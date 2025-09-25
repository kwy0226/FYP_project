import os
import base64
import time
import json
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# =======================================================
# Hugging Face 缓存目录修复
# =======================================================
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface/datasets"

# ---------- ML deps ----------
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from speechbrain.pretrained import EncoderClassifier
import torchaudio

# language detection
from langdetect import detect

# wikipedia
import wikipediaapi

# ---------- OpenAI ----------
from openai import OpenAI

# ---------- Firebase ----------
import firebase_admin
from firebase_admin import credentials, db

# =======================================================
# Init
# =======================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_openai_client = None
if OPENAI_API_KEY:
    try:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("[OpenAI] Client initialized")
    except Exception as e:
        print("[OpenAI init error]", e)

# Firebase init (用环境变量 JSON)
if not firebase_admin._apps:
    cred_json = os.getenv("FIREBASE_CREDENTIALS_JSON")
    if cred_json:
        try:
            cred_dict = json.loads(cred_json)
            cred = credentials.Certificate(cred_dict)
            firebase_admin.initialize_app(
                cred,
                {"databaseURL": os.getenv("FIREBASE_URL", "")},
            )
            print("[Firebase] Initialized from FIREBASE_CREDENTIALS_JSON")
        except Exception as e:
            print("[Firebase init error]", e)

# =======================================================
# FastAPI app
# =======================================================
app = FastAPI(title="Emotion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================================================
# Schemas
# =======================================================
class TextPayload(BaseModel):
    text: str = Field(..., min_length=1)
    uid: Optional[str] = None
    chatId: Optional[str] = None
    msgId: Optional[str] = None


class AudioPayload(BaseModel):
    wav_base64: str
    uid: Optional[str] = None
    chatId: Optional[str] = None
    msgId: Optional[str] = None


class ChatPayload(BaseModel):
    message: str
    emotion: Optional[Dict] = None
    history: Optional[List[Dict]] = None

    aiName: Optional[str] = None
    aiGender: Optional[str] = None
    aiBackground: Optional[str] = None
    aiPersonality: Optional[str] = None

    uid: Optional[str] = None
    chatId: Optional[str] = None
    msgId: Optional[str] = None


# =======================================================
# Models
# =======================================================
TXT_MODEL_EN = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
TXT_MODEL_ZH = "IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment"
_TXT_MODELS: dict[str, dict] = {}

SPEECH_MODEL_NAME = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
_SPEECH_EMO = None


def _load_text_model_once(key: str, model_name: str):
    if key in _TXT_MODELS:
        return
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    mdl.eval()
    if torch.cuda.is_available():
        mdl.to("cuda")
    try:
        id2label = mdl.config.id2label
        labels = [id2label[i] for i in range(len(id2label))]
    except Exception:
        labels = ["negative", "neutral", "positive"]
    _TXT_MODELS[key] = {"tok": tok, "mdl": mdl, "labels": labels}


def _pick_text_model_by_lang(text: str):
    try:
        lang = detect(text)
    except Exception:
        lang = "en"
    if lang.startswith("zh"):
        key, model_name = "zh", TXT_MODEL_ZH
    else:
        key, model_name = "multi", TXT_MODEL_EN
    _load_text_model_once(key, model_name)
    return _TXT_MODELS[key]


def _ensure_speech_model():
    global _SPEECH_EMO
    if _SPEECH_EMO is None:
        _SPEECH_EMO = EncoderClassifier.from_hparams(
            source=SPEECH_MODEL_NAME,
            savedir="/tmp/pretrained_models/speech_emotion",
        )

# =======================================================
# Helpers
# =======================================================
def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(logits, dim=-1)


def _strip_data_url_prefix(b64: str) -> str:
    if "," in b64 and "base64" in b64[:64]:
        return b64.split(",", 1)[1]
    return b64


def _base64_wav_to_tmpfile(b64: str) -> str:
    raw = base64.b64decode(_strip_data_url_prefix(b64), validate=True)
    path = f"/tmp/audio_{int(time.time() * 1000)}.wav"
    with open(path, "wb") as f:
        f.write(raw)
    return path


def _write_emotion_to_firebase(uid: str, chatId: str, msgId: str, emotion: Dict):
    if not uid or not chatId or not msgId:
        return
    ref = db.reference(f"chathistory/{uid}/{chatId}/messages/{msgId}")
    ref.update({"emotion": emotion})


def _write_reply_to_firebase(uid: str, chatId: str, msgId: str, reply: str):
    """把 AI 回复写进 Firebase 的同一个 msgId"""
    if not uid or not chatId or not msgId:
        return
    ref = db.reference(f"chathistory/{uid}/{chatId}/messages/{msgId}")
    ref.update({
        "reply": reply,
        "replyCreatedAt": int(time.time() * 1000),
    })


# ----------------- Character Enrichment -----------------
def _enrich_character_background(ai_name: str, ai_background: str) -> str:
    user_agent = "EmotionMate/1.0 (wkyeoh0226@gmail.com)"  # ⚠️改成你项目的标识

    wiki_zh = wikipediaapi.Wikipedia(user_agent=user_agent, language="zh")
    wiki_en = wikipediaapi.Wikipedia(user_agent=user_agent, language="en")

    query_name = ai_name.strip()
    query_bg = ai_background.strip()

    page = wiki_zh.page(query_name)
    if not page.exists():
        page = wiki_en.page(query_name)

    if page.exists():
        summary = page.summary[0:500]
        return ai_background + "\n\n[补全资料] " + summary
    return ai_background


def _character_key_for_user(uid: str, ai_name: str) -> str:
    safe_key = (ai_name or "").strip()
    if not safe_key:
        safe_key = "Character"
    return safe_key


def _upsert_user_character_profile(
    uid: str,
    ai_name: Optional[str],
    ai_gender: Optional[str],
    ai_personality: Optional[str],
    ai_background: Optional[str],
):
    if not uid or not ai_name:
        return
    now_ts = int(time.time() * 1000)
    char_key = _character_key_for_user(uid, ai_name)
    ref = db.reference(f"Profile/Character/{uid}/{char_key}")
    snap = ref.get() or {}

    enriched_bg = ai_background
    if ai_background and snap.get("background") != ai_background:
        enriched_bg = _enrich_character_background(ai_name, ai_background)

    updates = {
        "name": ai_name,
        "gender": ai_gender or snap.get("gender", ""),
        "personality": ai_personality or snap.get("personality", ""),
        "background": enriched_bg or snap.get("background", ""),
        "lastUpdated": now_ts,
    }
    ref.update(updates)


def _load_user_character_profile(uid: str, ai_name: Optional[str]) -> Dict:
    if not uid or not ai_name:
        return {}
    char_key = _character_key_for_user(uid, ai_name)
    ref = db.reference(f"Profile/Character/{uid}/{char_key}")
    snap = ref.get()
    return snap or {}


def _build_roleplay_system_prompt(profile: Dict) -> str:
    name = (profile.get("name") or "").strip()
    gender = (profile.get("gender") or "").strip()
    personality = (profile.get("personality") or "").strip()
    background = (profile.get("background") or "").strip()

    lines = [
        "You are strictly roleplaying a character. Stay fully in-character.",
        "Speak in first-person as the character. Be immersive and natural.",
        "Do not include AI disclaimers or meta commentary.",
        "",
        f"Name: {name}" if name else "",
        f"Gender: {gender}" if gender else "",
        f"Personality: {personality}" if personality else "",
        f"Background: {background}" if background else "",
    ]
    return "\n".join([ln for ln in lines if ln])

# =======================================================
# Routes
# =======================================================
@app.get("/")
def root():
    return {"status": "ok", "msg": "Emotion API root is alive"}


@app.post("/nlp/text-emotion")
def text_emotion(p: TextPayload):
    try:
        bundle = _pick_text_model_by_lang(p.text)
        tok, mdl, labels = bundle["tok"], bundle["mdl"], bundle["labels"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tok(p.text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = mdl(**inputs)
            probs = _softmax(outputs.logits)[0].cpu().tolist()
        idx = int(torch.tensor(probs).argmax().item())
        raw_label = labels[idx].lower()
        mapping = {"neg": "negative", "negative": "negative",
                   "neu": "neutral", "neutral": "neutral",
                   "pos": "positive", "positive": "positive"}
        std_label = mapping.get(raw_label, raw_label)
        result = {
            "label": std_label,
            "confidence": float(probs[idx]),
            "used_model": mdl.name_or_path,
        }
        _write_emotion_to_firebase(p.uid, p.chatId, p.msgId, result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"text inference error: {e}")


@app.post("/audio/emotion")
def audio_emotion(p: AudioPayload):
    try:
        _ensure_speech_model()
        wav_path = _base64_wav_to_tmpfile(p.wav_base64)
        wav, sr = torchaudio.load(wav_path)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            torchaudio.save(wav_path, wav, 16000)

        out = _SPEECH_EMO.classify_file(wav_path)
        os.remove(wav_path)

        label = out["predicted_label"]
        result = {"label": str(label), "model": SPEECH_MODEL_NAME}
        _write_emotion_to_firebase(p.uid, p.chatId, p.msgId, result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"audio inference error: {e}")


@app.post("/chat/reply")
def chat_reply(p: ChatPayload):
    if not _openai_client:
        return {"reply": "(AI disabled)"}

    if p.uid and p.aiName:
        _upsert_user_character_profile(
            uid=p.uid,
            ai_name=p.aiName,
            ai_gender=p.aiGender,
            ai_personality=p.aiPersonality,
            ai_background=p.aiBackground,
        )

    profile = _load_user_character_profile(p.uid or "", p.aiName or "")
    sys_prompt = _build_roleplay_system_prompt(profile)

    user_text = p.message
    if p.emotion:
        user_text = f"[Emotion hint: {p.emotion}] {p.message}"

    msgs = [{"role": "system", "content": sys_prompt}]
    if p.history:
        msgs.extend(p.history)
    msgs.append({"role": "user", "content": user_text})

    try:
        resp = _openai_client.chat.completions.create(
            model=OPENAI_MODEL, messages=msgs, temperature=0.7,
        )
        reply = resp.choices[0].message.content

        # 保存 AI 回复
        _write_reply_to_firebase(p.uid, p.chatId, reply)

        emo_check = text_emotion(TextPayload(text=reply))
        if emo_check.get("label") == "negative":
            resp = _openai_client.chat.completions.create(
                model=OPENAI_MODEL, messages=msgs, temperature=0.7,
            )
            reply = resp.choices[0].message.content
            _write_reply_to_firebase(p.uid, p.chatId, reply)

        return {"reply": reply}
    except Exception as e:
        return {"reply": "(error)", "error": str(e)}


@app.post("/audio/process")
def audio_process(p: AudioPayload):
    if not _openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")

    try:
        wav_path = _base64_wav_to_tmpfile(p.wav_base64)

        with open(wav_path, "rb") as f:
            transcript = _openai_client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f
            )
        user_text = transcript.text.strip()

        _ensure_speech_model()
        out = _SPEECH_EMO.classify_file(wav_path)
        os.remove(wav_path)

        label = out["predicted_label"]
        scores = out["scores"].squeeze().detach().cpu().tolist()
        classes = _SPEECH_EMO.hparams.label_encoder.decode_ndim(
            torch.arange(len(scores))
        )
        emotion = {
            "label": str(label),
            "confidence": float(max(scores)),
            "probs": {str(classes[i]): float(scores[i]) for i in range(len(scores))},
        }

        if p.uid and p.chatId and p.msgId:
            ref = db.reference(f"chathistory/{p.uid}/{p.chatId}/messages/{p.msgId}")
            ref.update({"text": user_text, "emotion": emotion})

        msgs = [{"role": "user", "content": f"[Emotion hint: {emotion}] {user_text}"}]
        resp = _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            temperature=0.7,
        )
        reply = resp.choices[0].message.content

        # 保存 AI 回复
        _write_reply_to_firebase(p.uid, p.chatId, reply)

        return {
            "text": user_text,
            "emotion": emotion,
            "reply": reply,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"audio_process error: {e}")

