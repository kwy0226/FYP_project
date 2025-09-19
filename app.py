import os
import base64
import time
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------- ML deps ----------
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from speechbrain.pretrained import EncoderClassifier
import torchaudio

# language detection
from langdetect import detect

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

# Firebase init
if not firebase_admin._apps:
    fb_key_path = os.getenv("FIREBASE_KEY_PATH", "firebase_key.json")
    if os.path.exists(fb_key_path):
        cred = credentials.Certificate(fb_key_path)
        firebase_admin.initialize_app(
            cred,
            {"databaseURL": os.getenv("FIREBASE_URL", "")},
        )

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

    # character profile fields (exclusive per user)
    aiName: Optional[str] = None
    aiGender: Optional[str] = None
    aiBackground: Optional[str] = None
    aiPersonality: Optional[str] = None  # new

    # routing
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
            savedir="pretrained_models/speech_emotion",
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
    ref = db.reference(f"users/{uid}/chats/{chatId}/messages/{msgId}")
    ref.update({"detectedEmotion": emotion})


# ----------------- Profile (exclusive) -----------------
def _character_key_for_user(uid: str, ai_name: str) -> str:
    """
    Use aiName as the character key under the user.
    If you prefer numbered keys like Character1/2..., replace this function
    to generate the next available index and return that key instead.
    """
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

    updates = {
        "name": ai_name,
        "gender": ai_gender or snap.get("gender", ""),
        "personality": ai_personality or snap.get("personality", ""),
        "background": ai_background or snap.get("background", ""),
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
        "Speak in first-person as the character. Be specific, warm and immersive.",
        "Avoid AI disclaimers or meta commentary.",
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
        mapping = {
            "neg": "negative",
            "negative": "negative",
            "neu": "neutral",
            "neutral": "neutral",
            "pos": "positive",
            "positive": "positive",
        }
        std_label = mapping.get(raw_label, raw_label)
        result = {
            "label": std_label,
            "confidence": float(probs[idx]),
            "probs": {
                mapping.get(labels[i].lower(), labels[i].lower()): float(probs[i])
                for i in range(len(labels))
            },
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
        try:
            wav, sr = torchaudio.load(wav_path)
            if sr != 16000:
                wav = torchaudio.functional.resample(wav, sr, 16000)
                torchaudio.save(wav_path, wav, 16000)
        except Exception:
            pass

        out = _SPEECH_EMO.classify_file(wav_path)
        os.remove(wav_path)

        label = out["predicted_label"]
        scores = out["scores"].squeeze().detach().cpu().tolist()
        classes = _SPEECH_EMO.hparams.label_encoder.decode_ndim(
            torch.arange(len(scores))
        )
        mx_idx = int(torch.tensor(scores).argmax().item())
        result = {
            "label": str(label),
            "confidence": float(scores[mx_idx]),
            "probs": {str(classes[i]): float(scores[i]) for i in range(len(scores))},
            "model": SPEECH_MODEL_NAME,
        }
        _write_emotion_to_firebase(p.uid, p.chatId, p.msgId, result)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"audio inference error: {e}")


@app.post("/chat/reply")
def chat_reply(p: ChatPayload):
    if not _openai_client:
        return {"reply": "(stub) AI disabled"}

    # persist/update user's exclusive character profile
    if p.uid and p.aiName:
        _upsert_user_character_profile(
            uid=p.uid,
            ai_name=p.aiName,
            ai_gender=p.aiGender,
            ai_personality=p.aiPersonality,
            ai_background=p.aiBackground,
        )

    # load profile for prompt
    profile = _load_user_character_profile(p.uid or "", p.aiName or "")
    sys_prompt = _build_roleplay_system_prompt(profile)

    # user message (attach emotion hint if present)
    user_text = p.message
    if p.emotion:
        user_text = f"[Emotion hint: {p.emotion}] {p.message}"

    msgs = [{"role": "system", "content": sys_prompt}]
    if p.history:
        msgs.extend(p.history)
    msgs.append({"role": "user", "content": user_text})

    try:
        resp = _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            temperature=0.7,
        )
        reply = resp.choices[0].message.content

        if p.emotion:
            _write_emotion_to_firebase(p.uid, p.chatId, p.msgId, p.emotion)

        return {"reply": reply}
    except Exception as e:
        return {"reply": "(stub) error", "error": str(e)}

