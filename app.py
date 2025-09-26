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
    if not uid or not chatId or not msgId:
        return
    ref = db.reference(f"chathistory/{uid}/{chatId}/messages/{msgId}")
    ref.update({
        "aiReply": {
            "content": reply,
            "role": "assistant",
            "type": "text",
            "createdAt": int(time.time() * 1000)
        }
    })


# =======================================================
# Character Profile Refinement
# =======================================================
def _refine_character_profile(ai_name: str, ai_background: str) -> Dict:
    """补全资料 + 自动生成 Personality"""
    user_agent = "EmotionMate/1.0 (wkyeoh0226@gmail.com)"
    wiki_zh = wikipediaapi.Wikipedia(user_agent=user_agent, language="zh")
    wiki_en = wikipediaapi.Wikipedia(user_agent=user_agent, language="en")

    page = wiki_zh.page(ai_name.strip())
    summary = ""
    if not page.exists():
        page = wiki_en.page(ai_name.strip())
        if page.exists():
            summary = page.summary[0:800]
            # 翻译成中文
            resp = _openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "你是一个翻译助手，把输入的英文简介翻译成中文，保持简洁自然。"},
                    {"role": "user", "content": summary}
                ]
            )
            summary = resp.choices[0].message.content.strip()
    else:
        summary = page.summary[0:800]

    full_bg = ai_background
    if summary:
        full_bg += "\n\n[补全资料] " + summary

    # 提炼性格 + 背景
    resp2 = _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "你是一个角色分析专家。"},
            {"role": "user", "content": f"以下是角色 {ai_name} 的背景资料：\n{full_bg}\n\n请帮我总结：\n1. 角色的性格特点（简洁中文描述）。\n2. 精炼一个适合 Roleplay 的背景描述。"}
        ]
    )
    result_text = resp2.choices[0].message.content.strip()

    return {
        "aiBackground": full_bg,
        "aiPersonality": result_text
    }


def _upsert_user_character_profile(
    uid: str,
    char_id: str,
    ai_name: Optional[str],
    ai_gender: Optional[str],
    ai_background: Optional[str],
):
    if not uid or not char_id or not ai_name:
        return

    now_ts = int(time.time() * 1000)
    ref = db.reference(f"character/{uid}/{char_id}")
    snap = ref.get() or {}

    enriched = {}
    if ai_background:
        enriched = _refine_character_profile(ai_name, ai_background)

    updates = {
        "aiName": ai_name or snap.get("aiName", ""),
        "aiGender": ai_gender or snap.get("aiGender", ""),
        "aiBackground": enriched.get("aiBackground", ai_background or snap.get("aiBackground", "")),
        "aiPersonality": enriched.get("aiPersonality", snap.get("aiPersonality", "")),
        "updatedAt": now_ts,
    }

    if not snap:
        updates["createdAt"] = now_ts

    ref.update(updates)


def _build_roleplay_system_prompt(profile: Dict) -> str:
    """基于数据库中的角色资料构造 Roleplay Prompt"""
    name = (profile.get("aiName") or "").strip()
    gender = (profile.get("aiGender") or "").strip()
    personality = (profile.get("aiPersonality") or "").strip()
    background = (profile.get("aiBackground") or "").strip()

    lines = [
        "你正在严格扮演这个角色，不要跳出角色。",
        "必须用第一人称说话，让对话自然沉浸。",
        "不要包含 AI 身份、免责声明或任何元信息。",
        "",
        f"名字: {name}" if name else "",
        f"性别: {gender}" if gender else "",
        f"性格: {personality}" if personality else "",
        f"背景: {background}" if background else "",
    ]
    return "\n".join([ln for ln in lines if ln])


# =======================================================
# Chat + Emotion Routes
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

    if p.uid and p.chatId and p.msgId:
        emo_result = text_emotion(TextPayload(text=p.message, uid=p.uid, chatId=p.chatId, msgId=p.msgId))
        ref = db.reference(f"chathistory/{p.uid}/{p.chatId}/messages/{p.msgId}")
        ref.update({"emotion": emo_result})

    if p.uid and p.aiName:
        _upsert_user_character_profile(
            uid=p.uid,
            char_id=p.chatId,
            ai_name=p.aiName,
            ai_gender=p.aiGender,
            ai_background=p.aiBackground,
        )

    ref = db.reference(f"character/{p.uid}/{p.chatId}")
    profile = ref.get() or {}
    sys_prompt = _build_roleplay_system_prompt(profile)

    msgs = [{"role": "system", "content": sys_prompt}]
    if p.history:
        msgs.extend(p.history)
    msgs.append({"role": "user", "content": p.message})

    try:
        resp = _openai_client.chat.completions.create(
            model=OPENAI_MODEL, messages=msgs, temperature=0.7,
        )
        reply = resp.choices[0].message.content
        _write_reply_to_firebase(p.uid, p.chatId, p.msgId, reply)
        return {"reply": reply}
    except Exception as e:
        _write_reply_to_firebase(p.uid, p.chatId, p.msgId, "(error)")
        return {"reply": "(error)", "error": str(e)}


@app.post("/audio/process")
def audio_process(p: AudioPayload):
    if not _openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")

    try:
        print("[audio_process] === START ===")
        print("[audio_process] payload length:", len(p.wav_base64) if p.wav_base64 else 0)

        # 保存临时 wav 文件
        wav_path = _base64_wav_to_tmpfile(p.wav_base64)
        print("[audio_process] temp wav saved:", wav_path)

        # 检查文件大小
        if os.path.exists(wav_path):
            size = os.path.getsize(wav_path)
            print(f"[audio_process] file exists, size={size} bytes")
        else:
            print("[audio_process][ERROR] wav file not found after save")

        # 调用 OpenAI 语音转写
        with open(wav_path, "rb") as f:
            print("[audio_process] sending file to OpenAI transcription...")
            transcript = _openai_client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=f
            )
        user_text = transcript.text.strip()
        print("[audio_process] transcription result:", user_text)

        # SpeechBrain 情绪识别
        _ensure_speech_model()
        print("[audio_process] running SpeechBrain classifier...")
        out = _SPEECH_EMO.classify_file(wav_path)
        print("[audio_process] speechbrain result:", out)
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
        print("[audio_process] final emotion dict:", emotion)

        # 写入 Firebase
        if p.uid and p.chatId and p.msgId:
            ref = db.reference(f"chathistory/{p.uid}/{p.chatId}/messages/{p.msgId}")
            ref.update({"text": user_text, "emotion": emotion})
            print("[audio_process] firebase updated")

        # 构建对话消息并调用 OpenAI
        msgs = [{"role": "user", "content": f"[Emotion hint: {emotion}] {user_text}"}]
        print("[audio_process] sending chat completion with msgs:", msgs)

        resp = _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=msgs,
            temperature=0.7,
        )
        reply = resp.choices[0].message.content
        print("[audio_process] reply from OpenAI:", reply)

        _write_reply_to_firebase(p.uid, p.chatId, p.msgId, reply)
        print("[audio_process] reply written to firebase")

        print("[audio_process] === DONE ===")
        return {
            "text": user_text,
            "emotion": emotion,
            "reply": reply,
        }

    except Exception as e:
        import traceback
        print("[audio_process][EXCEPTION]", str(e))
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"audio_process error: {e}")

