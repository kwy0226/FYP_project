import os
import io
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

# ---------- OpenAI chat ----------
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

_openai_client = None
if OPENAI_API_KEY:
    try:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        print("[OpenAI] Client initialized")
    except Exception as e:
        print("[OpenAI init error]", e)

# =======================================================
# FastAPI app
# =======================================================
app = FastAPI(title="Emotion API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发模式允许全部
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =======================================================
# Schemas
# =======================================================
class TextPayload(BaseModel):
    text: str = Field(..., min_length=1)

class AudioPayload(BaseModel):
    wav_base64: str = Field(..., description="Base64-encoded WAV audio")

class ChatPayload(BaseModel):
    message: str
    emotion: Optional[Dict] = None
    history: Optional[List[Dict]] = None

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
            savedir="pretrained_models/speech_emotion"
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
    path = f"/tmp/audio_{int(time.time()*1000)}.wav"
    with open(path, "wb") as f:
        f.write(raw)
    return path

# =======================================================
# Routes
# =======================================================
@app.get("/")
def root():
    return {"status": "ok", "msg": "Emotion API root is alive"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "text_model_multi": TXT_MODEL_EN,
        "text_model_zh": TXT_MODEL_ZH,
        "speech_model": SPEECH_MODEL_NAME,
        "openai_enabled": bool(_openai_client),
        "openai_model": OPENAI_MODEL if _openai_client else None,
    }

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
        return {
            "label": std_label,
            "confidence": float(probs[idx]),
            "probs": {mapping.get(labels[i].lower(), labels[i].lower()): float(probs[i]) for i in range(len(labels))},
            "used_model": mdl.name_or_path,
        }
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
        classes = _SPEECH_EMO.hparams.label_encoder.decode_ndim(torch.arange(len(scores)))
        mx_idx = int(torch.tensor(scores).argmax().item())
        return {
            "label": str(label),
            "confidence": float(scores[mx_idx]),
            "probs": {str(classes[i]): float(scores[i]) for i in range(len(scores))},
            "model": SPEECH_MODEL_NAME,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"audio inference error: {e}")

# ----------------- Chat -----------------
@app.post("/chat/reply")
def chat_reply(p: ChatPayload):
    if not _openai_client:
        return {
            "reply": "(stub) I hear you. Tell me more.",
            "note": "Set OPENAI_API_KEY in Hugging Face Secrets to enable real replies."
        }
    sys_prompt = (
        "You are a supportive, empathetic companion. "
        "Be concise, warm, and practical. If an emotion hint is provided, "
        "acknowledge it and respond accordingly."
    )
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
            temperature=0.6,
        )
        reply = resp.choices[0].message.content
        return {"reply": reply}
    except Exception as e:
        return {
            "reply": "(stub) Sorry, I'm having trouble replying right now.",
            "error": str(e),
        }

# ----------------- Extra GET for debug -----------------
@app.get("/chat/reply")
def chat_reply_get():
    return {
        "status": "ok",
        "note": "Use POST with JSON {message, emotion, history} to get a reply."
    }
