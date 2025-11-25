import os
import base64
import time
import json
import logging
import traceback
from typing import Optional, List, Dict, Iterator, Iterable

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# -----------------------
# Basic logging + env
# -----------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("emotion-api")

# Temporary huggingface caches (if models are used)
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface/datasets"

# ---------- ML deps ----------
import torch
import torchaudio
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoFeatureExtractor,            # For audio features
    AutoModelForAudioClassification, # Voice Emotion
)

# language detection for controlling response language
from langdetect import detect

# ---------- OpenAI ----------
from openai import OpenAI

# ---------- Firebase ----------
import firebase_admin
from firebase_admin import credentials, db

# -----------------------
# OpenAI and Firebase init
# -----------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
ENABLE_GPT_AUDIO = os.getenv("ENABLE_GPT_AUDIO", "1")  # Keep responses API audio path enabled by default

_openai_client = None
if OPENAI_API_KEY:
    try:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        log.info("[OpenAI] Client initialized")
    except Exception as e:
        log.exception("[OpenAI init error] %s", e)

# Firebase init using credentials JSON in env (as before)
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
            log.info("[Firebase] Initialized")
        except Exception as e:
            log.exception("[Firebase init error] %s", e)

# FastAPI app
app = FastAPI(title="Emotion API (no-autofill)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# -----------------------
# Request schemas
# -----------------------
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

    # IMPORTANT: These are accepted from client and saved as-is.
    aiName: Optional[str] = None
    aiGender: Optional[str] = None
    aiBackground: Optional[str] = None

    uid: Optional[str] = None
    chatId: Optional[str] = None
    msgId: Optional[str] = None


# -----------------------
# Models / mapping config
# -----------------------
# Text Sentiment models (you already used)
TXT_MODEL_EN = "SamLowe/roberta-base-go_emotions"
TXT_MODEL_ZH = "IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment"
_TXT_MODELS: dict[str, dict] = {}

# Voice emotion model
AUDIO_EMO_MODEL = "superb/hubert-large-superb-er"
_AUDIO_EMO: Dict[str, object] = {}

# Mapping of GoEmotions -> 4 categories
GOEMO_GROUPS = {
    "happy": {
        "excitement", "joy", "amusement", "relief", "pride",
        "gratitude", "love", "optimism", "desire"
    },
    "sad": {
        "sadness", "disappointment", "remorse", "grief"
    },
    "neutral": {
        "neutral", "approval", "curiosity", "realization",
        "admiration", "confusion", "surprise"
    },
    "angry": {
        "anger", "disgust", "disapproval", "embarrassment", "annoyance"
    },
}
ZH_SENTIMENT_MAP = {
    "positive": "happy",
    "negative": "sad",
    "neutral": "neutral",
}
AUDIO_GROUP_MAP = {
    "happy": "happy",
    "sad": "sad",
    "neutral": "neutral",
    "anger": "angry",
    "angry": "angry",
    "disgust": "angry",
    "fear": "neutral",
    "surprise": "neutral",
}
FOUR_KEYS = ["happy", "sad", "neutral", "angry"]


def _empty_grouped_dict() -> Dict[str, float]:
    return {k: 0.0 for k in FOUR_KEYS}


def _group_goemotions(labels: List[str], probs: List[float]) -> Dict:
    """
    Aggregate fine-grained GoEmotions labels into four broad categories,
    then return a dict containing label, confidence, grouped_probs and raw info.
    """
    grouped = _empty_grouped_dict()
    # raw label/confidence from argmax of original probs
    raw_idx = int(torch.tensor(probs).argmax().item())
    raw_label = labels[raw_idx]
    raw_conf = float(probs[raw_idx])

    # Aggregate probabilities into our four buckets
    for i, lab in enumerate(labels):
        p = float(probs[i])
        found = False
        for g, members in GOEMO_GROUPS.items():
            if lab in members:
                grouped[g] += p
                found = True
                break
        if not found:
            grouped["neutral"] += p

    # Normalize
    s = sum(grouped.values()) or 1.0
    for k in grouped:
        grouped[k] = grouped[k] / s

    best = max(grouped.items(), key=lambda x: x[1])[0]
    return {
        "label": best,
        "confidence": grouped[best],
        "grouped_probs": grouped,
        "raw": {
            "label": raw_label,
            "confidence": raw_conf,
            "labels": labels,
            "used_mapping": "goemotions→4"
        }
    }


def _group_zh_sentiment(labels: List[str], probs: List[float]) -> Dict:
    """
    Chinese sentiment mapping (positive/neutral/negative) -> four categories.
    Note: the Chinese model cannot reliably separate anger vs sadness; we map negative->sad.
    """
    raw_idx = int(torch.tensor(probs).argmax().item())
    raw_label = labels[raw_idx].lower()
    raw_conf = float(probs[raw_idx])

    grouped = _empty_grouped_dict()
    for i, lab in enumerate(labels):
        p = float(probs[i])
        lab = lab.lower()
        mapped = ZH_SENTIMENT_MAP.get(lab, "neutral")
        grouped[mapped] += p

    s = sum(grouped.values()) or 1.0
    for k in grouped:
        grouped[k] = grouped[k] / s

    best = max(grouped.items(), key=lambda x: x[1])[0]
    return {
        "label": best,
        "confidence": grouped[best],
        "grouped_probs": grouped,
        "raw": {
            "label": raw_label,
            "confidence": raw_conf,
            "labels": labels,
            "used_mapping": "zh-sentiment→4"
        }
    }


def _group_audio(labels: List[str], probs: List[float]) -> Dict:
    """
    Aggregate audio model outputs (7 categories) into our 4 categories.
    """
    raw_idx = int(torch.tensor(probs).argmax().item())
    raw_label = labels[raw_idx].lower()
    raw_conf = float(probs[raw_idx])

    grouped = _empty_grouped_dict()
    for i, lab in enumerate(labels):
        p = float(probs[i])
        lab = lab.lower()
        mapped = AUDIO_GROUP_MAP.get(lab, "neutral")
        grouped[mapped] += p

    s = sum(grouped.values()) or 1.0
    for k in grouped:
        grouped[k] = grouped[k] / s

    best = max(grouped.items(), key=lambda x: x[1])[0]
    return {
        "label": best,
        "confidence": grouped[best],
        "grouped_probs": grouped,
        "raw": {
            "label": raw_label,
            "confidence": raw_conf,
            "labels": labels,
            "used_mapping": "audio→4"
        }
    }


# -----------------------
# Model loaders (lazy)
# -----------------------
def _load_text_model_once(key: str, model_name: str):
    """
    Load a text model/tokenizer once and cache it in _TXT_MODELS.
    This function moves model to GPU if available.
    """
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
        labels = ["neutral", "happy", "sad", "angry"]
    _TXT_MODELS[key] = {"tok": tok, "mdl": mdl, "labels": labels}
    log.info("[TEXT] loaded %s", model_name)


def _pick_text_model_by_lang(text: str):
    """
    Choose text model by language detection — default to English if detection fails.
    """
    try:
        lang = detect(text)
    except Exception:
        lang = "en"
    if lang.startswith("zh"):
        key, model_name = "zh", TXT_MODEL_ZH
    else:
        key, model_name = "en", TXT_MODEL_EN
    _load_text_model_once(key, model_name)
    return _TXT_MODELS[key]


def _ensure_audio_emo():
    """
    Load audio emotion model/extractor once and cache.
    """
    if _AUDIO_EMO:
        return
    extractor = AutoFeatureExtractor.from_pretrained(AUDIO_EMO_MODEL)
    model = AutoModelForAudioClassification.from_pretrained(AUDIO_EMO_MODEL)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    _AUDIO_EMO["extractor"] = extractor
    _AUDIO_EMO["model"] = model
    _AUDIO_EMO["labels"] = labels
    log.info("[AUDIO] loaded %s", AUDIO_EMO_MODEL)


# -----------------------
# Utility helpers
# -----------------------
def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(logits, dim=-1)


def _strip_data_url_prefix(b64: str) -> str:
    """
    If the base64 string includes a data URL prefix, strip it.
    """
    if "," in b64 and "base64" in b64[:64]:
        return b64.split(",", 1)[1]
    return b64


def _base64_wav_to_tmpfile(b64: str) -> str:
    """
    Convert base64 wav -> temporary file path and return path.
    """
    raw = base64.b64decode(_strip_data_url_prefix(b64), validate=True)
    path = f"/tmp/audio_{int(time.time() * 1000)}.wav"
    with open(path, "wb") as f:
        f.write(raw)
    return path


def _write_emotion_to_firebase(uid: str, chatId: str, msgId: str, emotion: Dict):
    """
    Write the computed emotion object into the corresponding message node in Firebase.
    If any of uid/chatId/msgId is missing the function returns early.
    """
    if not uid or not chatId or not msgId:
        return
    ref = db.reference(f"chathistory/{uid}/{chatId}/messages/{msgId}")
    ref.update({"emotion": emotion})


def _push_reply_part(uid: str, chatId: str, msgId: str, content: str, part_idx: int):
    """
    Append an incremental 'aiReplyParts' chunk into Firebase for streaming UI.
    """
    if not uid or not chatId or not msgId:
        return
    ref = db.reference(f"chathistory/{uid}/{chatId}/messages/{msgId}").child("aiReplyParts")
    now_ms = int(time.time() * 1000)
    ref.push({
        "content": content,
        "role": "assistant",
        "type": "text",
        "createdAt": now_ms,
        "part": part_idx
    })


def _write_full_reply(uid: str, chatId: str, msgId: str, reply: str):
    """
    Write the final assembled assistant reply (aiReply) into Firebase.
    """
    if not uid or not chatId or not msgId:
        return
    ref = db.reference(f"chathistory/{uid}/{chatId}/messages/{msgId}")
    ref.update({
        "aiReply": {
            "content": reply,
            "role": "assistant",
            "type": "text",
            "createdAt": int(time.time() * 1000),
        }
    })


def _sse(data_obj: Dict) -> bytes:
    """
    Convert a JSON-able object into Server-Sent Event bytes with 'data:' prefix.
    """
    return f"data: {json.dumps(data_obj, ensure_ascii=False)}\n\n".encode("utf-8")


def _segment_ready(buf: str) -> bool:
    """
    Decide whether a text buffer is ready to flush as a chunk.
    Heuristics: punctuation or length threshold.
    """
    if any(p in buf for p in ["。", "！", "？", ".", "!", "?"]):
        return True
    return len(buf) >= 60


# -----------------------
# Character profile storage (SIMPLE)
# -----------------------
def _save_user_character_profile_simple(uid: str, char_id: str,
                                       ai_name: Optional[str], ai_gender: Optional[str],
                                       ai_background: Optional[str]):
    """
    Save character settings exactly as supplied by the client into Firebase.
    This function DOES NOT perform any enrichment, web lookup or auto-generation.
    It simply updates the character node and returns.
    """
    if not uid or not char_id:
        return
    now_ts = int(time.time() * 1000)
    ref = db.reference(f"character/{uid}/{char_id}")
    snap = ref.get() or {}

    updates = {
        "aiName": ai_name or snap.get("aiName", ""),
        "aiGender": ai_gender or snap.get("aiGender", ""),
        "aiBackground": ai_background if ai_background is not None else snap.get("aiBackground", ""),
        # do NOT create aiPersonality on server side
        "updatedAt": now_ts,
    }
    if not snap:
        updates["createdAt"] = now_ts
    ref.update(updates)

    # Also synchronize key metadata that the frontend expects (chat list / homepage)
    # Meta lives under chathistory/<uid>/<char_id>/meta
    meta_ref = db.reference(f"chathistory/{uid}/{char_id}/meta")
    try:
        meta_ref.update({
            "aiName": updates["aiName"],
            "updatedAt": now_ts,
        })
    except Exception:
        log.exception("meta sync failed in save_user_character_profile_simple")


# -----------------------
# Roleplay system prompt builder
# -----------------------
def _build_roleplay_system_prompt(profile: Dict) -> str:
    """
    Build a flexible roleplay system prompt where the AI's speech style,
    tone, pacing, and personality dynamically adapt to whatever the user
    writes in aiBackground.

    The more detailed the user writes, the more accurate the characterization.
    """

    name = (profile.get("aiName") or "").strip()
    gender = (profile.get("aiGender") or "").strip()
    background = (profile.get("aiBackground") or "").strip()

    # Dynamic personality shaping:
    # We explicitly instruct GPT to derive tone + speech patterns from background
    lines = [
        "You are strictly playing the described character. Never break character.",
        "Your tone, attitude, emotional expression, vocabulary, and speech style MUST be inferred directly from the user's background description.",
        "Do NOT use generic empathetic AI tone. Do NOT use counseling/therapist phrases.",
        "Speak like a real human texting, not an AI. Keep messages natural, expressive, and personality-driven.",
        "Do NOT be repetitive. Do NOT sound formal or robotic.",
        "Adjust your writing length naturally according to the personality.",
        "If the background describes a lively person, respond lively. If shy, respond shyly. If cold, respond coldly.",
        "You must adapt 100% of your speaking style to the background description."
    ]

    if name:
        lines.append(f"Character Name: {name}")
    if gender:
        lines.append(f"Gender: {gender}")
    if background:
        lines.append(f"Personality & Background Description:\n{background}")

    return "\n".join(lines)

# -----------------------
# OpenAI streaming helpers (unchanged)
# -----------------------
def _delta_iter_chat(messages: List[Dict]) -> Iterable[str]:
    """
    Create an OpenAI chat completion stream and yield incremental content deltas.
    """
    stream = _openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=0.8,
        max_tokens=400,
        stream=True,
    )
    for chunk in stream:
        try:
            delta = chunk.choices[0].delta.content or ""
        except Exception:
            delta = ""
        if delta:
            yield delta


def _delta_iter_audio_with_fallback(wav_path: str, emotion_label: str) -> Iterable[str]:
    """
    Try to use the Responses API to directly process audio. If not available or fails,
    fallback to transcription + text-based chat streaming.
    """
    # Attempt direct audio path via Responses API if enabled
    if ENABLE_GPT_AUDIO == "1":
        try:
            with open(wav_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            stream = _openai_client.responses.create(
                model=OPENAI_MODEL,
                stream=True,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text",
                             "text": f"Received a voice message. User's current mood:{emotion_label}。Please respond with empathy and in a conversational, human-like tone based on the user's content and emotional tone. Divide into 2–3 paragraphs, with 1–2 sentences per paragraph, totaling ≤50 words."},
                            {"type": "input_audio",
                             "audio": {"data": b64, "format": "wav"}}
                        ],
                    }
                ],
            )
            for event in stream:
                et = getattr(event, "type", None)
                if et == "response.output_text.delta":
                    delta = getattr(event, "delta", "") or ""
                    if delta:
                        yield delta
                elif et == "response.error":
                    err = getattr(event, "error", "")
                    log.error("Responses stream error: %s", err)
                elif et in ("response.completed", "response.done"):
                    break
            return
        except Exception:
            log.exception("Responses API audio path failed, fallback to transcription")

    # Fallback to whisper transcription + text-based chat
    try:
        with open(wav_path, "rb") as f:
            transcript = _openai_client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe", file=f
            )
        user_text = (transcript.text or "").strip() or "（Voice content could not be recognized.）"
    except Exception:
        log.exception("Transcription failed, use placeholder text")
        user_text = "（Voice content cannot be transcribed.）"

    messages = [
        {"role": "system", "content": "You are an empathetic emotional support assistant. Speak like a real person: Break into 2–3 paragraphs, each containing 1–2 sentences, with a total word count ≤50."},
        {"role": "user", "content": f"[Emotion: {emotion_label}] {user_text}"}
    ]
    yield from _delta_iter_chat(messages)


def _sse_stream_from_deltas(delta_iter: Iterable[str],
                            uid: str, chatId: str, msgId: str) -> Iterator[bytes]:
    """
    Convert an iterator of delta strings into SSE frames while also writing
    incremental parts and the final assembled reply into Firebase.
    """
    full_reply: List[str] = []
    buf = ""
    part_idx = 0
    try:
        for delta in delta_iter:
            buf += delta
            if _segment_ready(buf):
                part = buf.strip()
                buf = ""
                if part:
                    part_idx += 1
                    full_reply.append(part)
                    try:
                        _push_reply_part(uid, chatId, msgId, part, part_idx)
                    except Exception:
                        log.exception("push part to firebase failed")
                    yield _sse({"type": "chunk", "part": part_idx, "content": part})
        if buf.strip():
            part_idx += 1
            full_reply.append(buf.strip())
            try:
                _push_reply_part(uid, chatId, msgId, buf.strip(), part_idx)
            except Exception:
                log.exception("push last part to firebase failed")
            yield _sse({"type": "chunk", "part": part_idx, "content": buf.strip()})

        final_text = " ".join(full_reply)
        try:
            _write_full_reply(uid, chatId, msgId, final_text)
        except Exception:
            log.exception("write full reply failed")

        yield _sse({"type": "done"})
    except Exception:
        log.exception("sse_stream_from_deltas error")
        yield _sse({"type": "error", "message": "stream failed"})


# -----------------------
# HTTP Routes
# -----------------------
@app.get("/")
def root():
    return {"status": "ok", "msg": "Emotion API root is alive (no auto-fill)"}


# Text emotion route — unchanged behaviour, maps model outputs to 4 categories
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
            probs = _softmax(outputs.logits)[0].detach().cpu().tolist()

        # Decide mapping based on model labels
        if mdl.name_or_path.endswith("go_emotions") or len(labels) >= 10 or "joy" in [l.lower() for l in labels]:
            grouped = _group_goemotions([l.lower() for l in labels], probs)
        elif set([l.lower() for l in labels]) >= {"positive", "negative", "neutral"}:
            grouped = _group_zh_sentiment([l.lower() for l in labels], probs)
        else:
            raw_idx = int(torch.tensor(probs).argmax().item())
            raw_label = labels[raw_idx].lower()
            if raw_label in ("positive", "joy", "happiness", "amusement"):
                label = "happy"
            elif raw_label in ("negative", "sadness", "grief", "remorse", "disappointment"):
                label = "sad"
            elif raw_label in ("anger", "angry", "disgust", "disapproval", "annoyance"):
                label = "angry"
            else:
                label = "neutral"
            grouped = {
                "label": label,
                "confidence": float(probs[raw_idx]),
                "grouped_probs": _empty_grouped_dict() | {label: 1.0},
                "raw": {
                    "label": raw_label,
                    "confidence": float(probs[raw_idx]),
                    "labels": labels,
                    "used_mapping": "fallback"
                }
            }

        result = {
            "label": grouped["label"],
            "confidence": float(grouped["confidence"]),
            "grouped_probs": grouped["grouped_probs"],
            "raw": grouped["raw"],
            "used_model": mdl.name_or_path,
        }

        # Write emotion into firebase if identifiers provided
        _write_emotion_to_firebase(p.uid, p.chatId, p.msgId, result)
        return result
    except Exception as e:
        log.exception("text_emotion error")
        raise HTTPException(status_code=500, detail=f"text inference error: {e}")


# Chat SSE (Text messages) — this route now stores character profile AS-IS (no enrichment)
@app.post("/chat/reply")
def chat_reply(p: ChatPayload):
    """
    This endpoint receives a user message, optionally records sentiment and character settings,
    then streams an assistant reply via SSE. Character background is saved as-is (no auto lookup).
    """
    if not _openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    try:
        # 1) Text Sentiment Analysis and write to Firebase if identifiers available
        emo_result = None
        if p.uid and p.chatId and p.msgId:
            try:
                emo_result = text_emotion(TextPayload(text=p.message, uid=p.uid, chatId=p.chatId, msgId=p.msgId))
                ref = db.reference(f"chathistory/{p.uid}/{p.chatId}/messages/{p.msgId}")
                ref.update({"emotion": emo_result, "type": "text"})
            except Exception:
                log.exception("write text emotion failed")

        # 2) Save character profile EXACTLY as provided by client (NO enrichment)
        if p.uid and p.chatId:
            try:
                _save_user_character_profile_simple(
                    uid=p.uid, char_id=p.chatId,
                    ai_name=p.aiName, ai_gender=p.aiGender, ai_background=p.aiBackground
                )
            except Exception:
                log.exception("save_user_character_profile_simple failed")

        # 3) Retrieve the most recent character data from Firebase for prompt building
        try:
            ref = db.reference(f"character/{p.uid}/{p.chatId}")
            profile = ref.get() or {}
        except Exception:
            log.exception("read character profile failed")
            profile = {}

        # 4) Build roleplay system prompt using profile values verbatim
        sys_prompt = _build_roleplay_system_prompt(profile)

        # 5) Auto-detect language to instruct the assistant
        try:
            lang = detect(p.message)
        except Exception:
            lang = "en"
        if lang.startswith("zh"):
            lang_instr = "请使用中文回复用户。"
        else:
            lang_instr = "Please respond in English."

        # 6) Compose messages for OpenAI
        messages = [{"role": "system", "content": sys_prompt + "\n" + lang_instr}]
        if p.history:
            messages.extend(p.history)
        messages.append({"role": "user", "content": p.message})

        headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
        # Stream SSE using OpenAI deltas; Firebase writes happen inside _sse_stream_from_deltas
        return StreamingResponse(
            _sse_stream_from_deltas(_delta_iter_chat(messages), p.uid, p.chatId, p.msgId),
            media_type="text/event-stream",
            headers=headers
        )
    except Exception as e:
        log.exception("chat_reply error")
        raise HTTPException(status_code=500, detail=f"chat_reply error: {e}")


# Audio processing route (SER -> GPT response) — unchanged except uses simple profile storage above
@app.post("/audio/process")
def audio_process(p: AudioPayload):
    """
    Process uploaded base64 wav:
    1) Run audio emotion recognition and write emotion into Firebase
    2) Try to use Responses API for direct audio processing; fallback to transcription + chat streaming
    3) Stream assistant output via SSE, and write reply parts/full into Firebase
    """
    if not _openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    try:
        _ensure_audio_emo()
        wav_path = _base64_wav_to_tmpfile(p.wav_base64)

        # Use soundfile to read the wav to avoid torchaudio compatibility issues
        import soundfile as sf
        wav, sr = sf.read(wav_path, dtype="float32")  # returns numpy array
        wav = torch.tensor(wav).unsqueeze(0)  # convert to tensor

        # Ensure mono + 16k sampling rate expected by extractor
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000

        # 1) SER voice emotion detection
        extractor = _AUDIO_EMO["extractor"]
        model: AutoModelForAudioClassification = _AUDIO_EMO["model"]
        labels: List[str] = _AUDIO_EMO["labels"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = extractor(wav.squeeze().numpy(), sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = _softmax(logits)[0].detach().cpu().tolist()

        grouped = _group_audio([l.lower() for l in labels], probs)

        emotion = {
            "label": grouped["label"],
            "confidence": float(grouped["confidence"]),
            "grouped_probs": grouped["grouped_probs"],
            "raw": grouped["raw"],
            "used_model": AUDIO_EMO_MODEL,
        }

        # 2) Write emotion to Firebase if identifiers present
        if p.uid and p.chatId and p.msgId:
            try:
                ref = db.reference(f"chathistory/{p.uid}/{p.chatId}/messages/{p.msgId}")
                ref.update({"emotion": emotion, "type": "audio"})
            except Exception:
                log.exception("write audio emotion failed")

        # 3) Try to detect user's language by attempting transcription first
        user_lang = "en"
        try:
            with open(wav_path, "rb") as f:
                transcript = _openai_client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe", file=f
                )
            if transcript and transcript.text:
                user_lang = detect(transcript.text)
        except Exception:
            log.exception("audio detect language failed")

        if user_lang.startswith("zh"):
            lang_instr = "请用中文回复用户。"
        else:
            lang_instr = "Please respond in English."

        headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}

        # 4) Stream the assistant reply using delta iterator for audio responses
        stream = _sse_stream_from_deltas(
            _delta_iter_audio_with_fallback(
                wav_path,
                f"{emotion['label']} | {lang_instr}"
            ),
            p.uid,
            p.chatId,
            p.msgId,
        )

        # Wrap stream with cleanup: delete temp wav after streaming
        def stream_with_cleanup():
            try:
                for chunk in stream:
                    yield chunk
            finally:
                try:
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
                        log.info(f"[CLEANUP] Deleted temp wav file: {wav_path}")
                except Exception as e:
                    log.error(f"[CLEANUP] Failed to delete wav: {e}")

        return StreamingResponse(
            stream_with_cleanup(),
            media_type="text/event-stream",
            headers=headers,
        )

    except Exception as e:
        log.exception("audio_process error")
        raise HTTPException(status_code=500, detail=f"audio_process error: {e}")

