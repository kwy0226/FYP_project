import os
import base64
import time
import json
import logging
from typing import Optional, List, Dict, Iterator, Iterable

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ----------------------------------------------------------------------
# Basic Initialization
# ----------------------------------------------------------------------
# Configure logging for debugging and server monitoring.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("emotion-api")

# Temporary HuggingFace cache directories.
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface/datasets"

# ----------------------------------------------------------------------
# ML Dependencies
# ----------------------------------------------------------------------
import torch
import torchaudio
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoFeatureExtractor,
    AutoModelForAudioClassification,
)

# For language detection (used to determine reply language)
from langdetect import detect

# OpenAI API SDK
from openai import OpenAI

# Firebase Admin SDK
import firebase_admin
from firebase_admin import credentials, db

# ----------------------------------------------------------------------
# Environment Variables & SDK Initialization
# ----------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
ENABLE_GPT_AUDIO = os.getenv("ENABLE_GPT_AUDIO", "1")

_openai_client = None
if OPENAI_API_KEY:
    try:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        log.info("[OpenAI] Client initialized")
    except Exception as e:
        log.exception("[OpenAI init error] %s", e)

# Firebase setup
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

# FastAPI app initialization
app = FastAPI(title="Emotion API (Roleplay + Personality + Safety Guard)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# ----------------------------------------------------------------------
# Request Schemas
# ----------------------------------------------------------------------
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

    # User-defined character settings, always stored AS-IS.
    aiName: Optional[str] = None
    aiGender: Optional[str] = None
    aiBackground: Optional[str] = None

    uid: Optional[str] = None
    chatId: Optional[str] = None
    msgId: Optional[str] = None


# ----------------------------------------------------------------------
# Emotion Models & Mapping Configurations
# ----------------------------------------------------------------------
TXT_MODEL_EN = "SamLowe/roberta-base-go_emotions"
TXT_MODEL_ZH = "IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment"
_TXT_MODELS: dict[str, dict] = {}

AUDIO_EMO_MODEL = "superb/hubert-large-superb-er"
_AUDIO_EMO: Dict[str, object] = {}

GOEMO_GROUPS = {
    "happy": {"excitement", "joy", "amusement", "relief", "pride", "gratitude", "love", "optimism", "desire"},
    "sad": {"sadness", "disappointment", "remorse", "grief"},
    "neutral": {"neutral", "approval", "curiosity", "realization", "admiration", "confusion", "surprise"},
    "angry": {"anger", "disgust", "disapproval", "embarrassment", "annoyance"},
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

# ----------------------------------------------------------------------
# 🔒 Safety Guard — Harmful Content Detection
# ----------------------------------------------------------------------
def _contains_harmful_user_text(user_msg: str) -> bool:
    """
    Detect harmful or dangerous content in user input.
    This ensures the assistant avoids producing harmful output
    even when roleplaying.
    """
    if not user_msg:
        return False

    msg = user_msg.lower()

    banned_patterns = [
        # Self-harm / suicide
        "kill myself",
        "suicide",
        "i want to die",
        "i want to hurt myself",

        # Harm to others
        "kill him",
        "kill her",
        "kill them",
        "hurt someone",
        "murder",

        # Abuse / violence
        "abuse me",
        "abuse her",
        "abuse him",
        "rape",
        "sexual assault",
    ]

    return any(p in msg for p in banned_patterns)

def _safety_overwrite_response(user_msg: str) -> Optional[str]:
    """
    If user message contains harmful content, override and return a safe response.
    This does NOT break the roleplay logic because:
    - It activates ONLY for dangerous content
    - It avoids generating inappropriate roleplay replies
    """
    if not _contains_harmful_user_text(user_msg):
        return None

    # Minimal safe response template
    return (
        "I'm really sorry you're feeling like this. "
        "I care about your safety, and you deserve support. "
        "Please consider reaching out to someone you trust or a professional who can help."
    )

# ----------------------------------------------------------------------
# Helper: Softmax
# ----------------------------------------------------------------------
def _softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softmax(logits, dim=-1)

# (… functions for loading models, audio utils, Firebase write helpers … remain unchanged …)

# ----------------------------------------------------------------------
# Character Profile Saving (AS-IS, No Auto-Fill)
# ----------------------------------------------------------------------
def _save_user_character_profile_simple(uid: str, char_id: str,
                                       ai_name: Optional[str], ai_gender: Optional[str],
                                       ai_background: Optional[str]):
    """
    Save the user's character settings as-is into Firebase.
    No auto-generation, no Wikipedia enrichment.
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
        "updatedAt": now_ts,
    }

    if not snap:
        updates["createdAt"] = now_ts

    ref.update(updates)

    # Sync minimal metadata to chat list
    meta_ref = db.reference(f"chathistory/{uid}/{char_id}/meta")
    try:
        meta_ref.update({
            "aiName": updates["aiName"],
            "updatedAt": now_ts,
        })
    except Exception:
        log.exception("meta sync failed")

# ----------------------------------------------------------------------
# Build Roleplay System Prompt
# ----------------------------------------------------------------------
def _build_roleplay_system_prompt(profile: Dict) -> str:
    """
    Build personality-based system prompt for roleplay.
    This version ensures the assistant fully absorbs the
    user's custom-defined background and personality.
    """

    name = (profile.get("aiName") or "").strip()
    gender = (profile.get("aiGender") or "").strip()
    background = (profile.get("aiBackground") or "").strip()

    lines = [
        "You are strictly playing the described character. Never break character.",
        "Your tone, attitude, emotions, vocabulary, and speaking style must be shaped 100% by the background.",
        "Do NOT respond like an AI. Respond like a real human texting.",
        "Avoid generic empathy, avoid robotic patterns, avoid formal tone.",
        "Be natural, expressive, and consistent with the personality.",
        "If the character is lively, be lively. If shy, be shy. If cold, be cold.",
    ]

    if name:
        lines.append(f"Character Name: {name}")
    if gender:
        lines.append(f"Gender: {gender}")
    if background:
        lines.append("Personality & Background Description:")
        lines.append(background)

    return "\n".join(lines)

# -----------------------
# OpenAI streaming helpers + SSE writer + safety integration (Part 2)
# -----------------------
from typing import Iterator

def _delta_iter_chat(messages: List[Dict]) -> Iterable[str]:
    """
    Create an OpenAI chat completion stream and yield incremental content deltas.
    This function uses the same streaming API as before and yields text fragments.
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
    Try to use the Responses API to process audio directly. If that fails,
    fallback to transcription + text chat streaming. Yields delta strings.
    """
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
                             "text": f"Received a voice message. User's current mood:{emotion_label}. Please respond in a conversational, human-like tone."},
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
            log.exception("Responses API audio path failed, falling back to transcription")

    # Fallback route: transcribe with Whisper-style endpoint and feed to text chat
    try:
        with open(wav_path, "rb") as f:
            transcript = _openai_client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=f)
        user_text = (transcript.text or "").strip() or "(Voice could not be recognized.)"
    except Exception:
        log.exception("Transcription failed – using placeholder text")
        user_text = "(Voice could not be transcribed.)"

    messages = [
        {"role": "system", "content": "You are an empathetic assistant. Respond like a real person in concise paragraphs."},
        {"role": "user", "content": f"[Emotion: {emotion_label}] {user_text}"}
    ]
    yield from _delta_iter_chat(messages)


def _sse_stream_from_deltas(delta_iter: Iterable[str],
                            uid: str, chatId: str, msgId: str) -> Iterator[bytes]:
    """
    Convert an iterator of delta strings into SSE frames while writing:
      - incremental aiReplyParts (so frontend can show streaming chunks)
      - final assembled aiReply (aiReply)
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

        # Flush remaining buffer
        if buf.strip():
            part_idx += 1
            final_piece = buf.strip()
            full_reply.append(final_piece)
            try:
                _push_reply_part(uid, chatId, msgId, final_piece, part_idx)
            except Exception:
                log.exception("push last piece failed")
            yield _sse({"type": "chunk", "part": part_idx, "content": final_piece})

        # Assemble final reply and write it to Firebase
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
# HTTP Routes (Text emotion + Chat + Audio) with safety integration
# -----------------------
@app.get("/")
def root():
    return {"status": "ok", "msg": "Emotion API root is alive (no-autofill + safety guard)"}


@app.post("/nlp/text-emotion")
def text_emotion(p: TextPayload):
    """
    Unchanged mapping from text classifiers to four categories.
    Writes emotion back to Firebase if uid/chatId/msgId present.
    """
    try:
        bundle = _pick_text_model_by_lang(p.text)
        tok, mdl, labels = bundle["tok"], bundle["mdl"], bundle["labels"]
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = tok(p.text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = mdl(**inputs)
            probs = _softmax(outputs.logits)[0].detach().cpu().tolist()

        # choose mapping
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

        _write_emotion_to_firebase(p.uid, p.chatId, p.msgId, result)
        return result
    except Exception as e:
        log.exception("text_emotion error")
        raise HTTPException(status_code=500, detail=f"text inference error: {e}")


@app.post("/chat/reply")
def chat_reply(p: ChatPayload):
    """
    Main chat endpoint for text messages.
    Workflow:
      1) Optional: run text-emotion (and record)
      2) Save user-supplied character settings AS-IS (no enrichment)
      3) Build roleplay system prompt from saved profile
      4) Safety check: if user message triggers safety, send a safe canned reply immediately
      5) Otherwise: stream OpenAI completion and write streaming parts to Firebase
    """
    if not _openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    try:
        # 1) Text Sentiment Analysis and write to Firebase (non-blocking)
        if p.uid and p.chatId and p.msgId:
            try:
                emo_result = text_emotion(TextPayload(text=p.message, uid=p.uid, chatId=p.chatId, msgId=p.msgId))
                ref = db.reference(f"chathistory/{p.uid}/{p.chatId}/messages/{p.msgId}")
                ref.update({"emotion": emo_result, "type": "text"})
            except Exception:
                log.exception("write text emotion failed")

        # 2) Save character profile as-is (simple storage)
        if p.uid and p.chatId:
            try:
                _save_user_character_profile_simple(
                    uid=p.uid, char_id=p.chatId,
                    ai_name=p.aiName, ai_gender=p.aiGender, ai_background=p.aiBackground
                )
            except Exception:
                log.exception("save_user_character_profile_simple failed")

        # 3) Load profile for prompt building
        try:
            ref = db.reference(f"character/{p.uid}/{p.chatId}")
            profile = ref.get() or {}
        except Exception:
            log.exception("read character profile failed")
            profile = {}

        # 4) Safety check on user input. If dangerous, return immediate safe reply and write to Firebase.
        safety_reply = _safety_overwrite_response(p.message or "")
        if safety_reply:
            # Write safe reply into Firebase as assistant reply
            try:
                _write_full_reply(p.uid or "", p.chatId or "", p.msgId or "", safety_reply)
            except Exception:
                log.exception("write safety reply to firebase failed")

            # Stream the safe reply back as SSE (single chunk + done)
            def safe_stream():
                yield _sse({"type": "chunk", "part": 1, "content": safety_reply})
                yield _sse({"type": "done"})

            headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
            return StreamingResponse(safe_stream(), media_type="text/event-stream", headers=headers)

        # 5) Build roleplay prompt and message list
        sys_prompt = _build_roleplay_system_prompt(profile)

        try:
            lang = detect(p.message)
        except Exception:
            lang = "en"

        lang_instr = "请使用中文回复用户。" if lang.startswith("zh") else "Please respond in English."

        messages = [{"role": "system", "content": sys_prompt + "\n" + lang_instr}]
        if p.history:
            messages.extend(p.history)
        messages.append({"role": "user", "content": p.message})

        headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}

        # Stream OpenAI deltas → SSE + firebase parts
        return StreamingResponse(
            _sse_stream_from_deltas(_delta_iter_chat(messages), p.uid or "", p.chatId or "", p.msgId or ""),
            media_type="text/event-stream",
            headers=headers
        )
    except Exception as e:
        log.exception("chat_reply error")
        raise HTTPException(status_code=500, detail=f"chat_reply error: {e}")


@app.post("/audio/process")
def audio_process(p: AudioPayload):
    """
    Audio processing route:
      - Convert base64 → wav file and run SER
      - Attempt to transcribe + detect language
      - Perform safety check on transcribed text; if harmful, return a safe canned reply immediately
      - Otherwise stream assistant reply (Responses API or transcription fallback)
      - All streaming parts and final reply are saved to Firebase by _sse_stream_from_deltas
    """
    if not _openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    try:
        _ensure_audio_emo()
        wav_path = _base64_wav_to_tmpfile(p.wav_base64)

        # Read wav using soundfile to avoid torchaudio loading quirks
        import soundfile as sf
        wav_np, sr = sf.read(wav_path, dtype="float32")
        wav = torch.tensor(wav_np).unsqueeze(0)

        # Ensure mono and 16k sampling rate
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != 16000:
            import torchaudio
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000

        # 1) SER: run audio emotion classification
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

        # 2) Write audio emotion to Firebase (if identifiers present)
        if p.uid and p.chatId and p.msgId:
            try:
                ref = db.reference(f"chathistory/{p.uid}/{p.chatId}/messages/{p.msgId}")
                ref.update({"emotion": emotion, "type": "audio"})
            except Exception:
                log.exception("write audio emotion failed")

        # 3) Attempt transcription to determine language and to run safety check
        user_lang = "en"
        transcribed_text = ""
        try:
            with open(wav_path, "rb") as f:
                transcript = _openai_client.audio.transcriptions.create(model="gpt-4o-mini-transcribe", file=f)
            transcribed_text = (transcript.text or "").strip()
            if transcribed_text:
                user_lang = detect(transcribed_text)
        except Exception:
            log.exception("audio transcription failed or language detect failed")

        # 4) Safety check on transcribed content
        safety_reply = _safety_overwrite_response(transcribed_text or "")
        if safety_reply:
            # Write safe reply into Firebase
            try:
                _write_full_reply(p.uid or "", p.chatId or "", p.msgId or "", safety_reply)
            except Exception:
                log.exception("write safety reply to firebase failed")

            def safe_stream_audio():
                yield _sse({"type": "chunk", "part": 1, "content": safety_reply})
                yield _sse({"type": "done"})

            headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}

            # Cleanup wav after streaming
            def stream_and_cleanup():
                try:
                    for chunk in safe_stream_audio():
                        yield chunk
                finally:
                    try:
                        if os.path.exists(wav_path):
                            os.remove(wav_path)
                    except Exception:
                        log.exception("failed to remove wav in cleanup")

            return StreamingResponse(stream_and_cleanup(), media_type="text/event-stream", headers=headers)

        # 5) No safety issue → proceed with audio->assistant streaming
        if user_lang.startswith("zh"):
            lang_instr = "请用中文回复用户。"
        else:
            lang_instr = "Please respond in English."

        # Compose an instruction for audio processing. We include emotion label as hint.
        stream = _sse_stream_from_deltas(
            _delta_iter_audio_with_fallback(wav_path, f"{emotion['label']} | {lang_instr}"),
            p.uid or "",
            p.chatId or "",
            p.msgId or "",
        )

        # Wrap stream and delete wav on completion
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

        headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
        return StreamingResponse(stream_with_cleanup(), media_type="text/event-stream", headers=headers)

    except Exception as e:
        log.exception("audio_process error")
        raise HTTPException(status_code=500, detail=f"audio_process error: {e}")
        
