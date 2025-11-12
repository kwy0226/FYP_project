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

# =======================================================
# Logging
# =======================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("emotion-api")


# Hugging Face Cache Directory
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface/datasets"

# ---------- ML deps ----------
import torch
import logging

try:
    import torchaudio
    torchaudio._backend = None
except Exception as e:
    logging.warning(f"[CloudRun] torchaudio skipped: {e}")

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoFeatureExtractor,            # For audio features
    AutoModelForAudioClassification, # Speech emotion
)

# language detection
from langdetect import detect

# wikipedia
import wikipediaapi

# ---------- OpenAI ----------
from openai import OpenAI

# ---------- Firebase ----------
import firebase_admin
from firebase_admin import credentials, db

# Init
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
# Attempt to feed audio directly to GPT (Responses API); if unsuccessful, automatically fall back to transcription.
ENABLE_GPT_AUDIO = os.getenv("ENABLE_GPT_AUDIO", "1")  # “1” enables; other values disable

_openai_client = None
if OPENAI_API_KEY:
    try:
        _openai_client = OpenAI(api_key=OPENAI_API_KEY)
        log.info("[OpenAI] Client initialized")
    except Exception as e:
        log.exception("[OpenAI init error] %s", e)

# Firebase init
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
app = FastAPI(title="Emotion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

# Schemas

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


# Models
# Text Sentiment: Fine-Grained (English)
TXT_MODEL_EN = "SamLowe/roberta-base-go_emotions"            # English: Class 28
# Chinese Emotion (Positive/Neutral/Negative)
TXT_MODEL_ZH = "IDEA-CCNL/Erlangshen-Roberta-330M-Sentiment" # Chinese: Fine-grained emotion
_TXT_MODELS: dict[str, dict] = {}

# Speech Emotion Model (7 Categories)
AUDIO_EMO_MODEL = "superb/hubert-large-superb-er"
_AUDIO_EMO: Dict[str, object] = {}

# Four Major Categories Mapping (Text & Speech)
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
# Chinese Emotional Output (Positive/Neutral/Negative) → Four Categories
ZH_SENTIMENT_MAP = {
    "positive": "happy",
    "negative": "sad",   # The Chinese model cannot distinguish between anger and sadness, so both are uniformly categorized as “sad.” The frontend remains functional.
    "neutral": "neutral",
}
# Voice Category 7 → Category 4
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
    # Aggregate GoEmotions' fine-grained labels into four major categories and return the best aggregated label along with its distribution
    
    grouped = _empty_grouped_dict()
    raw_idx = int(torch.tensor(probs).argmax().item())
    raw_label = labels[raw_idx]
    raw_conf = float(probs[raw_idx])

    for i, lab in enumerate(labels):
        p = float(probs[i])
        found = False
        for g, members in GOEMO_GROUPS.items():
            if lab in members:
                grouped[g] += p
                found = True
                break
        if not found:
            # Minor tags not covered (e.g., fear, excitement are covered; if any remain overlooked) → Categorized as neutral
            grouped["neutral"] += p

    # Normalization (not required, but makes confidence more intuitive)
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
    
    # Chinese Sentiment (Positive/Neutral/Negative) → Four Categories (angry = 0)

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
    # Voice Category 7 Aggregated into Four Categories
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


# Loaders
def _load_text_model_once(key: str, model_name: str):
    if key in _TXT_MODELS:
        return
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    mdl.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    try:
        id2label = mdl.config.id2label
        labels = [id2label[i] for i in range(len(id2label))]
    except Exception:
        labels = ["neutral", "happy", "sad", "angry"]
    _TXT_MODELS[key] = {"tok": tok, "mdl": mdl, "labels": labels}
    log.info("[TEXT] loaded %s", model_name)


def _pick_text_model_by_lang(text: str):
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
    if _AUDIO_EMO:
        return
    extractor = AutoFeatureExtractor.from_pretrained(AUDIO_EMO_MODEL)
    model = AutoModelForAudioClassification.from_pretrained(AUDIO_EMO_MODEL)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    _AUDIO_EMO["extractor"] = extractor
    _AUDIO_EMO["model"] = model
    _AUDIO_EMO["labels"] = labels
    log.info("[AUDIO] loaded %s", AUDIO_EMO_MODEL)

# Helpers
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

def _push_reply_part(uid: str, chatId: str, msgId: str, content: str, part_idx: int):
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
    return f"data: {json.dumps(data_obj, ensure_ascii=False)}\n\n".encode("utf-8")

def _segment_ready(buf: str) -> bool:
    if any(p in buf for p in ["。", "！", "？", ".", "!", "?"]):
        return True
    return len(buf) >= 60  # Hem Length


# Character Profile Refinement（Keep）
def _refine_character_profile(ai_name: str, ai_background: str) -> Dict:
    user_agent = "EmotionMate/1.0"
    wiki_zh = wikipediaapi.Wikipedia(user_agent=user_agent, language="zh")
    wiki_en = wikipediaapi.Wikipedia(user_agent=user_agent, language="en")

    page = wiki_zh.page(ai_name.strip())
    summary = ""
    if not page.exists():
        page = wiki_en.page(ai_name.strip())
        if page.exists():
            summary = page.summary[0:800]
            resp = _openai_client.chat.completions.create(
                model="gpt-4o",   
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

    resp2 = _openai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "你是一个角色分析专家。"},
            {"role": "user", "content": f"以下是角色 {ai_name} 的背景资料：\n{full_bg}\n\n请帮我总结：\n1. 角色的性格特点。\n2. 精炼一个适合 Roleplay 的背景描述。\n3. 让角色的说话方式更接近真人。"}
        ]
    )
    result_text = resp2.choices[0].message.content.strip()

    return {
        "aiBackground": full_bg,
        "aiPersonality": result_text
    }


def _upsert_user_character_profile(uid: str, char_id: str, ai_name: Optional[str],
                                   ai_gender: Optional[str], ai_background: Optional[str]):
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
    name = (profile.get("aiName") or "").strip()
    gender = (profile.get("aiGender") or "").strip()
    personality = (profile.get("aiPersonality") or "").strip()
    background = (profile.get("aiBackground") or "").strip()
    lines = [
        "你正在严格扮演这个角色，不要跳出角色。",
        "必须用第一人称说话，让对话自然沉浸。",
        "避免AI身份、免责声明或元信息。",
        "每次回复≤80字，分2–3段，每段1–2句，像真人语气。",
        "语气可以带情绪波动，像真人聊天。"
    ]
    if name: lines.append(f"名字: {name}")
    if gender: lines.append(f"性别: {gender}")
    if personality: lines.append(f"性格: {personality}")
    if background: lines.append(f"背景: {background}")
    return "\n".join(lines)


# OpenAI Unified Incremental Text Iterator + SSE Wrapper
def _delta_iter_chat(messages: List[Dict]) -> Iterable[str]:
    # chat.completions stream, producing incremental text
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

    # Prioritize using the Responses API to directly understand audio; If unavailable, automatically fall back to Whisper transcription + text-based conversation.
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
                             "text": f"收到一段语音。用户当前情绪：{emotion_label}。请用同理心、口语化、简短的方式回复。分2–3段，每段1–2句，总字数≤80。"},
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

    # Rollback → Whisper Transcription
    try:
        with open(wav_path, "rb") as f:
            transcript = _openai_client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe", file=f
            )
        user_text = (transcript.text or "").strip() or "（语音内容未能识别）"
    except Exception:
        log.exception("Transcription failed, use placeholder text")
        user_text = "（语音内容无法转写）"

    messages = [
        {"role": "system", "content": "你是一个有同理心的情感支持助手。像真人一样说话：分成 2–3 段，每段 1–2 句，总字数≤80。"},
        {"role": "user", "content": f"[Emotion: {emotion_label}] {user_text}"}
    ]
    yield from _delta_iter_chat(messages)


def _sse_stream_from_deltas(delta_iter: Iterable[str],
                            uid: str, chatId: str, msgId: str) -> Iterator[bytes]:
    # Convert any incremental text into SSE + Firebase segmented & full-text writes 
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

# =======================================================
# Routes
# =======================================================
@app.get("/")
def root():
    return {"status": "ok", "msg": "Emotion API root is alive"}


# === Text Sentiment (Unified Mapping to Four Categories) ===
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

        # Aggregate by model type
        if mdl.name_or_path.endswith("go_emotions") or len(labels) >= 10 or "joy" in [l.lower() for l in labels]:
            grouped = _group_goemotions([l.lower() for l in labels], probs)
        elif set([l.lower() for l in labels]) >= {"positive", "negative", "neutral"}:
            grouped = _group_zh_sentiment([l.lower() for l in labels], probs)
        else:
            # Fallback: Directly take the argmax and perform a closest-match mapping.
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


# === Chat SSE (Text Messages) ===
@app.post("/chat/reply")
def chat_reply(p: ChatPayload):
    if not _openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    try:
        # 1. Text Sentiment Analysis (Four-Category Aggregation) and Writing to Firebase
        emo_result = None
        if p.uid and p.chatId and p.msgId:
            try:
                emo_result = text_emotion(TextPayload(text=p.message, uid=p.uid, chatId=p.chatId, msgId=p.msgId))
                ref = db.reference(f"chathistory/{p.uid}/{p.chatId}/messages/{p.msgId}")
                ref.update({"emotion": emo_result, "type": "text"})
            except Exception:
                log.exception("write text emotion failed")

        # 2. Update Character Profile
        if p.uid and p.aiName:
            try:
                _upsert_user_character_profile(
                    uid=p.uid, char_id=p.chatId,
                    ai_name=p.aiName, ai_gender=p.aiGender, ai_background=p.aiBackground,
                )
            except Exception:
                log.exception("upsert character profile failed")

        # 3. Retrieve Character Information
        try:
            ref = db.reference(f"character/{p.uid}/{p.chatId}")
            profile = ref.get() or {}
        except Exception:
            log.exception("read character profile failed")
            profile = {}

        sys_prompt = _build_roleplay_system_prompt(profile)

        # 4. Auto-detect language → Control GPT response language
        try:
            lang = detect(p.message)
        except Exception:
            lang = "en"
        if lang.startswith("zh"):
            lang_instr = "请使用中文回复用户。"
        else:
            lang_instr = "Please respond in English."

        messages = [{"role": "system", "content": sys_prompt + "\n" + lang_instr}]
        if p.history:
            messages.extend(p.history)
        messages.append({"role": "user", "content": p.message})

        headers = {"Cache-Control": "no-cache", "Connection": "keep-alive"}
        return StreamingResponse(
            _sse_stream_from_deltas(_delta_iter_chat(messages), p.uid, p.chatId, p.msgId),
            media_type="text/event-stream",
            headers=headers
        )
    except Exception as e:
        log.exception("chat_reply error")
        raise HTTPException(status_code=500, detail=f"chat_reply error: {e}")


# === Voice: SER → GPT Response (Four-Category Aggregation) ===
@app.post("/audio/process")
def audio_process(p: AudioPayload):
    if not _openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    try:
        _ensure_audio_emo()
        wav_path = _base64_wav_to_tmpfile(p.wav_base64)
        wav, sr = torchaudio.load(wav_path)

        # Mono & 16k
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000

        # 1. SER Voice Emotion Detection
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

        # 2. Write to Firebase
        if p.uid and p.chatId and p.msgId:
            try:
                ref = db.reference(f"chathistory/{p.uid}/{p.chatId}/messages/{p.msgId}")
                ref.update({"emotion": emotion, "type": "audio"})
            except Exception:
                log.exception("write audio emotion failed")

        # 3. Automatic Language Detection → Control Response Language
        user_lang = "en"
        try:
            # Attempt to transcribe and determine the language first
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

        stream = _sse_stream_from_deltas(
            _delta_iter_audio_with_fallback(wav_path, f"{emotion['label']} | {lang_instr}"),
            p.uid, p.chatId, p.msgId
        )

        try:
            os.remove(wav_path)
        except Exception:
            pass

        return StreamingResponse(stream, media_type="text/event-stream", headers=headers)
    except Exception as e:
        log.exception("audio_process error")
        raise HTTPException(status_code=500, detail=f"audio_process error: {e}")
