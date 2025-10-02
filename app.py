import os
import base64
import time
import json
import logging
import traceback
from typing import Optional, List, Dict, Iterator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# =======================================================
# Logging（让 Cloud Run 能看到真实报错）
# =======================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("emotion-api")

# =======================================================
# Hugging Face 缓存目录
# =======================================================
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface"
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface/datasets"

# ---------- ML deps ----------
import torch
import torchaudio
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoModelForAudioClassification,
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

# =======================================================
# Init
# =======================================================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_TRANSCRIBE_MODEL = "gpt-4o-mini-transcribe"

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

# =======================================================
# FastAPI app
# =======================================================
app = FastAPI(title="Emotion API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
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
# Models（细分类）
# =======================================================
TXT_MODEL_EN = "j-hartmann/emotion-english-distilroberta-base"         # 7类
TXT_MODEL_ZH = "uer/roberta-base-finetuned-dianping-chinese"  # 多类
_TXT_MODELS: dict[str, dict] = {}

# 语音情绪：使用 HF 模型，取代 speechbrain
AUDIO_EMO_MODEL = "superb/hubert-large-superb-er"  # 16kHz
_AUDIO_EMO: Dict[str, object] = {}

# =======================================================
# Loaders
# =======================================================
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
    processor = AutoProcessor.from_pretrained(AUDIO_EMO_MODEL)
    model = AutoModelForAudioClassification.from_pretrained(AUDIO_EMO_MODEL)
    model.eval()
    if torch.cuda.is_available():
        model.to("cuda")
    labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    _AUDIO_EMO["processor"] = processor
    _AUDIO_EMO["model"] = model
    _AUDIO_EMO["labels"] = labels
    log.info("[AUDIO] loaded %s", AUDIO_EMO_MODEL)

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
    return len(buf) >= 60  # 兜底长度，避免卡住

# =======================================================
# Character Profile Refinement（保持你原来的能力）
# =======================================================
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

    resp2 = _openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "你是一个角色分析专家。"},
            {"role": "user", "content": f"以下是角色 {ai_name} 的背景资料：\n{full_bg}\n\n请帮我总结：\n1. 角色的性格特点。\n2. 精炼一个适合 Roleplay 的背景描述。"}
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
        "不要包含 AI 身份、免责声明或任何元信息。",
        "每个回复像真人一样简短自然：分 2–3 段，每段 1–2 句，总字数≤80。",
    ]
    if name: lines.append(f"名字: {name}")
    if gender: lines.append(f"性别: {gender}")
    if personality: lines.append(f"性格: {personality}")
    if background: lines.append(f"背景: {background}")
    return "\n".join(lines)

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
            probs = _softmax(outputs.logits)[0].detach().cpu().tolist()
        idx = int(torch.tensor(probs).argmax().item())
        result = {
            "label": labels[idx],
            "confidence": float(probs[idx]),
            "probs": {labels[i]: float(probs[i]) for i in range(len(labels))},
            "used_model": mdl.name_or_path,
        }
        _write_emotion_to_firebase(p.uid, p.chatId, p.msgId, result)
        return result
    except Exception as e:
        log.exception("text_emotion error")
        raise HTTPException(status_code=500, detail=f"text inference error: {e}")

@app.post("/audio/emotion")
def audio_emotion(p: AudioPayload):
    try:
        _ensure_audio_emo()
        wav_path = _base64_wav_to_tmpfile(p.wav_base64)
        wav, sr = torchaudio.load(wav_path)
        os.remove(wav_path)
        # 单声道 & 16k
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000

        processor = _AUDIO_EMO["processor"]
        model: AutoModelForAudioClassification = _AUDIO_EMO["model"]
        labels: List[str] = _AUDIO_EMO["labels"]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        inputs = processor(wav.squeeze().numpy(), sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = _softmax(logits)[0].detach().cpu().tolist()
        idx = int(torch.tensor(probs).argmax().item())

        result = {
            "label": labels[idx],
            "confidence": float(probs[idx]),
            "probs": {labels[i]: float(probs[i]) for i in range(len(labels))}
        }
        _write_emotion_to_firebase(p.uid, p.chatId, p.msgId, result)
        return result
    except Exception as e:
        log.exception("audio_emotion error")
        raise HTTPException(status_code=500, detail=f"audio inference error: {e}")

# === SSE 流（逐条消息 + Firebase push）===
def _stream_chat(messages: List[Dict], uid: str, chatId: str, msgId: str) -> Iterator[bytes]:
    full_reply = []
    try:
        stream = _openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.8,
            max_tokens=400,
            stream=True,
        )
        buf = ""
        part_idx = 0
        for chunk in stream:
            try:
                delta = chunk.choices[0].delta.content or ""
            except Exception:
                delta = ""
            if not delta:
                continue
            buf += delta
            if _segment_ready(buf):
                part = buf.strip()
                buf = ""
                if part:
                    part_idx += 1
                    full_reply.append(part)
                    # 写 Firebase 单独的段落消息
                    try:
                        _push_reply_part(uid, chatId, msgId, part, part_idx)
                    except Exception:
                        log.exception("push part to firebase failed")
                    # SSE 发给前端
                    yield _sse({"type": "chunk", "part": part_idx, "content": part})
        if buf.strip():
            part_idx += 1
            full_reply.append(buf.strip())
            try:
                _push_reply_part(uid, chatId, msgId, buf.strip(), part_idx)
            except Exception:
                log.exception("push last part to firebase failed")
            yield _sse({"type": "chunk", "part": part_idx, "content": buf.strip()})

        # 可选：写完整文本
        final_text = " ".join(full_reply)
        try:
            _write_full_reply(uid, chatId, msgId, final_text)
        except Exception:
            log.exception("write full reply failed")

        yield _sse({"type": "done"})
    except Exception as e:
        log.exception("stream_chat error")
        # 不抛异常，避免 500；把错误通过 SSE 推给前端
        yield _sse({"type": "error", "message": str(e)})

@app.post("/chat/reply")
def chat_reply(p: ChatPayload):
    if not _openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    try:
        # 文字情绪（细分）
        if p.uid and p.chatId and p.msgId:
            try:
                emo_result = text_emotion(TextPayload(text=p.message, uid=p.uid, chatId=p.chatId, msgId=p.msgId))
                ref = db.reference(f"chathistory/{p.uid}/{p.chatId}/messages/{p.msgId}")
                ref.update({"emotion": emo_result})
            except Exception:
                log.exception("write text emotion failed")

        # 角色资料
        if p.uid and p.aiName:
            try:
                _upsert_user_character_profile(
                    uid=p.uid, char_id=p.chatId,
                    ai_name=p.aiName, ai_gender=p.aiGender, ai_background=p.aiBackground,
                )
            except Exception:
                log.exception("upsert character profile failed")

        # 系统提示
        try:
            ref = db.reference(f"character/{p.uid}/{p.chatId}")
            profile = ref.get() or {}
        except Exception:
            log.exception("read character profile failed")
            profile = {}
        sys_prompt = _build_roleplay_system_prompt(profile)

        msgs = [{"role": "system", "content": sys_prompt}]
        if p.history:
            msgs.extend(p.history)
        msgs.append({"role": "user", "content": p.message})

        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        return StreamingResponse(_stream_chat(msgs, p.uid, p.chatId, p.msgId),
                                 media_type="text/event-stream",
                                 headers=headers)
    except Exception as e:
        log.exception("chat_reply error")
        raise HTTPException(status_code=500, detail=f"chat_reply error: {e}")

@app.post("/audio/process")
def audio_process(p: AudioPayload):
    if not _openai_client:
        raise HTTPException(status_code=500, detail="OpenAI client not initialized")
    try:
        # 语音转文字
        wav_path = _base64_wav_to_tmpfile(p.wav_base64)
        with open(wav_path, "rb") as f:
            transcript = _openai_client.audio.transcriptions.create(
                model=OPENAI_TRANSCRIBE_MODEL, file=f
            )
        user_text = (transcript.text or "").strip()

        # 语音情绪（HF 模型）
        _ensure_audio_emo()
        wav, sr = torchaudio.load(wav_path)
        os.remove(wav_path)
        if wav.shape[0] > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.functional.resample(wav, sr, 16000)
            sr = 16000

        processor = _AUDIO_EMO["processor"]
        model: AutoModelForAudioClassification = _AUDIO_EMO["model"]
        labels: List[str] = _AUDIO_EMO["labels"]
        device = "cuda" if torch.cuda.is_available() else "cpu"

        inputs = processor(wav.squeeze().numpy(), sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = _softmax(logits)[0].detach().cpu().tolist()
        idx = int(torch.tensor(probs).argmax().item())

        emotion = {
            "label": labels[idx],
            "confidence": float(probs[idx]),
            "probs": {labels[i]: float(probs[i]) for i in range(len(labels))}
        }

        # 写入原消息节点（文本+情绪）
        if p.uid and p.chatId and p.msgId:
            try:
                ref = db.reference(f"chathistory/{p.uid}/{p.chatId}/messages/{p.msgId}")
                ref.update({"text": user_text, "emotion": emotion})
            except Exception:
                log.exception("write audio text/emotion failed")

        # 生成回复（同样走 SSE 流，分条 push）
        sys_prompt = "你是一个有同理心的情感支持助手。回复像真人说话，分成 2–3 段，每段 1–2 句，总体≤80 字。"
        msgs = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": f"[Emotion: {emotion['label']}] {user_text}"}
        ]
        headers = {
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
        return StreamingResponse(_stream_chat(msgs, p.uid, p.chatId, p.msgId),
                                 media_type="text/event-stream",
                                 headers=headers)
    except Exception as e:
        log.exception("audio_process error")
        raise HTTPException(status_code=500, detail=f"audio_process error: {e}")
