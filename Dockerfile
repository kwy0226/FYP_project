# ==========================================================
# ✅ Cloud Run verified FastAPI Dockerfile
# ==========================================================
FROM python:3.10-slim

# 避免缓存和编码问题
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 先复制依赖文件
COPY requirements.txt .

# 安装依赖（+ ffmpeg 避免 torchaudio 声音 backend 错误）
RUN apt-get update && apt-get install -y ffmpeg && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# 拷贝项目
COPY . .

# 暴露 Cloud Run 端口
EXPOSE 8080

# ✅ 关键点：使用 shell 模式执行，$PORT 才能被替换
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}
