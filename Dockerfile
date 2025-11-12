FROM python:3.10-slim

# 防止 pyc 缓存 & log 丢失
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 安装系统依赖（解决 torchaudio / torchcodec 问题）
RUN apt-get update && apt-get install -y \
    libsox-dev ffmpeg git && \
    rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目代码
COPY . .

# 暴露 Cloud Run 端口
EXPOSE 8080

# 启动 FastAPI 应用
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
