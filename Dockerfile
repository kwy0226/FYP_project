FROM python:3.10-slim

# 防止 pyc 缓存和日志丢失
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face 会注入 PORT 环境变量，使用 shell 模式保证变量展开
CMD bash -c "uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}"
