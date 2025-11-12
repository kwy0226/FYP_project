FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目代码
COPY . .

# Cloud Run 默认端口
EXPOSE 8080

# ✅ 使用 shell 模式 CMD，保证 Cloud Run 的 $PORT 被正确使用
CMD uvicorn app:app --host 0.0.0.0 --port $PORT
