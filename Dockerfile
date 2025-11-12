FROM python:3.10-slim

# 环境变量设置
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 安装系统依赖（这步非常关键）
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsox-dev \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# 安装 Python 依赖
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目
COPY . .

# 暴露端口
EXPOSE 8080

# 启动 FastAPI 服务
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
