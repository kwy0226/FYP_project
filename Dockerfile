FROM python:3.10-slim

# 防止 pyc 缓存 & log 丢失
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目代码
COPY . .

# 暴露 Cloud Run 默认端口
EXPOSE 8080

# 启动 FastAPI
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "$PORT"]
