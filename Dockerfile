FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# 시스템 deps (필요 시)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

# 파이썬 deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 소스
COPY . .

# Cloud Run이 PORT 환경변수로 포트를 넘겨줌
ENV PORT=8080

# Gunicorn + Uvicorn worker로 FastAPI 실행 (워커 1개 고정)
CMD exec gunicorn -w 1 -k uvicorn.workers.UvicornWorker app.main:app \
    --bind 0.0.0.0:${PORT} --timeout 120