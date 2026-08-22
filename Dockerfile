FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY afl ./afl
COPY serve ./serve
COPY scripts ./scripts
COPY config ./config

EXPOSE 8000 8501

CMD ["uvicorn", "serve.api:app", "--host", "0.0.0.0", "--port", "8000"]
