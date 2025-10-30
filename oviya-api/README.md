# Oviya API

Minimal FastAPI service exposing a demo empathy endpoint.

## Run locally

```bash
pip install -r requirements.txt
export OVIYA_DATA_ROOT=oviya-corpus/normalized/
uvicorn app.main:app --host 0.0.0.0 --port 8080
```

## Docker

```bash
docker build -t oviya-api .
docker run --rm -p 8080:8080 -e OVIYA_DATA_ROOT=oviya-corpus/normalized/ oviya-api
```

## Endpoints
- GET /healthz
- POST /empathy







