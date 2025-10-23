from fastapi import FastAPI
import os

app = FastAPI(title="Oviya API")

OVIYA_DATA_ROOT = os.environ.get("OVIYA_DATA_ROOT", "oviya-corpus/normalized/")

@app.get("/healthz")
def health():
    return {"status": "ok", "data_root": OVIYA_DATA_ROOT}

@app.post("/empathy")
def empathy(payload: dict):
    # Placeholder demo response for fundraising/demo purposes
    return {"reply": "I hear you. That sounds difficult. I'm here with you."}




