from fastapi import FastAPI
import os

app = FastAPI(title="Oviya API")

OVIYA_DATA_ROOT = os.environ.get("OVIYA_DATA_ROOT", "oviya-corpus/normalized/")

@app.get("/healthz")
def health():
    """
    Return the service health status and configured data root.
    
    Returns:
        dict: A mapping with keys:
            - "status": the string "ok".
            - "data_root": the configured OVIYA_DATA_ROOT path.
    """
    return {"status": "ok", "data_root": OVIYA_DATA_ROOT}

@app.post("/empathy")
def empathy(payload: dict):
    # Placeholder demo response for fundraising/demo purposes
    """
    Return a canned empathetic reply intended for demo or fundraising purposes.
    
    Parameters:
        payload (dict): Incoming request body; accepted but not inspected or used.
    
    Returns:
        dict: A dictionary with a single `reply` key containing a static empathetic message.
    """
    return {"reply": "I hear you. That sounds difficult. I'm here with you."}