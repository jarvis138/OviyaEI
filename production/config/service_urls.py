"""
Centralized Service URLs Configuration
Cloudflare Tunnel URLs for Oviya Services
RTX 5880 Ada Instance
"""

# Cloudflare Tunnel URLs (RTX 5880 Ada - VastAI) - overridable via env
import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
CSM_URL = os.getenv("CSM_URL", "http://localhost:19517/generate")
CSM_STREAM_URL = os.getenv("CSM_STREAM_URL", "http://localhost:19517/generate/stream")
WHISPERX_URL = os.getenv("WHISPERX_URL", "http://localhost:1111")

# WhisperX API endpoints
WHISPERX_HEALTH = f"{WHISPERX_URL}/health"
WHISPERX_TRANSCRIBE = f"{WHISPERX_URL}/transcribe"

# Legacy URLs (for reference)
# OLLAMA_URL_OLD = "https://0da53357866ee5.lhr.life/api/generate"  # localhost.run (expired)
# CSM_URL_OLD = "https://astronomy-initiative-paso-cream.trycloudflare.com/generate"  # ngrok

# Service Ports (on VastAI instance)
# - Ollama:   Port 11434
# - CSM:      Port 19517
# - WhisperX: Port 1111


