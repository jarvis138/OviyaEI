"""
Centralized Service URLs Configuration
Cloudflare Tunnel URLs for Oviya Services
RTX 5880 Ada Instance
"""

# Cloudflare Tunnel URLs (RTX 5880 Ada - VastAI)
OLLAMA_URL = "https://prime-show-visit-lock.trycloudflare.com/api/generate"
CSM_URL = "https://astronomy-initiative-paso-cream.trycloudflare.com/generate"
WHISPERX_URL = "https://msgid-enquiries-williams-lands.trycloudflare.com"

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


