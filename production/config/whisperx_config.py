"""
WhisperX Configuration for Oviya
RTX 5880 Ada Instance
"""

# WhisperX API endpoint (Cloudflare tunnel)
WHISPERX_URL = "https://msgid-enquiries-williams-lands.trycloudflare.com"

# API endpoints
WHISPERX_HEALTH = f"{WHISPERX_URL}/health"
WHISPERX_TRANSCRIBE = f"{WHISPERX_URL}/transcribe"

# Configuration
WHISPERX_CONFIG = {
    "batch_size": 8,
    "language": "en",
    "compute_type": "float16"
}

