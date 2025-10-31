#!/usr/bin/env python3
"""
HuggingFace Configuration for Oviya EI

Stores authentication tokens and model configuration for gated repositories.
Uses environment variables for security.
"""

import os

# HuggingFace Token for CSM-1B access (from environment variable)
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")

# Model configurations
CSM_1B_CONFIG = {
    "model_id": "sesame/csm-1b",
    "token": HUGGINGFACE_TOKEN,
    "device": "cuda" if __import__("torch").cuda.is_available() else "cpu"
}

def get_huggingface_token():
    """Get HuggingFace authentication token from environment"""
    token = os.getenv("HUGGINGFACE_TOKEN", "")
    if not token:
        raise ValueError(
            "HUGGINGFACE_TOKEN environment variable not set. "
            "Please set it with: export HUGGINGFACE_TOKEN='your_token_here'"
        )
    return token

def get_csm_config():
    """Get CSM-1B model configuration"""
    config = CSM_1B_CONFIG.copy()
    config["token"] = get_huggingface_token()  # Ensure fresh token from env
    return config