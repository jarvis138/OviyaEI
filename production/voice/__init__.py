"""
Voice Engine Module
==================

Core voice synthesis and processing systems for Oviya EI.
"""

# Import voice configuration
try:
    from ..config.production_voice_config import VOICE_CONFIG
    __all__ = ['VOICE_CONFIG']
except ImportError:
    VOICE_CONFIG = {
        "primary_engine": "csm_1b",
        "fallback_engine": "openvoice",
        "sample_rate": 24000,
        "channels": 1,
        "dtype": "float32"
    }
    __all__ = ['VOICE_CONFIG']
