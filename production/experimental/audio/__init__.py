"""
Experimental Audio Components
============================

Alternative audio processing pipelines and real-time audio systems.
"""

from experimental import register_component
from typing import TYPE_CHECKING

# Import and register experimental audio components
try:
    from .audio_input import AudioInputProcessor, VoiceActivityDetector, SpeechToTextProcessor
    register_component("audio_pipeline", lambda: AudioInputProcessor())
    register_component("voice_activity_detector", lambda: VoiceActivityDetector())
    register_component("speech_to_text", lambda: SpeechToTextProcessor())
except ImportError:
    pass

try:
    from .acoustic_emotion_detector import AcousticEmotionDetector
    register_component("acoustic_emotion", lambda: AcousticEmotionDetector())
except ImportError:
    pass

try:
    from .whisper_client import WhisperTurboClient
    register_component("whisper_turbo", lambda: WhisperTurboClient())
except ImportError:
    pass

if TYPE_CHECKING:
    from .audio_input import AudioInputProcessor, VoiceActivityDetector, SpeechToTextProcessor
    from .acoustic_emotion_detector import AcousticEmotionDetector
    from .whisper_client import WhisperTurboClient
