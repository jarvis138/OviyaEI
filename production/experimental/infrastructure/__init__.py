"""
Experimental Infrastructure Components
====================================

Alternative conversation systems and WebRTC infrastructure.
"""

from experimental import register_component
from typing import TYPE_CHECKING

# Import and register experimental infrastructure components
try:
    from .realtime_conversation import RealTimeConversation
    register_component("realtime_conversation", lambda: RealTimeConversation())
except ImportError:
    pass

try:
    from .voice_server_webrtc import VoiceServerWebRTC
    register_component("voice_server_webrtc", lambda: VoiceServerWebRTC())
except ImportError:
    pass

try:
    from .verify_csm_1b import CSMVerifier
    register_component("csm_verifier", lambda: CSMVerifier())
except ImportError:
    pass

if TYPE_CHECKING:
    from .realtime_conversation import RealTimeConversation
    from .voice_server_webrtc import VoiceServerWebRTC
    from .verify_csm_1b import CSMVerifier
