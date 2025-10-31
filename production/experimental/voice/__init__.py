"""
Experimental Voice Components
============================

Alternative voice synthesis servers and WebRTC systems.
"""

from experimental import register_component
from typing import TYPE_CHECKING

# Import and register experimental voice components
try:
    from .csm_server_real import CSMServerReal
    register_component("csm_server_real", lambda: CSMServerReal())
except ImportError:
    pass

try:
    from .csm_server_real_rvq import CSMServerRealRVQ
    register_component("csm_rvq_streaming", lambda: CSMServerRealRVQ())
except ImportError:
    pass

try:
    from .voice_csm_integration import CSMIntegration
    register_component("voice_csm_integration", lambda: CSMIntegration())
except ImportError:
    pass

if TYPE_CHECKING:
    from .csm_server_real import CSMServerReal
    from .csm_server_real_rvq import CSMServerRealRVQ
    from .voice_csm_integration import CSMIntegration
