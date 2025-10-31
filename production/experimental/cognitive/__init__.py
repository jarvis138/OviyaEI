"""
Experimental Cognitive Components
================================

Alternative personality computation and prosody systems.
"""

from experimental import register_component
from typing import TYPE_CHECKING

# Import and register experimental cognitive components
try:
    from .personality_system import FivePillarPersonality
    register_component("personality_system", lambda: FivePillarPersonality())
except ImportError:
    pass

try:
    from .prosody_engine import ProsodyEngine
    register_component("prosody_engine", lambda: ProsodyEngine())
except ImportError:
    pass

try:
    from .relationship_memory import RelationshipMemorySystem
    register_component("relationship_memory", lambda: RelationshipMemorySystem())
except ImportError:
    pass

if TYPE_CHECKING:
    from .personality_system import FivePillarPersonality
    from .prosody_engine import ProsodyEngine
    from .relationship_memory import RelationshipMemorySystem
