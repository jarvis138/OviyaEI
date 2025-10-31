"""
Experimental Safety Components
==============================

Enhanced safety routing and content filtering systems.
"""

from experimental import register_component
from typing import TYPE_CHECKING

# Import and register experimental safety components
try:
    from .safety_router import SafetyRouter
    register_component("safety_router", lambda: SafetyRouter({}))
except ImportError:
    pass

if TYPE_CHECKING:
    from .safety_router import SafetyRouter
