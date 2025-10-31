"""
Oviya EI Experimental Components
===============================

This namespace contains experimental features and alternative implementations
that are under evaluation for potential promotion to production.

All components here are:
- Feature-flagged (disabled by default)
- Safety-validated (crisis detection, PII redaction maintained)
- Performance-benchmarked (< 150ms p95 latency target)
- Contract-tested (input/output compatibility guaranteed)

Components graduate to production when they meet graduation criteria:
- Safety parity: 0 false negatives in crisis/PII detection
- Performance: ≤ 150ms p95, < 0.5% error rate
- Quality: ≥ 90% therapeutic effectiveness score
- Testing: 95%+ integration test coverage

Usage:
    from experimental import get_component
    component = get_component('audio_pipeline', context)
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

__version__ = "0.1.0-experimental"

@dataclass
class FeatureContext:
    """Context for feature flag evaluation"""
    user_id: str
    session_id: str
    user_load: int = 0
    system_load: float = 0.0
    risk_level: str = "low"
    experimental_enabled: bool = False
    performance_mode: str = "balanced"  # "performance", "balanced", "safety"

    @classmethod
    def from_env(cls) -> 'FeatureContext':
        """Create context from environment variables"""
        return cls(
            user_id=os.getenv("OVIYA_USER_ID", "anonymous"),
            session_id=os.getenv("OVIYA_SESSION_ID", "default"),
            user_load=int(os.getenv("OVIYA_USER_LOAD", "0")),
            system_load=float(os.getenv("OVIYA_SYSTEM_LOAD", "0.0")),
            risk_level=os.getenv("OVIYA_RISK_LEVEL", "low"),
            experimental_enabled=os.getenv("OVIYA_EXPERIMENTAL_ENABLED", "false").lower() == "true",
            performance_mode=os.getenv("OVIYA_PERFORMANCE_MODE", "balanced")
        )

class ExperimentalRegistry:
    """Registry for experimental components"""

    def __init__(self):
        self._components = {}
        self._feature_flags = self._load_feature_flags()

    def _load_feature_flags(self) -> Dict[str, bool]:
        """Load feature flags from configuration"""
        # Default feature flags (all disabled)
        flags = {
            # Audio components
            "experimental_audio_pipeline": False,
            "experimental_acoustic_emotion": False,
            "experimental_whisper_turbo": False,

            # Cognitive components
            "experimental_personality_system": False,
            "experimental_prosody_engine": False,
            "experimental_relationship_memory": False,

            # Voice components
            "experimental_csm_server_real": False,
            "experimental_csm_rvq_streaming": False,
            "experimental_voice_server_webrtc": False,

            # Safety components
            "experimental_safety_router": False,

            # Infrastructure
            "experimental_realtime_conversation": False,
            "experimental_monitoring_enhanced": False,
        }

        # Override from environment
        for flag in flags.keys():
            env_var = f"OVIYA_{flag.replace('experimental_', '').upper()}"
            if env_var in os.environ:
                flags[flag] = os.getenv(env_var, "false").lower() == "true"

        return flags

    def register_component(self, name: str, component_class: Any):
        """Register an experimental component"""
        self._components[name] = component_class

    def get_component(self, name: str, context: FeatureContext) -> Optional[Any]:
        """Get a component if its feature flag is enabled and context allows"""
        if not self._should_enable_component(name, context):
            return None

        component_class = self._components.get(name)
        if component_class:
            try:
                return component_class()
            except Exception as e:
                print(f"⚠️ Failed to initialize experimental component {name}: {e}")
                return None

        return None

    def _should_enable_component(self, name: str, context: FeatureContext) -> bool:
        """Determine if component should be enabled based on context"""
        flag_name = f"experimental_{name}"

        # Check feature flag
        if not self._feature_flags.get(flag_name, False):
            return False

        # Check context constraints
        if not context.experimental_enabled:
            return False

        # Performance-based constraints
        if context.performance_mode == "safety":
            # Only enable safety-enhancing components
            safety_components = ["safety_router", "acoustic_emotion"]
            return name in safety_components

        if context.system_load > 0.8:
            # High load - only enable lightweight components
            lightweight_components = ["relationship_memory", "safety_router"]
            return name in lightweight_components

        # Risk-based constraints
        if context.risk_level == "high":
            # Only enable thoroughly tested components
            tested_components = ["safety_router", "relationship_memory"]
            return name in tested_components

        return True

    def list_available_components(self) -> Dict[str, bool]:
        """List all registered components and their enabled status"""
        return {
            name: self._feature_flags.get(f"experimental_{name}", False)
            for name in self._components.keys()
        }

# Global registry instance
_registry = ExperimentalRegistry()

def get_component(name: str, context: Optional[FeatureContext] = None) -> Optional[Any]:
    """Get an experimental component if available and enabled"""
    if context is None:
        context = FeatureContext.from_env()
    return _registry.get_component(name, context)

def register_component(name: str, component_class: Any):
    """Register an experimental component"""
    _registry.register_component(name, component_class)

def list_experimental_features() -> Dict[str, bool]:
    """List all experimental features and their status"""
    return _registry.list_available_components()

# Export additional utilities
try:
    from ..shared.utils.circuit_breaker import get_graceful_degradation_manager
except ImportError:
    get_graceful_degradation_manager = None

try:
    from ..experimental.integration.phased_rollout import get_phased_integrator
except ImportError:
    get_phased_integrator = None

# Import graceful degradation and register experimental components
try:
    from ..shared.utils.circuit_breaker import (
        get_graceful_degradation_manager,
        register_experimental_component
    )
    _graceful_degradation_manager = get_graceful_degradation_manager()
except ImportError:
    _graceful_degradation_manager = None

# Import submodules for registration
try:
    from . import audio
    from . import cognitive
    from . import voice
    from . import safety
    from . import infrastructure
except ImportError:
    # Submodules may not be fully implemented yet
    pass
