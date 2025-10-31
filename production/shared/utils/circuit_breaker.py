"""
Circuit Breaker Pattern for Experimental Components
===================================================

Provides failure isolation and graceful degradation for experimental features.
Automatically disables failing components and enables recovery mechanisms.
"""

import time
import asyncio
from typing import Dict, Any, Optional, Callable
from enum import Enum
from contextlib import asynccontextmanager

class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, requests rejected
    HALF_OPEN = "half_open"  # Testing recovery

class CircuitBreaker:
    """Circuit breaker for experimental components"""

    def __init__(
        self,
        component_name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 60.0,
        expected_exception: Exception = Exception
    ):
        self.component_name = component_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception

        self.failures = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED

    def should_attempt(self) -> bool:
        """Check if request should be attempted"""
        if self.state == CircuitState.CLOSED:
            return True
        elif self.state == CircuitState.OPEN:
            if self._should_transition_to_half_open():
                self.state = CircuitState.HALF_OPEN
                return True
            return False
        elif self.state == CircuitState.HALF_OPEN:
            return True
        return False

    def record_success(self):
        """Record successful operation"""
        if self.state == CircuitState.HALF_OPEN:
            # Recovery successful
            self._reset()
        # For CLOSED state, no action needed

    def record_failure(self, exception: Exception = None):
        """Record failed operation"""
        self.failures += 1
        self.last_failure_time = time.time()

        if self.failures >= self.failure_threshold:
            self.state = CircuitState.OPEN
            print(f"🔴 Circuit breaker OPEN for {self.component_name} after {self.failures} failures")

    def _should_transition_to_half_open(self) -> bool:
        """Check if we should try recovery"""
        if self.last_failure_time is None:
            return True
        return (time.time() - self.last_failure_time) >= self.recovery_timeout

    def _reset(self):
        """Reset circuit breaker to closed state"""
        self.failures = 0
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        print(f"🟢 Circuit breaker CLOSED for {self.component_name} - recovered")

    def get_status(self) -> Dict[str, Any]:
        """Get circuit breaker status"""
        return {
            "component": self.component_name,
            "state": self.state.value,
            "failures": self.failures,
            "last_failure": self.last_failure_time,
            "next_attempt": self.last_failure_time + self.recovery_timeout if self.last_failure_time else None
        }

class GracefulDegradationManager:
    """Manages graceful degradation across experimental components"""

    def __init__(self):
        self.circuit_breakers = {}
        self.fallback_components = {}
        self.performance_thresholds = {}

    def register_component(
        self,
        component_name: str,
        component_factory: Callable,
        fallback_factory: Optional[Callable] = None,
        failure_threshold: int = 5,
        performance_threshold_ms: float = 150.0
    ):
        """Register a component with circuit breaker and fallback"""
        self.circuit_breakers[component_name] = CircuitBreaker(
            component_name=component_name,
            failure_threshold=failure_threshold
        )

        if fallback_factory:
            self.fallback_components[component_name] = fallback_factory

        self.performance_thresholds[component_name] = performance_threshold_ms

    def get_component(self, component_name: str, context=None):
        """Get a component with circuit breaker protection"""
        breaker = self.circuit_breakers.get(component_name)
        if not breaker:
            raise ValueError(f"No circuit breaker registered for {component_name}")

        if not breaker.should_attempt():
            # Try fallback if available
            fallback = self.fallback_components.get(component_name)
            if fallback:
                print(f"🔄 Using fallback for {component_name}")
                return fallback()
            else:
                raise RuntimeError(f"Component {component_name} circuit is open and no fallback available")

        try:
            # Get the actual component (this would be implemented by subclasses)
            component = self._get_component_instance(component_name, context)

            # Test component health
            start_time = time.time()
            health_check = self._perform_health_check(component)
            latency = (time.time() - start_time) * 1000

            if health_check and latency < self.performance_thresholds.get(component_name, 150.0):
                breaker.record_success()
                return component
            else:
                breaker.record_failure()
                raise RuntimeError(f"Component {component_name} health check failed")

        except Exception as e:
            breaker.record_failure(e)
            # Try fallback
            fallback = self.fallback_components.get(component_name)
            if fallback:
                print(f"🔄 Using fallback for {component_name} after error: {e}")
                return fallback()
            raise e

    def _get_component_instance(self, component_name: str, context=None):
        """Get actual component instance - to be implemented by subclasses"""
        # This would be implemented to actually instantiate components
        # For now, return a mock
        return MockComponent(component_name)

    def _perform_health_check(self, component) -> bool:
        """Perform basic health check on component"""
        try:
            # Basic health check - component should respond
            if hasattr(component, 'health_check'):
                return component.health_check()
            else:
                # Basic check - component exists and is callable
                return component is not None
        except Exception:
            return False

    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system health status"""
        status = {
            "circuit_breakers": {},
            "active_components": [],
            "failed_components": [],
            "fallback_active": []
        }

        for name, breaker in self.circuit_breakers.items():
            breaker_status = breaker.get_status()
            status["circuit_breakers"][name] = breaker_status

            if breaker.state == CircuitState.CLOSED:
                status["active_components"].append(name)
            elif breaker.state == CircuitState.OPEN:
                status["failed_components"].append(name)

            if name in self.fallback_components and breaker.state != CircuitState.CLOSED:
                status["fallback_active"].append(name)

        return status

class MockComponent:
    """Mock component for testing"""

    def __init__(self, name: str):
        self.name = name

    def health_check(self) -> bool:
        return True

    def process(self, input_data):
        return f"Mock processed: {input_data}"

# Global graceful degradation manager
_graceful_degradation = GracefulDegradationManager()

def get_graceful_degradation_manager() -> GracefulDegradationManager:
    """Get the global graceful degradation manager"""
    return _graceful_degradation

def register_experimental_component(
    component_name: str,
    component_factory: Callable,
    fallback_factory: Optional[Callable] = None,
    failure_threshold: int = 5,
    performance_threshold_ms: float = 150.0
):
    """Register an experimental component with circuit breaker protection"""
    _graceful_degradation.register_component(
        component_name=component_name,
        component_factory=component_factory,
        fallback_factory=fallback_factory,
        failure_threshold=failure_threshold,
        performance_threshold_ms=performance_threshold_ms
    )

@asynccontextmanager
async def experimental_context(component_name: str, context=None):
    """Context manager for safe experimental component usage"""
    try:
        component = _graceful_degradation.get_component(component_name, context)
        yield component
    except Exception as e:
        print(f"⚠️ Experimental component {component_name} failed: {e}")
        # Could log to monitoring system here
        raise
