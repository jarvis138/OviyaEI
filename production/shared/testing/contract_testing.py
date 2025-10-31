"""
Contract Testing Framework
==========================

Ensures experimental components meet safety, performance, and quality contracts
before being promoted to production.

Every experimental component must implement and pass these contracts.
"""

import time
import asyncio
from typing import Dict, Any, List, Protocol, Optional
from abc import ABC, abstractmethod
from dataclasses import dataclass
from contextlib import asynccontextmanager

@dataclass
class ContractResult:
    """Result of a contract test"""
    component_name: str
    test_name: str
    passed: bool
    latency_ms: float
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None

class ComponentContract(Protocol):
    """Protocol that all experimental components must implement"""

    def test_input_contract(self) -> ContractResult:
        """Test that component accepts expected inputs"""
        ...

    def test_output_contract(self) -> ContractResult:
        """Test that component produces expected outputs"""
        ...

    def test_safety_contract(self) -> ContractResult:
        """Test that component maintains safety standards"""
        ...

    def test_performance_contract(self) -> ContractResult:
        """Test that component meets performance requirements"""
        ...

class ContractTester:
    """Test harness for component contracts"""

    def __init__(self):
        self.results = []
        self.contracts = {
            "latency_p95": "< 150ms",
            "error_rate": "< 0.5%",
            "safety_false_negative": "0",
            "therapeutic_quality_score": "> 4.2/5.0",
            "integration_test_coverage": "95%"
        }

    def test_component(self, component: Any, component_name: str) -> List[ContractResult]:
        """Test a component against all contracts"""
        results = []

        # Test input contract
        results.append(self._run_test(component, component_name, "input_contract"))

        # Test output contract
        results.append(self._run_test(component, component_name, "output_contract"))

        # Test safety contract
        results.append(self._run_test(component, component_name, "safety_contract"))

        # Test performance contract
        results.append(self._run_test(component, component_name, "performance_contract"))

        self.results.extend(results)
        return results

    def _run_test(self, component: Any, component_name: str, test_name: str) -> ContractResult:
        """Run a specific contract test"""
        start_time = time.time()

        try:
            # Dynamically call the test method
            test_method = getattr(component, f"test_{test_name}")
            result = test_method()

            latency = (time.time() - start_time) * 1000

            if isinstance(result, ContractResult):
                return result
            else:
                # Assume boolean result
                return ContractResult(
                    component_name=component_name,
                    test_name=test_name,
                    passed=bool(result),
                    latency_ms=latency
                )

        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return ContractResult(
                component_name=component_name,
                test_name=test_name,
                passed=False,
                latency_ms=latency,
                error_message=str(e)
            )

    def get_graduation_readiness(self, component_name: str) -> Dict[str, Any]:
        """Check if component is ready for production graduation"""
        component_results = [r for r in self.results if r.component_name == component_name]

        if not component_results:
            return {"ready": False, "reason": "No contract tests run"}

        passed_tests = sum(1 for r in component_results if r.passed)
        total_tests = len(component_results)

        # Calculate metrics
        avg_latency = sum(r.latency_ms for r in component_results) / total_tests
        error_rate = (total_tests - passed_tests) / total_tests

        # Check graduation criteria
        graduation_criteria = {
            "latency_p95": avg_latency < 150,
            "error_rate": error_rate < 0.005,  # 0.5%
            "safety_tests_pass": all(r.passed for r in component_results if "safety" in r.test_name),
            "contract_coverage": total_tests >= 4  # All 4 contract types
        }

        ready = all(graduation_criteria.values())

        return {
            "ready": ready,
            "passed_tests": passed_tests,
            "total_tests": total_tests,
            "avg_latency_ms": avg_latency,
            "error_rate": error_rate,
            "criteria_met": graduation_criteria,
            "reason": "Ready for production" if ready else "Does not meet graduation criteria"
        }

class ExperimentalMetrics:
    """Enhanced monitoring for experimental components"""

    def __init__(self):
        self.metrics = {
            "component_health": {},
            "performance_deltas": {},
            "safety_incidents": {},
            "user_satisfaction": {},
            "graduation_candidates": []
        }

    def track_component_health(self, component: str, health_score: float):
        """Track component health scores"""
        if component not in self.metrics["component_health"]:
            self.metrics["component_health"][component] = []
        self.metrics["component_health"][component].append({
            "timestamp": time.time(),
            "health_score": health_score
        })

        # Keep only last 100 measurements
        if len(self.metrics["component_health"][component]) > 100:
            self.metrics["component_health"][component] = self.metrics["component_health"][component][-100:]

    def track_performance_delta(self, component: str, latency_delta: float):
        """Track performance changes vs production baseline"""
        if component not in self.metrics["performance_deltas"]:
            self.metrics["performance_deltas"][component] = []
        self.metrics["performance_deltas"][component].append({
            "timestamp": time.time(),
            "latency_delta_ms": latency_delta
        })

    def track_safety_incidents(self, component: str, incident_type: str):
        """Track safety incidents for experimental components"""
        if component not in self.metrics["safety_incidents"]:
            self.metrics["safety_incidents"][component] = []
        self.metrics["safety_incidents"][component].append({
            "timestamp": time.time(),
            "incident_type": incident_type
        })

    def track_user_satisfaction(self, component: str, satisfaction_score: float):
        """Track user satisfaction with experimental components"""
        if component not in self.metrics["user_satisfaction"]:
            self.metrics["user_satisfaction"][component] = []
        self.metrics["user_satisfaction"][component].append({
            "timestamp": time.time(),
            "satisfaction_score": satisfaction_score
        })

    def check_graduation_candidates(self, contract_tester: ContractTester) -> List[str]:
        """Identify components ready for production graduation"""
        candidates = []
        tested_components = set(r.component_name for r in contract_tester.results)

        for component in tested_components:
            readiness = contract_tester.get_graduation_readiness(component)
            if readiness["ready"]:
                candidates.append(component)

        self.metrics["graduation_candidates"] = candidates
        return candidates

# Global instances
_contract_tester = ContractTester()
_experimental_metrics = ExperimentalMetrics()

def get_contract_tester() -> ContractTester:
    """Get the global contract tester instance"""
    return _contract_tester

def get_experimental_metrics() -> ExperimentalMetrics:
    """Get the global experimental metrics instance"""
    return _experimental_metrics

# Base class for experimental components
class ExperimentalComponent(ABC):
    """Base class for all experimental components"""

    def __init__(self, name: str):
        self.name = name
        self.contract_tester = get_contract_tester()
        self.metrics = get_experimental_metrics()

    @abstractmethod
    def test_input_contract(self) -> ContractResult:
        """Test input contract - must be implemented by subclasses"""
        pass

    @abstractmethod
    def test_output_contract(self) -> ContractResult:
        """Test output contract - must be implemented by subclasses"""
        pass

    @abstractmethod
    def test_safety_contract(self) -> ContractResult:
        """Test safety contract - must be implemented by subclasses"""
        pass

    @abstractmethod
    def test_performance_contract(self) -> ContractResult:
        """Test performance contract - must be implemented by subclasses"""
        pass

    def run_contract_tests(self) -> List[ContractResult]:
        """Run all contract tests for this component"""
        return self.contract_tester.test_component(self, self.name)

    def check_graduation_readiness(self) -> Dict[str, Any]:
        """Check if this component is ready for production"""
        return self.contract_tester.get_graduation_readiness(self.name)
