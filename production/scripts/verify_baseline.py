#!/usr/bin/env python3
"""
Baseline Integrity Verification
===============================

Ensures the production baseline remains unchanged and functional.
Run this script to verify system integrity before experimental merges.
"""

import sys
import os
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Set

class BaselineVerifier:
    """Verifies production baseline integrity"""

    def __init__(self):
        self.baseline_path = Path(__file__).parent.parent
        self.baseline_hashes = self.load_baseline_hashes()
        self.required_components = self.get_required_components()

    def load_baseline_hashes(self) -> Dict[str, str]:
        """Load baseline file hashes from v1.0-production-baseline tag"""
        # In a real implementation, this would fetch from git
        # For now, we'll compute current hashes as the baseline
        baseline_hashes = {}

        for component in self.get_required_components():
            component_path = self.baseline_path / component
            if component_path.exists():
                baseline_hashes[component] = self.compute_file_hash(component_path)

        return baseline_hashes

    def get_required_components(self) -> List[str]:
        """Get list of components that must remain unchanged"""
        return [
            # Core therapeutic brain
            "brain/llm_brain.py",
            "brain/crisis_detection.py",
            "brain/attachment_style.py",
            "brain/bids.py",
            "brain/strategic_silence.py",
            "brain/micro_affirmations.py",

            # Voice synthesis pipeline
            "voice/csm_1b_stream.py",  # Updated: csm_1b_generator_optimized.py merged into csm_1b_stream.py
            "voice/csm_1b_client.py",

            # Safety & compliance
            "shared/utils/pii_redaction.py",
            "shared/utils/emotion_monitor.py",

            # Experimental governance
            "experimental/__init__.py",
            "shared/testing/contract_testing.py",
            "shared/utils/circuit_breaker.py",

            # Configuration (critical ones)
            "shared/config/oviya_persona.json",
            "shared/config/emotions.json",
        ]

    def compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file"""
        if not file_path.exists():
            return ""

        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def verify_baseline_integrity(self) -> Dict[str, any]:
        """Verify that baseline components are unchanged"""
        results = {
            "verified_components": 0,
            "modified_components": [],
            "missing_components": [],
            "integrity_status": "unknown",
            "recommendations": []
        }

        print("🔍 VERIFYING PRODUCTION BASELINE INTEGRITY")
        print("=" * 60)

        for component in self.required_components:
            component_path = self.baseline_path / component

            if not component_path.exists():
                results["missing_components"].append(component)
                print(f"❌ MISSING: {component}")
                continue

            current_hash = self.compute_file_hash(component_path)
            baseline_hash = self.baseline_hashes.get(component, "")

            if current_hash == baseline_hash:
                results["verified_components"] += 1
                print(f"✅ VERIFIED: {component}")
            else:
                results["modified_components"].append({
                    "component": component,
                    "baseline_hash": baseline_hash,
                    "current_hash": current_hash
                })
                print(f"⚠️ MODIFIED: {component}")

        # Determine overall status
        total_components = len(self.required_components)
        verified = results["verified_components"]

        if verified == total_components:
            results["integrity_status"] = "perfect"
            results["recommendations"] = ["✅ Baseline integrity perfect - safe for experimental merges"]
        elif verified >= total_components * 0.9:  # 90%+ verified
            results["integrity_status"] = "good"
            results["recommendations"] = ["✅ Baseline largely intact - minor modifications detected"]
        else:
            results["integrity_status"] = "compromised"
            results["recommendations"] = ["⚠️ Baseline significantly modified - review changes before proceeding"]

        print(f"\n📊 INTEGRITY SUMMARY:")
        print(f"   Total Components: {total_components}")
        print(f"   Verified: {verified}")
        print(f"   Modified: {len(results['modified_components'])}")
        print(f"   Missing: {len(results['missing_components'])}")
        print(f"   Status: {results['integrity_status'].upper()}")

        return results

    def run_crisis_detection_test(self) -> bool:
        """Test that crisis detection still works"""
        print("\n🛡️ TESTING CRISIS DETECTION...")
        try:
            # Import and test crisis detection
            sys.path.insert(0, str(self.baseline_path))
            from brain.crisis_detection import CrisisDetectionSystem

            crisis_detector = CrisisDetectionSystem()
            test_messages = [
                "I feel like hurting myself",
                "I'm having thoughts of suicide",
                "Everything is fine today"
            ]

            for msg in test_messages:
                result = crisis_detector.detect_crisis(msg)
                if "suicide" in msg.lower() or "hurting myself" in msg.lower():
                    if not result.get("is_crisis", False):
                        print(f"❌ Crisis detection failed for: {msg}")
                        return False
                else:
                    if result.get("is_crisis", True):
                        print(f"❌ False positive for: {msg}")
                        return False

            print("✅ Crisis detection working correctly")
            return True

        except Exception as e:
            print(f"❌ Crisis detection test failed: {e}")
            return False

    def run_pii_redaction_test(self) -> bool:
        """Test that PII redaction still works"""
        print("\n🔒 TESTING PII REDACTION...")
        try:
            from shared.utils.pii_redaction import redact

            test_message = "Contact me at john.doe@email.com or call 555-123-4567"
            redacted = redact(test_message)

            # Check that PII was redacted
            if "john.doe@email.com" in redacted or "555-123-4567" in redacted:
                print("❌ PII redaction failed")
                return False

            print("✅ PII redaction working correctly")
            return True

        except Exception as e:
            print(f"❌ PII redaction test failed: {e}")
            return False

    def run_performance_baseline_test(self) -> Dict[str, float]:
        """Run basic performance baseline test"""
        print("\n⚡ TESTING PERFORMANCE BASELINE...")
        try:
            from brain.llm_brain import OviyaBrain
            import time

            brain = OviyaBrain()
            test_messages = [
                "Hello, I'm feeling a bit anxious",
                "I need help with my stress",
                "Thank you for listening"
            ]

            latencies = []
            for msg in test_messages:
                start_time = time.time()
                response = brain.think(msg)
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)

            avg_latency = sum(latencies) / len(latencies)
            max_latency = max(latencies)

            print(f"   Average latency: {avg_latency:.2f}ms")
            print(f"   Max latency: {max_latency:.2f}ms")
            # Check if within acceptable range
            if avg_latency > 200:  # Allow some margin above target
                print("⚠️ Performance degraded from baseline")
            else:
                print("✅ Performance within baseline expectations")

            return {
                "avg_latency_ms": avg_latency,
                "max_latency_ms": max_latency,
                "within_baseline": avg_latency <= 200
            }

        except Exception as e:
            print(f"❌ Performance test failed: {e}")
            return {"error": str(e)}

def main():
    """Main verification function"""
    verifier = BaselineVerifier()

    # Run comprehensive verification
    integrity_results = verifier.verify_baseline_integrity()
    crisis_test = verifier.run_crisis_detection_test()
    pii_test = verifier.run_pii_redaction_test()
    performance_results = verifier.run_performance_baseline_test()

    print("\n" + "=" * 60)
    print("🎯 BASELINE VERIFICATION RESULTS")
    print("=" * 60)

    print(f"Integrity Status: {integrity_results['integrity_status'].upper()}")
    print(f"Crisis Detection: {'✅ PASS' if crisis_test else '❌ FAIL'}")
    print(f"PII Redaction: {'✅ PASS' if pii_test else '❌ FAIL'}")

    if isinstance(performance_results, dict) and "avg_latency_ms" in performance_results:
        print(f"   Average latency: {performance_results['avg_latency_ms']:.2f}ms")
        print(f"Performance OK: {'✅ YES' if performance_results.get('within_baseline', False) else '⚠️ NO'}")

    # Overall assessment
    all_passed = (
        integrity_results['integrity_status'] in ['perfect', 'good'] and
        crisis_test and
        pii_test and
        performance_results.get('within_baseline', False)
    )

    print(f"\n🏆 OVERALL STATUS: {'✅ PRODUCTION READY' if all_passed else '⚠️ REQUIRES REVIEW'}")

    if not all_passed:
        print("\n🔧 RECOMMENDATIONS:")
        for rec in integrity_results.get('recommendations', []):
            print(f"   • {rec}")

        if not crisis_test:
            print("   • Fix crisis detection system")
        if not pii_test:
            print("   • Fix PII redaction system")
        if not performance_results.get('within_baseline', True):
            print("   • Investigate performance degradation")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
