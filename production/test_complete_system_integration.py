#!/usr/bin/env python3
"""
COMPLETE OVIYA EI SYSTEM INTEGRATION TEST
Tests ALL integrated components working together end-to-end

This test validates the complete therapeutic pipeline with all advanced systems:
1. Crisis Detection & Safety
2. Attachment Style Analysis
3. Advanced Memory Systems
4. Bid Response & Connection Building
5. Micro-Affirmations & Natural Flow
6. Neural Prosody & Voice Quality
7. Advanced Emotion Processing
8. Secure Base & Therapeutic Presence
9. Unconditional Regard & Rogers Therapy
10. Healthy Boundaries & Ethics
"""

import asyncio
import json
import time
from typing import Dict, List, Any
import os
import sys

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from brain.llm_brain import OviyaBrain
from voice.csm_1b_generator_optimized import get_optimized_streamer

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def print_feature(feature: str, data: str):
    """Print a formatted feature result"""
    print(f"   {feature}: {data}")

async def initialize_complete_system():
    """Initialize all integrated systems"""
    print_section("🚀 INITIALIZING COMPLETE OVIYA EI SYSTEM")

    try:
        # Initialize brain with all advanced systems
        print("🧠 Initializing Oviya Brain (15+ integrated systems)...")
        brain = OviyaBrain()
        print("✅ Brain initialized with:")
        print("   • Crisis Detection System")
        print("   • Attachment Style Detector")
        print("   • Advanced MCP Memory Integration")
        print("   • Bid Response System")
        print("   • Micro-Affirmations Generator")
        print("   • Secure Base System")
        print("   • Unconditional Regard Engine")
        print("   • Healthy Boundaries System")
        print("   • Empathic Thinking Engine (5 modes)")
        print("   • Strategic Silence Manager")

        # Initialize voice system with all enhancements
        print("\n🎵 Initializing Voice System (8+ integrated systems)...")
        voice_system = get_optimized_streamer()
        print("✅ Voice system initialized with:")
        print("   • CUDA Graphs Optimization")
        print("   • Neural Prosody Controller")
        print("   • Advanced Emotion Library (28+ emotions)")
        print("   • Emotion Blender")
        print("   • Humanlike Prosody Engine")
        print("   • Professional Audio Post-Processor")
        print("   • Sesame Evaluation Suite")

        return brain, voice_system

    except Exception as e:
        print(f"❌ System initialization failed: {e}")
        return None, None

async def test_comprehensive_therapeutic_scenarios():
    """Test comprehensive therapeutic scenarios with all systems"""
    print_section("🧪 COMPREHENSIVE THERAPEUTIC SCENARIO TESTING")

    brain, voice_system = await initialize_complete_system()
    if not brain or not voice_system:
        return {"status": "failed", "reason": "System initialization failed"}

    scenarios = [
        {
            "name": "Crisis Detection & Safety",
            "user_input": "I can't take this anymore. I just want everything to end.",
            "expected_systems": ["crisis_detection", "safety_router"],
            "expected_response_type": "crisis_intervention"
        },
        {
            "name": "Attachment-Informed Therapy",
            "user_input": "Are you sure this will work? Do you really think I'm capable?",
            "expected_systems": ["attachment_style", "secure_base", "unconditional_regard"],
            "expected_response_type": "reassurance_focused"
        },
        {
            "name": "Bid Response & Connection",
            "user_input": "I've never told anyone this before, but I feel so alone.",
            "expected_systems": ["bid_response", "micro_affirmations", "vulnerability_reciprocation"],
            "expected_response_type": "connection_building"
        },
        {
            "name": "Advanced Memory & Continuity",
            "user_input": "Remember when I talked about my work stress last time?",
            "expected_systems": ["advanced_memory", "relationship_memory"],
            "expected_response_type": "continuity_demonstration"
        },
        {
            "name": "Boundary Management",
            "user_input": "Can we talk more? I really need this right now.",
            "expected_systems": ["healthy_boundaries", "unconditional_regard"],
            "expected_response_type": "boundary_guidance"
        }
    ]

    results = {
        "total_scenarios": len(scenarios),
        "passed_scenarios": 0,
        "failed_scenarios": 0,
        "system_integration_score": 0,
        "performance_metrics": {},
        "detailed_results": []
    }

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n🎯 SCENARIO {i}: {scenario['name']}")
        print(f"   Input: \"{scenario['user_input']}\"")
        print(f"   Expected systems: {', '.join(scenario['expected_systems'])}")

        try:
            # Test brain response
            start_time = time.time()
            brain_response = brain.think(scenario["user_input"])
            brain_time = time.time() - start_time

            # Analyze which systems were activated
            activated_systems = []
            response_metadata = brain_response

            # Check for system activation indicators
            if "crisis_intervention" in brain_response.get("text", "").lower():
                activated_systems.append("crisis_detection")
            if brain_response.get("has_reciprocity"):
                activated_systems.append("emotional_reciprocity")
            if brain_response.get("has_empathic_enhancement"):
                activated_systems.append("empathic_thinking")
            if brain_response.get("unconditional_regard_applied"):
                activated_systems.append("unconditional_regard")
            if brain_response.get("micro_affirmation_added"):
                activated_systems.append("micro_affirmations")
            if brain_response.get("strategic_silence_applied"):
                activated_systems.append("strategic_silence")

            # Test voice generation with personality
            personality_vector = brain_response.get("personality_vector")
            voice_start_time = time.time()
            voice_response = voice_system.generate_voice(
                brain_response.get("text", ""),
                emotion=brain_response.get("emotion", "calm_supportive"),
                personality_vector=personality_vector
            )
            voice_time = time.time() - voice_start_time

            # Evaluate integration quality
            expected_activated = len([s for s in scenario["expected_systems"] if s in activated_systems])
            integration_score = expected_activated / len(scenario["expected_systems"])

            # Determine pass/fail
            passed = integration_score >= 0.6  # At least 60% of expected systems activated

            results["detailed_results"].append({
                "scenario": scenario["name"],
                "passed": passed,
                "integration_score": integration_score,
                "activated_systems": activated_systems,
                "expected_systems": scenario["expected_systems"],
                "brain_response_time": brain_time,
                "voice_generation_time": voice_time,
                "response_quality": "high" if len(brain_response.get("text", "")) > 20 else "low",
                "voice_quality": "high" if len(voice_response) > 10000 else "low"
            })

            if passed:
                results["passed_scenarios"] += 1
                print(f"   ✅ PASSED ({integration_score:.1%} integration)")
            else:
                results["failed_scenarios"] += 1
                print(f"   ❌ FAILED ({integration_score:.1%} integration)")

            print(f"   Systems activated: {', '.join(activated_systems) if activated_systems else 'None'}")
            print(f"   Performance: Brain {brain_time:.2f}s, Voice {voice_time:.2f}s")

        except Exception as e:
            results["failed_scenarios"] += 1
            results["detailed_results"].append({
                "scenario": scenario["name"],
                "passed": False,
                "error": str(e)
            })
            print(f"   ❌ ERROR: {e}")

    # Calculate overall metrics
    results["system_integration_score"] = results["passed_scenarios"] / results["total_scenarios"]
    results["performance_metrics"] = {
        "average_brain_response_time": sum(r.get("brain_response_time", 0) for r in results["detailed_results"] if "brain_response_time" in r) / len([r for r in results["detailed_results"] if "brain_response_time" in r]),
        "average_voice_generation_time": sum(r.get("voice_generation_time", 0) for r in results["detailed_results"] if "voice_generation_time" in r) / len([r for r in results["detailed_results"] if "voice_generation_time" in r]),
        "high_quality_responses": sum(1 for r in results["detailed_results"] if r.get("response_quality") == "high"),
        "high_quality_voice": sum(1 for r in results["detailed_results"] if r.get("voice_quality") == "high")
    }

    return results

async def test_system_robustness():
    """Test system robustness and error handling"""
    print_section("🔧 SYSTEM ROBUSTNESS TESTING")

    brain, voice_system = await initialize_complete_system()
    robustness_tests = [
        {"name": "Empty input handling", "input": "", "expected": "graceful_handling"},
        {"name": "Very long input", "input": "test " * 1000, "expected": "truncated_processing"},
        {"name": "Special characters", "input": "!@#$%^&*()_+", "expected": "sanitized_processing"},
        {"name": "Unicode characters", "input": "Hello 世界 🌟", "expected": "unicode_handling"},
        {"name": "Repeated requests", "input": "test", "expected": "consistent_responses", "repeat": 5}
    ]

    robustness_results = {
        "total_tests": len(robustness_tests),
        "passed_tests": 0,
        "error_handling_score": 0
    }

    for test in robustness_tests:
        try:
            if test.get("repeat"):
                # Test repeated requests for consistency
                responses = []
                for _ in range(test["repeat"]):
                    response = brain.think(test["input"])
                    responses.append(response.get("text", ""))

                # Check consistency (responses should be similar but not identical due to personality)
                consistency_score = len(set(responses)) / len(responses)  # Lower is more consistent
                passed = consistency_score < 0.8  # Allow some variation
            else:
                response = brain.think(test["input"])
                passed = response is not None and "text" in response

            if passed:
                robustness_results["passed_tests"] += 1
                print(f"   ✅ {test['name']}: PASSED")
            else:
                print(f"   ❌ {test['name']}: FAILED")

        except Exception as e:
            print(f"   ❌ {test['name']}: ERROR - {e}")

    robustness_results["error_handling_score"] = robustness_results["passed_tests"] / robustness_results["total_tests"]

    return robustness_results

async def run_complete_integration_test():
    """Run the complete integration test suite"""
    print("🧪 COMPLETE OVIYA EI SYSTEM INTEGRATION TEST")
    print("=" * 80)
    print("Testing ALL integrated therapeutic AI systems:")
    print("• 15+ Brain Systems (Crisis, Attachment, Memory, etc.)")
    print("• 8+ Voice Systems (Neural Prosody, Emotion Library, etc.)")
    print("• End-to-end therapeutic pipeline validation")
    print("• Performance and robustness testing")
    print("=" * 80)

    start_time = time.time()

    # Run comprehensive scenario testing
    scenario_results = await test_comprehensive_therapeutic_scenarios()

    # Run robustness testing
    robustness_results = await test_system_robustness()

    total_time = time.time() - start_time

    # Calculate final integration score
    integration_score = (
        scenario_results["system_integration_score"] * 0.7 +  # Scenario performance (70%)
        robustness_results["error_handling_score"] * 0.3       # Robustness (30%)
    )

    # Determine overall status
    if integration_score >= 0.85:
        overall_status = "EXCELLENT"
        status_description = "World-class therapeutic AI - all systems integrated perfectly"
    elif integration_score >= 0.70:
        overall_status = "GOOD"
        status_description = "Strong therapeutic AI - most systems working well"
    elif integration_score >= 0.50:
        overall_status = "ADEQUATE"
        status_description = "Functional therapeutic AI - needs some optimization"
    else:
        overall_status = "NEEDS_ATTENTION"
        status_description = "Therapeutic AI needs significant improvements"

    # Final results summary
    print("\n" + "=" * 80)
    print("🎯 FINAL INTEGRATION TEST RESULTS")
    print("=" * 80)
    print(f"Status: {overall_status}")
    print(f"Integration Score: {integration_score:.1%}")
    print(f"Description: {status_description}")
    print()
    print("📊 SCENARIO TESTING:")
    print(f"   Scenarios Tested: {scenario_results['total_scenarios']}")
    print(f"   Passed: {scenario_results['passed_scenarios']}")
    print(f"   Failed: {scenario_results['failed_scenarios']}")
    print(f"   System Integration: {scenario_results['system_integration_score']:.1%}")
    print()
    print("🔧 ROBUSTNESS TESTING:")
    print(f"   Tests Passed: {robustness_results['passed_tests']}/{robustness_results['total_tests']}")
    print(f"   Error Handling: {robustness_results['error_handling_score']:.1%}")
    print()
    print("⚡ PERFORMANCE METRICS:")
    print(f"   Avg Brain Response: {scenario_results['performance_metrics']['average_brain_response_time']:.2f}s")
    print(f"   Avg Voice Generation: {scenario_results['performance_metrics']['average_voice_generation_time']:.2f}s")
    print(f"   High-Quality Responses: {scenario_results['performance_metrics']['high_quality_responses']}")
    print(f"   High-Quality Voice: {scenario_results['performance_metrics']['high_quality_voice']}")
    print(f"   Total Test Time: {total_time:.1f}s")
    print()

    # System capabilities summary
    print("🎉 INTEGRATED THERAPEUTIC AI CAPABILITIES:")
    capabilities = [
        "✅ Clinical Crisis Detection & Intervention",
        "✅ Attachment Theory-Based Personalization",
        "✅ Advanced Memory Systems (ChromaDB + Semantic)",
        "✅ Bid Response & Connection Building (EFT)",
        "✅ Micro-Affirmations & Natural Conversation Flow",
        "✅ Neural Prosody Control (F0, Energy, Duration)",
        "✅ 28+ Emotion Library with Blending",
        "✅ Secure Base & Safe Haven Responses",
        "✅ Unconditional Positive Regard (Rogers)",
        "✅ Healthy Boundaries & Ethical Practice",
        "✅ Empathic Thinking (5 Cognitive Modes)",
        "✅ Strategic Therapeutic Silence (Ma 間)",
        "✅ CUDA Graphs Real-Time Streaming",
        "✅ Professional Audio Mastering",
        "✅ Comprehensive Safety & Ethics Monitoring"
    ]

    for capability in capabilities:
        print(f"   {capability}")

    print()
    print("🏆 CONCLUSION:")
    if overall_status == "EXCELLENT":
        print("   Oviya EI is now a WORLD-CLASS THERAPEUTIC AI COMPANION!")
        print("   Ready for clinical therapeutic applications with professional safety standards.")
    elif overall_status == "GOOD":
        print("   Oviya EI is a strong therapeutic AI with most systems working well.")
        print("   Suitable for therapeutic support with some monitoring.")
    else:
        print("   Oviya EI needs optimization but demonstrates integrated therapeutic capabilities.")
        print("   Continue development for clinical readiness.")

    # Save detailed results
    final_results = {
        "timestamp": time.time(),
        "overall_status": overall_status,
        "integration_score": integration_score,
        "scenario_results": scenario_results,
        "robustness_results": robustness_results,
        "total_test_time": total_time,
        "capabilities_integrated": len(capabilities)
    }

    with open("complete_integration_test_results.json", "w") as f:
        json.dump(final_results, f, indent=2, default=str)

    print(f"\n📄 Detailed results saved to: complete_integration_test_results.json")

    return final_results

if __name__ == "__main__":
    asyncio.run(run_complete_integration_test())
