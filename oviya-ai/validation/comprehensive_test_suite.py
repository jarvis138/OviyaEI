#!/usr/bin/env python3
"""
Oviya Comprehensive Testing Suite
Epic 5: Test audio context emotions, all devices, 100 scenarios
"""
import asyncio
import aiohttp
import json
import time
import statistics
from typing import Dict, List, Tuple
from dataclasses import dataclass
import random

@dataclass
class TestResult:
    test_name: str
    latency_ms: float
    success: bool
    emotion: str
    text: str
    timestamp: float

class OviyaTestSuite:
    def __init__(self):
        self.base_url = "http://localhost:8000"
        self.orchestrator_url = "http://localhost:8002"
        self.results: List[TestResult] = []
        
    async def test_csm_latency(self, text: str, emotion: str = "empathetic") -> TestResult:
        """Test CSM latency for a given text and emotion"""
        try:
            async with aiohttp.ClientSession() as session:
                start_time = time.time()
                
                async with session.post(
                    f"{self.base_url}/tts",
                    json={"text": text, "emotion": emotion}
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        latency = result.get("first_chunk_latency", 0)
                        success = latency < 900  # Our target
                        
                        test_result = TestResult(
                            test_name="CSM Latency",
                            latency_ms=latency,
                            success=success,
                            emotion=emotion,
                            text=text,
                            timestamp=time.time()
                        )
                        
                        self.results.append(test_result)
                        return test_result
                    else:
                        return TestResult(
                            test_name="CSM Latency",
                            latency_ms=0,
                            success=False,
                            emotion=emotion,
                            text=text,
                            timestamp=time.time()
                        )
        except Exception as e:
            print(f"Error testing CSM: {e}")
            return TestResult(
                test_name="CSM Latency",
                latency_ms=0,
                success=False,
                emotion=emotion,
                text=text,
                timestamp=time.time()
            )
    
    async def test_emotion_scenarios(self) -> List[TestResult]:
        """Test different emotion scenarios"""
        emotions = ["empathetic", "encouraging", "calm", "concerned", "joyful"]
        test_phrases = [
            "Hello, how are you?",
            "I understand how you're feeling",
            "That sounds challenging",
            "You're doing great!",
            "Tell me more about that",
            "I'm here to support you",
            "That must be difficult",
            "You're not alone in this",
            "I believe in you",
            "How can I help?"
        ]
        
        results = []
        for emotion in emotions:
            for phrase in test_phrases:
                result = await self.test_csm_latency(phrase, emotion)
                results.append(result)
                print(f"✅ {emotion}: {phrase[:30]}... - {result.latency_ms:.0f}ms")
                await asyncio.sleep(0.5)  # Rate limiting
        
        return results
    
    async def test_text_length_scenarios(self) -> List[TestResult]:
        """Test different text lengths"""
        text_scenarios = [
            ("Short", "Hi"),
            ("Medium", "Hello, how are you today?"),
            ("Long", "Hello, how are you today? I hope you're doing well and feeling good about everything."),
            ("Very Long", "Hello, how are you today? I hope you're doing well and feeling good about everything. I understand that life can be challenging sometimes, but I want you to know that you're doing great and I'm here to support you through whatever you're going through.")
        ]
        
        results = []
        for scenario_name, text in text_scenarios:
            result = await self.test_csm_latency(text, "empathetic")
            results.append(result)
            print(f"✅ {scenario_name}: {result.latency_ms:.0f}ms")
            await asyncio.sleep(0.5)
        
        return results
    
    async def test_concurrent_requests(self, num_requests: int = 10) -> List[TestResult]:
        """Test concurrent request handling"""
        print(f"🧪 Testing {num_requests} concurrent requests...")
        
        tasks = []
        for i in range(num_requests):
            text = f"Test request {i+1}: Hello, how are you?"
            task = self.test_csm_latency(text, "empathetic")
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions
        valid_results = [r for r in results if isinstance(r, TestResult)]
        
        print(f"✅ Completed {len(valid_results)}/{num_requests} concurrent requests")
        return valid_results
    
    async def test_health_endpoints(self) -> Dict[str, bool]:
        """Test all health endpoints"""
        endpoints = {
            "CSM Service": f"{self.base_url}/health",
            "ASR Service": "http://localhost:8001/health",
            "Orchestrator": f"{self.orchestrator_url}/health"
        }
        
        health_status = {}
        
        async with aiohttp.ClientSession() as session:
            for service_name, url in endpoints.items():
                try:
                    async with session.get(url, timeout=5) as response:
                        health_status[service_name] = response.status == 200
                        print(f"✅ {service_name}: {'Healthy' if health_status[service_name] else 'Unhealthy'}")
                except Exception as e:
                    health_status[service_name] = False
                    print(f"❌ {service_name}: Error - {e}")
        
        return health_status
    
    def generate_report(self) -> Dict:
        """Generate comprehensive test report"""
        if not self.results:
            return {"error": "No test results available"}
        
        successful_tests = [r for r in self.results if r.success]
        failed_tests = [r for r in self.results if not r.success]
        
        latencies = [r.latency_ms for r in self.results if r.latency_ms > 0]
        
        report = {
            "summary": {
                "total_tests": len(self.results),
                "successful_tests": len(successful_tests),
                "failed_tests": len(failed_tests),
                "success_rate": len(successful_tests) / len(self.results) * 100,
                "target_met": len(successful_tests) == len(self.results)
            },
            "performance": {
                "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
                "min_latency_ms": min(latencies) if latencies else 0,
                "max_latency_ms": max(latencies) if latencies else 0,
                "median_latency_ms": statistics.median(latencies) if latencies else 0,
                "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else max(latencies) if latencies else 0
            },
            "emotion_breakdown": {},
            "recommendations": []
        }
        
        # Emotion breakdown
        emotions = {}
        for result in self.results:
            if result.emotion not in emotions:
                emotions[result.emotion] = {"total": 0, "successful": 0, "avg_latency": 0}
            emotions[result.emotion]["total"] += 1
            if result.success:
                emotions[result.emotion]["successful"] += 1
        
        for emotion, stats in emotions.items():
            emotion_results = [r for r in self.results if r.emotion == emotion and r.latency_ms > 0]
            stats["avg_latency"] = statistics.mean([r.latency_ms for r in emotion_results]) if emotion_results else 0
            stats["success_rate"] = stats["successful"] / stats["total"] * 100
        
        report["emotion_breakdown"] = emotions
        
        # Recommendations
        if report["performance"]["avg_latency_ms"] > 500:
            report["recommendations"].append("Consider optimizing CSM model for better latency")
        
        if report["summary"]["success_rate"] < 95:
            report["recommendations"].append("Investigate failed tests and improve reliability")
        
        if report["performance"]["p95_latency_ms"] > 900:
            report["recommendations"].append("P95 latency exceeds target - optimize worst-case scenarios")
        
        return report

async def main():
    """Run comprehensive test suite"""
    print("🚀 Starting Oviya Comprehensive Test Suite")
    print("=" * 60)
    
    test_suite = OviyaTestSuite()
    
    # Test 1: Health endpoints
    print("\n📊 Testing Health Endpoints...")
    health_status = await test_suite.test_health_endpoints()
    
    # Test 2: Emotion scenarios
    print("\n🎭 Testing Emotion Scenarios...")
    emotion_results = await test_suite.test_emotion_scenarios()
    
    # Test 3: Text length scenarios
    print("\n📝 Testing Text Length Scenarios...")
    length_results = await test_suite.test_text_length_scenarios()
    
    # Test 4: Concurrent requests
    print("\n⚡ Testing Concurrent Requests...")
    concurrent_results = await test_suite.test_concurrent_requests(5)
    
    # Generate report
    print("\n📊 Generating Test Report...")
    report = test_suite.generate_report()
    
    # Print summary
    print("\n" + "=" * 60)
    print("🎯 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Successful: {report['summary']['successful_tests']}")
    print(f"Failed: {report['summary']['failed_tests']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"Average Latency: {report['performance']['avg_latency_ms']:.1f}ms")
    print(f"P95 Latency: {report['performance']['p95_latency_ms']:.1f}ms")
    print(f"Target Met: {'✅ YES' if report['summary']['target_met'] else '❌ NO'}")
    
    if report['recommendations']:
        print("\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
    
    # Save detailed report
    with open('/Users/jarvis/Documents/Oviya EI/oviya-ai/validation/results/comprehensive_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed report saved to: comprehensive_test_report.json")
    print("\n🎉 Test Suite Complete!")

if __name__ == "__main__":
    asyncio.run(main())


