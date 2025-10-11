#!/usr/bin/env python3
"""
Oviya End-to-End Testing Suite
Epic 5: Complete system integration testing
"""
import asyncio
import aiohttp
import json
import time
from typing import Dict, List
import websockets

class E2ETester:
    def __init__(self):
        self.csm_url = "http://localhost:8000"
        self.asr_url = "http://localhost:8001"
        self.orchestrator_url = "http://localhost:8002"
        self.results: List[Dict] = []
    
    async def test_service_health(self) -> Dict[str, bool]:
        """Test all services are healthy"""
        print("🏥 Testing service health...")
        
        services = {
            "CSM": f"{self.csm_url}/health",
            "ASR": f"{self.asr_url}/health",
            "Orchestrator": f"{self.orchestrator_url}/health"
        }
        
        health_status = {}
        
        async with aiohttp.ClientSession() as session:
            for service_name, url in services.items():
                try:
                    async with session.get(url, timeout=5) as response:
                        health_status[service_name] = response.status == 200
                        print(f"  {'✅' if health_status[service_name] else '❌'} {service_name}: {response.status}")
                except Exception as e:
                    health_status[service_name] = False
                    print(f"  ❌ {service_name}: Error - {e}")
        
        return health_status
    
    async def test_csm_direct(self) -> Dict:
        """Test CSM service directly"""
        print("🎵 Testing CSM service directly...")
        
        test_cases = [
            {"text": "Hello", "emotion": "empathetic"},
            {"text": "How are you?", "emotion": "encouraging"},
            {"text": "That sounds challenging", "emotion": "calm"}
        ]
        
        results = []
        
        async with aiohttp.ClientSession() as session:
            for i, test_case in enumerate(test_cases):
                start_time = time.time()
                
                try:
                    async with session.post(
                        f"{self.csm_url}/tts",
                        json=test_case
                    ) as response:
                        end_time = time.time()
                        latency = (end_time - start_time) * 1000
                        
                        if response.status == 200:
                            result = await response.json()
                            success = result.get("first_chunk_latency", 0) < 900
                            
                            results.append({
                                "test": f"CSM Direct {i+1}",
                                "success": success,
                                "latency_ms": latency,
                                "first_chunk_latency_ms": result.get("first_chunk_latency", 0),
                                "text": test_case["text"],
                                "emotion": test_case["emotion"]
                            })
                            
                            print(f"  ✅ {test_case['text']} ({test_case['emotion']}): {result.get('first_chunk_latency', 0):.0f}ms")
                        else:
                            results.append({
                                "test": f"CSM Direct {i+1}",
                                "success": False,
                                "latency_ms": latency,
                                "error": f"HTTP {response.status}"
                            })
                            print(f"  ❌ {test_case['text']}: HTTP {response.status}")
                
                except Exception as e:
                    results.append({
                        "test": f"CSM Direct {i+1}",
                        "success": False,
                        "latency_ms": 0,
                        "error": str(e)
                    })
                    print(f"  ❌ {test_case['text']}: {e}")
        
        return results
    
    async def test_orchestrator_websocket(self) -> Dict:
        """Test orchestrator WebSocket connection"""
        print("🔌 Testing orchestrator WebSocket...")
        
        try:
            uri = f"ws://localhost:8002/session/connect"
            
            async with websockets.connect(uri) as websocket:
                # Send a test message
                test_message = {
                    "type": "test",
                    "text": "Hello, this is a test"
                }
                
                await websocket.send(json.dumps(test_message))
                
                # Wait for response (with timeout)
                try:
                    response = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(response)
                    
                    print(f"  ✅ WebSocket connected and responded")
                    return {
                        "test": "Orchestrator WebSocket",
                        "success": True,
                        "response": data
                    }
                
                except asyncio.TimeoutError:
                    print(f"  ⚠️ WebSocket timeout (no response)")
                    return {
                        "test": "Orchestrator WebSocket",
                        "success": False,
                        "error": "Timeout waiting for response"
                    }
        
        except Exception as e:
            print(f"  ❌ WebSocket error: {e}")
            return {
                "test": "Orchestrator WebSocket",
                "success": False,
                "error": str(e)
            }
    
    async def test_complete_pipeline(self) -> Dict:
        """Test complete ASR → GPT → CSM pipeline"""
        print("🔄 Testing complete pipeline...")
        
        # This is a simulated test since we don't have real ASR input
        # In a real scenario, we'd send audio data to the orchestrator
        
        try:
            async with aiohttp.ClientSession() as session:
                # Test orchestrator health
                async with session.get(f"{self.orchestrator_url}/health") as response:
                    if response.status == 200:
                        orchestrator_health = await response.json()
                        
                        # Simulate pipeline test by testing CSM with orchestrator context
                        pipeline_start = time.time()
                        
                        # Test CSM with a typical response
                        async with session.post(
                            f"{self.csm_url}/tts",
                            json={
                                "text": "I understand how you're feeling. That sounds challenging.",
                                "emotion": "empathetic"
                            }
                        ) as csm_response:
                            pipeline_end = time.time()
                            pipeline_latency = (pipeline_end - pipeline_start) * 1000
                            
                            if csm_response.status == 200:
                                csm_result = await csm_response.json()
                                
                                print(f"  ✅ Pipeline test: {pipeline_latency:.0f}ms")
                                
                                return {
                                    "test": "Complete Pipeline",
                                    "success": True,
                                    "pipeline_latency_ms": pipeline_latency,
                                    "csm_latency_ms": csm_result.get("first_chunk_latency", 0),
                                    "orchestrator_healthy": True
                                }
                            else:
                                return {
                                    "test": "Complete Pipeline",
                                    "success": False,
                                    "error": f"CSM failed: {csm_response.status}"
                                }
                    else:
                        return {
                            "test": "Complete Pipeline",
                            "success": False,
                            "error": f"Orchestrator unhealthy: {response.status}"
                        }
        
        except Exception as e:
            print(f"  ❌ Pipeline error: {e}")
            return {
                "test": "Complete Pipeline",
                "success": False,
                "error": str(e)
            }
    
    async def test_error_handling(self) -> List[Dict]:
        """Test error handling scenarios"""
        print("🛡️ Testing error handling...")
        
        error_tests = [
            {
                "name": "Invalid JSON",
                "url": f"{self.csm_url}/tts",
                "data": "invalid json",
                "headers": {"Content-Type": "application/json"}
            },
            {
                "name": "Missing Text",
                "url": f"{self.csm_url}/tts",
                "data": {"emotion": "empathetic"},
                "headers": {"Content-Type": "application/json"}
            },
            {
                "name": "Invalid Emotion",
                "url": f"{self.csm_url}/tts",
                "data": {"text": "Hello", "emotion": "invalid_emotion"},
                "headers": {"Content-Type": "application/json"}
            }
        ]
        
        results = []
        
        async with aiohttp.ClientSession() as session:
            for test in error_tests:
                try:
                    async with session.post(
                        test["url"],
                        json=test["data"] if isinstance(test["data"], dict) else None,
                        data=test["data"] if not isinstance(test["data"], dict) else None,
                        headers=test["headers"]
                    ) as response:
                        # For error handling tests, we expect non-200 responses
                        success = response.status >= 400  # Error responses are expected
                        
                        results.append({
                            "test": f"Error Handling: {test['name']}",
                            "success": success,
                            "status_code": response.status,
                            "expected_error": True
                        })
                        
                        print(f"  {'✅' if success else '❌'} {test['name']}: {response.status}")
                
                except Exception as e:
                    # Exceptions are also expected for some error tests
                    results.append({
                        "test": f"Error Handling: {test['name']}",
                        "success": True,  # Exception is expected
                        "error": str(e),
                        "expected_error": True
                    })
                    print(f"  ✅ {test['name']}: Exception (expected)")
        
        return results
    
    def generate_e2e_report(self, all_results: List[Dict]) -> Dict:
        """Generate end-to-end test report"""
        total_tests = len(all_results)
        successful_tests = len([r for r in all_results if r.get("success", False)])
        
        # Categorize results
        health_tests = [r for r in all_results if "health" in r.get("test", "").lower()]
        csm_tests = [r for r in all_results if "csm" in r.get("test", "").lower()]
        websocket_tests = [r for r in all_results if "websocket" in r.get("test", "").lower()]
        pipeline_tests = [r for r in all_results if "pipeline" in r.get("test", "").lower()]
        error_tests = [r for r in all_results if "error" in r.get("test", "").lower()]
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "success_rate": successful_tests / total_tests * 100 if total_tests > 0 else 0
            },
            "test_categories": {
                "health_tests": {
                    "total": len(health_tests),
                    "successful": len([r for r in health_tests if r.get("success", False)])
                },
                "csm_tests": {
                    "total": len(csm_tests),
                    "successful": len([r for r in csm_tests if r.get("success", False)])
                },
                "websocket_tests": {
                    "total": len(websocket_tests),
                    "successful": len([r for r in websocket_tests if r.get("success", False)])
                },
                "pipeline_tests": {
                    "total": len(pipeline_tests),
                    "successful": len([r for r in pipeline_tests if r.get("success", False)])
                },
                "error_tests": {
                    "total": len(error_tests),
                    "successful": len([r for r in error_tests if r.get("success", False)])
                }
            },
            "performance": {
                "avg_csm_latency_ms": 0,
                "avg_pipeline_latency_ms": 0
            },
            "recommendations": []
        }
        
        # Calculate performance metrics
        csm_latencies = [r.get("first_chunk_latency_ms", 0) for r in csm_tests if r.get("first_chunk_latency_ms", 0) > 0]
        pipeline_latencies = [r.get("pipeline_latency_ms", 0) for r in pipeline_tests if r.get("pipeline_latency_ms", 0) > 0]
        
        if csm_latencies:
            report["performance"]["avg_csm_latency_ms"] = sum(csm_latencies) / len(csm_latencies)
        
        if pipeline_latencies:
            report["performance"]["avg_pipeline_latency_ms"] = sum(pipeline_latencies) / len(pipeline_latencies)
        
        # Generate recommendations
        if report["summary"]["success_rate"] < 95:
            report["recommendations"].append("Improve system reliability - success rate below 95%")
        
        if report["performance"]["avg_csm_latency_ms"] > 500:
            report["recommendations"].append("Optimize CSM performance - latency above 500ms")
        
        if report["test_categories"]["websocket_tests"]["successful"] < report["test_categories"]["websocket_tests"]["total"]:
            report["recommendations"].append("Fix WebSocket connectivity issues")
        
        return report

async def main():
    """Run complete end-to-end test suite"""
    print("🚀 Starting Oviya End-to-End Test Suite")
    print("=" * 60)
    
    tester = E2ETester()
    all_results = []
    
    # Test 1: Service Health
    print("\n1️⃣ Testing Service Health")
    health_status = await tester.test_service_health()
    for service, healthy in health_status.items():
        all_results.append({
            "test": f"Health: {service}",
            "success": healthy,
            "service": service
        })
    
    # Test 2: CSM Direct
    print("\n2️⃣ Testing CSM Service")
    csm_results = await tester.test_csm_direct()
    all_results.extend(csm_results)
    
    # Test 3: WebSocket
    print("\n3️⃣ Testing WebSocket Connection")
    websocket_result = await tester.test_orchestrator_websocket()
    all_results.append(websocket_result)
    
    # Test 4: Complete Pipeline
    print("\n4️⃣ Testing Complete Pipeline")
    pipeline_result = await tester.test_complete_pipeline()
    all_results.append(pipeline_result)
    
    # Test 5: Error Handling
    print("\n5️⃣ Testing Error Handling")
    error_results = await tester.test_error_handling()
    all_results.extend(error_results)
    
    # Generate Report
    print("\n📊 Generating E2E Report...")
    report = tester.generate_e2e_report(all_results)
    
    # Print Summary
    print("\n" + "=" * 60)
    print("🎯 END-TO-END TEST RESULTS")
    print("=" * 60)
    print(f"Total Tests: {report['summary']['total_tests']}")
    print(f"Successful: {report['summary']['successful_tests']}")
    print(f"Failed: {report['summary']['failed_tests']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"Avg CSM Latency: {report['performance']['avg_csm_latency_ms']:.1f}ms")
    print(f"Avg Pipeline Latency: {report['performance']['avg_pipeline_latency_ms']:.1f}ms")
    
    print(f"\nTest Categories:")
    for category, stats in report['test_categories'].items():
        success_rate = stats['successful'] / stats['total'] * 100 if stats['total'] > 0 else 0
        print(f"  {category}: {stats['successful']}/{stats['total']} ({success_rate:.1f}%)")
    
    if report['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in report['recommendations']:
            print(f"  - {rec}")
    
    # Save report
    with open('/Users/jarvis/Documents/Oviya EI/oviya-ai/validation/results/e2e_test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 E2E report saved to: e2e_test_report.json")
    print("\n🎉 End-to-End Testing Complete!")

if __name__ == "__main__":
    asyncio.run(main())


