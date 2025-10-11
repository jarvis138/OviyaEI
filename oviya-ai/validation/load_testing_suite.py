#!/usr/bin/env python3
"""
Oviya Load Testing Suite
Epic 5: Performance testing under load
"""
import asyncio
import aiohttp
import time
import statistics
from typing import List, Dict
import random

class LoadTester:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[Dict] = []
        
    async def single_request(self, session: aiohttp.ClientSession, request_id: int) -> Dict:
        """Make a single TTS request"""
        start_time = time.time()
        
        try:
            async with session.post(
                f"{self.base_url}/tts",
                json={
                    "text": f"Load test request {request_id}: Hello, how are you?",
                    "emotion": random.choice(["empathetic", "encouraging", "calm"])
                }
            ) as response:
                end_time = time.time()
                latency = (end_time - start_time) * 1000
                
                if response.status == 200:
                    result = await response.json()
                    first_chunk_latency = result.get("first_chunk_latency", latency)
                    
                    return {
                        "request_id": request_id,
                        "success": True,
                        "latency_ms": latency,
                        "first_chunk_latency_ms": first_chunk_latency,
                        "status_code": response.status,
                        "timestamp": time.time()
                    }
                else:
                    return {
                        "request_id": request_id,
                        "success": False,
                        "latency_ms": latency,
                        "first_chunk_latency_ms": 0,
                        "status_code": response.status,
                        "timestamp": time.time()
                    }
        except Exception as e:
            end_time = time.time()
            latency = (end_time - start_time) * 1000
            
            return {
                "request_id": request_id,
                "success": False,
                "latency_ms": latency,
                "first_chunk_latency_ms": 0,
                "status_code": 0,
                "error": str(e),
                "timestamp": time.time()
            }
    
    async def run_load_test(self, concurrent_users: int = 10, requests_per_user: int = 5) -> Dict:
        """Run load test with specified concurrent users and requests per user"""
        print(f"🚀 Starting load test: {concurrent_users} users, {requests_per_user} requests each")
        
        start_time = time.time()
        all_results = []
        
        async with aiohttp.ClientSession() as session:
            # Create tasks for all requests
            tasks = []
            request_id = 0
            
            for user in range(concurrent_users):
                for request in range(requests_per_user):
                    request_id += 1
                    task = self.single_request(session, request_id)
                    tasks.append(task)
            
            # Execute all requests concurrently
            print(f"📡 Executing {len(tasks)} requests...")
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, dict):
                    all_results.append(result)
                else:
                    print(f"❌ Exception in request: {result}")
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Analyze results
        successful_requests = [r for r in all_results if r.get("success", False)]
        failed_requests = [r for r in all_results if not r.get("success", False)]
        
        latencies = [r["latency_ms"] for r in successful_requests]
        first_chunk_latencies = [r["first_chunk_latency_ms"] for r in successful_requests if r["first_chunk_latency_ms"] > 0]
        
        analysis = {
            "load_test_config": {
                "concurrent_users": concurrent_users,
                "requests_per_user": requests_per_user,
                "total_requests": len(all_results)
            },
            "performance": {
                "total_time_seconds": total_time,
                "requests_per_second": len(all_results) / total_time,
                "success_rate": len(successful_requests) / len(all_results) * 100,
                "failed_requests": len(failed_requests)
            },
            "latency_stats": {
                "avg_latency_ms": statistics.mean(latencies) if latencies else 0,
                "min_latency_ms": min(latencies) if latencies else 0,
                "max_latency_ms": max(latencies) if latencies else 0,
                "median_latency_ms": statistics.median(latencies) if latencies else 0,
                "p95_latency_ms": statistics.quantiles(latencies, n=20)[18] if len(latencies) > 20 else max(latencies) if latencies else 0
            },
            "first_chunk_latency_stats": {
                "avg_first_chunk_ms": statistics.mean(first_chunk_latencies) if first_chunk_latencies else 0,
                "min_first_chunk_ms": min(first_chunk_latencies) if first_chunk_latencies else 0,
                "max_first_chunk_ms": max(first_chunk_latencies) if first_chunk_latencies else 0,
                "median_first_chunk_ms": statistics.median(first_chunk_latencies) if first_chunk_latencies else 0
            },
            "target_compliance": {
                "latency_target_met": statistics.mean(latencies) < 900 if latencies else False,
                "first_chunk_target_met": statistics.mean(first_chunk_latencies) < 900 if first_chunk_latencies else False,
                "success_rate_target_met": len(successful_requests) / len(all_results) * 100 >= 95
            }
        }
        
        return analysis
    
    def print_results(self, analysis: Dict):
        """Print load test results"""
        print("\n" + "=" * 60)
        print("📊 LOAD TEST RESULTS")
        print("=" * 60)
        
        config = analysis["load_test_config"]
        perf = analysis["performance"]
        latency = analysis["latency_stats"]
        first_chunk = analysis["first_chunk_latency_stats"]
        targets = analysis["target_compliance"]
        
        print(f"Configuration:")
        print(f"  - Concurrent Users: {config['concurrent_users']}")
        print(f"  - Requests per User: {config['requests_per_user']}")
        print(f"  - Total Requests: {config['total_requests']}")
        
        print(f"\nPerformance:")
        print(f"  - Total Time: {perf['total_time_seconds']:.2f}s")
        print(f"  - Requests/sec: {perf['requests_per_second']:.2f}")
        print(f"  - Success Rate: {perf['success_rate']:.1f}%")
        print(f"  - Failed Requests: {perf['failed_requests']}")
        
        print(f"\nLatency Statistics:")
        print(f"  - Average: {latency['avg_latency_ms']:.1f}ms")
        print(f"  - Median: {latency['median_latency_ms']:.1f}ms")
        print(f"  - P95: {latency['p95_latency_ms']:.1f}ms")
        print(f"  - Min: {latency['min_latency_ms']:.1f}ms")
        print(f"  - Max: {latency['max_latency_ms']:.1f}ms")
        
        print(f"\nFirst Chunk Latency:")
        print(f"  - Average: {first_chunk['avg_first_chunk_ms']:.1f}ms")
        print(f"  - Median: {first_chunk['median_first_chunk_ms']:.1f}ms")
        print(f"  - Min: {first_chunk['min_first_chunk_ms']:.1f}ms")
        print(f"  - Max: {first_chunk['max_first_chunk_ms']:.1f}ms")
        
        print(f"\nTarget Compliance:")
        print(f"  - Latency <900ms: {'✅' if targets['latency_target_met'] else '❌'}")
        print(f"  - First Chunk <900ms: {'✅' if targets['first_chunk_target_met'] else '❌'}")
        print(f"  - Success Rate ≥95%: {'✅' if targets['success_rate_target_met'] else '❌'}")

async def main():
    """Run load tests with different configurations"""
    print("🚀 Starting Oviya Load Testing Suite")
    
    tester = LoadTester()
    
    # Test configurations
    test_configs = [
        {"users": 5, "requests": 3},   # Light load
        {"users": 10, "requests": 5},   # Medium load
        {"users": 20, "requests": 3},   # High concurrent users
    ]
    
    all_results = []
    
    for config in test_configs:
        print(f"\n🧪 Testing: {config['users']} users, {config['requests']} requests each")
        result = await tester.run_load_test(config["users"], config["requests"])
        tester.print_results(result)
        all_results.append(result)
        
        # Wait between tests
        print("\n⏳ Waiting 10 seconds before next test...")
        await asyncio.sleep(10)
    
    # Summary
    print("\n" + "=" * 60)
    print("🎯 LOAD TEST SUMMARY")
    print("=" * 60)
    
    for i, result in enumerate(all_results):
        config = result["load_test_config"]
        targets = result["target_compliance"]
        
        print(f"\nTest {i+1}: {config['concurrent_users']} users, {config['requests_per_user']} requests")
        print(f"  Success Rate: {result['performance']['success_rate']:.1f}%")
        print(f"  Avg Latency: {result['latency_stats']['avg_latency_ms']:.1f}ms")
        print(f"  Targets Met: {sum(targets.values())}/3")
    
    print("\n🎉 Load Testing Complete!")

if __name__ == "__main__":
    asyncio.run(main())


