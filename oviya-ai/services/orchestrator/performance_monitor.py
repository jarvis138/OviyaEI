#!/usr/bin/env python3
"""
Oviya Performance Monitoring System
Epic 6: Real-time metrics and latency tracking
"""
import asyncio
import time
import json
import psutil
import GPUtil
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import redis
import aiohttp

@dataclass
class LatencyMetrics:
    """Latency metrics for a single request"""
    request_id: str
    user_id: str
    timestamp: float
    
    # Component latencies
    vad_latency_ms: float = 0.0
    whisper_latency_ms: float = 0.0
    gpt_latency_ms: float = 0.0
    csm_latency_ms: float = 0.0
    network_latency_ms: float = 0.0
    
    # Total latency
    total_latency_ms: float = 0.0
    
    # Success/failure
    success: bool = True
    error_type: Optional[str] = None
    
    # Additional metadata
    emotion: Optional[str] = None
    text_length: int = 0
    audio_duration_ms: float = 0.0

@dataclass
class SystemMetrics:
    """System resource metrics"""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_utilization: float
    gpu_memory_percent: float
    active_sessions: int
    queue_depth: int
    error_rate: float

class PerformanceMonitor:
    """Main performance monitoring system"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.metrics_buffer: List[LatencyMetrics] = []
        self.system_metrics_buffer: List[SystemMetrics] = []
        
        # Thresholds for alerts
        self.thresholds = {
            "max_latency_ms": 2100,  # 2.1 seconds
            "p95_latency_ms": 1500,  # 1.5 seconds
            "error_rate_percent": 5.0,
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "gpu_memory_percent": 90.0
        }
    
    async def track_request(self, request_id: str, user_id: str) -> 'RequestTracker':
        """Start tracking a request"""
        return RequestTracker(self, request_id, user_id)
    
    async def record_latency_metrics(self, metrics: LatencyMetrics):
        """Record latency metrics"""
        self.metrics_buffer.append(metrics)
        
        # Flush buffer if it gets too large
        if len(self.metrics_buffer) >= 100:
            await self._flush_metrics_buffer()
        
        # Check for SLA violations
        if metrics.total_latency_ms > self.thresholds["max_latency_ms"]:
            await self._alert_sla_violation(metrics)
    
    async def record_system_metrics(self):
        """Record current system metrics"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # GPU metrics
            gpu_utilization = 0.0
            gpu_memory_percent = 0.0
            
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_utilization = gpu.load * 100
                    gpu_memory_percent = (gpu.memoryUsed / gpu.memoryTotal) * 100
            except:
                pass  # GPU monitoring not available
            
            # Application metrics
            active_sessions = await self._get_active_sessions()
            queue_depth = await self._get_queue_depth()
            error_rate = await self._get_error_rate()
            
            metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                gpu_utilization=gpu_utilization,
                gpu_memory_percent=gpu_memory_percent,
                active_sessions=active_sessions,
                queue_depth=queue_depth,
                error_rate=error_rate
            )
            
            self.system_metrics_buffer.append(metrics)
            
            # Check for system alerts
            await self._check_system_alerts(metrics)
            
            # Flush buffer if needed
            if len(self.system_metrics_buffer) >= 50:
                await self._flush_system_metrics_buffer()
                
        except Exception as e:
            print(f"Error recording system metrics: {e}")
    
    async def get_latency_summary(self, time_window_minutes: int = 60) -> Dict:
        """Get latency summary for time window"""
        cutoff_time = time.time() - (time_window_minutes * 60)
        
        # Get metrics from Redis
        metrics_key = f"latency_metrics:{datetime.now().strftime('%Y-%m-%d')}"
        metrics_data = await self.redis.lrange(metrics_key, 0, -1)
        
        recent_metrics = []
        for data in metrics_data:
            try:
                metric = LatencyMetrics(**json.loads(data))
                if metric.timestamp >= cutoff_time:
                    recent_metrics.append(metric)
            except:
                continue
        
        if not recent_metrics:
            return {"error": "No metrics available"}
        
        # Calculate statistics
        latencies = [m.total_latency_ms for m in recent_metrics]
        successful_requests = [m for m in recent_metrics if m.success]
        
        return {
            "time_window_minutes": time_window_minutes,
            "total_requests": len(recent_metrics),
            "successful_requests": len(successful_requests),
            "error_rate_percent": (len(recent_metrics) - len(successful_requests)) / len(recent_metrics) * 100,
            "avg_latency_ms": sum(latencies) / len(latencies),
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies),
            "p50_latency_ms": self._percentile(latencies, 50),
            "p95_latency_ms": self._percentile(latencies, 95),
            "p99_latency_ms": self._percentile(latencies, 99),
            "sla_compliance_percent": len([l for l in latencies if l <= self.thresholds["max_latency_ms"]]) / len(latencies) * 100
        }
    
    async def get_system_summary(self, time_window_minutes: int = 60) -> Dict:
        """Get system metrics summary"""
        cutoff_time = time.time() - (time_window_minutes * 60)
        
        # Get system metrics from Redis
        metrics_key = f"system_metrics:{datetime.now().strftime('%Y-%m-%d')}"
        metrics_data = await self.redis.lrange(metrics_key, 0, -1)
        
        recent_metrics = []
        for data in metrics_data:
            try:
                metric = SystemMetrics(**json.loads(data))
                if metric.timestamp >= cutoff_time:
                    recent_metrics.append(metric)
            except:
                continue
        
        if not recent_metrics:
            return {"error": "No system metrics available"}
        
        return {
            "time_window_minutes": time_window_minutes,
            "avg_cpu_percent": sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            "max_cpu_percent": max(m.cpu_percent for m in recent_metrics),
            "avg_memory_percent": sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            "max_memory_percent": max(m.memory_percent for m in recent_metrics),
            "avg_gpu_utilization": sum(m.gpu_utilization for m in recent_metrics) / len(recent_metrics),
            "max_gpu_utilization": max(m.gpu_utilization for m in recent_metrics),
            "avg_gpu_memory_percent": sum(m.gpu_memory_percent for m in recent_metrics) / len(recent_metrics),
            "max_gpu_memory_percent": max(m.gpu_memory_percent for m in recent_metrics),
            "avg_active_sessions": sum(m.active_sessions for m in recent_metrics) / len(recent_metrics),
            "max_active_sessions": max(m.active_sessions for m in recent_metrics),
            "avg_queue_depth": sum(m.queue_depth for m in recent_metrics) / len(recent_metrics),
            "max_queue_depth": max(m.queue_depth for m in recent_metrics),
            "avg_error_rate": sum(m.error_rate for m in recent_metrics) / len(recent_metrics),
            "max_error_rate": max(m.error_rate for m in recent_metrics)
        }
    
    async def _flush_metrics_buffer(self):
        """Flush latency metrics buffer to Redis"""
        if not self.metrics_buffer:
            return
        
        metrics_key = f"latency_metrics:{datetime.now().strftime('%Y-%m-%d')}"
        
        # Convert to JSON and store
        for metrics in self.metrics_buffer:
            await self.redis.lpush(metrics_key, json.dumps(asdict(metrics)))
        
        # Keep only last 1000 metrics per day
        await self.redis.ltrim(metrics_key, 0, 999)
        
        self.metrics_buffer.clear()
    
    async def _flush_system_metrics_buffer(self):
        """Flush system metrics buffer to Redis"""
        if not self.system_metrics_buffer:
            return
        
        metrics_key = f"system_metrics:{datetime.now().strftime('%Y-%m-%d')}"
        
        # Convert to JSON and store
        for metrics in self.system_metrics_buffer:
            await self.redis.lpush(metrics_key, json.dumps(asdict(metrics)))
        
        # Keep only last 500 metrics per day
        await self.redis.ltrim(metrics_key, 0, 499)
        
        self.system_metrics_buffer.clear()
    
    async def _alert_sla_violation(self, metrics: LatencyMetrics):
        """Alert on SLA violations"""
        alert_data = {
            "type": "sla_violation",
            "request_id": metrics.request_id,
            "user_id": metrics.user_id,
            "latency_ms": metrics.total_latency_ms,
            "threshold_ms": self.thresholds["max_latency_ms"],
            "timestamp": metrics.timestamp,
            "components": {
                "vad": metrics.vad_latency_ms,
                "whisper": metrics.whisper_latency_ms,
                "gpt": metrics.gpt_latency_ms,
                "csm": metrics.csm_latency_ms,
                "network": metrics.network_latency_ms
            }
        }
        
        # Store alert
        alert_key = f"alerts:{datetime.now().strftime('%Y-%m-%d')}"
        await self.redis.lpush(alert_key, json.dumps(alert_data))
        
        print(f"🚨 SLA VIOLATION: Request {metrics.request_id} took {metrics.total_latency_ms:.0f}ms")
    
    async def _check_system_alerts(self, metrics: SystemMetrics):
        """Check for system resource alerts"""
        alerts = []
        
        if metrics.cpu_percent > self.thresholds["cpu_percent"]:
            alerts.append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
        
        if metrics.memory_percent > self.thresholds["memory_percent"]:
            alerts.append(f"High memory usage: {metrics.memory_percent:.1f}%")
        
        if metrics.gpu_memory_percent > self.thresholds["gpu_memory_percent"]:
            alerts.append(f"High GPU memory usage: {metrics.gpu_memory_percent:.1f}%")
        
        if metrics.error_rate > self.thresholds["error_rate_percent"]:
            alerts.append(f"High error rate: {metrics.error_rate:.1f}%")
        
        if alerts:
            alert_data = {
                "type": "system_alert",
                "timestamp": metrics.timestamp,
                "alerts": alerts,
                "metrics": asdict(metrics)
            }
            
            alert_key = f"alerts:{datetime.now().strftime('%Y-%m-%d')}"
            await self.redis.lpush(alert_key, json.dumps(alert_data))
            
            print(f"⚠️ SYSTEM ALERT: {', '.join(alerts)}")
    
    async def _get_active_sessions(self) -> int:
        """Get number of active sessions"""
        # Count active sessions in Redis
        pattern = "concurrent_sessions:*"
        keys = await self.redis.keys(pattern)
        total_sessions = 0
        
        for key in keys:
            count = await self.redis.scard(key)
            total_sessions += count
        
        return total_sessions
    
    async def _get_queue_depth(self) -> int:
        """Get current queue depth"""
        # Check CSM queue depth
        queue_key = "csm_queue_depth"
        return await self.redis.get(queue_key) or 0
    
    async def _get_error_rate(self) -> float:
        """Get current error rate"""
        # Calculate error rate from recent metrics
        metrics_key = f"latency_metrics:{datetime.now().strftime('%Y-%m-%d')}"
        recent_metrics = await self.redis.lrange(metrics_key, 0, 99)  # Last 100 requests
        
        if not recent_metrics:
            return 0.0
        
        error_count = 0
        for data in recent_metrics:
            try:
                metric = LatencyMetrics(**json.loads(data))
                if not metric.success:
                    error_count += 1
            except:
                continue
        
        return (error_count / len(recent_metrics)) * 100
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile"""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]

class RequestTracker:
    """Tracks individual request metrics"""
    
    def __init__(self, monitor: PerformanceMonitor, request_id: str, user_id: str):
        self.monitor = monitor
        self.request_id = request_id
        self.user_id = user_id
        self.start_time = time.time()
        self.metrics = LatencyMetrics(
            request_id=request_id,
            user_id=user_id,
            timestamp=self.start_time
        )
    
    def record_vad_latency(self, latency_ms: float):
        """Record VAD latency"""
        self.metrics.vad_latency_ms = latency_ms
    
    def record_whisper_latency(self, latency_ms: float):
        """Record Whisper latency"""
        self.metrics.whisper_latency_ms = latency_ms
    
    def record_gpt_latency(self, latency_ms: float):
        """Record GPT latency"""
        self.metrics.gpt_latency_ms = latency_ms
    
    def record_csm_latency(self, latency_ms: float):
        """Record CSM latency"""
        self.metrics.csm_latency_ms = latency_ms
    
    def record_network_latency(self, latency_ms: float):
        """Record network latency"""
        self.metrics.network_latency_ms = latency_ms
    
    def set_metadata(self, emotion: str = None, text_length: int = 0, audio_duration_ms: float = 0.0):
        """Set request metadata"""
        self.metrics.emotion = emotion
        self.metrics.text_length = text_length
        self.metrics.audio_duration_ms = audio_duration_ms
    
    async def finish(self, success: bool = True, error_type: str = None):
        """Finish tracking and record metrics"""
        self.metrics.total_latency_ms = (time.time() - self.start_time) * 1000
        self.metrics.success = success
        self.metrics.error_type = error_type
        
        await self.monitor.record_latency_metrics(self.metrics)

# Usage example
async def main():
    """Test the performance monitoring system"""
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    monitor = PerformanceMonitor(redis_client)
    
    # Track a request
    tracker = await monitor.track_request("req_123", "user_456")
    tracker.record_vad_latency(45.0)
    tracker.record_whisper_latency(250.0)
    tracker.record_gpt_latency(400.0)
    tracker.record_csm_latency(300.0)
    tracker.record_network_latency(100.0)
    tracker.set_metadata(emotion="empathetic", text_length=50)
    await tracker.finish(success=True)
    
    # Record system metrics
    await monitor.record_system_metrics()
    
    # Get summaries
    latency_summary = await monitor.get_latency_summary(60)
    system_summary = await monitor.get_system_summary(60)
    
    print("Latency Summary:", latency_summary)
    print("System Summary:", system_summary)

if __name__ == "__main__":
    asyncio.run(main())


