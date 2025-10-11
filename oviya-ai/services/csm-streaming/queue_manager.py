#!/usr/bin/env python3
"""
Oviya CSM Queue Manager
Manages request queue with priority handling and overflow protection
"""
import asyncio
import time
import logging
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import json
from collections import deque

logger = logging.getLogger(__name__)

class Priority(Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

@dataclass
class QueueRequest:
    """Represents a request in the queue"""
    request_id: str
    session_id: str
    text: str
    emotion: str
    priority: Priority
    created_at: float
    timeout_at: float
    user_id: str = ""
    metadata: Dict = None

@dataclass
class QueueStats:
    """Queue statistics"""
    total_requests: int
    pending_requests: int
    completed_requests: int
    failed_requests: int
    avg_wait_time_ms: float
    avg_processing_time_ms: float
    queue_depth: int
    priority_distribution: Dict[str, int]

class QueueManager:
    """Manages request queue with priority handling"""
    
    def __init__(self, 
                 max_queue_size: int = 20,
                 max_priority_queue_size: int = 5,
                 request_timeout: float = 10.0,
                 max_wait_time: float = 30.0):
        
        self.max_queue_size = max_queue_size
        self.max_priority_queue_size = max_priority_queue_size
        self.request_timeout = request_timeout
        self.max_wait_time = max_wait_time
        
        # Priority queues
        self.urgent_queue = asyncio.Queue(maxsize=max_priority_queue_size)
        self.high_queue = asyncio.Queue(maxsize=max_priority_queue_size)
        self.normal_queue = asyncio.Queue(maxsize=max_queue_size)
        self.low_queue = asyncio.Queue(maxsize=max_queue_size)
        
        # Request tracking
        self.active_requests: Dict[str, QueueRequest] = {}
        self.completed_requests: deque = deque(maxlen=1000)
        self.failed_requests: deque = deque(maxlen=1000)
        
        # Statistics
        self.stats = QueueStats(
            total_requests=0,
            pending_requests=0,
            completed_requests=0,
            failed_requests=0,
            avg_wait_time_ms=0.0,
            avg_processing_time_ms=0.0,
            queue_depth=0,
            priority_distribution={"urgent": 0, "high": 0, "normal": 0, "low": 0}
        )
        
        # Background tasks
        self.cleanup_task = None
        self.stats_task = None
        
    async def start(self):
        """Start background tasks"""
        self.cleanup_task = asyncio.create_task(self._cleanup_expired_requests())
        self.stats_task = asyncio.create_task(self._update_stats())
        logger.info("Queue manager started")
    
    async def stop(self):
        """Stop background tasks"""
        if self.cleanup_task:
            self.cleanup_task.cancel()
        if self.stats_task:
            self.stats_task.cancel()
        logger.info("Queue manager stopped")
    
    async def add_request(self, 
                         request_id: str,
                         session_id: str,
                         text: str,
                         emotion: str,
                         priority: Priority = Priority.NORMAL,
                         user_id: str = "",
                         metadata: Dict = None) -> bool:
        """Add request to appropriate queue"""
        
        current_time = time.time()
        
        request = QueueRequest(
            request_id=request_id,
            session_id=session_id,
            text=text,
            emotion=emotion,
            priority=priority,
            created_at=current_time,
            timeout_at=current_time + self.request_timeout,
            user_id=user_id,
            metadata=metadata or {}
        )
        
        try:
            # Add to appropriate queue based on priority
            if priority == Priority.URGENT:
                await self.urgent_queue.put(request)
            elif priority == Priority.HIGH:
                await self.high_queue.put(request)
            elif priority == Priority.NORMAL:
                await self.normal_queue.put(request)
            else:  # LOW
                await self.low_queue.put(request)
            
            # Track active request
            self.active_requests[request_id] = request
            self.stats.total_requests += 1
            self.stats.priority_distribution[priority.value] += 1
            
            logger.debug(f"Added request {request_id} to {priority.value} queue")
            return True
            
        except asyncio.QueueFull:
            logger.warning(f"Queue full for priority {priority.value}, rejecting request {request_id}")
            return False
        except Exception as e:
            logger.error(f"Error adding request {request_id}: {e}")
            return False
    
    async def get_next_request(self, timeout: float = 5.0) -> Optional[QueueRequest]:
        """Get next request from queues in priority order"""
        
        try:
            # Check queues in priority order
            queues_to_check = [
                (self.urgent_queue, Priority.URGENT),
                (self.high_queue, Priority.HIGH),
                (self.normal_queue, Priority.NORMAL),
                (self.low_queue, Priority.LOW)
            ]
            
            for queue, priority in queues_to_check:
                if not queue.empty():
                    request = await asyncio.wait_for(queue.get(), timeout=timeout)
                    
                    # Check if request has expired
                    if time.time() > request.timeout_at:
                        logger.warning(f"Request {request.request_id} expired, skipping")
                        self._mark_request_failed(request, "timeout")
                        continue
                    
                    logger.debug(f"Got request {request.request_id} from {priority.value} queue")
                    return request
            
            return None
            
        except asyncio.TimeoutError:
            logger.debug("Timeout waiting for next request")
            return None
        except Exception as e:
            logger.error(f"Error getting next request: {e}")
            return None
    
    async def mark_request_completed(self, request_id: str, processing_time_ms: float):
        """Mark request as completed"""
        if request_id in self.active_requests:
            request = self.active_requests.pop(request_id)
            
            # Add to completed requests
            completed_request = {
                "request_id": request_id,
                "session_id": request.session_id,
                "priority": request.priority.value,
                "created_at": request.created_at,
                "completed_at": time.time(),
                "processing_time_ms": processing_time_ms,
                "wait_time_ms": (time.time() - request.created_at) * 1000
            }
            
            self.completed_requests.append(completed_request)
            self.stats.completed_requests += 1
            
            logger.debug(f"Marked request {request_id} as completed")
    
    def mark_request_failed(self, request_id: str, reason: str):
        """Mark request as failed"""
        if request_id in self.active_requests:
            request = self.active_requests.pop(request_id)
            
            # Add to failed requests
            failed_request = {
                "request_id": request_id,
                "session_id": request.session_id,
                "priority": request.priority.value,
                "created_at": request.created_at,
                "failed_at": time.time(),
                "reason": reason,
                "wait_time_ms": (time.time() - request.created_at) * 1000
            }
            
            self.failed_requests.append(failed_request)
            self.stats.failed_requests += 1
            
            logger.warning(f"Marked request {request_id} as failed: {reason}")
    
    def _mark_request_failed(self, request: QueueRequest, reason: str):
        """Internal method to mark request as failed"""
        if request.request_id in self.active_requests:
            self.active_requests.pop(request.request_id)
        
        failed_request = {
            "request_id": request.request_id,
            "session_id": request.session_id,
            "priority": request.priority.value,
            "created_at": request.created_at,
            "failed_at": time.time(),
            "reason": reason,
            "wait_time_ms": (time.time() - request.created_at) * 1000
        }
        
        self.failed_requests.append(failed_request)
        self.stats.failed_requests += 1
    
    async def _cleanup_expired_requests(self):
        """Background task to clean up expired requests"""
        while True:
            try:
                current_time = time.time()
                expired_requests = []
                
                # Find expired requests
                for request_id, request in self.active_requests.items():
                    if current_time > request.timeout_at:
                        expired_requests.append(request_id)
                
                # Remove expired requests
                for request_id in expired_requests:
                    request = self.active_requests.pop(request_id)
                    self._mark_request_failed(request, "timeout")
                    logger.warning(f"Cleaned up expired request: {request_id}")
                
                await asyncio.sleep(1)  # Check every second
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(1)
    
    async def _update_stats(self):
        """Background task to update statistics"""
        while True:
            try:
                # Update pending requests count
                self.stats.pending_requests = len(self.active_requests)
                
                # Update queue depth
                self.stats.queue_depth = (
                    self.urgent_queue.qsize() +
                    self.high_queue.qsize() +
                    self.normal_queue.qsize() +
                    self.low_queue.qsize()
                )
                
                # Calculate average wait time
                if self.completed_requests:
                    total_wait_time = sum(req["wait_time_ms"] for req in self.completed_requests)
                    self.stats.avg_wait_time_ms = total_wait_time / len(self.completed_requests)
                
                # Calculate average processing time
                if self.completed_requests:
                    total_processing_time = sum(req["processing_time_ms"] for req in self.completed_requests)
                    self.stats.avg_processing_time_ms = total_processing_time / len(self.completed_requests)
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Error updating stats: {e}")
                await asyncio.sleep(5)
    
    def get_queue_stats(self) -> Dict:
        """Get current queue statistics"""
        return {
            "total_requests": self.stats.total_requests,
            "pending_requests": self.stats.pending_requests,
            "completed_requests": self.stats.completed_requests,
            "failed_requests": self.stats.failed_requests,
            "avg_wait_time_ms": self.stats.avg_wait_time_ms,
            "avg_processing_time_ms": self.stats.avg_processing_time_ms,
            "queue_depth": self.stats.queue_depth,
            "priority_distribution": self.stats.priority_distribution,
            "queue_sizes": {
                "urgent": self.urgent_queue.qsize(),
                "high": self.high_queue.qsize(),
                "normal": self.normal_queue.qsize(),
                "low": self.low_queue.qsize()
            },
            "active_requests": len(self.active_requests),
            "max_queue_size": self.max_queue_size,
            "max_priority_queue_size": self.max_priority_queue_size
        }
    
    def get_recent_requests(self, limit: int = 50) -> List[Dict]:
        """Get recent completed and failed requests"""
        recent_requests = []
        
        # Add recent completed requests
        for request in list(self.completed_requests)[-limit//2:]:
            recent_requests.append({
                **request,
                "status": "completed"
            })
        
        # Add recent failed requests
        for request in list(self.failed_requests)[-limit//2:]:
            recent_requests.append({
                **request,
                "status": "failed"
            })
        
        # Sort by timestamp
        recent_requests.sort(key=lambda x: x.get("completed_at", x.get("failed_at", 0)), reverse=True)
        
        return recent_requests[:limit]
    
    def is_queue_healthy(self) -> bool:
        """Check if queue is healthy"""
        # Queue is unhealthy if:
        # 1. Too many failed requests
        # 2. Average wait time too high
        # 3. Queue depth consistently high
        
        failure_rate = self.stats.failed_requests / max(self.stats.total_requests, 1) * 100
        avg_wait_time_s = self.stats.avg_wait_time_ms / 1000
        
        if failure_rate > 10:  # More than 10% failure rate
            return False
        
        if avg_wait_time_s > 15:  # Average wait time more than 15 seconds
            return False
        
        if self.stats.queue_depth > self.max_queue_size * 0.8:  # Queue 80% full
            return False
        
        return True
    
    def get_health_status(self) -> Dict:
        """Get queue health status"""
        is_healthy = self.is_queue_healthy()
        
        return {
            "healthy": is_healthy,
            "status": "healthy" if is_healthy else "degraded",
            "metrics": {
                "failure_rate": self.stats.failed_requests / max(self.stats.total_requests, 1) * 100,
                "avg_wait_time_s": self.stats.avg_wait_time_ms / 1000,
                "queue_utilization": self.stats.queue_depth / self.max_queue_size * 100,
                "throughput_per_minute": self.stats.completed_requests / max(time.time() / 60, 1)
            },
            "recommendations": self._get_recommendations()
        }
    
    def _get_recommendations(self) -> List[str]:
        """Get recommendations for queue optimization"""
        recommendations = []
        
        failure_rate = self.stats.failed_requests / max(self.stats.total_requests, 1) * 100
        if failure_rate > 5:
            recommendations.append("High failure rate detected - check model instances")
        
        avg_wait_time_s = self.stats.avg_wait_time_ms / 1000
        if avg_wait_time_s > 10:
            recommendations.append("High wait times - consider increasing model pool size")
        
        queue_utilization = self.stats.queue_depth / self.max_queue_size * 100
        if queue_utilization > 80:
            recommendations.append("Queue utilization high - consider increasing queue size")
        
        if not recommendations:
            recommendations.append("Queue is operating normally")
        
        return recommendations

# Usage example
async def main():
    """Test the queue manager"""
    queue_manager = QueueManager()
    await queue_manager.start()
    
    # Add some test requests
    await queue_manager.add_request("req1", "session1", "Hello", "empathetic", Priority.NORMAL)
    await queue_manager.add_request("req2", "session2", "Urgent", "calm", Priority.URGENT)
    
    # Get next request
    request = await queue_manager.get_next_request()
    if request:
        print(f"Got request: {request.request_id}")
        
        # Simulate processing
        await asyncio.sleep(0.1)
        
        # Mark as completed
        await queue_manager.mark_request_completed(request.request_id, 100.0)
    
    # Print stats
    stats = queue_manager.get_queue_stats()
    print(f"Queue stats: {stats}")
    
    await queue_manager.stop()

if __name__ == "__main__":
    asyncio.run(main())


