#!/usr/bin/env python3
"""
Oviya CSM Model Pool Manager
Manages warm model instances for low latency
"""
import asyncio
import time
import torch
import logging
from typing import List, Optional
from dataclasses import dataclass
import psutil
import GPUtil

logger = logging.getLogger(__name__)

@dataclass
class ModelInstance:
    """Represents a model instance"""
    id: str
    pipeline: 'CSMGenerationPipeline'
    created_at: float
    last_used: float
    usage_count: int
    is_available: bool = True

class ModelPool:
    """Manages warm model instances for optimal performance"""
    
    def __init__(self, pool_size: int = 2, max_pool_size: int = 8):
        self.pool_size = pool_size
        self.max_pool_size = max_pool_size
        self.instances: List[ModelInstance] = []
        self.available_queue = asyncio.Queue()
        self.busy_instances = set()
        self.instance_counter = 0
        
        # Performance metrics
        self.total_requests = 0
        self.total_latency_ms = 0
        self.pool_hits = 0
        self.pool_misses = 0
        
    async def initialize(self):
        """Initialize model pool with warm instances"""
        logger.info(f"Initializing model pool with {self.pool_size} instances...")
        
        for i in range(self.pool_size):
            await self._create_instance()
        
        logger.info(f"Model pool initialized with {len(self.instances)} instances")
        logger.info(f"GPU Memory: {self._get_gpu_memory_info()}")
    
    async def get_instance(self, timeout: float = 10.0) -> Optional[ModelInstance]:
        """Get available model instance"""
        try:
            # Try to get from available queue
            instance = await asyncio.wait_for(
                self.available_queue.get(), 
                timeout=timeout
            )
            
            if instance.is_available:
                self.busy_instances.add(instance.id)
                instance.is_available = False
                instance.last_used = time.time()
                instance.usage_count += 1
                self.pool_hits += 1
                
                logger.debug(f"Got instance {instance.id} from pool")
                return instance
            
            # If instance is not available, try to create new one
            if len(self.instances) < self.max_pool_size:
                instance = await self._create_instance()
                if instance:
                    self.busy_instances.add(instance.id)
                    instance.is_available = False
                    instance.last_used = time.time()
                    instance.usage_count += 1
                    self.pool_misses += 1
                    return instance
            
            # No instances available
            logger.warning("No model instances available")
            return None
            
        except asyncio.TimeoutError:
            logger.warning("Timeout waiting for model instance")
            return None
        except Exception as e:
            logger.error(f"Error getting model instance: {e}")
            return None
    
    async def return_instance(self, instance: ModelInstance):
        """Return instance to pool"""
        try:
            instance.is_available = True
            self.busy_instances.discard(instance.id)
            
            # Add back to available queue
            await self.available_queue.put(instance)
            
            logger.debug(f"Returned instance {instance.id} to pool")
            
        except Exception as e:
            logger.error(f"Error returning instance: {e}")
    
    async def _create_instance(self) -> Optional[ModelInstance]:
        """Create new model instance"""
        try:
            instance_id = f"csm_instance_{self.instance_counter}"
            self.instance_counter += 1
            
            logger.info(f"Creating model instance: {instance_id}")
            
            # Import and initialize pipeline
            from server import CSMGenerationPipeline
            pipeline = CSMGenerationPipeline()
            await pipeline.initialize()
            
            instance = ModelInstance(
                id=instance_id,
                pipeline=pipeline,
                created_at=time.time(),
                last_used=time.time(),
                usage_count=0
            )
            
            self.instances.append(instance)
            await self.available_queue.put(instance)
            
            logger.info(f"Created model instance: {instance_id}")
            return instance
            
        except Exception as e:
            logger.error(f"Error creating model instance: {e}")
            return None
    
    async def cleanup_unused_instances(self, max_idle_time: float = 300.0):
        """Clean up unused instances to free memory"""
        current_time = time.time()
        instances_to_remove = []
        
        for instance in self.instances:
            idle_time = current_time - instance.last_used
            
            # Remove instances that have been idle too long
            if (idle_time > max_idle_time and 
                instance.is_available and 
                len(self.instances) > self.pool_size):
                
                instances_to_remove.append(instance)
        
        for instance in instances_to_remove:
            await self._remove_instance(instance)
            logger.info(f"Removed idle instance: {instance.id}")
    
    async def _remove_instance(self, instance: ModelInstance):
        """Remove instance from pool"""
        try:
            # Remove from instances list
            if instance in self.instances:
                self.instances.remove(instance)
            
            # Remove from busy instances
            self.busy_instances.discard(instance.id)
            
            # Clean up pipeline resources
            if hasattr(instance.pipeline, 'generator'):
                del instance.pipeline.generator
            
            logger.info(f"Removed instance: {instance.id}")
            
        except Exception as e:
            logger.error(f"Error removing instance: {e}")
    
    def get_pool_stats(self) -> dict:
        """Get pool statistics"""
        current_time = time.time()
        
        return {
            "total_instances": len(self.instances),
            "available_instances": len([i for i in self.instances if i.is_available]),
            "busy_instances": len(self.busy_instances),
            "pool_size": self.pool_size,
            "max_pool_size": self.max_pool_size,
            "total_requests": self.total_requests,
            "pool_hits": self.pool_hits,
            "pool_misses": self.pool_misses,
            "hit_rate": self.pool_hits / max(self.total_requests, 1) * 100,
            "avg_latency_ms": self.total_latency_ms / max(self.total_requests, 1),
            "gpu_memory_used": self._get_gpu_memory_usage(),
            "system_memory_used": psutil.virtual_memory().percent,
            "instances": [
                {
                    "id": instance.id,
                    "created_at": instance.created_at,
                    "last_used": instance.last_used,
                    "usage_count": instance.usage_count,
                    "is_available": instance.is_available,
                    "idle_time": current_time - instance.last_used
                }
                for instance in self.instances
            ]
        }
    
    def _get_gpu_memory_info(self) -> str:
        """Get GPU memory information"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                return f"{gpu.memoryUsed}MB / {gpu.memoryTotal}MB ({gpu.memoryUtil * 100:.1f}%)"
            return "No GPU detected"
        except:
            return "GPU info unavailable"
    
    def _get_gpu_memory_usage(self) -> float:
        """Get GPU memory usage percentage"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUtil * 100
            return 0.0
        except:
            return 0.0
    
    async def health_check(self) -> dict:
        """Perform health check on all instances"""
        health_status = {
            "overall_health": "healthy",
            "instances": [],
            "issues": []
        }
        
        for instance in self.instances:
            instance_health = {
                "id": instance.id,
                "status": "healthy",
                "issues": []
            }
            
            # Check if instance is responsive
            try:
                # Simple test generation
                test_audio = instance.pipeline.generator.generate(
                    text="test",
                    speaker=1,
                    context=[],
                    max_audio_length_ms=1000,
                    temperature=0.7,
                    do_sample=True
                )
                
                if test_audio is None or len(test_audio) == 0:
                    instance_health["status"] = "unhealthy"
                    instance_health["issues"].append("No audio generated")
                    health_status["issues"].append(f"Instance {instance.id}: No audio generated")
                
            except Exception as e:
                instance_health["status"] = "unhealthy"
                instance_health["issues"].append(f"Generation error: {str(e)}")
                health_status["issues"].append(f"Instance {instance.id}: {str(e)}")
            
            health_status["instances"].append(instance_health)
        
        # Overall health assessment
        unhealthy_instances = len([i for i in health_status["instances"] if i["status"] == "unhealthy"])
        if unhealthy_instances > 0:
            health_status["overall_health"] = "degraded"
        
        if unhealthy_instances >= len(self.instances) / 2:
            health_status["overall_health"] = "unhealthy"
        
        return health_status

# Background task for pool maintenance
async def pool_maintenance_task(model_pool: ModelPool):
    """Background task for pool maintenance"""
    while True:
        try:
            # Cleanup unused instances every 5 minutes
            await model_pool.cleanup_unused_instances(max_idle_time=300.0)
            
            # Log pool stats every minute
            stats = model_pool.get_pool_stats()
            logger.info(f"Pool stats: {stats['available_instances']}/{stats['total_instances']} available, "
                       f"hit rate: {stats['hit_rate']:.1f}%, "
                       f"GPU memory: {stats['gpu_memory_used']:.1f}%")
            
            await asyncio.sleep(60)  # Run every minute
            
        except Exception as e:
            logger.error(f"Error in pool maintenance: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    # Test the model pool
    async def test_pool():
        pool = ModelPool(pool_size=2)
        await pool.initialize()
        
        # Test getting instances
        instance1 = await pool.get_instance()
        instance2 = await pool.get_instance()
        
        print(f"Got instances: {instance1.id if instance1 else None}, {instance2.id if instance2 else None}")
        
        # Test returning instances
        if instance1:
            await pool.return_instance(instance1)
        if instance2:
            await pool.return_instance(instance2)
        
        # Print stats
        stats = pool.get_pool_stats()
        print(f"Pool stats: {stats}")
    
    asyncio.run(test_pool())


