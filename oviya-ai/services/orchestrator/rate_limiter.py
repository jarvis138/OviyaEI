#!/usr/bin/env python3
"""
Oviya Rate Limiting & Abuse Prevention System
Epic 6: Critical production protection component
"""
import asyncio
import redis
import json
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from dataclasses import dataclass
from enum import Enum

class RateLimitResult(Enum):
    ALLOWED = "allowed"
    HOURLY_LIMIT = "hourly_limit_exceeded"
    DAILY_LIMIT = "daily_limit_exceeded"
    CONCURRENT_LIMIT = "concurrent_limit_exceeded"
    COST_LIMIT = "cost_limit_exceeded"

@dataclass
class RateLimitResponse:
    allowed: bool
    reason: str
    retry_after: Optional[int] = None
    current_usage: Dict[str, int] = None

class RateLimiter:
    """Main rate limiting system"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        
        # Rate limits configuration
        self.limits = {
            "sessions_per_hour": 10,
            "sessions_per_day": 50,
            "total_minutes_per_day": 120,
            "concurrent_sessions": 1,
            "messages_per_minute": 20,
            "api_calls_per_hour": 1000
        }
        
        # Cost limits
        self.cost_limits = {
            "free_tier_daily": 1.0,  # $1/day
            "paid_tier_daily": 10.0,  # $10/day
            "emergency_limit": 50.0  # $50/day emergency
        }
    
    async def check_rate_limit(self, user_id: str, request_type: str = "session") -> RateLimitResponse:
        """Check if user is within rate limits"""
        
        # Check hourly limit
        hour_key = f"rate_limit:{user_id}:hour:{datetime.now().strftime('%Y-%m-%d-%H')}"
        hour_count = await self.redis.incr(hour_key)
        await self.redis.expire(hour_key, 3600)  # 1 hour TTL
        
        if hour_count > self.limits["sessions_per_hour"]:
            return RateLimitResponse(
                allowed=False,
                reason="Hourly session limit exceeded",
                retry_after=3600 - (datetime.now().minute * 60),
                current_usage={"hourly": hour_count}
            )
        
        # Check daily limit
        day_key = f"rate_limit:{user_id}:day:{datetime.now().strftime('%Y-%m-%d')}"
        day_count = await self.redis.incr(day_key)
        await self.redis.expire(day_key, 86400)  # 24 hours TTL
        
        if day_count > self.limits["sessions_per_day"]:
            return RateLimitResponse(
                allowed=False,
                reason="Daily session limit exceeded",
                retry_after=86400,
                current_usage={"daily": day_count}
            )
        
        # Check concurrent sessions
        concurrent_key = f"concurrent_sessions:{user_id}"
        concurrent_count = await self.redis.scard(concurrent_key)
        
        if concurrent_count >= self.limits["concurrent_sessions"]:
            return RateLimitResponse(
                allowed=False,
                reason="Concurrent session limit exceeded",
                retry_after=300,  # 5 minutes
                current_usage={"concurrent": concurrent_count}
            )
        
        # Check cost limits
        cost_check = await self._check_cost_limit(user_id)
        if not cost_check["allowed"]:
            return RateLimitResponse(
                allowed=False,
                reason="Daily cost limit exceeded",
                retry_after=86400,
                current_usage={"daily_cost": cost_check["current_cost"]}
            )
        
        return RateLimitResponse(
            allowed=True,
            reason="Rate limit check passed",
            current_usage={
                "hourly": hour_count,
                "daily": day_count,
                "concurrent": concurrent_count
            }
        )
    
    async def start_session(self, user_id: str, session_id: str) -> bool:
        """Track session start"""
        concurrent_key = f"concurrent_sessions:{user_id}"
        await self.redis.sadd(concurrent_key, session_id)
        await self.redis.expire(concurrent_key, 3600)  # 1 hour TTL
        return True
    
    async def end_session(self, user_id: str, session_id: str) -> bool:
        """Track session end"""
        concurrent_key = f"concurrent_sessions:{user_id}"
        await self.redis.srem(concurrent_key, session_id)
        return True
    
    async def track_api_call(self, user_id: str, cost: float = 0.001) -> bool:
        """Track API call and cost"""
        # Track API calls
        api_key = f"api_calls:{user_id}:hour:{datetime.now().strftime('%Y-%m-%d-%H')}"
        await self.redis.incr(api_key)
        await self.redis.expire(api_key, 3600)
        
        # Track cost
        cost_key = f"daily_cost:{user_id}:{datetime.now().strftime('%Y-%m-%d')}"
        await self.redis.hincrbyfloat(cost_key, "total", cost)
        await self.redis.expire(cost_key, 86400)
        
        return True
    
    async def _check_cost_limit(self, user_id: str) -> Dict:
        """Check if user has exceeded cost limits"""
        cost_key = f"daily_cost:{user_id}:{datetime.now().strftime('%Y-%m-%d')}"
        daily_cost = await self.redis.hget(cost_key, "total")
        
        if daily_cost is None:
            daily_cost = 0.0
        else:
            daily_cost = float(daily_cost)
        
        # Determine user tier (simplified)
        user_tier = await self._get_user_tier(user_id)
        max_cost = self.cost_limits[f"{user_tier}_tier_daily"]
        
        return {
            "allowed": daily_cost < max_cost,
            "current_cost": daily_cost,
            "max_cost": max_cost,
            "tier": user_tier
        }
    
    async def _get_user_tier(self, user_id: str) -> str:
        """Get user subscription tier"""
        # In production, this would check Firebase or database
        # For now, return "free" for all users
        return "free"

class AbuseDetector:
    """Detects abusive behavior patterns"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        
        # Abuse patterns
        self.abuse_patterns = {
            "rapid_fire": {
                "max_messages_per_minute": 30,
                "window_minutes": 1
            },
            "spam_content": {
                "max_similar_messages": 5,
                "window_minutes": 10
            },
            "excessive_interrupts": {
                "max_interrupts_per_session": 20,
                "window_minutes": 30
            }
        }
    
    async def detect_abuse(self, user_id: str, behavior_type: str, metadata: Dict) -> Dict:
        """Detect abusive behavior"""
        
        if behavior_type == "rapid_fire":
            return await self._detect_rapid_fire(user_id)
        elif behavior_type == "spam_content":
            return await self._detect_spam_content(user_id, metadata.get("message", ""))
        elif behavior_type == "excessive_interrupts":
            return await self._detect_excessive_interrupts(user_id)
        
        return {"is_abuse": False, "confidence": 0.0}
    
    async def _detect_rapid_fire(self, user_id: str) -> Dict:
        """Detect rapid-fire messaging"""
        minute_key = f"messages:{user_id}:minute:{datetime.now().strftime('%Y-%m-%d-%H-%M')}"
        message_count = await self.redis.incr(minute_key)
        await self.redis.expire(minute_key, 60)
        
        max_messages = self.abuse_patterns["rapid_fire"]["max_messages_per_minute"]
        
        return {
            "is_abuse": message_count > max_messages,
            "confidence": min(message_count / max_messages, 1.0),
            "message_count": message_count,
            "threshold": max_messages
        }
    
    async def _detect_spam_content(self, user_id: str, message: str) -> Dict:
        """Detect spam content"""
        # Simple spam detection based on message similarity
        message_hash = hash(message.lower().strip())
        spam_key = f"spam_messages:{user_id}:{datetime.now().strftime('%Y-%m-%d-%H')}"
        
        await self.redis.sadd(spam_key, message_hash)
        await self.redis.expire(spam_key, 3600)
        
        # Check for duplicate messages
        duplicate_count = await self.redis.scard(spam_key)
        max_similar = self.abuse_patterns["spam_content"]["max_similar_messages"]
        
        return {
            "is_abuse": duplicate_count > max_similar,
            "confidence": min(duplicate_count / max_similar, 1.0),
            "duplicate_count": duplicate_count,
            "threshold": max_similar
        }
    
    async def _detect_excessive_interrupts(self, user_id: str) -> Dict:
        """Detect excessive interrupt behavior"""
        interrupt_key = f"interrupts:{user_id}:session:{datetime.now().strftime('%Y-%m-%d-%H')}"
        interrupt_count = await self.redis.incr(interrupt_key)
        await self.redis.expire(interrupt_key, 1800)  # 30 minutes
        
        max_interrupts = self.abuse_patterns["excessive_interrupts"]["max_interrupts_per_session"]
        
        return {
            "is_abuse": interrupt_count > max_interrupts,
            "confidence": min(interrupt_count / max_interrupts, 1.0),
            "interrupt_count": interrupt_count,
            "threshold": max_interrupts
        }

class CostProtection:
    """Prevents runaway API costs"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        
        # Cost tracking
        self.cost_per_action = {
            "whisper_transcription": 0.006 / 60,  # $0.006 per minute
            "gpt4_generation": 0.03 / 1000,       # $0.03 per 1K tokens
            "csm_generation": 0.10 / 60,          # Estimated GPU cost
            "moderation_check": 0.0001             # $0.0001 per check
        }
    
    async def track_cost(self, user_id: str, action: str, metadata: Dict = None) -> Dict:
        """Track cost for an action"""
        cost = self.cost_per_action.get(action, 0.001)  # Default cost
        
        # Adjust cost based on metadata
        if action == "gpt4_generation" and metadata:
            token_count = metadata.get("token_count", 100)
            cost = cost * token_count
        elif action == "whisper_transcription" and metadata:
            duration_seconds = metadata.get("duration_seconds", 10)
            cost = cost * (duration_seconds / 60)
        
        # Track daily cost
        cost_key = f"daily_cost:{user_id}:{datetime.now().strftime('%Y-%m-%d')}"
        await self.redis.hincrbyfloat(cost_key, "total", cost)
        await self.redis.hincrbyfloat(cost_key, action, cost)
        await self.redis.expire(cost_key, 86400)
        
        # Check if user has exceeded limit
        daily_cost = await self.redis.hget(cost_key, "total")
        daily_cost = float(daily_cost) if daily_cost else 0.0
        
        user_tier = await self._get_user_tier(user_id)
        max_cost = 1.0 if user_tier == "free" else 10.0
        
        return {
            "action_cost": cost,
            "daily_total": daily_cost,
            "max_daily": max_cost,
            "exceeded": daily_cost > max_cost,
            "remaining": max_cost - daily_cost
        }
    
    async def get_cost_summary(self, user_id: str) -> Dict:
        """Get user's cost summary"""
        cost_key = f"daily_cost:{user_id}:{datetime.now().strftime('%Y-%m-%d')}"
        cost_data = await self.redis.hgetall(cost_key)
        
        if not cost_data:
            return {
                "daily_total": 0.0,
                "breakdown": {},
                "remaining": 1.0  # Free tier limit
            }
        
        daily_total = float(cost_data.get("total", 0))
        breakdown = {k: float(v) for k, v in cost_data.items() if k != "total"}
        
        user_tier = await self._get_user_tier(user_id)
        max_cost = 1.0 if user_tier == "free" else 10.0
        
        return {
            "daily_total": daily_total,
            "breakdown": breakdown,
            "remaining": max_cost - daily_total,
            "tier": user_tier
        }
    
    async def _get_user_tier(self, user_id: str) -> str:
        """Get user subscription tier"""
        # In production, check Firebase or database
        return "free"

# Usage example
async def main():
    """Test the rate limiting system"""
    redis_client = redis.Redis(host='localhost', port=6379, db=0)
    rate_limiter = RateLimiter(redis_client)
    abuse_detector = AbuseDetector(redis_client)
    cost_protection = CostProtection(redis_client)
    
    user_id = "test_user_123"
    
    # Test rate limiting
    result = await rate_limiter.check_rate_limit(user_id)
    print(f"Rate limit check: {result.allowed} - {result.reason}")
    
    # Test cost tracking
    cost_result = await cost_protection.track_cost(user_id, "gpt4_generation", {"token_count": 150})
    print(f"Cost tracking: ${cost_result['action_cost']:.4f} - Daily total: ${cost_result['daily_total']:.4f}")
    
    # Test abuse detection
    abuse_result = await abuse_detector.detect_abuse(user_id, "rapid_fire", {})
    print(f"Abuse detection: {abuse_result['is_abuse']} - Confidence: {abuse_result['confidence']}")

if __name__ == "__main__":
    asyncio.run(main())


