#!/usr/bin/env python3
"""
Redis MCP Server for Oviya EI
Provides high-performance caching, real-time state management, and rate limiting
"""

import asyncio
import json
import os
import sys
import redis.asyncio as redis
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import hashlib

# Add project paths for standalone execution
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class OviyaRedisServer:
    """
    MCP Server for Redis operations in Oviya EI

    Provides:
    - Personality vector caching
    - Session state management
    - Rate limiting
    - Real-time WebRTC session management
    - Cache invalidation and optimization
    """

    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.redis_client = None
        self.cache_ttl = {
            "personality_vector": 3600,  # 1 hour
            "session_state": 7200,      # 2 hours
            "user_profile": 1800,       # 30 minutes
            "analytics": 300,           # 5 minutes
            "rate_limit": 60,           # 1 minute
        }

    async def initialize_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)
            await self.redis_client.ping()
            print("Redis MCP Server initialized successfully")

            # Initialize rate limiting structures
            await self._initialize_rate_limits()

        except Exception as e:
            print(f"Failed to initialize Redis: {e}")
            raise

    async def _initialize_rate_limits(self):
        """Initialize rate limiting data structures"""
        # Create a simple Lua script for rate limiting
        self.rate_limit_script = await self.redis_client.script_load("""
            local key = KEYS[1]
            local limit = tonumber(ARGV[1])
            local window = tonumber(ARGV[2])
            local current = redis.call('INCR', key)
            if current == 1 then
                redis.call('EXPIRE', key, window)
            end
            return current <= limit
        """)

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests"""

        if request.get("method") == "tools/call":
            tool_name = request["params"]["name"]
            arguments = request["params"].get("arguments", {})

            try:
                if tool_name == "cache_personality_vector":
                    return await self._cache_personality_vector(
                        arguments.get("user_id"),
                        arguments.get("vector")
                    )
                elif tool_name == "get_cached_personality_vector":
                    return await self._get_cached_personality_vector(arguments.get("user_id"))
                elif tool_name == "set_session_state":
                    return await self._set_session_state(
                        arguments.get("session_id"),
                        arguments.get("state")
                    )
                elif tool_name == "get_session_state":
                    return await self._get_session_state(arguments.get("session_id"))
                elif tool_name == "check_rate_limit":
                    return await self._check_rate_limit(
                        arguments.get("identifier"),
                        arguments.get("limit", 60),
                        arguments.get("window", 60)
                    )
                elif tool_name == "cache_user_profile":
                    return await self._cache_user_profile(
                        arguments.get("user_id"),
                        arguments.get("profile")
                    )
                elif tool_name == "get_cached_user_profile":
                    return await self._get_cached_user_profile(arguments.get("user_id"))
                elif tool_name == "invalidate_cache":
                    return await self._invalidate_cache(arguments.get("pattern"))
                elif tool_name == "get_cache_stats":
                    return await self._get_cache_stats()
                elif tool_name == "set_webrtc_session":
                    return await self._set_webrtc_session(
                        arguments.get("session_id"),
                        arguments.get("webrtc_data")
                    )
                elif tool_name == "get_webrtc_session":
                    return await self._get_webrtc_session(arguments.get("session_id"))
                else:
                    return {"error": f"Unknown tool: {tool_name}"}

            except Exception as e:
                return {"error": str(e)}

        elif request.get("method") == "tools/list":
            return {
                "tools": [
                    {
                        "name": "cache_personality_vector",
                        "description": "Cache user's personality vector for fast access",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "string"},
                                "vector": {"type": "object"}
                            },
                            "required": ["user_id", "vector"]
                        }
                    },
                    {
                        "name": "get_cached_personality_vector",
                        "description": "Get cached personality vector",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "string"}
                            },
                            "required": ["user_id"]
                        }
                    },
                    {
                        "name": "set_session_state",
                        "description": "Set real-time session state",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "session_id": {"type": "string"},
                                "state": {"type": "object"}
                            },
                            "required": ["session_id", "state"]
                        }
                    },
                    {
                        "name": "get_session_state",
                        "description": "Get session state",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "session_id": {"type": "string"}
                            },
                            "required": ["session_id"]
                        }
                    },
                    {
                        "name": "check_rate_limit",
                        "description": "Check if request is within rate limits",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "identifier": {"type": "string"},
                                "limit": {"type": "integer", "default": 60},
                                "window": {"type": "integer", "default": 60}
                            },
                            "required": ["identifier"]
                        }
                    },
                    {
                        "name": "cache_user_profile",
                        "description": "Cache user profile data",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "string"},
                                "profile": {"type": "object"}
                            },
                            "required": ["user_id", "profile"]
                        }
                    },
                    {
                        "name": "get_cached_user_profile",
                        "description": "Get cached user profile",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "string"}
                            },
                            "required": ["user_id"]
                        }
                    },
                    {
                        "name": "invalidate_cache",
                        "description": "Invalidate cache entries matching pattern",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "pattern": {"type": "string"}
                            },
                            "required": ["pattern"]
                        }
                    },
                    {
                        "name": "get_cache_stats",
                        "description": "Get cache performance statistics",
                        "inputSchema": {"type": "object", "properties": {}}
                    },
                    {
                        "name": "set_webrtc_session",
                        "description": "Store WebRTC session data",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "session_id": {"type": "string"},
                                "webrtc_data": {"type": "object"}
                            },
                            "required": ["session_id", "webrtc_data"]
                        }
                    },
                    {
                        "name": "get_webrtc_session",
                        "description": "Get WebRTC session data",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "session_id": {"type": "string"}
                            },
                            "required": ["session_id"]
                        }
                    }
                ]
            }

        return {"error": "Method not supported"}

    async def _cache_personality_vector(self, user_id: str, vector: Dict[str, float]) -> Dict[str, Any]:
        """Cache personality vector with TTL"""
        key = f"personality:{user_id}"
        await self.redis_client.setex(
            key,
            self.cache_ttl["personality_vector"],
            json.dumps(vector)
        )

        # Also cache a hash for change detection
        vector_hash = hashlib.md5(json.dumps(vector, sort_keys=True).encode()).hexdigest()
        await self.redis_client.setex(
            f"personality_hash:{user_id}",
            self.cache_ttl["personality_vector"],
            vector_hash
        )

        return {
            "content": [{"type": "text", "text": json.dumps({
                "status": "cached",
                "user_id": user_id,
                "ttl": self.cache_ttl["personality_vector"]
            })}]
        }

    async def _get_cached_personality_vector(self, user_id: str) -> Dict[str, Any]:
        """Get cached personality vector"""
        key = f"personality:{user_id}"
        vector_json = await self.redis_client.get(key)

        if vector_json:
            vector = json.loads(vector_json)
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "status": "hit",
                    "user_id": user_id,
                    "vector": vector
                })}]
            }
        else:
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "status": "miss",
                    "user_id": user_id
                })}]
            }

    async def _set_session_state(self, session_id: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """Set real-time session state"""
        key = f"session:{session_id}"
        await self.redis_client.setex(
            key,
            self.cache_ttl["session_state"],
            json.dumps(state)
        )

        return {
            "content": [{"type": "text", "text": json.dumps({
                "status": "set",
                "session_id": session_id,
                "ttl": self.cache_ttl["session_state"]
            })}]
        }

    async def _get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Get session state"""
        key = f"session:{session_id}"
        state_json = await self.redis_client.get(key)

        if state_json:
            state = json.loads(state_json)
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "status": "hit",
                    "session_id": session_id,
                    "state": state
                })}]
            }
        else:
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "status": "miss",
                    "session_id": session_id
                })}]
            }

    async def _check_rate_limit(self, identifier: str, limit: int, window: int) -> Dict[str, Any]:
        """Check if request is within rate limits"""
        key = f"ratelimit:{identifier}"

        # Use Lua script for atomic rate limiting
        allowed = await self.redis_client.evalsha(
            self.rate_limit_script,
            keys=[key],
            args=[limit, window]
        )

        return {
            "content": [{"type": "text", "text": json.dumps({
                "allowed": bool(allowed),
                "identifier": identifier,
                "limit": limit,
                "window": window,
                "remaining": max(0, limit - (await self.redis_client.get(key) or 0))
            })}]
        }

    async def _cache_user_profile(self, user_id: str, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Cache user profile data"""
        key = f"profile:{user_id}"
        await self.redis_client.setex(
            key,
            self.cache_ttl["user_profile"],
            json.dumps(profile)
        )

        return {
            "content": [{"type": "text", "text": json.dumps({
                "status": "cached",
                "user_id": user_id,
                "ttl": self.cache_ttl["user_profile"]
            })}]
        }

    async def _get_cached_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get cached user profile"""
        key = f"profile:{user_id}"
        profile_json = await self.redis_client.get(key)

        if profile_json:
            profile = json.loads(profile_json)
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "status": "hit",
                    "user_id": user_id,
                    "profile": profile
                })}]
            }
        else:
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "status": "miss",
                    "user_id": user_id
                })}]
            }

    async def _invalidate_cache(self, pattern: str) -> Dict[str, Any]:
        """Invalidate cache entries matching pattern"""
        # Get all keys matching pattern
        keys = await self.redis_client.keys(pattern)

        if keys:
            await self.redis_client.delete(*keys)
            deleted_count = len(keys)
        else:
            deleted_count = 0

        return {
            "content": [{"type": "text", "text": json.dumps({
                "status": "invalidated",
                "pattern": pattern,
                "keys_deleted": deleted_count
            })}]
        }

    async def _get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        info = await self.redis_client.info()

        # Get cache hit/miss stats (simplified)
        personality_keys = await self.redis_client.keys("personality:*")
        session_keys = await self.redis_client.keys("session:*")
        profile_keys = await self.redis_client.keys("profile:*")

        stats = {
            "redis_info": {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "total_connections_received": info.get("total_connections_received", 0)
            },
            "cache_stats": {
                "personality_vectors_cached": len(personality_keys),
                "active_sessions": len(session_keys),
                "profiles_cached": len(profile_keys),
                "total_cached_items": len(personality_keys) + len(session_keys) + len(profile_keys)
            }
        }

        return {
            "content": [{"type": "text", "text": json.dumps(stats)}]
        }

    async def _set_webrtc_session(self, session_id: str, webrtc_data: Dict[str, Any]) -> Dict[str, Any]:
        """Store WebRTC session data for real-time voice"""
        key = f"webrtc:{session_id}"
        await self.redis_client.setex(
            key,
            3600,  # 1 hour for WebRTC sessions
            json.dumps(webrtc_data)
        )

        return {
            "content": [{"type": "text", "text": json.dumps({
                "status": "stored",
                "session_id": session_id,
                "ttl": 3600
            })}]
        }

    async def _get_webrtc_session(self, session_id: str) -> Dict[str, Any]:
        """Get WebRTC session data"""
        key = f"webrtc:{session_id}"
        webrtc_json = await self.redis_client.get(key)

        if webrtc_json:
            webrtc_data = json.loads(webrtc_json)
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "status": "found",
                    "session_id": session_id,
                    "webrtc_data": webrtc_data
                })}]
            }
        else:
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "status": "not_found",
                    "session_id": session_id
                })}]
            }

async def main():
    """Main MCP server loop"""
    server = OviyaRedisServer()

    try:
        await server.initialize_redis()

        # Read from stdin, write to stdout (MCP stdio protocol)
        for line in sys.stdin:
            try:
                request = json.loads(line.strip())
                response = await server.handle_request(request)
                print(json.dumps(response), flush=True)
            except json.JSONDecodeError:
                print(json.dumps({"error": "Invalid JSON"}), flush=True)
            except Exception as e:
                print(json.dumps({"error": str(e)}), flush=True)
    except Exception as e:
        print(f"Server initialization failed: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
