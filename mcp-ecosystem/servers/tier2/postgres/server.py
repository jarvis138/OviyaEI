#!/usr/bin/env python3
"""
PostgreSQL MCP Server for Oviya EI
Provides structured data storage and retrieval for user profiles, sessions, and analytics
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import asyncpg
import uuid

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.insert(0, project_root)

class OviyaPostgresServer:
    """
    MCP Server for PostgreSQL operations in Oviya EI

    Provides:
    - User profile management
    - Session analytics
    - Conversation logs
    - Billing and subscription data
    """

    def __init__(self):
        self.connection_pool = None
        self.database_url = os.getenv("DATABASE_URL", "postgresql://oviya:oviya_password@localhost:5432/oviya_db")

    async def initialize_database(self):
        """Initialize database connection pool and create tables"""
        try:
            self.connection_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )

            # Create tables if they don't exist
            await self._create_tables()
            print("PostgreSQL MCP Server initialized successfully")

        except Exception as e:
            print(f"Failed to initialize PostgreSQL: {e}")
            raise

    async def _create_tables(self):
        """Create necessary database tables"""

        # User profiles table
        await self.connection_pool.execute("""
            CREATE TABLE IF NOT EXISTS user_profiles (
                user_id VARCHAR(255) PRIMARY KEY,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                demographic_data JSONB,
                preferences JSONB DEFAULT '{}',
                subscription_status VARCHAR(50) DEFAULT 'free',
                total_sessions INTEGER DEFAULT 0,
                total_messages INTEGER DEFAULT 0,
                risk_assessment JSONB DEFAULT '{}',
                metadata JSONB DEFAULT '{}'
            )
        """)

        # Conversation sessions table
        await self.connection_pool.execute("""
            CREATE TABLE IF NOT EXISTS conversation_sessions (
                session_id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ended_at TIMESTAMP,
                duration_seconds INTEGER,
                message_count INTEGER DEFAULT 0,
                emotion_summary JSONB DEFAULT '{}',
                personality_vector JSONB DEFAULT '{}',
                crisis_events JSONB DEFAULT '[]',
                metadata JSONB DEFAULT '{}'
            )
        """)

        # Individual messages table
        await self.connection_pool.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id VARCHAR(255) PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL REFERENCES conversation_sessions(session_id),
                user_id VARCHAR(255) NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                message_type VARCHAR(20) CHECK (message_type IN ('user', 'oviya', 'system')),
                content TEXT,
                emotion VARCHAR(50),
                personality_vector JSONB,
                prosody_data JSONB DEFAULT '{}',
                crisis_flags JSONB DEFAULT '{}',
                metadata JSONB DEFAULT '{}'
            )
        """)

        # Analytics events table
        await self.connection_pool.execute("""
            CREATE TABLE IF NOT EXISTS analytics_events (
                event_id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255),
                session_id VARCHAR(255),
                event_type VARCHAR(100),
                event_data JSONB DEFAULT '{}',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                ip_address INET,
                user_agent TEXT
            )
        """)

        # Create indexes for performance
        await self.connection_pool.execute("""
            CREATE INDEX IF NOT EXISTS idx_user_profiles_last_seen ON user_profiles(last_seen);
            CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON conversation_sessions(user_id);
            CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id);
            CREATE INDEX IF NOT EXISTS idx_messages_user_id_timestamp ON messages(user_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_analytics_user_timestamp ON analytics_events(user_id, timestamp);
        """)

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests"""

        if request.get("method") == "tools/call":
            tool_name = request["params"]["name"]
            arguments = request["params"].get("arguments", {})

            try:
                if tool_name == "get_user_profile":
                    return await self._get_user_profile(arguments.get("user_id"))
                elif tool_name == "update_user_profile":
                    return await self._update_user_profile(
                        arguments.get("user_id"),
                        arguments.get("updates")
                    )
                elif tool_name == "create_session":
                    return await self._create_session(arguments.get("user_id"), arguments.get("metadata", {}))
                elif tool_name == "end_session":
                    return await self._end_session(arguments.get("session_id"))
                elif tool_name == "log_message":
                    return await self._log_message(arguments)
                elif tool_name == "get_user_analytics":
                    return await self._get_user_analytics(arguments.get("user_id"), arguments.get("days", 30))
                elif tool_name == "get_system_analytics":
                    return await self._get_system_analytics(arguments.get("days", 30))
                elif tool_name == "export_user_data":
                    return await self._export_user_data(arguments.get("user_id"))
                else:
                    return {"error": f"Unknown tool: {tool_name}"}

            except Exception as e:
                return {"error": str(e)}

        elif request.get("method") == "tools/list":
            return {
                "tools": [
                    {
                        "name": "get_user_profile",
                        "description": "Get complete user profile data",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "string"}
                            },
                            "required": ["user_id"]
                        }
                    },
                    {
                        "name": "update_user_profile",
                        "description": "Update user profile data",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "string"},
                                "updates": {"type": "object"}
                            },
                            "required": ["user_id", "updates"]
                        }
                    },
                    {
                        "name": "create_session",
                        "description": "Create new conversation session",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "string"},
                                "metadata": {"type": "object"}
                            },
                            "required": ["user_id"]
                        }
                    },
                    {
                        "name": "end_session",
                        "description": "End conversation session",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "session_id": {"type": "string"}
                            },
                            "required": ["session_id"]
                        }
                    },
                    {
                        "name": "log_message",
                        "description": "Log conversation message",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "session_id": {"type": "string"},
                                "user_id": {"type": "string"},
                                "message_type": {"type": "string", "enum": ["user", "oviya", "system"]},
                                "content": {"type": "string"},
                                "emotion": {"type": "string"},
                                "personality_vector": {"type": "object"},
                                "prosody_data": {"type": "object"},
                                "crisis_flags": {"type": "object"}
                            },
                            "required": ["session_id", "user_id", "message_type", "content"]
                        }
                    },
                    {
                        "name": "get_user_analytics",
                        "description": "Get user analytics data",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "string"},
                                "days": {"type": "integer", "default": 30}
                            },
                            "required": ["user_id"]
                        }
                    },
                    {
                        "name": "get_system_analytics",
                        "description": "Get system-wide analytics",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "days": {"type": "integer", "default": 30}
                            }
                        }
                    },
                    {
                        "name": "export_user_data",
                        "description": "Export all user data for GDPR compliance",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "string"}
                            },
                            "required": ["user_id"]
                        }
                    }
                ]
            }

        return {"error": "Method not supported"}

    async def _get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get complete user profile"""
        async with self.connection_pool.acquire() as conn:
            row = await conn.fetchrow("""
                SELECT * FROM user_profiles WHERE user_id = $1
            """, user_id)

            if row:
                return {
                    "content": [{"type": "text", "text": json.dumps(dict(row))}]
                }
            else:
                return {
                    "content": [{"type": "text", "text": json.dumps({"error": "User not found"})}]
                }

    async def _update_user_profile(self, user_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        """Update user profile data"""
        async with self.connection_pool.acquire() as conn:
            # First check if user exists
            existing = await conn.fetchval("""
                SELECT user_id FROM user_profiles WHERE user_id = $1
            """, user_id)

            if existing:
                # Update existing user
                set_clause = ", ".join(f"{k} = ${i+2}" for i, k in enumerate(updates.keys()))
                values = list(updates.values())
                values.insert(0, user_id)

                await conn.execute(f"""
                    UPDATE user_profiles
                    SET {set_clause}, last_seen = CURRENT_TIMESTAMP
                    WHERE user_id = $1
                """, *values)
            else:
                # Create new user
                columns = ["user_id"] + list(updates.keys())
                placeholders = ", ".join(f"${i+1}" for i in range(len(columns)))
                values = [user_id] + list(updates.values())

                await conn.execute(f"""
                    INSERT INTO user_profiles ({", ".join(columns)})
                    VALUES ({placeholders})
                """, *values)

            return {
                "content": [{"type": "text", "text": json.dumps({"status": "updated", "user_id": user_id})}]
            }

    async def _create_session(self, user_id: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create new conversation session"""
        session_id = str(uuid.uuid4())

        async with self.connection_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO conversation_sessions (session_id, user_id, metadata)
                VALUES ($1, $2, $3)
            """, session_id, user_id, json.dumps(metadata))

            # Update user session count
            await conn.execute("""
                UPDATE user_profiles
                SET total_sessions = total_sessions + 1, last_seen = CURRENT_TIMESTAMP
                WHERE user_id = $1
            """, user_id)

        return {
            "content": [{"type": "text", "text": json.dumps({"session_id": session_id, "status": "created"})}]
        }

    async def _end_session(self, session_id: str) -> Dict[str, Any]:
        """End conversation session"""
        async with self.connection_pool.acquire() as conn:
            # Calculate duration and message count
            session_data = await conn.fetchrow("""
                SELECT started_at, user_id,
                       EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - started_at)) as duration,
                       (SELECT COUNT(*) FROM messages WHERE session_id = $1) as message_count
                FROM conversation_sessions WHERE session_id = $1
            """, session_id)

            if session_data:
                await conn.execute("""
                    UPDATE conversation_sessions
                    SET ended_at = CURRENT_TIMESTAMP,
                        duration_seconds = $2,
                        message_count = $3
                    WHERE session_id = $1
                """, session_id, session_data['duration'], session_data['message_count'])

                # Update user message count
                await conn.execute("""
                    UPDATE user_profiles
                    SET total_messages = total_messages + $2, last_seen = CURRENT_TIMESTAMP
                    WHERE user_id = $1
                """, session_data['user_id'], session_data['message_count'])

        return {
            "content": [{"type": "text", "text": json.dumps({"session_id": session_id, "status": "ended"})}]
        }

    async def _log_message(self, message_data: Dict[str, Any]) -> Dict[str, Any]:
        """Log conversation message"""
        message_id = str(uuid.uuid4())

        async with self.connection_pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO messages (
                    message_id, session_id, user_id, message_type, content,
                    emotion, personality_vector, prosody_data, crisis_flags, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
            """,
            message_id,
            message_data.get("session_id"),
            message_data.get("user_id"),
            message_data.get("message_type"),
            message_data.get("content"),
            message_data.get("emotion"),
            json.dumps(message_data.get("personality_vector", {})),
            json.dumps(message_data.get("prosody_data", {})),
            json.dumps(message_data.get("crisis_flags", {})),
            json.dumps(message_data.get("metadata", {}))
            )

        return {
            "content": [{"type": "text", "text": json.dumps({"message_id": message_id, "status": "logged"})}]
        }

    async def _get_user_analytics(self, user_id: str, days: int) -> Dict[str, Any]:
        """Get user analytics data"""
        async with self.connection_pool.acquire() as conn:
            # Session statistics
            sessions = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_sessions,
                    AVG(duration_seconds) as avg_session_duration,
                    SUM(message_count) as total_messages
                FROM conversation_sessions
                WHERE user_id = $1 AND started_at >= CURRENT_TIMESTAMP - INTERVAL '%s days'
            """, user_id, days)

            # Emotion distribution
            emotions = await conn.fetch("""
                SELECT emotion, COUNT(*) as count
                FROM messages
                WHERE user_id = $1 AND timestamp >= CURRENT_TIMESTAMP - INTERVAL '%s days'
                AND emotion IS NOT NULL
                GROUP BY emotion
                ORDER BY count DESC
            """, user_id, days)

            # Crisis events
            crisis_count = await conn.fetchval("""
                SELECT COUNT(*)
                FROM messages
                WHERE user_id = $1 AND timestamp >= CURRENT_TIMESTAMP - INTERVAL '%s days'
                AND crisis_flags != '{}'
            """, user_id, days)

            analytics = {
                "user_id": user_id,
                "period_days": days,
                "session_stats": dict(sessions) if sessions else {},
                "emotion_distribution": [dict(row) for row in emotions],
                "crisis_events_count": crisis_count or 0,
                "most_common_emotion": emotions[0]['emotion'] if emotions else None
            }

        return {
            "content": [{"type": "text", "text": json.dumps(analytics)}]
        }

    async def _get_system_analytics(self, days: int) -> Dict[str, Any]:
        """Get system-wide analytics"""
        async with self.connection_pool.acquire() as conn:
            # User statistics
            users = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_users,
                    COUNT(*) FILTER (WHERE last_seen >= CURRENT_TIMESTAMP - INTERVAL '7 days') as active_users_7d,
                    COUNT(*) FILTER (WHERE subscription_status = 'premium') as premium_users
                FROM user_profiles
            """)

            # Session statistics
            sessions = await conn.fetchrow("""
                SELECT
                    COUNT(*) as total_sessions,
                    AVG(duration_seconds) as avg_session_duration,
                    SUM(message_count) as total_messages
                FROM conversation_sessions
                WHERE started_at >= CURRENT_TIMESTAMP - INTERVAL '%s days'
            """, days)

            # Crisis events
            crisis_events = await conn.fetchval("""
                SELECT COUNT(*)
                FROM messages
                WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '%s days'
                AND crisis_flags != '{}'
            """, days)

            analytics = {
                "period_days": days,
                "user_stats": dict(users) if users else {},
                "session_stats": dict(sessions) if sessions else {},
                "system_health": {
                    "crisis_events_total": crisis_events or 0,
                    "messages_per_day": (sessions['total_messages'] or 0) / max(days, 1)
                }
            }

        return {
            "content": [{"type": "text", "text": json.dumps(analytics)}]
        }

    async def _export_user_data(self, user_id: str) -> Dict[str, Any]:
        """Export all user data for GDPR compliance"""
        async with self.connection_pool.acquire() as conn:
            # Get user profile
            profile = await conn.fetchrow("""
                SELECT * FROM user_profiles WHERE user_id = $1
            """, user_id)

            # Get all sessions
            sessions = await conn.fetch("""
                SELECT * FROM conversation_sessions
                WHERE user_id = $1
                ORDER BY started_at DESC
            """, user_id)

            # Get all messages
            messages = await conn.fetch("""
                SELECT * FROM messages
                WHERE user_id = $1
                ORDER BY timestamp DESC
            """, user_id)

            # Get analytics events
            events = await conn.fetch("""
                SELECT * FROM analytics_events
                WHERE user_id = $1
                ORDER BY timestamp DESC
            """, user_id)

            export_data = {
                "user_id": user_id,
                "export_timestamp": datetime.now().isoformat(),
                "profile": dict(profile) if profile else None,
                "sessions": [dict(session) for session in sessions],
                "messages": [dict(message) for message in messages],
                "analytics_events": [dict(event) for event in events],
                "gdpr_compliant": True
            }

        return {
            "content": [{"type": "text", "text": json.dumps(export_data)}]
        }

async def main():
    """Main MCP server loop"""
    server = OviyaPostgresServer()

    try:
        await server.initialize_database()

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
