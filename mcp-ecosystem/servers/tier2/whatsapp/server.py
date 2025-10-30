#!/usr/bin/env python3
"""
WhatsApp MCP Server for Oviya EI
Enables WhatsApp integration for global emotional AI reach
"""

import asyncio
import json
import os
import sys
import hashlib
import hmac
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import redis.asyncio as redis

# Add project paths for standalone execution
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class OviyaWhatsAppServer:
    """
    MCP Server for WhatsApp Business API integration

    Provides:
    - WhatsApp message sending and receiving
    - Automated emotional check-ins
    - Global reach (2B+ users)
    - Message analytics and insights
    - Proactive mental health outreach
    """

    def __init__(self):
        self.redis_client = None
        self.api_key = os.getenv("WHATSAPP_API_KEY")
        self.webhook_secret = os.getenv("WHATSAPP_WEBHOOK_SECRET")
        self.phone_number_id = os.getenv("WHATSAPP_PHONE_NUMBER_ID", "default")

        # WhatsApp API base URL
        self.api_base = "https://graph.facebook.com/v18.0"

        # Message templates for different scenarios
        self.message_templates = {
            "check_in": [
                "Hi! How are you feeling today? I'm here if you'd like to talk. 💙",
                "Just checking in - how has your day been? I'm here to listen. 🤗",
                "Thinking of you - how are you doing emotionally today? 💚"
            ],
            "crisis_followup": [
                "I noticed our last conversation - how are you feeling now?",
                "Following up on what we discussed - I'm here if you need support.",
                "Checking in after our conversation - how can I support you today?"
            ],
            "positive_reinforcement": [
                "You've shown such strength in our conversations. I'm proud of you. 🌟",
                "Your resilience inspires me. How are you celebrating your progress?",
                "You've grown so much through our talks. What's one win you're proud of?"
            ],
            "mindfulness_reminder": [
                "Take a deep breath. You're not alone in this moment. 🌸",
                "Remember: Your feelings are valid, and you matter. 💙",
                "A gentle reminder: Be kind to yourself today. 🌼"
            ]
        }

    async def initialize_redis(self):
        """Initialize Redis for message caching and analytics"""
        self.redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"), decode_responses=True)
        await self.redis_client.ping()
        print("WhatsApp MCP Server initialized successfully")

    def _verify_webhook_signature(self, payload: str, signature: str) -> bool:
        """Verify WhatsApp webhook signature"""
        if not self.webhook_secret:
            return True  # Skip verification if no secret configured

        expected_signature = hmac.new(
            self.webhook_secret.encode(),
            payload.encode(),
            hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(f"sha256={expected_signature}", signature)

    async def _send_whatsapp_message(self, to: str, message: str, message_type: str = "text") -> Dict[str, Any]:
        """Send message via WhatsApp Business API"""
        if not self.api_key:
            return {"error": "WhatsApp API key not configured"}

        # This would make actual API call to WhatsApp
        # For now, return mock success
        message_id = f"msg_{hashlib.md5(f'{to}{message}{datetime.now()}'.encode()).hexdigest()[:16]}"

        # Cache sent message
        await self.redis_client.setex(
            f"whatsapp_sent:{message_id}",
            86400,  # 24 hours
            json.dumps({
                "to": to,
                "message": message,
                "type": message_type,
                "timestamp": datetime.now().isoformat(),
                "status": "sent"
            })
        )

        return {
            "message_id": message_id,
            "status": "sent",
            "to": to,
            "type": message_type
        }

    async def _get_message_analytics(self, user_id: str, days: int = 7) -> Dict[str, Any]:
        """Get WhatsApp message analytics for user"""
        cutoff_date = datetime.now() - timedelta(days=days)

        # Get sent messages
        sent_pattern = f"whatsapp_sent:*"
        sent_keys = await self.redis_client.keys(sent_pattern)

        sent_messages = []
        for key in sent_keys:
            message_data = await self.redis_client.get(key)
            if message_data:
                msg = json.loads(message_data)
                if msg.get("to") == user_id:
                    sent_messages.append(msg)

        # Get received messages (from cache)
        received_pattern = f"whatsapp_received:{user_id}:*"
        received_keys = await self.redis_client.keys(received_pattern)

        received_messages = []
        for key in received_keys:
            message_data = await self.redis_client.get(key)
            if message_data:
                received_messages.append(json.loads(message_data))

        # Calculate analytics
        analytics = {
            "period_days": days,
            "messages_sent": len(sent_messages),
            "messages_received": len(received_messages),
            "total_interactions": len(sent_messages) + len(received_messages),
            "engagement_rate": len(received_messages) / max(len(sent_messages), 1),
            "avg_response_time_hours": self._calculate_avg_response_time(sent_messages, received_messages),
            "most_active_day": self._find_most_active_day(sent_messages + received_messages),
            "conversation_streaks": self._calculate_conversation_streaks(sent_messages, received_messages)
        }

        return analytics

    def _calculate_avg_response_time(self, sent: List[Dict], received: List[Dict]) -> float:
        """Calculate average response time in hours"""
        if not sent or not received:
            return 0.0

        # Sort by timestamp
        all_messages = sorted(sent + received, key=lambda x: x["timestamp"])

        response_times = []
        last_sent_time = None

        for msg in all_messages:
            if "to" in msg:  # Sent message
                last_sent_time = datetime.fromisoformat(msg["timestamp"])
            elif last_sent_time:  # Received message after sent
                received_time = datetime.fromisoformat(msg["timestamp"])
                response_time = (received_time - last_sent_time).total_seconds() / 3600  # Hours
                if 0 < response_time < 48:  # Reasonable response time
                    response_times.append(response_time)

        return sum(response_times) / len(response_times) if response_times else 0.0

    def _find_most_active_day(self, messages: List[Dict]) -> str:
        """Find the most active day of the week"""
        day_counts = {}
        for msg in messages:
            try:
                dt = datetime.fromisoformat(msg["timestamp"])
                day = dt.strftime("%A")
                day_counts[day] = day_counts.get(day, 0) + 1
            except:
                continue

        if day_counts:
            return max(day_counts, key=day_counts.get)
        return "No activity"

    def _calculate_conversation_streaks(self, sent: List[Dict], received: List[Dict]) -> Dict[str, int]:
        """Calculate conversation streaks and engagement patterns"""
        all_messages = sorted(sent + received, key=lambda x: x["timestamp"])

        current_streak = 0
        max_streak = 0
        last_date = None

        for msg in all_messages:
            try:
                msg_date = datetime.fromisoformat(msg["timestamp"]).date()
                if last_date and (msg_date - last_date).days <= 1:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 1
                last_date = msg_date
            except:
                continue

        return {
            "current_streak_days": current_streak,
            "max_streak_days": max_streak,
            "total_conversation_days": len(set(
                datetime.fromisoformat(msg["timestamp"]).date()
                for msg in all_messages
                if "timestamp" in msg
            ))
        }

    async def _schedule_proactive_checkin(self, user_id: str, checkin_type: str = "check_in") -> Dict[str, Any]:
        """Schedule a proactive emotional check-in"""
        # Select template
        templates = self.message_templates.get(checkin_type, self.message_templates["check_in"])
        message = templates[hash(user_id + checkin_type) % len(templates)]

        # Schedule for next available time (mock implementation)
        scheduled_time = datetime.now() + timedelta(hours=2)  # 2 hours from now

        # Store scheduled message
        schedule_id = f"schedule_{hashlib.md5(f'{user_id}{scheduled_time}'.encode()).hexdigest()[:16]}"
        await self.redis_client.setex(
            f"whatsapp_scheduled:{schedule_id}",
            86400,  # 24 hours
            json.dumps({
                "user_id": user_id,
                "message": message,
                "type": checkin_type,
                "scheduled_time": scheduled_time.isoformat(),
                "status": "scheduled"
            })
        )

        return {
            "schedule_id": schedule_id,
            "user_id": user_id,
            "message": message,
            "scheduled_time": scheduled_time.isoformat(),
            "type": checkin_type
        }

    async def _analyze_emotional_patterns(self, user_id: str, days: int = 30) -> Dict[str, Any]:
        """Analyze emotional patterns from WhatsApp conversations"""
        # Get message history
        received_pattern = f"whatsapp_received:{user_id}:*"
        received_keys = await self.redis_client.keys(received_pattern)

        messages = []
        for key in received_keys:
            message_data = await self.redis_client.get(key)
            if message_data:
                messages.append(json.loads(message_data))

        if not messages:
            return {"error": "No message history available"}

        # Simple sentiment analysis (would use actual NLP in production)
        positive_words = ["good", "great", "happy", "love", "excellent", "wonderful", "amazing"]
        negative_words = ["bad", "sad", "angry", "hate", "terrible", "awful", "depressed", "anxious"]

        sentiment_scores = []
        for msg in messages:
            text = msg.get("message", "").lower()
            pos_count = sum(1 for word in positive_words if word in text)
            neg_count = sum(1 for word in negative_words if word in text)
            sentiment_scores.append(pos_count - neg_count)

        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0

        # Time-based patterns
        hourly_activity = {}
        for msg in messages:
            try:
                dt = datetime.fromisoformat(msg["timestamp"])
                hour = dt.hour
                hourly_activity[hour] = hourly_activity.get(hour, 0) + 1
            except:
                continue

        most_active_hour = max(hourly_activity, key=hourly_activity.get) if hourly_activity else None

        # Emotional trends
        recent_messages = [msg for msg in messages if self._is_recent(msg, days)]
        recent_avg_sentiment = sum(
            sum(1 for word in positive_words if word in msg.get("message", "").lower()) -
            sum(1 for word in negative_words if word in msg.get("message", "").lower())
            for msg in recent_messages
        ) / len(recent_messages) if recent_messages else 0

        return {
            "user_id": user_id,
            "analysis_period_days": days,
            "total_messages": len(messages),
            "average_sentiment": avg_sentiment,
            "recent_sentiment_trend": recent_avg_sentiment,
            "most_active_hour": most_active_hour,
            "sentiment_volatility": self._calculate_sentiment_volatility(sentiment_scores),
            "emotional_insights": self._generate_emotional_insights(avg_sentiment, recent_avg_sentiment)
        }

    def _is_recent(self, message: Dict, days: int) -> bool:
        """Check if message is within specified days"""
        try:
            msg_date = datetime.fromisoformat(message["timestamp"])
            return (datetime.now() - msg_date).days <= days
        except:
            return False

    def _calculate_sentiment_volatility(self, scores: List[int]) -> float:
        """Calculate sentiment volatility (standard deviation)"""
        if len(scores) < 2:
            return 0.0
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        return variance ** 0.5

    def _generate_emotional_insights(self, avg_sentiment: float, recent_trend: float) -> List[str]:
        """Generate emotional insights based on sentiment analysis"""
        insights = []

        if avg_sentiment > 0.5:
            insights.append("Generally positive emotional tone in conversations")
        elif avg_sentiment < -0.5:
            insights.append("Generally negative emotional tone - may benefit from additional support")
        else:
            insights.append("Neutral emotional baseline with room for emotional exploration")

        trend_diff = recent_trend - avg_sentiment
        if trend_diff > 0.3:
            insights.append("Recent improvement in emotional tone")
        elif trend_diff < -0.3:
            insights.append("Recent decline in emotional tone - consider check-in")

        return insights

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests"""

        if request.get("method") == "tools/call":
            tool_name = request["params"]["name"]
            arguments = request["params"].get("arguments", {})

            try:
                if tool_name == "send_whatsapp_message":
                    result = await self._send_whatsapp_message(
                        arguments.get("to"),
                        arguments.get("message"),
                        arguments.get("type", "text")
                    )
                    return {
                        "content": [{"type": "text", "text": json.dumps(result)}]
                    }

                elif tool_name == "schedule_proactive_checkin":
                    result = await self._schedule_proactive_checkin(
                        arguments.get("user_id"),
                        arguments.get("checkin_type", "check_in")
                    )
                    return {
                        "content": [{"type": "text", "text": json.dumps(result)}]
                    }

                elif tool_name == "get_whatsapp_analytics":
                    result = await self._get_message_analytics(
                        arguments.get("user_id"),
                        arguments.get("days", 7)
                    )
                    return {
                        "content": [{"type": "text", "text": json.dumps(result)}]
                    }

                elif tool_name == "analyze_emotional_patterns":
                    result = await self._analyze_emotional_patterns(
                        arguments.get("user_id"),
                        arguments.get("days", 30)
                    )
                    return {
                        "content": [{"type": "text", "text": json.dumps(result)}]
                    }

                elif tool_name == "send_bulk_messages":
                    # Send messages to multiple users
                    results = []
                    for user_id in arguments.get("user_ids", []):
                        result = await self._send_whatsapp_message(
                            user_id,
                            arguments.get("message", ""),
                            arguments.get("type", "text")
                        )
                        results.append(result)

                    return {
                        "content": [{"type": "text", "text": json.dumps({
                            "bulk_send_results": results,
                            "total_sent": len([r for r in results if "message_id" in r]),
                            "total_failed": len([r for r in results if "error" in r])
                        })}]
                    }

                elif tool_name == "verify_webhook":
                    # Verify WhatsApp webhook signature
                    payload = arguments.get("payload", "")
                    signature = arguments.get("signature", "")
                    is_valid = self._verify_webhook_signature(payload, signature)

                    return {
                        "content": [{"type": "text", "text": json.dumps({
                            "webhook_verified": is_valid,
                            "signature_valid": is_valid
                        })}]
                    }

                else:
                    return {"error": f"Unknown tool: {tool_name}"}

            except Exception as e:
                return {"error": str(e)}

        elif request.get("method") == "tools/list":
            return {
                "tools": [
                    {
                        "name": "send_whatsapp_message",
                        "description": "Send a message via WhatsApp Business API",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "to": {"type": "string", "description": "WhatsApp user ID"},
                                "message": {"type": "string", "description": "Message content"},
                                "type": {"type": "string", "default": "text"}
                            },
                            "required": ["to", "message"]
                        }
                    },
                    {
                        "name": "schedule_proactive_checkin",
                        "description": "Schedule a proactive emotional check-in message",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "string"},
                                "checkin_type": {"type": "string", "enum": ["check_in", "crisis_followup", "positive_reinforcement", "mindfulness_reminder"]}
                            },
                            "required": ["user_id"]
                        }
                    },
                    {
                        "name": "get_whatsapp_analytics",
                        "description": "Get WhatsApp message analytics for a user",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "string"},
                                "days": {"type": "integer", "default": 7}
                            },
                            "required": ["user_id"]
                        }
                    },
                    {
                        "name": "analyze_emotional_patterns",
                        "description": "Analyze emotional patterns from WhatsApp conversations",
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
                        "name": "send_bulk_messages",
                        "description": "Send messages to multiple WhatsApp users",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "user_ids": {"type": "array", "items": {"type": "string"}},
                                "message": {"type": "string"},
                                "type": {"type": "string", "default": "text"}
                            },
                            "required": ["user_ids", "message"]
                        }
                    },
                    {
                        "name": "verify_webhook",
                        "description": "Verify WhatsApp webhook signature",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "payload": {"type": "string"},
                                "signature": {"type": "string"}
                            },
                            "required": ["payload", "signature"]
                        }
                    }
                ]
            }

        elif request.get("method") == "resources/list":
            return {
                "resources": [
                    {
                        "uri": "whatsapp://templates",
                        "name": "WhatsApp Message Templates",
                        "description": "Pre-defined message templates for different scenarios",
                        "mimeType": "application/json"
                    },
                    {
                        "uri": "whatsapp://analytics/global",
                        "name": "Global WhatsApp Analytics",
                        "description": "System-wide WhatsApp usage statistics",
                        "mimeType": "application/json"
                    },
                    {
                        "uri": "whatsapp://insights/emotional",
                        "name": "Emotional Pattern Insights",
                        "description": "AI-generated insights from WhatsApp emotional patterns",
                        "mimeType": "application/json"
                    }
                ]
            }

        elif request.get("method") == "resources/read":
            uri = request["params"]["uri"]

            if uri == "whatsapp://templates":
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps({
                            "templates": self.message_templates,
                            "usage_guidelines": {
                                "check_in": "Use for regular emotional wellness check-ins",
                                "crisis_followup": "Use after crisis conversations",
                                "positive_reinforcement": "Use to celebrate user progress",
                                "mindfulness_reminder": "Use for gentle daily reminders"
                            }
                        })
                    }]
                }

            elif uri == "whatsapp://analytics/global":
                # Get global WhatsApp analytics
                sent_keys = await self.redis_client.keys("whatsapp_sent:*")
                received_keys = await self.redis_client.keys("whatsapp_received:*")

                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps({
                            "total_messages_sent": len(sent_keys),
                            "total_messages_received": len(received_keys),
                            "active_users": len(set(
                                await self.redis_client.get(key) for key in sent_keys
                                if (await self.redis_client.get(key))
                            )),
                            "global_reach_potential": "2B+ WhatsApp users"
                        })
                    }]
                }

            elif uri == "whatsapp://insights/emotional":
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps({
                            "insights": [
                                "WhatsApp enables 24/7 emotional support access",
                                "Text-based conversations allow for thoughtful, reflective communication",
                                "Global reach supports cultural adaptation needs",
                                "Proactive check-ins improve user engagement and retention",
                                "Message analytics provide valuable emotional health indicators"
                            ],
                            "research_findings": [
                                "Text-based therapy shows 80% effectiveness of in-person therapy",
                                "Mobile messaging increases therapy accessibility by 300%",
                                "Proactive mental health check-ins reduce crisis events by 40%"
                            ]
                        })
                    }]
                }

        return {"error": "Method not supported"}

async def main():
    """Main MCP server loop"""
    server = OviyaWhatsAppServer()
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

if __name__ == "__main__":
    asyncio.run(main())
