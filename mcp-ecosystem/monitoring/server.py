#!/usr/bin/env python3
"""
Monitoring & Analytics MCP Server for Oviya EI
Provides comprehensive monitoring, analytics, and insights across the entire ecosystem
"""

import asyncio
import json
import os
import sys
import psutil
import time
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque
import asyncpg
import redis.asyncio as redis
import logging

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

class OviyaMonitoringServer:
    """
    MCP Server for comprehensive monitoring and analytics

    Provides:
    - System health monitoring
    - User engagement analytics
    - Emotional health trends
    - Performance metrics
    - Crisis intervention tracking
    - Business intelligence insights
    """

    def __init__(self):
        # Database connections
        self.db_pool = None
        self.redis_client = None
        self.database_url = os.getenv("DATABASE_URL", "postgresql://oviya:oviya_password@localhost:5432/oviya_db")
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")

        # Real-time metrics storage
        self.metrics_buffer = deque(maxlen=1000)
        self.system_metrics = {}
        self.user_engagement = defaultdict(int)
        self.emotional_trends = defaultdict(list)

        # Alert thresholds
        self.alert_thresholds = {
            "response_time_ms": 5000,  # 5 seconds
            "error_rate_percent": 5.0,
            "crisis_events_per_hour": 10,
            "memory_usage_percent": 85.0,
            "cpu_usage_percent": 80.0
        }

        # Logging
        self.logger = logging.getLogger("oviya_monitoring")
        self.logger.setLevel(logging.INFO)

    async def initialize_connections(self):
        """Initialize database and Redis connections"""
        try:
            # Database connection
            self.db_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )

            # Redis connection
            self.redis_client = redis.from_url(self.redis_url, decode_responses=True)

            # Create monitoring tables
            await self._create_monitoring_tables()

            print("Monitoring MCP Server initialized successfully")

        except Exception as e:
            print(f"Failed to initialize monitoring server: {e}")
            raise

    async def _create_monitoring_tables(self):
        """Create monitoring and analytics tables"""

        # System metrics
        await self.db_pool.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                metric_id SERIAL PRIMARY KEY,
                metric_type VARCHAR(100) NOT NULL,
                metric_name VARCHAR(100) NOT NULL,
                metric_value NUMERIC,
                unit VARCHAR(50),
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                metadata JSONB DEFAULT '{}'
            )
        """)

        # User engagement metrics
        await self.db_pool.execute("""
            CREATE TABLE IF NOT EXISTS user_engagement (
                engagement_id SERIAL PRIMARY KEY,
                user_id VARCHAR(255),
                session_id VARCHAR(255),
                event_type VARCHAR(100) NOT NULL,
                event_data JSONB DEFAULT '{}',
                duration_seconds INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Emotional health trends
        await self.db_pool.execute("""
            CREATE TABLE IF NOT EXISTS emotional_trends (
                trend_id SERIAL PRIMARY KEY,
                user_id VARCHAR(255),
                emotion VARCHAR(50) NOT NULL,
                intensity NUMERIC(3,2),
                context VARCHAR(100),
                crisis_flags JSONB DEFAULT '[]',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Performance metrics
        await self.db_pool.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                perf_id SERIAL PRIMARY KEY,
                component VARCHAR(100) NOT NULL,
                operation VARCHAR(100) NOT NULL,
                duration_ms INTEGER,
                success BOOLEAN DEFAULT TRUE,
                error_message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Alert logs
        await self.db_pool.execute("""
            CREATE TABLE IF NOT EXISTS alert_logs (
                alert_id SERIAL PRIMARY KEY,
                alert_type VARCHAR(100) NOT NULL,
                severity VARCHAR(20) DEFAULT 'info',
                message TEXT NOT NULL,
                component VARCHAR(100),
                metric_value NUMERIC,
                threshold_value NUMERIC,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                resolved BOOLEAN DEFAULT FALSE,
                resolved_at TIMESTAMP
            )
        """)

        # Create indexes
        await self.db_pool.execute("""
            CREATE INDEX IF NOT EXISTS idx_system_metrics_timestamp ON system_metrics(timestamp);
            CREATE INDEX IF NOT EXISTS idx_user_engagement_user_timestamp ON user_engagement(user_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_emotional_trends_user_timestamp ON emotional_trends(user_id, timestamp);
            CREATE INDEX IF NOT EXISTS idx_performance_component_timestamp ON performance_metrics(component, timestamp);
            CREATE INDEX IF NOT EXISTS idx_alerts_timestamp ON alert_logs(timestamp);
        """)

    async def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect current system metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_gb = memory.used / (1024**3)

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent

            # Network I/O
            net_io = psutil.net_io_counters()
            bytes_sent_mb = net_io.bytes_sent / (1024**2)
            bytes_recv_mb = net_io.bytes_recv / (1024**2)

            metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "memory_used_gb": memory_used_gb,
                "disk_percent": disk_percent,
                "network_sent_mb": bytes_sent_mb,
                "network_recv_mb": bytes_recv_mb,
                "timestamp": datetime.now().isoformat()
            }

            # Store in database
            for metric_name, value in metrics.items():
                if metric_name != "timestamp":
                    await self.db_pool.execute("""
                        INSERT INTO system_metrics (metric_type, metric_name, metric_value, unit, metadata)
                        VALUES ($1, $2, $3, $4, $5)
                    """,
                    "system",
                    metric_name,
                    value,
                    "percent" if "percent" in metric_name else "gb" if "gb" in metric_name else "mb",
                    json.dumps({"collected_at": metrics["timestamp"]})
                    )

            # Check for alerts
            await self._check_alerts(metrics)

            return metrics

        except Exception as e:
            return {"error": str(e)}

    async def _check_alerts(self, metrics: Dict[str, Any]):
        """Check metrics against alert thresholds"""
        alerts = []

        # CPU alert
        if metrics.get("cpu_percent", 0) > self.alert_thresholds["cpu_usage_percent"]:
            alerts.append({
                "type": "high_cpu_usage",
                "severity": "warning",
                "message": f"CPU usage is {metrics['cpu_percent']:.1f}%, above threshold of {self.alert_thresholds['cpu_usage_percent']}%",
                "metric_value": metrics["cpu_percent"],
                "threshold_value": self.alert_thresholds["cpu_usage_percent"]
            })

        # Memory alert
        if metrics.get("memory_percent", 0) > self.alert_thresholds["memory_usage_percent"]:
            alerts.append({
                "type": "high_memory_usage",
                "severity": "warning",
                "message": f"Memory usage is {metrics['memory_percent']:.1f}%, above threshold of {self.alert_thresholds['memory_usage_percent']}%",
                "metric_value": metrics["memory_percent"],
                "threshold_value": self.alert_thresholds["memory_usage_percent"]
            })

        # Store alerts
        for alert in alerts:
            await self.db_pool.execute("""
                INSERT INTO alert_logs (alert_type, severity, message, component, metric_value, threshold_value)
                VALUES ($1, $2, $3, $4, $5, $6)
            """,
            alert["type"],
            alert["severity"],
            alert["message"],
            "system_monitoring",
            alert["metric_value"],
            alert["threshold_value"]
            )

    async def _get_user_engagement_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get comprehensive user engagement analytics"""
        try:
            # User activity metrics
            user_activity = await self.db_pool.fetchrow("""
                SELECT
                    COUNT(DISTINCT user_id) as active_users,
                    COUNT(*) as total_sessions,
                    AVG(duration_seconds) as avg_session_duration,
                    SUM(duration_seconds) as total_engagement_time
                FROM user_engagement
                WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '%s days'
                AND event_type = 'session_end'
            """, days)

            # Session distribution
            session_distribution = await self.db_pool.fetch("""
                SELECT
                    CASE
                        WHEN duration_seconds < 300 THEN '0-5min'
                        WHEN duration_seconds < 900 THEN '5-15min'
                        WHEN duration_seconds < 1800 THEN '15-30min'
                        ELSE '30min+'
                    END as duration_bucket,
                    COUNT(*) as session_count
                FROM user_engagement
                WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '%s days'
                AND event_type = 'session_end'
                GROUP BY duration_bucket
                ORDER BY duration_bucket
            """, days)

            # Daily active users trend
            daily_active = await self.db_pool.fetch("""
                SELECT
                    DATE(timestamp) as date,
                    COUNT(DISTINCT user_id) as daily_active_users
                FROM user_engagement
                WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '%s days'
                GROUP BY DATE(timestamp)
                ORDER BY date
            """, days)

            # Most common engagement events
            top_events = await self.db_pool.fetch("""
                SELECT
                    event_type,
                    COUNT(*) as event_count
                FROM user_engagement
                WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '%s days'
                GROUP BY event_type
                ORDER BY event_count DESC
                LIMIT 10
            """, days)

            return {
                "period_days": days,
                "user_activity": dict(user_activity) if user_activity else {},
                "session_distribution": [dict(row) for row in session_distribution],
                "daily_active_users_trend": [dict(row) for row in daily_active],
                "top_engagement_events": [dict(row) for row in top_events],
                "engagement_insights": self._generate_engagement_insights(
                    dict(user_activity) if user_activity else {},
                    [dict(row) for row in session_distribution]
                )
            }

        except Exception as e:
            return {"error": str(e)}

    async def _get_emotional_health_analytics(self, days: int = 30) -> Dict[str, Any]:
        """Get emotional health trends and analytics"""
        try:
            # Emotion distribution
            emotion_distribution = await self.db_pool.fetch("""
                SELECT
                    emotion,
                    COUNT(*) as occurrence_count,
                    AVG(intensity) as avg_intensity,
                    MIN(intensity) as min_intensity,
                    MAX(intensity) as max_intensity
                FROM emotional_trends
                WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '%s days'
                GROUP BY emotion
                ORDER BY occurrence_count DESC
            """, days)

            # Crisis event tracking
            crisis_events = await self.db_pool.fetchrow("""
                SELECT
                    COUNT(*) as total_crisis_events,
                    COUNT(DISTINCT user_id) as users_with_crisis,
                    AVG(timestamp - LAG(timestamp) OVER (ORDER BY timestamp)) as avg_time_between_crises
                FROM emotional_trends
                WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '%s days'
                AND crisis_flags != '[]'
            """, days)

            # Emotional trend over time
            emotion_trends = await self.db_pool.fetch("""
                SELECT
                    DATE(timestamp) as date,
                    emotion,
                    AVG(intensity) as avg_intensity,
                    COUNT(*) as daily_count
                FROM emotional_trends
                WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '%s days'
                GROUP BY DATE(timestamp), emotion
                ORDER BY date, avg_intensity DESC
            """, days)

            # User emotional stability
            user_stability = await self.db_pool.fetch("""
                SELECT
                    user_id,
                    COUNT(DISTINCT emotion) as unique_emotions,
                    AVG(intensity) as avg_emotional_intensity,
                    STDDEV(intensity) as emotional_volatility
                FROM emotional_trends
                WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '%s days'
                GROUP BY user_id
                HAVING COUNT(*) >= 5
                ORDER BY emotional_volatility DESC
            """, days)

            return {
                "period_days": days,
                "emotion_distribution": [dict(row) for row in emotion_distribution],
                "crisis_events": dict(crisis_events) if crisis_events else {},
                "emotion_trends_over_time": [dict(row) for row in emotion_trends],
                "user_emotional_stability": [dict(row) for row in user_stability],
                "emotional_health_insights": self._generate_emotional_insights(
                    [dict(row) for row in emotion_distribution],
                    dict(crisis_events) if crisis_events else {}
                )
            }

        except Exception as e:
            return {"error": str(e)}

    async def _get_performance_analytics(self, hours: int = 24) -> Dict[str, Any]:
        """Get system performance analytics"""
        try:
            # Component performance
            component_performance = await self.db_pool.fetch("""
                SELECT
                    component,
                    operation,
                    COUNT(*) as total_operations,
                    AVG(duration_ms) as avg_duration_ms,
                    MIN(duration_ms) as min_duration_ms,
                    MAX(duration_ms) as max_duration_ms,
                    COUNT(*) FILTER (WHERE success = false) as failed_operations,
                    COUNT(*) FILTER (WHERE success = true) as successful_operations
                FROM performance_metrics
                WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '%s hours'
                GROUP BY component, operation
                ORDER BY avg_duration_ms DESC
            """, hours)

            # Error rate trends
            error_trends = await self.db_pool.fetch("""
                SELECT
                    DATE_TRUNC('hour', timestamp) as hour,
                    COUNT(*) FILTER (WHERE success = false) * 100.0 / COUNT(*) as error_rate_percent,
                    COUNT(*) as total_operations
                FROM performance_metrics
                WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '%s hours'
                GROUP BY DATE_TRUNC('hour', timestamp)
                ORDER BY hour
            """, hours)

            # Slowest operations
            slow_operations = await self.db_pool.fetch("""
                SELECT
                    component,
                    operation,
                    duration_ms,
                    timestamp,
                    error_message
                FROM performance_metrics
                WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '%s hours'
                AND duration_ms > 1000
                ORDER BY duration_ms DESC
                LIMIT 20
            """, hours)

            # System uptime and reliability
            uptime_stats = await self.db_pool.fetchrow("""
                SELECT
                    COUNT(*) as total_operations,
                    COUNT(*) FILTER (WHERE success = true) as successful_operations,
                    AVG(duration_ms) as avg_response_time,
                    MIN(timestamp) as monitoring_start,
                    MAX(timestamp) as monitoring_end
                FROM performance_metrics
                WHERE timestamp >= CURRENT_TIMESTAMP - INTERVAL '%s hours'
            """, hours)

            return {
                "period_hours": hours,
                "component_performance": [dict(row) for row in component_performance],
                "error_rate_trends": [dict(row) for row in error_trends],
                "slowest_operations": [dict(row) for row in slow_operations],
                "system_uptime_stats": dict(uptime_stats) if uptime_stats else {},
                "performance_health_score": self._calculate_performance_health(
                    [dict(row) for row in component_performance],
                    dict(uptime_stats) if uptime_stats else {}
                )
            }

        except Exception as e:
            return {"error": str(e)}

    def _generate_engagement_insights(self, user_activity: Dict, session_distribution: List) -> List[str]:
        """Generate insights from engagement data"""
        insights = []

        active_users = user_activity.get("active_users", 0)
        avg_duration = user_activity.get("avg_session_duration", 0)

        if active_users > 100:
            insights.append("Strong user engagement with high active user count")
        elif active_users < 10:
            insights.append("Low user engagement - consider growth strategies")

        if avg_duration > 900:  # 15 minutes
            insights.append("Users spending significant time in sessions - high engagement")
        elif avg_duration < 300:  # 5 minutes
            insights.append("Short session durations - may indicate UX issues or quick tasks")

        # Session distribution insights
        short_sessions = sum(row["session_count"] for row in session_distribution
                           if row["duration_bucket"] in ["0-5min"])
        long_sessions = sum(row["session_count"] for row in session_distribution
                          if row["duration_bucket"] == "30min+")

        if short_sessions > long_sessions * 2:
            insights.append("Many short sessions - users may need quick emotional support")
        if long_sessions > short_sessions:
            insights.append("Many long sessions - deep emotional work happening")

        return insights

    def _generate_emotional_insights(self, emotion_dist: List, crisis_data: Dict) -> List[str]:
        """Generate insights from emotional health data"""
        insights = []

        if emotion_dist:
            top_emotion = emotion_dist[0]
            if top_emotion["emotion"] == "sad":
                insights.append("Sadness is the most common emotion - focus on grief support")
            elif top_emotion["emotion"] == "anxious":
                insights.append("Anxiety is prevalent - anxiety management tools needed")

            # Check for emotional diversity
            unique_emotions = len(emotion_dist)
            if unique_emotions < 3:
                insights.append("Limited emotional range detected - encourage emotional exploration")

        # Crisis insights
        crisis_count = crisis_data.get("total_crisis_events", 0)
        if crisis_count > 10:
            insights.append("High crisis event frequency - ensure adequate support resources")
        elif crisis_count == 0:
            insights.append("No crisis events detected - strong preventive support")

        return insights

    def _calculate_performance_health(self, component_perf: List, uptime_stats: Dict) -> Dict[str, Any]:
        """Calculate overall system performance health score"""
        if not component_perf or not uptime_stats:
            return {"health_score": 0, "status": "unknown"}

        # Calculate error rate
        total_ops = uptime_stats.get("total_operations", 0)
        successful_ops = uptime_stats.get("successful_operations", 0)
        error_rate = ((total_ops - successful_ops) / total_ops) * 100 if total_ops > 0 else 0

        # Calculate average response time
        avg_response_time = uptime_stats.get("avg_response_time", 0)

        # Component health
        slow_components = [comp for comp in component_perf if comp["avg_duration_ms"] > 1000]

        # Overall health score (0-100)
        health_score = 100

        # Deduct for high error rate
        if error_rate > 5:
            health_score -= min(30, error_rate * 2)
        elif error_rate > 1:
            health_score -= min(10, error_rate)

        # Deduct for slow response times
        if avg_response_time > 2000:
            health_score -= 20
        elif avg_response_time > 1000:
            health_score -= 10

        # Deduct for slow components
        health_score -= min(20, len(slow_components) * 5)

        # Determine status
        if health_score >= 90:
            status = "excellent"
        elif health_score >= 75:
            status = "good"
        elif health_score >= 60:
            status = "fair"
        elif health_score >= 40:
            status = "concerning"
        else:
            status = "critical"

        return {
            "health_score": max(0, health_score),
            "status": status,
            "error_rate_percent": error_rate,
            "avg_response_time_ms": avg_response_time,
            "slow_components_count": len(slow_components),
            "recommendations": self._generate_health_recommendations(health_score, error_rate, slow_components)
        }

    def _generate_health_recommendations(self, health_score: float, error_rate: float,
                                       slow_components: List) -> List[str]:
        """Generate health improvement recommendations"""
        recommendations = []

        if health_score < 60:
            recommendations.append("Critical performance issues detected - immediate attention required")

        if error_rate > 5:
            recommendations.append("High error rate - investigate error sources and implement fixes")

        if slow_components:
            slow_names = [comp["component"] for comp in slow_components[:3]]
            recommendations.append(f"Optimize slow components: {', '.join(slow_names)}")

        if health_score >= 90:
            recommendations.append("System performing excellently - maintain current standards")

        return recommendations

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests"""

        if request.get("method") == "tools/call":
            tool_name = request["params"]["name"]
            arguments = request["params"].get("arguments", {})

            try:
                if tool_name == "collect_system_metrics":
                    result = await self._collect_system_metrics()
                    return {
                        "content": [{"type": "text", "text": json.dumps(result)}]
                    }

                elif tool_name == "get_user_engagement_analytics":
                    result = await self._get_user_engagement_analytics(
                        arguments.get("days", 30)
                    )
                    return {
                        "content": [{"type": "text", "text": json.dumps(result)}]
                    }

                elif tool_name == "get_emotional_health_analytics":
                    result = await self._get_emotional_health_analytics(
                        arguments.get("days", 30)
                    )
                    return {
                        "content": [{"type": "text", "text": json.dumps(result)}]
                    }

                elif tool_name == "get_performance_analytics":
                    result = await self._get_performance_analytics(
                        arguments.get("hours", 24)
                    )
                    return {
                        "content": [{"type": "text", "text": json.dumps(result)}]
                    }

                elif tool_name == "log_performance_metric":
                    result = await self._log_performance_metric(
                        arguments.get("component"),
                        arguments.get("operation"),
                        arguments.get("duration_ms"),
                        arguments.get("success", True),
                        arguments.get("error_message")
                    )
                    return {
                        "content": [{"type": "text", "text": json.dumps(result)}]
                    }

                elif tool_name == "log_user_engagement":
                    result = await self._log_user_engagement(
                        arguments.get("user_id"),
                        arguments.get("session_id"),
                        arguments.get("event_type"),
                        arguments.get("event_data", {}),
                        arguments.get("duration_seconds")
                    )
                    return {
                        "content": [{"type": "text", "text": json.dumps(result)}]
                    }

                elif tool_name == "log_emotional_trend":
                    result = await self._log_emotional_trend(
                        arguments.get("user_id"),
                        arguments.get("emotion"),
                        arguments.get("intensity"),
                        arguments.get("context"),
                        arguments.get("crisis_flags", [])
                    )
                    return {
                        "content": [{"type": "text", "text": json.dumps(result)}]
                    }

                elif tool_name == "get_system_health_dashboard":
                    result = await self._get_system_health_dashboard()
                    return {
                        "content": [{"type": "text", "text": json.dumps(result)}]
                    }

                else:
                    return {"error": f"Unknown tool: {tool_name}"}

            except Exception as e:
                return {"error": str(e)}

        elif request.get("method") == "tools/list":
            return {
                "tools": [
                    {
                        "name": "collect_system_metrics",
                        "description": "Collect current system performance metrics",
                        "inputSchema": {"type": "object", "properties": {}}
                    },
                    {
                        "name": "get_user_engagement_analytics",
                        "description": "Get comprehensive user engagement analytics",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "days": {"type": "integer", "default": 30}
                            }
                        }
                    },
                    {
                        "name": "get_emotional_health_analytics",
                        "description": "Get emotional health trends and crisis analytics",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "days": {"type": "integer", "default": 30}
                            }
                        }
                    },
                    {
                        "name": "get_performance_analytics",
                        "description": "Get system performance and reliability metrics",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "hours": {"type": "integer", "default": 24}
                            }
                        }
                    },
                    {
                        "name": "log_performance_metric",
                        "description": "Log a performance metric for monitoring",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "component": {"type": "string"},
                                "operation": {"type": "string"},
                                "duration_ms": {"type": "integer"},
                                "success": {"type": "boolean", "default": True},
                                "error_message": {"type": "string"}
                            },
                            "required": ["component", "operation", "duration_ms"]
                        }
                    },
                    {
                        "name": "log_user_engagement",
                        "description": "Log user engagement event",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "string"},
                                "session_id": {"type": "string"},
                                "event_type": {"type": "string"},
                                "event_data": {"type": "object"},
                                "duration_seconds": {"type": "integer"}
                            },
                            "required": ["event_type"]
                        }
                    },
                    {
                        "name": "log_emotional_trend",
                        "description": "Log emotional health data point",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "string"},
                                "emotion": {"type": "string"},
                                "intensity": {"type": "number"},
                                "context": {"type": "string"},
                                "crisis_flags": {"type": "array"}
                            },
                            "required": ["emotion", "intensity"]
                        }
                    },
                    {
                        "name": "get_system_health_dashboard",
                        "description": "Get comprehensive system health dashboard",
                        "inputSchema": {"type": "object", "properties": {}}
                    }
                ]
            }

        elif request.get("method") == "resources/list":
            return {
                "resources": [
                    {
                        "uri": "monitoring://system/health",
                        "name": "System Health Dashboard",
                        "description": "Real-time system health and performance metrics",
                        "mimeType": "application/json"
                    },
                    {
                        "uri": "monitoring://analytics/user-engagement",
                        "name": "User Engagement Analytics",
                        "description": "Comprehensive user engagement metrics and trends",
                        "mimeType": "application/json"
                    },
                    {
                        "uri": "monitoring://analytics/emotional-health",
                        "name": "Emotional Health Analytics",
                        "description": "Emotional health trends and crisis intervention metrics",
                        "mimeType": "application/json"
                    },
                    {
                        "uri": "monitoring://alerts/active",
                        "name": "Active System Alerts",
                        "description": "Current system alerts and warnings",
                        "mimeType": "application/json"
                    }
                ]
            }

        elif request.get("method") == "resources/read":
            uri = request["params"]["uri"]

            if uri == "monitoring://system/health":
                dashboard = await self._get_system_health_dashboard()
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(dashboard)
                    }]
                }

            elif uri == "monitoring://analytics/user-engagement":
                analytics = await self._get_user_engagement_analytics()
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(analytics)
                    }]
                }

            elif uri == "monitoring://analytics/emotional-health":
                analytics = await self._get_emotional_health_analytics()
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(analytics)
                    }]
                }

            elif uri == "monitoring://alerts/active":
                alerts = await self._get_active_alerts()
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(alerts)
                    }]
                }

        return {"error": "Method not supported"}

    async def _log_performance_metric(self, component: str, operation: str, duration_ms: int,
                                    success: bool = True, error_message: str = None) -> Dict[str, Any]:
        """Log a performance metric"""
        await self.db_pool.execute("""
            INSERT INTO performance_metrics (component, operation, duration_ms, success, error_message)
            VALUES ($1, $2, $3, $4, $5)
        """, component, operation, duration_ms, success, error_message)

        return {"status": "logged", "component": component, "operation": operation}

    async def _log_user_engagement(self, user_id: str, session_id: str, event_type: str,
                                 event_data: Dict[str, Any], duration_seconds: int = None) -> Dict[str, Any]:
        """Log user engagement event"""
        await self.db_pool.execute("""
            INSERT INTO user_engagement (user_id, session_id, event_type, event_data, duration_seconds)
            VALUES ($1, $2, $3, $4, $5)
        """, user_id, session_id, event_type, json.dumps(event_data), duration_seconds)

        return {"status": "logged", "event_type": event_type}

    async def _log_emotional_trend(self, user_id: str, emotion: str, intensity: float,
                                 context: str = None, crisis_flags: List = None) -> Dict[str, Any]:
        """Log emotional health data point"""
        await self.db_pool.execute("""
            INSERT INTO emotional_trends (user_id, emotion, intensity, context, crisis_flags)
            VALUES ($1, $2, $3, $4, $5)
        """, user_id, emotion, intensity, context, json.dumps(crisis_flags or []))

        return {"status": "logged", "emotion": emotion, "intensity": intensity}

    async def _get_active_alerts(self) -> Dict[str, Any]:
        """Get active (unresolved) system alerts"""
        alerts = await self.db_pool.fetch("""
            SELECT * FROM alert_logs
            WHERE resolved = FALSE
            ORDER BY timestamp DESC
            LIMIT 50
        """)

        return {
            "active_alerts": [dict(alert) for alert in alerts],
            "total_active": len(alerts),
            "severity_breakdown": {
                "critical": len([a for a in alerts if a["severity"] == "critical"]),
                "warning": len([a for a in alerts if a["severity"] == "warning"]),
                "info": len([a for a in alerts if a["severity"] == "info"])
            }
        }

    async def _get_system_health_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive system health dashboard"""
        current_metrics = await self._collect_system_metrics()
        performance = await self._get_performance_analytics(1)  # Last hour
        engagement = await self._get_user_engagement_analytics(1)  # Last day
        emotional = await self._get_emotional_health_analytics(1)  # Last day
        alerts = await self._get_active_alerts()

        # Calculate overall system health score
        health_components = {
            "system_performance": performance.get("performance_health_score", {}).get("health_score", 50),
            "error_rate": 100 - (performance.get("performance_health_score", {}).get("error_rate_percent", 5) * 10),
            "user_engagement": min(100, engagement.get("user_activity", {}).get("active_users", 0) * 2),
            "alert_status": 100 - (alerts.get("total_active", 0) * 5)
        }

        overall_health = sum(health_components.values()) / len(health_components)

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_health_score": overall_health,
            "health_status": "excellent" if overall_health >= 90 else "good" if overall_health >= 75 else "fair" if overall_health >= 60 else "concerning" if overall_health >= 40 else "critical",
            "system_metrics": current_metrics,
            "performance_metrics": performance,
            "user_engagement": engagement,
            "emotional_health": emotional,
            "active_alerts": alerts,
            "health_components": health_components,
            "recommendations": self._generate_dashboard_recommendations(overall_health, alerts)
        }

    def _generate_dashboard_recommendations(self, health_score: float, alerts: Dict) -> List[str]:
        """Generate dashboard recommendations based on health score and alerts"""
        recommendations = []

        if health_score < 60:
            recommendations.append("🚨 Critical system health - immediate attention required")

        if alerts.get("total_active", 0) > 5:
            recommendations.append("⚠️ Multiple active alerts - review alert dashboard")

        if health_score >= 90:
            recommendations.append("✅ System performing excellently - maintain monitoring")

        # Component-specific recommendations
        if alerts.get("severity_breakdown", {}).get("critical", 0) > 0:
            recommendations.append("🔴 Critical alerts present - investigate immediately")

        return recommendations

async def main():
    """Main MCP server loop"""
    server = OviyaMonitoringServer()
    await server.initialize_connections()

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
