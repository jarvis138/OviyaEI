#!/usr/bin/env python3
"""
Stripe MCP Server for Oviya EI
Handles monetization, subscriptions, and enterprise billing
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import stripe
import asyncpg

# Add project paths for standalone execution
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class OviyaStripeServer:
    """
    MCP Server for Stripe payment processing and monetization

    Provides:
    - Subscription management
    - Payment processing
    - Enterprise billing
    - Usage-based pricing
    - Revenue analytics
    """

    def __init__(self):
        # Stripe configuration
        stripe.api_key = os.getenv("STRIPE_SECRET_KEY")
        self.publishable_key = os.getenv("STRIPE_PUBLISHABLE_KEY")

        # Database connection
        self.db_pool = None
        self.database_url = os.getenv("DATABASE_URL", "postgresql://oviya:oviya_password@localhost:5432/oviya_db")

        # Pricing tiers
        self.pricing_tiers = {
            "free": {
                "name": "Free",
                "monthly_messages": 50,
                "features": ["Basic emotional support", "Limited conversation history"],
                "price": 0
            },
            "personal": {
                "name": "Personal",
                "monthly_messages": 500,
                "features": ["Unlimited conversations", "Full memory retention", "Priority support"],
                "price": 9.99,
                "stripe_price_id": "price_personal_monthly"
            },
            "professional": {
                "name": "Professional",
                "monthly_messages": 2000,
                "features": ["All Personal features", "Advanced analytics", "Custom integrations", "Phone support"],
                "price": 29.99,
                "stripe_price_id": "price_professional_monthly"
            },
            "enterprise": {
                "name": "Enterprise",
                "monthly_messages": 10000,
                "features": ["All Professional features", "Dedicated account manager", "Custom deployment", "SLA guarantee"],
                "price": 99.99,
                "stripe_price_id": "price_enterprise_monthly"
            }
        }

    async def initialize_database(self):
        """Initialize database connection for billing data"""
        try:
            self.db_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=5,
                max_size=20,
                command_timeout=60
            )

            # Create billing tables
            await self._create_billing_tables()
            print("Stripe MCP Server initialized successfully")

        except Exception as e:
            print(f"Failed to initialize Stripe MCP: {e}")
            raise

    async def _create_billing_tables(self):
        """Create billing and subscription related tables"""

        # Customer subscriptions
        await self.db_pool.execute("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                subscription_id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                stripe_customer_id VARCHAR(255),
                stripe_subscription_id VARCHAR(255),
                plan_type VARCHAR(50) NOT NULL,
                status VARCHAR(50) DEFAULT 'active',
                current_period_start TIMESTAMP,
                current_period_end TIMESTAMP,
                messages_used INTEGER DEFAULT 0,
                messages_limit INTEGER DEFAULT 50,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Payment transactions
        await self.db_pool.execute("""
            CREATE TABLE IF NOT EXISTS payments (
                payment_id VARCHAR(255) PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                stripe_payment_intent_id VARCHAR(255),
                amount_cents INTEGER NOT NULL,
                currency VARCHAR(3) DEFAULT 'usd',
                status VARCHAR(50) NOT NULL,
                description TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed_at TIMESTAMP
            )
        """)

        # Usage tracking
        await self.db_pool.execute("""
            CREATE TABLE IF NOT EXISTS usage_tracking (
                usage_id SERIAL PRIMARY KEY,
                user_id VARCHAR(255) NOT NULL,
                subscription_id VARCHAR(255),
                feature_used VARCHAR(100) NOT NULL,
                quantity INTEGER DEFAULT 1,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Revenue analytics
        await self.db_pool.execute("""
            CREATE TABLE IF NOT EXISTS revenue_analytics (
                analytics_id SERIAL PRIMARY KEY,
                period_start DATE NOT NULL,
                period_end DATE NOT NULL,
                total_revenue_cents BIGINT DEFAULT 0,
                new_subscriptions INTEGER DEFAULT 0,
                churned_subscriptions INTEGER DEFAULT 0,
                active_subscriptions INTEGER DEFAULT 0,
                mrr_cents BIGINT DEFAULT 0,
                arr_cents BIGINT DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create indexes
        await self.db_pool.execute("""
            CREATE INDEX IF NOT EXISTS idx_subscriptions_user_id ON subscriptions(user_id);
            CREATE INDEX IF NOT EXISTS idx_subscriptions_status ON subscriptions(status);
            CREATE INDEX IF NOT EXISTS idx_payments_user_id ON payments(user_id);
            CREATE INDEX IF NOT EXISTS idx_usage_user_id ON usage_tracking(user_id);
            CREATE INDEX IF NOT EXISTS idx_usage_timestamp ON usage_tracking(timestamp);
        """)

    async def _create_stripe_customer(self, user_email: str, user_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Create a Stripe customer"""
        try:
            customer = stripe.Customer.create(
                email=user_email,
                metadata=user_metadata
            )

            # Store in database
            await self.db_pool.execute("""
                INSERT INTO customers (user_id, stripe_customer_id, email, metadata)
                VALUES ($1, $2, $3, $4)
                ON CONFLICT (user_id) DO UPDATE SET
                    stripe_customer_id = EXCLUDED.stripe_customer_id,
                    email = EXCLUDED.email,
                    metadata = EXCLUDED.metadata,
                    updated_at = CURRENT_TIMESTAMP
            """, user_metadata.get("user_id"), customer.id, user_email, json.dumps(user_metadata))

            return {
                "customer_id": customer.id,
                "status": "created",
                "email": user_email
            }

        except stripe.error.StripeError as e:
            return {"error": str(e)}

    async def _create_subscription(self, customer_id: str, price_id: str, user_id: str) -> Dict[str, Any]:
        """Create a Stripe subscription"""
        try:
            subscription = stripe.Subscription.create(
                customer=customer_id,
                items=[{"price": price_id}],
                metadata={"user_id": user_id}
            )

            # Store in database
            await self.db_pool.execute("""
                INSERT INTO subscriptions (
                    subscription_id, user_id, stripe_customer_id, stripe_subscription_id,
                    plan_type, status, current_period_start, current_period_end
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (subscription_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    current_period_start = EXCLUDED.current_period_start,
                    current_period_end = EXCLUDED.current_period_end,
                    updated_at = CURRENT_TIMESTAMP
            """,
            subscription.id,
            user_id,
            customer_id,
            subscription.id,
            self._get_plan_from_price_id(price_id),
            subscription.status,
            datetime.fromtimestamp(subscription.current_period_start),
            datetime.fromtimestamp(subscription.current_period_end)
            )

            return {
                "subscription_id": subscription.id,
                "status": subscription.status,
                "current_period_start": subscription.current_period_start,
                "current_period_end": subscription.current_period_end
            }

        except stripe.error.StripeError as e:
            return {"error": str(e)}

    def _get_plan_from_price_id(self, price_id: str) -> str:
        """Map Stripe price ID to plan type"""
        for plan_name, plan_data in self.pricing_tiers.items():
            if plan_data.get("stripe_price_id") == price_id:
                return plan_name
        return "unknown"

    async def _track_usage(self, user_id: str, feature: str, quantity: int = 1) -> Dict[str, Any]:
        """Track feature usage for billing"""
        try:
            # Insert usage record
            await self.db_pool.execute("""
                INSERT INTO usage_tracking (user_id, subscription_id, feature_used, quantity)
                SELECT $1, subscription_id, $2, $3
                FROM subscriptions
                WHERE user_id = $1 AND status = 'active'
                LIMIT 1
            """, user_id, feature, quantity)

            # Update subscription usage
            if feature == "message":
                await self.db_pool.execute("""
                    UPDATE subscriptions
                    SET messages_used = messages_used + $2, updated_at = CURRENT_TIMESTAMP
                    WHERE user_id = $1 AND status = 'active'
                """, user_id, quantity)

            return {"status": "tracked", "feature": feature, "quantity": quantity}

        except Exception as e:
            return {"error": str(e)}

    async def _check_limits(self, user_id: str, feature: str) -> Dict[str, Any]:
        """Check if user has exceeded usage limits"""
        try:
            # Get current subscription
            subscription = await self.db_pool.fetchrow("""
                SELECT plan_type, messages_used, messages_limit, status
                FROM subscriptions
                WHERE user_id = $1 AND status = 'active'
                ORDER BY created_at DESC
                LIMIT 1
            """, user_id)

            if not subscription:
                # Free tier limits
                free_limits = self.pricing_tiers["free"]
                current_usage = await self.db_pool.fetchval("""
                    SELECT COALESCE(SUM(quantity), 0)
                    FROM usage_tracking
                    WHERE user_id = $1 AND feature_used = $2
                    AND timestamp >= DATE_TRUNC('month', CURRENT_DATE)
                """, user_id, feature)

                return {
                    "plan_type": "free",
                    "limit": free_limits["monthly_messages"] if feature == "message" else float('inf'),
                    "used": current_usage,
                    "remaining": max(0, free_limits["monthly_messages"] - current_usage) if feature == "message" else float('inf'),
                    "within_limit": current_usage < free_limits["monthly_messages"] if feature == "message" else True
                }

            # Paid tier - check limits
            plan_limits = self.pricing_tiers.get(subscription["plan_type"], {})
            limit = plan_limits.get("monthly_messages", float('inf'))

            return {
                "plan_type": subscription["plan_type"],
                "limit": limit,
                "used": subscription["messages_used"],
                "remaining": max(0, limit - subscription["messages_used"]),
                "within_limit": subscription["messages_used"] < limit
            }

        except Exception as e:
            return {"error": str(e)}

    async def _get_revenue_analytics(self, period_days: int = 30) -> Dict[str, Any]:
        """Get revenue analytics for the specified period"""
        try:
            # Calculate period
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=period_days)

            # Get revenue data
            revenue_data = await self.db_pool.fetchrow("""
                SELECT
                    COALESCE(SUM(amount_cents), 0) as total_revenue,
                    COUNT(*) FILTER (WHERE status = 'succeeded') as successful_payments,
                    COUNT(*) FILTER (WHERE status != 'succeeded') as failed_payments,
                    AVG(amount_cents) FILTER (WHERE status = 'succeeded') as avg_payment
                FROM payments
                WHERE created_at >= $1 AND created_at <= $2
            """, start_date, end_date)

            # Get subscription metrics
            subscription_data = await self.db_pool.fetchrow("""
                SELECT
                    COUNT(*) FILTER (WHERE status = 'active') as active_subscriptions,
                    COUNT(*) FILTER (WHERE created_at >= $1) as new_subscriptions,
                    COUNT(*) FILTER (WHERE status = 'canceled' AND updated_at >= $1) as churned_subscriptions
                FROM subscriptions
            """, start_date)

            # Calculate MRR (Monthly Recurring Revenue)
            mrr_data = await self.db_pool.fetchrow("""
                SELECT COALESCE(SUM(p.price), 0) as mrr_cents
                FROM subscriptions s
                JOIN pricing_tiers p ON s.plan_type = p.name
                WHERE s.status = 'active'
            """)

            analytics = {
                "period_days": period_days,
                "period_start": start_date.isoformat(),
                "period_end": end_date.isoformat(),
                "revenue": {
                    "total_cents": revenue_data["total_revenue"] or 0,
                    "total_dollars": (revenue_data["total_revenue"] or 0) / 100,
                    "successful_payments": revenue_data["successful_payments"] or 0,
                    "failed_payments": revenue_data["failed_payments"] or 0,
                    "average_payment_cents": revenue_data["avg_payment"] or 0
                },
                "subscriptions": {
                    "active": subscription_data["active_subscriptions"] or 0,
                    "new": subscription_data["new_subscriptions"] or 0,
                    "churned": subscription_data["churned_subscriptions"] or 0,
                    "net_growth": (subscription_data["new_subscriptions"] or 0) - (subscription_data["churned_subscriptions"] or 0)
                },
                "mrr": {
                    "monthly_recurring_revenue_cents": mrr_data["mrr_cents"] or 0,
                    "monthly_recurring_revenue_dollars": (mrr_data["mrr_cents"] or 0) / 100
                },
                "metrics": {
                    "arpu": ((mrr_data["mrr_cents"] or 0) / 100) / max(subscription_data["active_subscriptions"] or 1, 1),
                    "churn_rate": (subscription_data["churned_subscriptions"] or 0) / max(subscription_data["active_subscriptions"] or 1, 1),
                    "conversion_rate": (subscription_data["active_subscriptions"] or 0) / max(subscription_data["new_subscriptions"] or 1, 1)
                }
            }

            return analytics

        except Exception as e:
            return {"error": str(e)}

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests"""

        if request.get("method") == "tools/call":
            tool_name = request["params"]["name"]
            arguments = request["params"].get("arguments", {})

            try:
                if tool_name == "create_customer":
                    result = await self._create_stripe_customer(
                        arguments.get("email"),
                        arguments.get("metadata", {})
                    )
                    return {
                        "content": [{"type": "text", "text": json.dumps(result)}]
                    }

                elif tool_name == "create_subscription":
                    result = await self._create_subscription(
                        arguments.get("customer_id"),
                        arguments.get("price_id"),
                        arguments.get("user_id")
                    )
                    return {
                        "content": [{"type": "text", "text": json.dumps(result)}]
                    }

                elif tool_name == "track_usage":
                    result = await self._track_usage(
                        arguments.get("user_id"),
                        arguments.get("feature"),
                        arguments.get("quantity", 1)
                    )
                    return {
                        "content": [{"type": "text", "text": json.dumps(result)}]
                    }

                elif tool_name == "check_limits":
                    result = await self._check_limits(
                        arguments.get("user_id"),
                        arguments.get("feature")
                    )
                    return {
                        "content": [{"type": "text", "text": json.dumps(result)}]
                    }

                elif tool_name == "get_revenue_analytics":
                    result = await self._get_revenue_analytics(
                        arguments.get("period_days", 30)
                    )
                    return {
                        "content": [{"type": "text", "text": json.dumps(result)}]
                    }

                elif tool_name == "process_payment":
                    # Process a one-time payment
                    result = await self._process_payment(
                        arguments.get("customer_id"),
                        arguments.get("amount_cents"),
                        arguments.get("description", "")
                    )
                    return {
                        "content": [{"type": "text", "text": json.dumps(result)}]
                    }

                elif tool_name == "get_pricing_tiers":
                    return {
                        "content": [{"type": "text", "text": json.dumps({
                            "tiers": self.pricing_tiers,
                            "currency": "USD"
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
                        "name": "create_customer",
                        "description": "Create a Stripe customer for billing",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "email": {"type": "string"},
                                "metadata": {"type": "object"}
                            },
                            "required": ["email"]
                        }
                    },
                    {
                        "name": "create_subscription",
                        "description": "Create a subscription for a customer",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "customer_id": {"type": "string"},
                                "price_id": {"type": "string"},
                                "user_id": {"type": "string"}
                            },
                            "required": ["customer_id", "price_id", "user_id"]
                        }
                    },
                    {
                        "name": "track_usage",
                        "description": "Track feature usage for billing",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "string"},
                                "feature": {"type": "string"},
                                "quantity": {"type": "integer", "default": 1}
                            },
                            "required": ["user_id", "feature"]
                        }
                    },
                    {
                        "name": "check_limits",
                        "description": "Check if user has exceeded usage limits",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "string"},
                                "feature": {"type": "string"}
                            },
                            "required": ["user_id", "feature"]
                        }
                    },
                    {
                        "name": "get_revenue_analytics",
                        "description": "Get revenue and subscription analytics",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "period_days": {"type": "integer", "default": 30}
                            }
                        }
                    },
                    {
                        "name": "process_payment",
                        "description": "Process a one-time payment",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "customer_id": {"type": "string"},
                                "amount_cents": {"type": "integer"},
                                "description": {"type": "string"}
                            },
                            "required": ["customer_id", "amount_cents"]
                        }
                    },
                    {
                        "name": "get_pricing_tiers",
                        "description": "Get available pricing tiers",
                        "inputSchema": {"type": "object", "properties": {}}
                    }
                ]
            }

        elif request.get("method") == "resources/list":
            return {
                "resources": [
                    {
                        "uri": "stripe://pricing",
                        "name": "Pricing Tiers",
                        "description": "Available subscription plans and pricing",
                        "mimeType": "application/json"
                    },
                    {
                        "uri": "stripe://analytics/revenue",
                        "name": "Revenue Analytics",
                        "description": "Real-time revenue and subscription metrics",
                        "mimeType": "application/json"
                    },
                    {
                        "uri": "stripe://metrics/usage",
                        "name": "Usage Metrics",
                        "description": "System-wide usage statistics",
                        "mimeType": "application/json"
                    }
                ]
            }

        elif request.get("method") == "resources/read":
            uri = request["params"]["uri"]

            if uri == "stripe://pricing":
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps({
                            "pricing_tiers": self.pricing_tiers,
                            "currency": "USD",
                            "billing_cycle": "monthly"
                        })
                    }]
                }

            elif uri == "stripe://analytics/revenue":
                analytics = await self._get_revenue_analytics(30)
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(analytics)
                    }]
                }

            elif uri == "stripe://metrics/usage":
                # Get usage metrics
                usage_stats = await self.db_pool.fetchrow("""
                    SELECT
                        COUNT(DISTINCT user_id) as active_users,
                        SUM(quantity) FILTER (WHERE feature_used = 'message') as total_messages,
                        COUNT(*) FILTER (WHERE feature_used = 'message') as message_count
                    FROM usage_tracking
                    WHERE timestamp >= CURRENT_DATE - INTERVAL '30 days'
                """)

                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps({
                            "period": "30 days",
                            "active_users": usage_stats["active_users"] or 0,
                            "total_messages": usage_stats["total_messages"] or 0,
                            "avg_messages_per_user": (usage_stats["total_messages"] or 0) / max(usage_stats["active_users"] or 1, 1)
                        })
                    }]
                }

        return {"error": "Method not supported"}

    async def _process_payment(self, customer_id: str, amount_cents: int, description: str) -> Dict[str, Any]:
        """Process a one-time payment"""
        try:
            payment_intent = stripe.PaymentIntent.create(
                amount=amount_cents,
                currency="usd",
                customer=customer_id,
                description=description,
                metadata={"processed_by": "oviya_mcp"}
            )

            # Store payment record
            await self.db_pool.execute("""
                INSERT INTO payments (
                    payment_id, user_id, stripe_payment_intent_id, amount_cents, status, description
                ) VALUES ($1, $2, $3, $4, $5, $6)
            """,
            payment_intent.id,
            "",  # user_id would need to be looked up from customer_id
            payment_intent.id,
            amount_cents,
            payment_intent.status,
            description
            )

            return {
                "payment_intent_id": payment_intent.id,
                "client_secret": payment_intent.client_secret,
                "status": payment_intent.status,
                "amount_cents": amount_cents
            }

        except stripe.error.StripeError as e:
            return {"error": str(e)}

async def main():
    """Main MCP server loop"""
    server = OviyaStripeServer()
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

if __name__ == "__main__":
    asyncio.run(main())
