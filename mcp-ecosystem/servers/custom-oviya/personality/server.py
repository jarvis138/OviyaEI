#!/usr/bin/env python3
"""
Oviya Personality MCP Server
Exposes the 5-pillar personality vector system as MCP tools
"""

import asyncio
import json
import os
import sys
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import numpy as np

# Add project paths for standalone execution
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

class OviyaPersonalityServer:
    """
    MCP Server that exposes Oviya's 5-pillar personality system

    Pillars:
    - Ma (Innovation/Creativity)
    - Ahimsa (Non-violence/Safety)
    - Jeong (Deep connection/Empathy)
    - Logos (Reason/Logic)
    - Lagom (Balance/Appropriateness)
    """

    def __init__(self):
        self.base_personality_vectors = {
            "calm_supportive": {
                "Ma": 0.2, "Ahimsa": 0.4, "Jeong": 0.3, "Logos": 0.05, "Lagom": 0.05
            },
            "empathetic_listener": {
                "Ma": 0.1, "Ahimsa": 0.3, "Jeong": 0.4, "Logos": 0.1, "Lagom": 0.1
            },
            "wise_counselor": {
                "Ma": 0.15, "Ahimsa": 0.25, "Jeong": 0.25, "Logos": 0.25, "Lagom": 0.1
            },
            "creative_problem_solver": {
                "Ma": 0.4, "Ahimsa": 0.2, "Jeong": 0.15, "Logos": 0.15, "Lagom": 0.1
            },
            "balanced_mediator": {
                "Ma": 0.1, "Ahimsa": 0.2, "Jeong": 0.2, "Logos": 0.2, "Lagom": 0.3
            }
        }

        # Personality evolution patterns
        self.evolution_patterns = {
            "trust_building": {"Jeong": +0.1, "Ahimsa": +0.05},
            "crisis_response": {"Ahimsa": +0.15, "Logos": +0.05, "Lagom": +0.05},
            "creative_session": {"Ma": +0.1, "Jeong": +0.05},
            "deep_reflection": {"Logos": +0.1, "Jeong": +0.1},
            "balance_restoration": {"Lagom": +0.15, "Ahimsa": +0.05}
        }

    def _normalize_vector(self, vector: Dict[str, float]) -> Dict[str, float]:
        """Normalize personality vector to ensure sum <= 1"""
        total = sum(vector.values())
        if total > 1.0:
            return {k: v / total for k, v in vector.items()}
        return vector

    def _compute_personality_vector(self, emotion: str, context: str, memory_state: Dict[str, Any]) -> Dict[str, float]:
        """
        Compute personality vector based on emotional context and conversation history

        This implements Oviya's core personality adaptation logic
        """
        # Start with base empathetic personality
        vector = self.base_personality_vectors["empathetic_listener"].copy()

        # Adapt based on emotion
        emotion_adaptations = {
            "sad": {"Jeong": +0.15, "Ahimsa": +0.1, "Lagom": +0.05},
            "anxious": {"Ahimsa": +0.2, "Lagom": +0.1, "Logos": +0.05},
            "angry": {"Ahimsa": +0.25, "Jeong": +0.1, "Lagom": +0.05},
            "confused": {"Logos": +0.2, "Jeong": +0.1, "Ahimsa": +0.05},
            "joyful": {"Ma": +0.15, "Jeong": +0.1, "Lagom": +0.05},
            "overwhelmed": {"Ahimsa": +0.15, "Lagom": +0.15, "Jeong": +0.1},
            "hopeful": {"Jeong": +0.1, "Ma": +0.1, "Lagom": +0.05},
            "frustrated": {"Logos": +0.1, "Ahimsa": +0.1, "Jeong": +0.1}
        }

        if emotion in emotion_adaptations:
            for pillar, adjustment in emotion_adaptations[emotion].items():
                vector[pillar] = min(1.0, vector[pillar] + adjustment)

        # Consider conversation context
        context_keywords = {
            "crisis": ["suicide", "kill myself", "end it all", "can't go on"],
            "creativity": ["create", "imagine", "design", "build"],
            "logic": ["understand", "explain", "reason", "analyze"],
            "balance": ["balance", "moderate", "sustainable", "harmony"],
            "connection": ["relationship", "friend", "family", "connect"]
        }

        for context_type, keywords in context_keywords.items():
            if any(keyword.lower() in context.lower() for keyword in keywords):
                if context_type == "crisis":
                    vector["Ahimsa"] = min(1.0, vector["Ahimsa"] + 0.2)
                    vector["Jeong"] = min(1.0, vector["Jeong"] + 0.1)
                elif context_type == "creativity":
                    vector["Ma"] = min(1.0, vector["Ma"] + 0.15)
                elif context_type == "logic":
                    vector["Logos"] = min(1.0, vector["Logos"] + 0.15)
                elif context_type == "balance":
                    vector["Lagom"] = min(1.0, vector["Lagom"] + 0.15)
                elif context_type == "connection":
                    vector["Jeong"] = min(1.0, vector["Jeong"] + 0.15)

        # Apply memory state evolution
        if memory_state:
            conversation_count = memory_state.get("conversation_count", 0)
            trust_level = memory_state.get("trust_level", 0.5)

            # Increase trust-based pillars over time
            if trust_level > 0.7:
                vector["Jeong"] = min(1.0, vector["Jeong"] + 0.1)
                vector["Ahimsa"] = min(1.0, vector["Ahimsa"] + 0.05)

            # Increase wisdom pillars with experience
            if conversation_count > 10:
                vector["Logos"] = min(1.0, vector["Logos"] + 0.05)
                vector["Lagom"] = min(1.0, vector["Lagom"] + 0.05)

        return self._normalize_vector(vector)

    def _update_personality_memory(self, user_id: str, new_vector: Dict[str, float]) -> Dict[str, Any]:
        """
        Update long-term personality memory (would integrate with memory system)
        """
        # This would typically store in persistent memory system
        # For now, return success status
        return {
            "user_id": user_id,
            "vector": new_vector,
            "timestamp": datetime.now().isoformat(),
            "status": "updated"
        }

    def _get_personality_history(self, user_id: str, days: int = 30) -> List[Dict[str, Any]]:
        """
        Get personality vector evolution over time
        """
        # This would query the memory system for historical vectors
        # For now, return sample data
        return [
            {
                "timestamp": (datetime.now() - timedelta(days=i)).isoformat(),
                "vector": self.base_personality_vectors["empathetic_listener"],
                "context": f"Day {i} conversation"
            } for i in range(min(days, 7))  # Sample last 7 days
        ]

    def _get_cultural_adaptation(self, culture: str) -> Dict[str, float]:
        """
        Adapt personality vector for cultural context
        """
        cultural_adaptations = {
            "western": {"Logos": +0.1, "Ma": +0.05},
            "eastern": {"Jeong": +0.1, "Lagom": +0.05, "Ahimsa": +0.05},
            "latin_american": {"Jeong": +0.15, "Ma": +0.05},
            "middle_eastern": {"Ahimsa": +0.1, "Jeong": +0.1},
            "african": {"Jeong": +0.1, "Lagom": +0.1}
        }

        base_vector = self.base_personality_vectors["empathetic_listener"].copy()
        if culture in cultural_adaptations:
            for pillar, adjustment in cultural_adaptations[culture].items():
                base_vector[pillar] = min(1.0, base_vector[pillar] + adjustment)

        return self._normalize_vector(base_vector)

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests"""

        if request.get("method") == "tools/call":
            tool_name = request["params"]["name"]
            arguments = request["params"].get("arguments", {})

            try:
                if tool_name == "compute_personality_vector":
                    result = self._compute_personality_vector(
                        arguments.get("emotion", ""),
                        arguments.get("context", ""),
                        arguments.get("memory_state", {})
                    )
                    return {
                        "content": [{"type": "text", "text": json.dumps({
                            "personality_vector": result,
                            "pillars": {
                                "Ma": "Innovation & Creativity",
                                "Ahimsa": "Non-violence & Safety",
                                "Jeong": "Deep Connection & Empathy",
                                "Logos": "Reason & Logic",
                                "Lagom": "Balance & Appropriateness"
                            },
                            "confidence": 0.85
                        })}]
                    }

                elif tool_name == "update_personality_memory":
                    result = self._update_personality_memory(
                        arguments.get("user_id"),
                        arguments.get("new_vector")
                    )
                    return {
                        "content": [{"type": "text", "text": json.dumps(result)}]
                    }

                elif tool_name == "get_personality_history":
                    result = self._get_personality_history(
                        arguments.get("user_id"),
                        arguments.get("days", 30)
                    )
                    return {
                        "content": [{"type": "text", "text": json.dumps({
                            "history": result,
                            "total_entries": len(result)
                        })}]
                    }

                elif tool_name == "adapt_cultural_personality":
                    result = self._get_cultural_adaptation(arguments.get("culture", "western"))
                    return {
                        "content": [{"type": "text", "text": json.dumps({
                            "culture": arguments.get("culture"),
                            "adapted_vector": result,
                            "cultural_notes": "Adapted for cultural communication preferences"
                        })}]
                    }

                elif tool_name == "analyze_personality_evolution":
                    # Analyze how personality has evolved
                    history = self._get_personality_history(arguments.get("user_id"))
                    evolution = self._analyze_evolution(history)
                    return {
                        "content": [{"type": "text", "text": json.dumps(evolution)}]
                    }

                else:
                    return {"error": f"Unknown tool: {tool_name}"}

            except Exception as e:
                return {"error": str(e)}

        elif request.get("method") == "tools/list":
            return {
                "tools": [
                    {
                        "name": "compute_personality_vector",
                        "description": "Compute the 5-pillar personality vector for Oviya's response",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "emotion": {"type": "string", "description": "User's current emotion"},
                                "context": {"type": "string", "description": "Conversation context"},
                                "memory_state": {"type": "object", "description": "User's memory state"}
                            },
                            "required": ["emotion", "context"]
                        }
                    },
                    {
                        "name": "update_personality_memory",
                        "description": "Update user's personality memory",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "user_id": {"type": "string"},
                                "new_vector": {"type": "object"}
                            },
                            "required": ["user_id", "new_vector"]
                        }
                    },
                    {
                        "name": "get_personality_history",
                        "description": "Get personality vector evolution over time",
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
                        "name": "adapt_cultural_personality",
                        "description": "Adapt personality vector for cultural context",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "culture": {"type": "string"}
                            },
                            "required": ["culture"]
                        }
                    },
                    {
                        "name": "analyze_personality_evolution",
                        "description": "Analyze how user's personality preferences have evolved",
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

        elif request.get("method") == "resources/list":
            return {
                "resources": [
                    {
                        "uri": "personality://pillars",
                        "name": "Oviya Personality Pillars",
                        "description": "The 5-pillar personality system definition",
                        "mimeType": "application/json"
                    },
                    {
                        "uri": "personality://cultural_profiles",
                        "name": "Cultural Personality Profiles",
                        "description": "Personality adaptations for different cultures",
                        "mimeType": "application/json"
                    }
                ]
            }

        elif request.get("method") == "resources/read":
            uri = request["params"]["uri"]

            if uri == "personality://pillars":
                pillars_info = {
                    "pillars": {
                        "Ma": {
                            "description": "Innovation and Creativity",
                            "core_value": "Bringing fresh perspectives and creative solutions",
                            "expression": "Novel approaches, imaginative solutions, artistic expression"
                        },
                        "Ahimsa": {
                            "description": "Non-violence and Safety",
                            "core_value": "Creating safe, non-judgmental spaces",
                            "expression": "Gentle communication, safety-first approach, harm prevention"
                        },
                        "Jeong": {
                            "description": "Deep Connection and Empathy",
                            "core_value": "Building profound human connections",
                            "expression": "Deep listening, emotional resonance, relationship building"
                        },
                        "Logos": {
                            "description": "Reason and Logic",
                            "core_value": "Clear thinking and rational understanding",
                            "expression": "Logical analysis, clear explanations, evidence-based insights"
                        },
                        "Lagom": {
                            "description": "Balance and Appropriateness",
                            "core_value": "Finding the right measure in all things",
                            "expression": "Balanced responses, contextual appropriateness, sustainable pacing"
                        }
                    }
                }
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(pillars_info)
                    }]
                }

            elif uri == "personality://cultural_profiles":
                cultural_profiles = {
                    "profiles": {
                        "western": "Logic and individual achievement focus",
                        "eastern": "Harmony and collective well-being focus",
                        "latin_american": "Warm relational connections focus",
                        "middle_eastern": "Hospitality and community focus",
                        "african": "Ubuntu - interconnected humanity focus"
                    }
                }
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps(cultural_profiles)
                    }]
                }

        return {"error": "Method not supported"}

    def _analyze_evolution(self, history: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze personality evolution patterns"""
        if not history:
            return {"evolution": "insufficient_data"}

        # Simple evolution analysis
        recent_vectors = [entry["vector"] for entry in history[:7]]  # Last 7 entries

        if len(recent_vectors) < 2:
            return {"evolution": "insufficient_data"}

        # Calculate trends
        trends = {}
        for pillar in ["Ma", "Ahimsa", "Jeong", "Logos", "Lagom"]:
            values = [v.get(pillar, 0) for v in recent_vectors]
            if len(values) > 1:
                trend = values[-1] - values[0]  # Simple trend
                trends[pillar] = {
                    "trend": "increasing" if trend > 0.05 else "decreasing" if trend < -0.05 else "stable",
                    "change": trend
                }

        return {
            "evolution_period": f"{len(recent_vectors)} most recent entries",
            "pillar_trends": trends,
            "insights": [
                "Jeong (deep connection) appears to be strengthening" if trends.get("Jeong", {}).get("trend") == "increasing" else "Personality showing stable adaptation patterns"
            ]
        }

async def main():
    """Main MCP server loop"""
    server = OviyaPersonalityServer()

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
