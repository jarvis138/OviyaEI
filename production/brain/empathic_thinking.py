#!/usr/bin/env python3
"""
Oviya Empathic Thinking Engine
Advanced cognitive modes for deep emotional intelligence
"""

import asyncio
import json
import re
from typing import Dict, List, Any, Optional
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class EmpathicThinkingEngine:
    """
    Advanced cognitive empathy system aligned with Oviya's personality pillars

    Thinking Modes:
    - Empathetic: Deep emotional understanding (Jeong/Ahimsa)
    - Dialectical: Resolving emotional conflicts
    - Reflective: Mirroring user's internal state
    - Metacognitive: Understanding thought patterns
    - Creative: Generating novel coping strategies
    """

    def __init__(self):
        # 🆕 REAL MCP CLIENT: Initialize MCP Thinking client
        try:
            from .mcp_client import get_mcp_client
            self.thinking_client = get_mcp_client("mcp-thinking")
            if self.thinking_client:
                # Initialize asynchronously (will be done on first use)
                self._thinking_initialized = False
                print("✅ MCP Thinking client ready")
            else:
                self.thinking_client = MockMCPClient("oviya-thinking")
                print("⚠️ MCP Thinking not configured, using mock")
        except Exception as e:
            print(f"⚠️ Failed to initialize MCP Thinking: {e}")
            self.thinking_client = MockMCPClient("oviya-thinking")
        self._thinking_initialized = False

        # Personality pillar mappings
        self.pillar_mappings = {
            "Ma": "innovation",
            "Ahimsa": "non-violence",
            "Jeong": "deep_connection",
            "Logos": "reason",
            "Lagom": "balance"
        }

    async def generate_empathic_response(self, user_input: str, personality_vector: Dict[str, float],
                                       emotion_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate deep empathic response using multiple cognitive modes

        Args:
            user_input: User's message
            personality_vector: 5-pillar personality weights
            emotion_context: Current emotional state

        Returns:
            Comprehensive empathic response with multiple thinking modes
        """

        # 1. Empathetic mode - core emotional understanding
        empathetic_analysis = await self._empathetic_thinking(
            user_input, personality_vector, emotion_context
        )

        # 2. Check for dialectical conflicts
        dialectical_insights = None
        if self._has_emotional_conflicts(emotion_context):
            dialectical_insights = await self._dialectical_thinking(
                emotion_context.get("conflicts", []),
                user_input
            )

        # 3. Reflective mirroring
        reflective_insights = await self._reflective_thinking(
            user_input, personality_vector
        )

        # 4. Metacognitive analysis
        metacognitive_insights = await self._metacognitive_thinking(
            empathetic_analysis.get("reasoning", ""),
            emotion_context.get("patterns", [])
        )

        # 5. Creative coping strategies (if needed)
        creative_strategies = None
        if emotion_context.get("intensity", 0) > 0.6:
            creative_strategies = await self._creative_thinking(
                user_input, personality_vector
            )

        # Combine into comprehensive response
        response = self._synthesize_response(
            empathetic_analysis,
            dialectical_insights,
            reflective_insights,
            metacognitive_insights,
            creative_strategies
        )

        return {
            "response": response["text"],
            "thinking_modes_used": response["modes"],
            "pillar_weights": personality_vector,
            "emotional_insights": {
                "primary_emotion": emotion_context.get("emotion", "unknown"),
                "intensity": emotion_context.get("intensity", 0.5),
                "conflicts_detected": dialectical_insights is not None,
                "patterns_identified": len(emotion_context.get("patterns", []))
            },
            "cognitive_depth": self._calculate_cognitive_depth(response["modes"]),
            "follow_up_suggestions": response.get("follow_ups", [])
        }

    async def _empathetic_thinking(self, user_input: str, personality_vector: Dict[str, float],
                                 emotion_context: Dict[str, Any]) -> Dict[str, Any]:
        """Core empathetic understanding aligned with personality pillars"""

        try:
            result = await self.thinking_client.call_tool("empathetic_thinking", {
                "input_text": user_input,
                "personality_context": personality_vector,
                "emotion_state": emotion_context
            })

            # Enhance with pillar-specific responses
            jeong_weight = personality_vector.get("Jeong", 0.15)
            ahimsa_weight = personality_vector.get("Ahimsa", 0.4)

            if jeong_weight > 0.2:
                result["response"] += " I'm here with you in this shared experience."

            if ahimsa_weight > 0.3:
                result["response"] += " Your feelings are valid and deserve gentle understanding."

            return result

        except Exception as e:
            # Fallback local empathetic response
            return self._fallback_empathetic_response(user_input, personality_vector, emotion_context)

    def _fallback_empathetic_response(self, user_input: str, personality_vector: Dict[str, float],
                                    emotion_context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback empathetic response when MCP is unavailable"""

        emotion = emotion_context.get("emotion", "unknown")
        intensity = emotion_context.get("intensity", 0.5)

        # Base empathetic responses
        empathy_templates = {
            "sad": [
                "I can feel the weight of what you're carrying",
                "This sadness feels so heavy for you right now",
                "Your heart is hurting, and that's so valid"
            ],
            "anxious": [
                "That anxiety must feel overwhelming",
                "Your nervous system is working so hard right now",
                "It's understandable to feel this worried"
            ],
            "angry": [
                "This anger has important information for you",
                "These strong feelings are trying to protect something important",
                "Your anger deserves to be heard and understood"
            ],
            "joyful": [
                "Your joy is beautiful and contagious",
                "This happiness feels so genuine",
                "I'm delighted to witness your joy"
            ]
        }

        responses = empathy_templates.get(emotion, [
            "I hear you and I'm here with you",
            "Your feelings matter to me",
            "I'm listening with my full attention"
        ])

        # Select response based on intensity
        response_index = min(int(intensity * len(responses)), len(responses) - 1)
        response = responses[response_index]

        # Add pillar-specific enhancements
        if personality_vector.get("Jeong", 0.15) > 0.2:
            response += " - we're connected in this moment."

        return {
            "response": response,
            "reasoning": f"Empathetic response for {emotion} emotion at intensity {intensity}",
            "pillar_weights": personality_vector,
            "fallback": True
        }

    async def _dialectical_thinking(self, conflicts: List[str], user_input: str) -> Dict[str, Any]:
        """
        Resolve emotional conflicts through dialectical thinking
        
        🆕 CSM-1B Compatible:
        - Uses MCP Thinking enhanced_thinking for complex conflicts
        - Returns structured reasoning that enhances CSM-1B context
        - Affects prosody (more measured when resolving conflicts)
        """
        
        # Initialize MCP client if needed
        if not self._thinking_initialized and hasattr(self.thinking_client, 'initialize'):
            try:
                await self.thinking_client.initialize()
                self._thinking_initialized = True
            except Exception as e:
                print(f"MCP Thinking initialization failed: {e}")
                self._thinking_initialized = False

        # Check complexity - use enhanced_thinking for complex conflicts
        is_complex = len(conflicts) > 2 or self._has_deep_conflicts(conflicts)

        if is_complex:
            try:
                # 🆕 USE ENHANCED THINKING: For complex dialectical reasoning
                # Format conflicts for MCP Thinking
                conflict_text = f"Emotional conflicts: {', '.join(conflicts)}. User statement: {user_input}"
                
                result = await self.thinking_client.call_tool("enhanced_thinking", {
                    "thought": conflict_text,
                    "thought_type": "DIALECTICAL_ANALYSIS",
                    "strategy": "DIALECTICAL",
                    "tags": ["emotion", "conflict", "therapy", "dialectical"]
                })
                
                # Parse enhanced thinking result
                if isinstance(result, dict):
                    # Extract reasoning from enhanced thinking
                    reasoning = result.get("reasoning", result.get("text", ""))
                    if isinstance(reasoning, str):
                        return {
                            "response": self._extract_synthesis_from_reasoning(reasoning),
                            "conflicts_resolved": conflicts,
                            "synthesis": reasoning,
                            "thinking_mode": "dialectical",
                            "enhanced": True
                        }
                
                # Fallback if parsing fails
                return self._fallback_dialectical_analysis(conflicts, user_input)
            
            except Exception as e:
                print(f"Enhanced thinking failed: {e}")
                # Fallback to local
                return self._fallback_dialectical_analysis(conflicts, user_input)
        
        else:
            # Use local dialectical thinking for simple conflicts (faster)
            return self._fallback_dialectical_analysis(conflicts, user_input)
    
    def _has_deep_conflicts(self, conflicts: List[str]) -> bool:
        """Check if conflicts are deep/complex"""
        # Deep conflicts involve fundamental tensions
        deep_keywords = ["love", "hate", "hope", "despair", "trust", "fear", "joy", "sadness"]
        conflict_text = " ".join(conflicts).lower()
        return any(keyword in conflict_text for keyword in deep_keywords)
    
    def _extract_synthesis_from_reasoning(self, reasoning: str) -> str:
        """Extract dialectical synthesis from enhanced thinking reasoning"""
        # Look for synthesis patterns in reasoning
        if "synthesis" in reasoning.lower() or "both" in reasoning.lower():
            # Extract synthesis sentence
            sentences = reasoning.split(". ")
            for sentence in sentences:
                if "both" in sentence.lower() or "synthesis" in sentence.lower():
                    return sentence
        return reasoning[:200]  # Return first 200 chars if no clear synthesis

    def _fallback_dialectical_analysis(self, conflicts: List[str], user_input: str) -> Dict[str, Any]:
        """Fallback dialectical conflict resolution"""

        if not conflicts:
            return {"synthesis": "No clear emotional conflicts detected"}

        # Simple synthesis for common conflicts
        syntheses = {
            "happy_sad": "Joy and sorrow can coexist - life contains both",
            "angry_loving": "Love and anger can both be present - caring deeply sometimes includes anger",
            "hopeful_doubtful": "Hope and doubt can balance each other - realistic optimism",
            "excited_anxious": "Excitement and anxiety often travel together - they're both about caring"
        }

        # Find matching synthesis
        conflict_key = "_".join(sorted(conflicts[:2]))  # Take first two conflicts
        synthesis = syntheses.get(conflict_key, f"Both {conflicts[0]} and {conflicts[1] if len(conflicts) > 1 else 'these feelings'} contain important truths")

        return {
            "response": f"I notice both {', '.join(conflicts)} present here. In dialectical thinking, {synthesis}.",
            "conflicts_resolved": conflicts,
            "synthesis": synthesis,
            "fallback": True
        }

    async def _reflective_thinking(self, user_input: str, personality_vector: Dict[str, float]) -> Dict[str, Any]:
        """Mirror user's internal experience back to them"""

        try:
            result = await self.thinking_client.call_tool("reflective_thinking", {
                "user_statement": user_input,
                "personality_context": personality_vector
            })
            return result

        except Exception as e:
            return self._fallback_reflective_response(user_input)

    def _fallback_reflective_response(self, user_input: str) -> Dict[str, Any]:
        """Fallback reflective mirroring"""

        # Identify potential underlying feelings
        reflection_prompts = {
            "tired": "You sound like you're carrying a heavy load right now",
            "overwhelmed": "This seems like a lot to process all at once",
            "confused": "Things feel unclear and you're searching for direction",
            "angry": "There's strong energy here that needs expression",
            "sad": "There's deep feeling beneath the surface that wants acknowledgment",
            "hopeful": "I can sense optimism and possibility in your words",
            "scared": "There's vulnerability here that takes courage to share"
        }

        reflection = "I'm hearing you express"
        for emotion, mirror in reflection_prompts.items():
            if emotion.lower() in user_input.lower():
                reflection = mirror
                break

        if reflection == "I'm hearing you express":
            reflection = "I hear the complexity of what you're experiencing"

        return {
            "response": f"What I'm hearing is: {reflection}. Does that resonate with your experience?",
            "reflection_type": "emotional_mirroring",
            "fallback": True
        }

    async def _metacognitive_thinking(self, reasoning_process: str, user_patterns: List[str]) -> Dict[str, Any]:
        """Help users understand their own thinking patterns"""

        try:
            result = await self.thinking_client.call_tool("metacognitive_thinking", {
                "thought_process": reasoning_process,
                "user_patterns": user_patterns
            })
            return result

        except Exception as e:
            return self._fallback_metacognitive_analysis(reasoning_process, user_patterns)

    def _fallback_metacognitive_analysis(self, reasoning_process: str, user_patterns: List[str]) -> Dict[str, Any]:
        """Fallback metacognitive pattern analysis"""

        insights = []

        # Analyze thinking patterns
        if re.search(r'\b(always|never|everyone|no one|every time)\b', reasoning_process, re.IGNORECASE):
            insights.append("I notice some all-or-nothing thinking patterns emerging")

        if len(user_patterns) > 2:
            insights.append("You seem to return to certain themes repeatedly - that's worth noticing")

        if "should" in reasoning_process.lower():
            insights.append("There are some 'should' statements that might be creating pressure")

        if not insights:
            insights.append("Your thinking shows flexibility and self-awareness")

        response = f"Looking at your thought patterns, {insights[0] if insights else 'you show thoughtful self-reflection'}."

        return {
            "insights": insights,
            "response": response,
            "patterns_identified": user_patterns,
            "fallback": True
        }

    async def _creative_thinking(self, user_input: str, personality_vector: Dict[str, float]) -> Dict[str, Any]:
        """Generate creative coping strategies based on personality"""

        try:
            result = await self.thinking_client.call_tool("creative_thinking", {
                "user_input": user_input,
                "personality_vector": personality_vector
            })
            return result

        except Exception as e:
            return self._fallback_creative_strategies(user_input, personality_vector)

    def _fallback_creative_strategies(self, user_input: str, personality_vector: Dict[str, float]) -> Dict[str, Any]:
        """Fallback creative coping strategy generation"""

        ma_weight = personality_vector.get("Ma", 0.3)      # Innovation
        lagom_weight = personality_vector.get("Lagom", 0.05)  # Balance
        jeong_weight = personality_vector.get("Jeong", 0.15)   # Connection

        strategies = []

        if ma_weight > 0.3:
            strategies.extend([
                "Try approaching this challenge from a completely new angle - what would an alien think?",
                "Invent a creative metaphor for what you're feeling",
                "Design a 'feeling map' showing where different emotions live in your body"
            ])

        if lagom_weight > 0.1:
            strategies.extend([
                "Find the sustainable middle path - what 'enough' looks like right now",
                "Balance immediate comfort with long-term growth",
                "Practice 'micro-balances' throughout your day"
            ])

        if jeong_weight > 0.2:
            strategies.extend([
                "Reach out to someone you trust - shared humanity helps",
                "Write a letter to yourself from a compassionate friend",
                "Imagine holding space for someone you love going through this"
            ])

        # General strategies
        strategies.extend([
            "Try the 'container' technique - imagine putting these feelings in a box you can visit later",
            "Create a 'feeling collage' with images or words that represent what you're experiencing",
            "Use 'future self' perspective - what advice would your wiser future self give?"
        ])

        selected_strategy = strategies[0] if strategies else "Take a moment to breathe and notice what you need right now"

        return {
            "response": f"Here's a creative approach: {selected_strategy}",
            "strategies_generated": strategies,
            "pillar_influence": {
                "Ma": ma_weight,
                "Lagom": lagom_weight,
                "Jeong": jeong_weight
            },
            "fallback": True
        }

    def _has_emotional_conflicts(self, emotion_context: Dict[str, Any]) -> bool:
        """Check if user has conflicting emotions"""
        conflicts = emotion_context.get("conflicts", [])
        return len(conflicts) > 0

    def _synthesize_response(self, empathetic: Dict, dialectical: Optional[Dict],
                           reflective: Dict, metacognitive: Dict,
                           creative: Optional[Dict]) -> Dict[str, Any]:
        """Synthesize comprehensive response from all thinking modes"""

        modes_used = ["empathetic"]
        response_parts = [empathetic["response"]]
        follow_ups = []

        # Add dialectical insights if conflicts exist
        if dialectical:
            modes_used.append("dialectical")
            response_parts.append(f" {dialectical['response']}")
            follow_ups.append("Explore the synthesis between these conflicting feelings")

        # Add reflective mirroring
        if reflective.get("reflection_type") == "emotional_mirroring":
            modes_used.append("reflective")
            response_parts.append(f" {reflective['response']}")

        # Add metacognitive insights
        if metacognitive.get("insights"):
            modes_used.append("metacognitive")
            response_parts.append(f" {metacognitive['response']}")

        # Add creative strategies for intense emotions
        if creative and len(modes_used) > 1:  # Only if using multiple modes
            modes_used.append("creative")
            response_parts.append(f" {creative['response']}")
            follow_ups.extend(["Try this creative approach", "See if this new perspective helps"])

        return {
            "text": " ".join(response_parts),
            "modes": modes_used,
            "follow_ups": follow_ups[:2]  # Limit to 2 suggestions
        }

    def _calculate_cognitive_depth(self, modes_used: List[str]) -> str:
        """Calculate cognitive depth based on thinking modes used"""

        depth_score = len(modes_used)

        if depth_score >= 4:
            return "deep_integration"
        elif depth_score >= 3:
            return "multi_perspective"
        elif depth_score >= 2:
            return "dual_awareness"
        else:
            return "focused_empathy"

class MockMCPClient:
    """Mock MCP client until real SDK is available"""

    def __init__(self, server_name: str):
        self.server_name = server_name

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock thinking tools - would be replaced with real MCP client"""

        if tool_name == "empathetic_thinking":
            # Simulate empathetic analysis
            emotion = params.get("emotion_state", {}).get("emotion", "unknown")
            intensity = params.get("emotion_state", {}).get("intensity", 0.5)

            responses = {
                "sad": "I can feel how heavy this sadness is for you",
                "anxious": "This anxiety must feel so overwhelming right now",
                "angry": "Your anger carries important information",
                "joyful": "Your joy is beautiful and deserves to be celebrated"
            }

            response = responses.get(emotion, "I hear the depth of what you're experiencing")

            return {
                "response": response,
                "reasoning": f"Empathetic response for {emotion} at intensity {intensity}",
                "pillar_weights": params.get("personality_context", {})
            }

        elif tool_name == "dialectical_thinking":
            conflicts = params.get("conflicting_emotions", [])
            if conflicts:
                return {
                    "response": f"I see both {conflicts[0]} and {conflicts[1]} present here. In dialectical thinking, these apparent opposites can lead to deeper understanding."
                }
            return {"response": "No clear emotional conflicts detected"}

        elif tool_name == "reflective_thinking":
            return {
                "response": "What I'm hearing is that you're searching for clarity in this moment. Does that resonate?",
                "reflection_type": "emotional_mirroring"
            }

        elif tool_name == "metacognitive_thinking":
            patterns = params.get("user_patterns", [])
            insights = []
            if patterns:
                insights.append("You return to certain themes - that's valuable self-awareness")
            else:
                insights.append("Your thinking shows thoughtful consideration")

            return {
                "insights": insights,
                "response": f"Looking at your thought patterns: {insights[0]}"
            }

        elif tool_name == "creative_thinking":
            return {
                "response": "Try viewing this from a new perspective - what would you advise a friend in this situation?",
                "strategies_generated": ["Perspective shift", "Friend advice technique"]
            }

        return {"mock": True, "tool": tool_name}

# Global instance
empathy_engine = EmpathicThinkingEngine()
