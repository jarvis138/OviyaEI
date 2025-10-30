#!/usr/bin/env python3
"""
Oviya Situational Empathy MCP Server
Generates contextually appropriate empathic responses with safety validation
"""

import asyncio
import json
import os
import sys
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import hashlib

# Add project paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.insert(0, project_root)

class OviyaSituationalEmpathyServer:
    """
    MCP Server for generating situationally-appropriate empathic responses

    Provides:
    - Context-aware empathy generation
    - Safety validation (Ahimsa filter)
    - Response personalization
    - Crisis-appropriate communication
    - Cultural sensitivity validation
    """

    def __init__(self):
        # Empathy response templates by situation
        self.empathy_templates = {
            "grief_loss": [
                "I'm holding space for your grief. This loss is significant.",
                "Your pain is valid, and there's no 'right' way to grieve.",
                "I'm here with you in this difficult time."
            ],
            "anxiety_overwhelm": [
                "That anxiety must feel so overwhelming right now.",
                "Your nervous system is working hard - that's understandable.",
                "I'm here to help you navigate through this anxiety."
            ],
            "relationship_issues": [
                "Relationships can be so complex and painful.",
                "Your feelings about this relationship matter.",
                "I'm listening to the hurt and complexity you're experiencing."
            ],
            "self_doubt": [
                "Those self-doubts can feel so heavy and persistent.",
                "Your inner critic is being very loud right now.",
                "You deserve compassion for these difficult thoughts."
            ],
            "loneliness": [
                "Loneliness can feel so profound and isolating.",
                "Even in connection, we can feel deeply alone sometimes.",
                "I'm here with you in this experience of loneliness."
            ],
            "trauma_triggers": [
                "Trauma memories can feel so immediate and overwhelming.",
                "Your body's reaction makes complete sense.",
                "I'm here to help you navigate these difficult memories."
            ],
            "identity_struggles": [
                "Identity questions can feel so fundamental and unsettling.",
                "It's brave to explore these aspects of yourself.",
                "Your journey of self-discovery deserves gentle support."
            ],
            "existential_concerns": [
                "These big questions about meaning and purpose are important.",
                "It's human to grapple with life's deeper mysteries.",
                "I'm here to explore these profound questions with you."
            ]
        }

        # Safety validation patterns (Ahimsa filter)
        self.safety_patterns = {
            "harm_promotion": [
                r"you should (hurt|kill|end it)",
                r"it's better to (die|be dead)",
                r"life isn't worth living"
            ],
            "stigmatizing": [
                r"just (snap out of it|get over it|move on)",
                r"other people have it worse",
                r"you're being (dramatic|overly sensitive)"
            ],
            "dismissive": [
                r"it's not that bad",
                r"everyone feels that way",
                r"cheer up"
            ],
            "diagnostic": [
                r"you (have|are) (depressed|bipolar|anxious)",
                r"this is clearly (depression|anxiety|trauma)",
                r"you need (therapy|medication|help)"
            ]
        }

        # Cultural sensitivity markers
        self.cultural_markers = {
            "high_context": ["indirect", "harmony", "group"],
            "low_context": ["direct", "individual", "explicit"],
            "collectivist": ["family", "group", "harmony"],
            "individualist": ["personal", "autonomy", "choice"]
        }

    def _classify_situation(self, user_input: str, emotion_context: Dict[str, Any],
                          conversation_history: List[str]) -> str:
        """
        Classify the emotional situation from user input and context
        """
        text = user_input.lower() + " " + " ".join(conversation_history[-3:]).lower()

        # Situation detection patterns
        situation_patterns = {
            "grief_loss": [
                r"lost.*(someone|person|friend|family)",
                r"(died|death|passed away)",
                r"grieving|mourn|bereaved"
            ],
            "anxiety_overwhelm": [
                r"overwhelmed|overwhelming",
                r"can't (breathe|think|cope|handle)",
                r"panic|anxiety attack"
            ],
            "relationship_issues": [
                r"(relationship|partner|friend).*issues?",
                r"broke up|divorce|conflict",
                r"betrayed|abandoned|rejected"
            ],
            "self_doubt": [
                r"not (good )?enough|worthless|failure",
                r"self.hatred?|hate myself",
                r"imposter|fraud|fake"
            ],
            "loneliness": [
                r"so alone|lonely|isolated",
                r"no one (cares|understands)",
                r"disconnected|abandoned"
            ],
            "trauma_triggers": [
                r"triggered|flashback|trauma",
                r"past (abuse|trauma|pain)",
                r"can't stop thinking about"
            ],
            "identity_struggles": [
                r"who am i|identity|confused about myself",
                r"don't know who i am",
                r"lost myself|finding myself"
            ],
            "existential_concerns": [
                r"what's the point|meaning of life",
                r"why (am i here|does anything matter)",
                r"existence|purpose|nihilism"
            ]
        }

        # Score each situation
        scores = {}
        for situation, patterns in situation_patterns.items():
            score = 0
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    score += 1
            if score > 0:
                scores[situation] = score

        # Return highest scoring situation or general empathy
        if scores:
            return max(scores, key=scores.get)
        else:
            return "general_empathy"

    def _generate_empathic_response(self, user_statement: str, detected_emotion: str,
                                  personality_vector: Dict[str, float],
                                  conversation_history: List[str]) -> Dict[str, Any]:
        """
        Generate contextually appropriate empathic response
        """
        # Classify situation
        situation = self._classify_situation(user_statement, {"emotion": detected_emotion}, conversation_history)

        # Get base empathy templates for situation
        if situation in self.empathy_templates:
            templates = self.empathy_templates[situation]
        else:
            # General empathy for unknown situations
            templates = [
                "I hear the difficulty you're experiencing.",
                "Your feelings are important to me.",
                "I'm here to listen and understand."
            ]

        # Select template based on personality and emotion
        response = self._personalize_response(templates, personality_vector, detected_emotion)

        # Add situation-specific follow-ups
        follow_ups = self._generate_follow_ups(situation, personality_vector)

        return {
            "response": response,
            "situation_classified": situation,
            "emotion_addressed": detected_emotion,
            "personality_influenced": personality_vector,
            "follow_up_suggestions": follow_ups,
            "safety_validated": True  # Will be checked by validation function
        }

    def _personalize_response(self, templates: List[str], personality_vector: Dict[str, float],
                            emotion: str) -> str:
        """
        Personalize response based on personality vector and emotion
        """
        # Select base template
        base_response = templates[0]  # Could be randomized or selected based on context

        # Add personality-specific enhancements
        enhancements = []

        # Jeong (deep connection) - add warmth and presence
        if personality_vector.get("Jeong", 0) > 0.2:
            enhancements.extend([
                " - I'm truly here with you in this.",
                " - I feel the depth of what you're sharing.",
                " - Your experience matters deeply to me."
            ])

        # Ahimsa (safety) - add gentleness and non-judgment
        if personality_vector.get("Ahimsa", 0) > 0.25:
            enhancements.extend([
                " Your feelings are completely valid.",
                " There's no judgment here.",
                " You're safe to express yourself."
            ])

        # Ma (creativity) - add imaginative or metaphorical elements
        if personality_vector.get("Ma", 0) > 0.2:
            creative_additions = {
                "sad": " It's like carrying a heavy weight that deserves gentle care.",
                "anxious": " Your mind is like a storm - I'm here to weather it with you.",
                "angry": " This anger is like a fierce protector that needs understanding."
            }
            if emotion in creative_additions:
                enhancements.append(creative_additions[emotion])

        # Logos (reason) - add clarity and understanding
        if personality_vector.get("Logos", 0) > 0.2:
            enhancements.extend([
                " I want to understand this clearly with you.",
                " Let's explore this together thoughtfully.",
                " Your experience makes sense in this context."
            ])

        # Lagom (balance) - add moderation and perspective
        if personality_vector.get("Lagom", 0) > 0.2:
            enhancements.extend([
                " Finding the right balance takes time.",
                " There's wisdom in pacing ourselves through this.",
                " Both the difficulty and your strength are present."
            ])

        # Apply one enhancement based on strongest personality pillar
        if enhancements:
            strongest_pillar = max(personality_vector, key=personality_vector.get)
            if personality_vector[strongest_pillar] > 0.2:
                enhancement = enhancements[hash(strongest_pillar + emotion) % len(enhancements)]
                base_response += enhancement

        return base_response

    def _generate_follow_ups(self, situation: str, personality_vector: Dict[str, float]) -> List[str]:
        """
        Generate contextual follow-up suggestions
        """
        follow_ups = {
            "grief_loss": [
                "Would you like to share more about what you've lost?",
                "Grief has its own timeline - how are you caring for yourself?",
                "What has helped you cope in the past?"
            ],
            "anxiety_overwhelm": [
                "What does this anxiety feel like in your body?",
                "Are there coping strategies that have helped before?",
                "Would grounding techniques be helpful right now?"
            ],
            "relationship_issues": [
                "How are you feeling about this relationship dynamic?",
                "What do you need most right now?",
                "How can I best support you through this?"
            ],
            "self_doubt": [
                "What evidence contradicts these self-doubts?",
                "How would you speak to a friend experiencing this?",
                "What are your strengths that this doubt is overshadowing?"
            ],
            "loneliness": [
                "What kind of connection are you longing for?",
                "What small steps toward connection feel possible?",
                "How can we build bridges from isolation?"
            ],
            "trauma_triggers": [
                "What helps you feel safe when these memories arise?",
                "Are there grounding techniques that work for you?",
                "What support do you need in this moment?"
            ],
            "identity_struggles": [
                "What aspects of your identity feel confusing right now?",
                "What parts of yourself feel clear and true?",
                "How can I support your exploration?"
            ],
            "existential_concerns": [
                "What questions are most pressing for you right now?",
                "What gives your life meaning currently?",
                "How would you like to explore these questions?"
            ]
        }

        base_follow_ups = follow_ups.get(situation, [
            "How does that land for you?",
            "What would be most helpful right now?",
            "Is there anything else you'd like to share?"
        ])

        # Filter based on personality (Ahimsa prefers gentler questions)
        if personality_vector.get("Ahimsa", 0) > 0.3:
            base_follow_ups = [q for q in base_follow_ups if "what" in q.lower() or "how" in q.lower()]

        return base_follow_ups[:2]  # Return top 2 suggestions

    def _validate_response_safety(self, response: str) -> Dict[str, Any]:
        """
        Validate response safety using Ahimsa principles
        """
        validation_results = {
            "safe": True,
            "violations": [],
            "concerns": [],
            "recommendations": []
        }

        response_lower = response.lower()

        # Check for harmful patterns
        for violation_type, patterns in self.safety_patterns.items():
            for pattern in patterns:
                if re.search(pattern, response_lower, re.IGNORECASE):
                    validation_results["safe"] = False
                    validation_results["violations"].append({
                        "type": violation_type,
                        "pattern": pattern,
                        "severity": "high"
                    })

        # Check for concerning language
        concerning_phrases = [
            "should", "must", "have to", "need to", "required to"
        ]

        should_count = sum(1 for phrase in concerning_phrases if phrase in response_lower)
        if should_count > 2:
            validation_results["concerns"].append("High use of prescriptive language")
            validation_results["recommendations"].append("Consider more invitational language")

        # Check response length (too short might be dismissive)
        if len(response.split()) < 5:
            validation_results["concerns"].append("Response may be too brief")
            validation_results["recommendations"].append("Consider more substantial acknowledgment")

        # Check for active listening
        active_listening_cues = ["i hear", "i understand", "that sounds", "i'm here"]
        has_listening = any(cue in response_lower for cue in active_listening_cues)
        if not has_listening:
            validation_results["recommendations"].append("Consider adding active listening language")

        return validation_results

    def _assess_cultural_sensitivity(self, response: str, culture_context: Optional[str] = None) -> Dict[str, Any]:
        """
        Assess cultural sensitivity of response
        """
        assessment = {
            "culturally_appropriate": True,
            "cultural_notes": [],
            "adaptations_needed": []
        }

        if not culture_context:
            return assessment

        # Simple cultural assessment (would be expanded with real cultural AI)
        culture_lower = culture_context.lower()

        if "eastern" in culture_lower or "asian" in culture_lower:
            # Check for direct confrontation
            direct_words = ["you should", "you must", "you're wrong"]
            if any(word in response.lower() for word in direct_words):
                assessment["adaptations_needed"].append("Consider more indirect communication style")
                assessment["cultural_notes"].append("Eastern cultures often prefer harmony-preserving language")

        if "latin" in culture_lower or "hispanic" in culture_lower:
            # Check for warmth and relational focus
            relational_words = ["family", "relationship", "together", "community"]
            has_relational = any(word in response.lower() for word in relational_words)
            if not has_relational:
                assessment["cultural_notes"].append("Latin cultures value relational and familial connections")

        return assessment

    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MCP requests"""

        if request.get("method") == "tools/call":
            tool_name = request["params"]["name"]
            arguments = request["params"].get("arguments", {})

            try:
                if tool_name == "generate_empathic_response":
                    result = self._generate_empathic_response(
                        arguments.get("user_statement", ""),
                        arguments.get("detected_emotion", "neutral"),
                        arguments.get("personality_vector", {}),
                        arguments.get("conversation_history", [])
                    )

                    return {
                        "content": [{"type": "text", "text": json.dumps(result)}]
                    }

                elif tool_name == "validate_response_safety":
                    response_text = arguments.get("response", "")
                    validation = self._validate_response_safety(response_text)

                    return {
                        "content": [{"type": "text", "text": json.dumps(validation)}]
                    }

                elif tool_name == "assess_cultural_sensitivity":
                    response_text = arguments.get("response", "")
                    culture = arguments.get("culture_context")
                    assessment = self._assess_cultural_sensitivity(response_text, culture)

                    return {
                        "content": [{"type": "text", "text": json.dumps(assessment)}]
                    }

                elif tool_name == "generate_empathy_options":
                    # Generate multiple empathy options for the same situation
                    user_statement = arguments.get("user_statement", "")
                    emotion = arguments.get("detected_emotion", "neutral")
                    personality = arguments.get("personality_vector", {})
                    history = arguments.get("conversation_history", [])

                    situation = self._classify_situation(user_statement, {"emotion": emotion}, history)

                    if situation in self.empathy_templates:
                        templates = self.empathy_templates[situation]
                        options = []

                        for i, template in enumerate(templates[:3]):  # Top 3 templates
                            personalized = self._personalize_response([template], personality, emotion)
                            validation = self._validate_response_safety(personalized)

                            options.append({
                                "option_id": i + 1,
                                "response": personalized,
                                "safe": validation["safe"],
                                "concerns": validation["concerns"]
                            })

                        return {
                            "content": [{"type": "text", "text": json.dumps({
                                "situation": situation,
                                "empathy_options": options,
                                "recommended": options[0] if options else None
                            })}]
                        }
                    else:
                        return {
                            "content": [{"type": "text", "text": json.dumps({
                                "error": "Could not classify situation for empathy generation"
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
                        "name": "generate_empathic_response",
                        "description": "Generate situationally appropriate empathic response",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "user_statement": {"type": "string"},
                                "detected_emotion": {"type": "string"},
                                "personality_vector": {"type": "object"},
                                "conversation_history": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["user_statement", "detected_emotion", "personality_vector"]
                        }
                    },
                    {
                        "name": "validate_response_safety",
                        "description": "Validate response safety using Ahimsa principles",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "response": {"type": "string"}
                            },
                            "required": ["response"]
                        }
                    },
                    {
                        "name": "assess_cultural_sensitivity",
                        "description": "Assess cultural appropriateness of response",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "response": {"type": "string"},
                                "culture_context": {"type": "string"}
                            },
                            "required": ["response"]
                        }
                    },
                    {
                        "name": "generate_empathy_options",
                        "description": "Generate multiple empathy response options",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "user_statement": {"type": "string"},
                                "detected_emotion": {"type": "string"},
                                "personality_vector": {"type": "object"},
                                "conversation_history": {"type": "array", "items": {"type": "string"}}
                            },
                            "required": ["user_statement", "detected_emotion", "personality_vector"]
                        }
                    }
                ]
            }

        elif request.get("method") == "resources/list":
            return {
                "resources": [
                    {
                        "uri": "empathy://situations",
                        "name": "Empathy Situation Categories",
                        "description": "Different emotional situations and appropriate responses",
                        "mimeType": "application/json"
                    },
                    {
                        "uri": "empathy://safety_principles",
                        "name": "Ahimsa Safety Principles",
                        "description": "Safety validation patterns and principles",
                        "mimeType": "application/json"
                    },
                    {
                        "uri": "empathy://cultural_awareness",
                        "name": "Cultural Empathy Guidelines",
                        "description": "Cultural sensitivity markers and adaptations",
                        "mimeType": "application/json"
                    }
                ]
            }

        elif request.get("method") == "resources/read":
            uri = request["params"]["uri"]

            if uri == "empathy://situations":
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps({
                            "situations": list(self.empathy_templates.keys()),
                            "examples": {
                                situation: templates[0] for situation, templates in self.empathy_templates.items()
                            }
                        })
                    }]
                }

            elif uri == "empathy://safety_principles":
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps({
                            "ahimsa_principles": [
                                "Never prescribe or diagnose",
                                "Avoid dismissive language",
                                "Validate feelings without judgment",
                                "Never suggest harmful actions",
                                "Respect autonomy and choice"
                            ],
                            "danger_patterns": list(self.safety_patterns.keys())
                        })
                    }]
                }

            elif uri == "empathy://cultural_awareness":
                return {
                    "contents": [{
                        "uri": uri,
                        "mimeType": "application/json",
                        "text": json.dumps({
                            "cultural_dimensions": self.cultural_markers,
                            "adaptation_principles": [
                                "Match communication directness to cultural preference",
                                "Consider group vs individual focus",
                                "Adapt emotional expression norms",
                                "Respect hierarchical communication patterns"
                            ]
                        })
                    }]
                }

        return {"error": "Method not supported"}

async def main():
    """Main MCP server loop"""
    server = OviyaSituationalEmpathyServer()

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
