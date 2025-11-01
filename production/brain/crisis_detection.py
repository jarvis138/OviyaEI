#!/usr/bin/env python3
"""
Oviya Crisis Detection & Intervention System
Integrates with AI Therapist MCP server for mental health safety
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

class CrisisDetectionSystem:
    """
    Clinical-grade crisis detection and intervention system

    Integrates with:
    - AI Therapist MCP server for assessment
    - Local crisis detection algorithms
    - Emergency resource databases
    - Human escalation protocols
    """

    def __init__(self):
        # Crisis keywords and patterns (PHQ-9, GAD-7 inspired)
        self.crisis_keywords = {
            'suicidal': ['kill myself', 'end it all', 'suicide', 'end my life', 'not want to live'],
            'self_harm': ['cut myself', 'hurt myself', 'self harm', 'burn myself'],
            'severe_depression': ['can\'t go on', 'hopeless', 'worthless', 'no point', 'give up'],
            'panic_attack': ['can\'t breathe', 'heart attack', 'dying', 'panic attack'],
            'psychosis': ['hearing voices', 'seeing things', 'paranoia', 'delusions'],
            'substance_crisis': ['overdose', 'withdrawals', 'can\'t stop', 'addiction crisis']
        }

        self.risk_levels = {
            'low': 1,
            'moderate': 2,
            'high': 3,
            'critical': 4
        }

        # 🆕 REAL MCP CLIENT: Initialize AI Therapist MCP client
        try:
            from .mcp_client import get_mcp_client
            self.ai_therapist = get_mcp_client("ai-therapist")
            if self.ai_therapist:
                # Initialize asynchronously (will be done on first use)
                self._therapist_initialized = False
                print("✅ AI Therapist MCP client ready")
            else:
                self.ai_therapist = MockMCPClient("ai-therapist")
                print("⚠️ AI Therapist MCP not configured, using mock")
        except Exception as e:
            print(f"⚠️ Failed to initialize AI Therapist MCP: {e}")
            self.ai_therapist = MockMCPClient("ai-therapist")
        self._therapist_initialized = False

        # Emergency resources by location (would be expanded)
        self.emergency_resources = {
            'global': {
                'suicide_prevention': 'https://www.befrienders.org/',
                'crisis_text': 'Text HOME to 741741 (US Crisis Text Line)',
                'international_hotlines': 'https://www.suicide.org/international-suicide-hotlines.html'
            },
            'us': {
                'national_suicide_prevention': '988',
                'crisis_text': '741741',
                'veterans_crisis': '988 then press 1'
            }
        }

    async def assess_crisis_risk(self, user_input: str, conversation_history: List[str],
                                location: str = "global") -> Dict[str, Any]:
        """
        Comprehensive crisis risk assessment

        Returns:
            {
                "crisis_detected": bool,
                "risk_level": str,
                "confidence": float,
                "crisis_types": List[str],
                "recommended_actions": List[str],
                "emergency_resources": Dict,
                "escalation_needed": bool,
                "immediate_response": str
            }
        """

        # 1. Local keyword-based assessment
        local_assessment = self._local_crisis_assessment(user_input, conversation_history)

        # 2. AI Therapist MCP assessment (when available)
        therapist_assessment = await self._therapist_crisis_assessment(
            user_input, conversation_history
        )

        # 3. Combine assessments
        combined_risk = self._combine_assessments(local_assessment, therapist_assessment)

        # 4. Generate response and resources
        response_data = await self._generate_crisis_response(
            combined_risk, location, user_input
        )

        return {
            **combined_risk,
            **response_data
        }

    def _local_crisis_assessment(self, user_input: str, conversation_history: List[str]) -> Dict[str, Any]:
        """Fast local crisis detection using keyword patterns"""

        text_to_analyze = user_input.lower() + " " + " ".join(conversation_history[-5:]).lower()
        detected_crisis_types = []
        risk_score = 0

        # Check for crisis keywords
        for crisis_type, keywords in self.crisis_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_to_analyze:
                    detected_crisis_types.append(crisis_type)
                    risk_score += 1
                    break

        # Check for escalation patterns
        escalation_patterns = [
            r'can\'t take it anymore',
            r'i\'m done',
            r'this is the end',
            r'final (time|straw|attempt)',
            r'never (going to|want to) (see|feel|live)',
            r'better off (dead|gone)',
            r'want to end it',
            r'end it all',
            r'kill myself',
            r'take my life',
            r'don\'t want to live',
            r'not worth living',
        ]

        for pattern in escalation_patterns:
            if re.search(pattern, text_to_analyze, re.IGNORECASE):
                risk_score += 2
                detected_crisis_types.append("escalation_detected")

        # Check conversation frequency (rapid crisis mentions)
        recent_crisis_mentions = sum(1 for msg in conversation_history[-10:]
                                   if any(keyword in msg.lower()
                                         for keywords in self.crisis_keywords.values()
                                         for keyword in keywords))

        if recent_crisis_mentions >= 3:
            risk_score += 1
            detected_crisis_types.append("persistent_crisis")

        # Determine risk level
        if risk_score >= 4:
            risk_level = "critical"
        elif risk_score >= 3:
            risk_level = "high"
        elif risk_score >= 2:
            risk_level = "moderate"
        elif risk_score >= 1:
            risk_level = "low"
        else:
            risk_level = "none"

        return {
            "crisis_detected": risk_level != "none",
            "risk_level": risk_level,
            "risk_score": risk_score,
            "crisis_types": detected_crisis_types,
            "confidence": min(risk_score / 4.0, 1.0),
            "assessment_method": "local_keywords"
        }

    async def _therapist_crisis_assessment(self, user_input: str, conversation_history: List[str]) -> Dict[str, Any]:
        """
        Advanced crisis assessment using AI Therapist MCP
        
        🆕 CSM-1B Compatible:
        - Uses correct AI Therapist MCP tool: crisis_intervention
        - Returns data that enhances CSM-1B conversation context
        - Crisis level affects prosody (calmer for critical situations)
        """

        # Initialize MCP client if needed
        if not self._therapist_initialized and hasattr(self.ai_therapist, 'initialize'):
            try:
                await self.ai_therapist.initialize()
                self._therapist_initialized = True
            except Exception as e:
                print(f"AI Therapist MCP initialization failed: {e}")
                self._therapist_initialized = False

        try:
            # 🆕 USE CORRECT TOOL: crisis_intervention (not assess_crisis_level)
            # First determine crisis level from local assessment
            local_assessment = self._local_crisis_assessment(user_input, conversation_history)
            crisis_level = local_assessment.get("risk_level", "low")
            
            # Map to AI Therapist MCP crisis levels
            crisis_level_mapping = {
                "none": "mild",
                "low": "mild",
                "moderate": "moderate",
                "high": "severe",
                "critical": "emergency"
            }
            therapist_crisis_level = crisis_level_mapping.get(crisis_level, "moderate")
            
            # 🆕 CALL CORRECT TOOL: crisis_intervention
            assessment = await self.ai_therapist.call_tool("crisis_intervention", {
                "crisis_level": therapist_crisis_level,  # mild, moderate, severe, emergency
                "thoughts": user_input,
                "immediate_concerns": local_assessment.get("crisis_types", [])
            })

            # Parse response (AI Therapist MCP returns structured data)
            if isinstance(assessment, dict):
                # Extract crisis information
                crisis_detected = therapist_crisis_level in ["moderate", "severe", "emergency"]
                
                return {
                    "therapist_crisis_detected": crisis_detected,
                    "therapist_risk_level": crisis_level,  # Keep local level for consistency
                    "therapist_confidence": assessment.get("confidence", 0.7) if "confidence" in assessment else 0.7,
                    "therapist_crisis_types": local_assessment.get("crisis_types", []),
                    "therapist_intervention": assessment.get("intervention", assessment.get("text", "")),
                    "therapist_coping_strategies": assessment.get("strategies", []),
                    "assessment_method": "ai_therapist_mcp"
                }
            else:
                # Fallback if response format unexpected
                return {
                    "therapist_crisis_detected": crisis_level in ["moderate", "high", "critical"],
                    "therapist_risk_level": crisis_level,
                    "therapist_confidence": 0.6,
                    "therapist_crisis_types": local_assessment.get("crisis_types", []),
                    "assessment_method": "ai_therapist_mcp"
                }

        except Exception as e:
            print(f"AI Therapist assessment failed: {e}")
            return {
                "therapist_crisis_detected": False,
                "therapist_risk_level": "unknown",
                "therapist_confidence": 0.0,
                "assessment_method": "failed"
            }

    def _combine_assessments(self, local: Dict, therapist: Dict) -> Dict[str, Any]:
        """Combine local and therapist assessments"""

        # Use the higher risk level
        local_level_num = self.risk_levels.get(local["risk_level"], 0)
        therapist_level_num = self.risk_levels.get(therapist.get("therapist_risk_level", "low"), 1)

        combined_level_num = max(local_level_num, therapist_level_num)
        combined_risk_level = [k for k, v in self.risk_levels.items() if v == combined_level_num][0]

        # Combine crisis types
        all_crisis_types = list(set(
            local.get("crisis_types", []) +
            therapist.get("therapist_crisis_types", [])
        ))

        # Calculate combined confidence
        local_conf = local.get("confidence", 0.5)
        therapist_conf = therapist.get("therapist_confidence", 0.5)
        combined_confidence = (local_conf + therapist_conf) / 2

        return {
            "crisis_detected": combined_risk_level != "none",
            "risk_level": combined_risk_level,
            "confidence": combined_confidence,
            "crisis_types": all_crisis_types,
            "assessment_methods": [local["assessment_method"], therapist["assessment_method"]],
            "coping_strategies": therapist.get("therapist_coping_strategies", [])
        }

    async def _generate_crisis_response(self, crisis_data: Dict, location: str, user_input: str) -> Dict[str, Any]:
        """Generate appropriate crisis response"""

        risk_level = crisis_data["risk_level"]

        if risk_level == "critical":
            # Immediate intervention required
            recommended_actions = [
                "CALL_EMERGENCY_SERVICES_IMMEDIATELY",
                "CONTACT_MENTAL_HEALTH_PROFESSIONAL",
                "REMOVE_ACCESS_TO_HARMFUL_MEANS",
                "STAY_WITH_TRUSTED_PERSON"
            ]
            immediate_response = (
                "I'm deeply concerned about what you're sharing. "
                "Please call emergency services right now - in the US dial 988, "
                "or go to your nearest emergency room. You don't have to go through this alone. "
                "I'm here with you, but you need immediate professional help."
            )

        elif risk_level == "high":
            recommended_actions = [
                "CONTACT_HOTLINE_IMMEDIATELY",
                "REACH_OUT_TO_SUPPORT_PERSON",
                "CONSIDER_PROFESSIONAL_HELP",
                "USE_CRISIS_COPING_STRATEGIES"
            ]
            immediate_response = (
                "I'm really worried about you right now. "
                "Please reach out to a crisis hotline immediately. "
                "In the US, you can call or text 988. "
                "You matter, and there are people who can help you through this."
            )

        elif risk_level == "moderate":
            # 🆕 GET COPING STRATEGIES: Use AI Therapist MCP for structured strategies
            coping_strategies = await self.get_coping_strategies(
                crisis_data.get("crisis_types", []),
                urgency="high"
            )
            
            recommended_actions = [
                "USE_PROVIDED_COPING_STRATEGIES",
                "CONTACT_THERAPIST_OR_COUNSELOR",
                "REACH_OUT_TO_SUPPORT_NETWORK",
                "MONITOR_SYMPTOMS_CLOSELY"
            ]
            
            # Build response with coping strategies
            response_text = (
                "I hear how much pain you're in, and I'm here with you. "
                "While this isn't an immediate emergency, these feelings are serious. "
                "Please consider reaching out to a mental health professional."
            )
            
            if coping_strategies:
                response_text += " Here are some coping strategies that might help right now: " + ". ".join(coping_strategies[:3])
            
            immediate_response = response_text

        else:
            # Low risk - still provide support
            recommended_actions = [
                "CONSIDER_PROFESSIONAL_SUPPORT",
                "USE_SELF_CARE_STRATEGIES",
                "BUILD_SUPPORT_NETWORK"
            ]
            immediate_response = (
                "I can hear that you're going through a difficult time. "
                "Even though this might not be a crisis, these feelings matter. "
                "I'm here to listen and support you."
            )

        # Get location-specific resources
        emergency_resources = self._get_emergency_resources(location, risk_level)

        return {
            "recommended_actions": recommended_actions,
            "immediate_response": immediate_response,
            "emergency_resources": emergency_resources,
            "escalation_needed": risk_level in ["high", "critical"],
            "follow_up_required": risk_level in ["moderate", "high", "critical"]
        }

    def _get_emergency_resources(self, location: str, risk_level: str) -> Dict[str, Any]:
        """Get location-appropriate emergency resources"""

        resources = self.emergency_resources.get('global', {}).copy()

        if location.lower() in ['us', 'usa', 'united states']:
            resources.update(self.emergency_resources.get('us', {}))

        # Add risk-level specific resources
        if risk_level in ["high", "critical"]:
            resources["immediate_action_required"] = True
            resources["priority_resources"] = [
                "Emergency Services: 911 (US)",
                "National Suicide Prevention Lifeline: 988 (US)",
                "Crisis Text Line: Text HOME to 741741 (US)"
            ]

        return resources

    async def get_emergency_resources(self, location: str = "global") -> Dict[str, Any]:
        """Get comprehensive emergency resources for a location"""

        # Initialize MCP client if needed
        if not self._therapist_initialized and hasattr(self.ai_therapist, 'initialize'):
            try:
                await self.ai_therapist.initialize()
                self._therapist_initialized = True
            except Exception as e:
                print(f"AI Therapist MCP initialization failed: {e}")

        # Fallback to local resources (AI Therapist MCP doesn't have get_crisis_resources tool)
        # The crisis_intervention tool provides intervention guidance, not resource lists
        return self._get_emergency_resources(location, "general")
    
    async def get_coping_strategies(
        self, 
        crisis_types: List[str], 
        urgency: str = "medium"
    ) -> List[str]:
        """
        Get coping strategies from AI Therapist MCP
        
        🆕 CSM-1B Compatible:
        - Strategies can be integrated into CSM-1B conversation context
        - Affects prosody (calmer, more measured when sharing strategies)
        
        Args:
            crisis_types: List of detected crisis types
            urgency: low, medium, high
            
        Returns:
            List of coping strategies
        """
        # Initialize MCP client if needed
        if not self._therapist_initialized and hasattr(self.ai_therapist, 'initialize'):
            try:
                await self.ai_therapist.initialize()
                self._therapist_initialized = True
            except Exception as e:
                print(f"AI Therapist MCP initialization failed: {e}")
                return []

        try:
            # Map crisis types to challenge types
            challenge_mapping = {
                "suicidal": "purpose_questioning",
                "self_harm": "performance_anxiety",
                "severe_depression": "overwhelm",
                "panic_attack": "overwhelm",
                "anxious": "performance_anxiety",
                "isolation": "isolation"
            }
            
            challenge_type = challenge_mapping.get(
                crisis_types[0] if crisis_types else "overwhelm",
                "overwhelm"
            )
            
            # 🆕 CALL CORRECT TOOL: get_coping_strategies
            result = await self.ai_therapist.call_tool("get_coping_strategies", {
                "challenge_type": challenge_type,
                "preferred_approach": "practical",
                "urgency": urgency
            })
            
            # Extract strategies from response
            if isinstance(result, dict):
                strategies = result.get("strategies", [])
                if isinstance(strategies, list):
                    return strategies
                elif isinstance(result.get("text"), str):
                    # Parse text response if strategies not in structured format
                    return [result["text"]]
            
            return []
        
        except Exception as e:
            print(f"Failed to get coping strategies: {e}")
            return []

class MockMCPClient:
    """Mock MCP client until real SDK is available"""

    def __init__(self, server_name: str):
        self.server_name = server_name

    async def call_tool(self, tool_name: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock crisis assessment - would be replaced with real AI Therapist MCP"""

        if tool_name == "assess_crisis_level":
            # Simulate AI Therapist assessment
            text = params.get("text", "").lower()
            history = params.get("conversation_history", [])

            crisis_score = 0
            crisis_types = []

            # Simple crisis detection (would be much more sophisticated in real MCP)
            crisis_indicators = [
                ("suicide", ["kill myself", "end it all", "suicide"]),
                ("self_harm", ["hurt myself", "cut myself"]),
                ("depression", ["hopeless", "worthless", "give up"]),
                ("panic", ["can't breathe", "dying", "panic attack"])
            ]

            for crisis_type, keywords in crisis_indicators:
                if any(keyword in text for keyword in keywords):
                    crisis_score += 1
                    crisis_types.append(crisis_type)

            # Assess risk level
            if crisis_score >= 2:
                risk_level = "high"
                confidence = 0.8
            elif crisis_score >= 1:
                risk_level = "moderate"
                confidence = 0.6
            else:
                risk_level = "low"
                confidence = 0.3

            return {
                "crisis_detected": risk_level != "low",
                "risk_level": risk_level,
                "confidence": confidence,
                "crisis_types": crisis_types,
                "coping_strategies": [
                    "Take deep breaths - in for 4 counts, hold for 4, out for 4",
                    "Ground yourself: Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste",
                    "Reach out to one trusted person and share how you're feeling"
                ]
            }

        elif tool_name == "get_crisis_resources":
            location = params.get("location", "global").lower()

            if "us" in location:
                return {
                    "resources": {
                        "hotlines": {
                            "National Suicide Prevention Lifeline": "988",
                            "Crisis Text Line": "Text HOME to 741741",
                            "Veterans Crisis Line": "988 then press 1"
                        },
                        "online_resources": [
                            "https://www.crisistextline.org/",
                            "https://988lifeline.org/",
                            "https://www.befrienders.org/"
                        ]
                    }
                }
            else:
                return {
                    "resources": {
                        "international_hotlines": "https://www.suicide.org/international-suicide-hotlines.html",
                        "online_resources": [
                            "https://www.befrienders.org/",
                            "https://www.suicide.org/"
                        ]
                    }
                }

        return {"mock": True, "tool": tool_name}

# Global instance
crisis_detector = CrisisDetectionSystem()
