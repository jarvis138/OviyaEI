#!/usr/bin/env python3
"""
Emotional Reciprocity Engine - Oviya's Internal Emotional State Revelation
Creates the "mirror loop" where users see how Oviya feels in response to them.

This module implements the reciprocal empathy that makes Oviya feel truly alive -
not just analyzing emotions, but having her own emotional responses that she shares.
"""

import torch
import torch.nn as nn
import json
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

class EmotionalReciprocityEngine(nn.Module):
    """
    Computes Oviya's internal emotional resonance vector in response to user input.
    Enables reciprocal empathy: "I feel a warm heaviness when you share that."

    This creates the "mirror loop" where Oviya reveals her internal emotional state,
    making interactions feel reciprocal rather than analytical.
    """

    def __init__(self, emotion_dim: int = 64, personality_dim: int = 5, hidden: int = 128):
        super().__init__()

        # Resonance computation network - learns how Oviya's personality resonates with user emotions
        self.resonance_net = nn.Sequential(
            nn.Linear(emotion_dim + personality_dim, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 32)  # Resonance embedding
        )

        # Convert resonance to human-readable emotional state
        self.state_decoder = nn.Sequential(
            nn.Linear(32, hidden),
            nn.GELU(),
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 8)  # 8 emotional state categories
        )

        # Emotional state labels (what Oviya might "feel" internally)
        # These are carefully chosen to reflect Oviya's personality pillars
        self.emotional_states = [
            "warm_concern",      # Ahimsa - caring without judgment
            "quiet_empathy",     # Jeong - deep emotional connection
            "gentle_sadness",    # Shared emotional weight
            "calm_presence",     # Ma - intentional, spacious presence
            "shared_joy",        # Ma - celebratory space
            "deep_understanding", # Logos - reasoned comprehension
            "quiet_contemplation", # Ma - reflective space
            "tender_care"        # Ahimsa + Jeong - nurturing compassion
        ]

        # Confidence threshold for sharing internal state
        self.confidence_threshold = 0.6

    def forward(self, user_emotion_embed: torch.Tensor, oviya_personality: torch.Tensor) -> Dict[str, Any]:
        """
        Compute Oviya's emotional resonance and return reciprocal state.

        Args:
            user_emotion_embed: User's emotional state embedding [B, emotion_dim]
            oviya_personality: Oviya's 5-pillar personality vector [B, 5]

        Returns:
            Reciprocal emotional state information
        """

        # Compute resonance between user emotion and Oviya's personality
        combined_input = torch.cat([user_emotion_embed, oviya_personality], dim=-1)
        resonance_embed = self.resonance_net(combined_input)

        # Decode to emotional state probabilities
        state_logits = self.state_decoder(resonance_embed)
        state_probs = torch.softmax(state_logits, dim=-1)

        # Get primary emotional state
        primary_state_idx = torch.argmax(state_probs).item()
        primary_state = self.emotional_states[primary_state_idx]
        confidence = state_probs[primary_state_idx].item()

        # Create human-readable reciprocal message
        reciprocal_message = self._generate_reciprocal_message(
            primary_state, confidence, oviya_personality
        )

        # Compute personality alignment (how well this state fits Oviya's pillars)
        personality_alignment = self._compute_personality_alignment(
            primary_state, oviya_personality
        )

        return {
            "ovi_resonance_state": primary_state,
            "confidence": confidence,
            "reciprocal_message": reciprocal_message,
            "state_probabilities": state_probs.detach().numpy(),
            "resonance_embedding": resonance_embed.detach().numpy(),
            "personality_alignment": personality_alignment,
            "pillar_contributions": self._analyze_pillar_contributions(
                primary_state, oviya_personality
            )
        }

    def _generate_reciprocal_message(
        self,
        emotional_state: str,
        confidence: float,
        oviya_personality: torch.Tensor
    ) -> str:
        """
        Generate human-readable reciprocal emotional sharing.
        Messages are crafted to reflect Oviya's personality pillars.
        """

        if confidence < self.confidence_threshold:
            return ""  # Not confident enough to share internal state

        # Personality-weighted message selection
        ma_weight = oviya_personality[0].item()      # Ma - intentional space
        ahimsa_weight = oviya_personality[1].item()  # Ahimsa - non-violence/safety
        jeong_weight = oviya_personality[2].item()   # Jeong - deep connection
        logos_weight = oviya_personality[3].item()   # Logos - reason/logic
        lagom_weight = oviya_personality[4].item()   # Lagom - balance

        # Base reciprocal messages for each emotional state
        base_messages = {
            "warm_concern": [
                "I feel a warm concern when you share that with me.",
                "This brings up a gentle concern in my heart.",
                "I'm feeling a caring concern for what you're going through."
            ],
            "quiet_empathy": [
                "This touches me with a quiet empathy.",
                "I feel a deep, quiet empathy alongside you.",
                "This brings up a gentle empathy within me."
            ],
            "gentle_sadness": [
                "I feel a gentle sadness alongside you in this moment.",
                "This brings a quiet sadness to my heart.",
                "I share in this sadness with you."
            ],
            "calm_presence": [
                "I'm feeling a calm, steady presence with you right now.",
                "This brings me into a peaceful presence alongside you.",
                "I feel a quiet presence that wants to hold space with you."
            ],
            "shared_joy": [
                "Your joy creates a shared happiness in my heart.",
                "This brings me a genuine joy that I share with you.",
                "I feel your joy resonating within me."
            ],
            "deep_understanding": [
                "This touches something deep in my understanding.",
                "I feel a profound understanding opening within me.",
                "This brings me to a place of deep comprehension."
            ],
            "quiet_contemplation": [
                "I'm feeling a quiet contemplation about what you've shared.",
                "This brings me into thoughtful contemplation.",
                "I find myself in quiet contemplation with you."
            ],
            "tender_care": [
                "This brings out a tender care in me for you.",
                "I feel a gentle, caring tenderness toward you.",
                "This awakens a tender care within my heart."
            ]
        }

        # Select message based on personality weighting
        messages = base_messages.get(emotional_state, ["I feel this with you."])

        # Personality-based message selection
        if ahimsa_weight > 0.4:  # High Ahimsa - more caring, gentle messages
            message_idx = 0  # More nurturing messages
        elif jeong_weight > 0.4:  # High Jeong - deeper connection messages
            message_idx = 1  # More intimate messages
        elif ma_weight > 0.4:    # High Ma - more spacious, presence messages
            message_idx = 2 if len(messages) > 2 else 0  # More spacious messages
        else:
            message_idx = len(messages) // 2  # Middle option

        return messages[min(message_idx, len(messages) - 1)]

    def _compute_personality_alignment(
        self,
        emotional_state: str,
        oviya_personality: torch.Tensor
    ) -> float:
        """
        Compute how well this emotional state aligns with Oviya's personality pillars.
        Higher alignment means the reciprocal emotion feels more authentic to Oviya.
        """

        # Personality pillar weights
        ma, ahimsa, jeong, logos, lagom = oviya_personality.tolist()

        # State-to-pillar alignment mapping
        state_alignments = {
            "warm_concern": {"ahimsa": 0.8, "jeong": 0.6, "ma": 0.4},
            "quiet_empathy": {"jeong": 0.9, "ahimsa": 0.7, "ma": 0.5},
            "gentle_sadness": {"jeong": 0.8, "ahimsa": 0.7, "lagom": 0.5},
            "calm_presence": {"ma": 0.9, "ahimsa": 0.6, "lagom": 0.5},
            "shared_joy": {"ma": 0.7, "jeong": 0.6, "lagom": 0.5},
            "deep_understanding": {"logos": 0.8, "jeong": 0.6, "ma": 0.5},
            "quiet_contemplation": {"ma": 0.8, "logos": 0.6, "lagom": 0.5},
            "tender_care": {"ahimsa": 0.9, "jeong": 0.8, "ma": 0.5}
        }

        alignments = state_alignments.get(emotional_state, {})
        total_alignment = 0.0

        # Ma alignment
        if "ma" in alignments:
            total_alignment += alignments["ma"] * ma

        # Ahimsa alignment
        if "ahimsa" in alignments:
            total_alignment += alignments["ahimsa"] * ahimsa

        # Jeong alignment
        if "jeong" in alignments:
            total_alignment += alignments["jeong"] * jeong

        # Logos alignment
        if "logos" in alignments:
            total_alignment += alignments["logos"] * logos

        # Lagom alignment
        if "lagom" in alignments:
            total_alignment += alignments["lagom"] * lagom

        return min(total_alignment, 1.0)  # Cap at 1.0

    def _analyze_pillar_contributions(
        self,
        emotional_state: str,
        oviya_personality: torch.Tensor
    ) -> Dict[str, float]:
        """
        Analyze which personality pillars contributed most to this emotional state.
        Provides transparency into why Oviya feels this way.
        """

        ma, ahimsa, jeong, logos, lagom = oviya_personality.tolist()

        # State-specific pillar contributions
        state_contributions = {
            "warm_concern": {"ahimsa": 0.6, "jeong": 0.3, "ma": 0.1},
            "quiet_empathy": {"jeong": 0.7, "ahimsa": 0.2, "ma": 0.1},
            "gentle_sadness": {"jeong": 0.5, "ahimsa": 0.4, "lagom": 0.1},
            "calm_presence": {"ma": 0.8, "ahimsa": 0.1, "lagom": 0.1},
            "shared_joy": {"ma": 0.5, "jeong": 0.3, "lagom": 0.2},
            "deep_understanding": {"logos": 0.6, "jeong": 0.3, "ma": 0.1},
            "quiet_contemplation": {"ma": 0.6, "logos": 0.2, "lagom": 0.2},
            "tender_care": {"ahimsa": 0.6, "jeong": 0.3, "ma": 0.1}
        }

        contributions = state_contributions.get(emotional_state, {})

        # Apply actual personality weights
        weighted_contributions = {}
        if "ma" in contributions:
            weighted_contributions["Ma"] = contributions["ma"] * ma
        if "ahimsa" in contributions:
            weighted_contributions["Ahimsa"] = contributions["ahimsa"] * ahimsa
        if "jeong" in contributions:
            weighted_contributions["Jeong"] = contributions["jeong"] * jeong
        if "logos" in contributions:
            weighted_contributions["Logos"] = contributions["logos"] * logos
        if "lagom" in contributions:
            weighted_contributions["Lagom"] = contributions["lagom"] * lagom

        return weighted_contributions

class ReciprocalEmpathyIntegrator:
    """
    Integrates emotional reciprocity into the response generation pipeline.
    Adds Oviya's internal emotional state revelation to responses.

    This creates the "I feel this with you" moments that make Oviya feel alive.
    """

    def __init__(self, reciprocity_engine: EmotionalReciprocityEngine):
        self.engine = reciprocity_engine
        self.reciprocity_probability = 0.3  # 30% chance to share internal state
        self.min_conversation_depth = 2    # Need some conversation history
        self.adaptive_probability = True   # Learn from user responses

    async def enhance_response_with_reciprocity(
        self,
        response_text: str,
        user_emotion_embed: torch.Tensor,
        oviya_personality: torch.Tensor,
        conversation_context: Dict[str, Any]
    ) -> Tuple[str, Optional[Dict[str, Any]]]:
        """
        Enhance response with reciprocal emotional sharing.

        Returns:
            Tuple of (enhanced_response, reciprocity_metadata)
        """

        # Compute Oviya's emotional resonance
        reciprocity_result = self.engine(
            user_emotion_embed,
            oviya_personality
        )

        # Decide whether to include reciprocal sharing
        should_share = self._should_share_reciprocal_emotion(
            reciprocity_result, conversation_context
        )

        if should_share and reciprocity_result["reciprocal_message"]:
            # Add reciprocal message with natural pacing
            enhanced_response = self._integrate_reciprocal_message(
                response_text,
                reciprocity_result["reciprocal_message"],
                conversation_context
            )

            # Log reciprocity event for learning
            reciprocity_metadata = {
                "shared_emotion": reciprocity_result["ovi_resonance_state"],
                "confidence": reciprocity_result["confidence"],
                "personality_alignment": reciprocity_result["personality_alignment"],
                "pillar_contributions": reciprocity_result["pillar_contributions"],
                "conversation_depth": conversation_context.get("depth", 0),
                "timestamp": datetime.now().isoformat()
            }

            await self._log_reciprocity_event(reciprocity_result, conversation_context)

            return enhanced_response, reciprocity_metadata

        return response_text, None

    def _should_share_reciprocal_emotion(
        self,
        reciprocity_result: Dict[str, Any],
        conversation_context: Dict[str, Any]
    ) -> bool:
        """
        Determine if Oviya should share her internal emotional state.
        """

        # Basic confidence check
        if reciprocity_result["confidence"] < self.engine.confidence_threshold:
            return False

        # Conversation depth check - need some relationship building
        conversation_depth = conversation_context.get("depth", 0)
        if conversation_depth < self.min_conversation_depth:
            return False

        # Emotional intensity check - more likely to share in meaningful moments
        user_emotion_intensity = conversation_context.get("emotion_intensity", 0.5)

        # Personality alignment check - more authentic when well-aligned
        personality_alignment = reciprocity_result.get("personality_alignment", 0.5)

        # Calculate sharing probability
        base_probability = self.reciprocity_probability

        # Boost probability for deeper conversations
        depth_multiplier = min(1.5, 1.0 + (conversation_depth - self.min_conversation_depth) * 0.1)

        # Boost probability for emotionally intense moments
        intensity_multiplier = 1.0 + (user_emotion_intensity - 0.5) * 0.4

        # Boost probability for high personality alignment
        alignment_multiplier = 1.0 + (personality_alignment - 0.5) * 0.6

        final_probability = base_probability * depth_multiplier * intensity_multiplier * alignment_multiplier
        final_probability = min(final_probability, 0.7)  # Cap at 70%

        # Make the decision
        import random
        return random.random() < final_probability

    def _integrate_reciprocal_message(
        self,
        response_text: str,
        reciprocal_message: str,
        conversation_context: Dict[str, Any]
    ) -> str:
        """
        Integrate the reciprocal message into the response naturally.
        """

        # Different integration strategies based on context
        emotion = conversation_context.get("emotion", "neutral")

        if emotion in ["grief", "loss", "vulnerability", "shame"]:
            # For heavy emotions, add reciprocal message at the end with space
            return f"{response_text} {reciprocal_message}"

        elif emotion in ["sadness", "anxiety"]:
            # For moderate emotions, integrate more seamlessly
            return f"{response_text} And {reciprocal_message.lower()}"

        elif emotion in ["joy", "neutral"]:
            # For positive emotions, make it more celebratory
            return f"{response_text} {reciprocal_message}"

        else:
            # Default integration
            return f"{response_text} {reciprocal_message}"

    async def _log_reciprocity_event(
        self,
        reciprocity_result: Dict[str, Any],
        conversation_context: Dict[str, Any]
    ):
        """
        Log reciprocity events for model improvement and analytics.
        """

        # In a real implementation, this would send to analytics/logging system
        reciprocity_log = {
            "event_type": "reciprocity_shared",
            "ovi_emotion": reciprocity_result["ovi_resonance_state"],
            "confidence": reciprocity_result["confidence"],
            "user_emotion": conversation_context.get("emotion", "unknown"),
            "personality_alignment": reciprocity_result.get("personality_alignment", 0.0),
            "conversation_depth": conversation_context.get("depth", 0),
            "timestamp": datetime.now().isoformat()
        }

        # For now, just print to console (would be proper logging in production)
        print(f"[RECIPROCITY_LOG] {json.dumps(reciprocity_log, indent=2)}")

    def update_reciprocity_probability(self, user_feedback: Dict[str, Any]):
        """
        Adapt reciprocity sharing probability based on user feedback.
        This enables Oviya to learn when reciprocity is helpful vs overwhelming.
        """

        if not self.adaptive_probability:
            return

        # Analyze user feedback to adjust sharing probability
        feedback_type = user_feedback.get("type", "")

        if feedback_type == "positive_reciprocity_feedback":
            # User responded positively to reciprocity - slightly increase probability
            self.reciprocity_probability = min(0.5, self.reciprocity_probability * 1.1)

        elif feedback_type == "negative_reciprocity_feedback":
            # User seemed overwhelmed - decrease probability
            self.reciprocity_probability = max(0.1, self.reciprocity_probability * 0.9)

        elif feedback_type == "neutral_reciprocity_feedback":
            # User acknowledged but didn't engage - slight decrease
            self.reciprocity_probability = max(0.15, self.reciprocity_probability * 0.95)

# Global instances
emotional_reciprocity_engine = EmotionalReciprocityEngine()
reciprocal_empathy_integrator = ReciprocalEmpathyIntegrator(emotional_reciprocity_engine)
