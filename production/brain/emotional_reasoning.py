#!/usr/bin/env python3
"""
Emotional Reasoning Engine - CSM-1B Compatible
Advanced emotional reasoning and inference for improved CSM-1B prompt conditioning

CSM-1B Integration:
- Emotional reasoning enhances CSM-1B prompt conditioning
- Cause-effect reasoning improves emotional context
- Goal inference informs therapeutic response generation
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import logging

logger = logging.getLogger(__name__)


class EmotionalReasoningEngine:
    """
    Advanced emotional reasoning and inference
    
    CSM-1B Compatible:
    - Reasoning results enhance CSM-1B prompt conditioning
    - Cause-effect chains improve emotional context
    - Goal inference informs therapeutic responses
    """
    
    def __init__(self):
        """Initialize emotional reasoning engine"""
        
        # Emotion cause-effect patterns
        self.cause_effect_patterns = {
            # Triggers -> Emotions
            "loss": ["sadness", "grief", "loneliness"],
            "rejection": ["sadness", "anger", "shame"],
            "failure": ["sadness", "shame", "fear"],
            "success": ["joy", "pride", "excitement"],
            "uncertainty": ["fear", "anxiety", "stress"],
            "conflict": ["anger", "frustration", "anxiety"],
            "isolation": ["loneliness", "sadness", "fear"],
            "validation": ["joy", "trust", "calm"],
            "criticism": ["anger", "sadness", "shame"],
            "support": ["joy", "trust", "calm"]
        }
        
        # Emotional goals (what emotions people seek)
        self.emotional_goals = {
            "sadness": ["comfort", "validation", "understanding"],
            "anger": ["validation", "justice", "acknowledgment"],
            "fear": ["safety", "reassurance", "control"],
            "anxiety": ["calm", "certainty", "support"],
            "joy": ["celebration", "sharing", "recognition"],
            "loneliness": ["connection", "belonging", "companionship"],
            "shame": ["acceptance", "forgiveness", "understanding"]
        }
        
        # Emotional needs (underlying needs behind emotions)
        self.emotional_needs = {
            "sadness": "need_for_comfort_and_validation",
            "anger": "need_for_acknowledgment_and_boundaries",
            "fear": "need_for_safety_and_reassurance",
            "anxiety": "need_for_certainty_and_control",
            "joy": "need_for_sharing_and_recognition",
            "loneliness": "need_for_connection_and_belonging",
            "shame": "need_for_acceptance_and_forgiveness"
        }
        
        logger.info("✅ EmotionalReasoningEngine initialized")
    
    def reason_emotional_cause(
        self,
        current_emotion: str,
        conversation_context: List[Dict],
        temporal_patterns: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Reason about emotional cause-effect
        
        CSM-1B Compatible:
        - Cause reasoning enhances prompt conditioning
        - Helps CSM-1B understand emotional context
        
        Args:
            current_emotion: Current detected emotion
            conversation_context: Recent conversation history
            temporal_patterns: Optional temporal emotion patterns
            
        Returns:
            Dict with cause analysis
        """
        # Analyze conversation for triggers
        triggers = self._identify_triggers(conversation_context)
        
        # Match triggers to emotions
        likely_causes = []
        for trigger, pattern_emotions in self.cause_effect_patterns.items():
            if trigger in triggers and current_emotion in pattern_emotions:
                likely_causes.append({
                    'trigger': trigger,
                    'emotion': current_emotion,
                    'confidence': 0.7,
                    'reasoning': f"{trigger} typically leads to {current_emotion}"
                })
        
        # Temporal analysis
        temporal_cause = None
        if temporal_patterns:
            trend = temporal_patterns.get('trend', 'stable')
            if trend == 'declining' and current_emotion in ['sadness', 'fear', 'anxiety']:
                temporal_cause = {
                    'type': 'cumulative_negative',
                    'confidence': 0.6,
                    'reasoning': 'Declining emotional trend suggests cumulative negative experiences'
                }
        
        return {
            'likely_causes': likely_causes,
            'temporal_cause': temporal_cause,
            'triggers_identified': triggers,
            'confidence': min(len(likely_causes) * 0.3 + 0.4, 1.0) if likely_causes else 0.3
        }
    
    def infer_emotional_goal(
        self,
        current_emotion: str,
        intensity: float,
        conversation_context: List[Dict]
    ) -> Dict[str, any]:
        """
        Infer emotional goal (what the user seeks emotionally)
        
        CSM-1B Compatible:
        - Goal inference informs therapeutic response generation
        - Helps CSM-1B understand what user needs
        
        Args:
            current_emotion: Current emotion
            intensity: Emotion intensity
            conversation_context: Conversation history
            
        Returns:
            Dict with goal inference
        """
        # Get emotional goals for this emotion
        goals = self.emotional_goals.get(current_emotion, ["understanding", "support"])
        
        # Get underlying need
        need = self.emotional_needs.get(current_emotion, "need_for_understanding")
        
        # Analyze conversation for explicit requests
        explicit_requests = self._identify_explicit_requests(conversation_context)
        
        # Prioritize goals based on intensity and context
        prioritized_goals = []
        for goal in goals:
            priority = 0.5  # Base priority
            
            # Higher intensity = higher priority
            priority += intensity * 0.3
            
            # Explicit requests boost priority
            if any(req in goal for req in explicit_requests):
                priority += 0.2
            
            prioritized_goals.append({
                'goal': goal,
                'priority': min(priority, 1.0)
            })
        
        # Sort by priority
        prioritized_goals.sort(key=lambda x: x['priority'], reverse=True)
        
        return {
            'primary_goal': prioritized_goals[0]['goal'] if prioritized_goals else "understanding",
            'all_goals': prioritized_goals,
            'underlying_need': need,
            'explicit_requests': explicit_requests,
            'confidence': min(intensity + 0.3, 1.0)
        }
    
    def predict_emotional_outcome(
        self,
        current_emotion: str,
        current_intensity: float,
        proposed_response: str,
        temporal_patterns: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Predict emotional outcome of proposed response
        
        CSM-1B Compatible:
        - Predictions can inform response generation
        - Helps CSM-1B choose appropriate emotional responses
        
        Args:
            current_emotion: Current emotion
            current_intensity: Current intensity
            proposed_response: Proposed response text
            temporal_patterns: Optional temporal patterns
            
        Returns:
            Dict with outcome prediction
        """
        # Analyze response for emotional cues
        response_emotion = self._analyze_response_emotion(proposed_response)
        
        # Predict transition
        transition_probability = self._predict_transition(
            current_emotion,
            response_emotion,
            temporal_patterns
        )
        
        # Predict intensity change
        intensity_change = self._predict_intensity_change(
            current_emotion,
            response_emotion,
            current_intensity
        )
        
        return {
            'predicted_emotion': response_emotion,
            'transition_probability': transition_probability,
            'predicted_intensity': max(0.0, min(1.0, current_intensity + intensity_change)),
            'likely_outcome': 'positive' if transition_probability > 0.6 else 'neutral',
            'confidence': transition_probability
        }
    
    def reason_emotional_chain(
        self,
        emotion_sequence: List[str],
        conversation_context: List[Dict]
    ) -> Dict[str, any]:
        """
        Reason about emotional chain (sequence of emotions)
        
        CSM-1B Compatible:
        - Chain reasoning enhances understanding of emotional journey
        - Informs CSM-1B about emotional progression
        
        Args:
            emotion_sequence: Sequence of emotions over time
            conversation_context: Conversation history
            
        Returns:
            Dict with chain analysis
        """
        if len(emotion_sequence) < 2:
            return {'chain_type': 'insufficient_data', 'confidence': 0.0}
        
        # Analyze chain patterns
        chain_type = None
        reasoning = ""
        
        # Check for healing trajectory (negative -> positive)
        if emotion_sequence[0] in ['sadness', 'fear', 'anxiety'] and \
           emotion_sequence[-1] in ['calm', 'joy', 'trust']:
            chain_type = 'healing_trajectory'
            reasoning = "Emotional progression from negative to positive suggests healing"
        
        # Check for escalation (neutral -> negative)
        elif emotion_sequence[0] in ['neutral', 'calm'] and \
             emotion_sequence[-1] in ['sadness', 'fear', 'anger']:
            chain_type = 'escalation'
            reasoning = "Emotional escalation detected - may need immediate support"
        
        # Check for stabilization (varied -> stable)
        elif len(set(emotion_sequence)) == 1:
            chain_type = 'stabilization'
            reasoning = "Emotional state stabilized - consistent support needed"
        
        # Check for volatility (rapid changes)
        elif len(set(emotion_sequence)) > len(emotion_sequence) * 0.7:
            chain_type = 'volatility'
            reasoning = "High emotional volatility - may need grounding support"
        
        else:
            chain_type = 'complex_pattern'
            reasoning = "Complex emotional pattern - requires nuanced understanding"
        
        return {
            'chain_type': chain_type,
            'reasoning': reasoning,
            'sequence': emotion_sequence,
            'confidence': min(len(emotion_sequence) / 10.0, 1.0)
        }
    
    def _identify_triggers(self, conversation_context: List[Dict]) -> List[str]:
        """Identify emotional triggers in conversation"""
        triggers = []
        
        # Analyze recent conversation turns
        recent_text = " ".join([
            turn.get('text', turn.get('q', turn.get('r', '')))
            for turn in conversation_context[-5:]
        ]).lower()
        
        # Check for trigger keywords
        trigger_keywords = {
            'loss': ['lost', 'died', 'death', 'gone', 'missing', 'passed away'],
            'rejection': ['rejected', 'refused', 'turned down', 'dismissed'],
            'failure': ['failed', 'failed', 'mistake', 'wrong', 'error'],
            'success': ['succeeded', 'achieved', 'completed', 'won', 'promoted'],
            'uncertainty': ['uncertain', 'unsure', 'don\'t know', 'confused', 'unclear'],
            'conflict': ['argued', 'disagreed', 'fight', 'conflict', 'dispute'],
            'isolation': ['alone', 'lonely', 'isolated', 'disconnected', 'separated'],
            'validation': ['validated', 'understood', 'heard', 'accepted'],
            'criticism': ['criticized', 'judged', 'blamed', 'attacked'],
            'support': ['supported', 'helped', 'cared for', 'there for me']
        }
        
        for trigger, keywords in trigger_keywords.items():
            if any(keyword in recent_text for keyword in keywords):
                triggers.append(trigger)
        
        return triggers
    
    def _identify_explicit_requests(self, conversation_context: List[Dict]) -> List[str]:
        """Identify explicit emotional requests"""
        requests = []
        
        recent_text = " ".join([
            turn.get('text', turn.get('q', turn.get('r', '')))
            for turn in conversation_context[-3:]
        ]).lower()
        
        request_patterns = {
            'comfort': ['comfort', 'soothe', 'make me feel better'],
            'validation': ['understand', 'validate', 'hear me', 'acknowledge'],
            'support': ['help', 'support', 'be there', 'guide'],
            'reassurance': ['reassure', 'tell me it\'s ok', 'everything will be fine'],
            'connection': ['connect', 'relate', 'share', 'together']
        }
        
        for request_type, patterns in request_patterns.items():
            if any(pattern in recent_text for pattern in patterns):
                requests.append(request_type)
        
        return requests
    
    def _analyze_response_emotion(self, response_text: str) -> str:
        """Analyze emotional tone of response"""
        text_lower = response_text.lower()
        
        # Emotional tone keywords
        emotion_keywords = {
            'calm': ['calm', 'peaceful', 'gentle', 'steady'],
            'supportive': ['support', 'here for you', 'with you', 'care'],
            'empathic': ['understand', 'feel', 'sense', 'hear'],
            'encouraging': ['can', 'will', 'strength', 'proud'],
            'validating': ['makes sense', 'valid', 'understandable', 'reasonable']
        }
        
        emotion_scores = {}
        for emotion, keywords in emotion_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                emotion_scores[emotion] = score
        
        if emotion_scores:
            return max(emotion_scores, key=emotion_scores.get)
        
        return 'neutral'
    
    def _predict_transition(
        self,
        current_emotion: str,
        response_emotion: str,
        temporal_patterns: Optional[Dict] = None
    ) -> float:
        """Predict probability of emotional transition"""
        # Base transition probability
        base_prob = 0.5
        
        # Compatible transitions are more likely
        compatible_transitions = {
            'sadness': ['calm', 'supportive', 'empathic'],
            'anger': ['calm', 'validating', 'supportive'],
            'fear': ['calm', 'reassuring', 'supportive'],
            'anxiety': ['calm', 'supportive', 'encouraging']
        }
        
        if current_emotion in compatible_transitions:
            if response_emotion in compatible_transitions[current_emotion]:
                base_prob = 0.8
        
        # Temporal patterns affect transition
        if temporal_patterns:
            trend = temporal_patterns.get('trend', 'stable')
            if trend == 'declining' and response_emotion == 'supportive':
                base_prob += 0.1
        
        return min(base_prob, 1.0)
    
    def _predict_intensity_change(
        self,
        current_emotion: str,
        response_emotion: str,
        current_intensity: float
    ) -> float:
        """Predict intensity change from response"""
        # Supportive responses typically reduce intensity
        if response_emotion in ['calm', 'supportive', 'empathic', 'validating']:
            return -0.2 * current_intensity  # Reduce intensity
        
        # Neutral responses maintain intensity
        elif response_emotion == 'neutral':
            return 0.0
        
        # Otherwise slight reduction
        return -0.1
    
    def get_reasoning_for_csm(
        self,
        current_emotion: str,
        intensity: float,
        conversation_context: List[Dict],
        temporal_patterns: Optional[Dict] = None
    ) -> Dict[str, any]:
        """
        Get comprehensive reasoning formatted for CSM-1B
        
        CSM-1B Compatible:
        - Returns reasoning dict that enhances prompt conditioning
        - Informs CSM-1B about emotional context and goals
        
        Args:
            current_emotion: Current emotion
            intensity: Emotion intensity
            conversation_context: Conversation history
            temporal_patterns: Optional temporal patterns
            
        Returns:
            Reasoning dict for CSM-1B integration
        """
        cause_analysis = self.reason_emotional_cause(
            current_emotion,
            conversation_context,
            temporal_patterns
        )
        
        goal_inference = self.infer_emotional_goal(
            current_emotion,
            intensity,
            conversation_context
        )
        
        return {
            'cause_analysis': cause_analysis,
            'goal_inference': goal_inference,
            'emotional_need': goal_inference.get('underlying_need', 'need_for_understanding'),
            'recommended_approach': self._recommend_approach(
                current_emotion,
                goal_inference.get('primary_goal', 'understanding')
            )
        }
    
    def _recommend_approach(
        self,
        emotion: str,
        goal: str
    ) -> str:
        """Recommend therapeutic approach based on emotion and goal"""
        approach_map = {
            ('sadness', 'comfort'): 'gentle_validation_and_comfort',
            ('anger', 'validation'): 'acknowledge_feelings_and_set_boundaries',
            ('fear', 'safety'): 'reassurance_and_grounding',
            ('anxiety', 'calm'): 'calming_presence_and_certainty',
            ('loneliness', 'connection'): 'empathetic_connection_and_presence'
        }
        
        return approach_map.get((emotion, goal), 'empathetic_listening_and_validation')

