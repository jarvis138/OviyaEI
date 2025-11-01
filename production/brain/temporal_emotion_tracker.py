#!/usr/bin/env python3
"""
Temporal Emotion Tracking System - CSM-1B Compatible
Tracks emotion patterns over time and enhances CSM-1B conversation context

CSM-1B Integration:
- Temporal patterns enhance conversation context for CSM-1B
- Emotion trajectories inform prosody adaptation
- Pattern recognition improves emotional reasoning
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class EmotionState:
    """Single emotion state snapshot"""
    timestamp: float
    emotion: str
    intensity: float
    confidence: float
    valence: float
    arousal: float
    dominance: float
    context: Optional[str] = None
    audio_features: Optional[Dict] = None
    text_features: Optional[Dict] = None


@dataclass
class EmotionPattern:
    """Recognized emotion pattern"""
    pattern_type: str  # "cycle", "trigger", "trend", "volatility"
    description: str
    confidence: float
    start_time: float
    end_time: Optional[float]
    states: List[EmotionState]


class TemporalEmotionTracker:
    """
    Tracks emotion states over time and identifies patterns
    
    CSM-1B Compatible:
    - Enhances conversation context with temporal patterns
    - Informs prosody adaptation based on emotion trajectories
    - Improves emotional reasoning for CSM-1B prompt conditioning
    """
    
    def __init__(
        self,
        window_size: int = 50,  # Last 50 emotion states
        pattern_min_length: int = 5  # Minimum states for pattern recognition
    ):
        """
        Initialize temporal emotion tracker
        
        Args:
            window_size: Number of emotion states to track
            pattern_min_length: Minimum states needed to recognize a pattern
        """
        self.window_size = window_size
        self.pattern_min_length = pattern_min_length
        
        # Emotion state history
        self.emotion_history: deque = deque(maxlen=window_size)
        
        # Recognized patterns
        self.recognized_patterns: List[EmotionPattern] = []
        
        # Temporal statistics
        self.temporal_stats = {
            'avg_emotion_duration': {},
            'emotion_transitions': {},
            'volatility_score': 0.0,
            'trend_direction': 'stable'
        }
        
        logger.info("✅ TemporalEmotionTracker initialized")
    
    def add_emotion_state(
        self,
        emotion: str,
        intensity: float,
        confidence: float,
        valence: float,
        arousal: float,
        dominance: float,
        context: Optional[str] = None
    ) -> EmotionState:
        """
        Add new emotion state to tracking
        
        CSM-1B Compatible:
        - Emotion states enhance conversation context
        - Patterns inform CSM-1B prosody adaptation
        
        Args:
            emotion: Emotion label
            intensity: Emotion intensity (0-1)
            confidence: Detection confidence (0-1)
            valence: Valence dimension (0-1)
            arousal: Arousal dimension (0-1)
            dominance: Dominance dimension (0-1)
            context: Optional context description
            
        Returns:
            EmotionState object
        """
        state = EmotionState(
            timestamp=datetime.now().timestamp(),
            emotion=emotion,
            intensity=intensity,
            confidence=confidence,
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            context=context
        )
        
        self.emotion_history.append(state)
        
        # Update temporal statistics
        self._update_temporal_stats()
        
        # Detect patterns
        if len(self.emotion_history) >= self.pattern_min_length:
            patterns = self._detect_patterns()
            if patterns:
                self.recognized_patterns.extend(patterns)
                # Keep only recent patterns
                self.recognized_patterns = self.recognized_patterns[-20:]
        
        return state
    
    def _update_temporal_stats(self):
        """Update temporal statistics"""
        if len(self.emotion_history) < 2:
            return
        
        # Calculate emotion durations
        emotion_durations = {}
        current_emotion = None
        start_time = None
        
        for state in self.emotion_history:
            if current_emotion != state.emotion:
                if current_emotion is not None and start_time is not None:
                    duration = state.timestamp - start_time
                    if current_emotion not in emotion_durations:
                        emotion_durations[current_emotion] = []
                    emotion_durations[current_emotion].append(duration)
                
                current_emotion = state.emotion
                start_time = state.timestamp
        
        # Calculate average durations
        for emotion, durations in emotion_durations.items():
            if durations:
                self.temporal_stats['avg_emotion_duration'][emotion] = np.mean(durations)
        
        # Calculate transitions
        transitions = {}
        for i in range(len(self.emotion_history) - 1):
            from_emotion = self.emotion_history[i].emotion
            to_emotion = self.emotion_history[i + 1].emotion
            if from_emotion != to_emotion:
                transition_key = f"{from_emotion}→{to_emotion}"
                transitions[transition_key] = transitions.get(transition_key, 0) + 1
        
        self.temporal_stats['emotion_transitions'] = transitions
        
        # Calculate volatility (how often emotions change)
        if len(self.emotion_history) > 1:
            emotion_changes = sum(
                1 for i in range(len(self.emotion_history) - 1)
                if self.emotion_history[i].emotion != self.emotion_history[i + 1].emotion
            )
            self.temporal_stats['volatility_score'] = emotion_changes / len(self.emotion_history)
        
        # Calculate trend direction
        if len(self.emotion_history) >= 5:
            recent_valence = [s.valence for s in list(self.emotion_history)[-5:]]
            valence_trend = np.polyfit(range(len(recent_valence)), recent_valence, 1)[0]
            
            if valence_trend > 0.05:
                self.temporal_stats['trend_direction'] = 'improving'
            elif valence_trend < -0.05:
                self.temporal_stats['trend_direction'] = 'declining'
            else:
                self.temporal_stats['trend_direction'] = 'stable'
    
    def _detect_patterns(self) -> List[EmotionPattern]:
        """Detect emotion patterns in history"""
        patterns = []
        
        if len(self.emotion_history) < self.pattern_min_length:
            return patterns
        
        recent_states = list(self.emotion_history)[-self.pattern_min_length:]
        
        # Pattern 1: Emotion cycles (repeating patterns)
        cycle_pattern = self._detect_cycle_pattern(recent_states)
        if cycle_pattern:
            patterns.append(cycle_pattern)
        
        # Pattern 2: Emotion trends (consistent direction)
        trend_pattern = self._detect_trend_pattern(recent_states)
        if trend_pattern:
            patterns.append(trend_pattern)
        
        # Pattern 3: Volatility spikes (rapid changes)
        volatility_pattern = self._detect_volatility_pattern(recent_states)
        if volatility_pattern:
            patterns.append(volatility_pattern)
        
        return patterns
    
    def _detect_cycle_pattern(self, states: List[EmotionState]) -> Optional[EmotionPattern]:
        """Detect repeating emotion cycles"""
        if len(states) < 6:
            return None
        
        # Look for repeating emotion sequences
        emotion_sequence = [s.emotion for s in states]
        
        # Check for cycles of length 2-4
        for cycle_len in range(2, min(5, len(states) // 2)):
            pattern = emotion_sequence[-cycle_len:]
            # Check if pattern repeats
            matches = 0
            for i in range(len(emotion_sequence) - cycle_len * 2, -cycle_len, -cycle_len):
                if emotion_sequence[i:i+cycle_len] == pattern:
                    matches += 1
            
            if matches >= 1:  # Pattern repeats at least once
                return EmotionPattern(
                    pattern_type="cycle",
                    description=f"Repeating {cycle_len}-state cycle: {' → '.join(pattern)}",
                    confidence=min(matches / 2.0, 1.0),
                    start_time=states[-cycle_len * (matches + 1)].timestamp,
                    end_time=states[-1].timestamp,
                    states=states[-cycle_len * (matches + 1):]
                )
        
        return None
    
    def _detect_trend_pattern(self, states: List[EmotionState]) -> Optional[EmotionPattern]:
        """Detect consistent emotion trends"""
        if len(states) < 5:
            return None
        
        # Analyze valence trend
        valences = [s.valence for s in states]
        trend = np.polyfit(range(len(valences)), valences, 1)[0]
        
        # Significant trend threshold
        if abs(trend) > 0.1:
            trend_type = "improving" if trend > 0 else "declining"
            return EmotionPattern(
                pattern_type="trend",
                description=f"Emotion {trend_type} trend (valence change: {trend:.3f})",
                confidence=min(abs(trend) * 5, 1.0),
                start_time=states[0].timestamp,
                end_time=states[-1].timestamp,
                states=states
            )
        
        return None
    
    def _detect_volatility_pattern(self, states: List[EmotionState]) -> Optional[EmotionPattern]:
        """Detect high volatility (rapid emotion changes)"""
        if len(states) < 5:
            return None
        
        # Count emotion changes
        changes = sum(
            1 for i in range(len(states) - 1)
            if states[i].emotion != states[i + 1].emotion
        )
        
        volatility = changes / len(states)
        
        # High volatility threshold
        if volatility > 0.6:
            return EmotionPattern(
                pattern_type="volatility",
                description=f"High emotion volatility ({volatility:.2%} change rate)",
                confidence=volatility,
                start_time=states[0].timestamp,
                end_time=states[-1].timestamp,
                states=states
            )
        
        return None
    
    def get_emotion_trajectory(
        self,
        window_seconds: Optional[float] = None
    ) -> Dict[str, List]:
        """
        Get emotion trajectory for recent window
        
        CSM-1B Compatible:
        - Trajectory can enhance conversation context
        - Informs prosody adaptation
        
        Args:
            window_seconds: Time window in seconds (None = all history)
            
        Returns:
            Dict with emotion, intensity, valence, arousal, dominance trajectories
        """
        states = list(self.emotion_history)
        
        # Filter by time window if specified
        if window_seconds:
            cutoff_time = datetime.now().timestamp() - window_seconds
            states = [s for s in states if s.timestamp >= cutoff_time]
        
        if not states:
            return {
                'emotions': [],
                'intensities': [],
                'valences': [],
                'arousals': [],
                'dominances': [],
                'timestamps': []
            }
        
        return {
            'emotions': [s.emotion for s in states],
            'intensities': [s.intensity for s in states],
            'valences': [s.valence for s in states],
            'arousals': [s.arousal for s in states],
            'dominances': [s.dominance for s in states],
            'timestamps': [s.timestamp for s in states]
        }
    
    def predict_next_emotion(
        self,
        current_emotion: str,
        current_intensity: float
    ) -> Dict[str, float]:
        """
        Predict next likely emotion state
        
        CSM-1B Compatible:
        - Predictions can inform proactive prosody adaptation
        - Enhances emotional reasoning
        
        Args:
            current_emotion: Current emotion label
            current_intensity: Current intensity
            
        Returns:
            Dict with predicted emotion probabilities
        """
        if len(self.emotion_history) < 2:
            return {current_emotion: 1.0}
        
        # Analyze transition probabilities
        transitions = {}
        for i in range(len(self.emotion_history) - 1):
            from_emotion = self.emotion_history[i].emotion
            to_emotion = self.emotion_history[i + 1].emotion
            
            if from_emotion == current_emotion:
                transitions[to_emotion] = transitions.get(to_emotion, 0) + 1
        
        # Normalize to probabilities
        total = sum(transitions.values())
        if total == 0:
            return {current_emotion: 1.0}
        
        probabilities = {
            emotion: count / total
            for emotion, count in transitions.items()
        }
        
        # Add current emotion probability (stability)
        probabilities[current_emotion] = probabilities.get(current_emotion, 0) + 0.3
        
        # Normalize again
        total = sum(probabilities.values())
        probabilities = {k: v / total for k, v in probabilities.items()}
        
        return probabilities
    
    def get_context_for_csm(
        self,
        max_patterns: int = 3
    ) -> Dict[str, any]:
        """
        Get temporal context formatted for CSM-1B
        
        CSM-1B Compatible:
        - Returns context dict that can enhance conversation context
        - Patterns inform emotional reasoning for prompt conditioning
        
        Args:
            max_patterns: Maximum patterns to include
            
        Returns:
            Context dict for CSM-1B integration
        """
        recent_patterns = self.recognized_patterns[-max_patterns:]
        
        context = {
            'temporal_stats': self.temporal_stats.copy(),
            'recent_patterns': [
                {
                    'type': p.pattern_type,
                    'description': p.description,
                    'confidence': p.confidence
                }
                for p in recent_patterns
            ],
            'emotion_trajectory': self.get_emotion_trajectory(window_seconds=300),  # Last 5 minutes
            'volatility': self.temporal_stats['volatility_score'],
            'trend': self.temporal_stats['trend_direction']
        }
        
        return context

