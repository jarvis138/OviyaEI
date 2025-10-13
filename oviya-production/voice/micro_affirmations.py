"""
Micro-Affirmations Module
Generates and manages verbal backchannels for natural conversation flow
"""

import random
from typing import Dict, List, Optional, Tuple
import torch
import numpy as np
from pathlib import Path


class MicroAffirmationGenerator:
    """Generates contextual micro-affirmations (verbal backchannels)"""
    
    # Backchannel types and their variations
    AFFIRMATIONS = {
        "acknowledgment": {
            "variants": ["mm-hmm", "uh-huh", "yeah", "right", "okay", "I see"],
            "emotions": ["neutral", "thoughtful", "encouraging"],
            "use_cases": ["listening", "understanding", "processing"]
        },
        "surprise": {
            "variants": ["oh", "oh wow", "really?", "seriously?", "no way"],
            "emotions": ["surprised", "curious", "intrigued"],
            "use_cases": ["new_information", "unexpected", "interesting"]
        },
        "empathy": {
            "variants": ["oh no", "I'm sorry", "that's tough", "I understand"],
            "emotions": ["empathetic_sad", "comforting", "concerned"],
            "use_cases": ["sad_content", "difficulty", "struggle"]
        },
        "excitement": {
            "variants": ["wow!", "amazing!", "that's great!", "awesome!"],
            "emotions": ["joyful_excited", "proud", "enthusiastic"],
            "use_cases": ["achievement", "good_news", "success"]
        },
        "thinking": {
            "variants": ["hmm", "let me see", "well", "so"],
            "emotions": ["thoughtful", "contemplative", "analytical"],
            "use_cases": ["processing", "considering", "analyzing"]
        },
        "agreement": {
            "variants": ["exactly", "absolutely", "yes", "that's right", "true"],
            "emotions": ["confident", "affirmative", "supportive"],
            "use_cases": ["agreement", "confirmation", "validation"]
        },
        "continuation": {
            "variants": ["go on", "and then?", "tell me more", "what happened?"],
            "emotions": ["curious", "engaged", "interested"],
            "use_cases": ["prompting", "encouraging_more", "active_listening"]
        }
    }
    
    # Timing parameters for backchannels
    TIMING_RULES = {
        "min_user_speaking_time": 3.0,  # seconds before first backchannel
        "backchannel_interval": 5.0,  # seconds between backchannels
        "pause_threshold": 1.5,  # seconds of pause to insert backchannel
        "max_backchannels_per_turn": 3  # limit per user turn
    }
    
    def __init__(self, sample_rate: int = 24000):
        """Initialize the micro-affirmation generator"""
        self.sample_rate = sample_rate
        self.last_backchannel_time = 0
        self.backchannels_in_turn = 0
        self.conversation_context = []
        
        # Audio generation parameters
        self.duration_ranges = {
            "short": (0.2, 0.4),  # mm-hmm, yeah
            "medium": (0.4, 0.8),  # I see, okay
            "long": (0.8, 1.5)  # that's interesting, tell me more
        }
    
    def select_backchannel(
        self,
        user_emotion: str = "neutral",
        content_type: str = "general",
        conversation_state: Dict = None
    ) -> Tuple[str, str, Dict]:
        """
        Select appropriate backchannel based on context
        
        Args:
            user_emotion: Detected emotion from user
            content_type: Type of content (sad, exciting, etc.)
            conversation_state: Current conversation state
            
        Returns:
            Tuple of (backchannel_text, backchannel_type, audio_params)
        """
        
        # Determine backchannel type based on context
        backchannel_type = self._determine_type(user_emotion, content_type)
        
        # Select variant
        variants = self.AFFIRMATIONS[backchannel_type]["variants"]
        selected = random.choice(variants)
        
        # Determine emotion for delivery
        emotions = self.AFFIRMATIONS[backchannel_type]["emotions"]
        emotion = random.choice(emotions)
        
        # Audio parameters
        audio_params = {
            "emotion": emotion,
            "intensity": 0.3 + random.random() * 0.4,  # 0.3-0.7
            "duration": self._get_duration(selected),
            "volume": 0.6 + random.random() * 0.2,  # 0.6-0.8 (quieter than main speech)
            "pitch_variation": random.uniform(0.9, 1.1)
        }
        
        return selected, backchannel_type, audio_params
    
    def _determine_type(self, user_emotion: str, content_type: str) -> str:
        """Determine appropriate backchannel type based on context"""
        
        # Map user emotions to backchannel types
        emotion_mapping = {
            "sad": "empathy",
            "angry": "acknowledgment",
            "happy": "excitement",
            "excited": "excitement",
            "worried": "empathy",
            "confused": "thinking",
            "neutral": "acknowledgment",
            "curious": "continuation"
        }
        
        # Content type overrides
        content_mapping = {
            "question": "thinking",
            "achievement": "excitement",
            "problem": "empathy",
            "story": "continuation",
            "opinion": "acknowledgment"
        }
        
        # Priority: content_type > user_emotion > default
        if content_type in content_mapping:
            return content_mapping[content_type]
        elif user_emotion in emotion_mapping:
            return emotion_mapping[user_emotion]
        else:
            return "acknowledgment"
    
    def _get_duration(self, text: str) -> float:
        """Get appropriate duration for backchannel"""
        word_count = len(text.split())
        
        if word_count <= 1:  # mm-hmm, yeah
            return random.uniform(*self.duration_ranges["short"])
        elif word_count <= 3:  # I see, that's right
            return random.uniform(*self.duration_ranges["medium"])
        else:  # longer phrases
            return random.uniform(*self.duration_ranges["long"])
    
    def should_insert_backchannel(
        self,
        user_speaking_time: float,
        time_since_last: float,
        detected_pause: bool
    ) -> bool:
        """
        Determine if a backchannel should be inserted
        
        Args:
            user_speaking_time: How long user has been speaking
            time_since_last: Time since last backchannel
            detected_pause: Whether a pause was detected
            
        Returns:
            Boolean indicating whether to insert backchannel
        """
        
        # Check if we've exceeded max backchannels
        if self.backchannels_in_turn >= self.TIMING_RULES["max_backchannels_per_turn"]:
            return False
        
        # User must speak minimum time first
        if user_speaking_time < self.TIMING_RULES["min_user_speaking_time"]:
            return False
        
        # Check interval since last backchannel
        if time_since_last < self.TIMING_RULES["backchannel_interval"]:
            return False
        
        # If pause detected, higher chance
        if detected_pause:
            return random.random() < 0.8  # 80% chance on pause
        
        # Random chance based on speaking time
        if user_speaking_time > 10.0:  # Long monologue
            return random.random() < 0.6  # 60% chance
        elif user_speaking_time > 5.0:
            return random.random() < 0.3  # 30% chance
        
        return False
    
    def generate_audio(
        self,
        text: str,
        emotion: str,
        params: Dict
    ) -> torch.Tensor:
        """
        Generate audio for backchannel (synthetic for now)
        
        Args:
            text: Backchannel text
            emotion: Emotion for delivery
            params: Audio parameters
            
        Returns:
            Audio tensor
        """
        
        # For now, generate synthetic placeholder
        # In production, this would call CSM or TTS with the backchannel text
        
        duration = params.get("duration", 0.5)
        samples = int(duration * self.sample_rate)
        
        # Generate base tone (placeholder)
        t = torch.linspace(0, duration, samples)
        
        # Different patterns for different backchannels
        if text in ["mm-hmm", "uh-huh"]:
            # Two-tone pattern
            freq1, freq2 = 150, 180
            audio = torch.sin(2 * np.pi * freq1 * t[:samples//2])
            audio = torch.cat([audio, torch.sin(2 * np.pi * freq2 * t[:samples//2])])
            
        elif text in ["hmm", "well"]:
            # Single falling tone
            freq = 200 * torch.exp(-t * 0.5)  # Exponential decay
            audio = torch.sin(2 * np.pi * freq * t)
            
        else:
            # Generic tone
            freq = 170
            audio = torch.sin(2 * np.pi * freq * t)
        
        # Apply envelope
        envelope = torch.exp(-t * 2) * 0.3  # Exponential decay
        audio = audio * envelope
        
        # Apply volume
        audio = audio * params.get("volume", 0.7)
        
        # Add slight noise for naturalness
        noise = torch.randn_like(audio) * 0.01
        audio = audio + noise
        
        return audio
    
    def reset_turn(self):
        """Reset backchannel counter for new turn"""
        self.backchannels_in_turn = 0
        self.last_backchannel_time = 0


class ConversationalDynamics:
    """Manages conversational dynamics and turn-taking"""
    
    def __init__(self):
        """Initialize conversation dynamics tracker"""
        self.user_speaking_time = 0
        self.oviya_speaking_time = 0
        self.last_speaker = None
        self.turn_count = 0
        self.interruption_threshold = 3.0  # seconds
        
        # Timing adjustments based on user state
        self.response_timings = {
            "anxious": 0.3,  # 300ms - quick response
            "sad": 0.6,      # 600ms - gentle pause
            "excited": 0.4,  # 400ms - match energy
            "thoughtful": 1.0,  # 1s - give space
            "angry": 0.8,    # 800ms - careful pause
            "neutral": 0.7   # 700ms - normal
        }
        
        # Track conversation flow
        self.conversation_pace = 1.0  # multiplier
        self.energy_level = 0.5  # 0-1
        
    def update_speaking_time(self, speaker: str, duration: float):
        """Update speaking time for speaker"""
        if speaker == "user":
            self.user_speaking_time += duration
        else:
            self.oviya_speaking_time += duration
        
        self.last_speaker = speaker
    
    def should_backchannel(self) -> bool:
        """Determine if Oviya should provide backchannel"""
        return self.user_speaking_time > self.interruption_threshold
    
    def get_response_timing(self, user_emotion: str = "neutral") -> float:
        """Get appropriate response timing based on user emotion"""
        base_timing = self.response_timings.get(user_emotion, 0.7)
        
        # Adjust based on conversation pace
        adjusted = base_timing * self.conversation_pace
        
        # Add slight randomness for naturalness
        jitter = random.uniform(-0.1, 0.1)
        
        return max(0.2, adjusted + jitter)  # Minimum 200ms
    
    def adjust_conversation_pace(self, user_response_time: float):
        """Adjust conversation pace based on user's response time"""
        if user_response_time < 0.5:  # Fast responses
            self.conversation_pace *= 0.95  # Speed up slightly
        elif user_response_time > 2.0:  # Slow responses
            self.conversation_pace *= 1.05  # Slow down slightly
        
        # Keep within bounds
        self.conversation_pace = max(0.7, min(1.3, self.conversation_pace))
    
    def reset_turn(self):
        """Reset for new conversation turn"""
        if self.last_speaker == "user":
            self.turn_count += 1
        
        # Decay speaking times
        self.user_speaking_time *= 0.5
        self.oviya_speaking_time *= 0.5


def test_micro_affirmations():
    """Test the micro-affirmation system"""
    
    generator = MicroAffirmationGenerator()
    dynamics = ConversationalDynamics()
    
    print("🗣️ Testing Micro-Affirmations System\n")
    print("=" * 60)
    
    # Test different contexts
    test_contexts = [
        ("neutral", "general", "User is explaining something"),
        ("sad", "problem", "User shares difficult experience"),
        ("excited", "achievement", "User shares good news"),
        ("confused", "question", "User asks for clarification"),
        ("curious", "story", "User tells a story")
    ]
    
    for user_emotion, content_type, description in test_contexts:
        print(f"\nContext: {description}")
        print(f"Emotion: {user_emotion}, Content: {content_type}")
        
        # Get backchannel
        text, bc_type, params = generator.select_backchannel(
            user_emotion, content_type
        )
        
        print(f"Backchannel: \"{text}\" (type: {bc_type})")
        print(f"Delivery: {params['emotion']} @ {params['intensity']:.2f} intensity")
        print(f"Audio: {params['duration']:.2f}s, volume: {params['volume']:.2f}")
        
        # Test timing
        response_time = dynamics.get_response_timing(user_emotion)
        print(f"Response timing: {response_time:.2f}s")
        print("-" * 40)
    
    # Test timing decisions
    print("\n📊 Testing Timing Decisions")
    print("=" * 60)
    
    timing_tests = [
        (2.0, 0, False, "Early in conversation, no pause"),
        (4.0, 6.0, False, "Enough time, good interval"),
        (5.0, 6.0, True, "With detected pause"),
        (12.0, 8.0, False, "Long monologue")
    ]
    
    for user_time, time_since, pause, description in timing_tests:
        should_insert = generator.should_insert_backchannel(
            user_time, time_since, pause
        )
        print(f"{description}: {'YES' if should_insert else 'NO'}")


if __name__ == "__main__":
    test_micro_affirmations()
