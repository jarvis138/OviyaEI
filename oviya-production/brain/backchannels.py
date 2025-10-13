"""
Backchannel/Micro-affirmation System for Oviya
Adds natural verbal and non-verbal responses during conversation
"""

import random
from typing import Dict, List, Optional, Tuple


class BackchannelSystem:
    """
    Manages micro-affirmations and backchannels for natural conversation flow.
    
    Backchannels are short verbal/non-verbal cues that show active listening:
    - "mm-hmm", "yeah", "uh-huh" (agreement)
    - "oh?", "really?" (curiosity)
    - "I see", "right" (understanding)
    - "oh no", "oh wow" (emotional resonance)
    """
    
    # Backchannel library organized by function
    BACKCHANNELS = {
        # Agreement & Understanding
        "agreement": [
            {"text": "mm-hmm", "prosody": "<gentle>mm-hmm</gentle>"},
            {"text": "yeah", "prosody": "<gentle>yeah</gentle>"},
            {"text": "uh-huh", "prosody": "<gentle>uh-huh</gentle>"},
            {"text": "right", "prosody": "right"},
            {"text": "I see", "prosody": "I see"},
            {"text": "okay", "prosody": "<gentle>okay</gentle>"}
        ],
        
        # Curiosity & Interest
        "curiosity": [
            {"text": "oh?", "prosody": "oh? <rising>"},
            {"text": "really?", "prosody": "really? <rising>"},
            {"text": "interesting", "prosody": "interesting"},
            {"text": "tell me more", "prosody": "tell me more"},
            {"text": "and then?", "prosody": "and then? <rising>"}
        ],
        
        # Emotional Resonance
        "positive_resonance": [
            {"text": "oh wow", "prosody": "oh wow <smile>"},
            {"text": "that's great", "prosody": "that's great <smile>"},
            {"text": "wonderful", "prosody": "wonderful <smile>"},
            {"text": "nice", "prosody": "nice <smile>"},
            {"text": "I'm glad", "prosody": "I'm glad <smile>"}
        ],
        
        "negative_resonance": [
            {"text": "oh no", "prosody": "<breath> oh no"},
            {"text": "I'm sorry", "prosody": "I'm sorry <gentle>"},
            {"text": "that's tough", "prosody": "that's tough"},
            {"text": "I hear you", "prosody": "I hear you"}
        ],
        
        # Thinking & Processing
        "thinking": [
            {"text": "hmm", "prosody": "hmm <pause>"},
            {"text": "let me think", "prosody": "let me think <pause>"},
            {"text": "well", "prosody": "well <pause>"},
            {"text": "you know", "prosody": "you know"}
        ],
        
        # Encouragement
        "encouragement": [
            {"text": "go on", "prosody": "go on"},
            {"text": "I'm listening", "prosody": "I'm listening"},
            {"text": "yes?", "prosody": "yes? <rising>"},
            {"text": "what happened?", "prosody": "what happened? <rising>"}
        ]
    }
    
    # Trigger conditions for each backchannel type
    TRIGGER_CONDITIONS = {
        "agreement": {
            "user_emotion": ["calm", "neutral", "confident"],
            "keywords": ["so", "and", "then", "because"],
            "probability": 0.3
        },
        "curiosity": {
            "user_emotion": ["excited", "happy", "surprised"],
            "keywords": ["guess what", "you won't believe", "something happened"],
            "probability": 0.4
        },
        "positive_resonance": {
            "user_emotion": ["happy", "excited", "joyful"],
            "keywords": ["great", "wonderful", "amazing", "excited", "happy"],
            "probability": 0.5
        },
        "negative_resonance": {
            "user_emotion": ["sad", "angry", "anxious", "stressed"],
            "keywords": ["sad", "upset", "worried", "stressed", "difficult"],
            "probability": 0.6
        },
        "thinking": {
            "user_emotion": ["thoughtful", "confused", "uncertain"],
            "keywords": ["maybe", "perhaps", "not sure", "don't know"],
            "probability": 0.35
        },
        "encouragement": {
            "user_emotion": ["hesitant", "uncertain", "shy"],
            "keywords": ["but", "well", "I guess", "kind of"],
            "probability": 0.4
        }
    }
    
    def __init__(self):
        self.backchannel_history = []  # Track recent backchannels
        self.max_history = 5
        self.cooldown_turns = 2  # Don't use same backchannel too often
    
    def should_inject_backchannel(
        self, 
        user_message: str,
        user_emotion: Optional[str] = None,
        conversation_length: int = 0
    ) -> Tuple[bool, Optional[str]]:
        """
        Decide if we should inject a backchannel.
        
        Args:
            user_message: User's input text
            user_emotion: Detected user emotion
            conversation_length: Number of turns in conversation
            
        Returns:
            (should_inject, backchannel_type)
        """
        
        # Don't inject on first turn
        if conversation_length < 1:
            return False, None
        
        # Don't inject too frequently (every 2-3 turns max)
        if len(self.backchannel_history) > 0:
            if conversation_length - self.backchannel_history[-1]["turn"] < self.cooldown_turns:
                return False, None
        
        # Check each backchannel type for triggers
        message_lower = user_message.lower()
        
        for bc_type, conditions in self.TRIGGER_CONDITIONS.items():
            # Check user emotion match
            emotion_match = False
            if user_emotion:
                for emotion_keyword in conditions["user_emotion"]:
                    if emotion_keyword in user_emotion.lower():
                        emotion_match = True
                        break
            
            # Check keyword match
            keyword_match = False
            for keyword in conditions["keywords"]:
                if keyword in message_lower:
                    keyword_match = True
                    break
            
            # Inject if either emotion or keyword matches (with probability)
            if (emotion_match or keyword_match):
                if random.random() < conditions["probability"]:
                    return True, bc_type
        
        return False, None
    
    def get_backchannel(self, bc_type: str) -> Dict[str, str]:
        """
        Get a backchannel of specified type.
        
        Args:
            bc_type: Type of backchannel to retrieve
            
        Returns:
            Dict with 'text' and 'prosody' keys
        """
        
        if bc_type not in self.BACKCHANNELS:
            bc_type = "agreement"  # Default
        
        # Get random backchannel from this type
        available = self.BACKCHANNELS[bc_type]
        
        # Avoid recently used backchannels
        recent_texts = [bc["text"] for bc in self.backchannel_history[-3:]]
        unused = [bc for bc in available if bc["text"] not in recent_texts]
        
        if not unused:
            unused = available  # Use any if all were recent
        
        backchannel = random.choice(unused)
        return backchannel
    
    def inject_backchannel(
        self,
        response_text: str,
        bc_type: str,
        position: str = "prefix"
    ) -> str:
        """
        Inject backchannel into response text.
        
        Args:
            response_text: Original response
            bc_type: Type of backchannel
            position: Where to inject ("prefix", "suffix", "standalone")
            
        Returns:
            Modified response with backchannel
        """
        
        backchannel = self.get_backchannel(bc_type)
        
        # Track in history
        self.backchannel_history.append({
            "text": backchannel["text"],
            "type": bc_type,
            "turn": len(self.backchannel_history)
        })
        
        # Keep history limited
        if len(self.backchannel_history) > self.max_history:
            self.backchannel_history.pop(0)
        
        # Inject based on position
        if position == "prefix":
            # Add backchannel before response
            return f"{backchannel['text']}, {response_text}"
        elif position == "suffix":
            # Add backchannel after response
            return f"{response_text} {backchannel['text']}"
        elif position == "standalone":
            # Backchannel only (for very short responses)
            return backchannel['text']
        else:
            return response_text
    
    def inject_into_prosodic_text(
        self,
        prosodic_text: str,
        bc_type: str,
        position: str = "prefix"
    ) -> str:
        """
        Inject backchannel into prosodic markup text.
        
        Args:
            prosodic_text: Text with prosodic markers
            bc_type: Type of backchannel
            position: Where to inject
            
        Returns:
            Modified prosodic text with backchannel
        """
        
        backchannel = self.get_backchannel(bc_type)
        prosodic_bc = backchannel["prosody"]
        
        # Track in history
        self.backchannel_history.append({
            "text": backchannel["text"],
            "type": bc_type,
            "turn": len(self.backchannel_history)
        })
        
        if len(self.backchannel_history) > self.max_history:
            self.backchannel_history.pop(0)
        
        # Inject with prosodic markers
        if position == "prefix":
            return f"{prosodic_bc} <pause> {prosodic_text}"
        elif position == "suffix":
            return f"{prosodic_text} <pause> {prosodic_bc}"
        elif position == "standalone":
            return prosodic_bc
        else:
            return prosodic_text
    
    def get_standalone_backchannel(self, user_emotion: str) -> Dict[str, str]:
        """
        Get a standalone backchannel response (for very short interactions).
        
        Args:
            user_emotion: User's current emotion
            
        Returns:
            Dict with text and prosodic_text
        """
        
        # Map emotion to backchannel type
        emotion_to_type = {
            "happy": "positive_resonance",
            "excited": "curiosity",
            "sad": "negative_resonance",
            "anxious": "negative_resonance",
            "stressed": "negative_resonance",
            "calm": "agreement",
            "neutral": "agreement"
        }
        
        bc_type = emotion_to_type.get(user_emotion, "agreement")
        backchannel = self.get_backchannel(bc_type)
        
        return {
            "text": backchannel["text"],
            "prosodic_text": backchannel["prosody"],
            "emotion": user_emotion,
            "intensity": 0.6,
            "is_backchannel": True
        }


# Example usage
if __name__ == "__main__":
    print("🎭 Testing Backchannel System\n")
    
    bc_system = BackchannelSystem()
    
    # Test scenarios
    test_scenarios = [
        {
            "user_message": "I'm feeling really stressed about work",
            "user_emotion": "stressed",
            "conversation_length": 2
        },
        {
            "user_message": "Guess what! I got promoted today!",
            "user_emotion": "excited",
            "conversation_length": 3
        },
        {
            "user_message": "I'm not sure what to do about this situation",
            "user_emotion": "uncertain",
            "conversation_length": 4
        },
        {
            "user_message": "So then I decided to talk to my manager",
            "user_emotion": "neutral",
            "conversation_length": 5
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"Scenario {i}:")
        print(f"  User: {scenario['user_message']}")
        print(f"  Emotion: {scenario['user_emotion']}")
        
        should_inject, bc_type = bc_system.should_inject_backchannel(
            scenario["user_message"],
            scenario["user_emotion"],
            scenario["conversation_length"]
        )
        
        if should_inject:
            backchannel = bc_system.get_backchannel(bc_type)
            print(f"  → Inject backchannel: '{backchannel['text']}' (type: {bc_type})")
            
            # Test injection
            original_response = "I understand how you feel."
            modified_response = bc_system.inject_backchannel(
                original_response,
                bc_type,
                position="prefix"
            )
            print(f"  → Modified response: '{modified_response}'")
        else:
            print(f"  → No backchannel injected")
        
        print()


