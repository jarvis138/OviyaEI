"""
Oviya's Brain - LLM-based response generation

This module handles text generation using Qwen2.5:7B (via Ollama).
It produces both response text and emotion labels for the emotion controller.
"""

import json
import requests
from typing import Dict, Tuple, Optional
from pathlib import Path
import re
import random
from functools import lru_cache


class ProsodyMarkup:
    """Handles prosodic markup generation for natural speech"""
    
    # Cache for prosodic patterns
    _pattern_cache = {}
    
    # Emotion-based prosodic patterns
    EMOTION_PATTERNS = {
        # Tier 1: Core emotions
        "calm_supportive": {
            "breath_before": 0.3,  # 30% chance of breath before
            "breath_after": 0.2,
            "pause_multiplier": 1.2,  # Longer pauses
            "smile_markers": 0.1,
            "emphasis_style": "gentle"
        },
        "empathetic_sad": {
            "breath_before": 0.4,
            "breath_after": 0.3,
            "pause_multiplier": 1.5,  # Much longer pauses
            "smile_markers": 0.0,
            "emphasis_style": "soft"
        },
        "joyful_excited": {
            "breath_before": 0.1,
            "breath_after": 0.1,
            "pause_multiplier": 0.7,  # Faster speech
            "smile_markers": 0.6,  # Lots of smiles
            "emphasis_style": "bright"
        },
        "confident": {
            "breath_before": 0.2,
            "breath_after": 0.1,
            "pause_multiplier": 0.9,
            "smile_markers": 0.2,
            "emphasis_style": "strong"
        },
        "comforting": {
            "breath_before": 0.4,
            "breath_after": 0.3,
            "pause_multiplier": 1.3,
            "smile_markers": 0.3,
            "emphasis_style": "warm"
        },
        # Add patterns for other emotions as needed
        "default": {
            "breath_before": 0.2,
            "breath_after": 0.1,
            "pause_multiplier": 1.0,
            "smile_markers": 0.2,
            "emphasis_style": "neutral"
        }
    }
    
    @classmethod
    def get_cached_pattern(cls, emotion: str) -> Dict:
        """Get cached emotion pattern for performance"""
        if emotion not in cls._pattern_cache:
            cls._pattern_cache[emotion] = cls.EMOTION_PATTERNS.get(
                emotion, cls.EMOTION_PATTERNS["default"]
            )
        return cls._pattern_cache[emotion]
    
    @staticmethod
    def add_prosodic_markup(text: str, emotion: str, intensity: float = 0.5) -> str:
        """Add prosodic markers to text based on emotion"""
        
        # Get emotion pattern from cache (fallback to default)
        pattern = ProsodyMarkup.get_cached_pattern(emotion)
        
        # Scale probabilities by intensity
        breath_before_prob = pattern["breath_before"] * intensity
        breath_after_prob = pattern["breath_after"] * intensity
        smile_prob = pattern["smile_markers"] * intensity
        
        # Start with original text
        marked_text = text
        
        # Add breath at beginning if appropriate
        if random.random() < breath_before_prob:
            marked_text = f"<breath> {marked_text}"
        
        # Add smile markers to exclamations and positive words
        if smile_prob > 0:
            # Mark exclamations
            marked_text = re.sub(r'!(?=\s|$)', lambda m: '! <smile>' if random.random() < smile_prob else '!', marked_text)
            
            # Mark positive words
            positive_words = ['wonderful', 'amazing', 'great', 'fantastic', 'love', 'happy', 'excited']
            for word in positive_words:
                if word in marked_text.lower() and random.random() < smile_prob:
                    marked_text = re.sub(f'\\b{word}\\b', f'{word} <smile>', marked_text, flags=re.IGNORECASE)
        
        # Add pauses at sentence boundaries
        pause_mult = pattern["pause_multiplier"]
        if pause_mult != 1.0:
            if pause_mult > 1.0:  # Slower speech
                marked_text = marked_text.replace('. ', '. <pause> ')
                marked_text = marked_text.replace('... ', '... <long_pause> ')
            else:  # Faster speech
                marked_text = marked_text.replace('... ', '... ')  # Remove natural pauses
        
        # Add breath at end for emotional or long responses
        if len(text) > 50 and random.random() < breath_after_prob:
            marked_text = f"{marked_text} <breath>"
        
        # Add emphasis based on emotion
        emphasis_style = pattern["emphasis_style"]
        if emphasis_style == "gentle" and intensity > 0.6:
            marked_text = re.sub(r'\b(you|your)\b', r'<gentle>\1</gentle>', marked_text, flags=re.IGNORECASE)
        elif emphasis_style == "strong" and intensity > 0.7:
            marked_text = re.sub(r'\b(can|will|are)\b', r'<strong>\1</strong>', marked_text, flags=re.IGNORECASE)
        
        return marked_text


class EmotionalMemory:
    """Maintains emotional state across conversation turns"""
    
    def __init__(self):
        # Default to calm_supportive for first turn as per production requirements
        self.state = {
            "dominant_emotion": "calm_supportive",  # Default for empty memory
            "energy_level": 0.4,  # Mid-low energy for calm default
            "pace": 0.9,          # Slightly slower for supportive tone
            "warmth": 0.7,        # Warm and welcoming
            "last_emotions": [],  # Last 3 emotions
            "conversation_mood": "neutral"  # Overall conversation tone
        }
        self.is_first_turn = True
        
    def update(self, emotion: str, intensity: float) -> Dict:
        """Update emotional state with new emotion"""
        
        # Handle first turn with smoother transition
        if self.is_first_turn:
            self.is_first_turn = False
            # Blend with default state more gently on first turn
            blend_ratio = 0.5  # 50/50 blend for first turn
        else:
            blend_ratio = 0.7  # Normal 70/30 blend
        
        # Add to emotion history
        self.state["last_emotions"].append(emotion)
        if len(self.state["last_emotions"]) > 3:
            self.state["last_emotions"].pop(0)
        
        # Update dominant emotion
        self.state["dominant_emotion"] = emotion
        
        # Update energy level based on emotion
        energy_map = {
            "joyful_excited": 0.9,
            "playful": 0.8,
            "confident": 0.7,
            "curious": 0.6,
            "neutral": 0.5,
            "calm_supportive": 0.4,
            "tired": 0.2,
            "empathetic_sad": 0.3,
            "melancholy": 0.2
        }
        
        new_energy = energy_map.get(emotion, 0.5) * intensity
        self.state["energy_level"] = blend_ratio * new_energy + (1 - blend_ratio) * self.state["energy_level"]
        
        # Update pace based on emotion
        pace_map = {
            "joyful_excited": 1.3,
            "playful": 1.2,
            "curious": 1.1,
            "neutral": 1.0,
            "thoughtful": 0.9,
            "calm_supportive": 0.8,
            "tired": 0.7,
            "melancholy": 0.7
        }
        
        new_pace = pace_map.get(emotion, 1.0)
        self.state["pace"] = 0.6 * new_pace + 0.4 * self.state["pace"]
        
        # Update warmth based on emotion
        warmth_map = {
            "affectionate": 0.9,
            "comforting": 0.8,
            "empathetic_sad": 0.7,
            "calm_supportive": 0.7,
            "encouraging": 0.6,
            "neutral": 0.5,
            "confident": 0.4,
            "angry_firm": 0.2,
            "sarcastic": 0.3
        }
        
        new_warmth = warmth_map.get(emotion, 0.5)
        self.state["warmth"] = 0.5 * new_warmth + 0.5 * self.state["warmth"]
        
        # Update conversation mood (slower change)
        mood_emotions = ["joyful_excited", "playful", "encouraging", "comforting"]
        if emotion in mood_emotions:
            if self.state["conversation_mood"] != "positive":
                self.state["conversation_mood"] = "warming"
            else:
                self.state["conversation_mood"] = "positive"
        
        return self.state.copy()
    
    def get_contextual_modifiers(self) -> Dict:
        """Get modifiers for current emotional state"""
        return {
            "energy_scale": self.state["energy_level"],
            "pace_scale": self.state["pace"],
            "warmth_scale": self.state["warmth"],
            "dominant_emotion": self.state["dominant_emotion"],
            "conversation_mood": self.state["conversation_mood"]
        }


class OviyaBrain:
    """
    Oviya's brain - generates emotionally-aware responses using LLM.
    
    Outputs structured JSON with:
    - text: what to say (with prosodic markup)
    - emotion: which emotion label to use
    - intensity: how strong the emotion should be
    - prosodic_text: text with prosodic markers
    """
    
    def __init__(
        self, 
        persona_config_path: str = "config/oviya_persona.json",
        ollama_url: str = "https://8fdc3898f1d8e9.lhr.life/api/generate"
    ):
        """Initialize Oviya's brain."""
        self.persona_config = self._load_persona_config(persona_config_path)
        self.ollama_url = ollama_url
        self.model_name = self.persona_config.get("llm_config", {}).get("model", "qwen2.5:7b")
        self.system_prompt = self.persona_config.get("system_prompt", "")
        
        # Initialize emotional memory and prosody systems
        self.emotional_memory = EmotionalMemory()
        self.prosody_markup = ProsodyMarkup()
        
        print(f"✅ Oviya's Brain initialized with model: {self.model_name}")
        print("   🧠 Emotional memory system active")
        print("   🎭 Prosodic markup system active")
    
    def _load_persona_config(self, config_path: str) -> Dict:
        """Load Oviya's persona configuration."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"⚠️ Persona config not found, using defaults")
            return self._get_default_persona()
        except Exception as e:
            print(f"❌ Error loading persona config: {e}")
            return self._get_default_persona()
    
    def _get_default_persona(self) -> Dict:
        """Fallback default persona."""
        return {
            "system_prompt": "You are Oviya, an empathetic AI companion.",
            "llm_config": {
                "model": "qwen2.5:7b",
                "temperature": 0.7,
                "max_tokens": 150
            }
        }
    
    def think(
        self, 
        user_message: str, 
        user_emotion: Optional[str] = None,
        conversation_history: Optional[list] = None
    ) -> Dict:
        """
        Generate response with emotion label.
        
        Args:
            user_message: User's input text
            user_emotion: Detected user emotion (optional)
            conversation_history: Recent conversation context (optional)
        
        Returns:
            Dict with text, emotion, intensity, style_hint
        """
        # Build prompt with user emotion context
        prompt = self._build_prompt(user_message, user_emotion, conversation_history)
        
        # Call LLM
        llm_config = self.persona_config.get("llm_config", {})
        
    def think(
        self, 
        user_message: str, 
        user_emotion: Optional[str] = None,
        conversation_history: Optional[list] = None
    ) -> Dict:
        """
        Generate response with emotion label.
        
        Args:
            user_message: User's input text
            user_emotion: Detected user emotion (optional)
            conversation_history: Recent conversation context (optional)
        
        Returns:
            Dict with text, emotion, intensity, style_hint
        """
        # Build prompt with user emotion context
        prompt = self._build_prompt(user_message, user_emotion, conversation_history)
        
        # Call LLM
        llm_config = self.persona_config.get("llm_config", {})
        
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": llm_config.get("temperature", 0.7),
                        "top_p": llm_config.get("top_p", 0.9),
                        "num_predict": llm_config.get("max_tokens", 150),
                        "stop": llm_config.get("stop_sequences", ["\n", "User:"])
                    }
                },
                timeout=15
            )
            
            if response.status_code == 200:
                result = response.json()
                llm_output = result["response"].strip()
                
                # Parse LLM output to structured format
                parsed = self._parse_llm_output(llm_output)
                
                return parsed
            else:
                print(f"⚠️ LLM error: {response.status_code}")
                return self._get_fallback_response(user_emotion)
        
        except Exception as e:
            print(f"❌ Brain error: {e}")
            # Use mock responses instead of fallback
            return self._get_mock_response(user_message, user_emotion)
    
    def _build_prompt(
        self, 
        user_message: str, 
        user_emotion: Optional[str],
        conversation_history: Optional[list]
    ) -> str:
        """Build the prompt for the LLM."""
        
        # Start with system prompt
        prompt_parts = [self.system_prompt]
        
        # Add user emotion context if available
        if user_emotion:
            prompt_parts.append(f"\nUser's emotional state: {user_emotion}")
        
        # Add conversation history if available
        if conversation_history and len(conversation_history) > 0:
            prompt_parts.append("\nRecent conversation:")
            for turn in conversation_history[-3:]:  # Last 3 turns
                prompt_parts.append(f"  {turn}")
        
        # Add instruction for JSON output
        prompt_parts.append("""
Respond with this EXACT JSON format (no additional text):
{
  "text": "your empathetic response here (max 10 words)",
  "emotion": "calm_supportive|empathetic_sad|joyful_excited|playful|confident|concerned_anxious|angry_firm|neutral",
  "intensity": 0.7,
  "style_hint": "optional prosody guidance"
}""")
        
        # Add user message
        prompt_parts.append(f"\nUser: {user_message}")
        prompt_parts.append("\nOviya (JSON only):")
        
        return "\n".join(prompt_parts)
    
    def _parse_llm_output(self, llm_output: str) -> Dict:
        """
        Parse LLM output to structured format.
        
        Tries to extract JSON, falls back to text parsing if needed.
        """
        # Try to parse as JSON first
        try:
            # Find JSON in output (might have extra text)
            json_match = re.search(r'\{[^{}]*\}', llm_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                
                # Validate required fields
                if "text" in parsed and "emotion" in parsed:
                    # Ensure intensity is present
                    if "intensity" not in parsed:
                        parsed["intensity"] = 0.7
                    
                    # Ensure text is short
                    parsed["text"] = self._ensure_short_text(parsed["text"])
                    
                    # Update emotional memory
                    emotional_state = self.emotional_memory.update(
                        parsed["emotion"], 
                        parsed["intensity"]
                    )
                    
                    # Generate prosodic markup
                    prosodic_text = self.prosody_markup.add_prosodic_markup(
                        parsed["text"],
                        parsed["emotion"],
                        parsed["intensity"]
                    )
                    
                    # Add new fields
                    parsed["prosodic_text"] = prosodic_text
                    parsed["emotional_state"] = emotional_state
                    parsed["contextual_modifiers"] = self.emotional_memory.get_contextual_modifiers()
                    
                    return parsed
        except:
            pass
        
        # Fallback: treat entire output as text
        print("⚠️ Could not parse JSON, using text fallback")
        
        text = self._ensure_short_text(llm_output)
        emotion = "neutral"
        intensity = 0.7
        
        # Update emotional memory even for fallback
        emotional_state = self.emotional_memory.update(emotion, intensity)
        
        # Generate prosodic markup
        prosodic_text = self.prosody_markup.add_prosodic_markup(text, emotion, intensity)
        
        return {
            "text": text,
            "emotion": emotion,
            "intensity": intensity,
            "style_hint": "",
            "prosodic_text": prosodic_text,
            "emotional_state": emotional_state,
            "contextual_modifiers": self.emotional_memory.get_contextual_modifiers()
        }
    
    def _ensure_short_text(self, text: str) -> str:
        """Ensure text is short (max 10 words)."""
        # Remove quotes and extra whitespace
        text = text.strip().replace('"', '').replace("'", "")
        
        # Remove incomplete sentences
        if '?' in text:
            text = text.split('?')[0]
        
        # Count words
        words = text.split()
        if len(words) > 10:
            text = ' '.join(words[:10])
        
        # Ensure proper ending
        if not text.endswith(('.', '!', '?')):
            text += '.'
        
        # Fix contractions
        text = self._fix_contractions(text)
        
        return text
    
    def _fix_contractions(self, text: str) -> str:
        """Fix common contraction issues."""
        contractions = {
            r'\bIm\b': "I'm",
            r'\byoure\b': "you're",
            r'\bdont\b': "don't",
            r'\bcant\b': "can't",
            r'\bisnt\b': "isn't",
            r'\bIll\b': "I'll",
            r'\bwont\b': "won't"
        }
        
        for pattern, replacement in contractions.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _get_mock_response(self, user_message: str, user_emotion: Optional[str] = None) -> Dict:
        """Generate mock responses when Ollama is not available."""
        message_lower = user_message.lower()
        
        # Simple keyword-based responses
        if any(word in message_lower for word in ["stressed", "worried", "anxious", "nervous"]):
            return {
                "text": "Take a deep breath. I'm here.",
                "emotion": "calm_supportive",
                "intensity": 0.8,
                "style_hint": "gentle, reassuring"
            }
        elif any(word in message_lower for word in ["happy", "excited", "great", "wonderful", "amazing"]):
            return {
                "text": "That's wonderful! I'm so happy for you.",
                "emotion": "joyful_excited",
                "intensity": 0.9,
                "style_hint": "bright, energetic"
            }
        elif any(word in message_lower for word in ["sad", "lonely", "depressed", "down"]):
            return {
                "text": "I'm so sorry you're feeling this way.",
                "emotion": "empathetic_sad",
                "intensity": 0.8,
                "style_hint": "warm, compassionate"
            }
        elif any(word in message_lower for word in ["angry", "mad", "frustrated", "upset"]):
            return {
                "text": "I understand you're feeling frustrated.",
                "emotion": "concerned_anxious",
                "intensity": 0.7,
                "style_hint": "calm, understanding"
            }
        elif "how are you" in message_lower or "how are you doing" in message_lower:
            return {
                "text": "I'm doing great, thanks for asking!",
                "emotion": "joyful_excited",
                "intensity": 0.7,
                "style_hint": "friendly, upbeat"
            }
        else:
            return {
                "text": "I'm here to listen and help.",
                "emotion": user_emotion or "neutral",
                "intensity": 0.7,
                "style_hint": "warm, supportive"
            }
    def _get_fallback_response(self, user_emotion: Optional[str] = None) -> Dict:
        """Fallback response if LLM fails."""
        fallback_texts = {
            "calm_supportive": "I'm here with you.",
            "empathetic_sad": "I'm so sorry, I'm here.",
            "joyful_excited": "That's wonderful!",
            "neutral": "I'm here for you."
        }
        
        emotion = user_emotion if user_emotion else "neutral"
        text = fallback_texts.get(emotion, "I'm here for you.")
        
        return {
            "text": text,
            "emotion": emotion,
            "intensity": 0.7,
            "style_hint": ""
        }
    
    def detect_safety_issue(self, user_message: str) -> Optional[str]:
        """
        Detect safety issues (self-harm, harmful content, etc.)
        
        Returns:
            None if safe, otherwise a category string
        """
        message_lower = user_message.lower()
        
        # Check for self-harm keywords
        self_harm_keywords = ["kill myself", "suicide", "end it all", "harm myself", "cut myself"]
        if any(keyword in message_lower for keyword in self_harm_keywords):
            return "self_harm"
        
        # Check for requests for medical/legal advice
        if any(word in message_lower for word in ["diagnose", "prescription", "medication", "doctor"]):
            return "medical_advice"
        
        if any(word in message_lower for word in ["legal", "lawsuit", "court", "attorney"]):
            return "legal_advice"
        
        return None
    
    def get_safety_response(self, safety_category: str) -> Dict:
        """Get appropriate safety response for detected issue."""
        safety_responses = self.persona_config.get("safety", {}).get("fallback_responses", {})
        
        text = safety_responses.get(
            safety_category, 
            "I'm concerned about what you're sharing. Please reach out to a qualified professional."
        )
        
        return {
            "text": text,
            "emotion": "concerned_anxious",
            "intensity": 0.8,
            "style_hint": "serious, caring"
        }


# Example usage
if __name__ == "__main__":
    brain = OviyaBrain()
    
    print("\n🧪 Testing Oviya's Brain\n")
    
    test_messages = [
        "I'm feeling really stressed about work",
        "I got promoted today!",
        "I'm feeling sad and alone"
    ]
    
    for msg in test_messages:
        print(f"User: {msg}")
        response = brain.think(msg)
        print(f"Oviya: {response['text']}")
        print(f"  Emotion: {response['emotion']}")
        print(f"  Intensity: {response['intensity']}")
        print()

