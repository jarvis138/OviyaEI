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


class OviyaBrain:
    """
    Oviya's brain - generates emotionally-aware responses using LLM.
    
    Outputs structured JSON with:
    - text: what to say
    - emotion: which emotion label to use
    - intensity: how strong the emotion should be
    """
    
    def __init__(
        self, 
        persona_config_path: str = "config/oviya_persona.json",
        ollama_url: str = "http://localhost:11434/api/generate"
    ):
        """Initialize Oviya's brain."""
        self.persona_config = self._load_persona_config(persona_config_path)
        self.ollama_url = ollama_url
        self.model_name = self.persona_config.get("llm_config", {}).get("model", "qwen2.5:7b")
        self.system_prompt = self.persona_config.get("system_prompt", "")
        
        print(f"✅ Oviya's Brain initialized with model: {self.model_name}")
    
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
            return self._get_fallback_response(user_emotion)
    
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
                    
                    return parsed
        except:
            pass
        
        # Fallback: treat entire output as text
        print("⚠️ Could not parse JSON, using text fallback")
        return {
            "text": self._ensure_short_text(llm_output),
            "emotion": "neutral",
            "intensity": 0.7,
            "style_hint": ""
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

