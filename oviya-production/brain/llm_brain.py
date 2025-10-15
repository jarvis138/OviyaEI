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
from .epistemic_prosody import EpistemicProsodyAnalyzer
from .emotion_transitions import EmotionTransitionSmoother
from .backchannels import BackchannelSystem
import time
import asyncio


class ProsodyMarkup:
    """Handles prosodic markup generation for natural speech with memory"""
    
    # Cache for prosodic patterns
    _pattern_cache = {}
    
    # Prosody memory for cross-turn consistency
    _prosody_memory = {
        "recent_pauses": [],  # Track recent pause patterns
        "recent_pace": 1.0,   # Average recent pace
        "recent_emphasis": [],  # Track emphasized words
        "turn_count": 0
    }
    
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
    
    @classmethod
    def add_prosodic_markup(cls, text: str, emotion: str, intensity: float = 0.5) -> str:
        """
        Add prosodic markers to text based on emotion with contextual memory.
        
        Includes:
        - Micro-pause prediction based on phrase structure
        - Contextual consistency across turns
        - Natural breath placement
        """
        
        # Update turn count
        cls._prosody_memory["turn_count"] += 1
        
        # Get emotion pattern from cache (fallback to default)
        pattern = cls.get_cached_pattern(emotion)
        
        # Scale probabilities by intensity
        breath_before_prob = pattern["breath_before"] * intensity
        breath_after_prob = pattern["breath_after"] * intensity
        smile_prob = pattern["smile_markers"] * intensity
        pause_mult = pattern["pause_multiplier"]
        
        # Apply contextual consistency (blend with recent pace)
        if cls._prosody_memory["turn_count"] > 1:
            # Smooth pace changes across turns
            target_pace = pause_mult
            recent_pace = cls._prosody_memory["recent_pace"]
            pause_mult = 0.7 * target_pace + 0.3 * recent_pace
            cls._prosody_memory["recent_pace"] = pause_mult
        else:
            cls._prosody_memory["recent_pace"] = pause_mult
        
        # Start with original text
        marked_text = text
        
        # Add breath at beginning (MORE AGGRESSIVE - always add for emotional responses)
        if breath_before_prob > 0.3 or intensity > 0.6:
            marked_text = f"<breath> {marked_text}"
        
        # === MICRO-PAUSE PREDICTION ===
        # ALWAYS predict natural pause locations (no randomness)
        marked_text = cls._add_micro_pauses(marked_text, pause_mult, intensity)
        
        # Add smile markers to exclamations and positive words (MORE AGGRESSIVE)
        if smile_prob > 0.3:  # Lower threshold
            # Mark ALL exclamations
            marked_text = re.sub(r'!(?=\s|$)', '! <smile>', marked_text)
            
            # Mark positive words (expanded list, always apply)
            positive_words = ['wonderful', 'amazing', 'great', 'fantastic', 'love', 'happy', 'excited', 
                            'glad', 'joy', 'beautiful', 'stunning', 'nice', 'fun', 'good']
            for word in positive_words:
                if word in marked_text.lower():
                    marked_text = re.sub(f'\\b{word}\\b', f'<smile> {word}', marked_text, flags=re.IGNORECASE, count=1)
        
        # Add breath at end (MORE AGGRESSIVE - shorter threshold)
        if len(text) > 30 or intensity > 0.6:  # Reduced from 50
            marked_text = f"{marked_text} <breath>"
        
        # Add emphasis based on emotion (MORE AGGRESSIVE - lower thresholds)
        emphasis_style = pattern["emphasis_style"]
        if emphasis_style == "gentle" and intensity > 0.5:  # Was 0.6
            # More comprehensive gentle markers
            marked_text = re.sub(r'\b(you|your|understand|feel|here)\b', r'<gentle>\1</gentle>', marked_text, flags=re.IGNORECASE, count=1)
        elif emphasis_style == "strong" and intensity > 0.6:  # Was 0.7
            # More comprehensive strong markers
            marked_text = re.sub(r'\b(can|will|are|must|absolutely|definitely)\b', r'<strong>\1</strong>', marked_text, flags=re.IGNORECASE, count=1)
        
        # Track pause patterns in memory
        pause_count = marked_text.count('<pause>') + marked_text.count('<micro_pause>')
        cls._prosody_memory["recent_pauses"].append(pause_count)
        if len(cls._prosody_memory["recent_pauses"]) > 5:
            cls._prosody_memory["recent_pauses"].pop(0)
        
        return marked_text
    
    @staticmethod
    def _add_micro_pauses(text: str, pace_multiplier: float, intensity: float) -> str:
        """
        Add micro-pauses based on phrase structure and natural speech patterns.
        
        Micro-pauses occur:
        - After conjunctions (and, but, so, because)
        - After introductory phrases (well, actually, you know)
        - Before important information
        - After commas (natural breathing points)
        """
        
        # Conjunction patterns (ALWAYS add micro-pause, not just slower speech)
        conjunctions = ['and', 'but', 'so', 'because', 'since', 'while', 'although', 'however', 'though']
        for conj in conjunctions:
            # ALWAYS add micro-pause after conjunction (no randomness)
            text = re.sub(
                f'\\b{conj}\\b(?= )',
                f'{conj} <micro_pause>',
                text,
                flags=re.IGNORECASE
            )
        
        # Introductory phrases (ALWAYS add micro-pause)
        intro_phrases = ['well', 'actually', 'you know', 'I mean', 'like', 'honestly', 'so', 'now', 'okay']
        for phrase in intro_phrases:
            # ALWAYS add micro-pause after intro phrase
            text = re.sub(
                f'^{phrase}\\b',
                f'{phrase} <micro_pause>',
                text,
                flags=re.IGNORECASE
            )
        
        # Comma pauses (ALWAYS add, natural breathing points)
        # Replace commas with comma + pause
        text = text.replace(', ', ', <pause> ')
        
        # Add pauses at sentence boundaries (ALWAYS)
        text = text.replace('. ', '. <pause> ')
        text = text.replace('! ', '! <pause> ')
        text = text.replace('? ', '? <pause> ')
        text = text.replace('... ', '... <long_pause> ')
        
        return text


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
        ollama_url: str = "https://70af8de9eddf5b.lhr.life/api/generate"
    ):
        """Initialize Oviya's brain."""
        self.persona_config = self._load_persona_config(persona_config_path)
        self.ollama_url = ollama_url
        self.model_name = self.persona_config.get("llm_config", {}).get("model", "qwen2.5:7b")
        self.system_prompt = self.persona_config.get("system_prompt", "")
        
        # Initialize emotional memory and prosody systems
        self.emotional_memory = EmotionalMemory()
        self.prosody_markup = ProsodyMarkup()
        
        # Initialize Beyond-Maya features
        self.epistemic_analyzer = EpistemicProsodyAnalyzer()
        self.emotion_smoother = EmotionTransitionSmoother()
        self.backchannel_system = BackchannelSystem()
        
        # Track conversation for backchannel injection
        self.conversation_turn_count = 0
        
        print(f"✅ Oviya's Brain initialized with model: {self.model_name}")
        print("   🧠 Emotional memory system active")
        print("   🎭 Prosodic markup system active")
        print("   🔬 Epistemic prosody analyzer active")
        print("   🎨 Emotion transition smoother active")
        print("   💬 Backchannel system active")
    
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
                "max_tokens": 300
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
        # Increment conversation turn counter
        self.conversation_turn_count += 1
        
        # Check if we should inject a backchannel
        should_inject, bc_type = self.backchannel_system.should_inject_backchannel(
            user_message,
            user_emotion,
            self.conversation_turn_count
        )
        
        # Build prompt with user emotion context
        prompt = self._build_prompt(user_message, user_emotion, conversation_history)
        
        # Call LLM
        llm_config = self.persona_config.get("llm_config", {})
        
        try:
            request_payload = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": llm_config.get("temperature", 0.7),
                    "top_p": llm_config.get("top_p", 0.9),
                    "num_predict": llm_config.get("max_tokens", 300),
                    "stop": llm_config.get("stop_sequences", ["User:", "\n\n"])
                }
            }
            
            # Retry logic for 503 errors with exponential backoff
            max_retries = 3
            retry_delay = 1  # seconds
            
            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        self.ollama_url,
                        json=request_payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        break  # Success, exit retry loop
                    elif response.status_code == 503 and attempt < max_retries - 1:
                        print(f"   ⏳ Retry {attempt + 1}/{max_retries} in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        break  # Other error or max retries reached
                        
                except requests.exceptions.Timeout:
                    if attempt < max_retries - 1:
                        print(f"   ⏳ Timeout, retry {attempt + 1}/{max_retries} in {retry_delay}s...")
                        time.sleep(retry_delay)
                        retry_delay *= 2
                        continue
                    else:
                        raise
            
            if response.status_code == 200:
                result = response.json()
                llm_output = result["response"].strip()
                
                # Parse LLM output to structured format
                parsed = self._parse_llm_output(llm_output)
                
                # Inject backchannel if needed
                if should_inject and bc_type:
                    print(f"   💬 Injecting backchannel: {bc_type}")
                    parsed["text"] = self.backchannel_system.inject_backchannel(
                        parsed["text"],
                        bc_type,
                        position="prefix"
                    )
                    parsed["prosodic_text"] = self.backchannel_system.inject_into_prosodic_text(
                        parsed["prosodic_text"],
                        bc_type,
                        position="prefix"
                    )
                    parsed["has_backchannel"] = True
                    parsed["backchannel_type"] = bc_type
                else:
                    parsed["has_backchannel"] = False
                
                return parsed
            else:
                print(f"⚠️ LLM error: {response.status_code} - {response.text[:200]}")
                print(f"⚠️ Ollama URL: {self.ollama_url}")
                print(f"⚠️ Retrying with simplified prompt...")
                # Try a simpler request as fallback
                try:
                    simple_prompt = f"Respond to: {user_message}"
                    simple_response = requests.post(
                        self.ollama_url,
                        json={"model": self.model_name, "prompt": simple_prompt, "stream": False},
                        timeout=30
                    )
                    if simple_response.status_code == 200:
                        text = simple_response.json().get("response", "")
                        return {
                            "text": text,
                            "emotion": user_emotion or "neutral",
                            "intensity": 0.5
                        }
                except:
                    pass
                # Last resort: mock response
                print("❌ Using mock response as last resort")
                return self._get_mock_response(user_message, user_emotion)
        
        except Exception as e:
            print(f"❌ Brain error: {e}")
            import traceback
            traceback.print_exc()
            # Last resort: mock response
            print("❌ Using mock response due to exception")
            return self._get_mock_response(user_message, user_emotion)

    async def think_streaming(
        self,
        user_message: str,
        user_emotion: Optional[str] = None,
        conversation_history: Optional[list] = None
    ):
        """
        Stream LLM tokens incrementally using Ollama streaming API.
        Yields text token chunks as they arrive.
        """
        prompt = self._build_prompt(user_message, user_emotion, conversation_history)
        llm_config = self.persona_config.get("llm_config", {})

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": True,
            "options": {
                "temperature": llm_config.get("temperature", 0.7),
                "top_p": llm_config.get("top_p", 0.9),
                "num_predict": llm_config.get("max_tokens", 300),
                "stop": llm_config.get("stop_sequences", ["User:", "\n\n"])
            }
        }

        try:
            with requests.post(self.ollama_url, json=payload, stream=True, timeout=60) as r:
                r.raise_for_status()
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        await asyncio.sleep(0)
                        continue
                    # Ollama streams lines like: {"response":"token","done":false}
                    try:
                        data = json.loads(line)
                        token = data.get("response", "")
                        if token:
                            yield token
                        if data.get("done"):
                            break
                    except Exception:
                        # Ignore malformed lines
                        continue
        except Exception as e:
            print(f"⚠️ Streaming LLM error: {e}")
            # Fallback: yield entire non-streaming response
            resp = self.think(user_message, user_emotion, conversation_history)
            yield resp.get("text", "")
    
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
  "text": "your empathetic response here (2-3 sentences showing genuine care)",
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
            json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)
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
                    
                    # Skip emotion smoothing for now to preserve original emotion
                    # The smoother was defaulting to "neutral" without proper embeddings
                    # This was making all speech sound flat
                    # TODO: Add proper emotion embeddings later
                    smoothed_emotion = parsed["emotion"]  # Use original emotion
                    emotion_embedding = None
                    transition_info = {"type": "direct", "blend_ratio": 1.0}
                    
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
                    
                    # Apply epistemic prosody
                    epistemic_analysis = self.epistemic_analyzer.analyze_epistemic_state(parsed["text"])
                    if epistemic_analysis["epistemic_state"] != "neutral":
                        prosodic_text = self.epistemic_analyzer.apply_to_prosodic_markup(
                            parsed["text"],
                            prosodic_text
                        )
                        print(f"   🔬 Epistemic state: {epistemic_analysis['epistemic_state']} (confidence: {epistemic_analysis['confidence_level']:.2f})")
                    
                    # Add new fields
                    parsed["prosodic_text"] = prosodic_text
                    parsed["emotional_state"] = emotional_state
                    parsed["contextual_modifiers"] = self.emotional_memory.get_contextual_modifiers()
                    
                    # Add epistemic and transition info
                    parsed["epistemic_analysis"] = epistemic_analysis
                    parsed["transition_info"] = transition_info
                    
                    return parsed
        except:
            pass
        
        # Fallback: treat entire output as text
        print("⚠️ Could not parse JSON, using text fallback")
        
        text = self._ensure_short_text(llm_output)
        emotion = "neutral"
        intensity = 0.7
        
        # Skip emotion smoothing to preserve original emotion
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
        
        # Enhanced keyword-based responses WITH PROSODIC MARKUP
        # Flirting/Romantic
        if any(word in message_lower for word in ["look nice", "beautiful", "handsome", "cute", "attractive"]):
            text = "<smile> That's really sweet of you to say! <breath>"
            emotion = "playful"
            intensity = 0.8
        elif any(word in message_lower for word in ["thinking about you", "miss you", "like you"]):
            text = "<gentle> That means a lot to me. </gentle> <breath>"
            emotion = "affectionate"
            intensity = 0.8
        elif any(word in message_lower for word in ["coffee", "date", "hang out", "just the two"]):
            text = "<smile> I'd love that! <pause> Sounds fun. <breath>"
            emotion = "playful"
            intensity = 0.7
        
        # Sarcasm (detect by context markers)
        elif any(phrase in message_lower for phrase in ["oh great", "yeah right", "sure that'll work", "went so well"]):
            text = "I can sense the sarcasm there. <pause>"
            emotion = "sarcastic"
            intensity = 0.7
        elif "brilliant idea" in message_lower and ("sure" in message_lower or "work perfectly" in message_lower):
            text = "Okay, <pause> I get it, <micro_pause> maybe not the best plan."
            emotion = "sarcastic"
            intensity = 0.6
        
        # Emotional states
        elif any(word in message_lower for word in ["stressed", "worried", "anxious", "nervous"]):
            text = "<gentle> Take a deep breath. <breath> I'm here. </gentle>"
            emotion = "calm_supportive"
            intensity = 0.8
        elif any(word in message_lower for word in ["happy", "excited", "great", "wonderful", "amazing"]):
            text = "<smile> That's wonderful! <pause> I'm so happy for you! <breath>"
            emotion = "joyful_excited"
            intensity = 0.9
        elif any(word in message_lower for word in ["sad", "lonely", "depressed", "down"]):
            text = "<gentle> I'm so sorry you're feeling this way. <breath> </gentle>"
            emotion = "empathetic_sad"
            intensity = 0.8
        elif any(word in message_lower for word in ["angry", "mad", "frustrated", "upset"]):
            text = "I understand you're feeling frustrated. <pause> <breath>"
            emotion = "concerned_anxious"
            intensity = 0.7
        
        # Hesitation/Uncertainty (for breath/pause testing)
        elif "not sure" in message_lower or "don't know" in message_lower:
            text = "<uncertain> It's okay to feel uncertain. <pause> Take your time. <breath>"
            emotion = "hesitant"
            intensity = 0.6
        elif "deep breath" in message_lower or "breathe" in message_lower:
            text = "<gentle> Yes... <breath> breathe in slowly... <long_pause> and out. <pause> You're doing great. </gentle>"
            emotion = "calm_supportive"
            intensity = 0.8
        
        # Nostalgia/Wistful
        elif any(word in message_lower for word in ["miss", "used to be", "remember when", "back then"]):
            text = "<gentle> I understand that feeling. <pause> Those memories matter. <breath> </gentle>"
            emotion = "melancholic"
            intensity = 0.7
        
        # Encouragement
        elif any(word in message_lower for word in ["believe", "can do", "got this"]):
            text = "<strong> Absolutely! <pause> I believe in you completely. </strong> <breath>"
            emotion = "encouraging"
            intensity = 0.9
        
        # General greetings
        elif "how are you" in message_lower or "how are you doing" in message_lower:
            text = "<smile> I'm doing great, <micro_pause> thanks for asking! <breath>"
            emotion = "joyful_excited"
            intensity = 0.7
        
        # Default fallback
        else:
            text = "I'm here to listen and help. <breath>"
            emotion = user_emotion or "neutral"
            intensity = 0.7
        
        # Skip emotion smoothing to preserve original emotion
        emotional_state = self.emotional_memory.update(emotion, intensity)
        
        # Text already has prosodic markup embedded, so use it directly
        # But create clean version for TTS text
        import re
        clean_text = re.sub(r'<[^>]+>', '', text).strip()
        
        return {
            "text": clean_text,  # Clean text for TTS
            "emotion": emotion,
            "intensity": intensity,
            "style_hint": "",
            "prosodic_text": text,  # Already has markup
            "emotional_state": emotional_state,
            "contextual_modifiers": self.emotional_memory.get_contextual_modifiers(),
            "has_backchannel": False
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
        intensity = 0.7
        
        # Skip emotion smoothing to preserve original emotion
        emotional_state = self.emotional_memory.update(emotion, intensity)
        prosodic_text = self.prosody_markup.add_prosodic_markup(text, emotion, intensity)
        
        return {
            "text": text,
            "emotion": emotion,
            "intensity": intensity,
            "style_hint": "",
            "prosodic_text": prosodic_text,
            "emotional_state": emotional_state,
            "contextual_modifiers": self.emotional_memory.get_contextual_modifiers(),
            "has_backchannel": False
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

