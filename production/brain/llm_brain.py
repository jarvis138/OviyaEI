"""
Oviya's Brain - LLM-based response generation

This module handles text generation using Qwen2.5:7B (via Ollama).
It produces both response text and emotion labels for the emotion controller.
"""

import json
import os
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
from .auto_decider import AutoDecider
from .unconditional_regard import UnconditionalRegardEngine
from .vulnerability import VulnerabilityReciprocationSystem
from .consistent_persona import ConsistentPersonaMemory
from .safety_router import SafetyRouter
from .global_soul import OviyaGlobalSoul
from core.data.bias_filter import CulturalBiasFilter
# Metrics imports with safe fallback (works whether running as a package or script)
try:
    from core.monitoring.psych_metrics import BIAS_FILTER_DROP, VECTOR_ENTROPY
except Exception:
    try:
        from ...core.monitoring.psych_metrics import BIAS_FILTER_DROP, VECTOR_ENTROPY  # type: ignore
    except Exception:
        class _Noop:
            def observe(self, *a, **k):
                pass
            def inc(self, *a, **k):
                pass
        BIAS_FILTER_DROP = _Noop()
        VECTOR_ENTROPY = _Noop()
from .empathy_fusion_head import EmpathyFusionHead
from .personality_vector import PersonalityEMA
import math
import os
import torch


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
        ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
    ):
        """Initialize Oviya's brain."""
        self.persona_config = self._load_persona_config(persona_config_path)
        self.ollama_url = ollama_url
        self.model_name = self.persona_config.get("llm_config", {}).get("model", "qwen2.5:7b")
        self.system_prompt = self.persona_config.get("system_prompt", "")
        self.context = ""  # Conversation context storage
        
        # Initialize emotional memory and prosody systems
        self.emotional_memory = EmotionalMemory()
        self.prosody_markup = ProsodyMarkup()
        self.auto_decider = AutoDecider(self.persona_config)
        # Post-processors
        self.upr = UnconditionalRegardEngine()
        ff = self.persona_config.get("feature_flags", {}) if isinstance(self.persona_config, dict) else {}
        self.vuln = VulnerabilityReciprocationSystem(enabled=bool(ff.get("vulnerability_reciprocation", False)))
        self.persona_consistency = ConsistentPersonaMemory()
        self.safety_router = SafetyRouter(self.persona_config)
        self.global_soul = OviyaGlobalSoul(self.persona_config)
        self._bias_filter = CulturalBiasFilter()
        # Personality conditioning flag and modules
        self.enable_personality = bool(ff.get("ENABLE_PERSONALITY_CONDITIONING", False))
        if self.enable_personality:
            # Simple embedding dims; swap with real embeddings when available
            self._fusion = EmpathyFusionHead(emotion_dim=8, context_dim=16, memory_dim=4)
            self._p_ema = PersonalityEMA()
            self._last_personality_vector = None
        
        # Initialize Beyond-Maya features
        self.epistemic_analyzer = EpistemicProsodyAnalyzer()
        self.emotion_smoother = EmotionTransitionSmoother()
        self.backchannel_system = BackchannelSystem()
        
        # Track conversation for backchannel injection
        self.conversation_turn_count = 0
        
        print(f"Oviya's Brain initialized with model: {self.model_name}")
        print("   Emotional memory system active")
        print("   Prosodic markup system active")
        print("   Epistemic prosody analyzer active")
        print("   Emotion transition smoother active")
        print("   Backchannel system active")
    
    def _load_persona_config(self, config_path: str) -> Dict:
        """Load Oviya's persona configuration."""
        try:
            with open(config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Persona config not found, using defaults")
            return self._get_default_persona()
        except Exception as e:
            print(f"Error loading persona config: {e}")
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
        
        # Auto-decide situation/emotion/intensity/style and safety
        dec = self.auto_decider.decide(
            user_message,
            user_emotion=user_emotion,
            conversation_history=conversation_history,
            prosody=None,
        )

        # Safety shortcut
        if dec.get("safety_flag"):
            return self.safety_router.route(dec.get("safety_category", "harmful_content"))

        # Build Global Soul plan hints
        gs_ctx = {
            "emotional_weight": dec.get("intensity_hint", 0.7),
            "intensity": dec.get("intensity_hint", 0.7),
            "validated_first": True,
            "primary_emotion": user_emotion or "neutral",
            "situation": dec.get("situation", "casual_chat"),
            "vulnerability": 0.4,
            "session_seconds": 300,
            "regular_checkin": False,
            "needs_meaning": False,
            "draft": "",
        }
        soul = self.global_soul.plan(user_id="global", ctx=gs_ctx)

        # Compute personality vector p (Ma, Ahimsa, Jeong, Logos, Lagom)
        if self.enable_personality:
            try:
                # rudimentary feature vectors (replace with real embeddings)
                emo = torch.zeros(1, 8)
                ctxv = torch.zeros(1, 16)
                mem = torch.zeros(1, 4)
                # bias a bit by situation/emotion
                if dec.get("situation") == "difficult_news":
                    emo[0, 0] = 1.0
                if dec.get("emotion_hint") == "joyful_excited":
                    emo[0, 1] = 1.0
                feats = {"emotion": emo, "context": ctxv, "memory": mem}
                p = self._fusion(feats)[0]
                p = self._p_ema.update(p)
                self._last_personality_vector = p.detach().cpu().tolist()
                # record vector entropy
                try:
                    ent = -sum([float(x)*math.log(max(1e-9, float(x))) for x in self._last_personality_vector])
                    VECTOR_ENTROPY.observe(ent)
                except Exception:
                    pass
            except Exception:
                self._last_personality_vector = None

        # Low-confidence fallback to neutral context
        if float(dec.get("confidence", 0.0)) < 0.6:
            dec["situation"] = dec.get("situation") or "casual_chat"
            dec["emotion_hint"] = dec.get("emotion_hint") or "neutral"

        # Build prompt with psych context and auto hints
        self._last_guidance_category = dec.get("situation")
        prompt = self._build_prompt(
            user_message,
            user_emotion,
            conversation_history,
            auto=dec,
        )
        # Inject personality conditioning line (if enabled)
        if self.enable_personality and self._last_personality_vector:
            try:
                ma, ah, je, lo, lg = self._last_personality_vector
                personality_line = f"\n[OVIYA_PERSONALITY: Ma={ma:.2f}, Ahimsa={ah:.2f}, Jeong={je:.2f}, Logos={lo:.2f}, Lagom={lg:.2f}]"
                prompt = prompt + personality_line
            except Exception:
                pass
        # Append cultural guidance as hints
        try:
            if self.persona_config.get("feature_flags", {}).get("global_soul", True):
                lines = [prompt, "\nCultural guidance:"]
                if soul.get("meaning", {}).get("logos"):
                    lines.append(f"- Logos: {soul['meaning']['logos']}")
                lines.append(f"- Jeong depth: {soul.get('jeong_depth',0.0):.2f}")
                lines.append(f"- Sattva tone: {soul.get('sattva',{}).get('tone','steady')}")
                lines.append(f"- Lagom: {soul.get('lagom','maintain_balance')}")
                lines.append("- Avoid fixing; honor Ma (space) and balance.")
                prompt = "\n".join(lines)
        except Exception:
            pass
        
        # Optional: simple FAISS retrieval to augment context (env-gated)
        try:
            if os.environ.get("OVIYA_USE_RAG", "0") == "1":
                extra = self._retrieve_similar(user_message)
                if extra:
                    prompt += "\nRelevant memory:\n" + "\n".join([f"- {e}" for e in extra[:3]])
        except Exception:
            pass

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
                        timeout=120  # Increased timeout for larger models
                    )
                    
                    if response.status_code == 200:
                        break  # Success, exit retry loop
                    elif response.status_code == 503 and attempt < max_retries - 1:
                        print(f"   Retry {attempt + 1}/{max_retries} in {retry_delay}s...")
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
                        timeout=120  # Increased timeout for larger models
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

    def _retrieve_similar(self, query: str) -> list:
        """Very small in-process RAG using FAISS if available (best-effort)."""
        try:
            import faiss  # type: ignore
            import numpy as np
            from sentence_transformers import SentenceTransformer
        except Exception:
            return []
        import os
        base = os.environ.get("OVIYA_MEMORY_DIR", "data")
        mem_path = Path(base) / "memory.txt"
        if not mem_path.exists():
            return []
        lines = [l.strip() for l in mem_path.read_text().splitlines() if l.strip()]
        if not lines:
            return []
        try:
            model = getattr(self, "_embed_model", None)
            if model is None:
                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                self._embed_model = model
            xb = model.encode(lines, convert_to_numpy=True, normalize_embeddings=True)
            qv = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            d = xb.shape[1]
            index = faiss.IndexFlatIP(d)
            index.add(xb)
            D, I = index.search(qv, k=min(5, len(lines)))
            results = []
            for i, score in zip(I[0], D[0]):
                if i < 0: continue
                if score < 0.30: continue
                results.append(lines[i])
            return results[:3]
        except Exception:
            return []

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
        dec = self.auto_decider.decide(
            user_message,
            user_emotion=user_emotion,
            conversation_history=conversation_history,
            prosody=None,
        )
        # Compute/update personality vector in streaming path as well
        try:
            if self.enable_personality:
                emo = torch.zeros(1, 8)
                ctxv = torch.zeros(1, 16)
                mem = torch.zeros(1, 4)
                if dec.get("situation") == "difficult_news":
                    emo[0, 0] = 1.0
                if dec.get("emotion_hint") == "joyful_excited":
                    emo[0, 1] = 1.0
                feats = {"emotion": emo, "context": ctxv, "memory": mem}
                p = self._fusion(feats)[0]
                p = self._p_ema.update(p)
                self._last_personality_vector = p.detach().cpu().tolist()
                try:
                    ent = -sum([float(x)*math.log(max(1e-9, float(x))) for x in self._last_personality_vector])
                    VECTOR_ENTROPY.observe(ent)
                except Exception:
                    pass
        except Exception:
            pass
        if dec.get("safety_flag"):
            # Yield safety response text then stop
            yield self.safety_router.route(dec.get("safety_category", "harmful_content")).get("text", "")
            return
        prompt = self._build_prompt(user_message, user_emotion, conversation_history, auto=dec)
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
        conversation_history: Optional[list],
        auto: Optional[dict] = None
    ) -> str:
        """Build the prompt with situational empathy (Rogers + ToM)."""

        # Heuristic situation classifier (lightweight, context-driven)
        msg = (user_message or "").lower()

        def classify_situation(text: str) -> str:
            if any(k in text for k in ["accident", "passed away", "lost my", "hospital", "bad news", "diagnosed"]):
                return "difficult_news"
            if any(k in text for k in ["debug", "bug", "fix", "stuck", "error", "hours", "compile", "build failed"]):
                return "frustration_technical"
            if any(k in text for k in ["promoted", "promotion", "got the job", "won", "passed exam", "launched", "shipped"]):
                return "celebrating_success"
            if any(k in text for k in ["should i", "decide", "choice", "choose", "thinking of whether", "which one"]):
                return "seeking_advice"
            if any(k in text for k in ["scared", "afraid", "worried", "anxious about", "uncertain", "what if"]):
                return "expressing_fear"
            if any(k in text for k in ["sorry", "apologize", "my fault"]):
                return "apology"
            if any(k in text for k in ["argue", "fight", "conflict", "disagree", "mad at"]):
                return "conflict"
            if any(k in text for k in ["boundary", "crossed the line", "say no", "stop doing"]):
                return "boundaries"
            if any(k in text for k in ["burnout", "burned out", "exhausted", "drained"]):
                return "burnout"
            if any(k in text for k in ["lonely", "alone", "no friends", "isolated"]):
                return "loneliness"
            if any(k in text for k in ["grief", "passed away", "funeral", "lost her", "lost him"]):
                return "grief_loss"
            if any(k in text for k in ["break up", "heartbroken", "relationship ended"]):
                return "breakup_heartbreak"
            if any(k in text for k in ["sick", "symptom", "diagnosis", "doctor", "appointment"]):
                return "health_concern"
            if any(k in text for k in ["rent", "bills", "debt", "money stress", "bank account"]):
                return "finances_stress"
            if any(k in text for k in ["party nervous", "meet people", "social anxiety"]):
                return "social_anxiety"
            if any(k in text for k in ["imposter", "fraud", "not good enough"]):
                return "imposter_syndrome"
            if any(k in text for k in ["deadline", "due tomorrow", "running out of time"]):
                return "time_pressure_deadlines"
            if any(k in text for k in ["writer's block", "creative block", "can't create"]):
                return "creative_block"
            if any(k in text for k in ["kid", "child", "parenting", "tantrum"]):
                return "parenting_stress"
            if any(k in text for k in ["exam", "study", "test", "midterm"]):
                return "study_exam_stress"
            if any(k in text for k in ["flight", "train", "delayed", "canceled", "airport"]):
                return "travel_disruption"
            if any(k in text for k in ["customer service", "support ticket", "refund"]):
                return "customer_service_issue"
            if any(k in text for k in ["i'm the worst", "i'm useless", "hate myself"]):
                return "self_criticism"
            if any(k in text for k in ["failed", "got rejected", "didn't work"]):
                return "failure_setback"
            if any(k in text for k in ["can't wait", "so excited", "looking forward"]):
                return "excitement_future_plans"
            if any(k in text for k in ["thank you", "grateful", "appreciate you"]):
                return "gratitude"
            return "casual_chat"

        situation = (auto or {}).get("situation") or classify_situation(msg)

        # Pull situational guidance from persona config
        sg = self.persona_config.get("situational_guidance", {})
        guidance = sg.get(situation, sg.get("casual_chat", {}))

        # Emotion style matrix proxies (map synonyms)
        esm = self.persona_config.get("emotion_style_matrix", {})
        proxies = (esm.get("proxies") or {})

        def map_proxy(em: str) -> str:
            return proxies.get(em, em)

        default_emotion = (auto or {}).get("emotion_hint") or map_proxy(guidance.get("default_emotion", "neutral"))
        hybrid_hint = (auto or {}).get("hybrid_hint", "")
        intensity_hint = (auto or {}).get("intensity_hint")
        style_hint = (auto or {}).get("style_hint")

        # Start with system prompt
        prompt_parts = [self.system_prompt]

        # Add Rogers + ToM framing
        prompt_parts.append("Empathy Framework:\n- Rogers: paraphrase essence with as-if quality\n- ToM: infer intentions, knowledge gaps, needs\n- Situational: validate → support → guide (context-first)")

        # Add situation context and guidance
        prompt_parts.append(f"\nSituation: {situation}")
        if guidance.get("prompt_rules"):
            prompt_parts.append("Situational Guidance:\n" + guidance["prompt_rules"])
        if guidance.get("starter_phrases"):
            prompt_parts.append("Starter Suggestions: " + ", ".join(guidance["starter_phrases"][:3]))
        if guidance.get("followups"):
            prompt_parts.append("Follow-ups: " + ", ".join(guidance["followups"][:3]))
        if hybrid_hint:
            prompt_parts.append(f"Hybrid Hint: {hybrid_hint} (context for tone only; output a single allowed emotion)")

        # Add user emotion context if available (context-only)
        if user_emotion:
            prompt_parts.append(f"\nDetected emotion (context only, do NOT label it back): {user_emotion}")

        # Add conversation history if available
        if conversation_history and len(conversation_history) > 0:
            prompt_parts.append("\nRecent conversation:")
            for turn in conversation_history[-3:]:
                prompt_parts.append(f"  {turn}")

        # Expressivity feedback loop (simple):
        # If last user turn was continuation after our validation-first phrasing,
        # reflect that the user reacted positively to validation.
        try:
            if conversation_history and len(conversation_history) >= 2:
                last_user = conversation_history[-1]
                prev_ai = conversation_history[-2]
                if isinstance(last_user, dict) and isinstance(prev_ai, dict):
                    if isinstance(prev_ai.get("text"), str) and prev_ai["text"].lower().startswith(("that makes sense", "i hear you", "it makes sense")):
                        prompt_parts.append("\nExpressivity feedback: User reacted positively to validation.")
        except Exception:
            pass

        # Add instruction for JSON output (allow smooth, longer responses)
        prompt_parts.append(f"""
Respond with this EXACT JSON format (no additional text):
{{
  "text": "Smooth, natural response sized to the situation (often 1–4 sentences). Avoid emotion labels.",
  "emotion": "calm_supportive|empathetic_sad|joyful_excited|playful|confident|concerned_anxious|angry_firm|neutral",
  "intensity": {intensity_hint if intensity_hint is not None else 0.7},
  "style_hint": "{style_hint or 'optional prosody guidance'}",
  "situation": "{situation}",
  "recommended_emotion_hint": "{default_emotion}"
}}
Rules:
- Use detected emotion as context only; do not say \"you sound ...\"
- Include both emotional validation AND a practical next step or question
- Match tone to intensity implied by the situation
- Avoid self-focus and generic platitudes
""")

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
                    
                    # Apply unconditional regard and persona consistency; optional vulnerability reciprocation
                    original_text = parsed["text"]
                    t1 = self.upr.apply(original_text)
                    t2 = self.persona_consistency.ensure_consistency(t1, user_id="global")
                    t3 = self.vuln.maybe_disclose(parsed["text"], t2)
                    # Ahimsa (non-harm) filter when enabled
                    try:
                        if self.persona_config.get("feature_flags", {}).get("ahimsa_filter", True):
                            from .culture.indian import IndianEmotionalWisdom
                            parsed["text"] = IndianEmotionalWisdom().apply_ahimsa(self._ensure_short_text(t3))
                        else:
                            parsed["text"] = self._ensure_short_text(t3)
                    except Exception:
                        parsed["text"] = self._ensure_short_text(t3)
                    # Bias filter (pattern-based) for cultural stereotyping
                    try:
                        keep, meta = self._bias_filter.filter_sample({
                            'response': parsed["text"],
                            'culture': self.persona_config.get('default_culture', 'en_us')
                        })
                        if not keep:
                            try:
                                BIAS_FILTER_DROP.inc()
                            except Exception:
                                pass
                            parsed["text"] = "Let's stay with what matters to you here, without assumptions. I'm here."
                    except Exception:
                        pass

                    # Add new fields
                    parsed["prosodic_text"] = prosodic_text
                    parsed["emotional_state"] = emotional_state
                    parsed["contextual_modifiers"] = self.emotional_memory.get_contextual_modifiers()
                    
                    # Add epistemic and transition info
                    parsed["epistemic_analysis"] = epistemic_analysis
                    parsed["transition_info"] = transition_info

                    # Inject style_hint from persona emotion_style_matrix (prosody hint)
                    try:
                        esm = self.persona_config.get("emotion_style_matrix", {})
                        proxies = (esm.get("proxies") or {})
                        base_emotion = proxies.get(parsed["emotion"], parsed["emotion"])
                        style = esm.get(base_emotion, {})
                        if style and not parsed.get("style_hint"):
                            hint = style.get("prosody_hint")
                            if hint:
                                parsed["style_hint"] = hint
                    except Exception:
                        pass

                    # Apply hybrid emotion policy intensity nudge if a category was selected
                    try:
                        category = getattr(self, "_last_guidance_category", None)
                        if category:
                            pol = self._apply_hybrid_policy(category)
                            delta = float(pol.get("intensity_adjust", 0.0))
                            parsed["intensity"] = max(0.0, min(1.0, float(parsed.get("intensity", 0.7)) + delta))
                    except Exception:
                        pass
                    
                    # attach personality vector for downstream decoder
                    if self.enable_personality and self._last_personality_vector:
                        parsed["personality_vector"] = self._last_personality_vector
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
        """Ensure text is conversational length (keep sentences; cap reasonably)."""
        # Normalize quotes/whitespace
        text = text.strip().replace('"', '').replace("'", "")

        # Keep up to 4 sentences to allow natural flow
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s for s in sentences if s.strip()]
        sentences = sentences[:4]
        clipped = " ".join(sentences)

        # Cap at ~80 words to avoid overlong monologues
        words = clipped.split()
        if len(words) > 80:
            clipped = " ".join(words[:80]) + "..."

        # Fix contractions
        clipped = self._fix_contractions(clipped)
        return clipped
    
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

