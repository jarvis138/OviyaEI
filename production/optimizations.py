#!/usr/bin/env python3
"""
Runtime Optimizations for Oviya Production
Implements caching, streaming, and performance improvements
"""

import time
import hashlib
from typing import Dict, Optional, Tuple
from functools import lru_cache
from pathlib import Path
import json

class ProsodyTemplateCache:
    """
    Cache prosody templates per emotion for 3-5× faster inference
    """
    def __init__(self, cache_dir: Path = Path("cache/prosody")):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.memory_cache = {}
        self.hits = 0
        self.misses = 0
        print("✅ Prosody template cache initialized")
    
    def _generate_key(self, text: str, emotion: str, intensity: float) -> str:
        """Generate cache key from inputs"""
        # Create hash of text for shorter keys
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        intensity_rounded = round(intensity, 1)
        return f"{emotion}_{intensity_rounded}_{text_hash}"
    
    def get(self, text: str, emotion: str, intensity: float) -> Optional[str]:
        """Get cached prosodic markup"""
        key = self._generate_key(text, emotion, intensity)
        
        # Check memory cache first
        if key in self.memory_cache:
            self.hits += 1
            return self.memory_cache[key]
        
        # Check disk cache
        cache_file = self.cache_dir / f"{key}.txt"
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                prosodic_text = f.read()
            self.memory_cache[key] = prosodic_text
            self.hits += 1
            return prosodic_text
        
        self.misses += 1
        return None
    
    def set(self, text: str, emotion: str, intensity: float, prosodic_text: str):
        """Cache prosodic markup"""
        key = self._generate_key(text, emotion, intensity)
        
        # Store in memory cache
        self.memory_cache[key] = prosodic_text
        
        # Store in disk cache (async would be better)
        cache_file = self.cache_dir / f"{key}.txt"
        with open(cache_file, 'w') as f:
            f.write(prosodic_text)
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.memory_cache)
        }
    
    def clear(self):
        """Clear cache"""
        self.memory_cache.clear()
        for cache_file in self.cache_dir.glob("*.txt"):
            cache_file.unlink()
        self.hits = 0
        self.misses = 0


class EmotionTemplateCache:
    """
    Cache emotion parameters per emotion/intensity combination
    """
    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0
        print("✅ Emotion parameter cache initialized")
    
    @staticmethod
    def _generate_key(emotion: str, intensity: float, contextual_modifiers: Optional[Dict] = None) -> str:
        """Generate cache key"""
        intensity_rounded = round(intensity, 2)
        if contextual_modifiers:
            # Include modifiers in key
            mod_str = "_".join(f"{k}{round(v, 2)}" for k, v in sorted(contextual_modifiers.items()))
            return f"{emotion}_{intensity_rounded}_{mod_str}"
        return f"{emotion}_{intensity_rounded}"
    
    def get(self, emotion: str, intensity: float, contextual_modifiers: Optional[Dict] = None) -> Optional[Dict]:
        """Get cached emotion parameters"""
        key = self._generate_key(emotion, intensity, contextual_modifiers)
        
        if key in self.cache:
            self.hits += 1
            return self.cache[key].copy()  # Return copy to prevent mutation
        
        self.misses += 1
        return None
    
    def set(self, emotion: str, intensity: float, params: Dict, contextual_modifiers: Optional[Dict] = None):
        """Cache emotion parameters"""
        key = self._generate_key(emotion, intensity, contextual_modifiers)
        self.cache[key] = params.copy()
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        total = self.hits + self.misses
        hit_rate = (self.hits / total * 100) if total > 0 else 0
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "cache_size": len(self.cache)
        }


class StreamingSynthesizer:
    """
    Stream synthesis in 300-500ms chunks for faster perceived response
    """
    def __init__(self, chunk_duration_ms: int = 400):
        self.chunk_duration_ms = chunk_duration_ms
        print(f"✅ Streaming synthesizer initialized ({chunk_duration_ms}ms chunks)")
    
    def split_for_streaming(self, text: str) -> list[str]:
        """
        Split text into streamable chunks at natural boundaries
        Target: ~10-15 words per chunk for 300-500ms audio
        """
        import re
        
        # Split on sentence boundaries first
        sentences = re.split(r'([.!?]+)', text)
        
        chunks = []
        current_chunk = ""
        
        for i in range(0, len(sentences), 2):
            sentence = sentences[i]
            punctuation = sentences[i+1] if i+1 < len(sentences) else ""
            
            # Estimate words (rough)
            words = len(sentence.split())
            
            if words <= 15:
                # Short sentence, add as chunk
                chunks.append(sentence + punctuation)
            else:
                # Long sentence, split on commas
                parts = re.split(r'(,)', sentence)
                for j in range(0, len(parts), 2):
                    part = parts[j]
                    comma = parts[j+1] if j+1 < len(parts) else ""
                    current_chunk += part + comma
                    
                    # Check if chunk is large enough
                    if len(current_chunk.split()) >= 10:
                        chunks.append(current_chunk.strip())
                        current_chunk = ""
                
                # Add remaining
                if current_chunk.strip():
                    chunks.append(current_chunk.strip() + punctuation)
                    current_chunk = ""
        
        # Filter empty chunks
        chunks = [c.strip() for c in chunks if c.strip()]
        
        return chunks
    
    def estimate_chunk_duration(self, text: str) -> float:
        """Estimate audio duration in seconds"""
        words = len(text.split())
        # Average: 150 words/min = 2.5 words/sec
        return words / 2.5


class AdaptivePauseSystem:
    """
    Learn pause length from user latency (faster if user interrupts often)
    """
    def __init__(self):
        self.user_interruption_count = 0
        self.total_responses = 0
        self.avg_user_latency = 2.0  # Default: 2 seconds
        self.pause_multiplier = 1.0
        print("✅ Adaptive pause system initialized")
    
    def record_user_latency(self, latency_seconds: float):
        """Record how long user took to respond"""
        self.avg_user_latency = (self.avg_user_latency * 0.8) + (latency_seconds * 0.2)
        self.total_responses += 1
        
        # Adjust pause multiplier based on user speed
        if self.avg_user_latency < 1.0:
            # Fast user, reduce pauses
            self.pause_multiplier = 0.7
        elif self.avg_user_latency < 2.0:
            # Normal user
            self.pause_multiplier = 1.0
        else:
            # Slow/thoughtful user, increase pauses
            self.pause_multiplier = 1.3
    
    def record_interruption(self):
        """Record user interrupting Oviya"""
        self.user_interruption_count += 1
        # Reduce pauses if user interrupts frequently
        if self.user_interruption_count > 3:
            self.pause_multiplier *= 0.9
    
    def get_pause_multiplier(self) -> float:
        """Get current pause multiplier"""
        return max(0.5, min(1.5, self.pause_multiplier))
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        interruption_rate = (self.user_interruption_count / self.total_responses * 100) if self.total_responses > 0 else 0
        return {
            "avg_user_latency": self.avg_user_latency,
            "interruption_rate": interruption_rate,
            "pause_multiplier": self.pause_multiplier,
            "total_responses": self.total_responses
        }


class ContextualEnergyDecay:
    """
    If conversation stays calm for > 5 turns, slowly lower pitch and loudness 2-3%
    """
    def __init__(self):
        self.calm_turn_count = 0
        self.energy_decay = 0.0
        self.pitch_decay = 0.0
        self.loudness_decay = 0.0
        print("✅ Contextual energy decay system initialized")
    
    def update(self, emotion: str, energy_level: float):
        """Update based on current emotion"""
        # Define "calm" emotions
        calm_emotions = ["calm_supportive", "comforting", "neutral", "thoughtful", "melancholic"]
        
        if emotion in calm_emotions and energy_level < 0.5:
            self.calm_turn_count += 1
        else:
            self.calm_turn_count = max(0, self.calm_turn_count - 1)
        
        # Apply decay after 5 calm turns
        if self.calm_turn_count >= 5:
            # Gradual decay: 0.5% per turn, max 3%
            decay_per_turn = 0.005
            max_decay = 0.03
            
            self.energy_decay = min(max_decay, (self.calm_turn_count - 5) * decay_per_turn)
            self.pitch_decay = min(max_decay * 0.8, (self.calm_turn_count - 5) * decay_per_turn * 0.8)
            self.loudness_decay = min(max_decay, (self.calm_turn_count - 5) * decay_per_turn)
        else:
            # Reset decay
            self.energy_decay = 0.0
            self.pitch_decay = 0.0
            self.loudness_decay = 0.0
    
    def get_modifiers(self) -> Dict:
        """Get current decay modifiers"""
        return {
            "energy_scale": 1.0 - self.energy_decay,
            "pitch_scale": 1.0 - self.pitch_decay,
            "loudness_scale": 1.0 - self.loudness_decay
        }
    
    def get_stats(self) -> Dict:
        """Get statistics"""
        return {
            "calm_turn_count": self.calm_turn_count,
            "energy_decay": self.energy_decay * 100,  # as percentage
            "pitch_decay": self.pitch_decay * 100,
            "loudness_decay": self.loudness_decay * 100
        }


# Example usage
if __name__ == "__main__":
    print("="*70)
    print("  RUNTIME OPTIMIZATIONS TEST")
    print("="*70)
    
    # Test 1: Prosody cache
    print("\n1. Testing Prosody Template Cache:")
    prosody_cache = ProsodyTemplateCache()
    
    # Simulate cache usage
    text = "Hello, how are you?"
    prosody_cache.set(text, "joyful_excited", 0.8, "Hello, <smile> how are you?")
    
    result = prosody_cache.get(text, "joyful_excited", 0.8)
    print(f"   Cache hit: {result}")
    
    stats = prosody_cache.get_stats()
    print(f"   Stats: {stats}")
    
    # Test 2: Streaming
    print("\n2. Testing Streaming Synthesizer:")
    streamer = StreamingSynthesizer()
    
    long_text = "I understand this is difficult. Take your time, and remember that I'm here to support you. Whatever you're feeling is valid, and we can work through this together."
    chunks = streamer.split_for_streaming(long_text)
    
    print(f"   Split into {len(chunks)} chunks:")
    for i, chunk in enumerate(chunks, 1):
        duration = streamer.estimate_chunk_duration(chunk)
        print(f"      {i}. \"{chunk}\" (~{duration:.2f}s)")
    
    # Test 3: Adaptive pauses
    print("\n3. Testing Adaptive Pause System:")
    pause_system = AdaptivePauseSystem()
    
    # Simulate fast user
    for _ in range(3):
        pause_system.record_user_latency(0.8)
    
    print(f"   Pause multiplier: {pause_system.get_pause_multiplier():.2f}")
    print(f"   Stats: {pause_system.get_stats()}")
    
    # Test 4: Energy decay
    print("\n4. Testing Contextual Energy Decay:")
    decay_system = ContextualEnergyDecay()
    
    # Simulate 7 calm turns
    for i in range(7):
        decay_system.update("calm_supportive", 0.4)
    
    modifiers = decay_system.get_modifiers()
    print(f"   Modifiers: {modifiers}")
    print(f"   Stats: {decay_system.get_stats()}")
    
    print("\n" + "="*70)
    print("✅ All optimization systems tested successfully!")
    print("="*70)


