#!/usr/bin/env python3
"""
Oviya CSM Audio Context Manager
Manages audio context for emotional speech generation
"""
import os
import torch
import torchaudio
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class EmotionPrompt:
    """Represents an emotion prompt"""
    emotion: str
    audio_path: str
    waveform: torch.Tensor
    sample_rate: int
    duration_ms: float
    text: str
    speaker_id: int = 1

@dataclass
class AudioContext:
    """Represents audio context for generation"""
    emotion_prompt: Optional[EmotionPrompt]
    user_audio: Optional[torch.Tensor]
    conversation_history: List[Dict]
    temperature: float
    do_sample: bool

class AudioContextManager:
    """Manages audio context for emotional speech generation"""
    
    def __init__(self, emotion_prompts_dir: str = "emotion_prompts"):
        self.emotion_prompts_dir = emotion_prompts_dir
        self.emotion_prompts: Dict[str, EmotionPrompt] = {}
        self.temperature_map = {
            "empathetic": 0.7,
            "encouraging": 0.85,
            "calm": 0.5,
            "concerned": 0.6,
            "joyful": 0.9,
            "neutral": 0.6
        }
        
        # Context limits
        self.max_context_turns = 3
        self.max_context_duration_ms = 30000  # 30 seconds
        self.max_context_tokens = 2048
        
        # Cache for processed contexts
        self.context_cache: Dict[str, AudioContext] = {}
        self.cache_max_size = 100
        
    async def initialize(self):
        """Initialize audio context manager"""
        logger.info("Initializing Audio Context Manager...")
        
        # Create emotion prompts directory if it doesn't exist
        os.makedirs(self.emotion_prompts_dir, exist_ok=True)
        
        # Load emotion prompts
        await self._load_emotion_prompts()
        
        # Create default emotion prompts if none exist
        if not self.emotion_prompts:
            await self._create_default_emotion_prompts()
        
        logger.info(f"Loaded {len(self.emotion_prompts)} emotion prompts")
    
    async def _load_emotion_prompts(self):
        """Load emotion prompts from directory"""
        try:
            prompts_dir = Path(self.emotion_prompts_dir)
            
            if not prompts_dir.exists():
                logger.warning(f"Emotion prompts directory not found: {self.emotion_prompts_dir}")
                return
            
            # Load emotion prompt files
            for emotion_file in prompts_dir.glob("*.wav"):
                emotion_name = emotion_file.stem
                
                try:
                    waveform, sample_rate = torchaudio.load(str(emotion_file))
                    duration_ms = waveform.shape[1] / sample_rate * 1000
                    
                    # Load metadata if available
                    metadata_file = emotion_file.with_suffix(".json")
                    text = f"Speaking {emotion_name}"
                    speaker_id = 1
                    
                    if metadata_file.exists():
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                            text = metadata.get("text", text)
                            speaker_id = metadata.get("speaker_id", speaker_id)
                    
                    emotion_prompt = EmotionPrompt(
                        emotion=emotion_name,
                        audio_path=str(emotion_file),
                        waveform=waveform,
                        sample_rate=sample_rate,
                        duration_ms=duration_ms,
                        text=text,
                        speaker_id=speaker_id
                    )
                    
                    self.emotion_prompts[emotion_name] = emotion_prompt
                    logger.info(f"Loaded emotion prompt: {emotion_name}")
                    
                except Exception as e:
                    logger.error(f"Error loading emotion prompt {emotion_file}: {e}")
        
        except Exception as e:
            logger.error(f"Error loading emotion prompts: {e}")
    
    async def _create_default_emotion_prompts(self):
        """Create default emotion prompts if none exist"""
        logger.info("Creating default emotion prompts...")
        
        emotions = ["empathetic", "encouraging", "calm", "concerned", "joyful"]
        
        for emotion in emotions:
            try:
                # Generate a simple tone for each emotion
                waveform = self._generate_emotion_tone(emotion)
                sample_rate = 16000
                duration_ms = 3000  # 3 seconds
                
                # Save audio file
                audio_path = os.path.join(self.emotion_prompts_dir, f"{emotion}.wav")
                torchaudio.save(audio_path, waveform, sample_rate)
                
                # Save metadata
                metadata = {
                    "text": f"Speaking {emotion}",
                    "speaker_id": 1,
                    "duration_ms": duration_ms,
                    "sample_rate": sample_rate
                }
                
                metadata_path = os.path.join(self.emotion_prompts_dir, f"{emotion}.json")
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                # Create emotion prompt
                emotion_prompt = EmotionPrompt(
                    emotion=emotion,
                    audio_path=audio_path,
                    waveform=waveform,
                    sample_rate=sample_rate,
                    duration_ms=duration_ms,
                    text=metadata["text"],
                    speaker_id=metadata["speaker_id"]
                )
                
                self.emotion_prompts[emotion] = emotion_prompt
                logger.info(f"Created default emotion prompt: {emotion}")
                
            except Exception as e:
                logger.error(f"Error creating default emotion prompt {emotion}: {e}")
    
    def _generate_emotion_tone(self, emotion: str) -> torch.Tensor:
        """Generate a simple tone for emotion (temporary implementation)"""
        sample_rate = 16000
        duration = 3.0  # 3 seconds
        samples = int(sample_rate * duration)
        
        # Generate different tones for different emotions
        if emotion == "empathetic":
            # Warm, gentle tone
            frequency = 200
            amplitude = 0.1
        elif emotion == "encouraging":
            # Bright, uplifting tone
            frequency = 300
            amplitude = 0.15
        elif emotion == "calm":
            # Soft, peaceful tone
            frequency = 150
            amplitude = 0.08
        elif emotion == "concerned":
            # Lower, worried tone
            frequency = 180
            amplitude = 0.12
        elif emotion == "joyful":
            # High, happy tone
            frequency = 400
            amplitude = 0.18
        else:
            # Neutral tone
            frequency = 250
            amplitude = 0.1
        
        # Generate sine wave
        t = torch.linspace(0, duration, samples)
        waveform = amplitude * torch.sin(2 * np.pi * frequency * t)
        
        # Add some variation
        noise = 0.01 * torch.randn(samples)
        waveform = waveform + noise
        
        # Apply fade in/out
        fade_samples = int(0.1 * sample_rate)  # 100ms fade
        fade_in = torch.linspace(0, 1, fade_samples)
        fade_out = torch.linspace(1, 0, fade_samples)
        
        waveform[:fade_samples] *= fade_in
        waveform[-fade_samples:] *= fade_out
        
        return waveform.unsqueeze(0)  # Add channel dimension
    
    async def prepare_context(self,
                            text: str,
                            emotion: str,
                            user_audio: Optional[torch.Tensor] = None,
                            conversation_history: Optional[List[Dict]] = None) -> AudioContext:
        """Prepare audio context for generation"""
        
        # Get emotion prompt
        emotion_prompt = self.emotion_prompts.get(emotion)
        if not emotion_prompt:
            logger.warning(f"Emotion prompt not found: {emotion}, using neutral")
            emotion_prompt = self.emotion_prompts.get("neutral")
        
        # Limit conversation history
        limited_history = []
        if conversation_history:
            limited_history = conversation_history[-self.max_context_turns:]
        
        # Create audio context
        context = AudioContext(
            emotion_prompt=emotion_prompt,
            user_audio=user_audio,
            conversation_history=limited_history,
            temperature=self.temperature_map.get(emotion, 0.7),
            do_sample=True
        )
        
        return context
    
    def prepare_csm_conversation(self, context: AudioContext, text: str) -> List[Dict]:
        """Prepare conversation format for CSM"""
        conversation = []
        
        # Add emotion prompt as first context
        if context.emotion_prompt:
            conversation.append({
                "role": str(context.emotion_prompt.speaker_id),
                "content": [
                    {"type": "text", "text": context.emotion_prompt.text},
                    {"type": "audio", "data": context.emotion_prompt.waveform}
                ]
            })
        
        # Add user audio if available
        if context.user_audio is not None:
            conversation.append({
                "role": "0",  # User speaker
                "content": [
                    {"type": "text", "text": "[user speech]"},
                    {"type": "audio", "data": context.user_audio}
                ]
            })
        
        # Add conversation history
        for turn in context.conversation_history:
            conversation.append(turn)
        
        # Add current text to generate
        conversation.append({
            "role": "1",  # AI speaker
            "content": [
                {"type": "text", "text": text}
            ]
        })
        
        return conversation
    
    def get_context_hash(self, text: str, emotion: str, user_audio: Optional[torch.Tensor] = None) -> str:
        """Get hash for context caching"""
        # Create hash from text, emotion, and audio features
        text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        emotion_hash = hashlib.md5(emotion.encode()).hexdigest()[:8]
        
        audio_hash = ""
        if user_audio is not None:
            # Use audio shape and first few values for hash
            audio_features = f"{user_audio.shape}_{user_audio.flatten()[:10].tolist()}"
            audio_hash = hashlib.md5(audio_features.encode()).hexdigest()[:8]
        
        return f"{text_hash}_{emotion_hash}_{audio_hash}"
    
    def cache_context(self, context_hash: str, context: AudioContext):
        """Cache audio context"""
        if len(self.context_cache) >= self.cache_max_size:
            # Remove oldest entry
            oldest_key = next(iter(self.context_cache))
            del self.context_cache[oldest_key]
        
        self.context_cache[context_hash] = context
    
    def get_cached_context(self, context_hash: str) -> Optional[AudioContext]:
        """Get cached audio context"""
        return self.context_cache.get(context_hash)
    
    def get_available_emotions(self) -> List[str]:
        """Get list of available emotions"""
        return list(self.emotion_prompts.keys())
    
    def get_emotion_info(self, emotion: str) -> Optional[Dict]:
        """Get information about an emotion"""
        emotion_prompt = self.emotion_prompts.get(emotion)
        if not emotion_prompt:
            return None
        
        return {
            "emotion": emotion,
            "text": emotion_prompt.text,
            "duration_ms": emotion_prompt.duration_ms,
            "sample_rate": emotion_prompt.sample_rate,
            "temperature": self.temperature_map.get(emotion, 0.7),
            "speaker_id": emotion_prompt.speaker_id
        }
    
    def get_context_stats(self) -> Dict:
        """Get context manager statistics"""
        return {
            "total_emotions": len(self.emotion_prompts),
            "available_emotions": list(self.emotion_prompts.keys()),
            "cached_contexts": len(self.context_cache),
            "max_context_turns": self.max_context_turns,
            "max_context_duration_ms": self.max_context_duration_ms,
            "temperature_map": self.temperature_map,
            "emotion_prompts_dir": self.emotion_prompts_dir
        }
    
    def validate_context(self, context: AudioContext) -> Tuple[bool, List[str]]:
        """Validate audio context"""
        issues = []
        
        # Check emotion prompt
        if not context.emotion_prompt:
            issues.append("No emotion prompt provided")
        
        # Check conversation history length
        if len(context.conversation_history) > self.max_context_turns:
            issues.append(f"Too many conversation turns: {len(context.conversation_history)}")
        
        # Check user audio duration
        if context.user_audio is not None:
            duration_ms = context.user_audio.shape[1] / 16000 * 1000
            if duration_ms > self.max_context_duration_ms:
                issues.append(f"User audio too long: {duration_ms}ms")
        
        # Check temperature range
        if not (0.1 <= context.temperature <= 2.0):
            issues.append(f"Temperature out of range: {context.temperature}")
        
        return len(issues) == 0, issues

# Usage example
async def main():
    """Test the audio context manager"""
    context_manager = AudioContextManager()
    await context_manager.initialize()
    
    # Test context preparation
    context = await context_manager.prepare_context(
        text="Hello, how are you?",
        emotion="empathetic",
        user_audio=None,
        conversation_history=[]
    )
    
    # Test CSM conversation format
    conversation = context_manager.prepare_csm_conversation(context, "Hello, how are you?")
    print(f"CSM conversation: {len(conversation)} turns")
    
    # Test emotion info
    emotion_info = context_manager.get_emotion_info("empathetic")
    print(f"Emotion info: {emotion_info}")
    
    # Test stats
    stats = context_manager.get_context_stats()
    print(f"Context stats: {stats}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
