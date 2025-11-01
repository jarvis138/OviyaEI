#!/usr/bin/env python3
"""
Unified VAD + STT Pipeline - Single Source of Truth
Optimized for ChatGPT-level smoothness

Replaces:
- production/voice/silero_vad_adapter.py
- services/services/asr-realtime/silero_vad_processor.py
- production/voice_server_webrtc.py SileroVAD class
- production/voice/whisper_client.py
- services/services/asr-realtime/whisper_stream.py
- production/websocket_server.py StreamingSTT class

Based on:
- Sesame CSM-1B recommendations
- Official Silero VAD documentation
- faster-whisper best practices
"""

import torch
import numpy as np
from faster_whisper import WhisperModel
from typing import AsyncGenerator, Dict, Optional, Tuple, List
from collections import deque
import asyncio
import time
import logging
import re

logger = logging.getLogger(__name__)


class OptimizedSileroVAD:
    """
    Unified Silero VAD implementation
    Best features from all implementations:
    - ONNX optimization (from asr-realtime)
    - State tracking (from WebRTC)
    - Streaming chunk processing (from asr-realtime)
    - Interrupt detection (from asr-realtime)
    """
    
    def __init__(
        self,
        device: str = "auto",
        threshold: float = 0.5,
        use_onnx: bool = True,
        chunk_size_ms: int = 32
    ):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.chunk_size_ms = chunk_size_ms
        self.chunk_size_samples = int(16000 * chunk_size_ms / 1000)  # 512 samples @ 16kHz
        self.sample_rate = 16000
        
        # Load model with ONNX
        self._load_model(use_onnx)
        
        # State tracking
        self.speech_buffer = deque(maxlen=100)
        self.is_speaking = False
        self.silence_counter_ms = 0.0
        self.speech_frames = 0
        self.silence_frames = 0
        
        # Thresholds (ChatGPT-like)
        self.min_speech_duration_ms = 250  # 250ms to detect speech start
        self.max_silence_duration_ms = 400  # 400ms to detect speech end (optimized)
        
        # Interrupt detection
        self.ai_is_speaking = False
        self.interrupt_callback = None
        
        logger.info(f"✅ OptimizedSileroVAD initialized: ONNX={use_onnx}, device={self.device}")
    
    def _load_model(self, use_onnx: bool):
        """Load Silero VAD with ONNX optimization"""
        try:
            model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=use_onnx
            )
            
            self.model = model.to(self.device).eval()
            (self.get_speech_timestamps, _, _, self.VADIterator, _) = utils
            
            logger.info(f"✅ VAD model loaded (ONNX={use_onnx})")
        except Exception as e:
            logger.error(f"❌ VAD load failed: {e}")
            raise
    
    def process_chunk(self, audio_chunk: np.ndarray) -> Tuple[bool, bool, Optional[np.ndarray]]:
        """
        Process audio chunk for VAD
        
        Returns:
            (is_speech_active, end_of_speech, audio_to_transcribe)
        """
        # Ensure correct format
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        if len(audio_chunk.shape) > 1:
            audio_chunk = audio_chunk.flatten()
        
        # Pad/truncate to exact chunk size
        if len(audio_chunk) < self.chunk_size_samples:
            audio_chunk = np.pad(audio_chunk, (0, self.chunk_size_samples - len(audio_chunk)))
        elif len(audio_chunk) > self.chunk_size_samples:
            audio_chunk = audio_chunk[:self.chunk_size_samples]
        
        # Fast VAD inference
        audio_tensor = torch.from_numpy(audio_chunk).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            speech_prob = self.model(audio_tensor, self.sample_rate).item()
        
        self.speech_buffer.append(audio_chunk)
        
        # Detect speech start
        if speech_prob >= self.threshold:
            self.speech_frames += 1
            self.silence_frames = 0
            self.silence_counter_ms = 0.0
            
            if not self.is_speaking:
                # Check if we have enough consecutive speech frames
                frames_needed = int(self.min_speech_duration_ms / self.chunk_size_ms)
                if self.speech_frames >= frames_needed:
                    self.is_speaking = True
                    # Check for interrupt
                    if self.ai_is_speaking and self.interrupt_callback:
                        asyncio.create_task(self.interrupt_callback())
                    return True, False, None
            else:
                return True, False, None
        else:
            # Silence detected
            if self.is_speaking:
                self.silence_frames += 1
                self.silence_counter_ms += self.chunk_size_ms
                
                # End of speech detection
                if self.silence_counter_ms >= self.max_silence_duration_ms:
                    self.is_speaking = False
                    self.speech_frames = 0
                    self.silence_frames = 0
                    
                    # Return buffered audio
                    audio_to_process = np.concatenate(list(self.speech_buffer))
                    self.speech_buffer.clear()
                    
                    return False, True, audio_to_process
            else:
                self.speech_frames = 0
        
        return self.is_speaking, False, None
    
    def set_ai_speaking_state(self, is_speaking: bool):
        """Update AI speaking state for interrupt detection"""
        self.ai_is_speaking = is_speaking
    
    def set_interrupt_callback(self, callback):
        """Set interrupt callback"""
        self.interrupt_callback = callback
    
    def reset(self):
        """Reset VAD state"""
        self.speech_buffer.clear()
        self.is_speaking = False
        self.silence_counter_ms = 0.0
        self.speech_frames = 0
        self.silence_frames = 0


class OptimizedWhisperSTT:
    """
    Unified Whisper STT implementation
    Best features:
    - faster-whisper (faster than transformers)
    - Async support
    - Partial transcription support
    - Performance tracking
    """
    
    def __init__(
        self,
        model_size: str = "base.en",
        device: str = "auto",
        partial_update_interval_ms: int = 250
    ):
        self.model_size = model_size
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.partial_update_interval_ms = partial_update_interval_ms
        
        # Performance settings
        self.compute_type = "int8_float16" if self.device == "cuda" else "int8"
        self.beam_size = 1  # Greedy for speed
        self.language = "en"
        
        # Model (initialized lazily)
        self.model = None
        self._initialized = False
        
        # Partial transcription state
        self.audio_buffer = deque(maxlen=1000)  # ~3 seconds
        self.last_partial_update = 0.0
        self.last_partial_text = ""
        
        logger.info(f"✅ OptimizedWhisperSTT initialized: {model_size}")
    
    async def initialize(self):
        """Initialize Whisper model asynchronously"""
        if self._initialized:
            return
        
        try:
            logger.info("Loading Whisper model...")
            start_time = time.time()
            
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
                num_workers=2
            )
            
            load_time = time.time() - start_time
            logger.info(f"✅ Whisper loaded in {load_time:.2f}s")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"❌ Whisper initialization failed: {e}")
            raise
    
    def add_audio(self, audio_chunk: np.ndarray):
        """Add audio chunk to buffer for partial transcription"""
        self.audio_buffer.extend(audio_chunk)
    
    async def get_partial_transcription(self) -> Optional[str]:
        """
        Get partial transcription update
        Returns None if no update needed
        """
        now_ms = time.time() * 1000
        
        # Throttle updates
        if now_ms - self.last_partial_update < self.partial_update_interval_ms:
            return None
        
        # Need minimum audio
        min_samples = int(16000 * 0.3)  # 300ms minimum
        if len(self.audio_buffer) < min_samples:
            return None
        
        if not self._initialized:
            await self.initialize()
        
        # Get recent window (last 2 seconds)
        window_samples = int(16000 * 2.0)
        audio_window = np.array(list(self.audio_buffer))[-window_samples:]
        
        try:
            segments, _ = self.model.transcribe(
                audio_window,
                beam_size=1,
                vad_filter=False,
                without_timestamps=True,
                language="en",
                condition_on_previous_text=True
            )
            
            text_parts = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    text_parts.append(text)
            
            partial_text = " ".join(text_parts).strip()
            
            if partial_text != self.last_partial_text:
                self.last_partial_text = partial_text
                self.last_partial_update = now_ms
                return partial_text
            
            return None
            
        except Exception as e:
            logger.error(f"Partial transcription error: {e}")
            return None
    
    async def transcribe_final(self, audio: np.ndarray) -> Dict:
        """
        Get final transcription after speech ends
        More accurate than partial
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            segments, info = self.model.transcribe(
                audio,
                beam_size=5,  # Better accuracy for final
                vad_filter=False,
                language="en",
                condition_on_previous_text=True
            )
            
            text_parts = []
            word_timestamps = []
            
            for segment in segments:
                text_parts.append(segment.text.strip())
                if hasattr(segment, 'words'):
                    for word in segment.words:
                        word_timestamps.append({
                            'word': word.word,
                            'start': word.start,
                            'end': word.end
                        })
            
            full_text = " ".join(text_parts).strip()
            
            return {
                'text': full_text,
                'words': word_timestamps,
                'language': info.language if hasattr(info, 'language') else 'en',
                'confidence': getattr(info, 'language_probability', 1.0)
            }
            
        except Exception as e:
            logger.error(f"Final transcription error: {e}")
            return {'text': '', 'words': [], 'language': 'en', 'confidence': 0.0}
    
    def reset(self):
        """Reset transcription state"""
        self.audio_buffer.clear()
        self.last_partial_text = ""
        self.last_partial_update = 0.0


class UnifiedVADSTTPipeline:
    """
    Complete unified pipeline combining VAD + STT
    Single source of truth for all voice processing
    """
    
    def __init__(
        self,
        vad_threshold: float = 0.5,
        whisper_model: str = "base.en",
        vad_onnx: bool = True,
        early_trigger_enabled: bool = True
    ):
        self.vad = OptimizedSileroVAD(threshold=vad_threshold, use_onnx=vad_onnx)
        self.stt = OptimizedWhisperSTT(model_size=whisper_model)
        self.early_trigger_enabled = early_trigger_enabled
        
        logger.info("✅ UnifiedVADSTTPipeline initialized")
    
    async def process_audio_chunk(self, audio_chunk: np.ndarray) -> Dict:
        """
        Process single audio chunk
        
        Returns:
            {
                'is_speech': bool,
                'end_of_speech': bool,
                'partial_text': Optional[str],
                'final_text': Optional[str],
                'audio_for_stt': Optional[np.ndarray]
            }
        """
        # Process VAD
        is_speech, end_of_speech, audio_for_stt = self.vad.process_chunk(audio_chunk)
        
        result = {
            'is_speech': is_speech,
            'end_of_speech': end_of_speech,
            'partial_text': None,
            'final_text': None,
            'audio_for_stt': audio_for_stt
        }
        
        # Add audio to STT buffer for partial transcription
        if is_speech:
            self.stt.add_audio(audio_chunk)
            # Get partial transcription
            partial_text = await self.stt.get_partial_transcription()
            if partial_text:
                result['partial_text'] = partial_text
        
        # Final transcription
        if end_of_speech and audio_for_stt is not None:
            final_result = await self.stt.transcribe_final(audio_for_stt)
            result['final_text'] = final_result['text']
            result['final_words'] = final_result.get('words', [])
            # Reset STT buffer
            self.stt.reset()
        
        return result
    
    def set_ai_speaking_state(self, is_speaking: bool):
        """Update AI speaking state for interrupt detection"""
        self.vad.set_ai_speaking_state(is_speaking)
    
    def set_interrupt_callback(self, callback):
        """Set interrupt callback"""
        self.vad.set_interrupt_callback(callback)
    
    def reset(self):
        """Reset pipeline state"""
        self.vad.reset()
        self.stt.reset()


# Singleton pattern for shared instances
_unified_pipeline_instance = None

def get_unified_pipeline() -> UnifiedVADSTTPipeline:
    """Get or create global unified pipeline instance"""
    global _unified_pipeline_instance
    if _unified_pipeline_instance is None:
        _unified_pipeline_instance = UnifiedVADSTTPipeline()
    return _unified_pipeline_instance

