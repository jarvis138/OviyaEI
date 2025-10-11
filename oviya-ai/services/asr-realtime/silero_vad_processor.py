#!/usr/bin/env python3
"""
Silero VAD v5 integration for voice activity detection
https://github.com/snakers4/silero-vad
"""
import torch
import numpy as np
from collections import deque
from typing import Dict, Optional, List, Callable
import asyncio
import logging

logger = logging.getLogger(__name__)

class SileroVADProcessor:
    """
    Silero VAD v5 integration for voice activity detection
    """
    
    def __init__(self):
        # Load Silero VAD according to official documentation
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=True  # Use ONNX for better performance
        )
        
        # Get utility functions according to official API
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = self.utils
        
        # VAD configuration
        self.sample_rate = 16000
        self.chunk_size_ms = 30  # Process 30ms chunks for low latency
        self.chunk_size_samples = int(self.sample_rate * self.chunk_size_ms / 1000)
        
        # Thresholds
        self.speech_threshold = 0.5  # Default Silero threshold
        self.min_speech_duration_ms = 250
        self.max_silence_duration_ms = 500
        
        # Buffers
        self.audio_buffer = deque(maxlen=100)  # ~3 seconds
        self.speech_buffer = []
        self.silence_counter = 0
        self.is_speaking = False
        
        # State for interrupt detection
        self.ai_is_speaking = False
        self.interrupt_callback = None
        
        logger.info(f"Silero VAD initialized: {self.chunk_size_samples} samples per chunk")
    
    async def process_audio_chunk(self, audio_chunk: bytes) -> Dict:
        """
        Process incoming audio chunk for VAD
        Returns: {
            'is_speech': bool,
            'speech_prob': float,
            'should_interrupt': bool,
            'final_transcript': bytes or None
        }
        """
        try:
            # Convert bytes to float tensor
            audio_float = np.frombuffer(audio_chunk, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Resample if needed (VAD expects 16kHz)
            if len(audio_float) != self.chunk_size_samples:
                audio_float = self._resample_chunk(audio_float)
            
            # Run VAD using official Silero VAD API
            speech_prob = self.model(torch.from_numpy(audio_float), self.sample_rate).item()
            
            # Add to buffer
            self.audio_buffer.append(audio_float)
            
            result = {
                'is_speech': speech_prob >= self.speech_threshold,
                'speech_prob': speech_prob,
                'should_interrupt': False,
                'final_transcript': None
            }
            
            # Handle speech state transitions
            if result['is_speech']:
                self.silence_counter = 0
                
                if not self.is_speaking:
                    # Speech started
                    self.is_speaking = True
                    self.speech_buffer = [audio_float]
                    
                    # Check for interrupt
                    if self.ai_is_speaking:
                        result['should_interrupt'] = True
                        await self._trigger_interrupt()
                else:
                    # Continue collecting speech
                    self.speech_buffer.append(audio_float)
            else:
                # Silence detected
                if self.is_speaking:
                    self.silence_counter += self.chunk_size_ms
                    self.speech_buffer.append(audio_float)
                    
                    # Check if end of speech
                    if self.silence_counter >= self.max_silence_duration_ms:
                        # Speech ended - prepare for transcription
                        self.is_speaking = False
                        
                        # Combine speech buffer
                        complete_audio = np.concatenate(self.speech_buffer)
                        
                        # Only process if speech was long enough
                        speech_duration_ms = len(complete_audio) * 1000 / self.sample_rate
                        if speech_duration_ms >= self.min_speech_duration_ms:
                            result['final_transcript'] = await self._prepare_for_whisper(complete_audio)
                        
                        # Reset buffers
                        self.speech_buffer = []
                        self.silence_counter = 0
            
            return result
            
        except Exception as e:
            logger.error(f"VAD processing error: {e}")
            return {
                'is_speech': False,
                'speech_prob': 0.0,
                'should_interrupt': False,
                'final_transcript': None
            }
    
    async def _trigger_interrupt(self):
        """Handle interrupt detection"""
        if self.interrupt_callback:
            await self.interrupt_callback()
    
    async def _prepare_for_whisper(self, audio_data: np.ndarray) -> bytes:
        """Prepare audio for Whisper transcription"""
        # Convert back to int16 for Whisper
        audio_int16 = (audio_data * 32768).astype(np.int16)
        return audio_int16.tobytes()
    
    def _resample_chunk(self, audio: np.ndarray) -> np.ndarray:
        """Simple resampling to match expected chunk size"""
        from scipy import signal
        
        current_samples = len(audio)
        if current_samples == self.chunk_size_samples:
            return audio
        
        # Resample to match chunk size
        resampled = signal.resample(audio, self.chunk_size_samples)
        return resampled.astype(np.float32)
    
    def set_ai_speaking_state(self, is_speaking: bool):
        """Update AI speaking state for interrupt detection"""
        self.ai_is_speaking = is_speaking
        logger.debug(f"AI speaking state: {is_speaking}")
    
    def set_interrupt_callback(self, callback: Callable):
        """Set interrupt callback function"""
        self.interrupt_callback = callback
    
    def get_speech_segments(self, audio_file: str) -> List:
        """
        Get speech timestamps from a complete audio file using official Silero VAD API
        Useful for post-processing or debugging
        """
        wav = self.read_audio(audio_file, sampling_rate=self.sample_rate)
        speech_timestamps = self.get_speech_timestamps(
            wav,
            self.model,
            sampling_rate=self.sample_rate,
            threshold=self.speech_threshold,
            min_speech_duration_ms=self.min_speech_duration_ms,
            max_speech_duration_s=float('inf'),
            return_seconds=True  # Return timestamps in seconds as per official docs
        )
        return speech_timestamps
    
    def get_stats(self) -> Dict:
        """Get VAD statistics for monitoring"""
        return {
            'is_speaking': self.is_speaking,
            'ai_is_speaking': self.ai_is_speaking,
            'buffer_size': len(self.audio_buffer),
            'speech_buffer_size': len(self.speech_buffer),
            'silence_counter_ms': self.silence_counter
        }
