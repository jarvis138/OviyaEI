#!/usr/bin/env python3
"""
Whisper ASR integration with faster-whisper
"""
from faster_whisper import WhisperModel
import logging
import time
from typing import List, Dict, Optional, Tuple
import asyncio

logger = logging.getLogger(__name__)

class WhisperStreamProcessor:
    """
    Whisper ASR processor optimized for real-time streaming
    """
    
    def __init__(self, model_size: str = "small.en", device: str = "auto"):
        self.model_size = model_size
        self.device = device
        self.model = None
        self._initialized = False
        
        # Performance settings
        self.compute_type = "int8"  # Faster inference
        self.num_workers = 2
        self.beam_size = 1  # Greedy for speed
        self.language = "en"
        
        # Latency tracking
        self.transcription_times = []
        
        logger.info(f"Whisper processor initialized: {model_size} on {device}")
    
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
                num_workers=self.num_workers,
                download_root="models/"
            )
            
            load_time = time.time() - start_time
            logger.info(f"Whisper model loaded in {load_time:.2f}s")
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper: {e}")
            raise
    
    async def transcribe_audio(self, audio_bytes: bytes, sample_rate: int = 16000) -> Dict:
        """
        Transcribe audio bytes to text
        Returns: {
            'text': str,
            'confidence': float,
            'latency_ms': float,
            'segments': List[Dict]
        }
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Convert bytes to numpy array
            import numpy as np
            audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Transcribe with Whisper
            segments, info = self.model.transcribe(
                audio_array,
                beam_size=self.beam_size,
                language=self.language,
                vad_filter=False,  # Already filtered by Silero VAD
                without_timestamps=True,
                condition_on_previous_text=False  # Faster for streaming
            )
            
            # Process segments
            text_parts = []
            all_segments = []
            total_confidence = 0
            
            for segment in segments:
                text_parts.append(segment.text.strip())
                all_segments.append({
                    'text': segment.text.strip(),
                    'start': getattr(segment, 'start', 0),
                    'end': getattr(segment, 'end', 0),
                    'confidence': getattr(segment, 'avg_logprob', 0)
                })
                total_confidence += getattr(segment, 'avg_logprob', 0)
            
            # Combine text
            full_text = " ".join(text_parts).strip()
            
            # Calculate metrics
            latency_ms = (time.time() - start_time) * 1000
            avg_confidence = total_confidence / len(all_segments) if all_segments else 0
            
            # Track latency
            self.transcription_times.append(latency_ms)
            if len(self.transcription_times) > 100:  # Keep last 100 measurements
                self.transcription_times.pop(0)
            
            result = {
                'text': full_text,
                'confidence': avg_confidence,
                'latency_ms': latency_ms,
                'segments': all_segments,
                'language': info.language if hasattr(info, 'language') else 'en',
                'language_probability': getattr(info, 'language_probability', 1.0)
            }
            
            logger.debug(f"Transcribed {len(audio_bytes)} bytes in {latency_ms:.1f}ms: '{full_text[:50]}...'")
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'latency_ms': (time.time() - start_time) * 1000,
                'segments': [],
                'language': 'en',
                'language_probability': 0.0
            }
    
    async def transcribe_stream(self, audio_stream) -> List[Dict]:
        """
        Transcribe a stream of audio chunks
        Returns list of transcription results
        """
        results = []
        
        async for audio_chunk in audio_stream:
            if audio_chunk:
                result = await self.transcribe_audio(audio_chunk)
                results.append(result)
        
        return results
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics"""
        if not self.transcription_times:
            return {
                'avg_latency_ms': 0,
                'min_latency_ms': 0,
                'max_latency_ms': 0,
                'total_transcriptions': 0
            }
        
        return {
            'avg_latency_ms': sum(self.transcription_times) / len(self.transcription_times),
            'min_latency_ms': min(self.transcription_times),
            'max_latency_ms': max(self.transcription_times),
            'total_transcriptions': len(self.transcription_times)
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self.transcription_times = []


