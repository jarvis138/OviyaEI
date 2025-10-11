#!/usr/bin/env python3
"""
Integrated Silero VAD + Whisper ASR Pipeline
"""
import asyncio
import logging
from typing import AsyncGenerator, Dict, Optional, Callable
import time

from silero_vad_processor import SileroVADProcessor
from whisper_stream import WhisperStreamProcessor

logger = logging.getLogger(__name__)

class WhisperSileroASR:
    """
    Combined VAD + ASR pipeline for real-time speech recognition
    """
    
    def __init__(self, whisper_model_size: str = "small.en"):
        self.vad = SileroVADProcessor()
        self.whisper = WhisperStreamProcessor(model_size=whisper_model_size)
        self.transcription_queue = asyncio.Queue()
        self.is_processing = False
        
        # Performance tracking
        self.total_chunks = 0
        self.speech_chunks = 0
        self.interrupts_detected = 0
        self.transcriptions_completed = 0
        
        logger.info("WhisperSileroASR pipeline initialized")
    
    async def initialize(self):
        """Initialize both VAD and Whisper models"""
        await self.whisper.initialize()
        logger.info("ASR pipeline fully initialized")
    
    async def process_stream(self, audio_stream: AsyncGenerator[bytes, None]) -> AsyncGenerator[Dict, None]:
        """
        Process audio stream with VAD → Whisper pipeline
        
        Yields:
        - {'type': 'interrupt', 'latency_ms': float} - User interrupt detected
        - {'type': 'transcript', 'text': str, 'confidence': float, 'latency_ms': float} - Final transcription
        - {'type': 'partial', 'text': str, 'confidence': float} - Partial results (if enabled)
        """
        self.is_processing = True
        
        try:
            async for audio_chunk in audio_stream:
                if not self.is_processing:
                    break
                
                self.total_chunks += 1
                
                # VAD processing (30ms target)
                vad_result = await self.vad.process_audio_chunk(audio_chunk)
                
                # Handle interrupt immediately
                if vad_result['should_interrupt']:
                    self.interrupts_detected += 1
                    yield {
                        'type': 'interrupt',
                        'latency_ms': 30,  # VAD latency
                        'timestamp': time.time()
                    }
                
                # Track speech detection
                if vad_result['is_speech']:
                    self.speech_chunks += 1
                
                # Transcribe complete utterances
                if vad_result['final_transcript']:
                    audio_bytes = vad_result['final_transcript']
                    
                    # Run Whisper (200ms target)
                    transcription_result = await self.whisper.transcribe_audio(audio_bytes)
                    
                    if transcription_result['text'].strip():
                        self.transcriptions_completed += 1
                        
                        yield {
                            'type': 'transcript',
                            'text': transcription_result['text'],
                            'confidence': transcription_result['confidence'],
                            'latency_ms': transcription_result['latency_ms'],
                            'language': transcription_result['language'],
                            'segments': transcription_result['segments'],
                            'timestamp': time.time()
                        }
                        
                        logger.info(f"Transcribed: '{transcription_result['text'][:50]}...' "
                                  f"({transcription_result['latency_ms']:.1f}ms)")
        
        except Exception as e:
            logger.error(f"ASR pipeline error: {e}")
            yield {
                'type': 'error',
                'message': str(e),
                'timestamp': time.time()
            }
        
        finally:
            self.is_processing = False
    
    async def process_audio_file(self, audio_file_path: str) -> Dict:
        """
        Process a complete audio file
        Returns transcription result
        """
        try:
            # Read audio file
            with open(audio_file_path, 'rb') as f:
                audio_data = f.read()
            
            # Convert to int16 array
            import numpy as np
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # Process as stream
            async def audio_generator():
                # Split into chunks for processing
                chunk_size = 480  # 30ms at 16kHz
                for i in range(0, len(audio_array), chunk_size):
                    chunk = audio_array[i:i + chunk_size]
                    if len(chunk) == chunk_size:
                        yield chunk.tobytes()
            
            # Process stream
            results = []
            async for result in self.process_stream(audio_generator()):
                if result['type'] == 'transcript':
                    results.append(result)
            
            # Return best result
            if results:
                best_result = max(results, key=lambda x: x['confidence'])
                return best_result
            else:
                return {
                    'text': '',
                    'confidence': 0.0,
                    'latency_ms': 0.0,
                    'language': 'en'
                }
        
        except Exception as e:
            logger.error(f"File processing error: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'latency_ms': 0.0,
                'language': 'en',
                'error': str(e)
            }
    
    def set_ai_speaking_state(self, is_speaking: bool):
        """Update AI speaking state for interrupt detection"""
        self.vad.set_ai_speaking_state(is_speaking)
    
    def set_interrupt_callback(self, callback: Callable):
        """Set interrupt callback function"""
        self.vad.set_interrupt_callback(callback)
    
    def stop_processing(self):
        """Stop the processing pipeline"""
        self.is_processing = False
        logger.info("ASR pipeline stopped")
    
    def get_performance_stats(self) -> Dict:
        """Get comprehensive performance statistics"""
        whisper_stats = self.whisper.get_performance_stats()
        vad_stats = self.vad.get_stats()
        
        return {
            'total_chunks_processed': self.total_chunks,
            'speech_chunks_detected': self.speech_chunks,
            'interrupts_detected': self.interrupts_detected,
            'transcriptions_completed': self.transcriptions_completed,
            'speech_detection_rate': self.speech_chunks / max(self.total_chunks, 1),
            'interrupt_rate': self.interrupts_detected / max(self.speech_chunks, 1),
            'transcription_success_rate': self.transcriptions_completed / max(self.speech_chunks, 1),
            'whisper_stats': whisper_stats,
            'vad_stats': vad_stats,
            'is_processing': self.is_processing
        }
    
    def reset_stats(self):
        """Reset all performance statistics"""
        self.total_chunks = 0
        self.speech_chunks = 0
        self.interrupts_detected = 0
        self.transcriptions_completed = 0
        self.whisper.reset_stats()
        logger.info("ASR pipeline stats reset")


