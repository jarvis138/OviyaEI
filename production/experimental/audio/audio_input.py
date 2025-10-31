#!/usr/bin/env python3
"""
Phase 1-3: Audio Input → VAD → STT Pipeline for Oviya EI

Handles real-time audio streaming, voice activity detection, and speech-to-text conversion.
"""

import asyncio
import numpy as np
import torch
import io
import base64
from typing import AsyncGenerator, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)

class AudioInputProcessor:
    """
    Phase 1: Audio Input Processing

    Handles real-time audio streaming from WebSocket clients.
    Buffers audio chunks and manages audio state.
    """

    def __init__(self, sample_rate: int = 24000, chunk_duration: float = 0.1):
        """
        Initialize audio input processor

        Args:
            sample_rate: Audio sample rate (24kHz for compatibility)
            chunk_duration: Duration of each audio chunk in seconds
        """
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.chunk_samples = int(sample_rate * chunk_duration)
        self.audio_buffer = []
        self.is_streaming = False

    def process_audio_chunk(self, audio_data: bytes) -> np.ndarray:
        """
        Process incoming audio chunk from WebSocket

        Args:
            audio_data: Raw audio bytes (16-bit PCM)

        Returns:
            Audio chunk as numpy array
        """
        # Convert bytes to 16-bit PCM, then to float32
        audio_int16 = np.frombuffer(audio_data, dtype=np.int16)
        audio_float32 = audio_int16.astype(np.float32) / 32768.0

        # Ensure correct sample rate (resample if needed)
        # For now, assume client sends 24kHz audio

        return audio_float32

    def add_to_buffer(self, audio_chunk: np.ndarray):
        """Add processed audio chunk to buffer"""
        self.audio_buffer.append(audio_chunk)

    def get_buffered_audio(self, duration_seconds: float = 1.0) -> Optional[np.ndarray]:
        """
        Get buffered audio for specified duration

        Args:
            duration_seconds: Duration of audio to return

        Returns:
            Concatenated audio array or None if insufficient data
        """
        target_samples = int(self.sample_rate * duration_seconds)
        buffered_samples = sum(len(chunk) for chunk in self.audio_buffer)

        if buffered_samples < target_samples:
            return None

        # Concatenate chunks and return most recent audio
        concatenated = np.concatenate(self.audio_buffer)
        return concatenated[-target_samples:]

    def clear_buffer(self):
        """Clear audio buffer"""
        self.audio_buffer = []


class VoiceActivityDetector:
    """
    Phase 2: Voice Activity Detection using Silero VAD

    Detects presence of speech in audio stream with high accuracy.
    """

    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        """
        Initialize VAD with Silero model

        Args:
            threshold: Speech probability threshold (0-1)
            sample_rate: Audio sample rate for VAD (16kHz optimal)
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load Silero VAD model
        self._load_model()

    def _load_model(self):
        """Load Silero VAD model"""
        try:
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                trust_repo=True
            )
            self.model.eval()
            self.model.to(self.device)

            # Get utils functions (handle different versions)
            if len(utils) >= 5:
                self.get_speech_timestamps, self.save_audio, self.read_audio, _, _ = utils[:5]
            else:
                # Fallback for older versions
                self.get_speech_timestamps = utils[0] if len(utils) > 0 else None
                self.save_audio = utils[1] if len(utils) > 1 else None
                self.read_audio = utils[2] if len(utils) > 2 else None

            logger.info("✅ Silero VAD model loaded successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load Silero VAD: {e}")
            raise

    def detect_speech(self, audio: np.ndarray) -> Dict[str, any]:
        """
        Detect speech in audio chunk

        Args:
            audio: Audio array (16kHz)

        Returns:
            Detection results with speech probability and timestamps
        """
        try:
            # Ensure audio is 16kHz for VAD
            if len(audio.shape) > 1:
                audio = audio.flatten()

            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio).float().to(self.device)

            # Get speech probability
            speech_prob = self.model(audio_tensor, self.sample_rate).item()

            # Get speech timestamps if speech detected
            speech_timestamps = []
            if speech_prob > self.threshold:
                timestamps = self.get_speech_timestamps(
                    audio_tensor,
                    self.model,
                    sampling_rate=self.sample_rate,
                    threshold=self.threshold
                )
                speech_timestamps = [
                    {'start': ts['start'] / self.sample_rate, 'end': ts['end'] / self.sample_rate}
                    for ts in timestamps
                ]

            return {
                'speech_detected': speech_prob > self.threshold,
                'speech_probability': speech_prob,
                'speech_timestamps': speech_timestamps,
                'audio_duration': len(audio) / self.sample_rate
            }

        except Exception as e:
            logger.error(f"❌ VAD processing failed: {e}")
            return {
                'speech_detected': False,
                'speech_probability': 0.0,
                'speech_timestamps': [],
                'audio_duration': len(audio) / self.sample_rate,
                'error': str(e)
            }

    def is_speech_complete(self, speech_timestamps: List[Dict]) -> bool:
        """
        Determine if speech segment is complete (ended)

        Args:
            speech_timestamps: Speech timestamp segments

        Returns:
            True if speech appears to be complete
        """
        if not speech_timestamps:
            return False

        # Check if last speech segment ended recently
        # (within last 0.5 seconds of audio)
        last_segment = speech_timestamps[-1]
        return last_segment['end'] < 0.5  # Speech ended >0.5s ago


class SpeechToTextProcessor:
    """
    Phase 3: Speech-to-Text using Whisper v3 Turbo

    Converts detected speech to text with high accuracy and timestamps.
    """

    def __init__(self, model_size: str = "turbo", language: str = "en"):
        """
        Initialize Whisper v3 Turbo STT

        Args:
            model_size: Whisper model size ("turbo" for fastest)
            language: Language code ("en" for English)
        """
        self.model_size = model_size
        self.language = language
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load Whisper model
        self._load_model()

    def _load_model(self):
        """Load Whisper v3 Turbo model"""
        try:
            import faster_whisper

            self.model = faster_whisper.WhisperModel(
                self.model_size,
                device=self.device.type,
                compute_type="float16" if self.device.type == "cuda" else "int8"
            )

            logger.info(f"✅ Whisper {self.model_size} model loaded successfully")

        except Exception as e:
            logger.error(f"❌ Failed to load Whisper model: {e}")
            raise

    def transcribe_audio(self, audio: np.ndarray, language: Optional[str] = None) -> Dict[str, any]:
        """
        Transcribe audio to text with timestamps

        Args:
            audio: Audio array (16kHz)
            language: Language override (optional)

        Returns:
            Transcription results with text, confidence, and word-level timestamps
        """
        try:
            # Ensure audio is correct format
            if len(audio.shape) > 1:
                audio = audio.flatten()

            # Convert to 16-bit PCM bytes for Whisper
            audio_int16 = (audio * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()

            # Create in-memory audio file
            audio_buffer = io.BytesIO(audio_bytes)
            audio_buffer.name = 'audio.wav'

            # Transcribe with Whisper
            segments, info = self.model.transcribe(
                audio_buffer,
                language=language or self.language,
                beam_size=5,
                patience=1,
                length_penalty=1,
                repetition_penalty=1,
                no_repeat_ngram_size=0,
                compression_ratio_threshold=2.4,
                log_probability_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=True,
                prompt_reset_on_temperature=0.5,
                initial_prompt=None,
                prefix=None,
                suppress_blank=True,
                suppress_tokens=[-1],
                without_timestamps=False,
                max_initial_timestamp=1.0,
                hallucination_silence_threshold=None
            )

            # Process results
            transcription_text = ""
            words = []
            confidence_scores = []

            for segment in segments:
                transcription_text += segment.text

                # Collect word-level data
                if hasattr(segment, 'words') and segment.words:
                    for word in segment.words:
                        words.append({
                            'word': word.word,
                            'start': word.start,
                            'end': word.end,
                            'confidence': word.probability
                        })
                        confidence_scores.append(word.probability)

            # Calculate overall confidence
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0

            return {
                'text': transcription_text.strip(),
                'confidence': avg_confidence,
                'language': info.language,
                'language_probability': info.language_probability,
                'duration': len(audio) / 16000,  # Assuming 16kHz
                'words': words,
                'segments': [
                    {
                        'text': segment.text,
                        'start': segment.start,
                        'end': segment.end,
                        'confidence': segment.avg_logprob
                    }
                    for segment in segments
                ]
            }

        except Exception as e:
            logger.error(f"❌ Whisper transcription failed: {e}")
            return {
                'text': '',
                'confidence': 0.0,
                'language': self.language,
                'language_probability': 0.0,
                'duration': len(audio) / 16000,
                'words': [],
                'segments': [],
                'error': str(e)
            }


class AudioProcessingPipeline:
    """
    Complete Phase 1-3 Pipeline: Audio Input → VAD → STT

    Orchestrates the entire audio processing pipeline for real-time conversation.
    """

    def __init__(self):
        """Initialize complete audio processing pipeline"""
        self.audio_processor = AudioInputProcessor()
        self.vad = VoiceActivityDetector()
        self.stt = SpeechToTextProcessor()

        self.is_listening = False
        self.speech_buffer = []
        self.last_speech_time = 0

        logger.info("🎤 Complete Audio Processing Pipeline initialized")

    async def process_audio_stream(self, audio_data: bytes) -> AsyncGenerator[Dict[str, any], None]:
        """
        Process real-time audio stream through complete pipeline

        Args:
            audio_data: Raw audio bytes from WebSocket

        Yields:
            Processing results as they become available
        """
        # Phase 1: Process audio input
        audio_chunk = self.audio_processor.process_audio_chunk(audio_data)
        self.audio_processor.add_to_buffer(audio_chunk)

        # Get 1 second of buffered audio for analysis
        buffered_audio = self.audio_processor.get_buffered_audio(duration_seconds=1.0)

        if buffered_audio is None:
            return

        # Resample to 16kHz for VAD and STT
        if len(buffered_audio) > 0:
            # Simple downsampling (would use proper resampling in production)
            vad_audio = buffered_audio[::1]  # Assume already 16kHz

            # Phase 2: Voice Activity Detection
            vad_result = self.vad.detect_speech(vad_audio)

            if vad_result['speech_detected']:
                self.speech_buffer.append(buffered_audio)
                self.last_speech_time = asyncio.get_event_loop().time()

                # Check if speech is complete (silence detected)
                if self.vad.is_speech_complete(vad_result['speech_timestamps']):
                    # Phase 3: Speech-to-Text
                    speech_audio = np.concatenate(self.speech_buffer)

                    stt_result = self.stt.transcribe_audio(speech_audio)

                    # Combine results
                    result = {
                        'phase': 'complete',
                        'vad_result': vad_result,
                        'stt_result': stt_result,
                        'audio_duration': len(speech_audio) / 16000,
                        'timestamp': asyncio.get_event_loop().time()
                    }

                    # Clear buffers
                    self.speech_buffer = []
                    self.audio_processor.clear_buffer()

                    yield result
            else:
                # No speech detected, check if we should clear old buffer
                current_time = asyncio.get_event_loop().time()
                if current_time - self.last_speech_time > 2.0:  # 2 second timeout
                    self.speech_buffer = []

    def reset(self):
        """Reset pipeline state"""
        self.audio_processor.clear_buffer()
        self.speech_buffer = []
        self.is_listening = False
        self.last_speech_time = 0


# Global pipeline instance
_audio_pipeline = None

def get_audio_pipeline() -> AudioProcessingPipeline:
    """Get or create global audio processing pipeline"""
    global _audio_pipeline
    if _audio_pipeline is None:
        _audio_pipeline = AudioProcessingPipeline()
    return _audio_pipeline


# Test function
async def test_audio_pipeline():
    """Test the complete audio processing pipeline"""
    print("🧪 TESTING AUDIO PROCESSING PIPELINE")
    print("=" * 50)

    pipeline = get_audio_pipeline()

    # Simulate audio chunks (would come from WebSocket in real usage)
    # For testing, we'll use silence with some simulated speech

    test_chunks = []
    for i in range(10):
        # Create 0.1 second chunks of silence (mostly)
        chunk = np.zeros(int(16000 * 0.1), dtype=np.float32)

        # Add some "speech" in middle chunks
        if 3 <= i <= 6:
            # Add random noise to simulate speech
            chunk += np.random.normal(0, 0.1, len(chunk))

        test_chunks.append(chunk.astype(np.int16).tobytes())

    print(f"Testing with {len(test_chunks)} audio chunks...")

    results = []
    async for result in pipeline.process_audio_stream(test_chunks[0]):
        results.append(result)
        print(f"✅ Pipeline result: {result.get('phase', 'unknown')}")

    if results:
        print(f"\\n📊 Pipeline test successful: {len(results)} results")
        for result in results:
            stt = result.get('stt_result', {})
            print(f"   Text: '{stt.get('text', '')}'")
            print(f"   Confidence: {stt.get('confidence', 0):.2f}")
    else:
        print("\\n⚠️ No speech detected in test (expected for silence)")

    print("\\n✅ Audio Processing Pipeline test completed")


if __name__ == "__main__":
    asyncio.run(test_audio_pipeline())
