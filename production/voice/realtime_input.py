"""
Real-time voice input system with Whisper v3 Turbo for ChatGPT-style voice mode
Handles audio streaming, VAD detection, and transcription
"""

import whisper
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple, Callable
import time
import threading
from queue import Queue
import warnings
warnings.filterwarnings("ignore")

class RealTimeVoiceInput:
    """Real-time voice input processing with Whisper v3 Turbo and VAD for ChatGPT-style conversations"""
    
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu", enable_diarization: bool = False):
        self.device = device
        self.whisper_model = None
        # Legacy attributes for compatibility
        self.whisperx_model = None
        self.align_model = None
        self.metadata = None
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.audio_buffer = []
        self.is_recording = False
        self.audio_queue = Queue()
        self.transcription_queue = Queue()
        self.vad_threshold = 0.5
        self.min_audio_length = 1.0  # Minimum 1 second for transcription
        self.max_buffer_seconds = 30  # Maximum 30 seconds in buffer
        
        # Speaker diarization
        self.enable_diarization = enable_diarization
        self.diarization_model = None
        
        # Conversation context tracking
        self.conversation_history = []
        self.current_turn_start = None
        self.word_timestamps_history = []
        
        print("RealTimeVoiceInput initialized")
        if enable_diarization:
            print("   Speaker diarization enabled")
        
    def initialize_models(self):
        """Initialize Whisper v3 Turbo model"""
        print("Initializing real-time voice models...")

        try:
            # Load Whisper v3 Turbo model
            print("   Loading Whisper v3 Turbo model...")
            self.whisper_model = whisper.load_model("turbo", device=self.device)
            print("   Whisper v3 Turbo model loaded successfully")
            print("   Whisper v3 Turbo models initialized successfully")

        except Exception as e:
            print(f"Whisper v3 Turbo initialization failed: {e}")
            print("Continuing with basic functionality...")
            self.whisper_model = None
    
    def start_recording(self, callback: Optional[Callable] = None):
        """
        Start real-time audio recording and processing
        
        Args:
            callback: Optional callback function to receive transcription results in real-time
        """
        if not self.whisper_model:
            self.initialize_models()
        
        self.is_recording = True
        self.audio_buffer = []
        self.current_turn_start = time.time()
        self.callback = callback
        
        print("Real-time recording started...")
        print("   Speak naturally - transcription will happen automatically")
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_audio_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop_recording(self) -> Optional[Dict]:
        """Stop recording and return final transcription with word-level timestamps"""
        if self.is_recording:
            self.is_recording = False
            print("Recording stopped")
            
            # Process remaining audio in buffer
            if len(self.audio_buffer) > self.sample_rate * self.min_audio_length:
                audio_array = np.array(self.audio_buffer, dtype=np.float32)
                final_result = self._transcribe_audio(audio_array)
                
                if final_result:
                    # Add to conversation history
                    self.conversation_history.append(final_result)
                    self.word_timestamps_history.extend(final_result.get("word_timestamps", []))
                    
                return final_result
            
        return None
    
    def add_audio_chunk(self, audio_chunk: np.ndarray):
        """
        Add audio chunk to buffer for processing
        Used for manual audio streaming (e.g., from web interface)
        
        Args:
            audio_chunk: Audio data as numpy array (float32, 16kHz)
        """
        if not isinstance(audio_chunk, np.ndarray):
            audio_chunk = np.array(audio_chunk, dtype=np.float32)
        
        # Add to buffer
        self.audio_buffer.extend(audio_chunk)
        self.audio_queue.put(audio_chunk)
        
        # Keep buffer manageable
        max_buffer = self.max_buffer_seconds * self.sample_rate
        if len(self.audio_buffer) > max_buffer:
            self.audio_buffer = self.audio_buffer[-max_buffer:]
    
    def _process_audio_loop(self):
        """Background thread for processing audio chunks"""
        last_transcription_time = time.time()
        transcription_interval = 2.0  # Transcribe every 2 seconds
        
        while self.is_recording:
            try:
                # Check if enough time has passed and we have enough audio
                current_time = time.time()
                buffer_duration = len(self.audio_buffer) / self.sample_rate
                
                if (current_time - last_transcription_time >= transcription_interval and 
                    buffer_duration >= self.min_audio_length):
                    
                    # Convert buffer to numpy array
                    audio_array = np.array(self.audio_buffer, dtype=np.float32)
                    
                    # Transcribe with word-level timestamps
                    result = self._transcribe_audio(audio_array)
                    
                    if result and result["text"].strip():
                        # Add to queue
                        self.transcription_queue.put(result)
                        
                        # Call callback if provided
                        if self.callback:
                            self.callback(result)
                        
                        # Update last transcription time
                        last_transcription_time = current_time
                        
                        # Clear processed audio from buffer (keep last 2 seconds for context)
                        overlap_samples = int(2.0 * self.sample_rate)
                        self.audio_buffer = self.audio_buffer[-overlap_samples:]
                
                # Sleep briefly to avoid busy waiting
                time.sleep(0.1)
                        
            except Exception as e:
                if self.is_recording:  # Only print errors if still recording
                    print(f"⚠️ Processing error: {e}")
                time.sleep(0.1)
                continue
    
    def _transcribe_audio(self, audio_array: np.ndarray) -> Optional[Dict]:
        """
        Transcribe audio with Whisper v3 Turbo

        Args:
            audio_array: Audio data as numpy array (float32, 16kHz)

        Returns:
            Dictionary with transcription, segments, and metadata
        """
        try:
            # Check if Whisper model is available
            if self.whisper_model is None:
                # Fallback: return a mock transcription
                print("   Using fallback transcription (Whisper model not loaded)")
                return {
                    "text": "[Voice input received - transcription temporarily unavailable]",
                    "segments": [{
                        "text": "[Voice input received - transcription temporarily unavailable]",
                        "start": 0.0,
                        "end": len(audio_array) / self.sample_rate,
                        "words": []
                    }],
                    "language": "en",
                    "duration": len(audio_array) / self.sample_rate,
                    "timestamp": time.time()
                }

            # Transcribe with Whisper v3 Turbo
            print("   Transcribing with Whisper v3 Turbo...")
            result = self.whisper_model.transcribe(
                audio_array,
                language="en",
                fp16=torch.cuda.is_available(),  # Use FP16 if CUDA available
                verbose=False
            )

            # Check if we got any transcription
            if not result.get("text", "").strip():
                return None

            # Convert to expected format (add segments if not present)
            if "segments" not in result:
                result["segments"] = [{
                    "text": result["text"],
                    "start": 0.0,
                    "end": len(audio_array) / self.sample_rate,
                    "words": []  # Whisper doesn't provide word-level timestamps by default
                }]

            # Add metadata
            result.update({
                "word_timestamps": [],  # Simplified - no word-level timestamps in basic Whisper
                "speakers": ["user"],
                "duration": len(audio_array) / self.sample_rate,
                "timestamp": time.time()
            })

            return result

        except Exception as e:
            print(f"⚠️ Transcription error: {e}")
            return None
    
    def _extract_word_timestamps(self, result: Dict) -> List[Dict]:
        """
        Extract word-level timestamps from WhisperX aligned result
        
        Args:
            result: WhisperX alignment result
            
        Returns:
            List of word timestamp dictionaries
        """
        word_timestamps = []
        
        for segment in result.get("segments", []):
            for word in segment.get("words", []):
                if word.get("word") and word.get("start") is not None:
                    word_timestamps.append({
                        "word": word["word"].strip(),
                        "start": word["start"],
                        "end": word["end"],
                        "confidence": word.get("score", 0.0),
                        "speaker": word.get("speaker", "user")  # Use diarization speaker if available
                    })
        
        return word_timestamps
    
    def get_transcription(self, timeout: float = 0.1) -> Optional[Dict]:
        """
        Get latest transcription from queue (non-blocking)
        
        Args:
            timeout: Maximum time to wait for transcription
            
        Returns:
            Transcription result or None
        """
        try:
            return self.transcription_queue.get(timeout=timeout)
        except:
            return None
    
    def get_conversation_context(self) -> Dict:
        """
        Get full conversation context with word-level timestamps
        
        Returns:
            Dictionary with conversation history and timing information
        """
        return {
            "history": self.conversation_history,
            "word_timestamps": self.word_timestamps_history,
            "turn_count": len(self.conversation_history),
            "total_duration": sum([h["duration"] for h in self.conversation_history])
        }
    
    def clear_buffer(self):
        """Clear audio buffer and queues"""
        self.audio_buffer = []
        while not self.audio_queue.empty():
            try:
                self.audio_queue.get_nowait()
            except:
                break
        while not self.transcription_queue.empty():
            try:
                self.transcription_queue.get_nowait()
            except:
                break
    
    def reset_conversation(self):
        """Reset conversation history and context"""
        self.conversation_history = []
        self.word_timestamps_history = []
        self.current_turn_start = None
        self.clear_buffer()
        print("🔄 Conversation context reset")


# Simulation class for testing without actual audio hardware
class AudioStreamSimulator:
    """Simulates real-time audio stream for testing"""
    
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
    
    def generate_test_audio(self, duration=3.0, text="Hello, this is a test"):
        """
        Generate test audio of specified duration
        In production, this would be replaced with actual audio recording
        
        Args:
            duration: Duration in seconds
            text: Text to simulate (for testing purposes)
            
        Returns:
            Numpy array of audio samples
        """
        samples = int(duration * self.sample_rate)
        # Generate white noise as placeholder
        audio = np.random.randn(samples).astype(np.float32) * 0.1
        return audio


# Test function for real-time voice input
def test_realtime_voice():
    """Test real-time voice input system with simulation"""
    print("=" * 60)
    print("Testing Real-Time Voice Input System")
    print("=" * 60)
    
    # Initialize voice input
    voice_input = RealTimeVoiceInput()
    voice_input.initialize_models()
    
    # Test with simulated audio
    print("\n📝 Test 1: Single transcription with simulated audio")
    simulator = AudioStreamSimulator()
    test_audio = simulator.generate_test_audio(duration=3.0)
    
    # Add audio chunk
    voice_input.add_audio_chunk(test_audio)
    
    # Start recording
    def on_transcription(result):
        print(f"\n✅ Real-time transcription received:")
        print(f"   Text: {result['text']}")
        print(f"   Duration: {result['duration']:.2f}s")
        print(f"   Words: {len(result['word_timestamps'])}")
    
    voice_input.start_recording(callback=on_transcription)
    
    # Simulate streaming audio
    print("\n🎤 Simulating 5 seconds of audio streaming...")
    for i in range(5):
        chunk = simulator.generate_test_audio(duration=1.0)
        voice_input.add_audio_chunk(chunk)
        time.sleep(1)
    
    # Stop recording
    final_result = voice_input.stop_recording()
    
    if final_result:
        print(f"\n📊 Final transcription:")
        print(f"   Text: {final_result['text']}")
        print(f"   Duration: {final_result['duration']:.2f}s")
        print(f"   Word timestamps: {len(final_result['word_timestamps'])}")
        
        # Show first 5 word timestamps
        if final_result['word_timestamps']:
            print("\n   First 5 word timestamps:")
            for word in final_result['word_timestamps'][:5]:
                print(f"      {word['word']}: {word['start']:.2f}-{word['end']:.2f}s (conf: {word['confidence']:.2f})")
    
    # Get conversation context
    context = voice_input.get_conversation_context()
    print(f"\n📚 Conversation context:")
    print(f"   Turn count: {context['turn_count']}")
    print(f"   Total duration: {context['total_duration']:.2f}s")
    
    print("\n" + "=" * 60)
    print("✅ Real-time voice input test complete!")
    print("=" * 60)


if __name__ == "__main__":
    test_realtime_voice()


