"""
Real-time conversation system for Oviya - ChatGPT-style voice mode
Integrates WhisperX real-time input with Oviya's brain and voice output
"""

import os
import sys
from pathlib import Path
import time
import numpy as np
from typing import Optional, Dict

sys.path.insert(0, str(Path(__file__).parent))

from .voice.realtime_input import RealTimeVoiceInput, AudioStreamSimulator
from .brain.llm_brain import OviyaBrain
from .emotion_controller.controller import EmotionController
from .voice.openvoice_tts import HybridVoiceEngine
import torch
import torchaudio


class RealTimeConversation:
    """
    Real-time conversation system for Oviya
    Flow: User speaks → WhisperX transcription → LLM brain → CSM voice response
    """
    
    def __init__(
        self,
        ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate"),
        csm_url: str = os.getenv("CSM_URL", "http://localhost:19517/generate")
    ):
        print("=" * 60)
        print("🎙️  Initializing Oviya Real-Time Conversation System")
        print("=" * 60)
        
        # Layer 1: Real-Time Voice Input (WhisperX)
        print("\n[1/4] Loading Real-Time Voice Input (WhisperX)...")
        self.voice_input = RealTimeVoiceInput()
        self.voice_input.initialize_models()
        print("✅ Real-time voice input ready")
        
        # Layer 2: Brain (LLM + Emotional Intelligence)
        print("\n[2/4] Loading Brain (LLM + Emotional Intelligence)...")
        self.brain = OviyaBrain(ollama_url=ollama_url)
        print("✅ Brain ready")
        
        # Layer 3: Emotion Controller (49-emotion library)
        print("\n[3/4] Loading Emotion Controller (49 emotions)...")
        self.emotion_controller = EmotionController()
        print("✅ Emotion controller ready")
        
        # Layer 4: Voice Output (CSM Hybrid Engine)
        print("\n[4/4] Loading Voice Output (CSM Hybrid Engine)...")
        self.voice_output = HybridVoiceEngine(
            csm_url=csm_url,
            default_engine="auto"
        )
        print("✅ Voice output ready")
        
        # Conversation state
        self.conversation_active = False
        self.turn_count = 0
        
        print("\n" + "=" * 60)
        print("✅ Oviya Real-Time Conversation System Ready!")
        print("=" * 60)
    
    def start_conversation(self):
        """Start a real-time conversation session"""
        print("\n🎙️  Starting conversation...")
        print("   Speak naturally - Oviya will respond with emotional voice")
        print("   Press Ctrl+C to stop\n")
        
        self.conversation_active = True
        self.turn_count = 0
        
        # Start voice input with callback
        self.voice_input.start_recording(callback=self._on_user_speech)
        
        try:
            # Keep conversation alive
            while self.conversation_active:
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n\n⏹️  Stopping conversation...")
            self.stop_conversation()
    
    def _on_user_speech(self, transcription_result: Dict):
        """
        Callback when user speech is transcribed
        Processes through brain and generates voice response
        """
        self.turn_count += 1
        
        user_text = transcription_result["text"]
        word_timestamps = transcription_result["word_timestamps"]
        
        print(f"\n{'='*60}")
        print(f"Turn {self.turn_count}")
        print(f"{'='*60}")
        print(f"👤 User: {user_text}")
        print(f"   Duration: {transcription_result['duration']:.2f}s")
        print(f"   Words: {len(word_timestamps)}")
        
        # Analyze user emotion from speech timing and content
        user_emotion = self._analyze_user_emotion(user_text, word_timestamps)
        print(f"   Detected emotion: {user_emotion}")
        
        # Process through brain
        print("\n🧠 Oviya thinking...")
        brain_response = self.brain.think(user_text, user_emotion)
        
        print(f"💭 Oviya: {brain_response['text']}")
        print(f"   Emotion: {brain_response['emotion']} (intensity: {brain_response['intensity']:.2f})")
        print(f"   Prosodic text: {brain_response['prosodic_text'][:80]}...")
        
        # Map emotion to acoustic parameters
        emotion_params = self.emotion_controller.map_emotion(
            brain_response["emotion"],
            brain_response["intensity"],
            brain_response.get("contextual_modifiers")
        )
        
        # Generate voice response
        print("\n🎤 Generating voice response...")
        try:
            audio = self.voice_output.generate(
                text=brain_response["text"],
                emotion_params=emotion_params,
                speaker_id="oviya_v1",
                prosodic_text=brain_response["prosodic_text"],
                emotional_state=brain_response["emotional_state"]
            )
            
            # Save audio
            output_path = f"output/realtime/turn_{self.turn_count}.wav"
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(output_path, audio.unsqueeze(0), 24000)
            print(f"✅ Voice response saved: {output_path}")
            
            # In production, this would play audio in real-time
            # For now, we just save it
            
        except Exception as e:
            print(f"❌ Voice generation error: {e}")
        
        print(f"{'='*60}\n")
    
    def _analyze_user_emotion(self, text: str, word_timestamps: list) -> str:
        """
        Analyze user emotion from text and speech timing
        Uses word timestamps to detect emotional cues
        """
        text_lower = text.lower()
        
        # Calculate speech rate (words per second)
        if word_timestamps:
            duration = word_timestamps[-1]["end"] - word_timestamps[0]["start"]
            words_per_second = len(word_timestamps) / duration if duration > 0 else 0
            
            # Fast speech often indicates excitement or anxiety
            if words_per_second > 3.5:
                if any(word in text_lower for word in ["amazing", "wow", "great", "awesome"]):
                    return "excited"
                else:
                    return "anxious"
            
            # Slow speech often indicates sadness or thoughtfulness
            elif words_per_second < 2.0:
                if any(word in text_lower for word in ["sad", "tired", "lonely", "depressed"]):
                    return "sad"
                else:
                    return "thoughtful"
        
        # Emotion keywords
        if any(word in text_lower for word in ["sad", "depressed", "lonely", "hurt"]):
            return "sad"
        elif any(word in text_lower for word in ["angry", "mad", "furious", "annoyed"]):
            return "angry"
        elif any(word in text_lower for word in ["worried", "anxious", "nervous", "scared"]):
            return "anxious"
        elif any(word in text_lower for word in ["happy", "excited", "great", "amazing"]):
            return "excited"
        elif any(word in text_lower for word in ["?", "how", "what", "why", "when"]):
            return "curious"
        else:
            return "neutral"
    
    def stop_conversation(self):
        """Stop the conversation session"""
        self.conversation_active = False
        final_result = self.voice_input.stop_recording()
        
        # Get conversation statistics
        context = self.voice_input.get_conversation_context()
        
        print("\n" + "=" * 60)
        print("📊 Conversation Summary")
        print("=" * 60)
        print(f"Total turns: {self.turn_count}")
        print(f"Total duration: {context['total_duration']:.2f}s")
        print(f"Total words: {len(context['word_timestamps'])}")
        print("=" * 60)
    
    def simulate_conversation(self, test_messages: list):
        """
        Simulate a conversation with pre-defined messages
        Useful for testing without actual audio input
        """
        print("\n🎭 Simulating conversation...")
        
        for i, message in enumerate(test_messages, 1):
            print(f"\n{'='*60}")
            print(f"Simulated Turn {i}")
            print(f"{'='*60}")
            print(f"👤 User (simulated): {message}")
            
            # Create simulated transcription result
            simulated_result = {
                "text": message,
                "duration": 2.0,
                "word_timestamps": [
                    {"word": word, "start": i*0.3, "end": (i+1)*0.3, "confidence": 0.95}
                    for i, word in enumerate(message.split())
                ]
            }
            
            # Process through pipeline
            self._on_user_speech(simulated_result)
            
            # Wait between turns
            time.sleep(1)
        
        print("\n✅ Simulated conversation complete!")


# Test scenarios
def test_realtime_conversation():
    """Test real-time conversation system with simulated input"""
    print("Testing Oviya Real-Time Conversation System\n")
    
    # Initialize system
    conversation = RealTimeConversation()
    
    # Test with simulated messages
    test_messages = [
        "Hey Oviya, how are you doing today?",
        "I'm feeling a bit anxious about my exam tomorrow.",
        "Can you help me feel better?",
        "Thank you so much, that really helps!",
        "What's the weather like today?"
    ]
    
    conversation.simulate_conversation(test_messages)


if __name__ == "__main__":
    test_realtime_conversation()

