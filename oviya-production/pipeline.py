#!/usr/bin/env python3
"""
Oviya Production Pipeline - Main Orchestrator

This is the main entry point that connects all four layers:
1. Emotion Detector - Analyzes user's emotional state
2. Brain (LLM) - Generates text + emotion label
3. Emotion Controller - Maps emotion to acoustic parameters
4. Voice (OpenVoiceV2) - Generates expressive speech

Usage:
    python pipeline.py
"""

import sys
from pathlib import Path

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from emotion_detector.detector import EmotionDetector
from brain.llm_brain import OviyaBrain
from emotion_controller.controller import EmotionController
from voice.openvoice_tts import HybridVoiceEngine
import time
from typing import Dict, Optional


class OviyaPipeline:
    """
    Main Oviya pipeline - orchestrates all four layers.
    
    Flow:
        User Input → Emotion Detector → Brain → Emotion Controller → Voice → Audio Output
    """
    
    def __init__(self):
        """Initialize all four layers."""
        print("🚀 Initializing Oviya Production Pipeline...")
        print("=" * 60)
        
        # Layer 1: Emotion Detector
        print("\n[1/4] Loading Emotion Detector...")
        self.emotion_detector = EmotionDetector(
            emotions_config_path="config/emotions.json"
        )
        
        # Layer 2: Brain (LLM)
        print("\n[2/4] Loading Brain (Qwen2.5:7B)...")
        self.brain = OviyaBrain(
            persona_config_path="config/oviya_persona.json"
        )
        
        # Layer 3: Emotion Controller
        print("\n[3/4] Loading Emotion Controller...")
        self.emotion_controller = EmotionController(
            emotions_config_path="config/emotions.json"
        )
        
        # Layer 4: Voice (Hybrid: CSM + OpenVoiceV2)
        print("\n[4/4] Loading Voice (Hybrid: CSM + OpenVoiceV2)...")
        self.tts = HybridVoiceEngine(
            csm_url="http://localhost:6006/generate",  # Your Vast.ai CSM service
            default_engine="auto"  # Auto-select best engine
        )
        
        print("\n" + "=" * 60)
        print("✅ Oviya Pipeline Ready!")
        print("=" * 60)
        
        # Conversation history
        self.conversation_history = []
    
    def process(
        self,
        user_message: str,
        user_emotion: Optional[str] = None
    ) -> Dict:
        """
        Process user input through the full pipeline.
        
        Args:
            user_message: User's input text
            user_emotion: Pre-detected user emotion (optional, will detect if not provided)
        
        Returns:
            Dict with text, emotion, audio, timing info
        """
        start_time = time.time()
        
        print(f"\n{'='*60}")
        print(f"User: {user_message}")
        print(f"{'='*60}")
        
        # Step 1: Detect user emotion (if not provided)
        if user_emotion is None:
            print("\n🔍 [Emotion Detector] Analyzing user emotion...")
            emotion_result = self.emotion_detector.detect_emotion(user_message)
            user_emotion = emotion_result["emotion"]
            print(f"  Detected: {emotion_result['emotion']}")
            print(f"  Intensity: {emotion_result['intensity']:.2f}")
            print(f"  Confidence: {emotion_result['confidence']:.2f}")
            print(f"  Keywords: {emotion_result['matched_keywords']}")
        else:
            print(f"\n🔍 [Emotion Detector] Using provided emotion: {user_emotion}")
        
        emotion_detect_time = time.time() - start_time
        
        # Step 2: Check for safety issues
        safety_issue = self.brain.detect_safety_issue(user_message)
        if safety_issue:
            print(f"🚨 Safety issue detected: {safety_issue}")
            brain_output = self.brain.get_safety_response(safety_issue)
        else:
            # Step 3: Brain - Generate response + emotion
            print("\n🧠 [Brain] Thinking...")
            brain_output = self.brain.think(
                user_message,
                user_emotion=user_emotion,
                conversation_history=self.conversation_history[-3:]  # Last 3 turns
            )
        
        think_time = time.time() - start_time - emotion_detect_time
        
        print(f"  Text: {brain_output['text']}")
        print(f"  Emotion: {brain_output['emotion']}")
        print(f"  Intensity: {brain_output['intensity']}")
        print(f"  ⚡ Think time: {think_time*1000:.0f}ms")
        
        # Step 4: Emotion Controller - Map to acoustic params
        print(f"\n❤️ [Emotion Controller] Mapping emotion...")
        emotion_params = self.emotion_controller.map_emotion(
            brain_output['emotion'],
            intensity=brain_output['intensity']
        )
        
        print(f"  Style Token: {emotion_params['style_token']}")
        print(f"  Pitch: {emotion_params['pitch_scale']:.2f}")
        print(f"  Rate: {emotion_params['rate_scale']:.2f}")
        print(f"  Energy: {emotion_params['energy_scale']:.2f}")
        
        # Step 5: Voice - Generate speech with conversation context
        print(f"\n🗣️ [Voice] Generating speech...")
        try:
            # Prepare conversation context for CSM
            conversation_context = self._prepare_conversation_context()
            
            audio = self.tts.generate(
                text=brain_output['text'],
                emotion_params=emotion_params,
                speaker_id="oviya_v1",
                conversation_context=conversation_context
            )
            
            duration = audio.shape[0] / 24000  # Default sample rate
            print(f"  Duration: {duration:.2f}s")
            print(f"  Samples: {audio.shape[0]}")
        except Exception as e:
            print(f"  ❌ Voice generation failed: {e}")
            audio = None
        
        total_time = time.time() - start_time
        print(f"\n⚡ Total time: {total_time*1000:.0f}ms")
        
        # Update conversation history
        self.conversation_history.append({
            "user": user_message,
            "user_emotion": user_emotion,
            "oviya": brain_output['text'],
            "oviya_emotion": brain_output['emotion']
        })
        
        # Keep only last 5 turns
        if len(self.conversation_history) > 5:
            self.conversation_history = self.conversation_history[-5:]
        
        return {
            "text": brain_output['text'],
            "emotion": brain_output['emotion'],
            "user_emotion": user_emotion,
            "emotion_params": emotion_params,
            "audio": audio,
            "emotion_detect_time_ms": emotion_detect_time * 1000,
            "think_time_ms": think_time * 1000,
            "total_time_ms": total_time * 1000,
            "timestamp": time.time()
        }
    
    def _prepare_conversation_context(self) -> list:
        """Prepare conversation context for CSM."""
        if not self.conversation_history:
            return []
        
        # Convert conversation history to CSM format
        context = []
        for turn in self.conversation_history[-3:]:  # Last 3 turns
            # CSM expects Segment objects with text, speaker, audio
            # For now, we'll use simplified format
            context.append({
                "text": turn["oviya"],
                "speaker": 0,  # Oviya is speaker 0
                "emotion": turn["oviya_emotion"]
            })
        
        return context
    
    def save_response_audio(self, audio, output_path: str = "oviya_response.wav"):
        """Save generated audio to file."""
        if audio is not None:
            self.tts.save_audio(audio, output_path)
        else:
            print("⚠️ No audio to save")


def main():
    """Main function - run interactive mode."""
    # Initialize pipeline
    pipeline = OviyaPipeline()
    
    print("\n" + "="*60)
    print("🎤 Oviya Interactive Mode")
    print("="*60)
    print("Type your message and press Enter.")
    print("Type 'quit' or 'exit' to stop.")
    print("="*60 + "\n")
    
    # Test cases for demo
    test_cases = [
        "I'm feeling really stressed about work today",
        "I got promoted! I'm so excited!",
        "I'm feeling sad and alone",
        "I'm frustrated with everything",
        "Just checking in, how are you?"
    ]
    
    print("🧪 Running test cases...\n")
    
    for i, test_message in enumerate(test_cases, 1):
        response = pipeline.process(test_message)
        
        # Save audio
        if response['audio'] is not None:
            output_path = f"output/test_{i}.wav"
            Path("output").mkdir(exist_ok=True)
            pipeline.save_response_audio(response['audio'], output_path)
        
        print()
    
    print("\n✅ Test cases completed!")
    print("📁 Audio files saved to output/")
    
    # Interactive mode
    print("\n" + "="*60)
    print("💬 Interactive mode (Ctrl+C to exit)")
    print("="*60 + "\n")
    
    try:
        while True:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            # Process input
            response = pipeline.process(user_input)
            
            # Save audio
            if response['audio'] is not None:
                output_path = "output/last_response.wav"
                Path("output").mkdir(exist_ok=True)
                pipeline.save_response_audio(response['audio'], output_path)
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")


if __name__ == "__main__":
    main()

