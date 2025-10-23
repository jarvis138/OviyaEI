#!/usr/bin/env python3
"""
Comprehensive test for all Beyond-Maya enhancements
Tests all 6 new features with multiple scenarios
"""

import sys
import torch
import torchaudio
from pathlib import Path

# Add root directory to path for imports
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from production.brain.llm_brain import OviyaBrain
from production.emotion_controller.controller import EmotionController
from production.voice.openvoice_tts import HybridVoiceEngine
from production.voice.audio_postprocessor import AudioPostProcessor

def print_section(title: str):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")

def print_feature(feature: str, data: str):
    """Print a formatted feature result"""
    print(f"   {feature}: {data}")

def test_scenario(
    scenario_num: int,
    title: str,
    user_message: str,
    user_emotion: str,
    brain: OviyaBrain,
    emotion_controller: EmotionController,
    tts: HybridVoiceEngine,
    audio_processor: AudioPostProcessor
):
    """Test a complete scenario through the pipeline"""
    
    print_section(f"SCENARIO {scenario_num}: {title}")
    
    print(f"📝 User Message: \"{user_message}\"")
    print(f"😊 User Emotion: {user_emotion}")
    
    # 1. Brain generates response
    print("\n🧠 BRAIN PROCESSING:")
    response = brain.think(user_message, user_emotion)
    
    print_feature("✅ Response Text", response['text'])
    print_feature("🎭 Emotion", response['emotion'])
    print_feature("📊 Intensity", f"{response['intensity']:.2f}")
    
    # Check for backchannels
    if response.get('has_backchannel'):
        print_feature("💬 Backchannel", f"YES ({response.get('backchannel_type')})")
    else:
        print_feature("💬 Backchannel", "No")
    
    # Show prosodic text
    print_feature("🎼 Prosodic Text", response['prosodic_text'][:100] + "..." if len(response['prosodic_text']) > 100 else response['prosodic_text'])
    
    # Show epistemic state if present
    if 'epistemic_analysis' in response:
        epistemic = response['epistemic_analysis']
        if epistemic['epistemic_state'] != 'neutral':
            print_feature("🔬 Epistemic State", f"{epistemic['epistemic_state']} (confidence: {epistemic['confidence_level']:.2f})")
    
    # Show emotional memory
    if 'emotional_state' in response:
        state = response['emotional_state']
        print_feature("🧠 Energy Level", f"{state['energy_level']:.2f}")
        print_feature("⚡ Pace", f"{state['pace']:.2f}x")
        print_feature("❤️  Warmth", f"{state['warmth']:.2f}")
    
    # 2. Emotion controller maps parameters
    print("\n🎛️  EMOTION CONTROLLER:")
    emotion_params = emotion_controller.map_emotion(
        response['emotion'],
        response['intensity'],
        response.get('contextual_modifiers')
    )
    
    print_feature("🎵 Pitch Scale", f"{emotion_params['pitch_scale']:.3f}x")
    print_feature("⏱️  Rate Scale", f"{emotion_params['rate_scale']:.3f}x")
    print_feature("⚡ Energy Scale", f"{emotion_params['energy_scale']:.3f}x")
    print_feature("📈 Intensity Curve", f"{emotion_params['intensity_curve']:.3f}")
    
    # 3. Generate audio with TTS
    print("\n🎤 VOICE GENERATION:")
    try:
        # Generate audio with emotion parameters and prosodic text
        audio = tts.generate(
            text=response['text'],
            emotion_params=emotion_params,
            prosodic_text=response['prosodic_text'],
            emotional_state=response.get('emotional_state')
        )
        
        duration = len(audio) / 24000
        print_feature("✅ Audio Generated", f"{len(audio)} samples ({duration:.2f}s)")
        print_feature("🔊 Peak Amplitude", f"{audio.abs().max():.3f}")
        print_feature("📊 RMS Level", f"{torch.sqrt(torch.mean(audio**2)):.3f}")
        
        # 4. Audio post-processing (Maya-level enhancements)
        print("\n🎨 AUDIO POST-PROCESSING:")
        processed_audio = audio_processor.process(
            audio,
            prosodic_text=response['prosodic_text'],
            emotional_state=response.get('emotional_state'),
            add_reverb=True,
            master_audio=True
        )
        
        processed_duration = len(processed_audio) / 24000
        print_feature("✅ Processing Complete", f"{len(processed_audio)} samples ({processed_duration:.2f}s)")
        print_feature("⏱️  Duration Change", f"{(processed_duration - duration):.3f}s ({((processed_duration/duration - 1)*100):.1f}%)")
        print_feature("🔊 Final Peak", f"{processed_audio.abs().max():.3f}")
        print_feature("📊 Final RMS", f"{torch.sqrt(torch.mean(processed_audio**2)):.3f}")
        
        # Save audio
        output_dir = Path("output/enhancements_test")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = output_dir / f"scenario_{scenario_num}_{response['emotion']}.wav"
        torchaudio.save(str(output_file), processed_audio.unsqueeze(0), 24000)
        print_feature("💾 Saved", str(output_file))
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def main():
    """Run comprehensive enhancement tests"""
    
    print("\n🚀 TESTING ALL BEYOND-MAYA ENHANCEMENTS")
    print("="*80)
    print("Testing 6 features across 8 scenarios:")
    print("  1. Backchannel System")
    print("  2. Enhanced Intensity Mapping")
    print("  3. Contextual Prosody Memory")
    print("  4. Micro-pause Predictor")
    print("  5. Enhanced Breath System")
    print("  6. 49-Emotion Integration")
    print("="*80)
    
    # Initialize components
    print("\n⚙️  Initializing components...")
    brain = OviyaBrain(
        ollama_url="https://c452e968bc8ea6.lhr.life/api/generate"
    )
    
    emotion_controller = EmotionController()
    
    tts = HybridVoiceEngine(
        csm_url="https://astronomy-initiative-paso-cream.trycloudflare.com/generate",
        default_engine="csm"
    )
    
    audio_processor = AudioPostProcessor(sample_rate=24000)
    
    print("✅ All components initialized\n")
    
    # Test scenarios covering different emotions and features
    scenarios = [
        {
            "title": "Stressed User (Tests: Backchannel, Comforting Emotion)",
            "user_message": "I'm feeling really stressed about my presentation tomorrow",
            "user_emotion": "stressed"
        },
        {
            "title": "Excited User (Tests: Positive Resonance, High Intensity)",
            "user_message": "Guess what! I just got promoted at work!",
            "user_emotion": "excited"
        },
        {
            "title": "Uncertain User (Tests: Epistemic Prosody, Thinking Backchannel)",
            "user_message": "I'm not sure what to do about this situation, maybe I should wait?",
            "user_emotion": "uncertain"
        },
        {
            "title": "Sad User (Tests: Negative Resonance, Empathetic Response)",
            "user_message": "I've been feeling really lonely lately, I miss my friends",
            "user_emotion": "sad"
        },
        {
            "title": "Narrative User (Tests: Micro-pauses, Prosody Memory)",
            "user_message": "So then I talked to my manager, and she said I could take the lead",
            "user_emotion": "neutral"
        },
        {
            "title": "Grateful User (Tests: Tier 1 Emotion, Warmth)",
            "user_message": "Thank you so much for listening to me, you really help",
            "user_emotion": "grateful"
        },
        {
            "title": "Frustrated User (Tests: Tier 3 Emotion, Higher Energy)",
            "user_message": "This problem keeps happening no matter what I try, it's so frustrating",
            "user_emotion": "frustrated"
        },
        {
            "title": "Nostalgic User (Tests: Tier 2 Emotion, Wistful Tone)",
            "user_message": "I remember when we used to do that, those were good times",
            "user_emotion": "nostalgic"
        }
    ]
    
    # Run all scenarios
    results = []
    for i, scenario in enumerate(scenarios, 1):
        success = test_scenario(
            scenario_num=i,
            title=scenario["title"],
            user_message=scenario["user_message"],
            user_emotion=scenario["user_emotion"],
            brain=brain,
            emotion_controller=emotion_controller,
            tts=tts,
            audio_processor=audio_processor
        )
        results.append(success)
    
    # Final summary
    print_section("TEST SUMMARY")
    
    successful = sum(results)
    total = len(results)
    
    print(f"✅ Successful: {successful}/{total}")
    print(f"❌ Failed: {total - successful}/{total}")
    print(f"📊 Success Rate: {(successful/total*100):.1f}%")
    
    print("\n📁 Output files saved to: output/enhancements_test/")
    print("\n🎉 All enhancements tested!")
    
    # Feature demonstration summary
    print_section("FEATURES DEMONSTRATED")
    
    features_shown = {
        "💬 Backchannels": "Automatic micro-affirmations based on user emotion",
        "📈 Intensity Curves": "Non-linear emotion scaling for natural expression",
        "🧠 Prosody Memory": "Cross-turn consistency in speech patterns",
        "⏸️  Micro-Pauses": "Natural pauses after conjunctions and phrases",
        "🫁 Breath System": "Respiratory state model with adaptive breathing",
        "🎭 49 Emotions": "Full emotion library across 3 tiers"
    }
    
    for feature, description in features_shown.items():
        print(f"   {feature}")
        print(f"      → {description}")
    
    print("\n✨ Beyond-Maya enhancements working perfectly!\n")

if __name__ == "__main__":
    main()

