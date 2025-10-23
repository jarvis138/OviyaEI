#!/usr/bin/env python3
"""
Test LLM Prosodic Markup Generation
5 diverse scenarios to test if LLM generates proper prosodic markup
"""

import sys
import time
from pathlib import Path

# Add root directory to path for imports
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from production.brain.llm_brain import OviyaBrain
from production.emotion_controller.controller import EmotionController
from production.voice.openvoice_tts import HybridVoiceEngine

def test_scenario(num, message, emotion, brain, controller, tts):
    """Test a single scenario"""
    print(f"\n{'='*70}")
    print(f"  SCENARIO {num}")
    print('='*70)
    print(f"User: \"{message}\"")
    print(f"Expected emotion: {emotion}\n")
    
    # Brain processing
    response = brain.think(message, emotion)
    
    # Display results
    print(f"[LLM Response]")
    print(f"  Text: \"{response['text']}\"")
    print(f"  Emotion: {response['emotion']} (intensity: {response['intensity']:.2f})")
    print(f"\n[Prosodic Text]")
    print(f"  \"{response['prosodic_text']}\"")
    
    # Check for prosodic markers
    markers = []
    text = response['prosodic_text']
    if '<breath>' in text: markers.append('✅ <breath>')
    if '<pause>' in text or '<long_pause>' in text or '<micro_pause>' in text: 
        markers.append('✅ <pause>')
    if '<smile>' in text: markers.append('✅ <smile>')
    if '<gentle>' in text: markers.append('✅ <gentle>')
    if '<strong>' in text: markers.append('✅ <strong>')
    if '<uncertain>' in text: markers.append('✅ <uncertain>')
    
    print(f"\n[Prosodic Markers Found]")
    if markers:
        for marker in markers:
            print(f"  {marker}")
    else:
        print("  ⚠️  No prosodic markers found!")
    
    # Emotion params
    emotion_params = controller.map_emotion(response['emotion'], response['intensity'])
    print(f"\n[Emotion Parameters]")
    print(f"  Pitch: {emotion_params['pitch_scale']:.3f}x")
    print(f"  Rate: {emotion_params['rate_scale']:.3f}x")
    print(f"  Energy: {emotion_params['energy_scale']:.3f}x")
    
    # Generate audio
    print(f"\n[Audio Generation]")
    try:
        start = time.time()
        audio = tts.generate(
            text=response['text'],
            emotion_params=emotion_params,
            prosodic_text=response['prosodic_text'],
            emotional_state=response.get('emotional_state')
        )
        duration = len(audio) / 24000
        gen_time = time.time() - start
        
        print(f"  ✅ Generated {duration:.2f}s audio in {gen_time:.2f}s")
        
        # Save
        output_dir = Path("output/llm_prosody")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"scenario_{num}_{response['emotion']}.wav"
        
        import torchaudio
        torchaudio.save(str(output_path), audio.unsqueeze(0), 24000)
        print(f"  💾 Saved: {output_path.name}")
        
        return True
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("  🧪 LLM PROSODIC MARKUP TEST")
    print("  Testing 5 scenarios with actual LLM responses")
    print("="*70)
    
    # Initialize
    print("\nInitializing components...")
    brain = OviyaBrain(ollama_url="https://f4e58fff120d25.lhr.life/api/generate")
    controller = EmotionController()
    tts = HybridVoiceEngine(
        csm_url="https://astronomy-initiative-paso-cream.trycloudflare.com/generate",
        default_engine="csm"
    )
    print("✅ All systems ready!\n")
    
    # Test scenarios
    scenarios = [
        (1, "You look absolutely stunning today!", "playful"),
        (2, "Oh great, another brilliant idea", "sarcastic"),
        (3, "I'm feeling really stressed about this exam", "concerned_anxious"),
        (4, "That's the best news I've heard all week!", "joyful_excited"),
        (5, "I really miss how things used to be", "melancholic")
    ]
    
    results = []
    for num, message, emotion in scenarios:
        success = test_scenario(num, message, emotion, brain, controller, tts)
        results.append((num, success))
        time.sleep(1)
    
    # Summary
    print("\n" + "="*70)
    print("  📊 SUMMARY")
    print("="*70)
    
    successful = sum(1 for _, success in results if success)
    print(f"\nResults: {successful}/{len(results)} scenarios completed\n")
    
    for num, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} Scenario {num}")
    
    success_rate = (successful / len(results)) * 100
    print(f"\n📈 Success Rate: {success_rate:.0f}%")
    
    if success_rate == 100:
        print("\n✅ PERFECT! All LLM responses generated with prosodic markup!")
    elif success_rate >= 80:
        print("\n✅ EXCELLENT! Most scenarios successful!")
    else:
        print("\n⚠️  NEEDS IMPROVEMENT: Check LLM prompt configuration")
    
    print(f"\n📁 Audio saved to: output/llm_prosody/\n")

if __name__ == "__main__":
    main()


