#!/usr/bin/env python3
"""
Diverse Scenario Testing with Prosodic Markup
Tests flirting, sarcasm, and validates specific prosodic markers
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from brain.llm_brain import OviyaBrain
from emotion_controller.controller import EmotionController
from voice.openvoice_tts import HybridVoiceEngine

def print_section(title, emoji="🎭"):
    print(f"\n{'='*70}")
    print(f"  {emoji} {title}")
    print('='*70 + "\n")

def test_scenario(scenario_num, user_message, user_emotion, brain, controller, tts, description):
    """Test a single scenario"""
    print(f"Scenario {scenario_num}: {description}")
    print(f"  User: \"{user_message}\"")
    print(f"  Emotion: {user_emotion}")
    
    # Brain processing
    start_time = time.time()
    response = brain.think(user_message, user_emotion)
    brain_time = time.time() - start_time
    
    print(f"\n  [Brain Response]")
    print(f"  💬 Text: \"{response['text'][:80]}...\"")
    print(f"  🎭 Emotion: {response['emotion']} (intensity: {response['intensity']:.2f})")
    print(f"  🎼 Prosodic Text: \"{response['prosodic_text'][:100]}...\"")
    
    # Check for specific prosodic markers
    markers_found = []
    if '<breath>' in response['prosodic_text']:
        markers_found.append('breath')
    if '<pause>' in response['prosodic_text'] or '<long_pause>' in response['prosodic_text'] or '<micro_pause>' in response['prosodic_text']:
        markers_found.append('pause')
    if '<smile>' in response['prosodic_text']:
        markers_found.append('smile')
    if '<gentle>' in response['prosodic_text']:
        markers_found.append('gentle')
    if '<strong>' in response['prosodic_text']:
        markers_found.append('strong')
    if '<uncertain>' in response['prosodic_text']:
        markers_found.append('uncertain')
    
    if markers_found:
        print(f"  ✅ Prosodic markers: {', '.join(markers_found)}")
    else:
        print(f"  ⚠️  No prosodic markers detected")
    
    # Backchannel check
    if response.get('has_backchannel'):
        print(f"  💬 Backchannel: {response.get('backchannel_type', 'unknown')}")
    
    # Emotional state
    if response.get('emotional_state'):
        state = response['emotional_state']
        print(f"  🧠 Emotional State: Energy={state['energy_level']:.2f}, Warmth={state['warmth']:.2f}")
    
    # Emotion controller
    emotion_params = controller.map_emotion(
        response['emotion'],
        response['intensity']
    )
    
    print(f"\n  [Emotion Controller]")
    print(f"  📊 Pitch: {emotion_params['pitch_scale']:.3f}x")
    print(f"  📊 Rate: {emotion_params['rate_scale']:.3f}x")
    print(f"  📊 Energy: {emotion_params['energy_scale']:.3f}x")
    
    # TTS generation
    print(f"\n  [Voice Generation]")
    tts_start = time.time()
    try:
        audio = tts.generate(
            text=response['text'],
            emotion_params=emotion_params,
            prosodic_text=response['prosodic_text'],
            emotional_state=response.get('emotional_state')
        )
        tts_time = time.time() - tts_start
        
        audio_duration = len(audio) / 24000
        print(f"  ✅ Generated: {audio_duration:.2f}s audio")
        print(f"  ⏱️  Brain time: {brain_time:.2f}s")
        print(f"  ⏱️  TTS time: {tts_time:.2f}s")
        print(f"  ⏱️  Total: {(brain_time + tts_time):.2f}s")
        
        # Save audio
        output_dir = Path("output/diverse_scenarios")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / f"scenario_{scenario_num}_{response['emotion']}.wav"
        
        import torchaudio
        torchaudio.save(str(output_path), audio.unsqueeze(0), 24000)
        print(f"  💾 Saved: {output_path.name}")
        
        return True
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        return False

def main():
    print("\n" + "="*70)
    print("  🎭 DIVERSE SCENARIO TESTING")
    print("  Testing: Flirting, Sarcasm, Prosodic Markup")
    print("="*70)
    
    # Initialize components
    print("\n[1/3] Initializing Brain...")
    brain = OviyaBrain(ollama_url="https://5799b9db7ee4fb.lhr.life/api/generate")
    
    print("\n[2/3] Initializing Emotion Controller...")
    controller = EmotionController()
    
    print("\n[3/3] Initializing Voice Engine...")
    tts = HybridVoiceEngine(
        csm_url="https://astronomy-initiative-paso-cream.trycloudflare.com/generate",
        default_engine="csm"
    )
    
    print("\n✅ All systems ready!\n")
    
    # Define test scenarios
    scenarios = [
        # Flirting scenarios
        {
            "num": 1,
            "message": "You look really nice today",
            "emotion": "playful",
            "description": "Flirty Compliment (should have <smile>)"
        },
        {
            "num": 2,
            "message": "I've been thinking about you all day",
            "emotion": "affectionate",
            "description": "Romantic Interest (should have warm tone)"
        },
        {
            "num": 3,
            "message": "Want to grab coffee sometime? Just the two of us?",
            "emotion": "playful",
            "description": "Flirty Invitation (should have playful energy)"
        },
        
        # Sarcastic scenarios
        {
            "num": 4,
            "message": "Oh great, another Monday morning",
            "emotion": "sarcastic",
            "description": "Sarcastic Complaint (should have ironic tone)"
        },
        {
            "num": 5,
            "message": "Wow, what a brilliant idea, I'm sure that'll work perfectly",
            "emotion": "sarcastic",
            "description": "Sarcastic Praise (should have exaggerated emphasis)"
        },
        {
            "num": 6,
            "message": "Yeah, because that went so well last time",
            "emotion": "sarcastic",
            "description": "Sarcastic Reminder (should have dry delivery)"
        },
        
        # Prosodic marker focused scenarios
        {
            "num": 7,
            "message": "Take a deep breath... everything is going to be okay",
            "emotion": "calm_supportive",
            "description": "Breathing Exercise (MUST have <breath> and <pause>)"
        },
        {
            "num": 8,
            "message": "Well... I'm not sure how to say this, but...",
            "emotion": "hesitant",
            "description": "Hesitation (MUST have <pause> markers)"
        },
        {
            "num": 9,
            "message": "That's wonderful news! I'm so happy for you!",
            "emotion": "joyful_excited",
            "description": "Excited Joy (MUST have <smile> marker)"
        },
        
        # Complex emotional scenarios
        {
            "num": 10,
            "message": "I really miss the way things used to be",
            "emotion": "melancholic",
            "description": "Wistful Nostalgia (should have gentle, slow delivery)"
        },
        {
            "num": 11,
            "message": "Stop. Just stop. I can't deal with this right now.",
            "emotion": "frustrated",
            "description": "Frustration (should have <strong> emphasis)"
        },
        {
            "num": 12,
            "message": "You know what? I actually believe in you. You've got this.",
            "emotion": "encouraging",
            "description": "Encouragement (should have warm, confident tone)"
        }
    ]
    
    # Run tests
    results = []
    
    for scenario in scenarios:
        print_section(f"SCENARIO {scenario['num']}/12", "🎬")
        
        success = test_scenario(
            scenario['num'],
            scenario['message'],
            scenario['emotion'],
            brain,
            controller,
            tts,
            scenario['description']
        )
        
        results.append({
            'scenario': scenario['num'],
            'description': scenario['description'],
            'success': success
        })
        
        time.sleep(3)  # Longer pause to prevent 503 errors
    
    # Summary
    print_section("SUMMARY", "📊")
    
    successful = sum(1 for r in results if r['success'])
    total = len(results)
    
    print(f"Results: {successful}/{total} scenarios completed successfully\n")
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"  {status} Scenario {result['scenario']}: {result['description']}")
    
    success_rate = (successful / total) * 100
    print(f"\n📈 Success Rate: {success_rate:.1f}%")
    
    # Prosodic marker analysis
    print("\n" + "="*70)
    print("  🎼 PROSODIC MARKER ANALYSIS")
    print("="*70 + "\n")
    
    print("Expected markers in specific scenarios:")
    print("  Scenario 7: <breath>, <pause> (Breathing exercise)")
    print("  Scenario 8: <pause>, <long_pause> (Hesitation)")
    print("  Scenario 9: <smile> (Excited joy)")
    print("\nCheck the console output above to verify these markers were generated.")
    
    print("\n" + "="*70)
    if success_rate >= 80:
        print("  ✅ EXCELLENT: All diverse scenarios tested successfully!")
    elif success_rate >= 60:
        print("  ⚠️  GOOD: Most scenarios completed, some issues detected")
    else:
        print("  ❌ NEEDS WORK: Multiple scenario failures")
    print("="*70 + "\n")
    
    print(f"📁 Audio files saved to: output/diverse_scenarios/\n")

if __name__ == "__main__":
    main()

