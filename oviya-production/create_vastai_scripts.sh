#!/bin/bash
# Run these commands on Vast.ai to create the scripts directly

echo "📝 Creating emotion reference extraction script..."

cat > /workspace/extract_emotion_references_vastai.py << 'SCRIPT_EOF'
"""
Emotion Reference Extractor for Vast.ai

Generates 8 emotion reference audio files using synthetic references.
These references are used as CSM context for emotionally expressive speech.
"""

import torch
import torchaudio
from pathlib import Path


def generate_synthetic_reference(emotion: str, text: str, sample_rate: int = 24000):
    """Generate synthetic emotional reference for testing."""
    duration = 2.0  # 2 seconds
    num_samples = int(duration * sample_rate)
    
    t = torch.linspace(0, duration, num_samples)
    
    # Different frequency patterns for different emotions
    emotion_freqs = {
        "calm_supportive": 200,
        "empathetic_sad": 180,
        "joyful_excited": 300,
        "playful": 280,
        "confident": 220,
        "concerned_anxious": 240,
        "angry_firm": 180,
        "neutral": 220
    }
    
    base_freq = emotion_freqs.get(emotion, 220)
    
    # Generate tone
    audio = 0.3 * torch.sin(2 * torch.pi * base_freq * t)
    
    # Add emotion-specific modulation
    if emotion == "joyful_excited":
        vibrato = 0.1 * torch.sin(2 * torch.pi * 5 * t)
        audio = audio * (1 + vibrato)
    elif emotion == "empathetic_sad":
        decay = torch.exp(-t * 0.5)
        audio = audio * decay
    
    return audio, sample_rate


def main():
    """Generate all emotion references."""
    print("\n" + "="*60)
    print("🎭 EMOTION REFERENCE EXTRACTION (SYNTHETIC)")
    print("="*60)
    
    emotions = {
        "calm_supportive": "Take a deep breath. Everything will be okay.",
        "empathetic_sad": "I'm so sorry you're going through this.",
        "joyful_excited": "That's amazing! I'm so happy for you!",
        "playful": "Hey there! This is going to be fun!",
        "confident": "You've got this. I believe in you.",
        "concerned_anxious": "Are you okay? I'm here if you need me.",
        "angry_firm": "That's not acceptable. This needs to stop.",
        "neutral": "Hello. How can I help you today?"
    }
    
    # Create output directory
    output_dir = Path("/workspace/emotion_references")
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output directory: {output_dir}")
    
    print(f"\n🎙️ Generating {len(emotions)} synthetic emotion references...")
    print("="*60)
    
    generated_refs = {}
    
    for emotion, text in emotions.items():
        print(f"\n[{emotion}]")
        print(f"   Text: \"{text}\"")
        
        # Generate synthetic reference
        audio, sr = generate_synthetic_reference(emotion, text)
        
        # Save reference
        output_path = output_dir / f"{emotion}.wav"
        torchaudio.save(str(output_path), audio.unsqueeze(0), sr)
        
        duration = audio.shape[0] / sr
        print(f"   ✅ Generated: {duration:.2f}s")
        print(f"   💾 Saved: {output_path}")
        
        generated_refs[emotion] = str(output_path)
    
    print("\n" + "="*60)
    print(f"✅ Successfully generated {len(generated_refs)} references")
    print(f"📁 Location: {output_dir}")
    print("="*60)
    
    print("\n📊 Summary:")
    for emotion, path in generated_refs.items():
        print(f"   {emotion}: {Path(path).name}")
    
    print("\n🎯 Next step:")
    print("   python3 update_csm_server_vastai.py")


if __name__ == "__main__":
    main()
SCRIPT_EOF

echo "✅ Created: /workspace/extract_emotion_references_vastai.py"

echo ""
echo "📝 Creating CSM server update script..."

cat > /workspace/update_csm_server_vastai.py << 'SCRIPT_EOF'
"""
CSM Server Update for Emotion References
"""

UPDATED_SERVER_CODE = '''#!/usr/bin/env python3
"""
Official CSM Server with Emotion Reference Support
"""

import torch
import torchaudio
from generator import load_csm_1b, Segment
from flask import Flask, request, jsonify
import io
import base64
import os
from pathlib import Path

os.environ["NO_TORCH_COMPILE"] = "1"

app = Flask(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CSM on device: {device}")
generator = load_csm_1b(device=device)
print("✅ CSM model loaded successfully!")

EMOTION_REF_DIR = Path("/workspace/emotion_references")

EMOTION_TEXTS = {
    "calm_supportive": "Take a deep breath. Everything will be okay.",
    "empathetic_sad": "I'm so sorry you're going through this.",
    "joyful_excited": "That's amazing! I'm so happy for you!",
    "playful": "Hey there! This is going to be fun!",
    "confident": "You've got this. I believe in you.",
    "concerned_anxious": "Are you okay? I'm here if you need me.",
    "angry_firm": "That's not acceptable. This needs to stop.",
    "neutral": "Hello. How can I help you today?"
}

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': True,
        'device': device,
        'sample_rate': generator.sample_rate,
        'emotion_references_available': EMOTION_REF_DIR.exists()
    })

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        text = data.get('text', '')
        speaker = data.get('speaker', 0)
        context = data.get('context', [])
        max_audio_length_ms = data.get('max_audio_length_ms', 10000)
        reference_emotion = data.get('reference_emotion', None)

        print(f"🎤 Generating: {text[:50]}...")
        if reference_emotion:
            print(f"   🎭 With emotion reference: {reference_emotion}")

        if reference_emotion and reference_emotion in EMOTION_TEXTS:
            ref_path = EMOTION_REF_DIR / f"{reference_emotion}.wav"
            
            if ref_path.exists():
                try:
                    ref_audio, ref_sr = torchaudio.load(str(ref_path))
                    ref_audio = ref_audio.squeeze(0)
                    
                    if ref_sr != generator.sample_rate:
                        ref_audio = torchaudio.functional.resample(
                            ref_audio,
                            orig_freq=ref_sr,
                            new_freq=generator.sample_rate
                        )
                    
                    ref_segment = Segment(
                        text=EMOTION_TEXTS[reference_emotion],
                        speaker=speaker,
                        audio=ref_audio
                    )
                    
                    context = [ref_segment] + (context if isinstance(context, list) else [])
                    
                    print(f"   ✅ Loaded emotion reference: {ref_path.name}")
                
                except Exception as e:
                    print(f"   ⚠️ Failed to load reference: {e}")
            else:
                print(f"   ⚠️ Reference not found: {ref_path}")

        audio = generator.generate(
            text=text,
            speaker=speaker,
            context=context,
            max_audio_length_ms=max_audio_length_ms,
        )

        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.unsqueeze(0).cpu(), generator.sample_rate, format='wav')
        buffer.seek(0)
        audio_bytes = buffer.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        duration = audio.shape[-1] / generator.sample_rate
        print(f"   ✅ Generated: {duration:.2f}s")

        return jsonify({
            'audio_base64': audio_base64,
            'text': text,
            'speaker': speaker,
            'duration': duration,
            'sample_rate': generator.sample_rate,
            'reference_emotion': reference_emotion,
            'status': 'success'
        })

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print('🚀 Starting CSM server with emotion reference support on port 6006...')
    print(f'📁 Emotion references directory: {EMOTION_REF_DIR}')
    app.run(host='0.0.0.0', port=6006)
'''


def main():
    """Update the CSM server script."""
    print("\n" + "="*60)
    print("🔧 CSM SERVER UPDATE FOR EMOTION REFERENCES")
    print("="*60)
    
    output_path = Path("/workspace/official_csm_server_with_emotions.py")
    
    print(f"\n📝 Writing updated server to: {output_path}")
    
    with open(output_path, 'w') as f:
        f.write(UPDATED_SERVER_CODE)
    
    print("✅ Server code updated successfully!")
    
    print("\n🎯 Next steps:")
    print("   1. Stop current CSM server (Ctrl+C)")
    print("   2. Run updated server:")
    print(f"      python3 {output_path}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
SCRIPT_EOF

echo "✅ Created: /workspace/update_csm_server_vastai.py"

echo ""
echo "="*60
echo "✅ SCRIPTS CREATED SUCCESSFULLY!"
echo "="*60
echo ""
echo "Now run:"
echo "  1. python3 /workspace/extract_emotion_references_vastai.py"
echo "  2. python3 /workspace/update_csm_server_vastai.py"
echo ""



