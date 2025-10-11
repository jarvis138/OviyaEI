"""
CSM Server Update for Emotion References

This script updates the official_csm_server.py to accept and use emotion references.
Run this on your Vast.ai server.

It modifies the /generate endpoint to:
1. Accept optional 'reference_emotion' parameter
2. Load corresponding emotion reference audio
3. Use it as Segment context for CSM generation
"""

UPDATED_SERVER_CODE = '''#!/usr/bin/env python3
"""
Official CSM Server with Emotion Reference Support

Updated to accept emotion references as context for emotionally expressive generation.
"""

import torch
import torchaudio
from generator import load_csm_1b, Segment
from flask import Flask, request, jsonify
import io
import base64
import os
from pathlib import Path

# Set environment variable
os.environ["NO_TORCH_COMPILE"] = "1"

app = Flask(__name__)

# Load CSM model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CSM on device: {device}")
generator = load_csm_1b(device=device)
print("✅ CSM model loaded successfully!")

# Emotion reference directory
EMOTION_REF_DIR = Path("/workspace/emotion_references")

# Emotion reference texts (for Segment creation)
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
        
        # NEW: Accept emotion reference
        reference_emotion = data.get('reference_emotion', None)

        print(f"🎤 Generating: {text[:50]}...")
        if reference_emotion:
            print(f"   🎭 With emotion reference: {reference_emotion}")

        # Load emotion reference if provided
        if reference_emotion and reference_emotion in EMOTION_TEXTS:
            ref_path = EMOTION_REF_DIR / f"{reference_emotion}.wav"
            
            if ref_path.exists():
                try:
                    # Load reference audio
                    ref_audio, ref_sr = torchaudio.load(str(ref_path))
                    ref_audio = ref_audio.squeeze(0)  # Remove channel dimension
                    
                    # Resample if needed
                    if ref_sr != generator.sample_rate:
                        ref_audio = torchaudio.functional.resample(
                            ref_audio,
                            orig_freq=ref_sr,
                            new_freq=generator.sample_rate
                        )
                    
                    # Create Segment with reference
                    ref_segment = Segment(
                        text=EMOTION_TEXTS[reference_emotion],
                        speaker=speaker,
                        audio=ref_audio
                    )
                    
                    # Prepend reference to context
                    context = [ref_segment] + (context if isinstance(context, list) else [])
                    
                    print(f"   ✅ Loaded emotion reference: {ref_path.name}")
                
                except Exception as e:
                    print(f"   ⚠️ Failed to load reference: {e}")
            else:
                print(f"   ⚠️ Reference not found: {ref_path}")

        # Generate audio
        audio = generator.generate(
            text=text,
            speaker=speaker,
            context=context,
            max_audio_length_ms=max_audio_length_ms,
        )

        # Convert to base64
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
    print("   3. Test with emotion references")
    
    print("\n📋 Changes made:")
    print("   ✅ Added 'reference_emotion' parameter support")
    print("   ✅ Loads emotion reference WAV files")
    print("   ✅ Creates Segment with reference audio")
    print("   ✅ Prepends reference to context")
    print("   ✅ Returns reference_emotion in response")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()


