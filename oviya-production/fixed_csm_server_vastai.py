#!/usr/bin/env python3
"""
Fixed CSM Server with Emotion Reference Support

This version ensures CSM generates NEW audio using the reference as context,
not returning the reference itself.
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

        # Load emotion reference if provided
        if reference_emotion and reference_emotion in EMOTION_TEXTS:
            ref_path = EMOTION_REF_DIR / f"{reference_emotion}.wav"
            
            if ref_path.exists():
                try:
                    # Load reference audio
                    ref_audio, ref_sr = torchaudio.load(str(ref_path))
                    ref_audio = ref_audio.squeeze(0)
                    
                    # Resample if needed
                    if ref_sr != generator.sample_rate:
                        ref_audio = torchaudio.functional.resample(
                            ref_audio,
                            orig_freq=ref_sr,
                            new_freq=generator.sample_rate
                        )
                    
                    # IMPORTANT: Use a DIFFERENT text for the reference segment
                    # This tells CSM "use this acoustic style" without repeating content
                    ref_segment = Segment(
                        text=EMOTION_TEXTS[reference_emotion],  # Different text
                        speaker=speaker,
                        audio=ref_audio
                    )
                    
                    # Prepend reference to context
                    # CSM will use this as acoustic conditioning
                    if isinstance(context, list):
                        context = [ref_segment] + context
                    else:
                        context = [ref_segment]
                    
                    print(f"   ✅ Loaded emotion reference: {ref_path.name}")
                    print(f"   📝 Reference text: '{EMOTION_TEXTS[reference_emotion]}'")
                    print(f"   🎯 Generating NEW text: '{text}'")
                
                except Exception as e:
                    print(f"   ⚠️ Failed to load reference: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"   ⚠️ Reference not found: {ref_path}")

        # Generate NEW audio for the requested text
        # CSM should use the reference context to guide prosody/emotion
        audio = generator.generate(
            text=text,  # The ACTUAL text to generate
            speaker=speaker,
            context=context,  # Includes reference as context
            max_audio_length_ms=max_audio_length_ms,
        )

        # Convert to base64
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.unsqueeze(0).cpu(), generator.sample_rate, format='wav')
        buffer.seek(0)
        audio_bytes = buffer.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')

        duration = audio.shape[-1] / generator.sample_rate
        print(f"   ✅ Generated NEW audio: {duration:.2f}s")
        print(f"   📊 Audio samples: {audio.shape[-1]}")

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
    print('🚀 Starting FIXED CSM server with emotion reference support on port 6006...')
    print(f'📁 Emotion references directory: {EMOTION_REF_DIR}')
    print('\n💡 How it works:')
    print('   1. Reference audio loaded as Segment')
    print('   2. Reference used as CONTEXT (not returned)')
    print('   3. CSM generates NEW audio for requested text')
    print('   4. Reference guides emotion/prosody\n')
    app.run(host='0.0.0.0', port=6006)


