"""
CSM Server with Expanded Emotion Library Support
Supports 28+ emotions across 3 tiers
"""

import sys
import os
sys.path.insert(0, '/workspace/csm/csm')
os.environ["NO_TORCH_COMPILE"] = "1"

from flask import Flask, request, jsonify
import torch
import torchaudio
from pathlib import Path
import base64
import io
from generator import load_csm_1b, Segment
from typing import List, Optional
import json

app = Flask(__name__)

# Global generator
generator = None
EMOTION_REF_DIR = Path("/workspace/emotion_references")

# Load emotion library config
EMOTION_CONFIG = None
def load_emotion_config():
    global EMOTION_CONFIG
    config_path = EMOTION_REF_DIR / "emotion_library.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            EMOTION_CONFIG = json.load(f)
        print(f"📚 Loaded emotion library: {EMOTION_CONFIG.get('total_emotions', 0)} emotions")
    else:
        print("⚠️  No emotion library config found, using defaults")
        EMOTION_CONFIG = {"emotion_texts": {}}

# Emotion aliases for backward compatibility
EMOTION_ALIASES = {
    "happy": "joyful_excited",
    "sad": "empathetic_sad",
    "angry": "angry_firm",
    "worried": "concerned_anxious",
    "calm": "calm_supportive",
    "excited": "joyful_excited",
    "supportive": "comforting",
    "understanding": "empathetic_sad",
    "enthusiastic": "encouraging",
    "caring": "affectionate",
    "peaceful": "calm_supportive",
    "stressed": "concerned_anxious",
    "frustrated": "angry_firm"
}

def resolve_emotion(emotion_label: str) -> str:
    """Resolve emotion label to library emotion"""
    # Direct match
    ref_file = EMOTION_REF_DIR / f"{emotion_label}.wav"
    if ref_file.exists():
        return emotion_label
    
    # Check aliases
    if emotion_label in EMOTION_ALIASES:
        resolved = EMOTION_ALIASES[emotion_label]
        if (EMOTION_REF_DIR / f"{resolved}.wav").exists():
            return resolved
    
    # Fallback
    return "neutral"

def load_reference_audio(emotion: str) -> Optional[torch.Tensor]:
    """Load emotion reference audio"""
    ref_path = EMOTION_REF_DIR / f"{emotion}.wav"
    
    if not ref_path.exists():
        print(f"⚠️  Reference not found: {ref_path}")
        return None
    
    try:
        audio, sr = torchaudio.load(ref_path)
        
        # Resample if needed
        if sr != generator.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, generator.sample_rate)
            audio = resampler(audio)
        
        # Convert to mono if stereo
        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        return audio.squeeze(0).to(generator.device)
    
    except Exception as e:
        print(f"❌ Failed to load reference {emotion}: {e}")
        return None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy" if generator is not None else "loading",
        "model_loaded": generator is not None,
        "emotion_library": EMOTION_CONFIG.get("total_emotions", 0) if EMOTION_CONFIG else 0,
        "reference_dir": str(EMOTION_REF_DIR),
        "available_emotions": len(list(EMOTION_REF_DIR.glob("*.wav"))) if EMOTION_REF_DIR.exists() else 0
    }
    return jsonify(status)

@app.route('/emotions', methods=['GET'])
def list_emotions():
    """List all available emotions"""
    available = []
    if EMOTION_REF_DIR.exists():
        available = [f.stem for f in EMOTION_REF_DIR.glob("*.wav")]
    
    return jsonify({
        "total": len(available),
        "emotions": sorted(available),
        "tiers": EMOTION_CONFIG.get("tiers", {}) if EMOTION_CONFIG else {}
    })

@app.route('/generate', methods=['POST'])
def generate():
    """Generate speech with emotion reference"""
    try:
        data = request.get_json()
        
        # Extract parameters
        text = data.get('text', '')
        speaker = data.get('speaker', 0)
        max_audio_length_ms = data.get('max_audio_length_ms', 10000)
        reference_emotion = data.get('reference_emotion', None)
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        print(f"\n🎤 Generate request:")
        print(f"   Text: \"{text[:60]}...\"")
        print(f"   Emotion: {reference_emotion}")
        
        # Prepare context with emotion reference
        context: List[Segment] = []
        
        if reference_emotion:
            # Resolve emotion (handle aliases)
            resolved_emotion = resolve_emotion(reference_emotion)
            print(f"   Resolved: {reference_emotion} → {resolved_emotion}")
            
            # Load reference audio
            ref_audio = load_reference_audio(resolved_emotion)
            
            if ref_audio is not None:
                # Get reference text from config
                ref_text = EMOTION_CONFIG.get("emotion_texts", {}).get(
                    resolved_emotion,
                    f"This is {resolved_emotion}."
                ) if EMOTION_CONFIG else f"This is {resolved_emotion}."
                
                # Create reference segment
                reference_segment = Segment(
                    text=ref_text,
                    speaker=0,
                    audio=ref_audio
                )
                context = [reference_segment]
                print(f"   ✅ Using emotion reference: {resolved_emotion}")
            else:
                print(f"   ⚠️  No reference available, generating without emotion")
        
        # Generate audio
        audio = generator.generate(
            text=text,
            speaker=speaker,
            context=context,
            max_audio_length_ms=max_audio_length_ms,
        )
        
        # Convert to base64
        audio_cpu = audio.cpu()
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_cpu.unsqueeze(0), generator.sample_rate, format="wav")
        buffer.seek(0)
        audio_bytes = buffer.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        duration = audio.shape[-1] / generator.sample_rate
        print(f"   ✅ Generated {duration:.2f}s audio\n")
        
        return jsonify({
            "audio_base64": audio_base64,
            "text": text,
            "emotion": reference_emotion,
            "resolved_emotion": resolved_emotion if reference_emotion else None,
            "duration": duration,
            "sample_rate": generator.sample_rate,
            "status": "success"
        })
    
    except Exception as e:
        print(f"❌ Generation error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

def main():
    """Start CSM server"""
    global generator
    
    print("=" * 70)
    print("🎨 CSM SERVER - EXPANDED EMOTION LIBRARY")
    print("=" * 70)
    
    # Load emotion config
    load_emotion_config()
    
    # Check emotion references
    if EMOTION_REF_DIR.exists():
        ref_count = len(list(EMOTION_REF_DIR.glob("*.wav")))
        print(f"📂 Found {ref_count} emotion references in {EMOTION_REF_DIR}")
    else:
        print(f"⚠️  Emotion reference directory not found: {EMOTION_REF_DIR}")
        EMOTION_REF_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load CSM model
    print("\n🔄 Loading CSM model...")
    generator = load_csm_1b(device="cuda")
    print("✅ CSM model loaded successfully!")
    
    # Start server
    print("\n🚀 Starting server on port 6006...")
    print("=" * 70)
    app.run(host='0.0.0.0', port=6006, debug=False)

if __name__ == '__main__':
    main()

