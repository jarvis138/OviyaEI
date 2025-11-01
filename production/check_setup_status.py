#!/usr/bin/env python3
"""
Setup Status Monitor
====================

Monitors the progress of the complete setup process and reports status.
"""

import os
import sys
import time
from pathlib import Path
import json
import importlib.util

def check_csm_ready():
    """Check if CSM-1B is ready"""
    try:
        from transformers import AutoProcessor, CsmForConditionalGeneration
        
        model_id = "sesame/csm-1b"
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        
        processor = AutoProcessor.from_pretrained(model_id, token=token)
        return True, "✅ CSM-1B model ready"
    except Exception as e:
        return False, f"❌ CSM-1B not ready: {e}"

def check_emotion_references():
    """Check if emotion references exist"""
    ref_dir = Path("data/emotion_references")
    map_file = ref_dir / "emotion_map.json"
    
    if map_file.exists():
        try:
            with open(map_file, 'r') as f:
                emotion_map = json.load(f)
            
            total_refs = sum(len(refs) for refs in emotion_map.values())
            return True, f"✅ Emotion references ready: {len(emotion_map)} emotions, {total_refs} references"
        except Exception:
            return False, "❌ Emotion map exists but invalid"
    else:
        return False, "⚠️ Emotion references not yet generated"

def check_tts_models():
    """Check if TTS models are downloaded"""
    models_status = {}
    
    # Check OpenVoiceV2
    openvoice_dir = Path("external/OpenVoice/checkpoints_v2")
    if openvoice_dir.exists() and any(openvoice_dir.iterdir()):
        models_status["OpenVoiceV2"] = "✅ Ready"
    else:
        models_status["OpenVoiceV2"] = "❌ Not downloaded"
    
    # Check Coqui TTS (with error handling)
    try:
        import importlib
        spec = importlib.util.find_spec("TTS.api")
        if spec is not None:
            models_status["Coqui TTS"] = "✅ Installed"
        else:
            models_status["Coqui TTS"] = "❌ Not installed"
    except Exception:
        models_status["Coqui TTS"] = "❌ Not installed"
    
    # Check Bark (with error handling)
    try:
        import importlib
        spec = importlib.util.find_spec("bark")
        if spec is not None:
            models_status["Bark"] = "✅ Installed"
        else:
            models_status["Bark"] = "❌ Not installed"
    except Exception:
        models_status["Bark"] = "❌ Not installed"
    
    return models_status

def main():
    """Main status check"""
    print("=" * 70)
    print("🔍 Setup Status Check")
    print("=" * 70)
    print()
    
    # Check CSM-1B
    print("1. CSM-1B Status:")
    csm_ready, csm_msg = check_csm_ready()
    print(f"   {csm_msg}")
    print()
    
    # Check TTS Models
    print("2. TTS Models Status:")
    tts_status = check_tts_models()
    for model, status in tts_status.items():
        print(f"   {model}: {status}")
    print()
    
    # Check Emotion References
    print("3. Emotion References Status:")
    refs_ready, refs_msg = check_emotion_references()
    print(f"   {refs_msg}")
    print()
    
    # Overall Status
    print("=" * 70)
    overall_ready = csm_ready and refs_ready
    if overall_ready:
        print("✅ Setup Complete - Ready for Production!")
    else:
        print("⚠️  Setup In Progress - Some components still downloading...")
        print()
        print("💡 To continue setup, run:")
        print("   export HUGGINGFACE_TOKEN=your_token")
        print("   python3 production/complete_setup.py")
    print("=" * 70)

if __name__ == "__main__":
    main()

