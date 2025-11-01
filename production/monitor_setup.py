#!/usr/bin/env python3
"""
Real-time Setup Progress Monitor
=================================

Monitors the setup process in real-time and provides progress updates.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
import json

def check_download_progress():
    """Check download progress"""
    status = {
        "csm_1b": False,
        "openvoice": False,
        "coqui": False,
        "bark": False,
        "emotion_refs": False
    }
    
    # Check CSM-1B
    try:
        from transformers import AutoProcessor
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        if any(cache_dir.glob("models--sesame--csm-1b*")):
            status["csm_1b"] = True
    except:
        pass
    
    # Check OpenVoiceV2
    openvoice_dir = Path("external/OpenVoice/checkpoints_v2")
    if openvoice_dir.exists():
        # Check if downloading (has .tmp files) or complete
        has_files = any(openvoice_dir.rglob("*"))
        has_tmp = any(openvoice_dir.rglob("*.tmp"))
        if has_files and not has_tmp:
            status["openvoice"] = True
        elif has_tmp:
            status["openvoice"] = "downloading"
    
    # Check Coqui TTS
    try:
        from TTS.api import TTS
        status["coqui"] = True
    except ImportError:
        pass
    
    # Check Bark
    try:
        from bark import generate_audio
        status["bark"] = True
    except ImportError:
        pass
    
    # Check emotion references
    ref_dir = Path("data/emotion_references")
    map_file = ref_dir / "emotion_map.json"
    if map_file.exists():
        try:
            with open(map_file, 'r') as f:
                emotion_map = json.load(f)
            if len(emotion_map) > 0:
                status["emotion_refs"] = True
        except:
            pass
    
    return status

def display_progress(status):
    """Display progress in a nice format"""
    print("\r" + " " * 80, end="")  # Clear line
    print("\r🔍 Progress: ", end="")
    
    components = []
    if status["csm_1b"]:
        components.append("✅ CSM-1B")
    else:
        components.append("⏳ CSM-1B")
    
    if status["openvoice"] == True:
        components.append("✅ OpenVoice")
    elif status["openvoice"] == "downloading":
        components.append("⬇️  OpenVoice")
    else:
        components.append("⏳ OpenVoice")
    
    if status["coqui"]:
        components.append("✅ Coqui")
    else:
        components.append("⏳ Coqui")
    
    if status["bark"]:
        components.append("✅ Bark")
    else:
        components.append("⏳ Bark")
    
    if status["emotion_refs"]:
        components.append("✅ Refs")
    else:
        components.append("⏳ Refs")
    
    print(" | ".join(components), end="", flush=True)

def main():
    """Main monitoring loop"""
    print("=" * 70)
    print("🚀 Setup Progress Monitor")
    print("=" * 70)
    print()
    print("Press Ctrl+C to stop monitoring")
    print()
    
    try:
        while True:
            status = check_download_progress()
            display_progress(status)
            
            # Check if everything is complete
            if all([
                status["csm_1b"] == True,
                status["openvoice"] == True,
                status["coqui"] == True,
                status["bark"] == True,
                status["emotion_refs"] == True
            ]):
                print()
                print()
                print("=" * 70)
                print("✅ Setup Complete!")
                print("=" * 70)
                print()
                print("All components are ready:")
                print("  ✅ CSM-1B model")
                print("  ✅ OpenVoiceV2")
                print("  ✅ Coqui TTS")
                print("  ✅ Bark")
                print("  ✅ Emotion references")
                print()
                print("🎯 Ready for production!")
                break
            
            time.sleep(2)  # Check every 2 seconds
            
    except KeyboardInterrupt:
        print()
        print()
        print("Monitoring stopped.")
        print("Run 'python3 check_setup_status.py' for final status.")

if __name__ == "__main__":
    main()

