#!/usr/bin/env python3
"""
Unified Multi-TTS Emotion Reference Setup
==========================================

This is the main entry point that orchestrates:
1. Multiple TTS model downloads (OpenVoiceV2, Coqui TTS, Bark, StyleTTS2)
2. Emotion dataset downloads
3. Reference generation from all TTS models
4. Dataset extraction
5. Complete integration for CSM-1B

Run this script for complete multi-TTS emotion reference extraction.
"""

import sys
import subprocess
from pathlib import Path

production_dir = Path(__file__).parent

def main():
    """Main entry point"""
    print("=" * 70)
    print("🚀 Multi-TTS Emotion Reference Extraction System")
    print("=" * 70)
    print()
    print("This will download and extract emotion references from:")
    print("  1. OpenVoiceV2 - Emotion-expressive voice cloning")
    print("  2. Coqui TTS (XTTS-v2) - Multilingual emotion control")
    print("  3. Bark (Suno AI) - Text-to-speech with emotion tags")
    print("  4. StyleTTS2 - Style transfer (may require manual download)")
    print("  5. Emotion datasets (RAVDESS, CREMA-D, MELD, EmoDB)")
    print()
    
    # Run multi-TTS setup
    multi_tts_script = production_dir / "setup_multi_tts_emotion_references.py"
    
    if not multi_tts_script.exists():
        print(f"❌ Multi-TTS script not found: {multi_tts_script}")
        return 1
    
    # Execute multi-TTS setup
    result = subprocess.run(
        [sys.executable, str(multi_tts_script)],
        cwd=str(production_dir)
    )
    
    if result.returncode == 0:
        # Also run dataset extraction
        extract_script = production_dir / "extract_all_emotions.py"
        if extract_script.exists():
            print("\n" + "=" * 70)
            print("Extracting Additional References from Datasets")
            print("=" * 70)
            subprocess.run(
                [sys.executable, str(extract_script)],
                cwd=str(production_dir)
            )
        
        print("\n" + "=" * 70)
        print("✅ Complete setup finished!")
        print("=" * 70)
        print("\n📋 Next steps:")
        print("   1. Emotion references are ready in data/emotion_references/")
        print("   2. CSM-1B will automatically use these references")
        print("   3. Start your server: python websocket_server.py")
        return 0
    else:
        print("\n❌ Setup failed with errors")
        return 1


if __name__ == "__main__":
    sys.exit(main())

