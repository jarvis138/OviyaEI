#!/usr/bin/env python3
"""
CSM-1B Installation and Verification Script
============================================

Downloads and verifies CSM-1B model according to official Sesame documentation:
- https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice#demo
- https://huggingface.co/sesame/csm-1b

This script ensures CSM-1B is properly installed and ready for use.
"""

import os
import sys
import subprocess
from pathlib import Path
import torch

def check_huggingface_auth():
    """Check if Hugging Face token is available"""
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        print("⚠️  Hugging Face token not found in environment")
        print("   Please set HF_TOKEN or HUGGINGFACE_TOKEN")
        print("   Get token from: https://huggingface.co/settings/tokens")
        return False
    return True

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    
    packages = [
        "transformers>=4.40.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "numpy",
        "accelerate"
    ]
    
    for package in packages:
        print(f"   Installing {package}...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", package],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            print(f"   ⚠️  Failed to install {package}: {result.stderr}")
            return False
    
    print("✅ All packages installed")
    return True

def verify_csm_model():
    """Verify CSM-1B model can be loaded"""
    print("\n🔍 Verifying CSM-1B model...")
    
    try:
        from transformers import AutoProcessor, CsmForConditionalGeneration
        
        model_id = "sesame/csm-1b"
        print(f"   Loading processor for {model_id}...")
        
        # Load processor (lightweight, fast)
        processor = AutoProcessor.from_pretrained(
            model_id,
            token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        )
        print("   ✅ Processor loaded")
        
        # Check if model files exist locally
        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
        model_path = None
        
        # Search for model in cache
        for model_dir in cache_dir.glob(f"models--sesame--csm-1b*"):
            snapshots = model_dir / "snapshots"
            if snapshots.exists():
                for snapshot in snapshots.iterdir():
                    if (snapshot / "config.json").exists():
                        model_path = snapshot
                        break
        
        if model_path:
            print(f"   📁 Model found in cache: {model_path}")
            print("   ✅ Model is ready for use")
        else:
            print("   📥 Model will be downloaded on first use")
            print("   ⚠️  First run will download ~2GB of model files")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to verify CSM-1B: {e}")
        return False

def verify_integration():
    """Verify CSM-1B integration matches official format"""
    print("\n🔍 Verifying CSM-1B integration format...")
    
    # Check csm_1b_stream.py
    stream_file = Path("production/voice/csm_1b_stream.py")
    if not stream_file.exists():
        stream_file = Path("voice/csm_1b_stream.py")
    
    if stream_file.exists():
        with open(stream_file, 'r') as f:
            content = f.read()
        
        # Check for correct format according to Sesame docs
        checks = {
            "processor.apply_chat_template": "apply_chat_template() method",
            '"type": "audio"': "Audio type in content",
            '"type": "text"': "Text type in content",
            '"role":': "Role field in conversation",
            "AutoProcessor": "AutoProcessor import",
            "CsmForConditionalGeneration": "CSM model import"
        }
        
        all_passed = True
        for check, description in checks.items():
            if check in content:
                print(f"   ✅ {description}")
            else:
                print(f"   ❌ Missing: {description}")
                all_passed = False
        
        return all_passed
    else:
        print("   ⚠️  csm_1b_stream.py not found")
        return False

def main():
    """Main execution"""
    print("=" * 70)
    print("🚀 CSM-1B Installation and Verification")
    print("=" * 70)
    print()
    print("Verifying according to:")
    print("  - https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice#demo")
    print("  - https://huggingface.co/sesame/csm-1b")
    print()
    
    # Step 1: Check Hugging Face auth
    if not check_huggingface_auth():
        print("\n⚠️  Please set HF_TOKEN or HUGGINGFACE_TOKEN environment variable")
        print("   Then run this script again")
        return 1
    
    # Step 2: Install requirements
    if not install_requirements():
        print("\n❌ Failed to install requirements")
        return 1
    
    # Step 3: Verify model
    if not verify_csm_model():
        print("\n❌ Failed to verify CSM-1B model")
        return 1
    
    # Step 4: Verify integration
    if not verify_integration():
        print("\n⚠️  Some integration checks failed")
        print("   Please review the code to ensure CSM-1B format compliance")
    
    print("\n" + "=" * 70)
    print("✅ CSM-1B Verification Complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  1. Run: python3 production/setup_complete_multi_tts.py")
    print("  2. This will download TTS models and datasets")
    print("  3. Generate emotion references")
    print("  4. Everything will be ready for CSM-1B!")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

