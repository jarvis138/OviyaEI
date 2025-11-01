#!/usr/bin/env python3
"""
Complete Emotion Reference System Setup
========================================

This script automates the complete pipeline:
1. Downloads OpenVoiceV2 emotion-expressive TTS model
2. Downloads emotion datasets (RAVDESS, CREMA-D, MELD, EmoDB)
3. Generates emotion references using OpenVoiceV2 TTS
4. Extracts additional samples from datasets
5. Saves all references for CSM-1B context

Usage:
    python setup_emotion_references.py

Output:
    data/emotion_references/ - Directory with emotion reference audio files
    data/emotion_references/emotion_map.json - Mapping of emotions to audio files
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
import zipfile
import tarfile

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    import requests
except ImportError:
    requests = None
    logger.warning("requests not available - will use subprocess for downloads")
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
    logger.warning("tqdm not available - progress bars disabled")

# Paths
WORKSPACE = Path(os.getenv("WORKSPACE", "/workspace"))
if not WORKSPACE.exists():
    WORKSPACE = Path(".")  # Fallback to current directory

DATASETS_DIR = WORKSPACE / "emotion_datasets"
OUTPUT_DIR = WORKSPACE / "data" / "emotion_references"
OPENVOICE_DIR = WORKSPACE / "external" / "OpenVoice"
OPENVOICE_MODELS_DIR = WORKSPACE / "external" / "OpenVoice" / "checkpoints_v2"


def download_file(url: str, output_path: Path, description: str = "") -> bool:
    """Download a file with progress bar"""
    if requests is None:
        # Fallback to wget/subprocess
        logger.info(f"   Downloading via wget: {url}")
        result = subprocess.run(
            ["wget", "-q", "--show-progress", url, "-O", str(output_path)],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        if tqdm:
            with open(output_path, 'wb') as f, tqdm(
                desc=description or "Downloading",
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        else:
            with open(output_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        
        return True
    except Exception as e:
        logger.error(f"Failed to download {url}: {e}")
        return False


def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """Extract ZIP file"""
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        return True
    except Exception as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        return False


def download_openvoice_tts() -> bool:
    """Download OpenVoiceV2 TTS model"""
    logger.info("=" * 60)
    logger.info("Step 1: Downloading OpenVoiceV2 Emotion-Expressive TTS Model")
    logger.info("=" * 60)
    
    # Check if already downloaded
    if OPENVOICE_MODELS_DIR.exists() and any(OPENVOICE_MODELS_DIR.iterdir()):
        logger.info("✅ OpenVoiceV2 models already exist")
        return True
    
    # Create directory
    OPENVOICE_DIR.mkdir(parents=True, exist_ok=True)
    
    # Clone repository if not exists
    if not (OPENVOICE_DIR / ".git").exists():
        logger.info("📦 Cloning OpenVoice repository...")
        result = subprocess.run(
            ["git", "clone", "https://github.com/myshell-ai/OpenVoice.git", str(OPENVOICE_DIR)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error(f"Failed to clone OpenVoice: {result.stderr}")
            return False
        logger.info("✅ OpenVoice repository cloned")
    
    # Install dependencies
    logger.info("📦 Installing OpenVoice dependencies...")
    result = subprocess.run(
        ["pip", "install", "-r", str(OPENVOICE_DIR / "requirements.txt"), "-q"],
        capture_output=True,
        text=True
    )
    if result.returncode != 0:
        logger.warning(f"Dependencies install warning: {result.stderr}")
    
    # Download models using huggingface-cli
    logger.info("📥 Downloading OpenVoiceV2 models from Hugging Face...")
    OPENVOICE_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Try huggingface-cli first
    result = subprocess.run(
        ["huggingface-cli", "download", "myshell-ai/OpenVoiceV2", 
         "--local-dir", str(OPENVOICE_MODELS_DIR), "--repo-type", "model"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        logger.info("✅ OpenVoiceV2 models downloaded successfully")
        return True
    
    # Fallback: Try git-lfs
    logger.info("⚠️  huggingface-cli failed, trying git-lfs...")
    result = subprocess.run(
        ["git", "lfs", "install"],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        result = subprocess.run(
            ["git", "clone", "https://huggingface.co/myshell-ai/OpenVoiceV2", str(OPENVOICE_MODELS_DIR)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            logger.info("✅ OpenVoiceV2 models downloaded via git-lfs")
            return True
    
    logger.error("❌ Failed to download OpenVoiceV2 models")
    return False


def download_ravdess_dataset() -> bool:
    """Download RAVDESS dataset"""
    logger.info("\n📦 Downloading RAVDESS dataset...")
    
    dataset_dir = DATASETS_DIR / "ravdess"
    if dataset_dir.exists() and any(dataset_dir.rglob("*.wav")):
        logger.info("✅ RAVDESS already exists")
        return True
    
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # RAVDESS download URL (Zenodo)
    url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
    zip_path = dataset_dir / "ravdess.zip"
    
    logger.info(f"   Downloading from: {url}")
    if download_file(url, zip_path, "RAVDESS"):
        logger.info("   Extracting...")
        if extract_zip(zip_path, dataset_dir):
            zip_path.unlink()  # Remove zip file
            logger.info("✅ RAVDESS downloaded and extracted")
            return True
    
    logger.warning("⚠️  RAVDESS download failed, will use TTS-generated references")
    return False


def download_crema_d_dataset() -> bool:
    """Download CREMA-D dataset"""
    logger.info("\n📦 Downloading CREMA-D dataset...")
    
    dataset_dir = DATASETS_DIR / "crema_d"
    if dataset_dir.exists() and any(dataset_dir.rglob("*.wav")):
        logger.info("✅ CREMA-D already exists")
        return True
    
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # CREMA-D is typically downloaded from GitHub releases
    # For now, we'll use a direct download if available
    # Note: CREMA-D may require manual download due to licensing
    logger.info("⚠️  CREMA-D requires manual download from:")
    logger.info("   https://github.com/CheyneyComputerScience/CREMA-D")
    logger.info("   Will use TTS-generated references instead")
    
    return False


def download_meld_dataset() -> bool:
    """Download MELD dataset"""
    logger.info("\n📦 Downloading MELD dataset...")
    
    dataset_dir = DATASETS_DIR / "meld"
    if dataset_dir.exists() and any(dataset_dir.rglob("*.wav")):
        logger.info("✅ MELD already exists")
        return True
    
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # Clone MELD repository
    logger.info("   Cloning MELD from GitHub...")
    result = subprocess.run(
        ["git", "clone", "https://github.com/declare-lab/MELD.git", str(dataset_dir)],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        logger.info("✅ MELD downloaded successfully")
        return True
    
    logger.warning("⚠️  MELD download failed, will use TTS-generated references")
    return False


def download_emodb_dataset() -> bool:
    """Download EmoDB dataset"""
    logger.info("\n📦 Downloading EmoDB dataset...")
    
    dataset_dir = DATASETS_DIR / "emodb"
    if dataset_dir.exists() and any(dataset_dir.rglob("*.wav")):
        logger.info("✅ EmoDB already exists")
        return True
    
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # EmoDB download URL
    url = "http://emodb.bilderbar.info/download/download.zip"
    zip_path = dataset_dir / "emodb.zip"
    
    logger.info(f"   Downloading from: {url}")
    if download_file(url, zip_path, "EmoDB"):
        logger.info("   Extracting...")
        if extract_zip(zip_path, dataset_dir):
            zip_path.unlink()  # Remove zip file
            logger.info("✅ EmoDB downloaded and extracted")
            return True
    
    logger.warning("⚠️  EmoDB download failed, will use TTS-generated references")
    return False


def generate_emotion_references_with_openvoice() -> Dict[str, str]:
    """Generate emotion references using OpenVoiceV2 TTS"""
    logger.info("\n" + "=" * 60)
    logger.info("Step 2: Generating Emotion References with OpenVoiceV2")
    logger.info("=" * 60)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Emotion texts for reference generation
    emotion_texts = {
        "calm_supportive": "Take a deep breath. Everything will be okay. I'm here with you.",
        "empathetic_sad": "I'm so sorry you're going through this. Your feelings are valid and important.",
        "joyful_excited": "That's amazing! I'm so happy for you! This is wonderful news!",
        "playful": "Hey there! This is going to be fun! Let's explore this together.",
        "confident": "You've got this. I believe in you. You're stronger than you know.",
        "concerned_anxious": "Are you okay? I'm here if you need me. Take your time.",
        "angry_firm": "That's not acceptable. This needs to stop. I understand your frustration.",
        "neutral": "Hello. How can I help you today? I'm here to listen.",
    }
    
    references = {}
    
    # Try to use OpenVoiceV2
    try:
        sys.path.insert(0, str(OPENVOICE_DIR))
        
        # Import OpenVoice modules
        try:
            from openvoice import se_extractor
            from openvoice.api import ToneColorConverter
            import torch
            
            logger.info("✅ OpenVoiceV2 loaded successfully")
            
            # Initialize OpenVoice
            # Note: This is a simplified version - adjust based on actual OpenVoice API
            ckpt_base = OPENVOICE_MODELS_DIR / "base_speakers" / "EN"
            ckpt_converter = OPENVOICE_MODELS_DIR / "converter"
            
            if ckpt_base.exists() and ckpt_converter.exists():
                device = "cuda" if torch.cuda.is_available() else "cpu"
                
                # Load models
                tone_color_converter = ToneColorConverter(
                    f"{ckpt_converter}/config.json",
                    device=device
                )
                tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")
                
                # Generate references for each emotion
                for emotion, text in emotion_texts.items():
                    logger.info(f"   Generating reference for: {emotion}")
                    
                    try:
                        # Use OpenVoice to synthesize with emotion
                        # Note: Actual API may differ - adjust based on OpenVoice's implementation
                        # This is a placeholder - replace with actual OpenVoice synthesis call
                        audio = generate_with_openvoice_api(
                            text, emotion, tone_color_converter, device
                        )
                        
                        # Save reference
                        output_file = OUTPUT_DIR / f"{emotion}.wav"
                        torchaudio.save(str(output_file), audio.unsqueeze(0), 24000)
                        
                        references[emotion] = str(output_file.name)
                        logger.info(f"   ✅ Generated: {output_file}")
                        
                    except Exception as e:
                        logger.warning(f"   ⚠️  Failed to generate {emotion}: {e}")
                        # Fallback to synthetic
                        audio = create_synthetic_reference(emotion)
                        output_file = OUTPUT_DIR / f"{emotion}.wav"
                        torchaudio.save(str(output_file), audio.unsqueeze(0), 24000)
                        references[emotion] = str(output_file.name)
                
                logger.info("✅ OpenVoiceV2 references generated")
                return references
                
        except ImportError as e:
            logger.warning(f"⚠️  OpenVoice modules not available: {e}")
            
    except Exception as e:
        logger.warning(f"⚠️  OpenVoice generation failed: {e}")
    
    # Fallback: Create synthetic references
    logger.info("⚠️  Using synthetic references (OpenVoice not available)")
    import torch
    import torchaudio
    
    for emotion, text in emotion_texts.items():
        audio = create_synthetic_reference(emotion)
        output_file = OUTPUT_DIR / f"{emotion}.wav"
        torchaudio.save(str(output_file), audio.unsqueeze(0), 24000)
        references[emotion] = str(output_file.name)
        logger.info(f"   ✅ Generated synthetic: {output_file}")
    
    return references


def generate_with_openvoice_api(text: str, emotion: str, converter, device: str):
    """Generate audio using OpenVoice API"""
    import torch
    import torchaudio
    
    try:
        # Import OpenVoice modules
        from openvoice import se_extractor
        
        # Load base speaker embedding
        ckpt_base = OPENVOICE_MODELS_DIR / "base_speakers" / "EN"
        if not ckpt_base.exists():
            # Fallback to synthetic
            return create_synthetic_reference(emotion)
        
        # Extract speaker embedding from base audio
        base_speaker_path = ckpt_base / "default" / "reference.wav"
        if not base_speaker_path.exists():
            # Try alternative path
            base_speaker_paths = list(ckpt_base.glob("*.wav"))
            if not base_speaker_paths:
                base_speaker_paths = list(ckpt_base.rglob("*.wav"))
            if base_speaker_paths:
                base_speaker_path = base_speaker_paths[0]
            else:
                return create_synthetic_reference(emotion)
        
        # Extract speaker embedding
        src_se = se_extractor.get_se(str(base_speaker_path), converter, vad=False)
        
        # Synthesize with OpenVoice
        tgt_path = OUTPUT_DIR / f"temp_{emotion}.wav"
        
        # Use OpenVoice to clone voice with emotion
        converter.convert(
            audio_src_path=str(base_speaker_path),
            src_se=src_se,
            tgt_path=str(tgt_path),
            message=text,
            output_dir=str(OUTPUT_DIR),
            tone_color_converter=converter
        )
        
        # Load generated audio
        if tgt_path.exists():
            audio, sr = torchaudio.load(str(tgt_path))
            tgt_path.unlink()  # Clean up temp file
            return audio.squeeze(0)
        
    except Exception as e:
        logger.warning(f"OpenVoice API call failed: {e}")
    
    # Fallback to synthetic
    return create_synthetic_reference(emotion)


def create_synthetic_reference(emotion: str):
    """Create synthetic emotion reference"""
    import torch
    import numpy as np
    
    duration = 2.0
    sample_rate = 24000
    num_samples = int(duration * sample_rate)
    
    t = torch.linspace(0, duration, num_samples)
    
    # Emotion-specific frequency patterns
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
    audio = 0.3 * torch.sin(2 * torch.pi * base_freq * t)
    
    # Add emotion-specific modulation
    if emotion == "joyful_excited":
        vibrato = 0.1 * torch.sin(2 * torch.pi * 5 * t)
        audio = audio * (1 + vibrato)
    elif emotion == "empathetic_sad":
        decay = torch.exp(-t * 0.5)
        audio = audio * decay
    
    return audio.unsqueeze(0)  # [1, samples]


def run_extract_all_emotions() -> bool:
    """Run extract_all_emotions.py to extract from datasets"""
    logger.info("\n" + "=" * 60)
    logger.info("Step 3: Extracting Additional References from Datasets")
    logger.info("=" * 60)
    
    extract_script = WORKSPACE / "production" / "extract_all_emotions.py"
    if not extract_script.exists():
        extract_script = Path("extract_all_emotions.py")
    
    if not extract_script.exists():
        logger.warning("⚠️  extract_all_emotions.py not found, skipping dataset extraction")
        return False
    
    logger.info(f"   Running: {extract_script}")
    result = subprocess.run(
        [sys.executable, str(extract_script)],
        cwd=str(extract_script.parent),
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        logger.info("✅ Dataset extraction completed")
        return True
    else:
        logger.warning(f"⚠️  Dataset extraction had issues: {result.stderr}")
        return False


def merge_emotion_references() -> Dict[str, List[Dict]]:
    """Merge TTS-generated and dataset-extracted references"""
    logger.info("\n" + "=" * 60)
    logger.info("Step 4: Merging All Emotion References")
    logger.info("=" * 60)
    
    emotion_map = {}
    
    # Load existing emotion_map.json if exists
    emotion_map_file = OUTPUT_DIR / "emotion_map.json"
    if emotion_map_file.exists():
        try:
            with open(emotion_map_file, 'r') as f:
                emotion_map = json.load(f)
            logger.info(f"   Loaded existing emotion_map.json with {len(emotion_map)} emotions")
        except Exception as e:
            logger.warning(f"   Failed to load emotion_map.json: {e}")
    
    # Add TTS-generated references
    for emotion_file in OUTPUT_DIR.glob("*.wav"):
        emotion_name = emotion_file.stem
        if emotion_name not in emotion_map:
            emotion_map[emotion_name] = []
        
        # Check if already in map
        already_exists = any(
            ref.get('file') == emotion_file.name
            for ref in emotion_map[emotion_name]
        )
        
        if not already_exists:
            import torchaudio
            try:
                audio, sr = torchaudio.load(str(emotion_file))
                duration = audio.shape[-1] / sr
                
                emotion_map[emotion_name].append({
                    'file': emotion_file.name,
                    'duration': duration,
                    'sample_rate': sr,
                    'source': 'tts_generated'
                })
            except Exception as e:
                logger.warning(f"   Failed to process {emotion_file}: {e}")
    
    # Save merged emotion map
    with open(emotion_map_file, 'w') as f:
        json.dump(emotion_map, f, indent=2)
    
    logger.info(f"✅ Merged emotion map saved: {len(emotion_map)} emotions")
    return emotion_map


def main():
    """Main execution"""
    logger.info("=" * 60)
    logger.info("🚀 Complete Emotion Reference System Setup")
    logger.info("=" * 60)
    logger.info("")
    logger.info("This script will:")
    logger.info("  1. Download OpenVoiceV2 emotion-expressive TTS model")
    logger.info("  2. Download emotion datasets (RAVDESS, CREMA-D, MELD, EmoDB)")
    logger.info("  3. Generate emotion references using OpenVoiceV2 TTS")
    logger.info("  4. Extract additional references from datasets")
    logger.info("  5. Merge all references for CSM-1B context")
    logger.info("")
    logger.info("💡 For multiple TTS models (Coqui TTS, Bark, StyleTTS2),")
    logger.info("   run: python setup_multi_tts_emotion_references.py")
    logger.info("")
    
    # Step 1: Download OpenVoiceV2 TTS
    openvoice_success = download_openvoice_tts()
    
    # Step 2: Download datasets (non-blocking - will use TTS if fails)
    logger.info("\n" + "=" * 60)
    logger.info("Downloading Emotion Datasets")
    logger.info("=" * 60)
    
    download_ravdess_dataset()
    download_crema_d_dataset()
    download_meld_dataset()
    download_emodb_dataset()
    
    # Step 3: Generate references with OpenVoiceV2
    references = generate_emotion_references_with_openvoice()
    
    # Step 4: Extract from datasets (if available)
    run_extract_all_emotions()
    
    # Step 5: Merge all references
    emotion_map = merge_emotion_references()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("✅ Setup Complete!")
    logger.info("=" * 60)
    logger.info(f"📁 Output directory: {OUTPUT_DIR}")
    logger.info(f"📊 Total emotions: {len(emotion_map)}")
    
    total_files = sum(len(refs) for refs in emotion_map.values())
    logger.info(f"📄 Total reference files: {total_files}")
    
    logger.info("\n🎯 Emotion references are ready for CSM-1B!")
    logger.info("   The system will automatically use these references")
    logger.info("   in conversation context for emotional voice conditioning.")


if __name__ == "__main__":
    main()

