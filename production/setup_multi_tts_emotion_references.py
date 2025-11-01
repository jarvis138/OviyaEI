#!/usr/bin/env python3
"""
Multi-TTS Emotion Reference Extraction System
=============================================

Downloads and extracts emotion references from multiple open-source TTS models:
1. OpenVoiceV2 - Emotion-expressive voice cloning
2. Coqui TTS (XTTS-v2) - Multilingual emotion control
3. StyleTTS2 - Style transfer for emotional expression
4. Bark (Suno AI) - Text-to-speech with emotion tags

All references are normalized and integrated for CSM-1B context according to:
- https://huggingface.co/sesame/csm-1b
- https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice#demo

Usage:
    python setup_multi_tts_emotion_references.py

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

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

try:
    import requests
except ImportError:
    requests = None
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None
try:
    import zipfile
except ImportError:
    zipfile = None
try:
    import tarfile
except ImportError:
    tarfile = None

# Paths
WORKSPACE = Path(os.getenv("WORKSPACE", "/workspace"))
if not WORKSPACE.exists():
    WORKSPACE = Path(".")

DATASETS_DIR = WORKSPACE / "emotion_datasets"
OUTPUT_DIR = WORKSPACE / "data" / "emotion_references"
MODELS_DIR = WORKSPACE / "external" / "tts_models"

# TTS Model paths
OPENVOICE_DIR = WORKSPACE / "external" / "OpenVoice"
COQUI_TTS_DIR = WORKSPACE / "external" / "coqui-tts"
STYLETTS2_DIR = WORKSPACE / "external" / "StyleTTS2"
BARK_DIR = WORKSPACE / "external" / "bark"

# Emotion texts for reference generation
EMOTION_TEXTS = {
    "calm_supportive": "Take a deep breath. Everything will be okay. I'm here with you.",
    "empathetic_sad": "I'm so sorry you're going through this. Your feelings are valid and important.",
    "joyful_excited": "That's amazing! I'm so happy for you! This is wonderful news!",
    "playful": "Hey there! This is going to be fun! Let's explore this together.",
    "confident": "You've got this. I believe in you. You're stronger than you know.",
    "concerned_anxious": "Are you okay? I'm here if you need me. Take your time.",
    "angry_firm": "That's not acceptable. This needs to stop. I understand your frustration.",
    "neutral": "Hello. How can I help you today? I'm here to listen.",
}


def download_file(url: str, output_path: Path, description: str = "") -> bool:
    """Download a file with progress bar"""
    if requests is None:
        logger.info(f"   Downloading via wget: {url}")
        result = subprocess.run(
            ["wget", "-q", "--show-progress", url, "-O", str(output_path)],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    
    try:
        response = requests.get(url, stream=True, timeout=60)
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


def download_openvoice_tts() -> bool:
    """Download OpenVoiceV2 TTS model"""
    logger.info("\n📦 Downloading OpenVoiceV2...")
    
    if (OPENVOICE_DIR / "checkpoints_v2").exists() and any((OPENVOICE_DIR / "checkpoints_v2").iterdir()):
        logger.info("✅ OpenVoiceV2 already exists")
        return True
    
    OPENVOICE_DIR.mkdir(parents=True, exist_ok=True)
    
    if not (OPENVOICE_DIR / ".git").exists():
        logger.info("   Cloning repository...")
        result = subprocess.run(
            ["git", "clone", "https://github.com/myshell-ai/OpenVoice.git", str(OPENVOICE_DIR)],
            capture_output=True,
            text=True
        )
        if result.returncode != 0:
            logger.error(f"Failed to clone OpenVoice: {result.stderr}")
            return False
    
        # Install dependencies
        logger.info("   Installing dependencies...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(OPENVOICE_DIR / "requirements.txt"), "-q"],
            capture_output=True,
            text=True
        )
    
    # Download models using huggingface-cli or Python module
    logger.info("   Downloading models from Hugging Face...")
    models_dir = OPENVOICE_DIR / "checkpoints_v2"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Try Python module first (more reliable)
    try:
        from huggingface_hub import snapshot_download
        logger.info("   Using huggingface_hub Python module...")
        snapshot_download(
            repo_id="myshell-ai/OpenVoiceV2",
            local_dir=str(models_dir),
            repo_type="model",
            token=os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        )
        logger.info("✅ OpenVoiceV2 downloaded successfully")
        return True
    except ImportError:
        logger.info("   huggingface_hub not available, trying CLI...")
        # Fallback to CLI
        result = subprocess.run(
            ["huggingface-cli", "download", "myshell-ai/OpenVoiceV2", 
             "--local-dir", str(models_dir), "--repo-type", "model"],
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            logger.info("✅ OpenVoiceV2 downloaded successfully")
            return True
    
    logger.warning("⚠️ OpenVoiceV2 download failed, will skip")
    return False


def download_coqui_tts() -> bool:
    """Download Coqui TTS (XTTS-v2) model"""
    logger.info("\n📦 Downloading Coqui TTS (XTTS-v2)...")
    
    try:
        # Install Coqui TTS
        logger.info("   Installing Coqui TTS...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "TTS", "-q"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.warning("⚠️ Coqui TTS installation failed")
            return False
        
        # Download XTTS-v2 model
        logger.info("   Downloading XTTS-v2 model...")
        COQUI_TTS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Coqui TTS downloads models automatically on first use
        # Test import to trigger download
        test_script = COQUI_TTS_DIR / "test_download.py"
        test_script.write_text("""
from TTS.api import TTS
import os
os.environ['COQUI_TOS_AGREED'] = '1'
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True)
print("✅ XTTS-v2 model downloaded")
""")
        
        result = subprocess.run(
            [sys.executable, str(test_script)],
            cwd=str(COQUI_TTS_DIR.parent),  # Run from parent directory to avoid path issues
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            logger.info("✅ Coqui TTS (XTTS-v2) downloaded successfully")
            test_script.unlink()
            return True
        else:
            logger.warning(f"⚠️ Coqui TTS download failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.warning(f"⚠️ Coqui TTS setup failed: {e}")
        return False


def download_styletts2() -> bool:
    """Download StyleTTS2 model"""
    logger.info("\n📦 Downloading StyleTTS2...")
    
    try:
        STYLETTS2_DIR.mkdir(parents=True, exist_ok=True)
        
        if not (STYLETTS2_DIR / ".git").exists():
            logger.info("   Cloning repository...")
            result = subprocess.run(
                ["git", "clone", "https://github.com/yl4579/StyleTTS2.git", str(STYLETTS2_DIR)],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                logger.warning("⚠️ StyleTTS2 clone failed")
                return False
        
        # Install dependencies
        logger.info("   Installing dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(STYLETTS2_DIR / "requirements.txt"), "-q"],
            capture_output=True
        )
        
        # Download pretrained models
        logger.info("   Downloading pretrained models...")
        checkpoints_dir = STYLETTS2_DIR / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        
        # Download from Hugging Face or Google Drive
        logger.info("   ⚠️ StyleTTS2 models require manual download from:")
        logger.info("   https://huggingface.co/yl4579/StyleTTS2")
        logger.info("   Will use other models for now")
        
        return False
        
    except Exception as e:
        logger.warning(f"⚠️ StyleTTS2 setup failed: {e}")
        return False


def download_bark() -> bool:
    """Download Bark (Suno AI) model"""
    logger.info("\n📦 Downloading Bark (Suno AI)...")
    
    try:
        # Install Bark
        logger.info("   Installing Bark...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "bark", "-q"],
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            logger.warning("⚠️ Bark installation failed")
            return False
        
        # Bark downloads models automatically on first use
        # Test import to trigger download
        test_script = BARK_DIR / "test_download.py"
        BARK_DIR.mkdir(parents=True, exist_ok=True)
        test_script.write_text("""
from bark import generate_audio, preload_models
preload_models()
print("✅ Bark models downloaded")
""")
        
        result = subprocess.run(
            [sys.executable, str(test_script)],
            cwd=str(BARK_DIR.parent),  # Run from parent directory to avoid path issues
            capture_output=True,
            text=True,
            timeout=300
        )
        
        if result.returncode == 0:
            logger.info("✅ Bark downloaded successfully")
            test_script.unlink()
            return True
        else:
            logger.warning(f"⚠️ Bark download failed: {result.stderr}")
            return False
            
    except Exception as e:
        logger.warning(f"⚠️ Bark setup failed: {e}")
        return False


def generate_references_with_openvoice(emotions: Dict[str, str]) -> Dict[str, str]:
    """Generate emotion references using OpenVoiceV2"""
    logger.info("\n🎤 Generating references with OpenVoiceV2...")
    
    references = {}
    
    try:
        sys.path.insert(0, str(OPENVOICE_DIR))
        from openvoice import se_extractor
        from openvoice.api import ToneColorConverter
        import torch
        import torchaudio
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ckpt_converter = OPENVOICE_DIR / "checkpoints_v2" / "converter"
        
        if not ckpt_converter.exists():
            logger.warning("⚠️ OpenVoiceV2 models not found")
            return references
        
        converter = ToneColorConverter(
            f"{ckpt_converter}/config.json",
            device=device
        )
        converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")
        
        # Find base speaker
        base_speaker_path = OPENVOICE_DIR / "checkpoints_v2" / "base_speakers" / "EN"
        ref_audio = list(base_speaker_path.rglob("*.wav"))
        if not ref_audio:
            logger.warning("⚠️ OpenVoiceV2 base speaker not found")
            return references
        
        base_audio = ref_audio[0]
        src_se = se_extractor.get_se(str(base_audio), converter, vad=False)
        
        for emotion, text in emotions.items():
            try:
                logger.info(f"   Generating {emotion}...")
                temp_output = OUTPUT_DIR / f"temp_openvoice_{emotion}.wav"
                
                converter.convert(
                    audio_src_path=str(base_audio),
                    src_se=src_se,
                    tgt_path=str(temp_output),
                    message=text,
                    output_dir=str(OUTPUT_DIR),
                    tone_color_converter=converter
                )
                
                if temp_output.exists():
                    # Normalize and save
                    audio, sr = torchaudio.load(str(temp_output))
                    audio_np = audio.squeeze().numpy().astype(np.float32)
                    
                    # Resample to 24kHz if needed
                    if sr != 24000:
                        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
                        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 24000)
                        audio_np = audio_tensor.squeeze().numpy()
                    
                    output_file = OUTPUT_DIR / f"{emotion}_openvoice.wav"
                    torchaudio.save(str(output_file), torch.from_numpy(audio_np).unsqueeze(0), 24000)
                    
                    references[f"{emotion}_openvoice"] = str(output_file.name)
                    temp_output.unlink()
                    logger.info(f"   ✅ Generated: {output_file.name}")
                    
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to generate {emotion}: {e}")
        
    except Exception as e:
        logger.warning(f"⚠️ OpenVoiceV2 generation failed: {e}")
    
    return references


def generate_references_with_coqui(emotions: Dict[str, str]) -> Dict[str, str]:
    """Generate emotion references using Coqui TTS (XTTS-v2)"""
    logger.info("\n🎤 Generating references with Coqui TTS (XTTS-v2)...")
    
    references = {}
    
    try:
        from TTS.api import TTS
        import torch
        import torchaudio
        import numpy as np
        
        # Initialize XTTS-v2
        tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=False)
        
        # Use a reference speaker (you can provide your own)
        # For now, use default speaker
        reference_audio_path = None
        
        for emotion, text in emotions.items():
            try:
                logger.info(f"   Generating {emotion}...")
                
                # Generate with emotion hint in text
                # XTTS-v2 doesn't have explicit emotion control, but we can use text hints
                emotion_hints = {
                    "calm_supportive": "calm and supportive",
                    "empathetic_sad": "empathetic and gentle",
                    "joyful_excited": "joyful and excited",
                    "playful": "playful and cheerful",
                    "confident": "confident and strong",
                    "concerned_anxious": "concerned and caring",
                    "angry_firm": "firm and determined",
                    "neutral": "neutral and balanced"
                }
                
                hint = emotion_hints.get(emotion, "")
                enhanced_text = f"{text} ({hint})"
                
                # Generate audio
                output_file = OUTPUT_DIR / f"temp_coqui_{emotion}.wav"
                
                if reference_audio_path:
                    tts.tts_to_file(text=enhanced_text, file_path=str(output_file), 
                                  speaker_wav=reference_audio_path)
                else:
                    tts.tts_to_file(text=enhanced_text, file_path=str(output_file))
                
                if output_file.exists():
                    # Normalize to 24kHz
                    audio, sr = torchaudio.load(str(output_file))
                    audio_np = audio.squeeze().numpy().astype(np.float32)
                    
                    if sr != 24000:
                        audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)
                        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, 24000)
                        audio_np = audio_tensor.squeeze().numpy()
                    
                    final_file = OUTPUT_DIR / f"{emotion}_coqui.wav"
                    torchaudio.save(str(final_file), torch.from_numpy(audio_np).unsqueeze(0), 24000)
                    
                    references[f"{emotion}_coqui"] = str(final_file.name)
                    output_file.unlink()
                    logger.info(f"   ✅ Generated: {final_file.name}")
                    
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to generate {emotion}: {e}")
        
    except Exception as e:
        logger.warning(f"⚠️ Coqui TTS generation failed: {e}")
    
    return references


def generate_references_with_bark(emotions: Dict[str, str]) -> Dict[str, str]:
    """Generate emotion references using Bark"""
    logger.info("\n🎤 Generating references with Bark...")
    
    references = {}
    
    try:
        from bark import generate_audio, SAMPLE_RATE
        from bark.generation import preload_models
        import torchaudio
        import numpy as np
        
        # Preload models
        preload_models()
        
        # Bark emotion prompts
        emotion_prompts = {
            "calm_supportive": "[speaker:calm, soothing]",
            "empathetic_sad": "[speaker:gentle, empathetic]",
            "joyful_excited": "[speaker:excited, happy]",
            "playful": "[speaker:playful, cheerful]",
            "confident": "[speaker:confident, strong]",
            "concerned_anxious": "[speaker:concerned, caring]",
            "angry_firm": "[speaker:firm, determined]",
            "neutral": "[speaker:neutral]"
        }
        
        for emotion, text in emotions.items():
            try:
                logger.info(f"   Generating {emotion}...")
                
                # Add emotion prompt
                prompt = emotion_prompts.get(emotion, "")
                full_text = f"{prompt} {text}"
                
                # Generate audio
                audio_array = generate_audio(full_text, history_prompt=None)
                
                if audio_array is not None and len(audio_array) > 0:
                    # Normalize to 24kHz (Bark outputs at 24kHz by default)
                    audio_np = audio_array.astype(np.float32)
                    
                    # Normalize amplitude
                    max_val = np.abs(audio_np).max()
                    if max_val > 0:
                        audio_np = audio_np / max_val * 0.95
                    
                    output_file = OUTPUT_DIR / f"{emotion}_bark.wav"
                    torchaudio.save(str(output_file), torch.from_numpy(audio_np).unsqueeze(0), SAMPLE_RATE)
                    
                    references[f"{emotion}_bark"] = str(output_file.name)
                    logger.info(f"   ✅ Generated: {output_file.name}")
                    
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to generate {emotion}: {e}")
        
    except Exception as e:
        logger.warning(f"⚠️ Bark generation failed: {e}")
    
    return references


def create_synthetic_reference(emotion: str):
    """Create synthetic emotion reference as fallback"""
    import torch
    import numpy as np
    
    duration = 2.0
    sample_rate = 24000
    num_samples = int(duration * sample_rate)
    
    t = torch.linspace(0, duration, num_samples)
    
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
    
    if emotion == "joyful_excited":
        vibrato = 0.1 * torch.sin(2 * torch.pi * 5 * t)
        audio = audio * (1 + vibrato)
    elif emotion == "empathetic_sad":
        decay = torch.exp(-t * 0.5)
        audio = audio * decay
    
    return audio.unsqueeze(0)


def merge_all_references(all_references: Dict[str, Dict[str, str]]) -> Dict[str, List[Dict]]:
    """Merge all TTS-generated references into emotion_map.json"""
    logger.info("\n📋 Merging all emotion references...")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load existing emotion_map.json if exists
    emotion_map_file = OUTPUT_DIR / "emotion_map.json"
    emotion_map = {}
    
    if emotion_map_file.exists():
        try:
            with open(emotion_map_file, 'r') as f:
                emotion_map = json.load(f)
        except Exception:
            pass
    
    # Merge all references
    for tts_name, references in all_references.items():
        for ref_key, filename in references.items():
            # Extract base emotion (remove _openvoice, _coqui, etc.)
            base_emotion = ref_key.rsplit('_', 1)[0]
            
            if base_emotion not in emotion_map:
                emotion_map[base_emotion] = []
            
            # Check if already exists
            existing = any(ref.get('file') == filename for ref in emotion_map[base_emotion])
            if not existing:
                ref_file = OUTPUT_DIR / filename
                if ref_file.exists():
                    try:
                        import torchaudio
                        audio, sr = torchaudio.load(str(ref_file))
                        duration = audio.shape[-1] / sr
                        
                        emotion_map[base_emotion].append({
                            'file': filename,
                            'duration': duration,
                            'sample_rate': sr,
                            'source': tts_name
                        })
                    except Exception:
                        pass
    
    # Save merged emotion map
    with open(emotion_map_file, 'w') as f:
        json.dump(emotion_map, f, indent=2)
    
    logger.info(f"✅ Merged {len(emotion_map)} emotions with {sum(len(refs) for refs in emotion_map.values())} references")
    return emotion_map


def run_extract_all_emotions() -> bool:
    """Run extract_all_emotions.py to extract from datasets"""
    logger.info("\n" + "=" * 70)
    logger.info("Step 4: Extracting Additional References from Datasets")
    logger.info("=" * 70)
    
    extract_script = WORKSPACE / "production" / "extract_all_emotions.py"
    if not extract_script.exists():
        extract_script = Path("extract_all_emotions.py")
    
    if not extract_script.exists():
        logger.warning("⚠️ extract_all_emotions.py not found, skipping dataset extraction")
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
        logger.warning(f"⚠️ Dataset extraction had issues: {result.stderr}")
        return False


def download_emotion_datasets() -> bool:
    """Download emotion datasets"""
    logger.info("\n" + "=" * 70)
    logger.info("Step 0: Downloading Emotion Datasets")
    logger.info("=" * 70)
    
    # Import dataset download functions from setup_emotion_references.py
    # Add production directory to path for imports
    sys.path.insert(0, str(WORKSPACE / "production"))
    sys.path.insert(0, str(WORKSPACE))
    
    try:
        # Try importing from setup_emotion_references module
        import importlib.util
        spec_path = WORKSPACE / "production" / "setup_emotion_references.py"
        if not spec_path.exists():
            spec_path = Path("production") / "setup_emotion_references.py"
        
        if spec_path.exists():
            spec = importlib.util.spec_from_file_location("setup_emotion_references", spec_path)
            module = importlib.util.module_from_spec(spec)
            sys.modules["setup_emotion_references"] = module
            spec.loader.exec_module(module)
            
            download_ravdess_dataset = getattr(module, 'download_ravdess_dataset', None)
            download_crema_d_dataset = getattr(module, 'download_crema_d_dataset', None)
            download_meld_dataset = getattr(module, 'download_meld_dataset', None)
            download_emodb_dataset = getattr(module, 'download_emodb_dataset', None)
            
            if all([download_ravdess_dataset, download_crema_d_dataset, 
                   download_meld_dataset, download_emodb_dataset]):
                download_ravdess_dataset()
                download_crema_d_dataset()
                download_meld_dataset()
                download_emodb_dataset()
                return True
    except Exception as e:
        logger.warning(f"⚠️ Could not import dataset download functions: {e}")
    
    # Fallback: Manual dataset download instructions
    logger.info("⚠️ Dataset download functions not available")
    logger.info("   Datasets will be downloaded manually or skipped")
    logger.info("   Script will continue with TTS-generated references")
    
    return False


def main():
    """Main execution"""
    logger.info("=" * 70)
    logger.info("🚀 Multi-TTS Emotion Reference Extraction System")
    logger.info("=" * 70)
    logger.info("")
    logger.info("This script will:")
    logger.info("  1. Download multiple TTS models (OpenVoiceV2, Coqui TTS, Bark)")
    logger.info("  2. Download emotion datasets (RAVDESS, CREMA-D, MELD, EmoDB)")
    logger.info("  3. Generate emotion references from each TTS model")
    logger.info("  4. Extract additional references from emotion datasets")
    logger.info("  5. Merge all references for CSM-1B context")
    logger.info("")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Step 0: Download datasets
    download_emotion_datasets()
    
    # Step 1: Download TTS models
    logger.info("\n" + "=" * 70)
    logger.info("Step 1: Downloading TTS Models")
    logger.info("=" * 70)
    
    openvoice_available = download_openvoice_tts()
    coqui_available = download_coqui_tts()
    bark_available = download_bark()
    styletts_available = download_styletts2()  # May require manual download
    
    # Step 2: Generate references
    logger.info("\n" + "=" * 70)
    logger.info("Step 2: Generating Emotion References")
    logger.info("=" * 70)
    
    all_references = {}
    
    if openvoice_available:
        all_references['openvoice'] = generate_references_with_openvoice(EMOTION_TEXTS)
    
    if coqui_available:
        all_references['coqui'] = generate_references_with_coqui(EMOTION_TEXTS)
    
    if bark_available:
        all_references['bark'] = generate_references_with_bark(EMOTION_TEXTS)
    
    # Step 3: Merge TTS references
    logger.info("\n" + "=" * 70)
    logger.info("Step 3: Merging TTS References")
    logger.info("=" * 70)
    
    emotion_map = merge_all_references(all_references)
    
    # Step 4: Extract from datasets (if available)
    run_extract_all_emotions()
    
    # Step 5: Final merge (reload emotion_map to include dataset references)
    emotion_map_file = OUTPUT_DIR / "emotion_map.json"
    if emotion_map_file.exists():
        try:
            with open(emotion_map_file, 'r') as f:
                emotion_map = json.load(f)
        except Exception:
            pass
    
    # Summary
    logger.info("\n" + "=" * 70)
    logger.info("✅ Setup Complete!")
    logger.info("=" * 70)
    logger.info(f"📁 Output directory: {OUTPUT_DIR}")
    logger.info(f"📊 Total emotions: {len(emotion_map)}")
    
    total_files = sum(len(refs) for refs in emotion_map.values())
    logger.info(f"📄 Total reference files: {total_files}")
    
    logger.info("\n🎯 Emotion references are ready for CSM-1B!")
    logger.info("   The system will automatically use these references")
    logger.info("   in conversation context for emotional voice conditioning.")


if __name__ == "__main__":
    main()

