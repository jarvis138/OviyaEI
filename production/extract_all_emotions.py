#!/usr/bin/env python3
"""
Extract Emotion Audio Samples from Datasets for CSM-1B Context

This script:
1. Extracts emotion-labeled audio samples from emotion datasets (RAVDESS, CREMA-D, MELD, EmoDB)
2. Normalizes and preprocesses audio for CSM-1B (24kHz, mono, float32)
3. Saves reference audio samples organized by emotion
4. Creates emotion-to-audio mapping for CSM conversation context

Usage:
    python extract_all_emotions.py
    
Output:
    data/emotion_references/ - Directory with emotion reference audio files
    data/emotion_references/emotion_map.json - Mapping of emotions to audio files
"""

import os
import json
import numpy as np
import torchaudio
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Target output format for CSM-1B
TARGET_SAMPLE_RATE = 24000
TARGET_CHANNELS = 1  # Mono
TARGET_DTYPE = np.float32

# Emotion mapping from datasets to Oviya's emotion taxonomy
EMOTION_MAPPING = {
    # RAVDESS emotions
    "neutral": "neutral",
    "calm": "calm_supportive",
    "happy": "joyful_excited",
    "sad": "empathetic_sad",
    "angry": "angry_firm",
    "fearful": "concerned_anxious",
    "disgust": "concerned_anxious",
    "surprised": "playful",
    
    # CREMA-D emotions
    "ANG": "angry_firm",
    "DIS": "concerned_anxious",
    "FEA": "concerned_anxious",
    "HAP": "joyful_excited",
    "NEU": "neutral",
    "SAD": "empathetic_sad",
    
    # MELD emotions
    "joy": "joyful_excited",
    "sadness": "empathetic_sad",
    "anger": "angry_firm",
    "fear": "concerned_anxious",
    "disgust": "concerned_anxious",
    "surprise": "playful",
    
    # EmoDB emotions
    "W": "neutral",
    "F": "happy",
    "T": "sad",
    "N": "angry",
    "E": "disgust",
    "A": "fear",
    "L": "bored"
}


def normalize_audio(audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, int]:
    """
    Normalize audio to CSM-1B format
    
    Args:
        audio: Audio array (any shape)
        sample_rate: Original sample rate
        
    Returns:
        (normalized_audio, target_sample_rate)
    """
    # Convert to numpy if tensor
    if isinstance(audio, torch.Tensor):
        audio = audio.numpy()
    
    # Ensure float32
    if audio.dtype != np.float32:
        if audio.dtype == np.int16:
            audio = audio.astype(np.float32) / 32768.0
        elif audio.dtype == np.int32:
            audio = audio.astype(np.float32) / 2147483648.0
        else:
            audio = audio.astype(np.float32)
    
    # Resample if needed
    if sample_rate != TARGET_SAMPLE_RATE:
        try:
            import torch
            audio_tensor = torch.from_numpy(audio)
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)  # [1, samples]
            
            resampler = torchaudio.transforms.Resample(
                sample_rate, TARGET_SAMPLE_RATE
            )
            audio_tensor = resampler(audio_tensor)
            audio = audio_tensor.squeeze(0).numpy()
        except Exception as e:
            logger.warning(f"Resampling failed: {e}, using scipy")
            from scipy import signal
            num_samples = int(len(audio) * TARGET_SAMPLE_RATE / sample_rate)
            audio = signal.resample(audio, num_samples).astype(np.float32)
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=0)
    
    # Normalize volume (peak normalization)
    max_val = np.abs(audio).max()
    if max_val > 0:
        audio = audio / max_val * 0.95  # Leave headroom
    
    return audio, TARGET_SAMPLE_RATE


def extract_ravdess_emotions(dataset_path: Path) -> Dict[str, List[Tuple[np.ndarray, int]]]:
    """
    Extract emotions from RAVDESS dataset
    
    RAVDESS filename format:
    Modality-Actor-Gender-Emotion-Statement-Repetition.wav
    Example: 03-01-06-01-01-01.wav
    Emotion codes: 01=neutral, 02=calm, 03=happy, 04=sad, 05=angry, 06=fearful, 07=disgust, 08=surprised
    """
    emotions = defaultdict(list)
    
    if not dataset_path.exists():
        logger.warning(f"RAVDESS dataset not found at {dataset_path}")
        return emotions
    
    emotion_codes = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised"
    }
    
    for audio_file in dataset_path.rglob("*.wav"):
        try:
            filename = audio_file.stem
            parts = filename.split('-')
            
            if len(parts) >= 4:
                emotion_code = parts[3]
                emotion_name = emotion_codes.get(emotion_code)
                
                if emotion_name:
                    # Load and normalize audio
                    waveform, sample_rate = torchaudio.load(str(audio_file))
                    audio_np = waveform.numpy()
                    normalized_audio, _ = normalize_audio(audio_np, sample_rate)
                    
                    # Map to Oviya emotion
                    oviya_emotion = EMOTION_MAPPING.get(emotion_name, emotion_name)
                    emotions[oviya_emotion].append((normalized_audio, TARGET_SAMPLE_RATE))
        except Exception as e:
            logger.warning(f"Failed to process {audio_file}: {e}")
    
    logger.info(f"Extracted {sum(len(v) for v in emotions.values())} samples from RAVDESS")
    return emotions


def extract_crema_d_emotions(dataset_path: Path) -> Dict[str, List[Tuple[np.ndarray, int]]]:
    """
    Extract emotions from CREMA-D dataset
    
    CREMA-D filename format: ActorID_Emotion_Statement_Repetition.wav
    Example: 1001_ANG_HAP_XX.wav
    Emotions: ANG, DIS, FEA, HAP, NEU, SAD
    """
    emotions = defaultdict(list)
    
    if not dataset_path.exists():
        logger.warning(f"CREMA-D dataset not found at {dataset_path}")
        return emotions
    
    for audio_file in dataset_path.rglob("*.wav"):
        try:
            filename = audio_file.stem
            parts = filename.split('_')
            
            if len(parts) >= 2:
                emotion_code = parts[1]  # ANG, DIS, FEA, HAP, NEU, SAD
                oviya_emotion = EMOTION_MAPPING.get(emotion_code, emotion_code.lower())
                
                # Load and normalize audio
                waveform, sample_rate = torchaudio.load(str(audio_file))
                audio_np = waveform.numpy()
                normalized_audio, _ = normalize_audio(audio_np, sample_rate)
                
                emotions[oviya_emotion].append((normalized_audio, TARGET_SAMPLE_RATE))
        except Exception as e:
            logger.warning(f"Failed to process {audio_file}: {e}")
    
    logger.info(f"Extracted {sum(len(v) for v in emotions.values())} samples from CREMA-D")
    return emotions


def extract_meld_emotions(dataset_path: Path) -> Dict[str, List[Tuple[np.ndarray, int]]]:
    """
    Extract emotions from MELD dataset
    
    MELD structure: audio/uttr_[id].wav, metadata in csv
    Emotions: joy, sadness, anger, fear, disgust, surprise, neutral
    """
    emotions = defaultdict(list)
    
    if not dataset_path.exists():
        logger.warning(f"MELD dataset not found at {dataset_path}")
        return emotions
    
    # Try to find metadata CSV
    csv_files = list(dataset_path.rglob("*.csv"))
    emotion_map = {}
    
    if csv_files:
        try:
            import pandas as pd
            df = pd.read_csv(csv_files[0])
            for _, row in df.iterrows():
                uttr_id = row.get('Utterance_ID', '')
                emotion = row.get('Emotion', '')
                if uttr_id and emotion:
                    emotion_map[uttr_id] = emotion
        except Exception as e:
            logger.warning(f"Failed to load MELD metadata: {e}")
    
    # Extract audio files
    for audio_file in dataset_path.rglob("*.wav"):
        try:
            # Extract utterance ID from filename
            filename = audio_file.stem
            uttr_id = filename.replace('uttr_', '').replace('dia_', '')
            
            # Get emotion from metadata or infer from filename
            emotion = emotion_map.get(uttr_id, 'neutral')
            oviya_emotion = EMOTION_MAPPING.get(emotion.lower(), emotion.lower())
            
            # Load and normalize audio
            waveform, sample_rate = torchaudio.load(str(audio_file))
            audio_np = waveform.numpy()
            normalized_audio, _ = normalize_audio(audio_np, sample_rate)
            
            emotions[oviya_emotion].append((normalized_audio, TARGET_SAMPLE_RATE))
        except Exception as e:
            logger.warning(f"Failed to process {audio_file}: {e}")
    
    logger.info(f"Extracted {sum(len(v) for v in emotions.values())} samples from MELD")
    return emotions


def extract_emodb_emotions(dataset_path: Path) -> Dict[str, List[Tuple[np.ndarray, int]]]:
    """
    Extract emotions from EmoDB dataset
    
    EmoDB filename format: [name][sex][emotion][text].wav
    Example: 03a01Fa.wav
    Emotion codes: W=neutral, F=happy, T=sad, N=angry, E=disgust, A=fear, L=bored
    """
    emotions = defaultdict(list)
    
    if not dataset_path.exists():
        logger.warning(f"EmoDB dataset not found at {dataset_path}")
        return emotions
    
    emotion_codes = {
        'W': 'neutral',
        'F': 'happy',
        'T': 'sad',
        'N': 'angry',
        'E': 'disgust',
        'A': 'fear',
        'L': 'bored'
    }
    
    for audio_file in dataset_path.rglob("*.wav"):
        try:
            filename = audio_file.name
            # Extract emotion code (3rd character in filename)
            if len(filename) >= 3:
                emotion_code = filename[2]
                emotion_name = emotion_codes.get(emotion_code)
                
                if emotion_name:
                    oviya_emotion = EMOTION_MAPPING.get(emotion_name, emotion_name)
                    
                    # Load and normalize audio
                    waveform, sample_rate = torchaudio.load(str(audio_file))
                    audio_np = waveform.numpy()
                    normalized_audio, _ = normalize_audio(audio_np, sample_rate)
                    
                    emotions[oviya_emotion].append((normalized_audio, TARGET_SAMPLE_RATE))
        except Exception as e:
            logger.warning(f"Failed to process {audio_file}: {e}")
    
    logger.info(f"Extracted {sum(len(v) for v in emotions.values())} samples from EmoDB")
    return emotions


def save_emotion_references(emotions: Dict[str, List[Tuple[np.ndarray, int]]], output_dir: Path):
    """
    Save emotion reference audio files
    
    Args:
        emotions: Dict mapping emotion names to lists of (audio, sample_rate) tuples
        output_dir: Output directory for emotion references
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    emotion_map = {}
    
    for emotion_name, audio_samples in emotions.items():
        if not audio_samples:
            continue
        
        # Select best sample (longest, most representative)
        # For now, just use first few samples
        best_samples = audio_samples[:5]  # Save up to 5 samples per emotion
        
        emotion_map[emotion_name] = []
        
        for idx, (audio, sample_rate) in enumerate(best_samples):
            # Save audio file
            output_file = output_dir / f"{emotion_name}_{idx}.wav"
            
            try:
                # Convert to tensor and save
                audio_tensor = torch.from_numpy(audio).unsqueeze(0)  # [1, samples]
                torchaudio.save(str(output_file), audio_tensor, sample_rate)
                
                emotion_map[emotion_name].append({
                    'file': str(output_file.name),
                    'duration': len(audio) / sample_rate,
                    'sample_rate': sample_rate
                })
                
                logger.debug(f"Saved {output_file}: {len(audio)/sample_rate:.2f}s")
            except Exception as e:
                logger.error(f"Failed to save {output_file}: {e}")
    
    # Save emotion mapping JSON
    mapping_file = output_dir / "emotion_map.json"
    with open(mapping_file, 'w') as f:
        json.dump(emotion_map, f, indent=2)
    
    logger.info(f"Saved {len(emotion_map)} emotion references to {output_dir}")
    logger.info(f"Emotion mapping saved to {mapping_file}")


def main():
    """Main extraction function"""
    logger.info("🎭 Starting Emotion Audio Extraction")
    logger.info("=" * 60)
    
    # Setup paths - support both local and VastAI deployments
    workspace = Path(os.getenv("WORKSPACE", "/workspace"))
    if not workspace.exists():
        workspace = Path(".")  # Fallback to current directory
    
    datasets_dir = workspace / "emotion_datasets"
    output_dir = workspace / "data" / "emotion_references"
    
    # Also try alternative paths
    if not datasets_dir.exists():
        datasets_dir = Path("emotion_datasets")
    if not output_dir.exists():
        output_dir = Path("data") / "emotion_references"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all emotions from all datasets
    all_emotions = defaultdict(list)
    
    # Extract from each dataset
    logger.info("\n📦 Extracting from RAVDESS...")
    ravdess_emotions = extract_ravdess_emotions(datasets_dir / "ravdess")
    for emotion, samples in ravdess_emotions.items():
        all_emotions[emotion].extend(samples)
    
    logger.info("\n📦 Extracting from CREMA-D...")
    crema_emotions = extract_crema_d_emotions(datasets_dir / "crema_d")
    for emotion, samples in crema_emotions.items():
        all_emotions[emotion].extend(samples)
    
    logger.info("\n📦 Extracting from MELD...")
    meld_emotions = extract_meld_emotions(datasets_dir / "meld")
    for emotion, samples in meld_emotions.items():
        all_emotions[emotion].extend(samples)
    
    logger.info("\n📦 Extracting from EmoDB...")
    emodb_emotions = extract_emodb_emotions(datasets_dir / "emodb")
    for emotion, samples in emodb_emotions.items():
        all_emotions[emotion].extend(samples)
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 Extraction Summary:")
    total_samples = sum(len(samples) for samples in all_emotions.values())
    logger.info(f"   Total samples extracted: {total_samples}")
    logger.info(f"   Unique emotions: {len(all_emotions)}")
    for emotion, samples in sorted(all_emotions.items()):
        logger.info(f"   {emotion}: {len(samples)} samples")
    
    # Save emotion references
    logger.info("\n💾 Saving emotion references...")
    save_emotion_references(all_emotions, output_dir)
    
    logger.info("\n✅ Emotion extraction complete!")
    logger.info(f"   Output directory: {output_dir}")


if __name__ == "__main__":
    main()

