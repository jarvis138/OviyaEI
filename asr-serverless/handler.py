#!/usr/bin/env python3
"""
RunPod Serverless ASR Handler
Silero VAD + Whisper ASR Pipeline
"""
import runpod
import torch
import torchaudio
import numpy as np
import base64
import io
import time
import logging
from typing import Dict, List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instances
vad_model = None
whisper_model = None
vad_utils = None

def load_models():
    """Load Silero VAD and Whisper models"""
    global vad_model, whisper_model, vad_utils
    
    try:
        logger.info("Loading Silero VAD model...")
        vad_model, vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=True
        )
        
        logger.info("Loading Whisper model...")
        from faster_whisper import WhisperModel
        whisper_model = WhisperModel(
            "small.en",
            device="cuda",
            compute_type="int8",
            num_workers=2
        )
        
        logger.info("Models loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        return False

def process_audio_chunk(audio_bytes: bytes, sample_rate: int = 16000) -> Dict:
    """Process audio chunk with VAD"""
    try:
        # Convert bytes to float tensor
        audio_float = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Run VAD
        speech_prob = vad_model(torch.from_numpy(audio_float), sample_rate).item()
        
        return {
            'is_speech': speech_prob >= 0.5,
            'speech_prob': speech_prob,
            'timestamp': time.time()
        }
        
    except Exception as e:
        logger.error(f"VAD processing error: {e}")
        return {
            'is_speech': False,
            'speech_prob': 0.0,
            'timestamp': time.time()
        }

def transcribe_audio(audio_bytes: bytes, sample_rate: int = 16000) -> Dict:
    """Transcribe audio with Whisper"""
    try:
        # Convert bytes to numpy array
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        # Transcribe with Whisper
        segments, info = whisper_model.transcribe(
            audio_array,
            beam_size=1,
            language="en",
            vad_filter=False,  # Already filtered by VAD
            without_timestamps=True,
            condition_on_previous_text=False
        )
        
        # Process segments
        text_parts = []
        all_segments = []
        total_confidence = 0
        
        for segment in segments:
            text_parts.append(segment.text.strip())
            all_segments.append({
                'text': segment.text.strip(),
                'start': getattr(segment, 'start', 0),
                'end': getattr(segment, 'end', 0),
                'confidence': getattr(segment, 'avg_logprob', 0)
            })
            total_confidence += getattr(segment, 'avg_logprob', 0)
        
        # Combine text
        full_text = " ".join(text_parts).strip()
        avg_confidence = total_confidence / len(all_segments) if all_segments else 0
        
        return {
            'text': full_text,
            'confidence': avg_confidence,
            'segments': all_segments,
            'language': info.language if hasattr(info, 'language') else 'en',
            'language_probability': getattr(info, 'language_probability', 1.0)
        }
        
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return {
            'text': '',
            'confidence': 0.0,
            'segments': [],
            'language': 'en',
            'language_probability': 0.0
        }

def handler(event):
    """Main RunPod serverless handler"""
    try:
        logger.info(f"Processing ASR request: {event}")
        
        # Extract input
        input_data = event.get("input", {})
        audio_base64 = input_data.get("audio", "")
        sample_rate = input_data.get("sample_rate", 16000)
        operation = input_data.get("operation", "transcribe")  # "vad", "transcribe", "both"
        
        if not audio_base64:
            return {"error": "No audio data provided"}
        
        # Decode audio
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception as e:
            return {"error": f"Failed to decode audio: {e}"}
        
        result = {
            "operation": operation,
            "timestamp": time.time(),
            "audio_size_bytes": len(audio_bytes)
        }
        
        # Process based on operation
        if operation == "vad":
            # Voice Activity Detection only
            vad_result = process_audio_chunk(audio_bytes, sample_rate)
            result.update(vad_result)
            
        elif operation == "transcribe":
            # Transcription only
            transcription_result = transcribe_audio(audio_bytes, sample_rate)
            result.update(transcription_result)
            
        elif operation == "both":
            # Both VAD and transcription
            vad_result = process_audio_chunk(audio_bytes, sample_rate)
            transcription_result = transcribe_audio(audio_bytes, sample_rate)
            
            result.update(vad_result)
            result.update(transcription_result)
            
        else:
            return {"error": f"Unknown operation: {operation}"}
        
        logger.info(f"ASR processing completed: {result.get('text', 'VAD only')[:50]}...")
        return result
        
    except Exception as e:
        logger.error(f"Handler error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": str(e), "traceback": traceback.format_exc()}

# Initialize models on startup
if __name__ == "__main__":
    logger.info("Starting ASR serverless handler...")
    
    # Load models
    if load_models():
        logger.info("ASR handler ready!")
        runpod.serverless.start({"handler": handler})
    else:
        logger.error("Failed to load models, exiting...")
        exit(1)


