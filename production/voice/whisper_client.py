import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
from typing import Optional, AsyncGenerator

# Import WhisperX configuration
try:
    from ..config.whisperx_config import WHISPERX_CONFIG
except ImportError:
    WHISPERX_CONFIG = {
        "batch_size": 8,
        "language": "en",
        "compute_type": "float16"
    }


class WhisperTurboClient:
    def __init__(self, device: str = "auto", model_id: str = "openai/whisper-large-v3-turbo", chunk_len_s: int = 30, batch_size: int = 8):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model_id = model_id

        # Use configuration values with fallbacks
        self.batch_size = batch_size or WHISPERX_CONFIG.get("batch_size", 8)
        self.language = WHISPERX_CONFIG.get("language", "en")
        self.compute_type = WHISPERX_CONFIG.get("compute_type", "float16")

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="flash_attention_2"
        ).to(self.device)

        # Optimizations
        try:
            self.model.generation_config.cache_implementation = "static"
            self.model.generation_config.max_new_tokens = 448
            self.model.forward = torch.compile(self.model.forward, mode="reduce-overhead", fullgraph=True)
        except Exception:
            pass

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            chunk_length_s=chunk_len_s,
            batch_size=batch_size
        )

        self._warmup()

    def _warmup(self):
        try:
            dummy_audio = np.random.randn(16000 * 2).astype(np.float32)
            for _ in range(2):
                _ = self.pipe(
                    dummy_audio,
                    generate_kwargs={
                        "min_new_tokens": 64,
                        "max_new_tokens": 64,
                        "language": "english",
                        "task": "transcribe"
                    }
                )
        except Exception:
            pass

    async def transcribe_audio(self, audio_data: np.ndarray, language: str = "english") -> dict:
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
        if np.max(np.abs(audio_data)) > 0:
            audio_data = audio_data / np.max(np.abs(audio_data)) * 0.9
        result = self.pipe(
            audio_data,
            generate_kwargs={
                "language": language,
                "task": "transcribe",
                "condition_on_prev_tokens": True,
                "compression_ratio_threshold": 1.35,
                "temperature": (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                "logprob_threshold": -1.0,
                "no_speech_threshold": 0.6,
            }
        )
        return {"text": result.get("text", "").strip(), "confidence": 0.9}

    async def transcribe_streaming(self, audio_chunks: AsyncGenerator[np.ndarray, None], language: str = "english"):
        audio_buffer = []
        buffer_duration = 0.0
        sample_rate = 16000
        async for chunk in audio_chunks:
            audio_buffer.extend(chunk)
            buffer_duration += len(chunk) / sample_rate
            if buffer_duration >= 2.0:
                audio_segment = np.array(audio_buffer, dtype=np.float32)
                result = await self.transcribe_audio(audio_segment, language)
                if result["text"]:
                    yield result
                overlap_samples = int(0.5 * sample_rate)
                audio_buffer = audio_buffer[-overlap_samples:]
                buffer_duration = len(audio_buffer) / sample_rate
        if audio_buffer:
            audio_segment = np.array(audio_buffer, dtype=np.float32)
            result = await self.transcribe_audio(audio_segment, language)
            if result["text"]:
                yield result


