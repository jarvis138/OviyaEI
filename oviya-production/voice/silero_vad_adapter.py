import numpy as np
import torch


def _load_silero(device: str = "cpu"):
    """
    Load Silero VAD from torch.hub.
    Returns (model, get_speech_timestamps).
    """
    model, utils = torch.hub.load(
        repo_or_dir='snakers4/silero-vad',
        model='silero_vad',
        force_reload=False
    )
    (get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils
    model.to(device).eval()
    return model, get_speech_timestamps


class SileroVAD:
    def __init__(self, device: str = "cpu", threshold: float = 0.5, min_speech_seconds: float = 0.2):
        self.device = device
        self.threshold = threshold
        self.min_speech_seconds = min_speech_seconds
        self.model, self.get_speech_timestamps = _load_silero(device=device)

    def is_speech(self, audio_np: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Return True if the chunk contains voice according to Silero.
        audio_np: mono PCM, float32 [-1,1] or int16
        """
        if audio_np.dtype == np.int16:
            audio_np = (audio_np.astype(np.float32) / 32768.0).copy()
        elif audio_np.dtype != np.float32:
            audio_np = audio_np.astype(np.float32).copy()
        min_len = int(self.min_speech_seconds * sample_rate)
        if audio_np.shape[0] < min_len:
            audio_np = np.pad(audio_np, (0, min_len - audio_np.shape[0]))
        ts = self.get_speech_timestamps(audio_np, self.model, sampling_rate=sample_rate)
        return len(ts) > 0


