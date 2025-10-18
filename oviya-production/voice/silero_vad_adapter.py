import numpy as np
import torch


def _load_silero(device: str = "cpu"):
    """
    Load the Silero VAD model and its speech-timestamp utility via torch.hub and move the model to the specified device for inference.
    
    Parameters:
        device (str): Target device for the model (e.g., "cpu" or "cuda").
    
    Returns:
        tuple: (model, get_speech_timestamps) where `model` is the loaded Silero VAD PyTorch module and `get_speech_timestamps` is a callable that extracts speech segments from an audio array.
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
        """
        Initialize the SileroVAD instance and load the Silero VAD model and its speech-timestamp helper.
        
        Parameters:
            device (str): PyTorch device identifier where the model will be placed (e.g., "cpu" or "cuda").
            threshold (float): Detection score threshold used to decide whether audio contains speech.
            min_speech_seconds (float): Minimum duration in seconds to consider when padding short audio chunks before detection.
        """
        self.device = device
        self.threshold = threshold
        self.min_speech_seconds = min_speech_seconds
        self.model, self.get_speech_timestamps = _load_silero(device=device)

    def is_speech(self, audio_np: np.ndarray, sample_rate: int = 16000) -> bool:
        """
        Determine whether the given mono PCM audio chunk contains speech.
        
        Parameters:
            audio_np (np.ndarray): Mono PCM audio. May be float32 samples in [-1, 1] or int16 PCM; inputs shorter than the instance's minimum speech duration are zero-padded.
            sample_rate (int): Sampling rate of audio_np in Hz.
        
        Returns:
            bool: True if Silero VAD detects any speech segments, False otherwise.
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

