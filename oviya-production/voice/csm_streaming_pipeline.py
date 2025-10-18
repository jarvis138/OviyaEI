from typing import Dict, Generator, Optional
import numpy as np


class CSMStreamingPipeline:
    def __init__(self, mimi_model: Optional[object] = None, csm_decoder: Optional[object] = None):
        # Expect initialized Mimi and CSM decoder passed in
        """
        Initialize the streaming pipeline with optional Mimi model and CSM decoder.
        
        Parameters:
            mimi_model (Optional[object]): An initialized Mimi model instance used to decode acoustic tokens into PCM. May be None to enable fallback behavior.
            csm_decoder (Optional[object]): An initialized CSM decoder instance that generates/refines acoustic tokens. May be None to enable fallback behavior.
        
        Notes:
            The constructor stores the provided objects on the instance without validation.
        """
        self.mimi = mimi_model
        self.decoder = csm_decoder

    def generate_stream(self, semantic_tokens: np.ndarray, style_vec: np.ndarray, fast_start: bool = True) -> Generator[np.ndarray, None, None]:
        """
        Stream decoded PCM audio from semantic tokens and a style vector using the Mimi model and CSM decoder.
        
        This generator yields consecutive PCM chunks (numpy.float32 arrays) representing decoded audio. If the pipeline components are unavailable or decoding fails, yields a short sequence of safe placeholder tones instead.
        
        Parameters:
            semantic_tokens (np.ndarray): Interleaved semantic token sequence used as decoder input.
            style_vec (np.ndarray): Length-5 style conditioning vector.
            fast_start (bool): If True, request fewer initial codebooks to reduce time-to-first-byte, with optional later refinement to full quality.
        
        Returns:
            Generator[np.ndarray, None, None]: A generator that yields PCM chunks (float32 numpy arrays, produced at a 24 kHz sample rate).
        """
        if self.mimi is None or self.decoder is None:
            # Fallback to tone beep (dev)
            sr = 24000
            t = np.linspace(0, 0.04, int(sr * 0.04), endpoint=False)
            for i in range(50):
                chunk = (0.02 * np.sin(2 * np.pi * 220 * (i+1) * t)).astype(np.float32)
                yield chunk
            return

        # Fast-start: decode initial 16 codebooks for quick TTFB, then refine to full 31
        initial_codebooks = 16 if fast_start else 31

        try:
            # Style conditioning: adapt 5D vector to decoder FiLM params if supported
            film_params = None
            try:
                import torch
                from .csm_style_adapter import style_film_params
                s = torch.tensor(style_vec, dtype=torch.float32)
                film_params = style_film_params(s)
            except Exception:
                film_params = None

            # Phase 1: generate with fewer codebooks for low TTFB
            if hasattr(self.decoder, 'generate') and hasattr(self.mimi, 'decode'):
                for acoustic_tokens in self.decoder.generate(
                    semantic_tokens,
                    style_vector=style_vec.tolist() if hasattr(style_vec, 'tolist') else list(style_vec),
                    codebooks=initial_codebooks,
                    film_params=film_params
                ):
                    pcm = self.mimi.decode(acoustic_tokens)
                    yield pcm

                # Phase 2: refine to full quality
                if initial_codebooks < 31 and hasattr(self.decoder, 'refine'):
                    for acoustic_tokens in self.decoder.refine(
                        codebooks=31,
                        film_params=film_params
                    ):
                        pcm = self.mimi.decode(acoustic_tokens)
                        yield pcm
                return
        except Exception:
            pass

        # Placeholder safe fallback if decoder API not available:
        sr = 24000
        t = np.linspace(0, 0.04, int(sr * 0.04), endpoint=False)
        for i in range(50):
            chunk = (0.02 * np.sin(2 * np.pi * 330 * (i+1) * t)).astype(np.float32)
            yield chunk

