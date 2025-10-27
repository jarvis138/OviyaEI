from typing import Dict, Generator, Optional
import numpy as np


class CSMStreamingPipeline:
    def __init__(self, mimi_model: Optional[object] = None, csm_decoder: Optional[object] = None):
        # Expect initialized Mimi and CSM decoder passed in
        self.mimi = mimi_model
        self.decoder = csm_decoder

    def generate_stream(self, semantic_tokens: np.ndarray, style_vec: np.ndarray, fast_start: bool = True) -> Generator[np.ndarray, None, None]:
        """
        Stream decoded PCM from semantic + acoustic tokens using Mimi + CSM.
        semantic_tokens: np.ndarray (interleaved text semantics or prepared semantic sequence)
        style_vec: np.ndarray length 5
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


