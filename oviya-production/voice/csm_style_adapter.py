from typing import Dict
import torch


def style_film_params(style_vec: torch.Tensor, scale: float = 0.5) -> Dict[str, torch.Tensor]:
    # Simple FiLM: gamma = 1 + scale*style, beta = scale*(style-mean)
    """
    Compute FiLM modulation parameters (gamma and beta) from a style vector.
    
    Parameters:
        style_vec (torch.Tensor): Input style vector (1D or broadcastable); reshaped to (1, N) before computation.
        scale (float): Scaling factor applied to the style vector to control modulation magnitude.
    
    Returns:
        Dict[str, torch.Tensor]: A dictionary with:
            - "gamma": tensor shaped (1, N) equal to 1.0 + scale * s
            - "beta": tensor shaped (1, N) equal to scale * (s - s.mean())
    """
    s = style_vec.view(1, -1)
    gamma = 1.0 + scale * s
    beta = scale * (s - s.mean())
    return {"gamma": gamma, "beta": beta}

