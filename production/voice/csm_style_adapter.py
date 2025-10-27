from typing import Dict
import torch


def style_film_params(style_vec: torch.Tensor, scale: float = 0.5) -> Dict[str, torch.Tensor]:
    # Simple FiLM: gamma = 1 + scale*style, beta = scale*(style-mean)
    s = style_vec.view(1, -1)
    gamma = 1.0 + scale * s
    beta = scale * (s - s.mean())
    return {"gamma": gamma, "beta": beta}




