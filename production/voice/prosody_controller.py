from typing import Dict, Tuple
import torch
import torch.nn as nn


class ProsodyController(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns [B,3] ≈ [F0_scale, energy_scale, duration_scale]
        return self.net(x)

def prosody_inference(model: ProsodyController, emotion_embed: torch.Tensor, p_vec: torch.Tensor) -> Dict[str, float]:
    x = torch.cat([emotion_embed, p_vec], dim=-1)
    out = model(x)
    f0_s, e_s, d_s = out[0].tolist()
    return {"f0_scale": float(f0_s), "energy_scale": float(e_s), "duration_scale": float(d_s)}




