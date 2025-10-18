from typing import Dict, Tuple
import torch
import torch.nn as nn


class ProsodyController(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128):
        """
        Initialize the ProsodyController with a three-output feedforward network that predicts prosodic scales.
        
        Parameters:
            in_dim (int): Dimensionality of the input feature vector (e.g., concatenated emotion embedding and prosody vector).
            hidden (int): Number of units in the hidden layers (default 128). The network architecture is Linear(in_dim -> hidden) -> GELU -> Linear(hidden -> hidden) -> GELU -> Linear(hidden -> 3), producing a [B, 3] tensor of [f0_scale, energy_scale, duration_scale].
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 3)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns [B,3] ≈ [F0_scale, energy_scale, duration_scale]
        """
        Compute prosody scale predictions from input feature vectors.
        
        Parameters:
            x (torch.Tensor): Input tensor with batch dimension first; features per example along the last dimension.
        
        Returns:
            torch.Tensor: Tensor of shape [B, 3] containing predicted prosody scales per batch entry:
                [f0_scale, energy_scale, duration_scale].
        """
        return self.net(x)

def prosody_inference(model: ProsodyController, emotion_embed: torch.Tensor, p_vec: torch.Tensor) -> Dict[str, float]:
    """
    Compute prosody scales from an emotion embedding and a prosody vector using a ProsodyController.
    
    Parameters:
        model (ProsodyController): The prosody controller network that maps concatenated features to three scale values.
        emotion_embed (torch.Tensor): Emotion embedding tensor; will be concatenated with `p_vec` along the last dimension.
        p_vec (torch.Tensor): Prosody feature tensor to concatenate with `emotion_embed`.
    
    Returns:
        dict: Mapping with keys `"f0_scale"`, `"energy_scale"`, and `"duration_scale"` containing the corresponding scale values as Python floats.
    """
    x = torch.cat([emotion_embed, p_vec], dim=-1)
    out = model(x)
    f0_s, e_s, d_s = out[0].tolist()
    return {"f0_scale": float(f0_s), "energy_scale": float(e_s), "duration_scale": float(d_s)}

