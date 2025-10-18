from typing import Dict
import torch
import torch.nn as nn


class EmpathyFusionHead(nn.Module):
    def __init__(self, emotion_dim: int, context_dim: int, memory_dim: int, hidden: int = 128, eps: float = 1e-6):
        """
        Initialize an EmpathyFusionHead that maps concatenated emotion, context, and memory features to a 5-class probability output.
        
        Parameters:
            emotion_dim (int): Dimension of the emotion feature vector.
            context_dim (int): Dimension of the context feature vector.
            memory_dim (int): Dimension of the memory feature vector.
            hidden (int): Hidden layer size for the internal MLP.
            eps (float): Small value used to clamp output probabilities away from zero.
        """
        super().__init__()
        self.linear1 = nn.Linear(emotion_dim + context_dim + memory_dim, hidden)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(hidden, 5)
        self.eps = eps

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        # feats: {'emotion': [B,De], 'context':[B,Dc], 'memory':[B,Dm]}
        """
        Fuse emotion, context, and memory features and produce a 5-class probability distribution.
        
        Concatenates feats['emotion'], feats['context'], and feats['memory'] along the last dimension, projects the result through a two-layer MLP with GELU activation, and returns a probability vector per input. Probabilities are clamped to be at least `self.eps` and renormalized so the last dimension sums to 1.
        
        Parameters:
            feats (Dict[str, torch.Tensor]): Dictionary containing:
                - 'emotion': tensor of shape [B, De]
                - 'context': tensor of shape [B, Dc]
                - 'memory': tensor of shape [B, Dm]
            All tensors must be compatible for concatenation along the last dimension.
        
        Returns:
            torch.Tensor: A tensor of shape [B, 5] containing per-sample probabilities; values are >= `self.eps` and sum to 1 along the last dimension.
        """
        x = torch.cat([feats['emotion'], feats['context'], feats['memory']], dim=-1)
        h = self.act(self.linear1(x))
        p = torch.softmax(self.linear2(h), dim=-1)
        # clamp then renormalize
        p = torch.clamp(p, min=self.eps)
        p = p / p.sum(dim=-1, keepdim=True)
        return p

