from typing import Dict
import torch
import torch.nn as nn


class EmpathyFusionHead(nn.Module):
    def __init__(self, emotion_dim: int, context_dim: int, memory_dim: int, hidden: int = 128, eps: float = 1e-6):
        super().__init__()
        self.linear1 = nn.Linear(emotion_dim + context_dim + memory_dim, hidden)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(hidden, 5)
        self.eps = eps

    def forward(self, feats: Dict[str, torch.Tensor]) -> torch.Tensor:
        # feats: {'emotion': [B,De], 'context':[B,Dc], 'memory':[B,Dm]}
        x = torch.cat([feats['emotion'], feats['context'], feats['memory']], dim=-1)
        h = self.act(self.linear1(x))
        p = torch.softmax(self.linear2(h), dim=-1)
        # clamp then renormalize
        p = torch.clamp(p, min=self.eps)
        p = p / p.sum(dim=-1, keepdim=True)
        return p




