from typing import Optional
import torch


class PersonalityEMA:
    def __init__(self, alpha: float = 0.85, eps: float = 1e-6):
        self.alpha = alpha
        self.eps = eps
        self._vec: Optional[torch.Tensor] = None  # [5]

    def update(self, new_vec: torch.Tensor) -> torch.Tensor:
        v = new_vec.detach()
        v = torch.clamp(v, min=self.eps)
        v = v / v.sum()
        if self._vec is None:
            self._vec = v
        else:
            self._vec = self.alpha * self._vec + (1.0 - self.alpha) * v
            self._vec = torch.clamp(self._vec, min=self.eps)
            self._vec = self._vec / self._vec.sum()
        return self._vec.clone()

    def current(self) -> Optional[torch.Tensor]:
        return None if self._vec is None else self._vec.clone()


