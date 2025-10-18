from typing import Optional
import torch


class PersonalityEMA:
    def __init__(self, alpha: float = 0.85, eps: float = 1e-6):
        """
        Configure the exponential moving average parameters and initialize the internal vector state.
        
        Parameters:
        	alpha (float): Smoothing coefficient in [0,1) that controls how much past state is retained; larger values weight past state more.
        	eps (float): Small positive constant used to clamp values for numerical stability.
        
        Notes:
        	Sets the internal vector `_vec` to `None` (uninitialized). 
        """
        self.alpha = alpha
        self.eps = eps
        self._vec: Optional[torch.Tensor] = None  # [5]

    def update(self, new_vec: torch.Tensor) -> torch.Tensor:
        """
        Update the internal exponentially weighted personality vector with a new input vector.
        
        The provided tensor is treated as non-negative weights, stabilized, and normalized; the internal EMA is initialized on the first call and subsequently updated using the configured smoothing factor.
        
        Parameters:
            new_vec (torch.Tensor): Input 5-element tensor of personality scores (will be clamped and normalized).
        
        Returns:
            torch.Tensor: A clone of the updated 5-element internal personality vector normalized to sum to 1.
        """
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
        """
        Get a copy of the current exponential moving average (EMA) vector, or None if it has not been initialized.
        
        Returns:
            torch.Tensor | None: A clone of the internal 5-element EMA tensor when available, or `None` if the EMA has not been set.
        """
        return None if self._vec is None else self._vec.clone()

