from typing import Union

import numpy as np
import torch


class EpsGreedy():

    def __init__(
        self,
        eps: float = 0.1,
        rng: np.random.Generator = np.random.default_rng(10)
    ) -> None:
        self.eps = eps
        self.rng = rng

    def __call__(
        self,
        x: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:

        if isinstance(x, np.ndarray):
            if self.rng.random() < self.eps:
                return np.full_like(x, 1 / np.size(x))
            return (x == x.max()).astype(np.float32)

        if isinstance(x, torch.Tensor):
            if self.rng.random() < self.eps:
                return torch.full_like(x, 1 / torch.numel(x))
            return (x == x.max()).float()

        raise TypeError('x must be either np.ndarray or torch.Tensor')
