from typing import Union

import numpy as np
import torch

class EpsGreedy():
    def __init__(self, eps=0.1) -> None:
        self.eps = eps

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if np.random.rand() < self.eps:
            return np.ones(len(x)) / len(x)
        else:
            ind = np.argmax(x)
            tmp = np.zeros(len(x))
            tmp[ind] = 1
            return tmp
