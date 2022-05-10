from typing import Union

import numpy as np
import torch


class Mellowmax():
    def __init__(self, omega=1) -> None:
        self.omega = omega

    def __call__(self, x: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(x, np.ndarray):
            ## Shifter
            c = x.max()
            x = x - c

            x = x * self.omega
            x = np.exp(x)

            s = np.sum(x, keepdims=True)
            s /= np.size(x)
            s = np.log(s)

        elif isinstance(x, torch.Tensor):
            ## Shifter
            c = x.max()
            x = x - c

            x = x * self.omega
            x = torch.exp(x)

            s = torch.sum(x)
            s /= torch.numel(x)
            s = torch.log(s)

        else:
            raise TypeError("x must be either np.ndarray or torch.Tensor")

        return s / self.omega + c
