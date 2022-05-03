from typing import Union

import torch
import numpy as np
from torch import Tensor
from numpy import ndarray


def boltzmax(x: Union[ndarray, Tensor], beta=1) -> Union[ndarray, Tensor]:
    if isinstance(x, ndarray):
        c = x.max()
        x_exp = np.exp(beta * (x - c))

        s = np.inner(x, x_exp)
        s /= np.sum(x_exp)
    elif isinstance(x, Tensor):
        c = x.max()
        x_exp = torch.exp(beta * (x - c))

        s = torch.inner(x, x_exp)
        s /= x_exp.sum()
    else:
        raise TypeError("x must be either np.ndarray or torch.Tensor")

    return s
