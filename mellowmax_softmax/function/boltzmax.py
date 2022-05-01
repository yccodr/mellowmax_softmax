from typing import Union

import numpy as np
import torch
from numpy import ndarray
from torch import Tensor


def boltzmax(x: Union[ndarray, Tensor], beta=1) -> Union[ndarray, Tensor]:
    if isinstance(x, ndarray):
        x_exp = np.exp(beta * x)

        s = np.inner(x, x_exp)
        s /= np.sum(x_exp)
    elif isinstance(x, Tensor):
        x_exp = torch.exp(beta * x)

        s = torch.inner(x, x_exp)
        s /= x_exp.sum()
    else:
        raise TypeError("x must be either np.ndarray or torch.Tensor")

    return s
