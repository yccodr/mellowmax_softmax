from typing import Union

import torch
import numpy as np
from torch import Tensor
from numpy import ndarray


def boltzmax(x: Union[ndarray, Tensor], beta=1) -> Union[ndarray, Tensor]:
    if isinstance(x, ndarray):
        x_exp = np.exp(beta * x)

        s = np.inner(x, x_exp)
        s /= np.sum(x_exp)
    elif isinstance(x, Tensor):
        x_exp = torch.exp(beta * x)

        s = torch.inner(x, x_exp)
        s /= x_exp.sum()    

    return s
