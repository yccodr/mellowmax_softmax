from plum import dispatch
import torch
from torch import Tensor
import numpy as np


@dispatch
def mellowmax(x: Tensor, omega=1) -> Tensor:
    ## Shifter
    c = x.max()
    x = x - c

    x = x * omega
    x = torch.exp(x)

    s = torch.sum(x)
    s /= torch.numel(x)
    s = torch.log(s)

    return s / omega + c


@dispatch
def mellowmax(x: np.ndarray, omega=1) -> np.ndarray:
    ## Shifter
    c = x.max()
    x = x - c

    x = x * omega
    x = np.exp(x)

    s = np.sum(x, keepdims=True)
    s /= np.size(x)
    s = np.log(s)

    return s / omega + c
