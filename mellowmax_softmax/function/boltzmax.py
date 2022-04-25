from plum import dispatch
import torch
import numpy as np


@dispatch
def boltzmax(x: torch.Tensor, beta=1) -> torch.Tensor:
    x_exp = torch.exp(beta * x)

    s = torch.inner(x, x_exp)
    s /= x_exp.sum()

    return s


@dispatch
def boltzmax(x: np.ndarray, beta=1) -> np.ndarray:
    x_exp = np.exp(beta * x)

    s = np.inner(x, x_exp)
    s /= np.sum(x_exp)

    return s
