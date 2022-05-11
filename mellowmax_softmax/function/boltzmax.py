from typing import Union

import numpy as np
import torch


class Boltzmax():

    def __init__(self, beta=1) -> None:
        self.beta = beta

    def __call__(
        self,
        x: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:

        if isinstance(x, np.ndarray):
            x_exp = np.exp(self.beta * (x - x.max()))
            return np.inner(x, x_exp) / np.sum(x_exp)

        if isinstance(x, torch.Tensor):
            x_exp = torch.exp(self.beta * (x - x.max()))
            return torch.inner(x, x_exp) / torch.sum(x_exp)

        raise TypeError('x must be either np.ndarray or torch.Tensor.')


class BoltzmannPolicy():

    def __init__(self, beta=1) -> None:
        self.beta = beta

    def __call__(
        self,
        x: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:

        if isinstance(x, np.ndarray):
            x_exp = np.exp(self.beta * (x - x.max()))
            return x_exp / np.sum(x_exp)

        if isinstance(x, torch.Tensor):
            x_exp = torch.exp(self.beta * (x - x.max()))
            return x_exp / torch.sum(x_exp)

        raise TypeError('x must be either np.ndarray or torch.Tensor.')
