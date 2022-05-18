from typing import Union

import numpy as np
import torch
from scipy import optimize


class Mellowmax():

    def __init__(self, omega=1) -> None:
        self.omega = omega

    def __call__(
            self, x: Union[np.ndarray,
                           torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
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


class MellowmaxPolicy():

    def __init__(self, omega=1) -> None:
        self.omega = omega
        self.mm = Mellowmax(self.omega)

    def __call__(
            self, x: Union[np.ndarray,
                           torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        beta = self.compute_beta(x)

        if isinstance(x, np.ndarray):
            c = x.max()
            x_exp = np.exp(beta * (x - c))
            s = x_exp / np.sum(x_exp)

        elif isinstance(x, torch.Tensor):
            c = x.max()
            x_exp = torch.exp(beta * (x - c))
            s = x_exp / torch.sum(x_exp)

        else:
            raise TypeError("x must be either np.ndarray or torch.Tensor")

        return s

    def compute_beta(self, x):
        tmp = x - self.mm(x)

        if isinstance(x, torch.Tensor):
            tmp = tmp.detach().numpy()

        def f(beta):
            return np.exp(beta * tmp - np.max(beta * tmp)) @ tmp.T

        return optimize.brentq(f, -10, 10)
