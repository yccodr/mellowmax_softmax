from plum import dispatch
from torch import Tensor
from numpy import ndarray


@dispatch
def mellowmax(x: Tensor) -> Tensor:
    raise NotImplementedError


@dispatch
def mellowmax(x: ndarray) -> ndarray:
    raise NotImplementedError
