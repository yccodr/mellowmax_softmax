from plum import dispatch
from numpy import ndarray
from torch import Tensor


@dispatch
def boltzmax(x: Tensor) -> Tensor:
    raise NotImplementedError


@dispatch
def boltzmax(x: ndarray) -> ndarray:
    raise NotImplementedError
