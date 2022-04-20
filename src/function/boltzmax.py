from plum import dispatch
from numpy.typing import ArrayLike
from torch import Tensor

@dispatch
def boltzmax(x: Tensor) -> Tensor:
    raise NotImplementedError

@dispatch
def boltzmax(x: ArrayLike) -> ArrayLike:
    raise NotImplementedError