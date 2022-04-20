from plum import dispatch
from torch import Tensor
from numpy.typing import ArrayLike

@dispatch
def mellowmax(x: Tensor) -> Tensor:
    raise NotImplementedError

@dispatch
def mellowmax(x: Tensor) -> ArrayLike:
    raise NotImplementedError
