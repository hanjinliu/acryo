from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray

# scipy is not well typed. Make patches here.
if TYPE_CHECKING:

    def fftn(img: NDArray[np.float32] | NDArray[np.complex64]) -> NDArray[np.complex64]:
        ...

    def ifftn(img: NDArray[np.complex64]) -> NDArray[np.complex64]:
        ...

    def rfftn(img: NDArray[np.float32]) -> NDArray[np.complex64]:
        ...

    def irfftn(img: NDArray[np.complex64]) -> NDArray[np.float32]:
        ...

else:
    pass
