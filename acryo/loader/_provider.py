from __future__ import annotations

from typing import Callable
from functools import partial
import numpy as np
from numpy.typing import NDArray
from acryo._types import nm
from acryo._reader import ImageReaderRegistry


class ImageProvider:
    """Function that provides an image at a given scale."""

    def __init__(self, provider: Callable[[nm], NDArray[np.float32]]):
        self._provider = provider

    def __call__(self, scale: nm) -> NDArray[np.float32]:
        return self._provider(scale)


class ImageReader(ImageProvider):
    def __init__(self, path: str):
        super().__init__(partial(ImageReaderRegistry.imread_array, path))
