from __future__ import annotations

from typing import Generic, Hashable, Iterator, TypeVar, TYPE_CHECKING
import weakref
import numpy as np
from numpy.typing import NDArray
from dask import array as da

if TYPE_CHECKING:
    from acryo.loader._base import LoaderBase

_K = TypeVar("_K", bound=Hashable)


class LoaderGroup(Generic[_K]):
    """A groupby-like object for subtomogram loaders."""

    def __init__(
        self,
        loader: LoaderBase,
        by: _K,
        order: int,
        scale: float,
        output_shape: tuple[int, ...],
        corner_safe: bool,
    ):
        self._loader = weakref.ref(loader)
        self._by = by
        self._order = order
        self._scale = scale
        self._output_shape = output_shape
        self._corner_safe = corner_safe

    def __iter__(self) -> Iterator[tuple[_K, LoaderBase]]:
        loader = self._loader()
        for key, mole in loader.molecules.groupby(self._by):
            _loader = loader.replace(
                molecules=mole,
                order=self._order,
                scale=self._scale,
                output_shape=self._output_shape,
                corner_safe=self._corner_safe,
            )
            yield key, _loader

    def average(
        self, output_shape: tuple[int, ...] | None = None
    ) -> dict[_K, NDArray[np.float32]]:
        """Calculate average image."""
        if output_shape is None:
            output_shape = self._output_shape
        tasks = []
        keys: list[str] = []
        for key, loader in self:
            keys.append(key)
            dsk = loader.construct_dask(output_shape)
            tasks.append(da.mean(dsk, axis=0))

        out: list[NDArray[np.float32]] = da.compute(tasks)[0]
        return {key: img for key, img in zip(keys, out)}
