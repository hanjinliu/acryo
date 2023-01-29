from __future__ import annotations

from typing import TYPE_CHECKING, Hashable, Iterator
import numpy as np
import pandas as pd

from dask import array as da
from dask.array.core import Array as daskArray

from .loader import SubtomogramLoader, Unset, check_input
from .molecules import Molecules
from ._types import nm, pixel

if TYPE_CHECKING:
    from typing_extensions import Self

IMAGE_ID_LABEL = "image"


class TomogramCollection:
    def __init__(
        self,
        order: int = 3,
        scale: nm = 1.0,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset = Unset(),
        corner_safe: bool = False,
    ) -> None:

        self._order, self._output_shape, self._scale, self._corner_safe = check_input(
            order, output_shape, scale, corner_safe, 3
        )
        self._images: dict[Hashable, np.ndarray | daskArray] = {}
        self._molecules: Molecules = Molecules.empty([IMAGE_ID_LABEL])

    def add_tomogram(
        self,
        image: np.ndarray | daskArray,
        molecules: Molecules,
    ) -> Self:
        idx = len(self._images)
        while idx in self._images:
            idx += 1
        molecules = molecules.copy()
        molecules.features[IMAGE_ID_LABEL] = idx
        _molecules_new = self._molecules.concat_with(molecules)

        self._images[idx] = image
        self._molecules = _molecules_new
        return self

    @property
    def molecules(self) -> Molecules:
        """Collect all the molecules from all the loaders"""
        return self._molecules

    @property
    def features(self) -> pd.DataFrame:
        """Collect all the features from all the molecules"""
        return self.molecules.features

    @property
    def scale(self) -> nm:
        """Get the physical scale of tomogram."""
        return self._scale

    @property
    def output_shape(self) -> tuple[pixel, ...] | Unset:
        """Return the output subtomogram shape."""
        return self._output_shape

    @property
    def order(self) -> int:
        """Return the interpolation order."""
        return self._order

    @property
    def corner_safe(self) -> bool:
        return self._corner_safe

    def replace(
        self,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset | None = None,
        order: int | None = None,
        scale: float | None = None,
        corner_safe: bool | None = None,
    ) -> Self:
        """Return a new instance with different parameter(s)."""
        if output_shape is None:
            output_shape = self.output_shape
        if order is None:
            order = self.order
        if scale is None:
            scale = self.scale
        if corner_safe is None:
            corner_safe = self.corner_safe
        out = self.__class__(
            output_shape=output_shape,
            order=order,
            scale=scale,
            corner_safe=corner_safe,
        )
        out._images = self._images.copy()
        return out

    def iter_loader(self) -> Iterator[SubtomogramLoader]:
        for key, group in self.molecules.groupby(IMAGE_ID_LABEL):
            image = self._images[key]
            loader = SubtomogramLoader(
                image,
                group,
                self.order,
                self.scale,
                self.output_shape,
                self.corner_safe,
            )
            yield loader

    def average(
        self,
        output_shape: pixel | tuple[pixel] | None = None,
    ) -> np.ndarray:
        """
        Calculate the average of subtomograms from all the loaders.

        Returns
        -------
        np.ndarray
            Averaged image
        """
        dask_arrays: list[daskArray] = []
        for loader in self.iter_loader():
            dask_arrays.append(loader.construct_dask(output_shape=output_shape))
        dask_array = da.stack(dask_arrays, axis=0)  # type: ignore
        return da.compute(da.mean(dask_array, axis=0))[0]  # type: ignore

    # def filter(self, predicate):
    #     if callable(predicate):
    #         features = predicate(self.features)
    #     elif isinstance(predicate, str):
    #         features = self.features.eval()
    #     features = self.features[predicate]
    #     mole = Molecules.from_dataframe(features)
    #     return self.__class__()
