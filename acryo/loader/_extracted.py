# pyright: reportPrivateImportUsage=false
from __future__ import annotations
from typing import TYPE_CHECKING
from dask import array as da
import numpy as np
from acryo.backend._api import Backend
from acryo.loader._base import LoaderBase, Unset
from acryo._types import nm, pixel
from acryo.molecules import Molecules
from acryo.tilt import TiltSeriesModel, NoWedge
from acryo._dask import DaskArrayList

if TYPE_CHECKING:
    from typing_extensions import Self

SUBVOLUME_INDEX = "subvolume_index"


class ExtractedSubvolumeLoader(LoaderBase):
    """A class for loading extracted subvolumes directly from files."""

    def __init__(
        self,
        dask_array: da.Array,
        molecules: Molecules,
        order: int = 3,
        scale: nm = 1.0,
        tilt_model: TiltSeriesModel | None = None,
    ) -> None:
        if dask_array.ndim != 4:
            raise ValueError(f"Expected a 4D dask array, got {dask_array.ndim}D.")
        if SUBVOLUME_INDEX not in molecules.features.columns:
            raise ValueError(f"Molecules object must have {SUBVOLUME_INDEX!r} feature.")
        output_shape = dask_array.shape[1:]
        super().__init__(order=order, scale=scale, output_shape=output_shape)
        self._dask_array = dask_array
        self._molecules = molecules
        self._tilt_model = tilt_model or NoWedge()

    @property
    def molecules(self) -> Molecules:
        """All the molecules"""
        return self._molecules

    @property
    def tilt_model(self) -> TiltSeriesModel:
        """Return the tilt model of the subtomogram loader."""
        return self._tilt_model

    def construct_loading_tasks(
        self,
        output_shape=None,
        backend: Backend | None = None,
    ) -> DaskArrayList[np.number]:
        if output_shape is not None and output_shape != self.output_shape:
            raise ValueError("Cannot change output_shape for this class.")
        return DaskArrayList(list(self._dask_array))

    def construct_dask(
        self,
        output_shape: pixel | tuple[pixel, ...] | None = None,
        backend: Backend | None = None,
    ) -> da.Array:
        return self._dask_array

    def replace(
        self,
        molecules: Molecules | None = None,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset | None = None,
        order: int | None = None,
        scale: float | None = None,
    ) -> Self:
        """Return a new loader with updated attributes."""
        dsk = self._dask_array
        if molecules is None:
            molecules = self.molecules
        else:
            indices = molecules.features[SUBVOLUME_INDEX].to_list()
            dsk = self._dask_array[indices]
        if output_shape is None and output_shape != self.output_shape:
            raise ValueError("Cannot change output_shape for this class.")
        if order is None:
            order = self.order
        if scale is None:
            scale = self.scale
        return self.__class__(
            dsk,
            molecules=molecules,
            order=order,
            scale=scale,
            tilt_model=self.tilt_model,
        )

    def _default_align_kwargs(self):
        return {"tilt": self.tilt_model}
