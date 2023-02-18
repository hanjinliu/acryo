# pyright: reportPrivateImportUsage=false

from __future__ import annotations
from types import MappingProxyType

import weakref
from typing import Hashable, Iterator, TYPE_CHECKING
import numpy as np
import polars as pl

from dask import array as da

from acryo.molecules import Molecules
from acryo._types import nm, pixel
from acryo.loader._base import LoaderBase
from acryo.loader._loader import SubtomogramLoader, Unset

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing_extensions import Self
    from acryo.loader._base import _ShapeType

IMAGE_ID_LABEL = "image"


class TomogramCollection(LoaderBase):
    """
    Collection of tomograms and their molecules.

    A `TomogramCollection` is similar to a list of `SubtomogramLoader` objects, but with
    better consistency in loader parameters and more convenient access to the molecules.
    This class is useful for processing many tomograms at once, especially when you want
    to get the average of all the molecules available.
    """

    def __init__(
        self,
        order: int = 3,
        scale: nm = 1.0,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset = Unset(),
        corner_safe: bool = False,
    ) -> None:

        super().__init__(order, scale, output_shape, corner_safe)
        self._images: dict[Hashable, NDArray[np.float32] | da.Array] = {}
        self._molecules: Molecules = Molecules.empty([IMAGE_ID_LABEL])

    @property
    def loaders(self) -> LoaderAccessor:
        """Interface to access the subtomogram loaders."""
        return LoaderAccessor(self)

    @property
    def images(self) -> MappingProxyType:
        """All the images in the collection."""
        return MappingProxyType(self._images)

    def add_tomogram(
        self,
        image: np.ndarray | da.Array,
        molecules: Molecules,
        image_id: Hashable = None,
    ) -> Self:
        """Add a tomogram and its molecules to the collection."""
        if image_id is None:
            image_id = len(self._images)
            while image_id in self._images:
                image_id += 1
        molecules = molecules.copy()
        molecules.features = molecules.features.with_columns(
            pl.Series(IMAGE_ID_LABEL, np.full((len(molecules)), image_id))
        )
        _molecules_new = self._molecules.concat_with(molecules)

        self._images[image_id] = image
        self._molecules = _molecules_new
        return self

    @property
    def molecules(self) -> Molecules:
        """Collect all the molecules from all the loaders"""
        return self._molecules

    def replace(
        self,
        molecules: Molecules | None = None,
        output_shape: _ShapeType | Unset = None,
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
            order=order,
            scale=scale,
            output_shape=output_shape,
            corner_safe=corner_safe,
        )
        out._images = self._images.copy()
        if molecules is None:
            out._molecules = self._molecules.copy()
        else:
            out._molecules = molecules
            _id_exists = set(molecules.features[IMAGE_ID_LABEL].unique())
            for k in list(out._images.keys()):
                if k not in _id_exists:
                    out._images.pop(k)
        return out

    def construct_loading_tasks(
        self, output_shape: _ShapeType = None
    ) -> list[da.Array]:
        return sum(
            (
                loader.construct_loading_tasks(output_shape=output_shape)
                for loader in self.loaders
            ),
            start=[],
        )

    def construct_dask(self, output_shape: _ShapeType = None) -> da.Array:
        dask_arrays: list[da.Array] = []
        for loader in self.loaders:
            dask_arrays.append(loader.construct_dask(output_shape=output_shape))
        return da.concatenate(dask_arrays, axis=0)


class LoaderAccessor:
    def __init__(self, collection: TomogramCollection):
        self._collection = weakref.ref(collection)

    def __getitem__(self, idx: int) -> SubtomogramLoader:
        col = self._collection()
        mole = col.filter(pl.col(IMAGE_ID_LABEL) == idx).molecules
        if mole.features.shape[0] == 0:
            raise KeyError(idx)
        image = col._images[idx]
        loader = SubtomogramLoader(
            image,
            mole,
            col.order,
            col.scale,
            col.output_shape,
            col.corner_safe,
        )
        return loader

    def __iter__(self) -> Iterator[SubtomogramLoader]:
        col = self._collection()

        for key, group in col.molecules.groupby(IMAGE_ID_LABEL):
            image = col._images[key]
            loader = SubtomogramLoader(
                image,
                group,
                col.order,
                col.scale,
                col.output_shape,
                col.corner_safe,
            )
            yield loader
