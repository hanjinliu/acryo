# pyright: reportPrivateImportUsage=false
from __future__ import annotations
from types import MappingProxyType

from typing import Hashable, Iterator, TYPE_CHECKING
import numpy as np
import polars as pl

from dask import array as da

from acryo.loader import SubtomogramLoader, Unset, check_input
from acryo.molecules import Molecules
from acryo.alignment import (
    BaseAlignmentModel,
    ZNCCAlignment,
)
from acryo._types import nm, pixel

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from typing_extensions import Self

IMAGE_ID_LABEL = "image"


class TomogramCollection:
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

        self._order, self._output_shape, self._scale, self._corner_safe = check_input(
            order, output_shape, scale, corner_safe, 3
        )
        self._images: dict[Hashable, NDArray[np.float32] | da.Array] = {}
        self._molecules: Molecules = Molecules.empty([IMAGE_ID_LABEL])
        self._loaders = LoaderAccessor(self)

    @property
    def loaders(self) -> LoaderAccessor:
        """Interface to access the subtomogram loaders."""
        return self._loaders

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

    @property
    def features(self) -> pl.DataFrame:
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
        images: dict[Hashable, NDArray[np.float32] | da.Array] | None = None,
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
        if images is None:
            out._images = self._images.copy()
        else:
            out._images = dict(images)
        return out

    def copy(self) -> Self:
        return self.replace()

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
        dask_arrays: list[da.Array] = []
        for loader in self.loaders:
            dask_arrays.append(loader.construct_dask(output_shape=output_shape))
        dask_array = da.concatenate(dask_arrays, axis=0)
        return da.compute(da.mean(dask_array, axis=0))[0]

    def align(
        self,
        template: NDArray[np.float32],
        *,
        mask: NDArray[np.float32] | None = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1,
        alignment_model: type[BaseAlignmentModel] = ZNCCAlignment,
        **align_kwargs,
    ):
        new = self.replace(images={})
        for loader in self.loaders:
            result = loader.align(
                template=template,
                mask=mask,
                max_shifts=max_shifts,
                alignment_model=alignment_model,
                **align_kwargs,
            )
            new.add_tomogram(result.image, result.molecules)
        return new

    def filter(
        self,
        predicate: pl.Expr | str | pl.Series | list[bool] | np.ndarray,
    ) -> Self:
        """
        Filter the molecules and tomograms based on the predicate.

        This method comes from `polars.DataFrame.filter`.

        Examples
        --------
        >>> collection.filter(pl.col("score") > 0.5)

        Parameters
        ----------
        predicate : pl.Expr | str | pl.Series | list[bool] | np.ndarray
            Predicate to filter on.

        Returns
        -------
        TomogramCollection
            A collection with filtered molecules and tomograms.
        """
        features = self.molecules.to_dataframe().filter(predicate)
        mole = Molecules.from_dataframe(features)
        new = self.copy()
        new._molecules = mole
        _id_exists = set(
            mole.features.select(IMAGE_ID_LABEL).unique().to_numpy().ravel()
        )
        for k in list(new._images.keys()):
            if k not in _id_exists:
                new._images.pop(k)
        return new


class LoaderAccessor:
    def __init__(self, collection: TomogramCollection):
        self._collection = collection

    @property
    def collection(self) -> TomogramCollection:
        return self._collection

    def __getitem__(self, idx: int) -> SubtomogramLoader:
        col = self.collection
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
        col = self.collection

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
