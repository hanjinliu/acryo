# pyright: reportPrivateImportUsage=false
from __future__ import annotations
from typing import TYPE_CHECKING
from dask import array as da
import numpy as np
import polars as pl
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from acryo.alignment._base import AlignmentResult
from acryo.backend._api import Backend
from acryo.loader._base import LoaderBase, Unset
from acryo._types import nm, pixel
from acryo.molecules import Molecules
from acryo.tilt import TiltSeriesModel, NoWedge
from acryo._dask import DaskArrayList, map_subvolumes_to_array

if TYPE_CHECKING:
    from typing_extensions import Self

SUBVOLUME_INDEX = "subvolume_index"


class ExtractedSubvolumeLoader(LoaderBase):
    """A class for loading extracted subvolumes directly from files.

    Parameters
    ----------
    dask_array : da.Array
        4D array of all the extracted subvolumes. The ``subvolume_index`` feature of
        the molecules is the index of the subvolume in this array.
    molecules : Molecules
        Molecules with the ``subvolume_index`` feature.
    pre_transform : (N, 4, 4) array, optional
        Affine matrices that map the output coordinates to the coordinates of each
        subvolume in ``dask_array``. If given, subvolumes are transformed by them on
        loading, and ``output_shape`` must also be given.
    output_shape : tuple of int, optional
        Shape of the output subvolumes. Same as the shape of ``dask_array`` if not
        given.
    """

    def __init__(
        self,
        dask_array: da.Array,
        molecules: Molecules,
        order: int = 3,
        scale: nm = 1.0,
        tilt_model: TiltSeriesModel | None = None,
        is_array_prefiltered: bool = False,
        pre_transform: NDArray[np.float32] | None = None,
        output_shape: tuple[pixel, pixel, pixel] | None = None,
    ) -> None:
        if dask_array.ndim != 4:
            raise ValueError(f"Expected a 4D dask array, got {dask_array.ndim}D.")
        if SUBVOLUME_INDEX not in molecules.features.columns:
            raise ValueError(f"Molecules object must have {SUBVOLUME_INDEX!r} feature.")
        if pre_transform is None:
            if output_shape is not None and tuple(output_shape) != dask_array.shape[1:]:
                raise ValueError(
                    "output_shape must be the same as the shape of the subvolumes if "
                    "pre_transform is not given."
                )
            output_shape = dask_array.shape[1:]
        else:
            pre_transform = np.asarray(pre_transform, dtype=np.float32)
            if pre_transform.shape != (dask_array.shape[0], 4, 4):
                raise ValueError(
                    f"Shape of pre_transform must be ({dask_array.shape[0]}, 4, 4), "
                    f"got {pre_transform.shape}."
                )
            if output_shape is None:
                raise ValueError("output_shape must be given with pre_transform.")
        super().__init__(order=order, scale=scale, output_shape=tuple(output_shape))
        self._dask_array = dask_array
        self._molecules = molecules
        self._tilt_model = tilt_model or NoWedge()
        # if True, `dask_array` is the spline coefficients for interpolation of `order`
        self._is_array_prefiltered = is_array_prefiltered
        self._pre_transform = pre_transform

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
        return DaskArrayList(self.construct_dask(output_shape, backend))

    def construct_dask(
        self,
        output_shape: pixel | tuple[pixel, ...] | None = None,
        backend: Backend | None = None,
    ) -> da.Array:
        if self._get_output_shape(output_shape) != self.output_shape:
            raise ValueError("Cannot change output_shape for this class.")
        shape = self.output_shape
        nmole = self.molecules.count()
        if nmole == 0:
            return da.zeros((0, *shape), dtype=np.float32)
        xp = backend or Backend()
        dsk, pre_transform = self._get_stored_arrays()
        shift, rotation = self._get_alignment_results()
        has_alignment = shift is not None or rotation is not None
        if shift is None:
            shift = np.zeros((nmole, 3), dtype=np.float32)
        if self._is_array_prefiltered:
            prefilter = False
        else:
            prefilter = self.order > 1
        if pre_transform is None:
            if not has_alignment:
                return self._get_image_array(dsk, xp)
            if rotation is None or np.allclose(
                rotation.as_quat(), np.array([0, 0, 0, 1], dtype=np.float32)
            ):
                # No rotation. Use ndi.shift
                shift_px = shift / self.scale
                return map_subvolumes_to_array(
                    dsk,
                    _shift_subvolume,
                    shape,
                    dtype=np.float32,
                    var_kwarg={"shift": shift_px},
                    order=self.order,
                    prefilter=prefilter,
                    backend=xp,
                )
        if has_alignment:
            if rotation is None:
                rotation = Rotation.identity(nmole)
            # Molecules are updated by `Molecules.linear_transform` after alignment,
            # which rotates the shift before translation. Do the same here so that the
            # results are consistent with the subvolumes cropped at the aligned molecules.
            shift_px = rotation.apply(shift) / self.scale
            matrices = np.stack(
                [
                    AlignmentResult(0, sh, quat, 0.0).affine_matrix(shape)
                    for sh, quat in zip(shift_px, rotation.as_quat())
                ],
                axis=0,
            )
            if pre_transform is not None:
                # pre-transform is applied to the output of the alignment transform
                matrices = np.matmul(pre_transform, matrices)
        else:
            matrices = pre_transform
        return map_subvolumes_to_array(
            dsk,
            _transform_subvolume,
            shape,
            dtype=np.float32,
            var_kwarg={"matrix": matrices},
            order=self.order,
            output_shape=shape,
            prefilter=prefilter,
            backend=xp,
        )

    def replace(
        self,
        molecules: Molecules | None = None,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset | None = None,
        order: int | None = None,
        scale: float | None = None,
    ) -> Self:
        """Return a new loader with updated attributes."""
        if molecules is None:
            molecules = self.molecules
        if output_shape is not None and output_shape != self.output_shape:
            raise ValueError("Cannot change output_shape for this class.")
        if order is None:
            order = self.order
        if scale is None:
            scale = self.scale
        is_array_prefiltered = self._is_array_prefiltered
        if is_array_prefiltered and order != self.order:
            raise ValueError("Cannot change order for prefiltered array.")
        # subvolumes are selected by the `SUBVOLUME_INDEX` feature on loading, so the
        # stored arrays are passed as is.
        return self.__class__(
            self._dask_array,
            molecules=molecules,
            order=order,
            scale=scale,
            tilt_model=self.tilt_model,
            is_array_prefiltered=is_array_prefiltered,
            pre_transform=self._pre_transform,
            output_shape=self.output_shape,
        )

    def _default_align_kwargs(self):
        return {"tilt": self.tilt_model}

    def _get_stored_arrays(self) -> tuple[da.Array, NDArray[np.float32] | None]:
        """Return the stored subvolumes and pre-transforms of current molecules."""
        indices = self.molecules.features[SUBVOLUME_INDEX].to_numpy()
        dsk = self._dask_array
        pre_transform = self._pre_transform
        if indices.size == dsk.shape[0] and np.array_equal(
            indices, np.arange(indices.size)
        ):
            return dsk, pre_transform
        if pre_transform is not None:
            pre_transform = pre_transform[indices]
        return dsk[indices], pre_transform

    def _get_image_array(self, dsk: da.Array, backend: Backend) -> da.Array:
        """Return the stacked subvolume images (not the spline coefficients)."""
        if not self._is_array_prefiltered:
            return dsk
        # interpolation at the grid points converts the coefficients to the image
        return map_subvolumes_to_array(
            dsk,
            _shift_subvolume,
            self.output_shape,
            dtype=np.float32,
            shift=np.zeros(3, dtype=np.float32),
            order=self.order,
            prefilter=False,
            backend=backend,
        )

    def _get_alignment_results(
        self,
    ) -> tuple[NDArray[np.float32] | None, Rotation | None]:
        df = self.molecules.features
        try:
            shift_df = df.select([f"align-d{r}" for r in "zyx"])
        except pl.exceptions.ColumnNotFoundError:
            shift = None
        else:
            shift = shift_df.to_numpy().astype(np.float32)
        try:
            rotation_df = df.select([f"align-d{r}rot" for r in "zyx"])
        except pl.exceptions.ColumnNotFoundError:
            rotation = None
        else:
            rotation = Rotation.from_rotvec(rotation_df.to_numpy())
        return shift, rotation


def _transform_subvolume(
    img: NDArray[np.float32],
    matrix: NDArray[np.float32],
    order: int,
    output_shape: tuple[int, int, int],
    backend: Backend,
    prefilter: bool = True,
):
    xp = backend
    img = xp.asarray(img)
    return xp.rotated_crop(
        img, matrix, shape=output_shape, order=order, cval=xp.mean, prefilter=prefilter
    )


def _shift_subvolume(
    img: NDArray[np.float32],
    shift: NDArray[np.float32],
    order: int,
    backend: Backend,
    prefilter: bool = True,
):
    xp = backend
    img = xp.asarray(img)
    return xp.shift(img, -shift, order=order, cval=xp.mean, prefilter=prefilter)
