from __future__ import annotations
from typing import (
    Callable,
    TYPE_CHECKING,
    TypeVar,
)
import warnings
import weakref
import tempfile
import pandas as pd
from scipy.spatial.transform import Rotation
import numpy as np
from dask import array as da, delayed

from .alignment import (
    BaseAlignmentModel,
    ZNCCAlignment,
    SupportRotation,
)
from ._types import nm, pixel
from .molecules import Molecules
from . import _utils

if TYPE_CHECKING:
    from typing_extensions import Self

_A = TypeVar("_A", np.ndarray, da.core.Array)
_R = TypeVar("_R")


def subtomogram_loader(
    image: np.ndarray | da.core.Array,
    molecules: Molecules,
    output_shape: pixel | tuple[pixel, pixel, pixel],
    order: int = 3,
    scale: nm = 1.0,
    chunksize: int = 1,
) -> SubtomogramLoader | ChunkedSubtomogramLoader:
    """
    Return a proper subtomogram loader object.

    Parameters
    ----------
    image : np.ndarray or da.core.Array
        Tomogram image. Must be 3-D.
    molecules : Molecules
        Molecules object that represents positions and orientations of
        subtomograms.
    output_shape : int or tuple of int
        Shape of output subtomogram in pixel.
    order : int, default is 3
        Interpolation order of subtomogram sampling.
        - 0 = Nearest neighbor
        - 1 = Linear interpolation
        - 3 = Cubic interpolation
    scale : float, default is 1.0
        Physical scale of pixel, such as nm. This value does not affect
        averaging/alignment results but molecule coordinates are multiplied
        by this value. This parameter is useful when another loader with
        binned image is created.
    chunksize : int, default is 1
        Set to >1 if adjacent molecules are physically adjacent.

    Returns
    -------
    SubtomogramLoader or ChunkedSubtomogramLoader object
        Proper subtomogram loader object.
    """
    if not image.ndim == 3:
        raise TypeError("Input image must be 3D.")
    kwargs = dict(
        image=image,
        molecules=molecules,
        output_shape=output_shape,
        order=order,
        scale=scale,
    )
    if chunksize == 1:
        return SubtomogramLoader(**kwargs)  # type: ignore
    else:
        return ChunkedSubtomogramLoader(**kwargs, chunksize=chunksize)  # type: ignore


class SubtomogramLoader:
    """A basic subtomogram loader class."""

    def __init__(
        self,
        image: _A,
        molecules: Molecules,
        output_shape: pixel | tuple[pixel, pixel, pixel],
        order: int = 3,
        scale: nm = 1.0,
    ) -> None:
        """
        A class for efficient loading of subtomograms.

        A ``SubtomogramLoader`` instance is basically composed of two elements,
        an image and a Molecules object. A subtomogram is loaded by creating a
        local rotated Cartesian coordinate at a molecule and calculating mapping
        from the image to the subtomogram.

        Parameters
        ----------
        image : np.ndarray or da.core.Array
            Tomogram image. Must be 3-D.
        molecules : Molecules
            Molecules object that represents positions and orientations of
            subtomograms.
        output_shape : int or tuple of int
            Shape of output subtomogram in pixel.
        order : int, default is 3
            Interpolation order of subtomogram sampling.
            - 0 = Nearest neighbor
            - 1 = Linear interpolation
            - 3 = Cubic interpolation
        scale : float, default is 1.0
            Physical scale of pixel, such as nm. This value does not affect
            averaging/alignment results but molecule coordinates are multiplied
            by this value. This parameter is useful when another loader with
            binned image is created.
        """
        # check type of input image
        if isinstance(image, np.ndarray):
            self._image_ref = weakref.ref(image)
        elif isinstance(image, da.core.Array):
            self._image_ref = image
        else:
            raise TypeError(
                "Input image of a SubtomogramLoader instance must be np.ndarray "
                f"or dask.core.Array, got {type(image)}."
            )

        # check type of molecules
        if not isinstance(molecules, Molecules):
            raise TypeError(
                "The second argument 'molecules' must be a Molecules object, got"
                f"{type(molecules)}."
            )
        self._molecules = molecules

        # check interpolation order
        if order not in (0, 1, 3):
            raise ValueError(
                f"The third argument 'order' must be 0, 1 or 3, got {order!r}."
            )
        self._order = order

        # check output_shape
        if isinstance(output_shape, int):
            self._output_shape = (output_shape,) * image.ndim
        else:
            self._output_shape = tuple(output_shape)

        # check scale
        self._scale = float(scale)
        if self._scale <= 0:
            raise ValueError("Negative scale is not allowed.")

        self._cached_dask_array: da.core.Array | None = None

    def __repr__(self) -> str:
        shape = "x".join(map(str, self.image.shape))
        mole_repr = repr(self.molecules)
        return (
            f"{self.__class__.__name__}(tomogram=[{shape}], molecules={mole_repr}, "
            f"output_shape={self.output_shape}, order={self.order}, "
            f"scale={self.scale:.4f})"
        )

    @property
    def image(self) -> _A:
        """Return tomogram image."""
        if isinstance(self._image_ref, weakref.ReferenceType):
            image = self._image_ref()
        else:
            image = self._image_ref
        if image is None:
            raise ValueError("No tomogram found.")
        return image  # type: ignore

    @property
    def scale(self) -> nm:
        """Get the physical scale of tomogram."""
        return self._scale

    @property
    def output_shape(self) -> tuple[pixel, ...]:
        """Return the output subtomogram shape."""
        return self._output_shape

    @property
    def molecules(self) -> Molecules:
        """Return the molecules of the subtomogram loader."""
        return self._molecules

    @property
    def order(self) -> int:
        """Return the interpolation order."""
        return self._order

    def __len__(self) -> int:
        """Return the number of subtomograms."""
        return self.molecules.pos.shape[0]

    def replace(
        self,
        molecules: Molecules | None = None,
        output_shape: pixel | tuple[pixel, pixel, pixel] | None = None,
        order: int | None = None,
        scale: float | None = None,
    ) -> Self:
        """Return a new instance with different parameter(s)."""
        if molecules is None:
            molecules = self.molecules
        if output_shape is None:
            output_shape = self.output_shape
        if order is None:
            order = self.order
        if scale is None:
            scale = self.scale
        return self.__class__(
            self.image,
            molecules=molecules,
            output_shape=output_shape,
            order=order,
            scale=scale,
        )

    def copy(self) -> Self:
        """Create a shallow copy of the loader."""
        return self.replace()

    def binning(self, binsize: pixel = 2, *, compute: bool = True) -> Self:
        tr = -(binsize - 1) / 2 * self.scale
        molecules = self.molecules.translate([tr, tr, tr])
        binned_image = _utils.bin_image(self.image, binsize=binsize)
        if isinstance(binned_image, da.core.Array) and compute:
            binned_image = binned_image.compute()
        out = self.replace(
            molecules=molecules,
            scale=self.scale * binsize,
        )

        out._image_ref = weakref.ref(binned_image)
        return out

    def _check_shape(self, template: np.ndarray, name: str = "template") -> None:
        if template.shape != self.output_shape:
            warnings.warn(
                f"'output_shape' of {self.__class__.__name__} object "
                f"{self.output_shape!r} differs from the shape of {name} image "
                f"{template.shape!r}. 'output_shape' is updated.",
                UserWarning,
            )
            self._output_shape = template.shape
        return None

    def get_subtomogram(self, i: int) -> np.ndarray:
        if self._cached_dask_array is not None:
            return da.compute(self._cached_dask_array[i])[0]

        image = self.image
        scale = self.scale
        coords = self.molecules.cartesian_at(i, self.output_shape, scale)
        mapped = _utils.map_coordinates(image, coords, order=self.order, cval=np.mean)
        subvol = np.stack(list(mapped), axis=0)
        return subvol

    def construct_dask(self) -> da.core.Array:
        """
        Construct a dask array of subtomograms.

        This function is always needed before parallel processing. If subtomograms
        are cached in a memory-map it will be used instead.

        Returns
        -------
        da.core.Array
            An 4-D array with axes "pzyx".
        """
        if self._cached_dask_array is not None:
            return self._cached_dask_array
        delayed_get_subtomogram = delayed(self.get_subtomogram)

        arrays = [
            da.from_delayed(
                delayed_get_subtomogram(i),
                shape=self.output_shape,
                dtype=np.float32,
            )
            for i in range(len(self))
        ]
        return da.stack(arrays, axis=0)

    def construct_map(self, f: Callable, *args, **kwargs) -> list[da.core.Delayed]:
        dask_array = self.construct_dask()
        delayed_f = delayed(f)
        tasks = [delayed_f(ar, *args, **kwargs) for ar in dask_array]
        return tasks

    def create_cache(self, path: str | None = None) -> da.core.Array:
        """
        Create cached stack of subtomograms.

        Parameters
        ----------
        path : str, optional
            File path of the temporary file. If not given file will be created  by
            ``tempfile.NamedTemporaryFile`` function.

        Returns
        -------
        da.core.Array
            A lazy-loading array that uses the memory-mapped array.

        Examples
        --------
        1. Get i-th subtomogram.

        >>> arr = loader.to_lazy_imgarray()  # axes = "pzyx"
        >>> arr[i]

        2. Subtomogram averaging.

        >>> arr = loader.to_lazy_imgarray()  # axes = "pzyx"
        >>> avg = arr.proj("p")  # identical to np.mean(arr, axis=0)

        """
        dask_array = self.construct_dask()
        shape = (len(self.molecules),) + self.output_shape
        kwargs = dict(dtype=np.float32, mode="w+", shape=shape)
        if path is None:
            with tempfile.NamedTemporaryFile() as ntf:
                mmap = np.memmap(ntf, **kwargs)
        else:
            mmap = np.memmap(path, **kwargs)

        mmap[:] = dask_array[:]
        darr = da.from_array(
            mmap,
            chunks=(1,) + self.output_shape,  # type: ignore
            meta=np.array([], dtype=np.float32),
        )
        return darr

    def asnumpy(self) -> np.ndarray:
        """Create a 4D image stack of all the subtomograms."""
        arr = self.construct_dask()
        return arr.compute()

    def average(
        self,
    ) -> np.ndarray:
        """
        Calculate the average of subtomograms.

        This function execute so-called "subtomogram averaging". The size of
        subtomograms is determined by the ``self.output_shape`` attribute.

        Returns
        -------
        np.ndarray
            Averaged image
        """
        dask_array = self.construct_dask()
        return da.compute(da.mean(dask_array, axis=0))[0]

    def average_split(
        self,
        n_set: int = 1,
        seed: int | None = 0,
        squeeze: bool = True,
    ) -> np.ndarray:
        """
        Split subtomograms into two set and average separately.

        This method executes pairwise subtomogram averaging using randomly
        selected molecules, which is useful for calculation of such as Fourier
        shell correlation.

        Parameters
        ----------
        n_set : int, default is 1
            Number of split set of averaged image.
        seed : random seed, default is 0
            Random seed to determine how subtomograms will be split.

        Returns
        -------
        ImgArray
            Averaged image
        """
        nmole = len(self)
        np.random.seed(seed)
        tasks = []
        for _ in range(n_set):
            sl = np.random.choice(np.arange(nmole), nmole // 2).tolist()
            indices0 = np.zeros(nmole, dtype=bool)
            indices0[sl] = True
            indices1 = np.ones(nmole, dtype=bool)
            indices1[sl] = False
            dask_array = self.construct_dask()
            dask_avg0 = da.mean(dask_array[indices0], axis=0)
            dask_avg1 = da.mean(dask_array[indices1], axis=0)
            tasks.extend([dask_avg0, dask_avg1])
        np.random.seed(None)
        out = da.compute(tasks)[0]
        stack = np.stack(out, axis=0).reshape(n_set, 2, *self.output_shape)
        if squeeze and n_set == 1:
            stack = stack[0]
        return stack

    def align(
        self,
        template: np.ndarray,
        *,
        mask: np.ndarray | None = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        alignment_model: type[BaseAlignmentModel] = ZNCCAlignment,
        **align_kwargs,
    ) -> Self:
        """
        Align subtomograms to the template image.

        This method conduct so called "subtomogram alignment". Only shifts and rotations
        are calculated in this method. To get averaged image, you'll have to run "average"
        method using the resulting SubtomogramLoader instance.

        Parameters
        ----------
        template : np.ndarray, optional
            Template image.
        mask : np.ndarray, optional
            Mask image. Must in the same shae as the template.
        max_shifts : int or tuple of int, default is (1., 1., 1.)
            Maximum shift between subtomograms and template.
        alignment_model : subclass of BaseAlignmentModel, optional
            Alignment model class used for subtomogram alignment. By default,
            ``ZNCCAlignment`` will be used.
        align_kwargs : optional keyword arguments
            Additional keyword arguments passed to the input alignment model.

        Returns
        -------
        SubtomogramLoader
            An loader instance with updated molecules.
        """
        self._check_shape(template)

        _max_shifts_px = np.asarray(max_shifts) / self.scale

        model = alignment_model(
            template=template,
            mask=mask,
            **align_kwargs,
        )
        tasks = self.construct_map(model.align, _max_shifts_px)
        all_results = da.compute(tasks)[0]

        local_shifts, local_rot, scores = _allocate(len(self))
        for i, result in enumerate(all_results):
            _, local_shifts[i], local_rot[i], scores[i] = result

        rotator = Rotation.from_quat(local_rot)
        mole_aligned = self.molecules.linear_transform(
            local_shifts * self.scale,
            rotator,
        )

        mole_aligned.features = update_features(
            self.molecules.features.copy(),
            get_features(scores, local_shifts, rotator.as_rotvec()),
        )

        return self.replace(molecules=mole_aligned)

    def align_no_template(
        self,
        *,
        mask: np.ndarray | Callable[[np.ndarray], np.ndarray] | None = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        alignment_model: type[BaseAlignmentModel] = ZNCCAlignment,
        **align_kwargs,
    ) -> Self:
        """
        Align subtomograms without template image.

        A template-free version of :func:`iter_align`. This method first
        calculates averaged image and uses it for the alignment template. To
        avoid loading same subtomograms twice, a memory-mapped array is created
        internally (so the second subtomogram loading is faster).

        Parameters
        ----------
        mask : np.ndarray or callable, optional
            Mask image. Must in the same shape as the template.
        max_shifts : int or tuple of int, default is (1., 1., 1.)
            Maximum shift between subtomograms and template.
        alignment_model : subclass of BaseAlignmentModel, optional
            Alignment model class used for subtomogram alignment. By default,
            ``ZNCCAlignment`` will be used.
        align_kwargs : optional keyword arguments
            Additional keyword arguments passed to the input alignment model.

        Returns
        -------
        SubtomogramLoader
            An loader instance with updated molecules.
        """
        all_subvols = self.create_cache()

        template = da.compute(da.mean(all_subvols, axis=0))[0]

        # get mask image
        if isinstance(mask, np.ndarray):
            mask = mask
        elif callable(mask):
            mask = mask(template)
        else:
            mask = mask
        return self.align(
            template,
            mask=mask,
            max_shifts=max_shifts,
            alignment_model=alignment_model,
            **align_kwargs,
        )

    def align_multi_templates(
        self,
        templates: list[np.ndarray],
        *,
        mask: np.ndarray | None = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        alignment_model: type[BaseAlignmentModel] = ZNCCAlignment,
        **align_kwargs,
    ) -> Self:
        """
        Align subtomograms with multiple template images.

        A multi-template version of :func:`iter_align`. This method calculate cross
        correlation for every template and uses the best local shift, rotation and
        template.

        Parameters
        ----------
        templates: list of ImgArray
            Template images.
        mask : np.ndarray, optional
            Mask image. Must in the same shape as the template.
        max_shifts : int or tuple of int, default is (1., 1., 1.)
            Maximum shift between subtomograms and template.
        alignment_model : subclass of BaseAlignmentModel, optional
            Alignment model class used for subtomogram alignment. By default,
            ``ZNCCAlignment`` will be used.
        align_kwargs : optional keyword arguments
            Additional keyword arguments passed to the input alignment model.

        Returns
        -------
        SubtomogramLoader
            An loader instance with updated molecules.
        """
        n_templates = len(templates)
        self._check_shape(templates[0])

        local_shifts, local_rot, corr_max = _allocate(len(self))

        # optimal template ID
        labels = np.zeros(len(self), dtype=np.uint32)

        _max_shifts_px = np.asarray(max_shifts) / self.scale

        model = alignment_model(
            template=np.stack(list(templates), axis=0),
            mask=mask,
            **align_kwargs,
        )
        tasks = self.construct_map(model.align, _max_shifts_px)
        all_results = da.compute(tasks)[0]

        local_shifts, local_rot, scores = _allocate(len(self))
        for i, result in enumerate(all_results):
            labels[i], local_shifts[i], local_rot[i], scores[i] = result

        rotator = Rotation.from_quat(local_rot)
        mole_aligned = self.molecules.linear_transform(
            local_shifts * self.scale,
            rotator,
        )

        if isinstance(model, SupportRotation) and model._n_rotations > 1:
            labels %= n_templates
        labels = labels.astype(np.uint8)

        mole_aligned.features = pd.concat(
            [
                self.molecules.features,
                get_features(corr_max, local_shifts, rotator.as_rotvec()),
                pd.DataFrame({"labels": labels}),
            ],
            axis=1,
        )

        return self.replace(molecules=mole_aligned)

    def subtomoprops(
        self,
        template: np.ndarray,
        func: Callable[[np.ndarray, np.ndarray], _R],
        mask: np.ndarray | None = None,
    ) -> list[_R]:

        if mask is None:
            _mask = 1
        else:
            _mask = mask
        template_masked = template * _mask

        tasks = self.construct_map(func, template_masked)
        all_results: list[_R] = da.compute(tasks)[0]

        return all_results

    def fsc(
        self,
        mask: np.ndarray | None = None,
        seed: int | None = 0,
        n_set: int = 1,
        dfreq: float = 0.05,
    ) -> pd.DataFrame:
        """
        Calculate Fourier shell correlation.

        Parameters
        ----------
        mask : np.ndarray, optional
            Mask image, by default None
        seed : random seed, default is 0
            Random seed used to split subtomograms.
        n_set : int, default is 1
            Number of split set of averaged images.
        dfreq : float, default is 0.05
            Frequency sampling width.

        Returns
        -------
        pd.DataFrame
            A data frame with FSC results.
        """
        if mask is None:
            _mask = 1
        else:
            self._check_shape(mask, "mask")
            _mask = mask

        if n_set <= 0:
            raise ValueError("'n_set' must be positive.")

        img = self.average_split(n_set=n_set, seed=seed, squeeze=False)
        fsc_all: dict[str, np.ndarray] = {}
        for i in range(n_set):
            img0, img1 = img[i]
            freq, fsc = _utils.fourier_shell_correlation(
                img0 * _mask, img1 * _mask, dfreq=dfreq
            )
            fsc_all[f"FSC-{i}"] = fsc

        out: dict[str, np.ndarray] = {"freq": freq}  # type: ignore
        out.update(fsc_all)
        return pd.DataFrame(out)


class ChunkedSubtomogramLoader(SubtomogramLoader):
    """A subtomogram loader class for chunkwise loading."""

    def __init__(
        self,
        image: _A,
        molecules: Molecules,
        output_shape: int | tuple[int, int, int],
        order: int = 3,
        scale: nm = 1.0,
        chunksize: int = 100,
    ) -> None:
        """
        A SubtomogramLoader class equipped with chunkwise subtomogram loading.

        Parameters
        ----------
        image : np.ndarray or da.core.Array
            Tomogram image. Must be 3-D.
        molecules : Molecules
            Molecules object that represents positions and orientations of
            subtomograms.
        output_shape : int or tuple of int
            Shape of output subtomogram in pixel.
        order : int, default is 3
            Interpolation order of subtomogram sampling.
            - 0 = Nearest neighbor
            - 1 = Linear interpolation
            - 3 = Cubic interpolation
        scale : float, default is 1.0
            Physical scale of pixel, such as nm. This value does not affect
            averaging/alignment results but molecule coordinates are multiplied
            by this value. This parameter is useful when another loader with
            binned image is created.
        chunksize : int, default is 1
            How many subtomogram will be loaded at the same time.
        """
        super().__init__(image, molecules, output_shape, order, scale=scale)
        self._chunksize = chunksize

    @property
    def chunksize(self) -> int:
        """Return the chunk size on subtomogram loading."""
        return self._chunksize

    def replace(
        self,
        molecules: Molecules | None = None,
        output_shape: int | tuple[int, int, int] | None = None,
        order: int | None = None,
        scale: float | None = None,
        chunksize: int | None = None,
    ) -> Self:
        """Return a new instance with different parameter(s)."""
        if molecules is None:
            molecules = self.molecules
        if output_shape is None:
            output_shape = self.output_shape
        if order is None:
            order = self.order
        if scale is None:
            scale = self.scale
        if chunksize is None:
            chunksize = self.chunksize
        return self.__class__(
            self.image,
            molecules=molecules,
            output_shape=output_shape,
            order=order,
            scale=scale,
            chunksize=chunksize,
        )

    def get_subtomogram_chunk(self, chunk_index: int) -> np.ndarray:
        """Generate subtomogram list chunk-wise."""
        image = self.image
        sl = slice(chunk_index * self.chunksize, (chunk_index + 1) * self.chunksize)
        coords = self.molecules.cartesian_at(sl, self.output_shape, self.scale)
        subvols = _utils.multi_map_coordinates(
            image, coords, order=self.order, cval=np.mean
        )
        return subvols

    def construct_dask(self) -> da.core.Array:
        delayed_get_subtomogram = delayed(self.get_subtomogram_chunk)
        n_full_chunk, resid = divmod(len(self), self.chunksize)
        shapes = [self.chunksize] * n_full_chunk + [resid]

        arrays = [
            da.from_delayed(
                delayed_get_subtomogram(i),
                shape=(c,) + self.output_shape,
                dtype=np.float32,
            )
            for i, c in enumerate(shapes)
        ]
        return da.concatenate(arrays, axis=0)

    zShift = "shift-z"
    yShift = "shift-y"
    xShift = "shift-x"
    zRotvec = "rotvec-z"
    yRotvec = "rotvec-y"
    xRotvec = "rotvec-x"


def get_features(corr_max, local_shifts, rotvec) -> pd.DataFrame:
    features = {
        "score": corr_max,
        "shift-z": np.round(local_shifts[:, 0], 2),
        "shift-y": np.round(local_shifts[:, 1], 2),
        "shift-x": np.round(local_shifts[:, 2], 2),
        "rotvec-z": np.round(rotvec[:, 0], 5),
        "rotvec-y": np.round(rotvec[:, 1], 5),
        "rotvec-x": np.round(rotvec[:, 2], 5),
    }
    return pd.DataFrame(features)


def update_features(
    features: pd.DataFrame,
    values: dict | pd.DataFrame,
):
    """Update features with new values."""
    for name, value in values.items():
        features[name] = value
    return features


def _allocate(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # shift in local Cartesian
    local_shifts = np.zeros((size, 3))

    # maximum ZNCC
    corr_max = np.zeros(size)

    # rotation (quaternion) in local Cartesian
    local_rot = np.zeros((size, 4))
    local_rot[:, 3] = 1  # identity map in quaternion

    return local_shifts, local_rot, corr_max
