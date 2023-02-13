# pyright: reportPrivateImportUsage=false

from __future__ import annotations
import psutil
from typing import (
    Callable,
    TYPE_CHECKING,
    Iterable,
    Iterator,
    NamedTuple,
    TypeVar,
    Any,
)
import tempfile
from scipy.spatial.transform import Rotation
import numpy as np
from dask import array as da
from dask.delayed import delayed
import polars as pl

from acryo.alignment import (
    BaseAlignmentModel,
    ZNCCAlignment,
    RotationImplemented,
)
from acryo._types import nm, pixel
from acryo._reader import imread
from acryo.molecules import Molecules
from acryo import _utils

if TYPE_CHECKING:
    from typing_extensions import Self
    from dask.delayed import Delayed
    from numpy.typing import NDArray
    from acryo.classification import PcaClassifier


_R = TypeVar("_R")
MEMORY_LIMIT = psutil.virtual_memory().total


class Unset:
    def __bool__(self) -> bool:
        return False

    def __repr__(self) -> str:
        return "Unset"


class SubtomogramLoader:
    """
    A class for efficient loading of subtomograms.

    A ``SubtomogramLoader`` instance is basically composed of two elements,
    an image and a Molecules object. A subtomogram is loaded by creating a
    local rotated Cartesian coordinate at a molecule and calculating mapping
    from the image to the subtomogram.

    Parameters
    ----------
    image : np.ndarray or da.Array
        Tomogram image. Must be 3-D.
    molecules : Molecules
        Molecules object that represents positions and orientations of
        subtomograms.
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
    output_shape : int or tuple of int, optional
        Shape of output subtomogram in pixel. This parameter is not required
        if template (or mask) image is available immediately.
    corner_safe : bool, default is False
        If true, regions around molecules will be cropped at a volume larger
        than ``output_shape`` so that densities at the corners will not be
        lost due to rotation. If target density is globular, this parameter
        should be set false to save computation time.
    """

    def __init__(
        self,
        image: np.ndarray | da.Array,
        molecules: Molecules,
        order: int = 3,
        scale: nm = 1.0,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset = Unset(),
        corner_safe: bool = False,
    ) -> None:
        # check type of input image
        if not isinstance(image, (np.ndarray, da.Array)):
            raise TypeError(
                "Input image of a SubtomogramLoader instance must be np.ndarray "
                f"or dask.Array, got {type(image)}."
            )

        self._image = image

        # check type of molecules
        if not isinstance(molecules, Molecules):
            raise TypeError(
                "The second argument 'molecules' must be a Molecules object, got"
                f"{type(molecules)}."
            )
        self._molecules = molecules

        self._order, self._output_shape, self._scale, self._corner_safe = check_input(
            order, output_shape, scale, corner_safe, image.ndim
        )
        self._cached_dask_array: da.Array | None = None

    def __repr__(self) -> str:
        shape = self.image.shape
        mole_repr = repr(self.molecules)
        return (
            f"{self.__class__.__name__}(tomogram={shape}, molecules={mole_repr}, "
            f"output_shape={self.output_shape}, order={self.order}, "
            f"scale={self.scale:.4f})"
        )

    @classmethod
    def imread(
        cls,
        path: str,
        molecules: Molecules,
        order: int = 3,
        scale: nm | None = None,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset = Unset(),
        corner_safe: bool = False,
        chunks: Any = "auto",
    ):
        dask_array, _scale = imread(path, chunks)
        if scale is None:
            scale = _scale
        return cls(
            dask_array,
            molecules=molecules,
            order=order,
            scale=scale,
            output_shape=output_shape,
            corner_safe=corner_safe,
        )

    @property
    def image(self) -> NDArray[np.float32] | da.Array:
        """Return tomogram image."""
        return self._image

    @property
    def scale(self) -> nm:
        """Get the physical scale of tomogram."""
        return self._scale

    @property
    def output_shape(self) -> tuple[pixel, ...] | Unset:
        """Return the output subtomogram shape."""
        return self._output_shape

    @property
    def molecules(self) -> Molecules:
        """Return the molecules of the subtomogram loader."""
        return self._molecules

    @property
    def features(self) -> pl.DataFrame:
        """The features of molecules."""
        return self._molecules.features

    @property
    def order(self) -> int:
        """Return the interpolation order."""
        return self._order

    @property
    def corner_safe(self) -> bool:
        return self._corner_safe

    def __len__(self) -> int:
        """Return the number of subtomograms."""
        return self.molecules.pos.shape[0]

    def replace(
        self,
        molecules: Molecules | None = None,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset | None = None,
        order: int | None = None,
        scale: float | None = None,
        corner_safe: bool | None = None,
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
        if corner_safe is None:
            corner_safe = self.corner_safe
        return self.__class__(
            self.image,
            molecules=molecules,
            output_shape=output_shape,
            order=order,
            scale=scale,
            corner_safe=corner_safe,
        )

    def copy(self) -> Self:
        """Create a shallow copy of the loader."""
        return self.replace()

    def binning(self, binsize: pixel = 2, *, compute: bool = True) -> Self:
        tr = -(binsize - 1) / 2 * self.scale
        molecules = self.molecules.translate([tr, tr, tr])
        binned_image = _utils.bin_image(self.image, binsize=binsize)
        if isinstance(binned_image, da.Array) and compute:
            binned_image = binned_image.compute()
        out = self.replace(
            molecules=molecules,
            scale=self.scale * binsize,
        )

        out._image = binned_image
        return out

    def construct_loading_tasks(
        self,
        output_shape: pixel | tuple[pixel, ...] | None = None,
    ) -> list[Delayed]:
        """
        Construct a list of subtomogram lazy loader.

        Returns
        -------
        list of Delayed object
            Each object returns a subtomogram on execution by ``da.compute``.
        """
        image = self.image
        scale = self.scale
        if isinstance(image, np.ndarray):
            image = da.from_array(image)

        output_shape = self._get_output_shape(output_shape)

        if self.corner_safe:
            _prep = _utils.prepare_affine_cornersafe
        else:
            _prep = _utils.prepare_affine
        tasks = []
        for i in range(len(self)):
            subvol, mtx = _prep(
                image,
                center=self.molecules.pos[i] / scale,
                output_shape=output_shape,
                rot=self.molecules.rotator[i],
            )
            task = _utils.rotated_crop(
                subvol,
                mtx,
                shape=output_shape,
                order=self.order,
                mode="constant",
                cval=np.mean,
            )
            tasks.append(task)

        return tasks

    def construct_dask(
        self,
        output_shape: pixel | tuple[pixel, ...] | None = None,
    ) -> da.Array:
        """
        Construct a dask array of subtomograms.

        This function is always needed before parallel processing. If subtomograms
        are cached in a memory-map it will be used instead.

        Returns
        -------
        da.Array
            An 4-D array which ``arr[i]`` corresponds to the ``i``-th subtomogram.
        """
        if self._cached_dask_array is not None:
            return self._cached_dask_array

        output_shape = self._get_output_shape(output_shape)

        tasks = self.construct_loading_tasks(output_shape=output_shape)
        arrays = []
        for task in tasks:
            arrays.append(da.from_delayed(task, shape=output_shape, dtype=np.float32))

        out = da.stack(arrays, axis=0)
        return out

    def iter_mapping_tasks(
        self,
        func: Callable,
        *const_args,
        output_shape: pixel | tuple[pixel, ...] | None = None,
        var_kwarg: dict[str, Iterable[Any]] | None = None,
        **const_kwargs,
    ) -> Iterator[Delayed]:
        """
        Iterate over delayed mapping tasks using subtomograms.

        Parameters
        ----------
        func : Callable
            Mapping function. The first argument of the function is the subtomogram.
        output_shape : int or tuple of int, optional
            Shape of subtomograms.
        var_kwarg : dict, optional
            Variable keyword arguments. The length of each argument must be the same
            as the number of subtomograms.

        Yields
        ------
        da.delayed.Delayed object
            Delayed tasks that are ready for ``da.compute``.
        """
        dask_array = self.construct_loading_tasks(output_shape=output_shape)
        delayed_f = delayed(func)
        if var_kwarg is None:
            it = (delayed_f(ar, *const_args, **const_kwargs) for ar in dask_array)
        else:
            it = (
                delayed_f(ar, *const_args, **const_kwargs, **kw)
                for ar, kw in zip(dask_array, _dict_iterrows(var_kwarg))
            )
        return it

    def construct_mapping_tasks(
        self,
        func: Callable,
        *const_args,
        output_shape: pixel | tuple[pixel, ...] | None = None,
        var_kwarg: dict[str, Iterable[Any]] | None = None,
        **const_kwargs,
    ) -> list[Delayed]:
        """
        Construct delayed mapping tasks using subtomograms.

        Parameters
        ----------
        func : Callable
            Mapping function. The first argument of the function is the subtomogram.
        output_shape : int or tuple of int, optional
            Shape of subtomograms.
        var_kwarg : dict, optional
            Variable keyword arguments. The length of each argument must be the same
            as the number of subtomograms.

        Returns
        -------
        list of da.delayed.Delayed object
            List of delayed tasks that are ready for ``da.compute``.
        """
        return list(
            self.iter_mapping_tasks(
                func,
                *const_args,
                output_shape=output_shape,
                var_kwarg=var_kwarg,
                **const_kwargs,
            )
        )

    def create_cache(
        self,
        output_shape: pixel | tuple[pixel] | None = None,
        path: str | None = None,
    ) -> da.Array:
        """
        Create cached stack of subtomograms.

        Parameters
        ----------
        path : str, optional
            File path of the temporary file. If not given file will be created by
            ``tempfile.NamedTemporaryFile`` function.

        Returns
        -------
        da.Array
            A lazy-loading array that uses the memory-mapped array.

        Examples
        --------
        1. Get i-th subtomogram.

        >>> arr = loader.create_cache()
        >>> arr[i]

        2. Subtomogram averaging.

        >>> arr = loader.create_cache()
        >>> avg = np.mean(arr, axis=0).mean()

        """
        output_shape = self._get_output_shape(output_shape)
        dask_array = self.construct_dask(output_shape=output_shape)
        shape = (len(self.molecules),) + output_shape
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

    def asnumpy(self, *, lim: int = MEMORY_LIMIT) -> NDArray[np.float32]:
        """Create a 4D image stack of all the subtomograms."""
        arr = self.construct_dask()
        if arr.nbytes > lim:
            raise MemoryError("The array is too large to be loaded into memory.")
        return arr.compute()

    def average(
        self,
        output_shape: pixel | tuple[pixel] | None = None,
    ) -> NDArray[np.float32]:
        """
        Calculate the average of subtomograms.

        This function execute so-called "subtomogram averaging". The size of
        subtomograms is determined by the ``self.output_shape`` attribute.

        Returns
        -------
        np.ndarray
            Averaged image
        """
        dask_array = self.construct_dask(output_shape=output_shape)
        return da.compute(da.mean(dask_array, axis=0))[0]

    def average_split(
        self,
        n_set: int = 1,
        seed: int | None = 0,
        squeeze: bool = True,
        output_shape: pixel | tuple[pixel] | None = None,
    ) -> NDArray[np.float32]:
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
        np.ndarray
            Averaged image
        """
        output_shape = self._get_output_shape(output_shape)
        rng = np.random.default_rng(seed=seed)

        tasks: list[da.Array] = []
        dask_array = self.construct_dask(output_shape=output_shape)
        nmole = dask_array.shape[0]
        for _ in range(n_set):
            ind0, ind1 = _utils.random_splitter(rng, nmole)
            dask_avg0 = da.mean(dask_array[ind0], axis=0)
            dask_avg1 = da.mean(dask_array[ind1], axis=0)
            tasks.extend([dask_avg0, dask_avg1])

        out = da.compute(tasks)[0]
        stack = np.stack(out, axis=0).reshape(n_set, 2, *output_shape)
        if squeeze and n_set == 1:
            stack = stack[0]
        return stack

    def align(
        self,
        template: NDArray[np.float32],
        *,
        mask: NDArray[np.float32] | None = None,
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
        _max_shifts_px = np.asarray(max_shifts) / self.scale

        model = alignment_model(
            template=template,
            mask=mask,
            **align_kwargs,
        )
        tasks = self.construct_mapping_tasks(
            model.align,
            max_shifts=_max_shifts_px,
            output_shape=template.shape,
            var_kwarg=dict(quaternion=self.molecules.quaternion()),
        )
        all_results = da.compute(tasks)[0]

        local_shifts, local_rot, scores = _allocate(len(self))
        for i, result in enumerate(all_results):
            _, local_shifts[i], local_rot[i], scores[i] = result

        rotator = Rotation.from_quat(local_rot)
        mole_aligned = self.molecules.linear_transform(
            local_shifts * self.scale,
            rotator,
        )

        mole_aligned.features = self.molecules.features.with_columns(
            get_feature_list(scores, local_shifts, rotator.as_rotvec()),
        )

        return self.replace(molecules=mole_aligned, output_shape=template.shape)

    def align_no_template(
        self,
        *,
        mask: NDArray[np.float32]
        | Callable[[NDArray[np.float32]], NDArray[np.float32]]
        | None = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        output_shape: pixel | tuple[pixel] | None = None,
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
        if output_shape is None and isinstance(mask, np.ndarray):
            output_shape = mask.shape

        all_subvols = self.create_cache(output_shape=output_shape)

        template: NDArray[np.float32] = da.compute(da.mean(all_subvols, axis=0))[0]

        # get mask image
        if isinstance(mask, np.ndarray):
            _mask = mask
        elif callable(mask):
            _mask = mask(template)
        else:
            _mask = mask
        return self.align(
            template,
            mask=_mask,
            max_shifts=max_shifts,
            alignment_model=alignment_model,
            **align_kwargs,
        )

    def align_multi_templates(
        self,
        templates: list[NDArray[np.float32]],
        *,
        mask: NDArray[np.float32] | None = None,
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

        local_shifts, local_rot, corr_max = _allocate(len(self))

        # optimal template ID
        labels = np.zeros(len(self), dtype=np.uint32)

        _max_shifts_px = np.asarray(max_shifts) / self.scale

        model = alignment_model(
            template=np.stack(list(templates), axis=0),
            mask=mask,
            **align_kwargs,
        )
        tasks = self.construct_mapping_tasks(
            model.align,
            max_shifts=_max_shifts_px,
            output_shape=templates[0].shape,
            var_kwarg=dict(quaternion=self.molecules.quaternion()),
        )
        all_results = da.compute(tasks)[0]

        local_shifts, local_rot, scores = _allocate(len(self))
        for i, result in enumerate(all_results):
            labels[i], local_shifts[i], local_rot[i], scores[i] = result

        rotator = Rotation.from_quat(local_rot)
        mole_aligned = self.molecules.linear_transform(
            local_shifts * self.scale,
            rotator,
        )

        if isinstance(model, RotationImplemented) and model._n_rotations > 1:
            labels %= n_templates
        labels = labels.astype(np.uint8)

        feature_list = get_feature_list(corr_max, local_shifts, rotator.as_rotvec())
        mole_aligned.features = self.molecules.features.with_columns(
            feature_list + pl.Series("labels", labels)
        )

        return self.replace(molecules=mole_aligned, output_shape=templates[0].shape)

    def subtomoprops(
        self,
        template: NDArray[np.float32],
        func: Callable[[NDArray[np.float32], NDArray[np.float32]], _R],
        mask: NDArray[np.float32] | None = None,
    ) -> list[_R]:
        if mask is None:
            _mask = 1.0
        else:
            _mask = mask
        template_masked = template * _mask

        tasks = self.construct_mapping_tasks(
            func, template_masked, output_shape=template.shape
        )
        all_results: list[_R] = da.compute(tasks)[0]

        return all_results

    def fsc(
        self,
        mask: NDArray[np.float32] | None = None,
        seed: int | None = 0,
        n_set: int = 1,
        dfreq: float = 0.05,
    ) -> pl.DataFrame:
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
        pl.DataFrame
            A data frame with FSC results.
        """

        if mask is None:
            _mask = 1.0
            output_shape = self.output_shape
            if isinstance(output_shape, Unset):
                raise TypeError("Output shape is unknown.")
        else:
            _mask = mask
            output_shape = mask.shape

        if n_set <= 0:
            raise ValueError("'n_set' must be positive.")

        img = self.average_split(
            n_set=n_set,
            seed=seed,
            squeeze=False,
            output_shape=output_shape,
        )
        fsc_all: dict[str, np.ndarray] = {}
        freq = np.zeros(0, dtype=np.float32)
        for i in range(n_set):
            img0, img1 = img[i]
            freq, fsc = _utils.fourier_shell_correlation(
                img0 * _mask, img1 * _mask, dfreq=dfreq
            )
            fsc_all[f"FSC-{i}"] = fsc

        out: dict[str, NDArray[np.float32]] = {"freq": freq}
        out.update(fsc_all)
        return pl.DataFrame(out)

    def classify(
        self,
        template: NDArray[np.float32] | None = None,
        mask: NDArray[np.float32] | None = None,
        *,
        n_components: int = 2,
        n_clusters: int = 2,
        tilt_range: tuple[float, float] | None = None,
        seed: int = 0,
        label_name: str = "labels",
    ) -> ClassificationResult:
        from acryo.classification import PcaClassifier

        if isinstance(self.output_shape, Unset):
            if mask is None:
                raise ValueError("Cannot determine output shape.")
            shape = mask.shape
        else:
            shape = self.output_shape

        if template is None:
            template = self.average(shape)

        model = ZNCCAlignment(template, mask, tilt_range=tilt_range)
        tasks: list[da.Array] = []
        for task in self.iter_mapping_tasks(
            model.masked_difference,
            output_shape=shape,
            var_kwarg=dict(quaternion=self.molecules.quaternion()),
        ):
            tasks.append(da.from_delayed(task, shape=shape, dtype=np.float32))

        stack = da.stack(tasks, axis=0)

        clf = PcaClassifier(
            stack,
            mask,
            n_components=n_components,
            n_clusters=n_clusters,
            seed=seed,
        )
        clf.run()
        mole = self.molecules.copy()
        mole.features = mole.features.with_columns(pl.Series(label_name, clf._labels))
        new = self.replace(molecules=mole)
        return ClassificationResult(new, clf)

    def filter(
        self,
        predicate: pl.Expr | str | pl.Series | list[bool] | np.ndarray,
    ) -> Self:
        """Return a new loader with filtered molecules."""
        return self.replace(molecules=self.molecules.filter(predicate))

    def _get_output_shape(
        self, output_shape: pixel | tuple[pixel] | None
    ) -> tuple[pixel, ...]:
        if output_shape is None:
            if isinstance(self.output_shape, Unset):
                raise ValueError("Output shape is unknown.")
            _output_shape = self.output_shape
        else:
            _output_shape = _utils.normalize_shape(output_shape, ndim=self.image.ndim)
        return _output_shape


class ClassificationResult(NamedTuple):
    """Tuple of classification results."""

    loader: SubtomogramLoader
    classifier: PcaClassifier


def get_feature_list(corr_max, local_shifts, rotvec) -> list[pl.Series]:
    features = [
        pl.Series("score", corr_max),
        pl.Series("shift-z", np.round(local_shifts[:, 0], 2)),
        pl.Series("shift-y", np.round(local_shifts[:, 1], 2)),
        pl.Series("shift-x", np.round(local_shifts[:, 2], 2)),
        pl.Series("rotvec-z", np.round(rotvec[:, 0], 5)),
        pl.Series("rotvec-y", np.round(rotvec[:, 1], 5)),
        pl.Series("rotvec-x", np.round(rotvec[:, 2], 5)),
    ]
    return features


def check_input(
    order: int,
    output_shape: pixel | tuple[pixel, pixel, pixel] | Unset,
    scale: float,
    corner_safe: bool,
    ndim: int,
):
    # check interpolation order
    if order not in (0, 1, 3):
        raise ValueError(
            f"The third argument 'order' must be 0, 1 or 3, got {order!r}."
        )

    # check output_shape
    if isinstance(output_shape, Unset):
        _output_shape = output_shape
    else:
        _output_shape = _utils.normalize_shape(output_shape, ndim=ndim)

    # check scale
    _scale = float(scale)
    if _scale <= 0:
        raise ValueError("Negative scale is not allowed.")

    _corner_safe = bool(corner_safe)
    return order, _output_shape, _scale, _corner_safe


def _allocate(size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # shift in local Cartesian
    local_shifts = np.zeros((size, 3))

    # maximum ZNCC
    corr_max = np.zeros(size)

    # rotation (quaternion) in local Cartesian
    local_rot = np.zeros((size, 4))
    local_rot[:, 3] = 1  # identity map in quaternion

    return local_shifts, local_rot, corr_max


def _dict_iterrows(d: dict[str, Iterable[Any]]):
    """
    Generater similar to pl.DataFrame.iterrows().

    >>> _dict_iterrows({'a': [1, 2, 3], 'b': [4, 5, 6]})

    will yield {'a': 1, 'b': 4}, {'a': 2, 'b': 5}, {'a': 3, 'b': 6}.
    """
    keys = d.keys()
    value_iters = [iter(v) for v in d.values()]

    dict_out = dict.fromkeys(keys, None)
    while True:
        try:
            for k, viter in zip(keys, value_iters):
                dict_out[k] = next(viter)
            yield dict_out
        except StopIteration:
            break
