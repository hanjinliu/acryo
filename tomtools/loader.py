from __future__ import annotations
from typing import (
    Any,
    Callable,
    Generator,
    Iterator,
    TYPE_CHECKING,
)
import warnings
import weakref
import tempfile
import pandas as pd
from scipy.spatial.transform import Rotation
import numpy as np
from numpy.typing import ArrayLike
import impy as ip
from dask import array as da
from .alignment import BaseAlignmentModel, ZNCCAlignment, SupportRotation
from .molecules import Molecules
from . import _utils
from .const import Align

if TYPE_CHECKING:
    from .alignment import AlignmentResult
    from .alignment._types import Ranges

nm = float  # alias


class BaseLoader:
    def __init__(
        self,
        image: ip.ImgArray | ip.LazyImgArray | np.ndarray | da.core.Array,
        mole: Molecules,
        output_shape: int | tuple[int, int, int],
        order: int = 3,
    ) -> None:
        ndim = 3
        if not isinstance(image, (ip.ImgArray, ip.LazyImgArray)):
            if isinstance(image, np.ndarray) and image.ndim == ndim:
                image = ip.asarray(image, axes="zyx")
            elif isinstance(image, da.core.Array) and image.ndim == ndim:
                image = ip.LazyImgArray(image, axes="zyx")
            else:
                raise TypeError("'image' must be a 3D numpy ndarray like object.")
        self.image = image
        self._molecules = mole
        self._order = order
        if isinstance(output_shape, int):
            output_shape = (output_shape,) * ndim
        else:
            output_shape = tuple(output_shape)
        self._output_shape = output_shape

    def __repr__(self) -> str:
        shape = str(self.image.shape).lstrip("AxesShape")
        mole_repr = repr(self.molecules)
        return f"{self.__class__.__name__}(tomogram={shape}, molecules={mole_repr})"

    @property
    def image(self) -> ip.ImgArray | ip.LazyImgArray:
        """Return tomogram image."""
        image = self._image_ref()
        if image is None:
            raise ValueError("No tomogram found.")
        return image

    @image.setter
    def image(self, image: ip.ImgArray | ip.LazyImgArray):
        """Set tomogram as a weak reference."""
        self._image_ref = weakref.ref(image)

    @property
    def scale(self) -> nm:
        """Get the scale (nm/px) of tomogram."""
        return self.image.scale.x

    @property
    def output_shape(self) -> tuple[int, ...]:
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
        output_shape: int | tuple[int, int, int] | None = None,
        order: int | None = None,
    ):
        """Return a new instance with different parameter(s)."""
        if molecules is None:
            molecules = self.molecules
        if output_shape is None:
            output_shape = self.output_shape
        if order is None:
            order = self.order
        return self.__class__(
            self.image,
            molecules=molecules,
            output_shape=output_shape,
            order=order,
        )
    
    
    def _check_shape(self, template: ip.ImgArray, name: str = "template") -> None:
        if template.shape != self.output_shape:
            warnings.warn(
                f"'output_shape' of {self.__class__.__name__} object "
                f"{self.output_shape!r} differs from the shape of {name} image "
                f"{template.shape!r}. 'output_shape' is updated.",
                UserWarning,
            )
            self.output_shape = template.shape
        return None

    def _resolve_iterator(
        self, it: Generator[Any, Any, ip.ImgArray], callback: Callable
    ) -> ip.ImgArray:
        """Iterate over an iterator until it returns something."""
        if callback is None:
            callback = lambda x: None
        while True:
            try:
                next(it)
                callback(self)
            except StopIteration as e:
                results = e.value
                break

        return results
    
    def get_subtomogram(self, i: int) -> ip.ImgArray:
        image = self.image
        scale = image.scale.x
        coords = self.molecules.cartesian_at(i, self.output_shape, scale)
        with ip.use("cupy"):
            subvols = np.stack(
                _utils.map_coordinates(
                    image, coords, order=self.order, cval=np.mean
                ),
                axis=0,
            )
        subvols = ip.asarray(subvols, axes="zyx")
        subvols.set_scale(image)
        return subvols
    
    def iter_subtomograms(self) -> Iterator[ip.ImgArray]:
        for i in range(len(self)):
            yield self.get_subtomogram(i)

    def iter_to_memmap(self, path: str | None = None):
        """
        Create an iterator that convert all the subtomograms into a memory-mapped array.

        This function is useful when the same set of subtomograms will be used for many
        times but it should not be fully loaded into memory. A temporary file will be
        created to store subtomograms by default.

        Parameters
        ----------
        path : str, optional
            File path of the temporary file. If not given file will be created  by
            ``tempfile.NamedTemporaryFile`` function.

        Returns
        -------
        LazyImgArray
            A lazy-loading array that uses the memory-mapped array.

        Yields
        ------
        ImgArray
            Subtomogram at each position.
        """
        shape = (len(self.molecules),) + self.output_shape
        kwargs = dict(dtype=np.float32, mode="w+", shape=shape)
        if path is None:
            with tempfile.NamedTemporaryFile() as ntf:
                mmap = np.memmap(ntf, **kwargs)
        else:
            mmap = np.memmap(path, **kwargs)

        for i, subvol in enumerate(self.iter_subtomograms()):
            mmap[i] = subvol
            yield subvol
        darr = da.from_array(
            mmap, chunks=(1,) + self.output_shape, meta=np.array([], dtype=np.float32)
        )
        arr = ip.LazyImgArray(darr, name="All_subtomograms", axes="pzyx")
        arr.set_scale(self.image)
        return arr

    def to_lazy_imgarray(self, path: str | None = None) -> ip.LazyImgArray:
        """
        An non-iterator version of :func:`iter_to_memmap`.

        Parameters
        ----------
        path : str, optional
            File path of the temporary file. If not given file will be created  by
            ``tempfile.NamedTemporaryFile`` function.

        Returns
        -------
        LazyImgArray
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
        it = self.iter_to_memmap(path)
        return self._resolve_iterator(it, lambda x: None)

    def to_stack(self, binsize: int = 1) -> ip.ImgArray:
        """Create a 4D image stack of all the subtomograms."""
        images = list(self.iter_subtomograms(binsize=binsize))
        stack: ip.ImgArray = np.stack(images, axis="p")
        stack.set_scale(xyz=self.image.scale.x * binsize)
        return stack

    def iter_average(
        self,
        classifier: Callable[[np.ndarray], bool] | None = None,
    ) -> Generator[ip.ImgArray, None, ip.ImgArray]:
        """
        Create an iterator that calculate the averaged image from a tomogram.

        This function execute so-called "subtomogram averaging". The size of
        subtomograms is determined by the ``self.output_shape`` attribute.

        Parameters
        ----------
        classifier : callable, optional
            If provided, only subtomograms that satisfy ``classifier(img)==True`` will
            be used.

        Returns
        -------
        ImgArray
            Averaged image

        Yields
        ------
        ImgArray
            Subtomogram at each position
        """
        aligned = np.zeros(self.output_shape, dtype=np.float32)
        n = 0
        if classifier is None:
            classifier = lambda x: True
        for subvol in self.iter_subtomograms():
            if classifier(subvol):
                aligned += subvol.value
            n += 1
            yield aligned
        avg = ip.asarray(aligned / n, name="Average", axes="zyx")
        avg.set_scale(self.image)
        return avg

    def iter_align(
        self,
        template: ip.ImgArray,
        *,
        mask: ip.ImgArray = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        alignment_model: type[BaseAlignmentModel] = ZNCCAlignment,
        **align_kwargs,
    ) -> Generator[AlignmentResult, None, SubtomogramLoader]:
        """
        Create an iterator that align subtomograms to the template image.

        This method conduct so called "subtomogram alignment". Only shifts and rotations
        are calculated in this method. To get averaged image, you'll have to run "average"
        method using the resulting SubtomogramLoader instance.

        Parameters
        ----------
        template : ip.ImgArray, optional
            Template image.
        mask : ip.ImgArray, optional
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

        Yields
        ------
        AlignmentResult
            An tuple representing the current alignment result.
        """

        self._check_shape(template)

        local_shifts, local_rot, corr_max = _allocate(len(self))
        _max_shifts_px = np.asarray(max_shifts) / self.scale

        with ip.use("cupy"):
            model = alignment_model(
                template=template,
                mask=mask,
                **align_kwargs,
            )
            for i, subvol in enumerate(self.iter_subtomograms()):
                result = model.align(subvol, _max_shifts_px)
                _, local_shifts[i], local_rot[i], corr_max[i] = result
                yield result

        rotator = Rotation.from_quat(local_rot)
        mole_aligned = self.molecules.linear_transform(
            local_shifts * self.scale,
            rotator,
        )

        mole_aligned.features = update_features(
            self.molecules.features.copy(),
            get_features(corr_max, local_shifts, rotator.as_rotvec()),
        )

        return self.replace(molecules=mole_aligned)

    def iter_align_no_template(
        self,
        *,
        mask_params: np.ndarray | Callable[[np.ndarray], np.ndarray] | None = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        alignment_model: type[BaseAlignmentModel] = ZNCCAlignment,
        **align_kwargs,
    ) -> Generator[AlignmentResult, None, SubtomogramLoader]:
        """
        Create an iterator that align subtomograms without template image.

        A template-free version of :func:`iter_align`. This method first 
        calculates averaged image and uses it for the alignment template. To
        avoid loading same subtomograms twice, a memory-mapped array is created 
        internally (so the second subtomogram loading is faster).

        Parameters
        ----------
        template : ip.ImgArray, optional
            Template image.
        mask : ip.ImgArray, optional
            Mask image. Must in the same shap as the template.
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

        Yields
        ------
        AlignmentResult
            An tuple representing the current alignment result.
        """
        local_shifts, local_rot, corr_max = _allocate(len(self))
        _max_shifts_px = np.asarray(max_shifts) / self.scale
        all_subvols = yield from self.iter_to_memmap(path=None)

        template = all_subvols.proj("p").compute()

        # get mask image
        if isinstance(mask_params, np.ndarray):
            mask = mask_params
        elif callable(mask_params):
            mask = mask_params(template)
        else:
            mask = mask_params

        with ip.use("cupy"):
            model = alignment_model(
                template=template,
                mask=mask,
                **align_kwargs,
            )
            for i, subvol in enumerate(all_subvols):
                result = model.align(subvol.compute(), _max_shifts_px)
                _, local_shifts[i], local_rot[i], corr_max[i] = result
                yield result

        rotator = Rotation.from_quat(local_rot)
        mole_aligned = self.molecules.linear_transform(
            local_shifts * self.scale,
            rotator,
        )

        mole_aligned.features = update_features(
            self.molecules.features.copy(),
            get_features(corr_max, local_shifts, rotator.as_rotvec()),
        )

        return self.replace(molecules=mole_aligned)

    def iter_align_multi_templates(
        self,
        templates: list[ip.ImgArray],
        *,
        mask: ip.ImgArray | None = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        alignment_model: type[BaseAlignmentModel] = ZNCCAlignment,
        **align_kwargs,
    ) -> Generator[AlignmentResult, None, SubtomogramLoader]:
        """
        Create an iterator that align subtomograms with multiple template images.

        A multi-template version of :func:`iter_align`. This method calculate cross
        correlation for every template and uses the best local shift, rotation and
        template.

        Parameters
        ----------
        templates: list of ImgArray
            Template images.
        mask : ip.ImgArray, optional
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

        Yields
        ------
        AlignmentResult
            An tuple representing the current alignment result.
        """
        n_templates = len(templates)
        self._check_shape(templates[0])

        local_shifts, local_rot, corr_max = _allocate(len(self))

        # optimal template ID
        labels = np.zeros(len(self), dtype=np.uint32)

        _max_shifts_px = np.asarray(max_shifts) / self.scale
        with ip.use("cupy"):
            model = alignment_model(
                template=np.stack(list(templates), axis="p"),
                mask=mask,
                **align_kwargs,
            )

            for i, subvol in enumerate(self.iter_subtomograms()):
                result = model.align(subvol, _max_shifts_px)
                labels[i], local_shifts[i], local_rot[i], corr_max[i] = result
                yield result
        
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

    def iter_subtomoprops(
        self,
        template: ip.ImgArray = None,
        mask: ip.ImgArray = None,
        properties=(ip.zncc,),
    ) -> Generator[None, None, pd.DataFrame]:
        results = {
            f.__name__: np.zeros(len(self), dtype=np.float32) for f in properties
        }
        n = 0
        if mask is None:
            mask = 1
        template_masked = template * mask
        for i, subvol in enumerate(self.iter_subtomograms()):
            for _prop in properties:
                prop = _prop(subvol * mask, template_masked)
                results[_prop.__name__][i] = prop
            n += 1
            yield

        return pd.DataFrame(results)


class SubtomogramLoader(BaseLoader):
    """
    A class for efficient loading of subtomograms.

    A ``SubtomogramLoader`` instance is basically composed of two elements,
    an image and a Molecules object. A subtomogram is loaded by creating a
    local rotated Cartesian coordinate at a molecule and calculating mapping
    from the image to the subtomogram.

    Parameters
    ----------
    image : ImgArray, LazyImgArray, np.ndarray or da.core.Array
        Tomogram image.
    mole : Molecules
        Molecules object that defines position and rotation of subtomograms.
    output_shape : int or tuple of int
        Shape (in pixel) of output subtomograms.
    order : int, default is 1
        Interpolation order of subtomogram sampling.
        - 0 = Nearest neighbor
        - 1 = Linear interpolation
        - 3 = Cubic interpolation
    chunksize : int, optional
        Chunk size used when loading subtomograms. This parameter controls the
        number of subtomograms to be loaded at the same time. Larger chunk size
        results in better performance if adjacent subtomograms are near to each
        other.
    """

    def __init__(
        self,
        image: ip.ImgArray | ip.LazyImgArray | np.ndarray | da.core.Array,
        mole: Molecules,
        output_shape: int | tuple[int, int, int],
        order: int = 3,
        chunksize: int = 1,
    ) -> None:
        ndim = 3
        if not isinstance(image, (ip.ImgArray, ip.LazyImgArray)):
            if isinstance(image, np.ndarray) and image.ndim == ndim:
                image = ip.asarray(image, axes="zyx")
            elif isinstance(image, da.core.Array) and image.ndim == ndim:
                image = ip.LazyImgArray(image, axes="zyx")
            else:
                raise TypeError("'image' must be a 3D numpy ndarray like object.")
        self.image = image
        self._molecules = mole
        self._order = order
        self._chunksize = max(chunksize, 1)
        if isinstance(output_shape, int):
            output_shape = (output_shape,) * ndim
        else:
            output_shape = tuple(output_shape)
        self._output_shape = output_shape

    @property
    def chunksize(self) -> int:
        """Return the chunk size on subtomogram loading."""
        return self._chunksize

    def __iter__(self) -> Iterator[ip.ImgArray]:
        return self.iter_subtomograms()

    def replace(
        self,
        output_shape: int | tuple[int, int, int] | None = None,
        order: int | None = None,
        chunksize: int | None = None,
    ):
        """Return a new instance with different parameter(s)."""
        if output_shape is None:
            output_shape = self.output_shape
        if order is None:
            order = self.order
        if chunksize is None:
            chunksize = self.chunksize
        return self.__class__(
            self.image,
            self.molecules,
            output_shape=output_shape,
            order=order,
            chunksize=chunksize,
        )

    def iter_average_split(
        self,
        *,
        n_set: int = 1,
        seed: int | float | str | bytes | bytearray | None = 0,
    ):
        """
        Create an iterator that calculate the averaged images for split pairs.

        This method executes pairwise subtomogram averaging using randomly selected
        molecules, which is useful for calculation of such as Fourier shell
        correlation.

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

        Yields
        ------
        ImgArray
            Split images in a stack.
        """
        np.random.seed(seed)
        try:
            sum_images = ip.zeros((n_set, 2) + self.output_shape, dtype=np.float32)
            res = 0
            for subvols in self._iter_chunks():
                lc, res = divmod(len(subvols) + res, 2)
                for i_set in range(n_set):
                    np.random.shuffle(subvols)
                    sum_images[i_set, 0] += np.sum(subvols[:lc], axis=0)
                    sum_images[i_set, 1] += np.sum(subvols[lc:], axis=0)
                yield sum_images
        finally:
            np.random.seed(None)

        img = ip.asarray(sum_images, axes="pqzyx")
        img.set_scale(self.image)

        return img

    def average(
        self,
        *,
        classifier=None,
        callback: Callable[[SubtomogramLoader], Any] = None,
    ) -> ip.ImgArray:
        """
        A non-iterator version of :func:`iter_average`.

        Parameters
        ----------
        classifier : callable, optional
            If given, only those subvolumes that satisfy ``classifier(subvol) == True``
            will be collected.
        callback : callable, optional
            If given, ``callback(self)`` will be called for each iteration of subtomogram
            loading.

        Returns
        -------
        ImgArray
            Averaged image.
        """
        average_iter = self.iter_average(classifier=classifier)
        return self._resolve_iterator(average_iter, callback)

    def align(
        self,
        *,
        template: ip.ImgArray = None,
        mask: ip.ImgArray = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        rotations: Ranges | None = None,
        cutoff: float = 0.5,
        method: str = "zncc",
        callback: Callable[[SubtomogramLoader], Any] = None,
    ) -> SubtomogramLoader:
        """
        A non-iterator version of :func:`iter_align`.

        Parameters
        ----------
        template : ip.ImgArray, optional
            Template image.
        mask : ip.ImgArray, optional
            Mask image. Must in the same shae as the template.
        max_shifts : int or tuple of int, default is (1., 1., 1.)
            Maximum shift between subtomograms and template.
        rotations : (float, float) or three-tuple of (float, float) or None, optional
            Rotation between subtomograms and template in external Euler angles.
        cutoff : float, default is 0.5
            Cutoff frequency of low-pass filter applied in each subtomogram.
        callback : callable, optional
            Callback function that will get called after each iteration.

        Returns
        -------
        SubtomogramLoader
            Refined molecule object is bound.
        """

        align_iter = self.iter_align(
            template=template,
            mask=mask,
            max_shifts=max_shifts,
            rotations=rotations,
            cutoff=cutoff,
            method=method,
        )

        return self._resolve_iterator(align_iter, callback)

    def average_split(
        self,
        *,
        n_set: int = 1,
        seed: int | float | str | bytes | bytearray | None = 0,
        callback: Callable[[SubtomogramLoader], Any] = None,
    ) -> tuple[ip.ImgArray, ip.ImgArray]:
        """
        A non-iterator version of :func:`iter_average_split`.

        Parameters
        ----------
        n_set : int, default is 1
            Number of split set of averaged image.
        seed : random seed, default is 0
            Random seed to determine how subtomograms will be split.

        Returns
        -------
        ImgArray
            Averaged image.
        """
        it = self.iter_average_split(n_set=n_set, seed=seed)
        return self._resolve_iterator(it, callback)

    def fsc(
        self,
        mask: ip.ImgArray | None = None,
        seed: int | float | str | bytes | bytearray | None = 0,
        n_set: int = 1,
        dfreq: float = 0.05,
    ) -> pd.DataFrame:
        """
        Calculate Fourier shell correlation.

        Parameters
        ----------
        mask : ip.ImgArray, optional
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
            mask = 1
        else:
            self._check_shape(mask, "mask")

        img = self.average_split(n_set=n_set, seed=seed)
        fsc_all: dict[str, np.ndarray] = {}
        for i in range(n_set):
            img0, img1 = img[i]
            freq, fsc = ip.fsc(img0 * mask, img1 * mask, dfreq=dfreq)
            fsc_all[f"FSC-{i}"] = fsc

        df = pd.DataFrame({"freq": freq})
        return df.update(fsc_all)

    # def get_classifier(
    #     self,
    #     mask: ip.ImgArray | None = None,
    #     n_components: int = 5,
    #     n_clusters: int = 2,
    #     binsize: int = 1,
    #     seed: int | None = 0,
    # ) -> PcaClassifier:
    #     image_stack = self.to_stack(binsize=binsize)
    #     from .classification import PcaClassifier
    #     clf = PcaClassifier(
    #         image_stack,
    #         mask_image=mask,
    #         n_components=n_components,
    #         n_clusters=n_clusters,
    #         seed=seed,
    #     )
    #     return clf


    def _iter_chunks(self) -> Iterator[ip.ImgArray]:  # axes: pzyx
        """Generate subtomogram list chunk-wise."""
        image = self.image
        scale = image.scale.x

        for coords in self.molecules.iter_cartesian(
            self.output_shape, scale, self.chunksize
        ):
            with ip.use("cupy"):
                subvols = np.stack(
                    _utils.multi_map_coordinates(
                        image, coords, order=self.order, cval=np.mean
                    ),
                    axis=0,
                )
            subvols = ip.asarray(subvols, axes="pzyx")
            subvols.set_scale(image)
            yield subvols



class ChunkedSubtomogramLoader(SubtomogramLoader):
    ...


def get_features(corr_max, local_shifts, rotvec) -> pd.DataFrame:
    features = {
        "score": corr_max,
        Align.zShift: np.round(local_shifts[:, 0], 2),
        Align.yShift: np.round(local_shifts[:, 1], 2),
        Align.xShift: np.round(local_shifts[:, 2], 2),
        Align.zRotvec: np.round(rotvec[:, 0], 5),
        Align.yRotvec: np.round(rotvec[:, 1], 5),
        Align.xRotvec: np.round(rotvec[:, 2], 5),
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
