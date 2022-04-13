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
import impy as ip
from dask import array as da, delayed
from .alignment import BaseAlignmentModel, ZNCCAlignment, SupportRotation
from .molecules import Molecules
from . import _utils
from .const import Align

if TYPE_CHECKING:
    from .alignment import AlignmentResult
    from typing_extensions import Self

nm = float  # alias


def subtomogram_loader(
    image: ip.ImgArray | ip.LazyImgArray | np.ndarray | da.core.Array,
    molecules: Molecules,
    output_shape: int | tuple[int, int, int],
    order: int = 3,
    chunksize: int = 1,
):
    if chunksize == 1:
        return SubtomogramLoader(image, molecules, output_shape, order)
    else:
        return ChunkedSubtomogramLoader(image, molecules, output_shape, order, chunksize)

class SubtomogramLoader:
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
    """
    def __init__(
        self,
        image: ip.ImgArray | ip.LazyImgArray | np.ndarray | da.core.Array,
        molecules: Molecules,
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
        self._molecules = molecules
        self._order = order
        if isinstance(output_shape, int):
            output_shape = (output_shape,) * ndim
        else:
            output_shape = tuple(output_shape)
        self._output_shape = output_shape
        self._cached_lazy_imgarray = None

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
    ) -> Self:
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
        if self._cached_lazy_imgarray is not None:
            return self._cached_lazy_imgarray[i].compute()
        image = self.image
        scale = self.scale
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

    def construct_dask(self) -> da.core.Array:
        from dask import delayed
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
        

    def iter_create_cache(self, path: str | None = None):
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
        arr = ip.LazyImgArray(darr, name="Subtomograms", axes="pzyx")
        arr.set_scale(self.image)
        self._cached_lazy_imgarray = arr
        return arr
        
    def construct_map(self, f: Callable, *args, **kwargs):
        dask_array = self.construct_dask()
        delayed_f = delayed(f)
        tasks = [delayed_f(ar, *args, **kwargs) for ar in dask_array]
        return tasks

    def create_cache(self, path: str | None = None) -> ip.LazyImgArray:
        """
        An non-iterator version of :func:`iter_create_cache`.

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
        it = self.iter_create_cache(path)
        return self._resolve_iterator(it, lambda x: None)

    def to_stack(self, binsize: int = 1) -> ip.ImgArray:
        """Create a 4D image stack of all the subtomograms."""
        images = list(self.iter_subtomograms(binsize=binsize))
        stack: ip.ImgArray = np.stack(images, axis="p")
        stack.set_scale(xyz=self.scale * binsize)
        return stack

    def iter_average(
        self,
    ) -> Generator[ip.ImgArray, None, ip.ImgArray]:
        """
        Create an iterator that calculate the averaged image from a tomogram.

        This function execute so-called "subtomogram averaging". The size of
        subtomograms is determined by the ``self.output_shape`` attribute.

        Returns
        -------
        ImgArray
            Averaged image

        Yields
        ------
        ImgArray
            Subtomogram at each position
        """
        sum_img = np.zeros(self.output_shape, dtype=np.float32)
        for subvol in self.iter_subtomograms():
            sum_img += subvol.value
            yield sum_img

        avg = ip.asarray(sum_img / len(self), name="Average", axes="zyx")
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
    ) -> Generator[AlignmentResult, None, Self]:
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

        local_shifts, local_rot, scores = _allocate(len(self))
        _max_shifts_px = np.asarray(max_shifts) / self.scale

        with ip.use("cupy"):
            model = alignment_model(
                template=template,
                mask=mask,
                **align_kwargs,
            )
            for i, subvol in enumerate(self.iter_subtomograms()):
                result = model.align(subvol, _max_shifts_px)
                _, local_shifts[i], local_rot[i], scores[i] = result
                yield result

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

    def parallel_average(
        self,
    ) -> ip.ImgArray:
        with ip.use("cupy"):
            dask_array = self.construct_dask()
        return da.mean(dask_array, axis=0).compute()
        

    def parallel_align(
        self,
        template: ip.ImgArray,
        *,
        mask: ip.ImgArray = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        alignment_model: type[BaseAlignmentModel] = ZNCCAlignment,
        **align_kwargs,
    ) -> Generator[AlignmentResult, None, Self]:
        
        self._check_shape(template)

        _max_shifts_px = np.asarray(max_shifts) / self.scale

        with ip.use("cupy"):
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
            local_shifts * self.scale, rotator,
        )

        mole_aligned.features = update_features(
            self.molecules.features.copy(),
            get_features(scores, local_shifts, rotator.as_rotvec()),
        )

        return self.replace(molecules=mole_aligned)
        
    def iter_align_no_template(
        self,
        *,
        mask_params: np.ndarray | Callable[[np.ndarray], np.ndarray] | None = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        alignment_model: type[BaseAlignmentModel] = ZNCCAlignment,
        **align_kwargs,
    ) -> Generator[AlignmentResult, None, Self]:
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
        all_subvols = yield from self.iter_create_cache(path=None)

        template = all_subvols.proj("p").compute()

        # get mask image
        if isinstance(mask_params, np.ndarray):
            mask = mask_params
        elif callable(mask_params):
            mask = mask_params(template)
        else:
            mask = mask_params
        out = yield from self.iter_align(
            tmeplate=template,
            mask=mask,
            max_shifts=max_shifts,
            alignment_model=alignment_model,
            **align_kwargs,
        )
        return out

    def iter_align_multi_templates(
        self,
        templates: list[ip.ImgArray],
        *,
        mask: ip.ImgArray | None = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        alignment_model: type[BaseAlignmentModel] = ZNCCAlignment,
        **align_kwargs,
    ) -> Generator[AlignmentResult, None, Self]:
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
        callback: Callable[[Self], Any] = None,
    ) -> ip.ImgArray:
        """
        A non-iterator version of :func:`iter_average`.

        Parameters
        ----------
        callback : callable, optional
            If given, ``callback(self)`` will be called for each iteration of subtomogram
            loading.

        Returns
        -------
        ImgArray
            Averaged image.
        """
        average_iter = self.iter_average()
        return self._resolve_iterator(average_iter, callback)

    def align(
        self,
        *,
        template: ip.ImgArray = None,
        mask: ip.ImgArray = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        alignment_model: type[BaseAlignmentModel] = ZNCCAlignment,
        callback: Callable[[Self], Any] = None,
        **align_kwargs,
    ) -> Self:
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
            alignment_model=alignment_model,
            **align_kwargs,
        )

        return self._resolve_iterator(align_iter, callback)

    def average_split(
        self,
        *,
        n_set: int = 1,
        seed: int | float | str | bytes | bytearray | None = 0,
        callback: Callable[[Self], Any] = None,
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

class ChunkedSubtomogramLoader(SubtomogramLoader):
    
    def __init__(
        self,
        image: ip.ImgArray | ip.LazyImgArray | np.ndarray | da.core.Array,
        mole: Molecules,
        output_shape: int | tuple[int, int, int],
        order: int = 3,
        chunksize: int = 100,
    ) -> None:
        super().__init__(image, mole, output_shape, order)
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
        chunksize: int | None = None,
    ) -> Self:
        """Return a new instance with different parameter(s)."""
        if molecules is None:
            molecules = self.molecules
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
    
    def iter_subtomograms(self) -> Iterator[ip.ImgArray]:
        for subvols in self.get_subtomogram_chunk():
            for subvol in subvols:
                yield subvol

    def get_subtomogram_chunk(self, chunk_index: int) -> Iterator[ip.ImgArray]:  # axes: pzyx
        """Generate subtomogram list chunk-wise."""
        image = self.image
        sl = slice(chunk_index*self.chunksize, (chunk_index + 1) * self.chunksize)
        coords = self.molecules.cartesian_at(sl, self.output_shape, self.scale)
        with ip.use("cupy"):
            subvols = _utils.multi_map_coordinates(
                    image, coords, order=self.order, cval=np.mean
                )
        subvols = ip.asarray(subvols, axes="pzyx")
        subvols.set_scale(image)
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
