from __future__ import annotations
from typing import (
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
    ndim = 3
    if not isinstance(image, (ip.ImgArray, ip.LazyImgArray)):
        if isinstance(image, np.ndarray) and image.ndim == ndim:
            image = ip.asarray(image, axes="zyx")
        elif isinstance(image, da.core.Array) and image.ndim == ndim:
            image = ip.LazyImgArray(image, axes="zyx")
        else:
            raise TypeError("'image' must be a 3D numpy ndarray like object.")
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
        image: ip.ImgArray | ip.LazyImgArray,
        molecules: Molecules,
        output_shape: int | tuple[int, int, int],
        order: int = 3,
    ) -> None:
        ndim = 3
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
    
    def get_subtomogram(self, i: int) -> ip.ImgArray:
        if self._cached_lazy_imgarray is not None:
            return self._cached_lazy_imgarray[i].compute()
        image = self.image
        scale = self.scale
        coords = self.molecules.cartesian_at(i, self.output_shape, scale)
        subvol = np.stack(
            _utils.map_coordinates(
                image, coords, order=self.order, cval=np.mean
            ),
            axis=0,
        )
        subvol = ip.asarray(subvol, axes="zyx")
        subvol.set_scale(image)
        return subvol
    
    def _get_subtomogram_4d(self, i: int) -> ip.ImgArray:
        sv = self.get_subtomogram(i)
        subvol = sv.value[np.newaxis]
        subvol = ip.asarray(subvol, axes="pzyx")
        subvol.set_scale(sv)
        return subvol
    
    def iter_subtomograms(self) -> Iterator[ip.ImgArray]:
        for i in range(len(self)):
            yield self.get_subtomogram(i)

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
        if self._cached_lazy_imgarray is not None:
            return self._cached_lazy_imgarray.value
        delayed_get_subtomogram = delayed(self._get_subtomogram_4d)
        
        arrays = [
            da.from_delayed(
                delayed_get_subtomogram(i),
                shape=(1,) + self.output_shape,
                dtype=np.float32,
            )
            for i in range(len(self))
        ]
        return da.concatenate(arrays, axis=0)
    
    def construct_lazyimgarray(self) -> ip.LazyImgArray:
        arr = ip.aslazy(self.construct_dask(), axes="zyx")
        return arr.set_scale(self.image)

        
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
            mmap, chunks=(1,) + self.output_shape, meta=np.array([], dtype=np.float32)
        )
        arr = ip.aslazy(darr, name="Subtomograms", axes="pzyx")
        arr.set_scale(self.image)
        self._cached_lazy_imgarray = arr
        return arr

    def to_stack(self, binsize: int = 1) -> ip.ImgArray:
        """Create a 4D image stack of all the subtomograms."""
        arr = self.construct_lazyimgarray()
        if binsize > 1:
            arr = arr.binning(binsize, check_edges=False)
        return arr.compute()

    def average(
        self,
    ) -> ip.ImgArray:
        """
        Calculate the average of subtomograms.

        This function execute so-called "subtomogram averaging". The size of
        subtomograms is determined by the ``self.output_shape`` attribute.

        Returns
        -------
        ImgArray
            Averaged image
        """
        dask_array = self.construct_dask()
        avg: ip.ImgArray = da.compute(
            da.mean(dask_array, axis=0),
        )[0]
        
        avg.axes = "zyx"
        avg.set_scale(self.image)
        return avg
    
    def average_split(
        self,
        n_set: int = 1,
        seed: int = 0,
        squeeze: bool = True,
    ) -> ip.ImgArray:
        """
        Split subtomograms into two set and average separately.
        
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
        """
        nmole = len(self)
        np.random.seed(seed)
        tasks = []
        for _ in range(n_set):
            sl = np.random.choice(np.arange(nmole), nmole//2).tolist()
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
        ip_stack = ip.asarray(stack, name="split average", axes="pqzyx")
        ip_stack.set_scale(self.image)
        if squeeze and n_set == 1:
            ip_stack = ip_stack[0]
        return ip_stack

    def align(
        self,
        template: ip.ImgArray,
        *,
        mask: ip.ImgArray = None,
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
            local_shifts * self.scale, rotator,
        )

        mole_aligned.features = update_features(
            self.molecules.features.copy(),
            get_features(scores, local_shifts, rotator.as_rotvec()),
        )

        return self.replace(molecules=mole_aligned)
        
    def align_no_template(
        self,
        *,
        mask_params: np.ndarray | Callable[[np.ndarray], np.ndarray] | None = None,
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
        """
        all_subvols = self.create_cache()

        template = all_subvols.proj("p").compute()

        # get mask image
        if isinstance(mask_params, np.ndarray):
            mask = mask_params
        elif callable(mask_params):
            mask = mask_params(template)
        else:
            mask = mask_params
        return self.align(
            template,
            mask=mask,
            max_shifts=max_shifts,
            alignment_model=alignment_model,
            **align_kwargs
        )

    def align_multi_templates(
        self,
        templates: list[ip.ImgArray],
        *,
        mask: ip.ImgArray | None = None,
        max_shifts: nm | tuple[nm, nm, nm] = 1.0,
        alignment_model: type[BaseAlignmentModel] = ZNCCAlignment,
        **align_kwargs,
    ) -> Generator[AlignmentResult, None, Self]:
        """
        Align subtomograms with multiple template images.

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
        
        model = alignment_model(
            template=np.stack(list(templates), axis="p"),
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

        img = self.average_split(n_set=n_set, seed=seed, squeeze=False)
        fsc_all: dict[str, np.ndarray] = {}
        for i in range(n_set):
            img0, img1 = img[i]
            freq, fsc = ip.fsc(img0 * mask, img1 * mask, dfreq=dfreq)
            fsc_all[f"FSC-{i}"] = fsc

        df = pd.DataFrame({"freq": freq})
        return pd.concat([df, fsc_all], axis=1)


class ChunkedSubtomogramLoader(SubtomogramLoader):
    
    def __init__(
        self,
        image: ip.ImgArray | ip.LazyImgArray | np.ndarray | da.core.Array,
        molecules: Molecules,
        output_shape: int | tuple[int, int, int],
        order: int = 3,
        chunksize: int = 100,
    ) -> None:
        super().__init__(image, molecules, output_shape, order)
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
            molecules=molecules,
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
