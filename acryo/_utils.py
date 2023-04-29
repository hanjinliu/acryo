# pyright: reportPrivateImportUsage=false
from __future__ import annotations

import warnings
from typing import Callable, Sequence, TYPE_CHECKING, TypeVar
from functools import lru_cache, reduce, wraps

import numpy as np
from numpy.typing import NDArray
from dask import array as da
from dask.delayed import delayed
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation
from acryo._typed_scipy import fftn, rfftn, irfftn

if TYPE_CHECKING:
    from acryo._types import degree


def make_slice_and_pad(
    z0: int, z1: int, size: int
) -> tuple[slice, tuple[int, int], bool]:
    """
    Helper function for cropping images.

    This function calculates what slicing and padding are needed when an array
    is sliced by ``z0:z1``. Array must be padded when z0 is negative or z1 is
    outside the array size.
    """
    z0_pad = z1_pad = 0
    if z0 < 0:
        z0_pad = -z0
        z0 = 0
    elif size < z0:
        raise ValueError(f"Specified size is {size} but need to slice at {z0}:{z1}.")

    if size < z1:
        z1_pad = z1 - size
        z1 = size
    elif z1 < 0:
        raise ValueError(f"Specified size is {size} but need to slice at {z0}:{z1}.")

    out_of_bound = z0_pad != 0 or z1_pad != 0
    return slice(z0, z1), (z0_pad, z1_pad), out_of_bound


def compose_matrices(
    center: Sequence[float] | np.ndarray,
    rotators: Rotation | list[Rotation],
    output_center: Sequence[float] | np.ndarray | None = None,
) -> list[NDArray[np.float32]]:
    """Compose Affine matrices from an array shape and a Rotation object."""

    dz, dy, dx = center
    if output_center is None:
        output_center = center
    # center to corner
    translation_0 = np.array(
        [
            [1.0, 0.0, 0.0, dz],
            [0.0, 1.0, 0.0, dy],
            [0.0, 0.0, 1.0, dx],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    # corner to center
    dz, dy, dx = output_center
    translation_1 = np.array(
        [
            [1.0, 0.0, 0.0, -dz],
            [0.0, 1.0, 0.0, -dy],
            [0.0, 0.0, 1.0, -dx],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    matrices: list[np.ndarray] = []
    for rot in rotators:
        e_ = np.eye(4)
        e_[:3, :3] = rot.as_matrix()
        matrices.append(translation_0 @ e_ @ translation_1)
    return matrices


def fourier_shell_correlation(
    img0: np.ndarray,
    img1: np.ndarray,
    dfreq: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate Fourier shell correlation for resolution analysis.

    Parameters
    ----------
    img0 : np.ndarray
        First input image.
    img1 : np.ndarray
        Second input image.
    dfreq : float, default is 0.02
        Difference of sampling frequency.

    Returns
    -------
    np.ndarray and np.ndarray
        Frequency and FSC.
    """
    shape = img0.shape

    freqs = np.meshgrid(
        *[np.fft.fftshift(np.fft.fftfreq(s, d=1.0)) for s in shape], indexing="ij"
    )

    r: np.ndarray = np.sqrt(sum(f**2 for f in freqs))

    # make radially separated labels
    labels = (r / dfreq).astype(np.uint16)
    nlabels = labels.max()

    out = np.empty(nlabels, dtype=np.float32)

    def radial_sum(arr: NDArray[np.float32]) -> NDArray[np.float32]:
        arr = np.asarray(arr)
        return ndi.sum_labels(arr, labels=labels, index=np.arange(0, nlabels))

    f0 = np.fft.fftshift(fftn(img0))
    f1 = np.fft.fftshift(fftn(img1))

    cov = f0.real * f1.real + f0.imag * f1.imag
    pw0 = f0.real**2 + f0.imag**2
    pw1 = f1.real**2 + f1.imag**2
    out = radial_sum(cov) / np.sqrt(radial_sum(pw0) * radial_sum(pw1))
    freq = (np.arange(len(out)) + 0.5) * dfreq
    return freq, out


def bin_image(img: np.ndarray | da.Array, binsize: int) -> np.ndarray:
    """Bin an image."""
    _slices: list[slice] = []
    _shapes: list[int] = []
    for s in img.shape:
        npix, res = divmod(s, binsize)
        _slices.append(slice(None, s - res))
        _shapes.extend([npix, binsize])
    slices = tuple(_slices)
    shapes = tuple(_shapes)
    img_reshaped = np.reshape(img[slices], shapes)
    axis = tuple(i * 2 + 1 for i in range(img.ndim))
    return np.sum(img_reshaped, axis=axis)


def prepare_affine(
    img: da.Array,
    center: Sequence[float],
    output_shape: Sequence[int],
    rot: Rotation,
    order: int = 1,
) -> tuple[da.Array, NDArray[np.float32]]:
    output_center = np.array(output_shape) / 2 - 0.5
    slices: list[slice] = []
    pads: list[tuple[int, ...]] = []
    new_center: list[float] = []
    need_pad = False
    for c, s, s0 in zip(center, output_shape, img.shape):
        x0 = int(c - s / 2 - order)
        x1 = int(x0 + s + 2 * order + 1)
        _sl, _pad, _need_pad = make_slice_and_pad(x0, x1, s0)
        slices.append(_sl)
        pads.append(_pad)
        new_center.append(c - x0)
        need_pad = need_pad or _need_pad

    img0 = img[tuple(slices)]
    if need_pad:
        input = da.pad(img0, pads, mode="mean")
    else:
        input = img0
    mtx = compose_matrices(new_center, [rot], output_center=output_center)[0]
    return input, mtx


def prepare_affine_cornersafe(
    img: da.Array,
    center: Sequence[float],
    output_shape: Sequence[int],
    rot: Rotation,
    order: int = 1,
) -> tuple[da.Array, NDArray[np.float32]]:
    max_len = np.sqrt(np.sum(np.asarray(output_shape, dtype=np.float32) ** 2))
    output_center = np.array(output_shape) / 2 - 0.5
    half_len = max_len / 2
    slices: list[slice] = []
    pads: list[tuple[int, ...]] = []
    new_center: list[float] = []
    need_pad = False
    for c, s0 in zip(center, img.shape):
        x0 = int(c - half_len - order)
        x1 = int(x0 + max_len + 2 * order + 1)
        _sl, _pad, _need_pad = make_slice_and_pad(x0, x1, s0)
        slices.append(_sl)
        pads.append(_pad)
        new_center.append(c - x0)
        need_pad = need_pad or _need_pad

    img0 = img[tuple(slices)]
    if need_pad:
        input = da.pad(img0, pads, mode="mean")
    else:
        input = img0
    mtx = compose_matrices(new_center, [rot], output_center=output_center)[0]
    return input, mtx


_delayed_affine_transform = delayed(ndi.affine_transform)


def delayed_affine(
    input: np.ndarray,
    matrix: np.ndarray,
    order: int = 3,
    mode: str = "constant",
    cval: float = 0.0,
    prefilter: bool = True,
) -> da.Array:
    out = _delayed_affine_transform(
        input, matrix, order=order, mode=mode, cval=cval, prefilter=prefilter
    )
    return da.from_delayed(out, shape=input.shape, dtype=input.dtype)  # type: ignore


def missing_wedge_mask(
    rotator: Rotation,
    tilt_range: tuple[degree, degree],
    shape: tuple[int, int, int],
) -> NDArray[np.float32]:
    """
    Create a binary mask that covers tomographical missing wedge.

    Note that the mask is not shifted to the center of the Fourier domain.
    ``np.fft.fftn(img) * mask`` will be the correct way to apply the mask.

    Parameters
    ----------
    rotator : Rotation
        The rotation object that describes the direction of the mask.
    tilt_range : tuple of float
        The range of tilt angles in degrees.
    shape : tuple of int
        The shape of the mask.

    Returns
    -------
    np.ndarray or float
        Missing wedge mask. If ``tilt_range`` is None, 1 will be returned.
    """
    normal0, normal1 = _get_unrotated_normals(tilt_range)
    shape_vector = np.array(shape, dtype=np.float32)
    rotator_inv = rotator.inv()
    normal0 = rotator_inv.apply(normal0 * shape_vector)
    normal1 = rotator_inv.apply(normal1 * shape_vector)
    vectors = _get_indices(shape)
    dot0 = vectors.dot(normal0)
    dot1 = vectors.dot(normal1)
    missing = dot0 * dot1 <= 0
    return missing


@lru_cache(maxsize=8)
def _get_unrotated_normals(
    tilt_range: tuple[degree, degree]
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    radmin, radmax = np.deg2rad(tilt_range)
    ang0 = np.pi - radmin
    ang1 = np.pi - radmax
    return (
        np.array([np.cos(ang0), 0, np.sin(ang0)], dtype=np.float32),
        np.array([np.cos(ang1), 0, np.sin(ang1)], dtype=np.float32),
    )


@lru_cache(maxsize=32)
def _get_indices(shape: tuple[int, ...]) -> NDArray[np.float32]:
    inds = np.indices(shape, dtype=np.float32)
    for ind, s in zip(inds, shape):
        # Note that the shifts in indices must resemble the shifts in fftshift.
        ind -= np.ceil(s / 2)
    return np.fft.fftshift(np.stack(list(inds), axis=-1), axes=(0, 1, 2))


# lowpass filter
# Modified from skimage.filters._fft_based
def lowpass_filter_ft(
    img: NDArray[np.float32], cutoff: float, order: int = 2
) -> NDArray[np.complex64]:
    """Apply a low-pass filter and return the result in Fourier space."""
    if cutoff >= 0.5 * np.sqrt(img.ndim) or cutoff <= 0:
        return fftn(img)
    weight = _get_ND_butterworth_filter(
        img.shape,
        cutoff,
        order,
        real=False,
    )
    return weight * fftn(img)


def lowpass_filter(
    img: NDArray[np.float32], cutoff: float, order: int = 2
) -> NDArray[np.float32]:
    """Apply a low-pass filter and return the result in real space."""
    if cutoff >= 0.5 * np.sqrt(img.ndim) or cutoff <= 0:
        return img
    weight = _get_ND_butterworth_filter(
        img.shape,
        cutoff,
        order,
        real=True,
    )
    out = irfftn(weight * rfftn(img))
    return out.real


def highpass_filter_ft(
    img: NDArray[np.float32], cutoff: float, order: int = 2
) -> NDArray[np.complex64]:
    """Apply a high-pass filter and return the result in Fourier space."""
    if cutoff >= 0.5 * np.sqrt(img.ndim) or cutoff <= 0:
        return np.zeros_like(img)
    weight = 1 - _get_ND_butterworth_filter(
        img.shape,
        cutoff,
        order,
        real=False,
    )
    return weight * fftn(img)


def highpass_filter(
    img: NDArray[np.float32], cutoff: float, order: int = 2
) -> NDArray[np.float32]:
    """Apply a high-pass filter and return the result in real space."""
    if cutoff >= 0.5 * np.sqrt(img.ndim) or cutoff <= 0:
        return np.zeros_like(img)
    weight = 1 - _get_ND_butterworth_filter(
        img.shape,
        cutoff,
        order,
        real=True,
    )
    out = irfftn(weight * rfftn(img))
    return out.real


@lru_cache(maxsize=4)
def _get_ND_butterworth_filter(
    shape: tuple[int, ...],
    cutoff: float,
    order: int,
    real: bool,
) -> NDArray[np.float32]:
    ranges = []
    for d in shape:
        axis = np.arange(-(d - 1) // 2, (d - 1) // 2 + 1, dtype=np.float32) / (
            d * cutoff
        )
        ranges.append(np.fft.ifftshift(axis**2))
    if real:
        limit = shape[-1] // 2 + 1
        ranges[-1] = ranges[-1][:limit]
    q2 = reduce(np.add, np.meshgrid(*ranges, indexing="ij", sparse=True))
    wfilt = 1 / (1 + q2**order)
    return wfilt


_F = TypeVar("_F", bound=Callable)


def deprecated_kwarg(old: str, new: str) -> Callable[[_F], _F]:
    def _inner(f):
        @wraps(f)
        def _func(*args, **kwargs):
            if old in kwargs:
                warnings.warn(
                    f"`{old}` is deprecated, use `{new}` instead",
                    DeprecationWarning,
                )
                kwargs[new] = kwargs.pop(old)
            return f(*args, **kwargs)

        return _func

    return _inner  # type: ignore
