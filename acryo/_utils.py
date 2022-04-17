from __future__ import annotations
import numpy as np
from dask import array as da, delayed
from scipy import ndimage as ndi
from scipy.fft import fftn
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from scipy.spatial.transform import Rotation


def _make_slice_and_pad(
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

    return slice(z0, z1), (z0_pad, z1_pad), z0_pad != 0 or z1_pad != 0


def compose_matrices(
    center: Sequence[float] | np.ndarray,
    rotators: list[Rotation],
    output_center: Sequence[float] | np.ndarray | None = None,
):
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

    matrices = []
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
        *[np.fft.fftshift(np.fft.fftfreq(s)) for s in shape], indexing="ij"
    )

    r: np.ndarray = np.sqrt(sum(f ** 2 for f in freqs))

    # make radially separated labels
    labels = (r / dfreq).astype(np.uint16)
    nlabels = labels.max()

    out = np.empty(nlabels, dtype=np.float32)

    def radial_sum(arr):
        arr = np.asarray(arr)
        return ndi.sum_labels(arr, labels=labels, index=np.arange(0, nlabels))

    f0: np.ndarray = np.fft.fftshift(fftn(img0))
    f1: np.ndarray = np.fft.fftshift(fftn(img1))

    cov = f0.real * f1.real + f0.imag * f1.imag  # type: ignore
    pw0 = f0.real ** 2 + f0.imag ** 2  # type: ignore
    pw1 = f1.real ** 2 + f1.imag ** 2  # type: ignore

    out = radial_sum(cov) / np.sqrt(radial_sum(pw0) * radial_sum(pw1))
    freq = (np.arange(len(out)) + 0.5) * dfreq
    return freq, out


def bin_image(img: np.ndarray | da.core.Array, binsize: int) -> np.ndarray:
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
    img: da.core.Array,
    center: tuple[int, ...],
    output_shape,
    rot: Rotation,
    order: int = 1,
):
    output_center = np.array(output_shape) / 2 - 0.5
    slices: list[slice] = []
    pads: list[tuple[int, ...]] = []
    new_center: list[float] = []
    need_pad = False
    for c, s, s0 in zip(center, output_shape, img.shape):
        x0 = int(c - s / 2 - order)
        x1 = int(x0 + s + 2 * order + 1)
        _sl, _pad, _need_pad = _make_slice_and_pad(x0, x1, s0)
        slices.append(_sl)
        pads.append(_pad)
        new_center.append(c - x0)
        need_pad = need_pad or _need_pad

    img0 = img[tuple(slices)]
    if need_pad:
        input = da.pad(img0, pads)
        lower = np.array([l for l, r in pads])
        new_center += lower
        output_center += lower
    else:
        input = img0
    mtx = compose_matrices(new_center, [rot], output_center=output_center)[0]
    return input, mtx


def prepare_affine_cornersafe(
    img: da.core.Array,
    center: tuple[int, ...],
    shape,
    rot: Rotation,
    order: int = 1,
):
    max_len = np.sqrt(np.sum(np.asarray(shape, dtype=np.float32) ** 2))
    output_center = np.array(shape) / 2 - 0.5
    half_len = max_len / 2
    slices: list[slice] = []
    pads: list[tuple[int, ...]] = []
    new_center: list[float] = []
    need_pad = False
    for c, s0 in zip(center, img.shape):
        x0 = int(c - half_len - order)
        x1 = int(x0 + max_len + 2 * order + 1)
        _sl, _pad, _need_pad = _make_slice_and_pad(x0, x1, s0)
        slices.append(_sl)
        pads.append(_pad)
        new_center.append(c - x0)
        need_pad = need_pad or _need_pad

    img0 = img[tuple(slices)]
    if need_pad:
        input = da.pad(img0, pads)
        lower = np.array([l for l, r in pads])
        new_center += lower
        output_center += lower
    else:
        input = img0
    mtx = compose_matrices(new_center, [rot], output_center=output_center)[0]
    return input, mtx


@delayed
def rotated_crop(subimg: np.ndarray, mtx: np.ndarray, shape, order, mode, cval):
    if callable(cval):
        cval = cval(subimg)

    out = ndi.affine_transform(
        subimg,
        matrix=mtx,
        output_shape=shape,
        order=order,
        prefilter=order > 1,
        mode=mode,
        cval=cval,
    )
    return out
