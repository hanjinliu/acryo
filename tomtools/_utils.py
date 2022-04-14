from __future__ import annotations
import numpy as np
from dask import array as da
from scipy import ndimage as ndi
from scipy.fft import fftn
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from scipy.spatial.transform import Rotation


def map_coordinates(
    input: np.ndarray | da.core.Array,
    coordinates: np.ndarray,
    order: int = 3,
    mode: str = "constant",
    cval: float | Callable[[np.ndarray], float] = 0.0,
) -> np.ndarray:
    """
    Crop image at the edges of coordinates before calling map_coordinates to avoid
    loading entire array into memory.
    """
    coordinates = coordinates.copy()
    shape = input.shape
    sl = []
    for i in range(input.ndim):
        imin = int(np.min(coordinates[i])) - order
        imax = int(np.ceil(np.max(coordinates[i]))) + order + 1
        _sl, _pad = make_slice_and_pad(imin, imax, shape[i])
        sl.append(_sl)
        coordinates[i] -= _sl.start

    img = input[tuple(sl)]
    if isinstance(img, da.core.Array):
        img = img.compute()
    if callable(cval):
        cval = cval(img)

    return ndi.map_coordinates(
        img,
        coordinates=coordinates,
        order=order,
        mode=mode,
        cval=cval,
        prefilter=order > 1,
    )


def multi_map_coordinates(
    input: np.ndarray | da.core.Array,
    coordinates: np.ndarray,
    order: int = 3,
    mode: str = "constant",
    cval: float | Callable[[np.ndarray], float] = 0.0,
) -> np.ndarray:
    """
    Multiple map-coordinate in parallel.

    Result of this function is identical to following code.

    .. code-block:: python

        outputs = []
        for i in range(len(coordinates)):
            out = ndi.map_coordinates(input, coordinates[i], ...)
            outputs.append(out)

    """
    shape = input.shape
    coordinates = coordinates.copy()

    if coordinates.ndim != input.ndim + 2:
        if coordinates.ndim == input.ndim + 1:
            coordinates = coordinates[np.newaxis]
        else:
            raise ValueError(f"Coordinates have wrong dimension: {coordinates.shape}.")

    sl = []
    for i in range(coordinates.shape[1]):
        imin = int(np.min(coordinates[:, i])) - order
        imax = int(np.ceil(np.max(coordinates[:, i]))) + order + 1
        _sl, _pad = make_slice_and_pad(imin, imax, shape[i])
        sl.append(_sl)
        coordinates[:, i] -= _sl.start

    img = input[tuple(sl)]
    if isinstance(img, da.core.Array):
        img = img.compute()
    if callable(cval):
        cval = cval(img)
    input_img = img

    imgs: list[np.ndarray] = []
    for crds in coordinates:
        imgs.append(
            ndi.map_coordinates(
                input_img,
                crds,
                mode=mode,
                cval=cval,
                order=order,
                prefilter=order > 1,
            )
        )

    return np.stack(imgs, axis=0)


def make_slice_and_pad(z0: int, z1: int, size: int) -> tuple[slice, tuple[int, int]]:
    """
    This function calculates what slicing and padding are needed when an array is sliced
    by ``z0:z1``. Array must be padded when z0 is negative or z1 is outside the array size.
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

    return slice(z0, z1), (z0_pad, z1_pad)


def compose_matrices(
    shape: tuple[int, int, int],
    rotators: list[Rotation],
):
    dz, dy, dx = (np.array(shape) - 1) / 2
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
    tuple[np.ndarray, np.ndarray]
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
