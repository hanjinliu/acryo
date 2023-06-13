from __future__ import annotations

from pathlib import Path
from typing import Iterable, Union

import numpy as np
from numpy.typing import NDArray
from acryo.pipe._curry import provider_function
from acryo._reader import REG
from acryo._typed_scipy import zoom
from acryo._types import nm

PathLike = Union[str, Path, bytes]


@provider_function
def from_file(
    scale: nm,
    path: PathLike,
    original_scale: float | None = None,
    tol: float = 0.01,
) -> NDArray[np.float32]:
    """
    An image provider function with rescaling.

    This function will provide a subtomogram loader with a resized image from a file.
    Will be used for the template images or the mask images.

    >>> loader.align(
    ...     template=from_file("path/to/template.mrc"),
    ...     mask=from_file("path/to/mask.mrc"),
    ... )

    Parameters
    ----------
    path : path-like
        Path to the image.
    original_scale : float, optional
        If given, this value will be used as the image scale (nm/pixel) instead
        of the scale extracted from the image metadata.
    tol : float
        Tolerance of the scale difference. If the relative difference is smaller than
        this, the image will not be resized.
    """
    if original_scale is not None and original_scale <= 0:
        raise ValueError("original_scale must be positive")
    img, img_scale = REG.imread_array(str(path))
    if original_scale is None:
        original_scale = img_scale
    ratio = original_scale / scale
    if abs(ratio - 1) < tol:
        return img
    return zoom(img, ratio, order=3, prefilter=True, mode="reflect")


@provider_function
def from_files(
    scale: nm,
    path: Iterable[PathLike],
    original_scale: float | None = None,
    tol: float = 0.01,
) -> list[NDArray[np.float32]]:
    """
    Batch image provider function with rescaling.

    This function will provide a subtomogram loader with resized images from files.
    Will be used for the template images.

    >>> from glob import glob
    >>> loader.align(
    ...     template=from_files(glob("path/to/template_*.mrc")),
    ...     mask=from_file("path/to/mask.mrc"),
    ... )

    Parameters
    ----------
    paths : iterable of path-like
        Paths to the image.
    original_scale : float, optional
        If given, this value will be used as the image scale (nm/pixel) instead
        of the scale extracted from the image metadata.
    tol : float
        Tolerance of the scale difference. If the relative difference is smaller than
        this, the image will not be resized.
    """
    return [from_file(p, original_scale, tol).provide(scale) for p in path]


@provider_function
def from_gaussian(
    scale: nm,
    shape: tuple[nm, nm, nm],
    sigma: nm | tuple[nm, nm, nm] = 1.0,
    shift: tuple[nm, nm, nm] = (0.0, 0.0, 0.0),
) -> NDArray[np.float32]:
    """
    An image provider function by a Gaussian function.

    This function will provide a Gaussian particle with given shape, sigma and shift from
    the center.

    >>> loader.align(
    ...     template=from_gaussian(shape=(4.8, 4.8, 4.8), sigma=1.2),
    ...     mask=from_file("path/to/mask.mrc"),
    ... )

    Parameters
    ----------
    shape : float or tuple of float
        Shape of the output image in nm.
    sigma : float or tuple of float
        Standard deviation of the Gaussian particle in nm.
    shift : tuple of float, optional
        Shift of the Gaussian particle from the center in nm.
    """
    sigma_px = _as_3_array(sigma) / scale
    shape_subpix = _as_3_array(shape) / scale
    center_subpix = shape_subpix + np.array(shift) / scale
    shape_px = tuple(np.round(shape_subpix).astype(np.int32))

    crds = np.indices(shape_px, dtype=np.float32)

    return np.exp(
        -0.5
        * sum((xx - c) / sg for xx, c, sg in zip(crds, center_subpix, sigma_px)) ** 2
    )


@provider_function
def from_array(
    scale: nm,
    img: NDArray[np.float32],
    original_scale: float = 1.0,
    tol: float = 0.01,
) -> NDArray[np.float32]:
    """
    An image provider function using existing image array.

    This function will provide a subtomogram loader with a resized image from an array.
    Will be used for the template images or the mask images.

    >>> loader.align(
    ...     template=from_array(img, original_scale=0.28),
    ...     mask=from_file("path/to/mask.mrc"),
    ... )

    Parameters
    ----------
    img : np.ndarray
        Input image array. Must be 3D.
    original_scale : float, optional
        If given, this value will be used as the image scale (nm/pixel) instead
        of the scale extracted from the image metadata.
    tol : float
        Tolerance of the scale difference. If the relative difference is smaller than
        this, the image will not be resized.
    """
    if original_scale is not None and original_scale <= 0:
        raise ValueError("original_scale must be positive")
    if img.ndim != 3:
        raise ValueError("img must be 3D")
    ratio = original_scale / scale
    if abs(ratio - 1) < tol:
        return img
    out = zoom(img, ratio, order=3, prefilter=True, mode="reflect")
    if not out.dtype == np.float32:
        out = out.astype(np.float32)
    return out


@provider_function
def from_arrays(
    scale: nm,
    imgs: Iterable[NDArray[np.float32]],
    original_scale: float = 1.0,
    tol: float = 0.01,
) -> list[NDArray[np.float32]]:
    return [from_array(img, original_scale, tol).provide(scale) for img in imgs]


@provider_function
def from_atoms(
    scale: nm,
    atoms: np.ndarray,
    weights: np.ndarray | None = None,
    center: tuple[nm, nm, nm] | None = None,
) -> NDArray[np.float32]:
    """
    An image provider function using a point cloud.

    Given an array of atoms, such as data extracted from a PDB file, this function
    can generate a 3D image of the atoms by simply building a histogram.

    Parameters
    ----------
    atoms : (N, 3) array
        Atoms coordinates in nanometer.
    weights : np.ndarray, optional
        weights of the atoms.
    center : tuple of float, optional
        Coordinates of the image center. If not given, the geometric center of the atoms
        will be used.
    """
    if atoms.ndim != 2 or atoms.shape[1] != 3:
        raise ValueError("atoms must be a 2D array with shape (n, 3)")
    if center is None:
        center = np.mean(atoms, axis=0)
    _center = np.asarray(center)[np.newaxis]
    coords = (atoms - _center) / scale
    rmax = np.max(np.sqrt(np.sum(coords**2, axis=1)))  # the furthest in pixels
    size = int(np.ceil(rmax * 2))
    lims = (-size / 2, size / 2)

    counts, _ = np.histogramdd(
        coords, bins=(size,) * 3, range=(lims,) * 3, weights=weights
    )
    return counts


def _as_3_array(x) -> NDArray[np.floating]:
    if np.isscalar(x):
        return np.array([x, x, x])  # type: ignore
    return np.asarray(x)
