"""Normalized cross correlation of 2D images, such as tilt images and projections.

The 2D versions of the functions in ``_zncc.py``. They are computed in batches over the
leading dimensions of numpy arrays.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import fft as sfft

_EPS = 1e-12  # floor of the denominators


def ncc_2d(
    img0: NDArray[np.floating], img1: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Normalized cross correlation (..., H, W) -> (...) of the images."""
    img0 = np.asarray(img0, dtype=np.float64)
    img1 = np.asarray(img1, dtype=np.float64)
    num = (img0 * img1).sum(axis=(-2, -1))
    denom = np.sqrt((img0**2).sum(axis=(-2, -1)) * (img1**2).sum(axis=(-2, -1)))
    return num / np.maximum(denom, _EPS)


def zncc_2d(
    img0: NDArray[np.floating], img1: NDArray[np.floating]
) -> NDArray[np.float64]:
    """Zero-mean normalized cross correlation (..., H, W) -> (...) of the images."""
    img0 = np.asarray(img0, dtype=np.float64)
    img1 = np.asarray(img1, dtype=np.float64)
    return ncc_2d(
        img0 - img0.mean(axis=(-2, -1), keepdims=True),
        img1 - img1.mean(axis=(-2, -1), keepdims=True),
    )


def zncc_landscape_2d(
    ft0: NDArray[np.complexfloating],
    ft1: NDArray[np.complexfloating],
    radius: int,
) -> NDArray[np.float32]:
    """ZNCC maps (..., 2 * radius + 1, 2 * radius + 1) of the shifts of the images.

    ``ft0`` and ``ft1`` are the (..., n, n // 2 + 1) rfft of square images. The images
    are circularly shifted, and normalized by their norms over the whole images. The
    center of the maps is no shift, and the value at the shift ``(dy, dx)`` is the
    correlation of the first image with the second image shifted by ``(dy, dx)``. The
    center is the same as ``zncc_2d`` of the images.
    """
    size = ft0.shape[-2]
    # zero-mean images
    ft0 = ft0.copy()
    ft1 = ft1.copy()
    ft0[..., 0, 0] = 0
    ft1[..., 0, 0] = 0
    norm0 = _norm(ft0, size)
    norm1 = _norm(ft1, size)
    cc = sfft.irfft2(ft0 * ft1.conj(), s=(size, size))
    cc = sfft.fftshift(cc, axes=(-2, -1))
    c = size // 2
    cc = cc[..., c - radius : c + radius + 1, c - radius : c + radius + 1]
    denom = np.maximum(norm0 * norm1, _EPS)
    return (cc / denom[..., np.newaxis, np.newaxis]).astype(np.float32)


def _norm(ft: NDArray[np.complexfloating], size: int) -> NDArray[np.float64]:
    """L2 norms of the real images of the (..., size, size // 2 + 1) rfft."""
    return np.linalg.norm(sfft.irfft2(ft, s=(size, size)), axis=(-2, -1))
