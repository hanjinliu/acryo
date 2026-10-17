from __future__ import annotations

import math
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import fft as sfft


def read_patch(
    image: NDArray[np.float32],
    start: Sequence[int],
    shape: Sequence[int],
) -> NDArray[np.float32]:
    """Read a patch of a 2D image, with zero padding outside the image.

    Only the region of the patch is read, so the image can be memory-mapped.

    Parameters
    ----------
    image : (H, W) array
        The image.
    start : (int, int)
        (y, x) index of the first pixel of the patch.
    shape : (int, int)
        Shape of the patch.
    """
    patch = np.zeros(tuple(shape), dtype=np.float32)
    y0, x0 = start
    ys0, xs0 = max(y0, 0), max(x0, 0)
    ys1 = min(y0 + patch.shape[0], image.shape[0])
    xs1 = min(x0 + patch.shape[1], image.shape[1])
    if ys0 < ys1 and xs0 < xs1:
        patch[ys0 - y0 : ys1 - y0, xs0 - x0 : xs1 - x0] = image[ys0:ys1, xs0:xs1]
    return patch


def centered_patches_rfft(
    images: NDArray[np.float32],
    positions: NDArray[np.floating],
    shape: int | Sequence[int],
) -> NDArray[np.complex64]:
    """The rfft of the patches of images, whose origins are at the given positions.

    A patch around each position is read, and shifted with subpixel precision in
    Fourier space so that the position is at the origin (index 0). The rfft is the
    input of the central slice insertion.

    Parameters
    ----------
    images : (T, H, W) array
        Images, such as a tilt series. It can be memory-mapped, because only the
        patches are read.
    positions : (..., T, 2) array
        (y, x) positions in pixels, one for each image. Leading dimensions are for
        multiple sets of positions, such as different particles.
    shape : int or (int, int)
        Shape of the patches.

    Returns
    -------
    (..., T, ny, nx // 2 + 1) complex array
        The rfft of the patches.
    """
    ny, nx = (shape, shape) if isinstance(shape, int) else tuple(shape)
    positions = np.asarray(positions, dtype=np.float64)
    if positions.shape[-2:] != (len(images), 2):
        raise ValueError(
            f"Shape of positions must be (..., {len(images)}, 2), got "
            f"{positions.shape}."
        )
    batch_shape = positions.shape[:-2]
    positions = positions.reshape(-1, len(images), 2)
    patches = np.empty(positions.shape[:2] + (ny, nx), dtype=np.float32)
    shifts = np.empty(positions.shape, dtype=np.float64)
    for t, image in enumerate(images):
        for i, (y, x) in enumerate(positions[:, t]):
            iy, ix = math.floor(y), math.floor(x)
            patches[i, t] = read_patch(image, (iy - ny // 2, ix - nx // 2), (ny, nx))
            # position of the point in the patch
            shifts[i, t] = ny // 2 + y - iy, nx // 2 + x - ix
    ft = sfft.rfft2(patches, axes=(-2, -1))
    # shift the point to the origin
    ramp_y = np.exp(2j * np.pi * shifts[..., 0:1] * sfft.fftfreq(ny))
    ramp_x = np.exp(2j * np.pi * shifts[..., 1:2] * sfft.rfftfreq(nx))
    ft *= ramp_y[..., :, np.newaxis] * ramp_x[..., np.newaxis, :]
    return ft.astype(np.complex64).reshape(batch_shape + ft.shape[-3:])
