from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import os
from pathlib import Path
import threading
from typing import Sequence

import numpy as np
from numpy.typing import DTypeLike, NDArray

from acryo._types import nm
from acryo.projection._io import VolumeWriter
from acryo.projection._reconstruct import DirectFourierReconstructor
from acryo.projection._tilt_series import TiltSeries


def reconstruct_tomogram(
    tilt_series: TiltSeries,
    images: NDArray[np.float32],
    shape: tuple[int, int, int],
    scale: nm | None = None,
    *,
    patch_size: int = 64,
    blend_margin: int | None = None,
    padding: float = 2.0,
    save_path: str | Path | None = None,
    overwrite: bool = False,
    num_workers: int | None = None,
    dtype: DTypeLike = np.float16,
) -> NDArray[np.floating]:
    """Reconstruct a tomogram from tilt images by direct Fourier inversion.

    The tomogram is tiled by cubic patches, which are reconstructed by
    :class:`DirectFourierReconstructor` and blended with cosine-tapered margins. The
    voxel ``(i, j, k)`` is at ``((i, j, k) - shape // 2) * scale`` nm in the tomogram
    space of the tilt series. The contrast of the tilt images is kept.

    The patches are reconstructed row by row along the z axis, and the finished slabs
    are written to the output, so only a slab of the size of the patches is in memory
    besides the output. If ``save_path`` is given, the tomogram is written to a
    pre-allocated MRC file and not kept in memory.

    Parameters
    ----------
    tilt_series : TiltSeries
        Geometry of the tilt series. ``scale`` must be the pixel size of ``images``.
    images : (T, H, W) array
        Tilt images (usually preprocessed), whose centers ``(H // 2, W // 2)`` are
        the origin of the detector.
    shape : (int, int, int)
        Shape of the tomogram.
    scale : float, optional
        Voxel size of the tomogram in nm. Default is the pixel size of the images.
    patch_size : int, default is 64
        Distance between the centers of the patches in voxels.
    blend_margin : int, optional
        Margin of each patch in voxels, which is blended with the neighboring
        patches. Default is ``patch_size // 4``.
    padding : float, default is 2.0
        Padding factor of the reconstruction of each patch.
    save_path : str or Path, optional
        If given, the tomogram is written to this MRC file.
    overwrite : bool, default is False
        Whether to overwrite the existing file at ``save_path``.
    num_workers : int, optional
        Number of threads to reconstruct the patches. Default is the number of the
        CPUs. Each thread needs memory for the 3D DFT of a padded patch (about
        ``8 * (padding * (patch_size + 2 * blend_margin)) ** 3 / 2`` bytes), so a
        smaller value uses less memory.
    dtype : float16 or float32, default is float16
        Data type of the tomogram. float16 is half the size of float32. If
        ``save_path`` is given, the MRC file is of mode 12 (float16) or mode 2
        (float32). The patches are always reconstructed in float32.

    Returns
    -------
    (D, H, W) array
        The tomogram. If ``save_path`` is given, a read-only memory map of the file.
    """
    shape = _normalize_shape(shape)
    images = np.asarray(images)
    if tilt_series.scale is None:
        raise ValueError(
            "'scale' of the tilt series (pixel size of images) is not set."
        )
    if images.ndim != 3 or images.shape[0] != tilt_series.ntilts:
        raise ValueError(
            f"Shape of images must be ({tilt_series.ntilts}, H, W), got {images.shape}."
        )
    if patch_size <= 0:
        raise ValueError(f"'patch_size' must be positive, got {patch_size}.")
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    if num_workers <= 0:
        raise ValueError(f"'num_workers' must be positive, got {num_workers}.")
    if blend_margin is None:
        blend_margin = patch_size // 4
    if blend_margin < 0:
        raise ValueError(f"'blend_margin' must be >= 0, got {blend_margin}.")
    if scale is None:
        scale = tilt_series.scale

    length = patch_size + 2 * blend_margin
    half = length // 2
    reconstructor = DirectFourierReconstructor(
        images,
        tilt_series.local_detector2tomo,
        input_spacing=tilt_series.scale,
        output_spacing=scale,
        sidelength=length,
        padding=padding,
    )
    window = _cosine_taper(patch_size, blend_margin)
    window_3d = (
        window[:, np.newaxis, np.newaxis]
        * window[np.newaxis, :, np.newaxis]
        * window[np.newaxis, np.newaxis, :]
    )
    # patch centers and the sum of the windows of each axis
    centers = [
        np.arange(patch_size // 2, size + patch_size // 2, patch_size) for size in shape
    ]
    weights = [_window_sum(size, cs - half, window) for size, cs in zip(shape, centers)]
    weight_yx = weights[1][:, np.newaxis] * weights[2][np.newaxis, :]
    volume_center = np.array(shape) // 2
    image_center = np.array(images.shape[1:]) // 2
    yx_centers = [(cy, cx) for cy in centers[1] for cx in centers[2]]
    nrows = len(centers[0])

    def _add_patch(row: _PatchRow, cz: int, cy: int, cx: int):
        point = (np.array([cz, cy, cx]) - volume_center) * scale
        positions = tilt_series.project_points(point) / tilt_series.scale
        patch = reconstructor(positions + image_center)
        patch *= window_3d
        row.add((cz - half, cy - half, cx - half), patch)

    writer = VolumeWriter(
        shape, save_path, scale=scale, overwrite=overwrite, dtype=dtype
    )
    # Two rows of patches are reconstructed at the same time, so that the threads are
    # not idle while waiting for the last patches of a row.
    rows: deque[_PatchRow] = deque()
    try:
        with ThreadPoolExecutor(max_workers=num_workers) as executor:

            def _submit(k: int):
                cz = int(centers[0][k])
                row = _PatchRow(cz - half, length, shape)
                row.futures = [
                    executor.submit(_add_patch, row, cz, int(cy), int(cx))
                    for cy, cx in yx_centers
                ]
                rows.append(row)

            for k in range(min(2, nrows)):
                _submit(k)
            for k in range(nrows):
                row = rows.popleft()
                for future in row.futures:
                    future.result()
                # the region before the next row of patches is finished
                if k + 1 < nrows:
                    next_start = int(centers[0][k + 1]) - half
                else:
                    next_start = shape[0]
                finished = min(max(next_start, row.start), row.stop)
                done = row.slab[: finished - row.start]
                done /= weights[0][row.start : finished, np.newaxis, np.newaxis]
                done /= weight_yx
                writer.write(row.start, done)
                if rows:
                    # the rest is added to the next row
                    rows[0].add_slab(row.slab[finished - row.start :])
                del row, done
                if k + 2 < nrows:
                    _submit(k + 2)
    except BaseException:
        for row in rows:
            for future in row.futures:
                future.cancel()
        writer.abort()
        raise
    return writer.close()


class _PatchRow:
    """Accumulation of a row of patches along the z axis."""

    def __init__(self, z0: int, length: int, shape: tuple[int, int, int]):
        self.z0 = z0
        self.length = length
        self.start, self.stop = max(z0, 0), min(z0 + length, shape[0])
        self.slab = np.zeros(
            (self.stop - self.start, shape[1], shape[2]), dtype=np.float32
        )
        self.futures: list[Future] = []
        self._lock = threading.Lock()

    def add(self, start: tuple[int, int, int], patch: NDArray[np.float32]):
        """Add a patch whose first voxel is at ``start`` of the volume."""
        src, dst = _clip_region(start, self.length, (self.start, 0, 0), self.slab.shape)
        with self._lock:
            self.slab[dst] += patch[src]

    def add_slab(self, slab: NDArray[np.float32]):
        """Add a slab that starts at the first z of this row."""
        with self._lock:
            self.slab[: slab.shape[0]] += slab


def _normalize_shape(shape: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(shape, (int, np.integer)):
        out = (int(shape),) * 3
    else:
        out = tuple(int(s) for s in shape)
    if len(out) != 3 or any(s <= 0 for s in out):
        raise ValueError(f"'shape' must be positive (D, H, W), got {shape!r}.")
    return out  # type: ignore


def _cosine_taper(core: int, margin: int) -> NDArray[np.float32]:
    """1D window, 1 over the core and cosine-tapered over the margins."""
    if margin == 0:
        return np.ones(core, dtype=np.float32)
    ramp = 0.5 * (1 - np.cos(np.linspace(0, np.pi, margin)))
    return np.concatenate([ramp, np.ones(core), ramp[::-1]]).astype(np.float32)


def _window_sum(
    size: int, starts: NDArray[np.intp], window: NDArray[np.float32]
) -> NDArray[np.float32]:
    """Sum of the windows starting at ``starts`` within ``[0, size)``."""
    out = np.zeros(size, dtype=np.float64)
    for start in starts:
        lo, hi = max(start, 0), min(start + window.size, size)
        out[lo:hi] += window[lo - start : hi - start]
    return out.astype(np.float32)


def _clip_region(
    start: tuple[int, int, int],
    length: int,
    origin: tuple[int, int, int],
    shape: tuple[int, ...],
) -> tuple[tuple[slice, ...], tuple[slice, ...]]:
    """Slices of a cube at ``start`` and the overlapping region of an array.

    The array of ``shape`` starts at ``origin`` in the same coordinates.
    """
    src: list[slice] = []
    dst: list[slice] = []
    for s, o, size in zip(start, origin, shape):
        lo, hi = max(s, o), min(s + length, o + size)
        src.append(slice(lo - s, max(hi - s, lo - s)))
        dst.append(slice(lo - o, max(hi - o, lo - o)))
    return tuple(src), tuple(dst)
