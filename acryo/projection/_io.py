from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Sequence
import warnings

import numpy as np
from numpy.typing import DTypeLike, NDArray

from acryo._types import nm


class MrcImageStack:
    """An MRC file of a stack of 2D images, such as a tilt series.

    Images are read one by one from the file, so the whole stack is never read nor
    memory-mapped.
    """

    def __init__(self, path: str | Path):
        import mrcfile
        from mrcfile.utils import data_dtype_from_header

        with mrcfile.open(path, mode="r", header_only=True, permissive=True) as mrc:
            header = mrc.header
            self.shape = (int(header.nz), int(header.ny), int(header.nx))
            self.dtype = data_dtype_from_header(header)
            # the data block follows the main header and the extended header
            self.offset = header.nbytes + int(header.nsymbt)
            # pixel size in Å (0 if not set)
            self.voxel_size = float(mrc.voxel_size.x)
        self.path = path

    def iter_images(self, indices: Iterable[int]) -> Iterator[NDArray[np.float32]]:
        """Iterate over the images of given indices as float32 arrays."""
        nimages, ny, nx = self.shape
        with open(self.path, "rb") as f:
            for index in indices:
                if not -nimages <= index < nimages:
                    raise IndexError(
                        f"Index {index} is out of range of {nimages} images in "
                        f"{self.path}."
                    )
                f.seek(self.offset + (index % nimages) * ny * nx * self.dtype.itemsize)
                image = np.fromfile(f, dtype=self.dtype, count=ny * nx)
                if image.size != ny * nx:
                    raise ValueError(f"The data block of {self.path} is truncated.")
                yield image.reshape(ny, nx).astype(np.float32, copy=False)

    def memmap(self) -> np.memmap:
        """Read-only memory map of the whole data block."""
        return np.memmap(
            self.path, dtype=self.dtype, mode="r", offset=self.offset, shape=self.shape
        )


class TiltImageSource:
    """The tilt images of a tilt series, read one by one from MRC file(s).

    The images are either in a stack (``image_path`` is a path) or in separate files
    (``image_path`` is a sequence of paths, one for each tilt). ``image_indices`` are
    the indices of the images in the stack, or in each file.
    """

    def __init__(
        self,
        image_path: Path | Sequence[Path],
        image_indices: Sequence[int] | None,
        ntilts: int,
    ):
        if isinstance(image_path, (str, Path)):
            stack = MrcImageStack(image_path)
            if image_indices is None:
                indices = list(range(stack.shape[0]))
            else:
                indices = [int(i) for i in image_indices]
            if len(indices) != ntilts:
                raise ValueError(
                    f"The tilt series has {ntilts} tilts, but {len(indices)} images "
                    f"are found in {image_path}. Set 'image_indices' of the tilt "
                    "series."
                )
            self._stacks = [stack]
            self._indices = indices
        else:
            paths = list(image_path)
            if len(paths) != ntilts:
                raise ValueError(
                    f"The tilt series has {ntilts} tilts, but {len(paths)} image files "
                    "are given."
                )
            self._stacks = [MrcImageStack(path) for path in paths]
            if image_indices is None:
                self._indices = [0] * ntilts
            else:
                self._indices = [int(i) for i in image_indices]
            shapes = {stack.shape[1:] for stack in self._stacks}
            if len(shapes) != 1:
                raise ValueError(f"Shapes of the tilt images differ: {shapes}.")

    @property
    def image_shape(self) -> tuple[int, int]:
        """(H, W) shape of the tilt images."""
        return self._stacks[0].shape[1:]  # type: ignore

    @property
    def voxel_size(self) -> float:
        """Pixel size in Å written in the header of the (first) file."""
        return self._stacks[0].voxel_size

    def iter_images(self) -> Iterator[NDArray[np.float32]]:
        """Iterate over the tilt images as float32 arrays."""
        if len(self._stacks) == 1:
            yield from self._stacks[0].iter_images(self._indices)
        else:
            for stack, index in zip(self._stacks, self._indices):
                yield from stack.iter_images([index])


# MRC modes of the supported data types
_MRC_MODES = {np.dtype(np.float32): 2, np.dtype(np.float16): 12}


class VolumeWriter:
    """Write a 3D volume slab by slab, in memory or to a pre-allocated MRC file.

    If ``path`` is given, an MRC file of the volume shape is allocated, and the slabs
    are written to the memory-mapped data, so the whole volume is never in memory.
    The statistics in the header are computed slab by slab. The data type is float32
    (MRC mode 2) or float16 (MRC mode 12).
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        path: str | Path | None = None,
        scale: nm = 1.0,
        overwrite: bool = False,
        dtype: DTypeLike = np.float32,
    ):
        self.shape = shape
        self.path = None if path is None else Path(path)
        self.dtype = _check_dtype(dtype)
        self._mrc = None
        if self.path is None:
            self._data: NDArray[np.floating] = np.empty(shape, dtype=self.dtype)
        else:
            import mrcfile

            self._mrc = mrcfile.new_mmap(
                self.path,
                shape=shape,
                mrc_mode=_MRC_MODES[self.dtype],
                overwrite=overwrite,
            )
            self._mrc.voxel_size = scale * 10  # Å
            self._data = self._mrc.data
        self._min = np.inf
        self._max = -np.inf
        self._sum = 0.0
        self._sum_sq = 0.0
        self._warned = False

    def write(self, start: int, slab: NDArray[np.float32]) -> None:
        """Write the slab at the z index ``start``."""
        if slab.size == 0:
            return
        if self.dtype != slab.dtype:
            info = np.finfo(self.dtype)
            if not self._warned and (slab.min() < info.min or slab.max() > info.max):
                warnings.warn(
                    f"Values out of the range of {self.dtype} are saturated to "
                    "infinity. Use float32 instead.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._warned = True
            with np.errstate(over="ignore"):
                slab = slab.astype(self.dtype)
        self._data[start : start + slab.shape[0]] = slab
        # statistics of the stored values (not finite after overflow, which is warned)
        with np.errstate(invalid="ignore", over="ignore"):
            self._min = min(self._min, float(slab.min()))
            self._max = max(self._max, float(slab.max()))
            self._sum += float(slab.sum(dtype=np.float64))
            self._sum_sq += float(np.square(slab, dtype=np.float64).sum())

    def close(self) -> NDArray[np.floating]:
        """Finish writing and return the volume.

        If the volume is written to a file, a read-only memory map of the file is
        returned.
        """
        if self._mrc is None:
            return self._data
        header = self._mrc.header
        count = int(np.prod(self.shape))
        mean = self._sum / count
        header.dmin = self._min
        header.dmax = self._max
        header.dmean = mean
        header.rms = np.sqrt(max(self._sum_sq / count - mean**2, 0.0))
        self._mrc.close()
        self._mrc = None
        del self._data
        assert self.path is not None
        return MrcImageStack(self.path).memmap()

    def abort(self) -> None:
        """Close the file without finishing."""
        if self._mrc is not None:
            # the memory map must be released to close the file
            self._data = None  # type: ignore
            self._mrc.close()
            self._mrc = None


def _check_dtype(dtype: DTypeLike) -> np.dtype:
    out = np.dtype(dtype)
    if out not in _MRC_MODES:
        raise ValueError(f"'dtype' must be float32 or float16, got {dtype!r}.")
    return out
