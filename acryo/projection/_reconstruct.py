from __future__ import annotations

from collections import OrderedDict
import math
import threading

import numpy as np
from numpy.typing import NDArray
from scipy import fft as sfft

from acryo.projection._insertion import (
    CentralSliceInsertion,
    LocalSliceInsertion,
    slice_stretch,
)
from acryo.projection._patch import centered_patches_rfft


class DirectFourierReconstructor:
    """Reconstruct cubes from tilt images by direct Fourier inversion.

    The patches of the tilt images around the projections of a point are inserted
    into the 3D DFT of a padded cube, and the central region is returned. The point
    is at ``sidelength // 2`` of the output cube. The insertion is computed only once
    for all the points, unless the matrices of each cube are given (see
    :meth:`__call__`). Then, the insertion is computed for each cube, or shared by the
    cubes of almost the same matrices, such as the neighboring particles.

    Parameters
    ----------
    images : (T, H, W) array
        Tilt images, the projections of the volume. It can be memory-mapped, because
        only the patches are read.
    rotations : (T, 3, 3) array
        Matrices that map the (z, y, x) coordinates on each tilt image (z = 0) to
        the (z, y, x) coordinates of the cube. The image plane may also be scaled,
        such as :attr:`TiltSeries.local_detector2tomo` of magnified tilt images.
    input_spacing : float
        Pixel spacing of the tilt images.
    output_spacing : float
        Pixel spacing of the cube, in the same unit as ``input_spacing``.
    sidelength : int
        Size of the cube.
    padding : float, default is 2.0
        Size of the 3D DFT relative to the cube. Larger padding reduces the artifacts
        of interpolation and the aliasing of the densities outside the cube, but is
        slower.
    """

    def __init__(
        self,
        images: NDArray[np.float32],
        rotations: NDArray[np.floating],
        input_spacing: float,
        output_spacing: float,
        sidelength: int,
        padding: float = 2.0,
    ):
        if len(images) != len(rotations):
            raise ValueError(
                f"Number of images ({len(images)}) and rotations ({len(rotations)}) "
                "do not match."
            )
        if sidelength <= 0:
            raise ValueError(f"'sidelength' must be positive, got {sidelength}.")
        if padding < 1:
            raise ValueError(f"'padding' must be >= 1, got {padding}.")
        self.images = images
        self.sidelength = sidelength
        self._rotations = np.asarray(rotations, dtype=np.float64)
        self._padded = padded = sfft.next_fast_len(
            math.ceil(padding * sidelength), real=True
        )
        self._ratio = ratio = output_spacing / input_spacing
        # The patches cover the padded cube. If the images are magnified, the size is
        # scaled by the mean magnification, so that the period of the patches is
        # close to that of the padded cube for all the images, which is more accurate
        # than covering the cube in all the images.
        stretch = float(np.exp(np.log(slice_stretch(self._rotations)).mean()))
        self.patch_size = math.ceil(padded * ratio * stretch)
        self.local_insertion = LocalSliceInsertion(self.patch_size, padded, scale=ratio)
        # The precomputed insertion is created when it is first needed, because it is
        # not used if the matrices of each cube are given.
        self._insertion: CentralSliceInsertion | None = None
        self._lock = threading.Lock()
        # The insertions shared by the cubes of almost the same matrices. The matrices
        # of a cube are close enough to those of an anchor if the positions in the
        # cube differ by less than 0.1 voxel at the faces.
        self._anchors = OrderedDict[int, _Anchor]()
        self._anchor_tolerance = 0.2 / sidelength
        self._anchor_count = 0

    @property
    def insertion(self) -> CentralSliceInsertion:
        """The precomputed insertion of the slices of ``rotations``."""
        with self._lock:
            if self._insertion is None:
                self._insertion = self._create_insertion(self._rotations)
            return self._insertion

    @property
    def nbytes(self) -> int:
        """Approximate memory usage in bytes, except for the images."""
        with self._lock:
            insertions = [a.insertion for a in self._anchors.values()]
            insertions.append(self._insertion)
        return sum(ins.nbytes for ins in insertions if ins is not None)

    def __call__(
        self,
        positions: NDArray[np.floating],
        rotations: NDArray[np.floating] | None = None,
    ) -> NDArray[np.float32]:
        """Reconstruct cubes.

        Parameters
        ----------
        positions : (..., T, 2) array
            (y, x) positions (pixels) of the center of the cube on each tilt image.
            Leading dimensions are for multiple cubes.
        rotations : (..., T, 3, 3) array, optional
            Matrices of each cube, which replace ``rotations`` of the constructor, if
            the projection depends on the position, such as
            :meth:`TiltSeries.local_detector2tomo_at` of a tilt series with local
            alignments. The slices of each cube are inserted separately (see
            :class:`LocalSliceInsertion`), but the insertion is precomputed and shared
            by the next cubes of almost the same matrices, which are reconstructed
            within 0.1 voxel at the faces.

        Returns
        -------
        (..., L, L, L) array
            The reconstructed cubes.
        """
        slices = centered_patches_rfft(self.images, positions, self.patch_size)
        if rotations is None:
            return self.insertion.reconstruct(slices, output_shape=self.sidelength)
        rotations = np.asarray(rotations, dtype=np.float64)
        batch_shape = slices.shape[:-3]
        if rotations.shape != batch_shape + self._rotations.shape:
            raise ValueError(
                f"Shape of rotations must be {batch_shape + self._rotations.shape}, "
                f"got {rotations.shape}."
            )
        slices = slices.reshape((-1,) + slices.shape[-3:])
        rotations = rotations.reshape((-1,) + self._rotations.shape)
        out = np.empty((slices.shape[0],) + (self.sidelength,) * 3, dtype=np.float32)
        for i, (sl, rot) in enumerate(zip(slices, rotations)):
            if (insertion := self._shared_insertion(rot)) is None:
                out[i] = self.local_insertion.reconstruct(
                    sl, rot, output_shape=self.sidelength
                )
            else:
                out[i] = insertion.reconstruct(sl, output_shape=self.sidelength)
        return out.reshape(batch_shape + out.shape[1:])

    def _create_insertion(
        self, rotations: NDArray[np.float64]
    ) -> CentralSliceInsertion:
        return CentralSliceInsertion(
            rotations, self.patch_size, self._padded, scale=self._ratio
        )

    def _shared_insertion(
        self, rotations: NDArray[np.float64]
    ) -> CentralSliceInsertion | None:
        """The insertion shared by the cubes of almost the same matrices, if any.

        The first cube of new matrices is inserted separately, and its matrices are
        registered as an anchor. The insertion of the anchor is created for the next
        cube close to it, so isolated cubes are never slower than the separate
        insertion.
        """
        with self._lock:
            found = None
            for key, anchor in reversed(self._anchors.items()):
                diff = np.linalg.norm(rotations - anchor.rotations, axis=(-2, -1))
                if diff.max() <= self._anchor_tolerance:
                    found = key
                    break
            if found is None:
                self._anchors[self._anchor_count] = _Anchor(rotations)
                self._anchor_count += 1
                while len(self._anchors) > _MAX_ANCHORS:
                    self._anchors.popitem(last=False)
                return None
            self._anchors.move_to_end(found)
            anchor = self._anchors[found]
        return anchor.get(self._create_insertion)


# Maximum number of the shared insertions of a reconstructor. Each one is as large as
# the precomputed insertion (about 100 MB for cubes of 100 voxels).
_MAX_ANCHORS = 4


class _Anchor:
    """Matrices of a cube and the insertion created when it is first shared."""

    def __init__(self, rotations: NDArray[np.float64]):
        self.rotations = rotations
        self.insertion: CentralSliceInsertion | None = None
        self._lock = threading.Lock()

    def get(self, create) -> CentralSliceInsertion:
        # created only once, even if requested by many threads at the same time
        with self._lock:
            if self.insertion is None:
                self.insertion = create(self.rotations)
            return self.insertion
