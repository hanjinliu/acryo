from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray
from scipy import fft as sfft

from acryo.projection._insertion import CentralSliceInsertion, slice_stretch
from acryo.projection._patch import centered_patches_rfft


class DirectFourierReconstructor:
    """Reconstruct cubes from tilt images by direct Fourier inversion.

    The patches of the tilt images around the projections of a point are inserted
    into the 3D DFT of a padded cube, and the central region is returned. The point
    is at ``sidelength // 2`` of the output cube. The insertion is computed only once
    for all the points.

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
        padded = sfft.next_fast_len(math.ceil(padding * sidelength), real=True)
        ratio = output_spacing / input_spacing
        # The patches cover the padded cube. If the images are magnified, the size is
        # scaled by the mean magnification, so that the period of the patches is
        # close to that of the padded cube for all the images, which is more accurate
        # than covering the cube in all the images.
        stretch = float(np.exp(np.log(slice_stretch(rotations)).mean()))
        self.patch_size = math.ceil(padded * ratio * stretch)
        self.insertion = CentralSliceInsertion(
            rotations, self.patch_size, padded, scale=ratio
        )

    @property
    def nbytes(self) -> int:
        """Approximate memory usage in bytes, except for the images."""
        return self.insertion.nbytes

    def __call__(self, positions: NDArray[np.floating]) -> NDArray[np.float32]:
        """Reconstruct cubes.

        Parameters
        ----------
        positions : (..., T, 2) array
            (y, x) positions (pixels) of the center of the cube on each tilt image.
            Leading dimensions are for multiple cubes.

        Returns
        -------
        (..., L, L, L) array
            The reconstructed cubes.
        """
        slices = centered_patches_rfft(self.images, positions, self.patch_size)
        return self.insertion.reconstruct(slices, output_shape=self.sidelength)
