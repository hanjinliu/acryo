from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import fft as sfft

from acryo.projection._insertion import _gridding_correction


class VolumeProjector:
    """Projection of a volume by extracting central slices from its 3D DFT.

    The volume is padded and corrected for the trilinear interpolation, and its DFT is
    computed once. Each projection is the inverse transform of a central slice of the
    DFT (Fourier slice theorem), so many projections of different orientations (such
    as the projections of a template in the tilt images) are computed quickly. This
    is the inverse operation of :class:`CentralSliceInsertion`.

    The average of the voxels near the faces of the volume is subtracted before
    padding, so the padded volume is continuous, and added back to the projections.

    Parameters
    ----------
    volume : (D, D, D) array
        A cubic volume, whose center is at ``D // 2``.
    padding : float, default is 2.0
        Factor of the size of the padded volume to that of the volume. A larger value
        reduces the artifacts of interpolation.
    """

    def __init__(self, volume: NDArray[np.floating], padding: float = 2.0):
        volume = np.asarray(volume, dtype=np.float32)
        if volume.ndim != 3 or len(set(volume.shape)) != 1:
            raise ValueError(f"Volume must be a cube, got shape {volume.shape}.")
        if padding < 1:
            raise ValueError(f"'padding' must be >= 1, got {padding}.")
        size = volume.shape[0]
        self._size = size
        self._edge = float(_face_average(volume))
        self._pad = pad = int(size * (padding - 1) // 2)
        padded = np.pad(volume - self._edge, pad)
        shape = padded.shape
        padded = padded / _gridding_correction(shape, shape)
        # origin of the volume at index 0
        self._dft = sfft.rfftn(sfft.ifftshift(padded)).astype(np.complex64)

    @property
    def size(self) -> int:
        """Size of the volume and the projections."""
        return self._size

    @property
    def padded_size(self) -> int:
        """Size of the padded volume."""
        return self._dft.shape[0]

    def __call__(self, rotations: NDArray[np.floating]) -> NDArray[np.float32]:
        """Project the volume.

        Parameters
        ----------
        rotations : (..., 3, 3) array
            Matrices that map the (z, y, x) coordinates of each projection (whose z
            axis is the direction of the projection) to the (z, y, x) coordinates of
            the volume.

        Returns
        -------
        (..., D, D) array
            The projections, that is, the sums of the voxels along the z axis of each
            projection. Their centers are at ``D // 2``.
        """
        rotations = np.asarray(rotations, dtype=np.float64)
        if rotations.shape[-2:] != (3, 3):
            raise ValueError(
                f"Shape of rotations must be (..., 3, 3), got {rotations.shape}."
            )
        batch_shape = rotations.shape[:-2]
        n = self.padded_size
        slices = self._extract(rotations.reshape(-1, 3, 3))
        images = sfft.irfft2(slices, s=(n, n), overwrite_x=True)
        # crop the center of the padded projections
        indices = (np.arange(self._size) - self._size // 2) % n
        images = images[:, indices][:, :, indices] + self._edge * self._size
        return images.astype(np.float32).reshape(batch_shape + images.shape[1:])

    def _extract(self, rotations: NDArray[np.float64]) -> NDArray[np.complex64]:
        """Central slices (M, n, n // 2 + 1) of the DFT with trilinear interpolation."""
        n = self.padded_size
        nx = self._dft.shape[2]
        qy, qx = np.meshgrid(sfft.fftfreq(n), sfft.rfftfreq(n), indexing="ij")
        # Frequencies beyond the Nyquist frequency are not extracted. Rotations keep
        # the norms, so they are selected before the rotation.
        inside = np.flatnonzero((qy**2 + qx**2).ravel() <= 0.25)
        freqs = np.stack([np.zeros_like(qy), qy, qx], axis=-1).reshape(-1, 3)
        freqs = (freqs[inside] * n).astype(np.float32)
        dft = self._dft.ravel()
        out = np.zeros((rotations.shape[0], n * nx), dtype=np.complex64)
        # chunks of about 2^20 points to limit the memory
        chunk = max(1, (1 << 20) // inside.size)
        for start in range(0, rotations.shape[0], chunk):
            rot = rotations[start : start + chunk].astype(np.float32)
            # (m, K, 3) indices of the frequencies in the DFT
            index = freqs @ rot.transpose(0, 2, 1)
            # the rfft only has x >= 0, where the conjugate values are read
            is_conj = index[..., 2] < 0
            index[is_conj] *= -1
            base = np.floor(index)
            frac = index - base
            base = base.astype(np.int32)
            # flat indices of the corners, which are periodic along z and y
            iz = (base[..., 0] % n, (base[..., 0] + 1) % n)
            iy = (base[..., 1] % n, (base[..., 1] + 1) % n)
            ix = (base[..., 2], np.minimum(base[..., 2] + 1, nx - 1))
            fz, fy, fx = frac[..., 0], frac[..., 1], frac[..., 2]
            wz = (1 - fz, fz)
            wy = (1 - fy, fy)
            wx = (1 - fx, fx)
            values = np.zeros(index.shape[:2], dtype=np.complex64)
            for cz in (0, 1):
                for cy in (0, 1):
                    row = (iz[cz] * n + iy[cy]) * nx
                    wzy = wz[cz] * wy[cy]
                    for cx in (0, 1):
                        values += (wzy * wx[cx]) * dft.take(row + ix[cx])
            np.conjugate(values, out=values, where=is_conj)
            out[start : start + chunk, inside] = values
        return out.reshape(-1, n, nx)


def _face_average(volume: NDArray[np.floating], width: int = 4) -> np.floating:
    """Average of the voxels within ``width`` voxels from the faces of the volume."""
    inner = tuple(slice(width, max(s - width, width)) for s in volume.shape)
    mask = np.ones(volume.shape, dtype=bool)
    mask[inner] = False
    return volume[mask].mean()
