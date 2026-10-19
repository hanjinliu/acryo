from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import fft as sfft
from scipy import sparse


class CentralSliceInsertion:
    """Insertion of central slices into a 3D DFT, as a precomputed sparse operator.

    The 2D DFTs (rfft) of images are inserted as central slices into the 3D DFT of a
    volume with trilinear interpolation, and the volume is reconstructed by direct
    Fourier inversion. The insertion depends only on the orientations of the slices,
    so it is computed once, and many volumes of the same geometry (such as
    subtomograms of different molecules) can be reconstructed quickly.

    The images are regarded as the projections of the volume along the z axis of
    each slice, that is, the sum over the voxels. Images of a smaller pixel size
    (``scale > 1``) are normalized by the pixel area.

    Parameters
    ----------
    rotations : (T, 3, 3) array
        Matrices that map the (z, y, x) coordinates of each slice (z = 0) to the
        (z, y, x) coordinates of the volume. They are usually rotations, but the
        image plane may also be scaled or sheared, such as a magnified image (see
        :attr:`TiltSeries.local_detector2tomo`). Then, the frequencies are mapped by
        the inverse transpose, and the image is normalized by the area of its pixels
        in the volume.
    slice_shape : int or (int, int)
        Shape of the images whose rfft are inserted.
    volume_shape : int or (int, int, int)
        Shape of the volume. A padded shape (such as twice the size of the output) is
        usually used to reduce the artifacts of interpolation.
    scale : float, default is 1.0
        Pixel size of the volume divided by the pixel size of the images. Frequencies
        beyond the Nyquist frequency of the volume are not inserted.
    """

    def __init__(
        self,
        rotations: NDArray[np.floating],
        slice_shape: int | Sequence[int],
        volume_shape: int | Sequence[int],
        scale: float = 1.0,
    ):
        rotations = np.asarray(rotations, dtype=np.float64)
        if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
            raise ValueError(
                f"Shape of rotations must be (T, 3, 3), got {rotations.shape}."
            )
        self.slice_shape = ny, nx = _normalize_shape(slice_shape, 2)
        self.volume_shape = vshape = _normalize_shape(volume_shape, 3)
        self.nslices = rotations.shape[0]
        qy, qx = np.meshgrid(
            sfft.fftfreq(ny).astype(np.float32),
            sfft.rfftfreq(nx).astype(np.float32),
            indexing="ij",
        )
        # (z, y, x) frequencies of the slices in cycles per pixel of the volume
        freqs = np.stack([np.zeros_like(qy), qy, qx], axis=-1).reshape(-1, 3)
        freqs *= np.float32(scale)
        # Only up to the Nyquist frequency of the volume. Half of the frequencies of
        # x = 0 are redundant because of the Friedel symmetry.
        stretch_min = slice_stretch(rotations).min()
        keep = (np.linalg.norm(freqs, axis=-1) <= 0.5 / stretch_min) & ~(
            (freqs[:, 2] == 0) & (freqs[:, 1] < 0)
        )
        self._keep = np.flatnonzero(keep)
        freqs = freqs[self._keep]
        nkeep = self._keep.size

        self._dft_shape = dft_shape = vshape[:2] + (vshape[2] // 2 + 1,)
        direct: list[tuple[NDArray, NDArray, NDArray]] = []
        conjugate: list[tuple[NDArray, NDArray, NDArray]] = []
        matrices = _frequency_matrices(rotations)
        for t, matrix in enumerate(matrices.astype(np.float32)):
            coords = freqs @ matrix.T
            columns = t * nkeep + np.arange(nkeep)
            # the frequencies of each slice beyond the Nyquist frequency of the volume
            # (only if the slice is stretched)
            inside = np.linalg.norm(coords, axis=-1) <= 0.5 * (1 + 1e-6)
            if not inside.all():
                coords, columns = coords[inside], columns[inside]
            # the rfft only has x >= 0, where the conjugate values are inserted
            is_conj = coords[:, 2] < 0
            coords[is_conj] *= -1
            index = coords * np.array(vshape, dtype=np.float32)
            base = np.floor(index).astype(np.int64)
            frac = index - base
            for corner in np.ndindex(2, 2, 2):
                weight = np.prod(
                    [f if c else 1 - f for c, f in zip(corner, frac.T)], axis=0
                )
                iz = (base[:, 0] + corner[0]) % vshape[0]
                iy = (base[:, 1] + corner[1]) % vshape[1]
                ix = base[:, 2] + corner[2]
                valid = ix < dft_shape[2]
                rows = np.ravel_multi_index(
                    (iz[valid], iy[valid], ix[valid]), dft_shape
                )
                conj = is_conj[valid]
                entries = (rows, columns[valid], weight[valid])
                direct.append(tuple(e[~conj] for e in entries))
                conjugate.append(tuple(e[conj] for e in entries))
        shape = (int(np.prod(dft_shape)), self.nslices * nkeep)
        self._direct = _to_csr(direct, shape)
        self._conjugate = _to_csr(conjugate, shape)
        weights = np.asarray(
            self._direct.sum(axis=1) + self._conjugate.sum(axis=1)
        ).real.ravel()
        valid = weights > 1e-3
        self._normalizer = np.zeros(weights.shape, dtype=np.float32)
        # normalize by the pixel area of the images
        self._normalizer[valid] = 1 / weights[valid] / scale**2
        areas = _slice_areas(matrices)
        if not np.allclose(areas, 1.0, rtol=0, atol=1e-6):
            # The pixels of a stretched slice are smaller in the volume. The values
            # of each slice are scaled after the weights are summed.
            factors = sparse.diags(np.repeat(1 / areas, nkeep).astype(np.complex64))
            self._direct = (self._direct @ factors).tocsr()
            self._conjugate = (self._conjugate @ factors).tocsr()

    @property
    def nbytes(self) -> int:
        """Approximate memory usage in bytes."""
        mats = [self._direct, self._conjugate]
        return sum(a.data.nbytes + a.indices.nbytes + a.indptr.nbytes for a in mats)

    def insert(self, slices: NDArray[np.complexfloating]) -> NDArray[np.complex64]:
        """Insert the slices and return the 3D DFT normalized by the weights.

        Parameters
        ----------
        slices : (..., T, ny, nx // 2 + 1) complex array
            The rfft of the images (not fftshifted), whose origin is the origin of
            the volume. Leading dimensions are for multiple volumes.

        Returns
        -------
        (..., Z, Y, X // 2 + 1) complex array
            The rfft of the volumes (not fftshifted).
        """
        slices = np.asarray(slices)
        expected = (self.nslices, self.slice_shape[0], self.slice_shape[1] // 2 + 1)
        if slices.shape[-3:] != expected:
            raise ValueError(
                f"Shape of slices must be (..., {expected[0]}, {expected[1]}, "
                f"{expected[2]}), got {slices.shape}."
            )
        batch_shape = slices.shape[:-3]
        values = slices.reshape(-1, self.nslices, expected[1] * expected[2])
        # (T * nkeep, batch) matrix of the values to be inserted
        values = values[:, :, self._keep].reshape(values.shape[0], -1).T
        values = values.astype(np.complex64, copy=False)
        dft = self._direct @ values
        dft += self._conjugate @ values.conj()
        dft *= self._normalizer[:, np.newaxis]
        return np.moveaxis(dft, -1, 0).reshape(batch_shape + self._dft_shape)

    def reconstruct(
        self,
        slices: NDArray[np.complexfloating],
        output_shape: int | Sequence[int] | None = None,
    ) -> NDArray[np.float32]:
        """Reconstruct the volume from the slices.

        Parameters
        ----------
        slices : (..., T, ny, nx // 2 + 1) complex array
            The rfft of the images (not fftshifted), whose origin is the origin of
            the volume. Leading dimensions are for multiple volumes.
        output_shape : int or (int, int, int), optional
            If given, the central region of this shape is returned. The origin of the
            volume is at ``output_shape // 2`` of the output. By default, the whole
            volume is returned with the origin at ``volume_shape // 2``.

        Returns
        -------
        (..., *output_shape) float32 array
            The reconstructed volumes, corrected for the trilinear interpolation.
        """
        vshape = self.volume_shape
        if output_shape is None:
            output_shape = vshape
        output_shape = _normalize_shape(output_shape, 3)
        # indices of the output, whose origin is moved to the center
        indices = [
            (np.arange(size) - size // 2) % length
            for size, length in zip(output_shape, vshape)
        ]
        # The inverse transform is computed axis by axis, and each axis is cropped
        # right after the transform, so the later transforms are smaller.
        volume = sfft.ifft(self.insert(slices), axis=-3, overwrite_x=True)
        volume = sfft.ifft(volume[..., indices[0], :, :], axis=-2, overwrite_x=True)
        volume = sfft.irfft(volume[..., indices[1], :], n=vshape[2], overwrite_x=True)
        volume = volume[..., indices[2]].astype(np.float32, copy=False)
        volume /= _gridding_correction(vshape, output_shape)
        return volume


def _normalize_shape(shape: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    if isinstance(shape, (int, np.integer)):
        return (int(shape),) * ndim
    out = tuple(int(s) for s in shape)
    if len(out) != ndim:
        raise ValueError(f"Shape must be an integer or a {ndim}-tuple, got {shape!r}.")
    return out


def slice_stretch(rotations: NDArray[np.floating]) -> NDArray[np.float64]:
    """(T, 2) factors of the frequencies of each slice in the volume.

    They are the singular values of the image plane of the matrices of
    :class:`CentralSliceInsertion`, which are exactly 1 for rotations. The frequencies
    of a magnified image are stretched (factor > 1).
    """
    rotations = np.asarray(rotations, dtype=np.float64)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(
            f"Shape of rotations must be (T, 3, 3), got {rotations.shape}."
        )
    if _is_orthogonal(rotations):
        return np.ones((rotations.shape[0], 2))
    planes = _frequency_matrices(rotations)[:, :, 1:]
    return np.linalg.svd(planes, compute_uv=False)


def _is_orthogonal(matrices: NDArray[np.float64]) -> bool:
    gram = matrices @ matrices.transpose(0, 2, 1)
    return bool(np.allclose(gram, np.eye(3), rtol=0, atol=1e-8))


def _frequency_matrices(rotations: NDArray[np.float64]) -> NDArray[np.float64]:
    """Matrices that map the frequencies of the slices to those of the volume.

    The frequencies are mapped by the inverse transpose of the matrices of the
    coordinates, which are the same for rotations.
    """
    if _is_orthogonal(rotations):
        return rotations
    return np.linalg.inv(rotations).transpose(0, 2, 1)


def _slice_areas(matrices: NDArray[np.float64]) -> NDArray[np.float64]:
    """Areas of the unit square of the frequencies of each slice in the volume."""
    return np.linalg.norm(np.cross(matrices[:, :, 1], matrices[:, :, 2]), axis=-1)


def _to_csr(entries, shape) -> sparse.csr_matrix:
    rows, cols, vals = (np.concatenate(e) for e in zip(*entries))
    return sparse.csr_matrix((vals.astype(np.complex64), (rows, cols)), shape=shape)


def _gridding_correction(
    volume_shape: tuple[int, ...], output_shape: tuple[int, ...]
) -> NDArray[np.float32]:
    """Correction of the trilinear interpolation in Fourier space (separable sinc^2)."""
    correction = np.ones(output_shape, dtype=np.float32)
    for axis, (length, size) in enumerate(zip(volume_shape, output_shape)):
        sinc2 = np.sinc((np.arange(size) - size // 2) / length) ** 2
        shape = [1] * len(output_shape)
        shape[axis] = size
        correction *= sinc2.astype(np.float32).reshape(shape)
    return correction
