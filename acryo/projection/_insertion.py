from __future__ import annotations

import threading
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import fft as sfft
from scipy import sparse

from acryo.backend import Backend
from acryo.backend._gpu import backend_of


class CentralSliceInsertion:
    """Insertion of central slices into a 3D DFT, as a precomputed sparse operator.

    The 2D DFTs (rfft) of images are inserted as central slices into the 3D DFT of a
    volume with trilinear interpolation, and the volume is reconstructed by direct
    Fourier inversion. The insertion depends only on the orientations of the slices,
    so it is computed once, and many volumes of the same geometry (such as
    subtomograms of different molecules) can be reconstructed quickly.

    The images are regarded as the projections of the volume along the z axis of
    each slice, that is, the sum over the voxels. Images of a smaller pixel size
    (``scale > 1``) are normalized by the pixel area. The slices can be numpy or cupy
    arrays, and the volume is computed on the same device.

    Parameters
    ----------
    rotations : (T, 3, 3) array
        Matrices that map the (z, y, x) coordinates of each slice (z = 0) to the
        (z, y, x) coordinates of the volume.
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
        keep = (np.linalg.norm(freqs, axis=-1) <= 0.5) & ~(
            (freqs[:, 2] == 0) & (freqs[:, 1] < 0)
        )
        self._keep = np.flatnonzero(keep)
        freqs = freqs[self._keep]
        nkeep = self._keep.size

        self._dft_shape = dft_shape = vshape[:2] + (vshape[2] // 2 + 1,)
        direct: list[tuple[NDArray, NDArray, NDArray]] = []
        conjugate: list[tuple[NDArray, NDArray, NDArray]] = []
        for t, rotation in enumerate(rotations.astype(np.float32)):
            coords = freqs @ rotation.T
            # the rfft only has x >= 0, where the conjugate values are inserted
            is_conj = coords[:, 2] < 0
            coords[is_conj] *= -1
            index = coords * np.array(vshape, dtype=np.float32)
            base = np.floor(index).astype(np.int64)
            frac = index - base
            columns = t * nkeep + np.arange(nkeep)
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
        self._device_lock = threading.Lock()
        self._device_operators: tuple[Any, Any, Any, Any] | None = None

    @property
    def nbytes(self) -> int:
        """Approximate memory usage in bytes (on the host)."""
        mats = [self._direct, self._conjugate]
        return sum(a.data.nbytes + a.indices.nbytes + a.indptr.nbytes for a in mats)

    @property
    def nkeep(self) -> int:
        """Number of the frequencies of each slice that are inserted."""
        return self._keep.size

    def to_device(self, backend: Backend) -> None:
        """Copy the operators to the device of the backend beforehand."""
        self._operators(backend)

    def _operators(self, backend: Backend) -> tuple[Any, Any, Any, Any]:
        """The sparse matrices, the normalizer and the kept indices on the device."""
        if not backend.is_gpu:
            return self._direct, self._conjugate, self._normalizer, self._keep
        with self._device_lock:
            if self._device_operators is None:
                from cupyx.scipy import sparse as csparse

                xp = backend.xp
                self._device_operators = (
                    csparse.csr_matrix(self._direct),
                    csparse.csr_matrix(self._conjugate),
                    xp.asarray(self._normalizer),
                    xp.asarray(self._keep),
                )
                # the operators are shared by the streams of the threads
                xp.cuda.get_current_stream().synchronize()
            return self._device_operators

    def _device_nbytes(self, nbatch: int, output_shape: int | Sequence[int]) -> int:
        """Approximate peak device memory to reconstruct ``nbatch`` volumes."""
        ny, nx = self.slice_shape
        dft_size = int(np.prod(self._dft_shape))
        out_size = int(np.prod(_normalize_shape(output_shape, 3)))
        per_volume = (
            self.nslices * (4 * ny * nx + 16 * ny * (nx // 2 + 1))  # rfft and plan
            + 24 * self.nslices * self.nkeep  # inserted values and their conjugates
            + 32 * dft_size  # 3D DFT, the products and the inverse transforms
            + 4 * out_size
        )
        return nbatch * per_volume

    def insert(self, slices: NDArray[np.complexfloating]) -> NDArray[np.complex64]:
        """Insert the slices and return the 3D DFT normalized by the weights.

        Parameters
        ----------
        slices : (..., T, ny, nx // 2 + 1) complex array
            The rfft of the images (not fftshifted), whose origin is the origin of
            the volume. Leading dimensions are for multiple volumes.

        Returns
        -------
        (..., Z, Y, X // 2 + 1) complex64 array
            The rfft of the volumes (not fftshifted), on the same device as
            ``slices``.
        """
        backend = backend_of(slices)
        slices = backend.asarray(slices)
        expected = (self.nslices, self.slice_shape[0], self.slice_shape[1] // 2 + 1)
        if slices.shape[-3:] != expected:
            raise ValueError(
                f"Shape of slices must be (..., {expected[0]}, {expected[1]}, "
                f"{expected[2]}), got {slices.shape}."
            )
        batch_shape = slices.shape[:-3]
        direct, conjugate, normalizer, keep = self._operators(backend)
        values = slices.reshape(-1, self.nslices, expected[1] * expected[2])
        # (T * nkeep, batch) matrix of the values to be inserted
        values = values[:, :, keep].reshape(values.shape[0], -1).T
        values = values.astype(np.complex64, copy=False)
        dft = direct @ values
        dft += conjugate @ values.conj()
        dft *= normalizer[:, np.newaxis]
        return backend.xp.moveaxis(dft, -1, 0).reshape(batch_shape + self._dft_shape)

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
            The reconstructed volumes, corrected for the trilinear interpolation, on
            the same device as ``slices``.
        """
        vshape = self.volume_shape
        if output_shape is None:
            output_shape = vshape
        output_shape = _normalize_shape(output_shape, 3)
        backend = backend_of(slices)
        xp, fft = backend.xp, backend.fft
        # indices of the output, whose origin is moved to the center
        indices = [
            xp.asarray((np.arange(size) - size // 2) % length)
            for size, length in zip(output_shape, vshape)
        ]
        # The inverse transform is computed axis by axis, and each axis is cropped
        # right after the transform, so the later transforms are smaller.
        volume = fft.ifft(self.insert(slices), axis=-3, overwrite_x=True)
        volume = fft.ifft(volume[..., indices[0], :, :], axis=-2, overwrite_x=True)
        volume = fft.irfft(volume[..., indices[1], :], n=vshape[2], overwrite_x=True)
        volume = volume[..., indices[2]].astype(np.float32, copy=False)
        volume /= xp.asarray(_gridding_correction(vshape, output_shape))
        return volume


def _normalize_shape(shape: int | Sequence[int], ndim: int) -> tuple[int, ...]:
    if isinstance(shape, (int, np.integer)):
        return (int(shape),) * ndim
    out = tuple(int(s) for s in shape)
    if len(out) != ndim:
        raise ValueError(f"Shape must be an integer or a {ndim}-tuple, got {shape!r}.")
    return out


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
