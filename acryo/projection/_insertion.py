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
        self.slice_shape = _normalize_shape(slice_shape, 2)
        self.volume_shape = vshape = _normalize_shape(volume_shape, 3)
        self.nslices = rotations.shape[0]
        self._keep, freqs = _slice_frequencies(
            self.slice_shape, scale, slice_stretch(rotations).min()
        )
        nkeep = self._keep.size

        self._dft_shape = dft_shape = _rfft_shape(vshape)
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
            is_conj, rows, weights = _trilinear_corners(coords, vshape)
            for row, weight in zip(rows, weights):
                valid = weight > 0
                conj = is_conj[valid]
                entries = (row[valid], columns[valid], weight[valid])
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
        return _inverse_rfft(self.insert(slices), self.volume_shape, output_shape)


class LocalSliceInsertion:
    """Insertion of central slices whose orientations differ for each volume.

    Same as :class:`CentralSliceInsertion`, but the matrices of the slices are given
    for each volume when the slices are inserted, such as the linear maps of the
    projection around each particle of a locally distorted tilt series (see
    :meth:`TiltSeries.local_detector2tomo_at`). The insertion is computed for each
    volume, which is a few times slower than the precomputed operator of
    :class:`CentralSliceInsertion`.

    Parameters
    ----------
    slice_shape : int or (int, int)
        Shape of the images whose rfft are inserted.
    volume_shape : int or (int, int, int)
        Shape of the volume.
    scale : float, default is 1.0
        Pixel size of the volume divided by the pixel size of the images.
    """

    def __init__(
        self,
        slice_shape: int | Sequence[int],
        volume_shape: int | Sequence[int],
        scale: float = 1.0,
    ):
        self.slice_shape = _normalize_shape(slice_shape, 2)
        self.volume_shape = _normalize_shape(volume_shape, 3)
        self.scale = float(scale)
        self._dft_shape = _rfft_shape(self.volume_shape)

    def insert(
        self,
        slices: NDArray[np.complexfloating],
        rotations: NDArray[np.floating],
    ) -> NDArray[np.complex64]:
        """Insert the slices and return the 3D DFT normalized by the weights.

        Parameters
        ----------
        slices : (..., T, ny, nx // 2 + 1) complex array
            The rfft of the images (not fftshifted), whose origin is the origin of
            the volume. Leading dimensions are for multiple volumes.
        rotations : (..., T, 3, 3) array
            Matrices of the slices of each volume, same as those of
            :class:`CentralSliceInsertion`.

        Returns
        -------
        (..., Z, Y, X // 2 + 1) complex array
            The rfft of the volumes (not fftshifted).
        """
        slices = np.asarray(slices)
        rotations = np.asarray(rotations, dtype=np.float64)
        ny, nx = self.slice_shape
        if slices.ndim < 3 or slices.shape[-2:] != (ny, nx // 2 + 1):
            raise ValueError(
                f"Shape of slices must be (..., T, {ny}, {nx // 2 + 1}), got "
                f"{slices.shape}."
            )
        batch_shape, nslices = slices.shape[:-3], slices.shape[-3]
        if rotations.shape != batch_shape + (nslices, 3, 3):
            raise ValueError(
                f"Shape of rotations must be {batch_shape + (nslices, 3, 3)}, got "
                f"{rotations.shape}."
            )
        values = slices.reshape(-1, nslices, ny * (nx // 2 + 1))
        matrices = rotations.reshape(-1, nslices, 3, 3)
        out = np.empty((values.shape[0],) + self._dft_shape, dtype=np.complex64)
        for i in range(values.shape[0]):
            out[i] = self._insert_one(values[i], matrices[i])
        return out.reshape(batch_shape + self._dft_shape)

    def reconstruct(
        self,
        slices: NDArray[np.complexfloating],
        rotations: NDArray[np.floating],
        output_shape: int | Sequence[int] | None = None,
    ) -> NDArray[np.float32]:
        """Reconstruct the volumes from the slices.

        Same as :meth:`CentralSliceInsertion.reconstruct`, but the matrices of the
        slices are given for each volume (see :meth:`insert`).
        """
        dft = self.insert(slices, rotations)
        return _inverse_rfft(dft, self.volume_shape, output_shape)

    def _insert_one(
        self, values: NDArray[np.complexfloating], rotations: NDArray[np.float64]
    ) -> NDArray[np.complex64]:
        """Insert (T, ny * (nx // 2 + 1)) values of the slices of a volume."""
        matrices = _frequency_matrices(rotations)
        keep, freqs = _slice_frequencies(
            self.slice_shape, self.scale, slice_stretch(rotations).min()
        )
        coords = freqs @ matrices.astype(np.float32).transpose(0, 2, 1)
        coords = coords.reshape(-1, 3)
        values = values[:, keep].astype(np.complex64, copy=False)
        # the pixels of a stretched slice are smaller in the volume
        factors = (1 / _slice_areas(matrices)).astype(np.float32)
        values = (values * factors[:, np.newaxis]).ravel()
        # the frequencies beyond the Nyquist frequency of the volume
        inside = np.einsum("ij,ij->i", coords, coords) <= 0.25 * (1 + 1e-6) ** 2
        if not inside.all():
            coords, values = coords[inside], values[inside]
        is_conj, rows, weights = _trilinear_corners(coords, self.volume_shape)
        real = values.real
        imag = np.where(is_conj, -values.imag, values.imag)
        rows = rows.ravel()
        size = int(np.prod(self._dft_shape))
        weight_sum = np.bincount(rows, weights.ravel(), minlength=size)
        dft = np.empty(size, dtype=np.complex64)
        dft.real = np.bincount(rows, (weights * real).ravel(), minlength=size)
        dft.imag = np.bincount(rows, (weights * imag).ravel(), minlength=size)
        valid = weight_sum > 1e-3
        # normalize by the weights and by the pixel area of the images
        normalizer = np.zeros(size, dtype=np.float32)
        normalizer[valid] = 1 / weight_sum[valid] / self.scale**2
        dft *= normalizer
        return dft.reshape(self._dft_shape)


def _inverse_rfft(
    dft: NDArray[np.complex64],
    volume_shape: tuple[int, int, int],
    output_shape: int | Sequence[int] | None,
) -> NDArray[np.float32]:
    """Inverse transform of the inserted 3D DFT, cropped to the output shape."""
    vshape = volume_shape
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
    volume = sfft.ifft(dft, axis=-3, overwrite_x=True)
    volume = sfft.ifft(volume[..., indices[0], :, :], axis=-2, overwrite_x=True)
    volume = sfft.irfft(volume[..., indices[1], :], n=vshape[2], overwrite_x=True)
    volume = volume[..., indices[2]].astype(np.float32, copy=False)
    volume /= _gridding_correction(vshape, output_shape)
    return volume


def _rfft_shape(volume_shape: tuple[int, ...]) -> tuple[int, int, int]:
    return volume_shape[0], volume_shape[1], volume_shape[2] // 2 + 1


def _slice_frequencies(
    slice_shape: tuple[int, ...], scale: float, stretch_min: float
) -> tuple[NDArray[np.intp], NDArray[np.float32]]:
    """Indices of the rfft of the slices to insert, and their (z, y, x) frequencies.

    The frequencies are in cycles per pixel of the volume. Only those up to the
    Nyquist frequency of the volume are inserted. Half of the frequencies of x = 0 are
    redundant because of the Friedel symmetry.
    """
    ny, nx = slice_shape
    qy, qx = np.meshgrid(
        sfft.fftfreq(ny).astype(np.float32),
        sfft.rfftfreq(nx).astype(np.float32),
        indexing="ij",
    )
    freqs = np.stack([np.zeros_like(qy), qy, qx], axis=-1).reshape(-1, 3)
    freqs *= np.float32(scale)
    keep = (np.linalg.norm(freqs, axis=-1) <= 0.5 / stretch_min) & ~(
        (freqs[:, 2] == 0) & (freqs[:, 1] < 0)
    )
    keep = np.flatnonzero(keep)
    return keep, freqs[keep]


def _trilinear_corners(
    coords: NDArray[np.float32], vshape: tuple[int, ...]
) -> tuple[NDArray[np.bool_], NDArray[np.int64], NDArray[np.float32]]:
    """Trilinear interpolation of (N, 3) frequencies in the rfft of the volume.

    The rfft only has x >= 0, so the frequencies of x < 0 are inserted at the
    opposite frequencies as conjugate values. Returns the (N,) mask of them, and the
    (8, N) rows (indices of the flattened rfft) and weights of the 8 corners. The
    weights of the corners outside the rfft are 0.
    """
    nz, ny, nxh = _rfft_shape(vshape)
    is_conj = coords[:, 2] < 0
    coords = np.where(is_conj[:, np.newaxis], -coords, coords)
    index = coords * np.array(vshape, dtype=np.float32)
    floor = np.floor(index)
    frac = index - floor
    # |frequencies| <= 0.5, so the indices are in [-size // 2 - 1, size // 2]
    base = floor.astype(np.int32)
    # (2, N) of the two corners along each axis, wrapped around along z and y
    iz = np.where(base[:, 0] < 0, base[:, 0] + nz, base[:, 0])
    iy = np.where(base[:, 1] < 0, base[:, 1] + ny, base[:, 1])
    ix = base[:, 2]
    rows_z = np.stack([iz, np.where(iz + 1 == nz, 0, iz + 1)]).astype(np.int64)
    rows_z *= ny * nxh
    rows_y = np.stack([iy, np.where(iy + 1 == ny, 0, iy + 1)]) * nxh
    rows_x = np.stack([ix, np.minimum(ix + 1, nxh - 1)])
    weights_z = np.stack([1 - frac[:, 0], frac[:, 0]])
    weights_y = np.stack([1 - frac[:, 1], frac[:, 1]])
    weights_x = np.stack([1 - frac[:, 2], np.where(ix + 1 < nxh, frac[:, 2], 0)])
    rows = (
        rows_z[:, np.newaxis, np.newaxis]
        + (rows_y[:, np.newaxis] + rows_x[np.newaxis, :])[np.newaxis]
    )
    weights = (
        weights_z[:, np.newaxis, np.newaxis]
        * weights_y[np.newaxis, :, np.newaxis]
        * weights_x[np.newaxis, np.newaxis, :]
    )
    return is_conj, rows.reshape(8, -1), weights.reshape(8, -1)


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
