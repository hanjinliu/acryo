from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache, partial
import math
from typing import TYPE_CHECKING, Callable, Literal, NamedTuple, Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import fft as sfft

from acryo._types import nm
from acryo.ctf import CTFCorrection

if TYPE_CHECKING:
    from typing_extensions import Self
    from acryo.ctf import CTFModel


class TiltImageFilter(Protocol):
    """Protocol of a Fourier filter applied to each tilt image in preprocessing.

    The filter is called with the spatial frequencies and the index of the tilt, and
    returns the values multiplied to the Fourier transform of the tilt image. The tilt
    images are filtered patch by patch (see :class:`TiltImagePreprocessor`). Usually,
    the filter is called once for each tilt image, and the values are multiplied to
    all the patches. If the filter has the attribute ``position_dependent = True``
    (such as :class:`acryo.ctf.PatchCTFCorrection`), it is called for each patch with
    the additional keyword argument ``position``, the (y, x) coordinates of the center
    of the patch in nm from the center of the tilt image (``size // 2``).
    """

    def __call__(
        self,
        freq_y: NDArray[np.float32],
        freq_x: NDArray[np.float32],
        index: int,
    ) -> NDArray[np.floating] | float:
        """Filter values of the tilt image.

        Parameters
        ----------
        freq_y, freq_x : float32 array
            (y, x) spatial frequencies in 1/nm. They are broadcastable to the shape
            of the rfft of the (padded) patches. If the tilt image is binned in
            preprocessing, only the frequencies kept by the binning are given. Values
            are multiplied in float32, so float32 values are enough.
        index : int
            Index of the tilt in the tilt series, which corresponds to the tilt
            angles (not the index in the image file).
        """


@dataclass(frozen=True)
class _SelectedTiltsFilter:
    """A filter of the selected tilts, called with the indices before the selection."""

    filter: TiltImageFilter
    indices: tuple[int, ...]

    @property
    def position_dependent(self) -> bool:
        return _is_position_dependent(self.filter)

    def __call__(
        self,
        freq_y: NDArray[np.float32],
        freq_x: NDArray[np.float32],
        index: int,
        **kwargs,
    ) -> NDArray[np.floating] | float:
        return self.filter(freq_y, freq_x, self.indices[index], **kwargs)

    def select_tilts(self, indices: ArrayLike) -> _SelectedTiltsFilter:
        idx = np.asarray(indices, dtype=np.intp)
        return _SelectedTiltsFilter(self.filter, tuple(self.indices[i] for i in idx))


def _select_filter_tilts(
    filt: TiltImageFilter, indices: NDArray[np.intp]
) -> TiltImageFilter:
    if (select_tilts := getattr(filt, "select_tilts", None)) is not None:
        return select_tilts(indices)
    return _SelectedTiltsFilter(filt, tuple(int(i) for i in indices))


def _is_position_dependent(filt: TiltImageFilter) -> bool:
    return bool(getattr(filt, "position_dependent", False))


@dataclass(frozen=True)
class TiltImagePreprocessor:
    """Preprocessing of tilt images, applied to each tilt image independently.

    0. If the image is binned by ``real_space_binning`` or more, the image is first
       binned in real space (averaged) by the largest integer that does not exceed the
       binning factor and divides the image size. The rest of the binning, which may
       be fractional, is done in Fourier space by the following steps. The resolution
       near the Nyquist frequency of the binned image is slightly attenuated and
       aliased, but the preprocessing is much faster.
    1. A linear background plane is subtracted (if ``subtract_background``).
    2. The image is split into ``patch_grid`` patches, which overlap with their
       neighbors by ``overlap`` pixels. Each patch is extended by ``padding`` pixels
       of the image around it (mirror-padded outside the image) with a soft mask over
       the padding, to suppress the edge artifacts of filtering, and zero-padded to a
       size suitable for FFT.
    3. Each patch is filtered in Fourier space by the bandpass filter and
       ``filters``, such as :class:`acryo.ctf.CTFCorrection`. Filters that depend on
       the position, such as :class:`acryo.ctf.PatchCTFCorrection`, are evaluated at
       the center of each patch. If the image is binned, the Fourier transform of each
       patch is cropped to the binned size before the inverse transform (Fourier
       binning).
    4. The padding of the patches is cropped, and the patches are blended with
       smooth weights in the overlapping regions.
    5. The image is normalized to zero mean and unit variance on the central crop (if
       ``normalize``), or the mean is subtracted.

    The preprocessing is ported from ``preprocess_tilt_series_images`` of
    `torch-tilt-series <https://github.com/teamtomo/torch-tilt-series>`_, with the
    additional ``filters`` and patches. The patches need less memory than the whole
    image, and the filters can depend on the position in the image.

    Parameters
    ----------
    low, high : float
        Cutoff frequencies of the bandpass filter in cycles per pixel (0 to 0.5).
        If ``low`` is 0, the mean of each patch is kept and the mean of the whole
        image is subtracted, so the DC component is always excluded.
    falloff : float, default is 0.0
        Width of the cosine falloff at each cutoff, in cycles per pixel.
    padding : int, default is 128
        Width of the margin of each patch in pixels.
    subtract_background : bool, default is True
        Whether to subtract the background plane.
    normalize : bool, default is True
        Whether to normalize the image on the central crop.
    filters : sequence of callable, optional
        Additional Fourier filters. See :class:`TiltImageFilter`.
    patch_grid : int or (int, int), default is (6, 6)
        Number of patches along y and x. ``(1, 1)`` filters the whole image at once.
    overlap : int, default is 64
        Width of the overlap of neighboring patches in pixels, over which the
        filtered patches are blended.
    real_space_binning : int, optional, default is 4
        Binning factor from which the image is binned in real space first (see
        above). If None, the image is binned only in Fourier space.

    Examples
    --------
    >>> from acryo.ctf import CTFModel
    >>> ctf = CTFModel.from_kv(300, spherical_aberration=2.7)
    >>> preprocessor = TiltImagePreprocessor().with_ctf(ctf, defocus=defocus_list)
    """

    low: float = 0.0
    high: float = 0.5
    falloff: float = 0.0
    padding: int = 128
    subtract_background: bool = True
    normalize: bool = True
    filters: tuple[TiltImageFilter, ...] = ()
    patch_grid: tuple[int, int] = (6, 6)
    overlap: int = 64
    real_space_binning: int | None = 4

    def __post_init__(self):
        if self.padding < 0:
            raise ValueError(f"'padding' must be non-negative, got {self.padding}.")
        if self.falloff < 0:
            raise ValueError(f"'falloff' must be non-negative, got {self.falloff}.")
        if self.overlap < 0:
            raise ValueError(f"'overlap' must be non-negative, got {self.overlap}.")
        grid = self.patch_grid
        grid = (grid, grid) if isinstance(grid, int) else tuple(grid)
        if len(grid) != 2 or any(int(n) != n or n < 1 for n in grid):
            raise ValueError(
                f"'patch_grid' must be a positive int or a pair of them, got "
                f"{self.patch_grid!r}."
            )
        object.__setattr__(self, "patch_grid", (int(grid[0]), int(grid[1])))
        object.__setattr__(self, "filters", tuple(self.filters))
        if self.real_space_binning is not None and self.real_space_binning < 2:
            raise ValueError(
                "'real_space_binning' must be None or >= 2, got "
                f"{self.real_space_binning!r}."
            )

    def replace(self, **kwargs) -> Self:
        """Return a new preprocessor with the given parameters replaced."""
        return dataclasses.replace(self, **kwargs)

    def with_filters(self, *filters: TiltImageFilter) -> Self:
        """Return a new preprocessor with additional Fourier filters."""
        return self.replace(filters=self.filters + filters)

    def with_ctf(
        self,
        ctf: CTFModel | Sequence[CTFModel],
        defocus: ArrayLike | None = None,
        method: Literal["phase_flip", "wiener"] = "phase_flip",
        wiener_constant: float = 0.1,
    ) -> Self:
        """Return a new preprocessor with CTF correction.

        Parameters
        ----------
        ctf : CTFModel or sequence of CTFModel
            CTF model(s). A sequence is the models of each tilt.
        defocus : (T,) array, optional
            Defocus of each tilt in μm, which replaces the defocus of ``ctf``.
        method : {"phase_flip", "wiener"}, default is "phase_flip"
            Method of CTF correction.
        wiener_constant : float, default is 0.1
            Regularization of the Wiener filter.
        """
        if defocus is None:
            correction = CTFCorrection(
                ctf, method=method, wiener_constant=wiener_constant  # type: ignore
            )
        else:
            correction = CTFCorrection.from_defocus(
                ctf, defocus, method=method, wiener_constant=wiener_constant  # type: ignore
            )
        return self.with_filters(correction)

    def select_tilts(self, indices: ArrayLike) -> Self:
        """Return the preprocessor of the selected tilts.

        The ``filters`` are converted so that each tilt of the selection is filtered
        as before, such as by the CTF model of the tilt. A filter that has the
        ``select_tilts`` method is converted by it, and the others are called with the
        indices of the tilts before the selection.

        Parameters
        ----------
        indices : sequence of int
            Indices of the tilts to keep, as :meth:`TiltSeries.select_tilts`.
        """
        idx = np.asarray(indices, dtype=np.intp)
        return self.replace(
            filters=tuple(_select_filter_tilts(filt, idx) for filt in self.filters)
        )

    def __call__(
        self,
        image: NDArray[np.floating],
        index: int = 0,
        scale: nm = 1.0,
        binsize: int = 1,
    ) -> NDArray[np.float32]:
        """Preprocess a tilt image.

        Parameters
        ----------
        image : (H, W) array
            The tilt image.
        index : int, default is 0
            Index of the tilt in the tilt series, which is passed to ``filters``.
        scale : float, default is 1.0
            Pixel size of the image in nm, which is used by ``filters``.
        binsize : int, default is 1
            If larger than 1, the preprocessed image is binned by Fourier cropping of
            each patch (after binning in real space if ``binsize`` is
            ``real_space_binning`` or more). The image is cropped around the center
            in the same way as :func:`fourier_bin`, and the binned image is
            normalized by its own statistics.
        """
        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 2:
            raise ValueError(f"Image must be 2D, got shape {image.shape}.")
        if binsize < 1 or int(binsize) != binsize:
            raise ValueError(f"'binsize' must be a positive integer, got {binsize!r}.")
        binsize = int(binsize)
        image_shape = image.shape
        prebin = _real_space_binsize(image_shape, binsize, self.real_space_binning)
        if prebin > 1:
            image = _bin_real_space(image, prebin)
        layouts = [
            _axis_layout(size, binsize, n, self.overlap, self.padding, prebin=prebin)
            for size, n in zip(image_shape, self.patch_grid)
        ]
        fft_shape = (layouts[0].fft_size, layouts[1].fft_size)
        # the Fourier transform of the patches cropped to the binned size
        shape = (layouts[0].crop_size, layouts[1].crop_size)
        spacing = scale * binsize
        freq_y = sfft.fftfreq(shape[0], spacing).astype(np.float32)[:, np.newaxis]
        freq_x = sfft.rfftfreq(shape[1], spacing).astype(np.float32)[np.newaxis, :]
        weight = _patch_bandpass(shape, binsize, self.low, self.high, self.falloff)
        if shape != fft_shape:
            # the inverse FFT of the smaller size is larger by the ratio of the sizes
            ratio = shape[0] * shape[1] / (fft_shape[0] * fft_shape[1])
            weight = weight * np.float32(ratio)
        # position-dependent filters of this tilt as functions of the position
        local_filters: list[Callable[..., NDArray[np.floating] | float]] = []
        for filt in self.filters:
            if not _is_position_dependent(filt):
                weight = weight * np.asarray(filt(freq_y, freq_x, index), np.float32)
            elif (for_tilt := getattr(filt, "_for_tilt", None)) is not None:
                local_filters.append(for_tilt(freq_y, freq_x, index))
            else:
                local_filters.append(partial(filt, freq_y, freq_x, index))
        terms = _plane_terms(image) if self.subtract_background else None
        center = np.array(image_shape) // 2
        out = np.zeros((layouts[0].size_out, layouts[1].size_out), dtype=np.float32)
        buffer = np.empty(fft_shape, dtype=np.float32)
        for patch_y in layouts[0].patches:
            for patch_x in layouts[1].patches:
                _read_patch(image, (patch_y, patch_x), terms, buffer)
                ft = sfft.rfft2(buffer)
                if shape != fft_shape:
                    ft = _crop_spectrum(ft, shape)
                if patch_y.shift or patch_x.shift:
                    _shift_spectrum(ft, fft_shape, patch_y.shift, patch_x.shift)
                ft *= weight
                if local_filters:
                    position = (
                        float(patch_y.center - center[0]) * scale,
                        float(patch_x.center - center[1]) * scale,
                    )
                    for func in local_filters:
                        ft *= np.asarray(func(position=position), dtype=np.float32)
                filtered = sfft.irfft2(ft, s=shape, overwrite_x=True)
                _blend_patch(out, filtered, patch_y, patch_x)
        if self.normalize:
            mean, std = _central_crop_stats(out)
            out -= mean
            out /= std
        else:
            out -= np.float32(out.mean(dtype=np.float64))
        return out

    def _workspace_nbytes(self, image_shape: tuple[int, int]) -> int:
        """Approximate memory in bytes used to preprocess an image (without binning)."""
        sizes = [
            _axis_layout(size, 1, n, self.overlap, self.padding).fft_size
            for size, n in zip(image_shape, self.patch_grid)
        ]
        # the output, and the patch with its Fourier transform and filters
        return image_shape[0] * image_shape[1] * 4 + sizes[0] * sizes[1] * 4 * 5


def fourier_bin(image: NDArray[np.floating], binsize: int) -> NDArray[np.float32]:
    """Bin a 2D image by Fourier cropping, keeping the position of the image center.

    The image center (``size // 2``) is the origin of the detector, so the image is
    first cropped symmetrically around the center to a multiple of ``2 * binsize``.
    The pixel size of the output is exactly ``binsize`` times larger, and the mean
    is preserved. This is ported from ``fourier_rescale_2d`` of
    `torch-fourier-rescale <https://github.com/teamtomo/torch-fourier-rescale>`_.
    """
    image = np.asarray(image, dtype=np.float32)
    if binsize == 1:
        return image
    if binsize < 1 or int(binsize) != binsize:
        raise ValueError(f"'binsize' must be a positive integer, got {binsize!r}.")
    slices: list[slice] = []
    target: list[int] = []
    for size in image.shape:
        cropped = size - size % (2 * binsize)
        if cropped == 0:
            raise ValueError(f"Image of shape {image.shape} is too small to bin.")
        start = size // 2 - cropped // 2
        slices.append(slice(start, start + cropped))
        target.append(cropped // binsize)
    # The cropped size is ``binsize`` times the even output size, so the center is
    # kept without shifting the origin to the center.
    ft = sfft.rfft2(image[tuple(slices)])
    th, tw = target
    ft = np.concatenate([ft[: th // 2], ft[ft.shape[0] - th // 2 :]], axis=0)
    ft = ft[:, : tw // 2 + 1]
    out = sfft.irfft2(ft, s=(th, tw), overwrite_x=True)
    out /= binsize**2
    return out.astype(np.float32, copy=False)


class _AxisPatch(NamedTuple):
    """A patch along one axis of the image."""

    start: int  # index of the first pixel of the extended patch in the image
    length: int  # number of pixels read from the image (mirrored outside)
    out_start: int  # output pixels written by the patch
    out_stop: int
    offset: int  # index of the pixel ``out_start`` in the (binned) patch
    weight: NDArray[np.float32]  # blending weights of the output pixels
    blend: tuple[int, int]  # number of the weights < 1 at the start and the stop
    center: float  # position of the center of the patch in the original image
    taper: NDArray[np.float32]  # soft mask of the pixels read from the image
    margins: tuple[int, int]  # number of the taper values < 1 at the start and stop
    shift: float  # shift (pixel) of the samples of the binned patch from the grid


class _AxisLayout(NamedTuple):
    """Layout of the patches along one axis of the image."""

    size_out: int  # size of the output (binned) image
    fft_size: int  # size of the patches for FFT (before Fourier binning)
    crop_size: int  # size of the patches after Fourier binning
    patches: tuple[_AxisPatch, ...]


def _axis_layout(
    size: int,
    binsize: int,
    npatches: int,
    overlap: int,
    padding: int,
    prebin: int = 1,
) -> _AxisLayout:
    """Layout of the overlapping patches along an axis of the image.

    The output pixel ``j`` is at ``size // 2 + (j - size_out // 2) * binsize`` of the
    image, where the image is cropped around the center as :func:`fourier_bin`. The
    output is split into ``npatches`` cores, and each patch covers its core and half
    the overlap beyond it. Each patch is extended by ``padding`` (rounded up to a
    multiple of ``binsize``).

    The patches are read from the image binned by ``prebin`` in real space (``prebin``
    divides ``size``), whose pixel ``i`` is at ``i * prebin + (prebin - 1) / 2`` of
    the image. The output pixels are ``binsize / prebin = p / q`` pixels apart in the
    binned image, so the Fourier transform of each patch of the size ``p * m`` is
    cropped to the size ``q * m``, whose samples are on the grid of the output after
    shifting the patch by ``shift`` (less than a pixel). Without real space binning,
    ``shift`` is always 0.
    """
    if binsize == 1:
        size_out = size
    else:
        cropped = size - size % (2 * binsize)
        if cropped == 0:
            raise ValueError(f"Image of size {size} is too small to bin by {binsize}.")
        size_out = cropped // binsize
    if size % prebin != 0:
        raise ValueError(f"Size {size} is not divisible by {prebin}.")
    # distance between the output pixels in the pixels of the binned image
    step = Fraction(binsize, prebin)

    def _position(j: int) -> Fraction:
        """Position of the output pixel ``j`` in the image binned by ``prebin``."""
        position = size // 2 + (j - size_out // 2) * binsize
        return (position - Fraction(prebin - 1, 2)) / prebin

    npatches = min(npatches, size_out)
    bounds = [i * size_out // npatches for i in range(npatches + 1)]
    min_core = min(b - a for a, b in zip(bounds[:-1], bounds[1:]))
    # half width of the blending region in output pixels
    half = min(-(-overlap // (2 * binsize)), min_core // 2)
    # padding in output pixels and in the pixels of the binned image
    npad = -(-max(min(padding, size - 1), 0) // binsize)
    pad = npad * step
    spans = [
        (
            bounds[i] - half if i > 0 else 0,
            bounds[i + 1] + half if i < npatches - 1 else size_out,
        )
        for i in range(npatches)
    ]
    # the first pixel of each patch, and the shift of the first output pixel from it
    firsts = [_position(lo - npad) for lo, _ in spans]
    starts = [math.floor(first) for first in firsts]
    nsamples = [hi - lo + 2 * npad for lo, hi in spans]
    lengths = [
        math.ceil(first + n * step) - start
        for first, start, n in zip(firsts, starts, nsamples)
    ]
    # FFT of the size p * m is cropped to the even size q * m
    p, q = step.numerator, step.denominator
    m = max(-(-max(lengths) // p), -(-max(nsamples) // q), 1)
    while (q * m) % 2 != 0 or not _is_5_smooth(m):
        m += 1
    blend_t = (np.arange(2 * half) + 0.5) / max(2 * half, 1)
    patches = []
    for i, (lo, hi) in enumerate(spans):
        weight = np.ones(hi - lo, dtype=np.float32)
        nstart = 2 * half if i > 0 else 0
        nstop = 2 * half if i < npatches - 1 else 0
        if nstart > 0:
            weight[:nstart] = np.sin(np.pi / 2 * blend_t) ** 2
        if nstop > 0:
            weight[weight.size - nstop :] = np.cos(np.pi / 2 * blend_t) ** 2
        shift = firsts[i] - starts[i]
        # the output pixels are in [inner_start, inner_stop) of the read pixels
        inner_start = shift + pad
        taper, margins = _taper(
            lengths[i], inner_start, inner_start + (hi - lo) * step, pad
        )
        patches.append(
            _AxisPatch(
                start=starts[i],
                length=lengths[i],
                out_start=lo,
                out_stop=hi,
                offset=npad,
                weight=weight,
                blend=(nstart, nstop),
                center=size // 2
                + ((bounds[i] + bounds[i + 1] - 1) / 2 - size_out // 2) * binsize,
                taper=taper,
                margins=margins,
                shift=float(shift),
            )
        )
    return _AxisLayout(size_out, p * m, q * m, tuple(patches))


def _taper(
    length: int, inner_start: Fraction, inner_stop: Fraction, width: Fraction
) -> tuple[NDArray[np.float32], tuple[int, int]]:
    """Soft mask that is 1 in ``[inner_start, inner_stop)`` with cosine edges.

    The mask falls to 0 over ``width`` pixels outside. The number of the values < 1
    at the start and the stop are also returned.
    """
    index = np.arange(length, dtype=np.float64)
    distance = np.maximum(
        np.maximum(float(inner_start) - index, index - float(inner_stop) + 1), 0
    )
    if width > 0:
        taper = np.cos(np.pi / 2 * np.minimum(distance / float(width), 1))
    else:
        taper = np.ones(length)
    inner = np.flatnonzero(taper >= 1)
    if inner.size == 0:
        margins = (length, 0)
    else:
        margins = (int(inner[0]), int(length - 1 - inner[-1]))
    return taper.astype(np.float32), margins


def _is_5_smooth(n: int) -> bool:
    for p in (2, 3, 5):
        while n % p == 0:
            n //= p
    return n == 1


def _reflect_indices(start: int, length: int, size: int) -> NDArray[np.intp]:
    """Indices of the image mirrored outside, same as ``np.pad(mode="reflect")``."""
    if size == 1:
        return np.zeros(length, dtype=np.intp)
    period = 2 * (size - 1)
    idx = np.arange(start, start + length) % period
    return np.where(idx >= size, period - idx, idx)


def _reflect_runs(
    start: int, length: int, size: int
) -> list[tuple[slice, slice]] | None:
    """Pairs of the slices of the destination and the source of mirrored indices.

    None is returned if the indices are mirrored more than once.
    """
    stop = start + length
    if start < 1 - size or stop > 2 * size - 1 or stop <= 0 or start >= size:
        return None
    runs = []
    if start < 0:
        runs.append((slice(0, -start), slice(-start, 0, -1)))
    lo, hi = max(start, 0), min(stop, size)
    runs.append((slice(lo - start, hi - start), slice(lo, hi)))
    if stop > size:
        end = 2 * size - 2 - stop
        runs.append(
            (
                slice(size - start, length),
                slice(size - 2, end if end >= 0 else None, -1),
            )
        )
    return runs


def _read_patch(
    image: NDArray[np.float32],
    patches: tuple[_AxisPatch, _AxisPatch],
    terms: tuple[NDArray[np.float32], NDArray[np.float32]] | None,
    out: NDArray[np.float32],
) -> None:
    """Read the extended patch into ``out``, with the soft mask and zero padding."""
    patch_y, patch_x = patches
    ny, nx = patch_y.length, patch_x.length
    view = out[:ny, :nx]
    iy = _reflect_indices(patch_y.start, ny, image.shape[0])
    ix = _reflect_indices(patch_x.start, nx, image.shape[1])
    if terms is None:
        term_y = np.zeros((ny, 1), dtype=np.float32)
    else:
        term_y = terms[0][iy, np.newaxis]
    runs_y = _reflect_runs(patch_y.start, ny, image.shape[0])
    runs_x = _reflect_runs(patch_x.start, nx, image.shape[1])
    if runs_y is None or runs_x is None:
        np.subtract(image[np.ix_(iy, ix)], term_y, out=view)
    else:
        # the background plane is subtracted while reading
        for dst_y, src_y in runs_y:
            for dst_x, src_x in runs_x:
                np.subtract(image[src_y, src_x], term_y[dst_y], out=view[dst_y, dst_x])
    if terms is not None:
        view -= terms[1][np.newaxis, ix]
    # only the pixels in the margins are masked
    for patch, axis in [(patch_y, 0), (patch_x, 1)]:
        n0, n1 = patch.margins
        if n0 > 0:
            _multiply_along(view, patch.taper, slice(0, n0), axis)
        if n1 > 0:
            _multiply_along(view, patch.taper, slice(patch.length - n1, None), axis)
    out[ny:] = 0.0
    out[:ny, nx:] = 0.0


def _shift_spectrum(
    ft: NDArray[np.complex64],
    fft_shape: tuple[int, int],
    shift_y: float,
    shift_x: float,
) -> None:
    """Shift the (cropped) rfft of a patch of ``fft_shape`` by ``-shift`` in place.

    The samples of the inverse transform are moved by ``shift`` pixels of the patch.
    """
    ny, nx = ft.shape[0], (ft.shape[1] - 1) * 2
    if shift_y:
        freq = sfft.fftfreq(ny) * ny / fft_shape[0]  # cycles per pixel of the patch
        ft *= np.exp(2j * np.pi * freq * shift_y).astype(np.complex64)[:, np.newaxis]
    if shift_x:
        freq = sfft.rfftfreq(nx) * nx / fft_shape[1]
        ft *= np.exp(2j * np.pi * freq * shift_x).astype(np.complex64)[np.newaxis, :]


def _real_space_binsize(
    shape: tuple[int, ...], binsize: int, threshold: int | None
) -> int:
    """The largest factor <= binsize that divides the shape, if binsize >= threshold."""
    if threshold is None or binsize < threshold:
        return 1
    for factor in range(binsize, 1, -1):
        if all(size % factor == 0 for size in shape):
            return factor
    return 1


def _bin_real_space(image: NDArray[np.float32], factor: int) -> NDArray[np.float32]:
    """Bin the image (divisible by the factor) by averaging in real space."""
    h, w = image.shape
    # rows first (contiguous), and then the columns of the smaller array
    rows = image.reshape(h // factor, factor, w).sum(axis=1)
    out = rows[:, ::factor].copy()
    for k in range(1, factor):
        out += rows[:, k::factor]
    out /= factor**2
    return out


def _crop_spectrum(
    ft: NDArray[np.complex64], shape: tuple[int, int]
) -> NDArray[np.complex64]:
    """Crop the rfft to that of the shape (even sizes), same as :func:`fourier_bin`."""
    ny, nx = shape
    out = np.empty((ny, nx // 2 + 1), dtype=ft.dtype)
    out[: ny // 2] = ft[: ny // 2, : nx // 2 + 1]
    out[ny // 2 :] = ft[ft.shape[0] - ny // 2 :, : nx // 2 + 1]
    return out


def _blend_patch(
    out: NDArray[np.float32],
    filtered: NDArray[np.float32],
    patch_y: _AxisPatch,
    patch_x: _AxisPatch,
) -> None:
    """Add the filtered patch to the output with the blending weights."""
    ny = patch_y.out_stop - patch_y.out_start
    nx = patch_x.out_stop - patch_x.out_start
    region = filtered[
        patch_y.offset : patch_y.offset + ny, patch_x.offset : patch_x.offset + nx
    ]
    # only the pixels in the overlapping regions are weighted
    for weight, (n0, n1), axis in [
        (patch_y.weight, patch_y.blend, 0),
        (patch_x.weight, patch_x.blend, 1),
    ]:
        if n0 > 0:
            _multiply_along(region, weight, slice(0, n0), axis)
        if n1 > 0:
            _multiply_along(region, weight, slice(weight.size - n1, weight.size), axis)
    out[
        patch_y.out_start : patch_y.out_stop, patch_x.out_start : patch_x.out_stop
    ] += region


def _multiply_along(
    arr: NDArray[np.float32], weight: NDArray[np.float32], sl: slice, axis: int
) -> None:
    if axis == 0:
        arr[sl] *= weight[sl, np.newaxis]
    else:
        arr[:, sl] *= weight[np.newaxis, sl]


def _subtract_plane(image: NDArray[np.float32]) -> NDArray[np.float32]:
    """Subtract the least-squares plane ``a * y + b * x + c``."""
    y_term, x_term = _plane_terms(image)
    out = image - y_term[:, np.newaxis]
    out -= x_term[np.newaxis, :]
    return out


def _plane_terms(
    image: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """The terms ``a * y + c`` and ``b * x`` of the least-squares plane."""
    h, w = image.shape
    y = np.arange(h, dtype=np.float64)
    x = np.arange(w, dtype=np.float64)
    rows = image.sum(axis=1, dtype=np.float64)
    cols = image.sum(axis=0, dtype=np.float64)
    # normal equations of the design matrix [y, x, 1]
    lhs = np.array(
        [
            [w * (y**2).sum(), y.sum() * x.sum(), w * y.sum()],
            [y.sum() * x.sum(), h * (x**2).sum(), h * x.sum()],
            [w * y.sum(), h * x.sum(), h * w],
        ]
    )
    rhs = np.array([y @ rows, x @ cols, rows.sum()])
    a, b, c = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    return (a * y + c).astype(np.float32), (b * x).astype(np.float32)


def _central_crop_stats(image: NDArray[np.float32]) -> tuple[np.float32, np.float32]:
    """Mean and standard deviation of the central 25% crop."""
    h, w = image.shape
    crop = image[int(0.375 * h) : int(0.625 * h), int(0.375 * w) : int(0.625 * w)]
    # accumulated in float64 without float64 copies of the crop
    mean = np.float32(crop.mean(dtype=np.float64))
    var = np.square(crop - mean).mean(dtype=np.float64)
    return mean, np.float32(math.sqrt(var))


@lru_cache(maxsize=4)
def _patch_bandpass(
    shape: tuple[int, int], binsize: int, low: float, high: float, falloff: float
) -> NDArray[np.float32]:
    """Bandpass filter of the binned rfft of the patches.

    The cutoffs are in cycles per pixel of the image before binning. If ``low`` is 0,
    the mean of the patch is kept, and the mean of the whole image is subtracted later.
    """
    out = _bandpass(shape, low * binsize, high * binsize, falloff * binsize).copy()
    if low <= 0:
        out[0, 0] = 1.0
    out.setflags(write=False)
    return out


@lru_cache(maxsize=2)
def _bandpass(
    shape: tuple[int, int], low: float, high: float, falloff: float
) -> NDArray[np.float32]:
    """Bandpass filter for the rfft of an image of the shape.

    Ported from ``bandpass_filter`` of torch-fourier-filter: 1 on ``(low, high]``
    with cosine falloff of the width ``falloff`` outside.
    """
    freq_y = sfft.fftfreq(shape[0]).astype(np.float32)[:, np.newaxis]
    freq_x = sfft.rfftfreq(shape[1]).astype(np.float32)[np.newaxis, :]
    freq = np.sqrt(freq_y**2 + freq_x**2)
    band = (freq > low) & (freq <= high)
    out = band.astype(np.float32)
    if falloff > 0:
        soft = (freq > low - falloff) & (freq <= high + falloff) & ~band
        distance = np.minimum(np.abs(freq[soft] - low), np.abs(freq[soft] - high))
        out[soft] = np.cos(distance / falloff * (math.pi / 2))
    out.setflags(write=False)
    return out
