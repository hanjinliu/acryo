from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from functools import lru_cache
import math
from typing import TYPE_CHECKING, Literal, Protocol, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import fft as sfft

from acryo._types import nm

if TYPE_CHECKING:
    from typing_extensions import Self
    from acryo.ctf import CTFModel


class TiltImageFilter(Protocol):
    """Protocol of a Fourier filter applied to each tilt image in preprocessing.

    The filter is called with the spatial frequencies and the index of the tilt, and
    returns the values multiplied to the Fourier transform of the tilt image.
    """

    def __call__(
        self,
        freq_y: NDArray[np.float64],
        freq_x: NDArray[np.float64],
        index: int,
    ) -> NDArray[np.floating] | float:
        """Filter values of the tilt image.

        Parameters
        ----------
        freq_y, freq_x : array
            (y, x) spatial frequencies in 1/nm. They are broadcastable to the shape
            of the rfft of the (padded) image.
        index : int
            Index of the tilt in the tilt series, which corresponds to the tilt
            angles (not the index in the image file).
        """


@dataclass(frozen=True)
class CTFCorrection:
    """CTF correction of each tilt image, as a filter in preprocessing.

    The CTF is corrected by phase flipping or Wiener filtering of the whole tilt
    image. The variation of the defocus within a tilted image is not considered.

    Parameters
    ----------
    models : CTFModel or sequence of CTFModel
        CTF models of each tilt. If only one model is given, it is used for all the
        tilts.
    method : {"phase_flip", "wiener"}, default is "phase_flip"
        Method of CTF correction. ``"phase_flip"`` multiplies the sign of the CTF,
        and ``"wiener"`` multiplies ``ctf / (ctf^2 + wiener_constant)``.
    wiener_constant : float, default is 0.1
        Regularization of the Wiener filter (inverse of the SNR).
    """

    models: tuple[CTFModel, ...]
    method: Literal["phase_flip", "wiener"] = "phase_flip"
    wiener_constant: float = 0.1

    def __post_init__(self):
        from acryo.ctf import CTFModel

        models = self.models
        models = (models,) if isinstance(models, CTFModel) else tuple(models)
        if len(models) == 0:
            raise ValueError("At least one CTF model is required.")
        if self.method not in ("phase_flip", "wiener"):
            raise ValueError(
                f"'method' must be 'phase_flip' or 'wiener', got {self.method!r}."
            )
        if self.wiener_constant <= 0:
            raise ValueError(
                f"'wiener_constant' must be positive, got {self.wiener_constant}."
            )
        object.__setattr__(self, "models", models)

    @classmethod
    def from_defocus(
        cls,
        model: CTFModel,
        defocus: ArrayLike,
        method: Literal["phase_flip", "wiener"] = "phase_flip",
        wiener_constant: float = 0.1,
    ) -> CTFCorrection:
        """Create the CTF correction from a model and the defocus of each tilt.

        Parameters
        ----------
        model : CTFModel
            CTF model whose defocus is replaced.
        defocus : (T,) array
            Defocus of each tilt in μm. Negative value means underfocus.
        """
        models = tuple(
            dataclasses.replace(model, defocus=float(d))
            for d in np.atleast_1d(np.asarray(defocus, dtype=np.float64))
        )
        return cls(models, method=method, wiener_constant=wiener_constant)

    def __call__(
        self,
        freq_y: NDArray[np.float64],
        freq_x: NDArray[np.float64],
        index: int,
    ) -> NDArray[np.float64]:
        if len(self.models) == 1:
            model = self.models[0]
        elif 0 <= index < len(self.models):
            model = self.models[index]
        else:
            raise IndexError(
                f"No CTF model for the tilt {index} ({len(self.models)} models)."
            )
        ctf = model.simulate(np.sqrt(freq_y**2 + freq_x**2))
        if self.method == "phase_flip":
            return np.sign(ctf)
        return ctf / (ctf**2 + self.wiener_constant)


@dataclass(frozen=True)
class TiltImagePreprocessor:
    """Preprocessing of tilt images, applied to each tilt image independently.

    1. A linear background plane is subtracted (if ``subtract_background``).
    2. The image is mirror-padded by ``padding`` pixels with a soft rectangular mask
       over the padding, to suppress the edge artifacts of filtering.
    3. The image is filtered in Fourier space by the bandpass filter and
       ``filters``, such as :class:`CTFCorrection`.
    4. The padding is cropped, and the image is normalized to zero mean and unit
       variance on the central crop (if ``normalize``).

    Without ``filters``, this is the same as
    ``torch_tilt_series.preprocess_tilt_series_images``.

    Parameters
    ----------
    low, high : float
        Cutoff frequencies of the bandpass filter in cycles per pixel (0 to 0.5).
        The DC component is always excluded.
    falloff : float, default is 0.0
        Width of the cosine falloff at each cutoff, in cycles per pixel.
    padding : int, default is 128
        Width of the mirror padding in pixels.
    subtract_background : bool, default is True
        Whether to subtract the background plane.
    normalize : bool, default is True
        Whether to normalize the image on the central crop.
    filters : sequence of callable, optional
        Additional Fourier filters. See :class:`TiltImageFilter`.

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

    def __post_init__(self):
        if self.padding < 0:
            raise ValueError(f"'padding' must be non-negative, got {self.padding}.")
        if self.falloff < 0:
            raise ValueError(f"'falloff' must be non-negative, got {self.falloff}.")
        object.__setattr__(self, "filters", tuple(self.filters))

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
            If larger than 1, the preprocessed image is binned by :func:`fourier_bin`.
            The image is preprocessed at the original pixel size, so this is the same
            as binning the output, but faster.
        """
        image = np.asarray(image, dtype=np.float32)
        if image.ndim != 2:
            raise ValueError(f"Image must be 2D, got shape {image.shape}.")
        h, w = image.shape
        pad = max(0, min(self.padding, h - 1, w - 1))
        padded = _pad_reflect(image, pad, self.subtract_background)
        padded *= _soft_rectangle((h, w), pad)
        shape = padded.shape
        ft = sfft.rfft2(padded)
        del padded
        ft *= _bandpass(shape, self.low, self.high, self.falloff)
        if self.filters:
            freq_y = sfft.fftfreq(shape[0], scale)[:, np.newaxis]
            freq_x = sfft.rfftfreq(shape[1], scale)[np.newaxis, :]
            for filt in self.filters:
                ft *= np.asarray(filt(freq_y, freq_x, index), dtype=np.float32)
        out = sfft.irfft2(ft, s=shape)[pad : pad + h, pad : pad + w]
        del ft
        if self.normalize:
            # Binning is linear and preserves constants, so the binned image can be
            # normalized by the statistics of the original one.
            mean, std = _central_crop_stats(out)
            out = fourier_bin(out, binsize)
            out -= mean
            out /= std
        else:
            out = fourier_bin(out, binsize)
        return out.astype(np.float32, copy=False)


def fourier_bin(image: NDArray[np.floating], binsize: int) -> NDArray[np.float32]:
    """Bin a 2D image by Fourier cropping, keeping the position of the image center.

    The image center (``size // 2``) is the origin of the detector, so the image is
    first cropped symmetrically around the center to a multiple of ``2 * binsize``.
    The pixel size of the output is exactly ``binsize`` times larger, and the mean
    is preserved. This is the same as ``torch_fourier_rescale.fourier_rescale_2d``.
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


def _pad_reflect(
    image: NDArray[np.float32], pad: int, subtract_background: bool
) -> NDArray[np.float32]:
    """Mirror-pad the image (optionally without the background plane).

    Same as ``np.pad(image, pad, mode="reflect")`` but the output is filled in place.
    """
    h, w = image.shape
    padded = np.empty((h + 2 * pad, w + 2 * pad), dtype=np.float32)
    center = padded[pad : pad + h, pad : pad + w]
    if subtract_background:
        y_term, x_term = _plane_terms(image)
        np.subtract(image, y_term[:, np.newaxis], out=center)
        center -= x_term[np.newaxis, :]
    else:
        center[:] = image
    if pad > 0:
        # reflect without repeating the edges, along y and then along x
        rows = slice(pad, pad + w)
        padded[:pad, rows] = padded[2 * pad : pad : -1, rows]
        padded[pad + h :, rows] = padded[
            pad + h - 2 : h - 2 if h > 1 else None : -1, rows
        ]
        padded[:, :pad] = padded[:, 2 * pad : pad : -1]
        padded[:, pad + w :] = padded[:, pad + w - 2 : w - 2 if w > 1 else None : -1]
    return padded


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
    return np.float32(crop.mean(dtype=np.float64)), np.float32(
        crop.std(dtype=np.float64)
    )


@lru_cache(maxsize=2)
def _soft_rectangle(shape: tuple[int, int], pad: int) -> NDArray[np.float32]:
    """The mask of the image region in the padded image, with a cosine soft edge.

    Same as ``torch_grid_utils.rectangle`` with ``smoothing_radius=pad``.
    """
    distances = []
    for size in shape:
        # the rectangle is strictly within half the size from the center
        coords = np.arange(size + 2 * pad) - (size + 2 * pad) // 2
        inside = np.flatnonzero((coords > -size / 2) & (coords < size / 2))
        index = np.arange(size + 2 * pad)
        distances.append(
            np.maximum(np.maximum(inside[0] - index, index - inside[-1]), 0)
        )
    dist = np.hypot(distances[0][:, np.newaxis], distances[1][np.newaxis, :])
    mask = (dist == 0).astype(np.float32)
    if pad > 0:
        edge = (dist > 0) & (dist <= pad)
        mask[edge] = np.cos(np.pi / 2 * dist[edge] / pad)
    mask.setflags(write=False)
    return mask


@lru_cache(maxsize=2)
def _bandpass(
    shape: tuple[int, int], low: float, high: float, falloff: float
) -> NDArray[np.float32]:
    """Bandpass filter for the rfft of an image of the shape.

    Same as ``torch_fourier_filter.bandpass.bandpass_filter``: 1 on ``(low, high]``
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
