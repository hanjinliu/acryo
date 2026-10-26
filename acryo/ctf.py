from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, ClassVar, Literal
import warnings

import numpy as np
from numpy.typing import ArrayLike, NDArray
from acryo._typed_scipy import fftn, ifftn
from acryo._types import nm
from acryo.deconv import wiener_deconv

if TYPE_CHECKING:
    from typing_extensions import Self


@dataclass
class CTFModel:
    """A model for Contrast Transfer Function.

    The sign of the CTF is chosen so that the first lobe is positive under the
    standard (underfocus) imaging condition. Applying the CTF to an image therefore
    preserves its contrast at low frequencies (bright density stays bright).

    Attributes
    ----------
    spherical_aberration : float or callable
        Spherical aberration in mm.
    defocus : float or callable
        Defocus in μm. Negative value means underfocus.
    wave_length : float or callable
        Wave length in angstrom.
    bfactor : float
        B-factor in nm².
    amplitude : float
        Fraction of amplitude contrast, between 0 and 1.
    """

    spherical_aberration: float
    wave_length: float
    defocus: float = -1.0
    bfactor: float = 0.0
    amplitude: float = 0.07

    def __post_init__(self):
        if not 0.0 <= self.amplitude <= 1.0:
            raise ValueError(
                f"Amplitude contrast must be between 0 and 1, got {self.amplitude}."
            )

    @classmethod
    def from_kv(
        cls,
        kv: float,
        spherical_aberration: float,
        defocus: float = -1.0,
        bfactor: float = 0.0,
        amplitude: float = 0.07,
    ) -> CTFModel:
        wave_length = _voltage_to_wave_length(kv)
        return cls(
            spherical_aberration=spherical_aberration,
            wave_length=wave_length,
            defocus=defocus,
            bfactor=bfactor,
            amplitude=amplitude,
        )

    def simulate_image(
        self,
        shape: tuple[int, int],
        scale: nm,
    ) -> NDArray[np.float32]:
        """Simulate the CTF of the given shape."""
        yfreq = np.fft.fftfreq(shape[0], scale).astype(np.float32)
        xfreq = np.fft.fftfreq(shape[1], scale).astype(np.float32)
        freq = np.sqrt(yfreq[:, np.newaxis] ** 2 + xfreq[np.newaxis, :] ** 2)
        return self.simulate(freq)

    def apply_ctf(
        self,
        img: NDArray[np.floating],
        scale: nm,
    ) -> NDArray[np.floating]:
        """Apply the CTF to the image (convolve)."""
        ctf = self.simulate_image(img.shape[-2:], scale)
        img_ft = fftn(img, axes=(-2, -1))
        return ifftn(_multiply_multi(ctf, img_ft), axes=(-2, -1)).real

    def phase_flip(
        self,
        img: NDArray[np.floating],
        scale: nm,
    ) -> NDArray[np.floating]:
        """Flip the phase of the image according to the CTF model."""
        ctf = self.simulate_image(img.shape[-2:], scale)
        img_ft = fftn(img, axes=(-2, -1))
        return ifftn(_multiply_multi(np.sign(ctf), img_ft), axes=(-2, -1)).real

    def simulate(self, freq):
        """Simulate the CTF at the given spatial frequencies (in 1/nm).

        The output has the same floating point type as ``freq`` (float32 for float32).
        """
        f2 = freq**2
        # Python floats so that the type of the arrays is kept
        cs = float(self.spherical_aberration) * 1e6  # mm -> nm
        underfocus = -float(self.defocus) * 1e3  # μm -> nm
        lmd = float(self.wave_length) / 10  # Å -> nm
        # phase shift caused by the lens aberration. Defocus and spherical aberration
        # terms have opposite signs in underfocus.
        wave_aberration = (
            math.pi * lmd * underfocus * f2 - math.pi / 2 * cs * lmd**3 * f2**2
        )
        # The CTF with amplitude contrast A is sqrt(1 - A²) * sin(χ) + A * cos(χ),
        # which is identical to sin(χ + arcsin(A)).
        amplitude_phase = math.asin(float(self.amplitude))
        envelope = np.exp(-float(self.bfactor) * f2 / 4)
        return np.sin(wave_aberration + amplitude_phase) * envelope

    def deconvolve(
        self,
        img: NDArray[np.float32],
        scale: float,
        snr_falloff: float = 1.0,
        deconv_strength: float = 1.0,
        highpass_nyquist: float = 0.02,
        phaseflipped: bool = False,
    ) -> NDArray[np.float32]:
        """Deconvolve the image using Wiener deconvolution."""
        return wiener_deconv(
            img,
            scale,
            ctf_model=self,
            snr_falloff=snr_falloff,
            deconv_strength=deconv_strength,
            highpass_nyquist=highpass_nyquist,
            phaseflipped=phaseflipped,
        )


@dataclass(frozen=True)
class CTFCorrection:
    """CTF correction of each tilt image, as a filter in preprocessing.

    The CTF is corrected by phase flipping or Wiener filtering of the whole tilt
    image (see :class:`acryo.projection.TiltImagePreprocessor`). The variation of the
    defocus within a tilted image is not considered (see :class:`PatchCTFCorrection`).

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
        freq_y: NDArray[np.float32],
        freq_x: NDArray[np.float32],
        index: int,
    ) -> NDArray[np.float32]:
        if len(self.models) == 1:
            model = self.models[0]
        elif 0 <= index < len(self.models):
            model = self.models[index]
        else:
            raise IndexError(
                f"No CTF model for the tilt {index} ({len(self.models)} models)."
            )
        freq_y = np.asarray(freq_y, dtype=np.float32)
        freq_x = np.asarray(freq_x, dtype=np.float32)
        ctf = model.simulate(np.sqrt(freq_y**2 + freq_x**2))
        if self.method == "phase_flip":
            return np.sign(ctf)
        return ctf / (ctf**2 + float(self.wiener_constant))

    def select_tilts(self, indices: ArrayLike) -> CTFCorrection:
        """Return the CTF correction of the selected tilts."""
        if len(self.models) == 1:
            return self
        models = tuple(self.models[i] for i in np.asarray(indices, dtype=np.intp))
        return dataclasses.replace(self, models=models)


@dataclass(frozen=True, eq=False)
class TiltSeriesCTF:
    """CTF parameters of each tilt of a tilt series.

    The defocus of each tilt is the defocus at the projection of the tomogram center,
    or at the center of the tilt image (see ``defocus_at``). The defocus in tilted
    images changes with the depth of the specimen, which is considered by
    :class:`PatchCTFCorrection` (see
    :meth:`acryo.projection.TiltSeries.ctf_correction`).

    Parameters
    ----------
    defocus_u : (T,) array
        Defocus of each tilt in μm along the axis of ``astigmatism_angle``. Negative
        value means underfocus, as :class:`CTFModel`.
    defocus_v : (T,) array, optional
        Defocus of each tilt in μm perpendicular to the axis. Same as ``defocus_u``
        by default (no astigmatism).
    astigmatism_angle : float or (T,) array, default is 0.0
        Angle of the axis of ``defocus_u`` in degrees in the Fourier space of the tilt
        images, counterclockwise from the x axis (from x toward y).
    voltage : float, default is 300.0
        Acceleration voltage in kV.
    spherical_aberration : float, default is 2.7
        Spherical aberration in mm.
    amplitude_contrast : float, default is 0.07
        Fraction of amplitude contrast.
    defocus_at : {"tomogram_center", "image_center"}, default is "tomogram_center"
        Where the defocus of each tilt is. "tomogram_center" is the projection of the
        tomogram center, as RELION. "image_center" is the center of the tilt image,
        where ctfplotter of IMOD measures the defocus on the raw tilt images. They
        differ by the depth between them, which is large if the tilt images are
        shifted by the alignment.
    """

    defocus_u: NDArray[np.float64]
    defocus_v: NDArray[np.float64] = None  # type: ignore
    astigmatism_angle: NDArray[np.float64] = 0.0  # type: ignore
    voltage: float = 300.0
    spherical_aberration: float = 2.7
    amplitude_contrast: float = 0.07
    defocus_at: Literal["tomogram_center", "image_center"] = "tomogram_center"

    def __post_init__(self):
        if self.defocus_at not in ("tomogram_center", "image_center"):
            raise ValueError(
                "'defocus_at' must be 'tomogram_center' or 'image_center', got "
                f"{self.defocus_at!r}."
            )
        defocus_u = np.asarray(self.defocus_u, dtype=np.float64)
        if defocus_u.ndim != 1 or defocus_u.size == 0:
            raise ValueError(
                f"'defocus_u' must be a non-empty 1D array, got {defocus_u.shape}."
            )
        ntilts = defocus_u.size
        defocus_v = defocus_u if self.defocus_v is None else self.defocus_v
        _set = object.__setattr__
        _set(self, "defocus_u", _readonly(defocus_u))
        for name, value in [
            ("defocus_v", defocus_v),
            ("astigmatism_angle", self.astigmatism_angle),
        ]:
            arr = np.broadcast_to(np.asarray(value, dtype=np.float64), (ntilts,))
            _set(self, name, _readonly(arr))
        if self.voltage <= 0:
            raise ValueError(f"'voltage' must be positive, got {self.voltage}.")
        if self.spherical_aberration < 0:
            raise ValueError(
                "'spherical_aberration' must be non-negative, got "
                f"{self.spherical_aberration}."
            )
        if not 0.0 <= self.amplitude_contrast <= 1.0:
            raise ValueError(
                "'amplitude_contrast' must be between 0 and 1, got "
                f"{self.amplitude_contrast}."
            )
        _set(self, "voltage", float(self.voltage))
        _set(self, "spherical_aberration", float(self.spherical_aberration))
        _set(self, "amplitude_contrast", float(self.amplitude_contrast))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(ntilts={self.ntilts}, voltage={self.voltage}, "
            f"spherical_aberration={self.spherical_aberration}, "
            f"amplitude_contrast={self.amplitude_contrast})"
        )

    @property
    def ntilts(self) -> int:
        """Number of tilts."""
        return self.defocus_u.size

    @property
    def wave_length(self) -> nm:
        """Wave length of the electrons in nm."""
        return float(_voltage_to_wave_length(self.voltage)) / 10

    def select_tilts(self, indices: ArrayLike) -> Self:
        """Return the CTF parameters of the selected tilts."""
        idx = np.asarray(indices, dtype=np.intp)
        return type(self)(
            defocus_u=self.defocus_u[idx],
            defocus_v=self.defocus_v[idx],
            astigmatism_angle=self.astigmatism_angle[idx],
            voltage=self.voltage,
            spherical_aberration=self.spherical_aberration,
            amplitude_contrast=self.amplitude_contrast,
            defocus_at=self.defocus_at,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to a JSON-serializable dict."""
        return {
            "defocus_u": self.defocus_u.tolist(),
            "defocus_v": self.defocus_v.tolist(),
            "astigmatism_angle": self.astigmatism_angle.tolist(),
            "voltage": self.voltage,
            "spherical_aberration": self.spherical_aberration,
            "amplitude_contrast": self.amplitude_contrast,
            "defocus_at": self.defocus_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TiltSeriesCTF:
        """Create from a dict made by :meth:`to_dict`."""
        return cls(**data)


@dataclass(frozen=True, eq=False)
class PatchCTFCorrection:
    """CTF correction by phase flipping with the defocus at each position.

    This is a position-dependent filter of
    :class:`acryo.projection.TiltImagePreprocessor`, so each patch of a tilt image is
    phase-flipped by the CTF at the center of the patch. The defocus at a position on
    the detector is that of the plane z = 0 of the tomogram projected there, that is,
    the defocus of :class:`TiltSeriesCTF` changed by the depth of the plane relative
    to the tomogram center. Use :meth:`acryo.projection.TiltSeries.ctf_correction` to
    create it from a tilt series.

    Parameters
    ----------
    ctf : TiltSeriesCTF
        CTF parameters of each tilt.
    depth_gradient : (T, 2) array
        Change of the depth (nm) of the plane z = 0 of the tomogram per nm along
        (y, x) on the detector. The depth is the first coordinate of the projection
        by the tilt series.
    center : (T, 2) array
        (y, x) coordinates of the projection of the tomogram center in nm from the
        center of the detector.
    hand : {1, -1}, default is -1
        Handedness of the defocus gradient, same as ``rlnTomoHand`` of RELION. The
        underfocus at a point is ``hand`` times its depth larger than that of the
        tomogram center. -1 is the standard geometry, where the right side (+x) of the
        tilt images aligned to the vertical tilt axis is more underfocused at positive
        tilt angles. This is the convention of IMOD (``InvertTiltAngles`` inverts it).
    """

    ctf: TiltSeriesCTF
    depth_gradient: NDArray[np.float64]
    center: NDArray[np.float64]
    hand: int = -1

    position_dependent: ClassVar[bool] = True

    def __post_init__(self):
        for name in ["depth_gradient", "center"]:
            arr = np.asarray(getattr(self, name), dtype=np.float64)
            if arr.shape != (self.ctf.ntilts, 2):
                raise ValueError(
                    f"Shape of '{name}' must be ({self.ctf.ntilts}, 2), got "
                    f"{arr.shape}."
                )
            object.__setattr__(self, name, _readonly(arr))
        if self.hand not in (1, -1):
            raise ValueError(f"'hand' must be 1 or -1, got {self.hand!r}.")
        object.__setattr__(self, "hand", int(self.hand))

    def defocus(
        self, index: int, position: tuple[nm, nm] = (0.0, 0.0)
    ) -> tuple[float, float]:
        """Defocus (u, v) in μm of the tilt at the (y, x) position (nm) on the detector."""
        depth = float(
            self.depth_gradient[index] @ (np.asarray(position) - self.center[index])
        )
        # depth and underfocus are in nm, defocus in μm (negative for underfocus)
        shift = -self.hand * depth / 1000
        return (
            float(self.ctf.defocus_u[index]) + shift,
            float(self.ctf.defocus_v[index]) + shift,
        )

    def __call__(
        self,
        freq_y: NDArray[np.float32],
        freq_x: NDArray[np.float32],
        index: int,
        position: tuple[nm, nm] = (0.0, 0.0),
    ) -> NDArray[np.float32]:
        return self._for_tilt(freq_y, freq_x, index)(position)

    def _for_tilt(
        self, freq_y: NDArray[np.float32], freq_x: NDArray[np.float32], index: int
    ) -> Callable[[tuple[nm, nm]], NDArray[np.float32]]:
        """The filter of a tilt as a function of the position.

        The terms of the CTF that do not depend on the position are computed only
        once for all the patches of a tilt image (see TiltImagePreprocessor).
        """
        if not 0 <= index < self.ctf.ntilts:
            raise IndexError(f"No CTF for the tilt {index} ({self.ctf.ntilts} tilts).")
        ctf = self.ctf
        # the position changes the defocus u and v by the same amount
        phase = _CTFPhase(
            np.asarray(freq_y, dtype=np.float32),
            np.asarray(freq_x, dtype=np.float32),
            astigmatism=(ctf.defocus_u[index] - ctf.defocus_v[index]) / 2,
            astigmatism_angle=math.radians(ctf.astigmatism_angle[index]),
            wave_length=ctf.wave_length,
            spherical_aberration=ctf.spherical_aberration,
            amplitude_contrast=ctf.amplitude_contrast,
        )

        def _filter(position: tuple[nm, nm]) -> NDArray[np.float32]:
            defocus_u, defocus_v = self.defocus(index, position)
            return phase.sign((defocus_u + defocus_v) / 2)

        return _filter

    def select_tilts(self, indices: ArrayLike) -> PatchCTFCorrection:
        """Return the CTF correction of the selected tilts."""
        idx = np.asarray(indices, dtype=np.intp)
        return PatchCTFCorrection(
            self.ctf.select_tilts(idx),
            self.depth_gradient[idx],
            self.center[idx],
            hand=self.hand,
        )


class _CTFPhase:
    """Phase of the astigmatic CTF, in the same convention as :class:`CTFModel`.

    The phase is ``pi * lmd * underfocus * f^2 - pi / 2 * cs * lmd^3 * f^4`` plus
    ``arcsin(A)``. The underfocus along the direction of the frequency at the angle
    ``a`` is ``mean + astigmatism * cos(2 * (a - astigmatism_angle))``, where
    ``f^2 * cos(2a) = fx^2 - fy^2`` and ``f^2 * sin(2a) = 2 * fx * fy``. Only the term
    of the mean defocus is computed for each defocus.
    """

    def __init__(
        self,
        freq_y: NDArray[np.float32],
        freq_x: NDArray[np.float32],
        *,
        astigmatism: float,
        astigmatism_angle: float,
        wave_length: nm,
        spherical_aberration: float,
        amplitude_contrast: float,
    ):
        self._lmd = wave_length
        cs = spherical_aberration * 1e6  # mm -> nm
        fy2 = freq_y * freq_y
        fx2 = freq_x * freq_x
        f2 = fy2 + fx2
        base = f2 * f2
        base *= np.float32(-math.pi / 2 * cs * wave_length**3)
        base += np.float32(math.asin(amplitude_contrast))
        if astigmatism != 0:
            angle = 2 * astigmatism_angle
            # astigmatism in μm (negative for underfocus) -> underfocus in nm
            coef = -math.pi * wave_length * astigmatism * 1e3
            base += (fx2 - fy2) * np.float32(coef * math.cos(angle))
            base += (freq_x * np.float32(2 * coef * math.sin(angle))) * freq_y
        self._f2 = f2
        self._base = base

    def value(self, defocus: float | NDArray[np.floating]) -> NDArray[np.float32]:
        """CTF of the mean defocus in μm (negative for underfocus).

        If ``defocus`` is an array, the CTFs of each defocus are stacked along the
        leading axes.
        """
        coef = np.asarray(defocus, dtype=np.float32)
        coef = coef * np.float32(-math.pi * self._lmd * 1e3)
        chi = coef.reshape(coef.shape + (1,) * self._f2.ndim) * self._f2
        chi += self._base
        return np.sin(chi, out=chi)

    def sign(self, defocus: float) -> NDArray[np.float32]:
        """Sign of the CTF of the mean defocus in μm (negative for underfocus)."""
        chi = self.value(defocus)
        return np.sign(chi, out=chi)


@dataclass(frozen=True)
class _ImodDefocus:
    """Defocus file of IMOD (output of ctfplotter).

    Each entry is fit to a range of views (numbered from 1). Defocus values are in nm
    (underfocus positive), and astigmatism angles are in degrees.
    """

    start_view: NDArray[np.float64]
    end_view: NDArray[np.float64]
    defocus1: NDArray[np.float64]
    defocus2: NDArray[np.float64]
    astigmatism_angle: NDArray[np.float64]
    phase_shift: NDArray[np.float64]

    @classmethod
    def read(cls, path: str | Path) -> _ImodDefocus:
        """Read the defocus file of version 1, 2 or 3."""
        path = Path(path)
        rows = [
            [float(v) for v in line.split()]
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        if not rows:
            raise ValueError(f"Defocus file {path} is empty.")
        flags, version = 0, 1
        if len(rows[0]) == 6:
            # the version number at the end of the first line
            version = int(rows[0][5])
            if version >= 3:
                # header line of "flags 0 0. 0. 0 3"
                flags = int(rows[0][0])
                rows = rows[1:]
            else:
                rows[0] = rows[0][:5]
        astig, phase = bool(flags & 1), bool(flags & 4)
        ncols = 5 + 2 * astig + phase
        if not rows or any(len(row) < ncols for row in rows):
            raise ValueError(
                f"Each line of the defocus file {path} must have at least {ncols} "
                "values."
            )
        data = np.array([row[:ncols] for row in rows], dtype=np.float64)
        start, end, defocus1 = data[:, 0], data[:, 1], data[:, 4]
        if version < 2 and start.min() == 0:
            # ctfplotter before IMOD 4.0.29 numbered the views from 0
            start, end = start + 1, end + 1
        if astig:
            defocus2 = data[:, 5]
            angle = data[:, 6] if not flags & 2 else np.rad2deg(data[:, 6])
        else:
            defocus2, angle = defocus1, np.zeros_like(defocus1)
        if phase:
            shift = data[:, ncols - 1]
            shift = shift if not flags & 8 else np.rad2deg(shift)
        else:
            shift = np.zeros_like(defocus1)
        return cls(start, end, defocus1, defocus2, angle, shift)

    def at_views(
        self, views: ArrayLike
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Defocus values and astigmatism angles of the views (numbered from 1).

        As ctfphaseflip of IMOD, each defocus value is at the middle of the range of
        views, and the defocus of a view is linearly interpolated between them, or the
        nearest one outside. The astigmatism angle is that of the nearest entry.
        """
        views = np.asarray(views, dtype=np.float64)
        middle = (self.start_view + self.end_view) / 2
        order = np.argsort(middle, kind="stable")
        middle = middle[order]
        defocus1 = np.interp(views, middle, self.defocus1[order])
        defocus2 = np.interp(views, middle, self.defocus2[order])
        nearest = np.abs(views[:, np.newaxis] - middle[np.newaxis, :]).argmin(axis=1)
        angle = self.astigmatism_angle[order][nearest]
        return defocus1, defocus2, angle


def _warn_ignored(message: str) -> None:
    """Warn that a parameter of the CTF in a file is ignored."""
    # stacklevel of the caller of TiltSeries.from_*
    warnings.warn(message, UserWarning, stacklevel=4)


def _voltage_to_wave_length(kv: float) -> float:
    """Convert kV to wave length in angstrom."""
    # energy of an electron is sum of relativistic momentum and kinetic energy
    return np.sqrt(0.1504 / (kv + 0.978e-3 * kv**2))


def _multiply_multi(
    a: NDArray[np.floating], b: NDArray[np.floating]
) -> NDArray[np.floating]:
    """Multiply two arrays considering their dimensions."""
    if b.ndim == a.ndim:
        return a * b
    newaxis = (np.newaxis,) * (b.ndim - a.ndim)
    return a[newaxis] * b


def _readonly(arr: ArrayLike) -> NDArray:
    arr = np.array(arr)  # copy
    arr.setflags(write=False)
    return arr
