from __future__ import annotations

from ._base import LoaderBase, Unset
from typing import TYPE_CHECKING, Iterable, Sequence
import numpy as np
from numpy.typing import NDArray
from dask import array as da, delayed
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation

from acryo import _utils
from acryo._types import nm, pixel
from acryo._fft import fftn, ifftn
from acryo.molecules import Molecules
from acryo.pipe._classes import ImageProvider

if TYPE_CHECKING:
    from typing_extensions import Self
    from dask.delayed import Delayed
    from ._base import _ShapeType


class MockLoader(LoaderBase):
    """
    A subtomogram loader from a virtual tomogram.

    This loader is for testing purpose only. The tomogram does not actually exist
    but subtomograms are generated on the fly based on the template image. The
    true center of the molecules is always at (0, 0, 0) and the true rotation is
    always the identity rotation. Thus, given molecule positions/rotations will
    directly be the values to be tested.

    Parameters
    ----------
    template : 3D array, ImageProvider
        The template image.
    molecules : Molecules
        The molecules to be loaded. Note that the true center/rotation is always
        (0, 0, 0) and the identity rotation, respectively.
    noise : float, optional
        Standard deviation of the Gaussian noise to be added to the subtomograms
        projection slices.
    degrees : array-like, optional
        The rotation angles in degrees. If not provided, projection/back-projection
        will not be simulated.
    central_axis : tuple, default is (0.0, 1.0, 0.0)
        The central axis of the rotation during tomogram acquisition.
    """

    def __init__(
        self,
        template: NDArray[np.float32] | ImageProvider,
        molecules: Molecules,
        noise: float = 0.0,
        degrees: Sequence[float] | None = None,
        central_axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
        order: int = 3,
        scale: nm = 1.0,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset = Unset(),
        corner_safe: bool = False,
    ) -> None:
        if noise < 0:
            raise ValueError("Noise must be non-negative.")
        super().__init__(order, scale, output_shape, corner_safe)
        self._template = template
        self._noise = noise
        if degrees is None:
            if noise > 0:
                # TODO: implement this
                raise NotImplementedError(
                    "Noise simulation without degrees is not implemented yet."
                )
            self._degrees = None
        else:
            self._degrees = np.asarray(degrees, dtype=np.float32)
        self._molecules = molecules
        self._central_axis = np.array(central_axis, dtype=np.float32)

    @property
    def molecules(self) -> Molecules:
        """All the molecules"""
        return self._molecules

    def construct_loading_tasks(
        self, output_shape: _ShapeType = None
    ) -> list[da.Array]:
        tasks: list[Delayed] = []
        if isinstance(self._template, ImageProvider):
            template = self._template(self.scale)
        else:
            template = self._template
        if output_shape is None:
            output_shape = template.shape
        elif output_shape != template.shape:
            raise ValueError("Mismatched output shape and template shape.")

        # spline prefilter in advance
        template = ndi.spline_filter(
            template, order=self.order, mode="constant", output=np.float32
        )
        center = np.array(template.shape) / 2 - 0.5
        matrices = self.molecules.affine_matrix(
            center, center + self.molecules.pos / self.scale
        )
        for i, mtx in enumerate(matrices):
            img = _utils.delayed_affine(
                template, mtx, order=self.order, prefilter=False
            )
            if self._degrees is None:
                tasks.append(img)
            else:
                task = simulate_noise(
                    img,
                    self._central_axis,
                    self._degrees,
                    self._noise,
                    seed=i,
                )
                tasks.append(task)
        return tasks

    def replace(
        self,
        molecules: Molecules | None = None,
        output_shape: pixel | tuple[pixel, pixel, pixel] | Unset | None = None,
        order: int | None = None,
        scale: float | None = None,
        corner_safe: bool | None = None,
    ) -> Self:
        if molecules is None:
            molecules = self.molecules
        if output_shape is None:
            output_shape = self.output_shape
        if order is None:
            order = self.order
        if scale is None:
            scale = self.scale
        if corner_safe is None:
            corner_safe = self.corner_safe
        return self.__class__(
            self._template,
            molecules=molecules,
            noise=self._noise,
            degrees=self._degrees,
            output_shape=output_shape,
            order=order,
            scale=scale,
            corner_safe=corner_safe,
        )


def simulate_noise(
    img: NDArray[np.float32],
    central_axis,
    degrees: NDArray[np.float32],
    noise: float,
    seed: int,
):
    # img: spline filtered
    matrices, output_shape = normalize_radon_input(img, central_axis, degrees)
    sino: da.Array = da.stack(
        [
            da.from_delayed(
                radon_single(img, mtx, order=3, output_shape=output_shape),
                shape=output_shape[1:],
                dtype=np.float32,
            )
            for mtx in matrices
        ],
        axis=0,
    )
    rng = np.random.default_rng(seed=seed)
    sino += rng.normal(0, noise, sino.shape).astype(np.float32)
    out = da.stack(
        [
            da.from_delayed(
                iradon(sino[:, i].T, degrees, img.shape[:2]),
                shape=img.shape[:2],
                dtype=np.float32,
            )
            for i in range(img.shape[2])
        ],
        axis=1,
    )
    return out


# Radon transform


def normalize_radon_input(
    self: NDArray[np.float32],
    central_axis: NDArray[np.float32],
    degrees: NDArray[np.float32] | float,
):
    radians = np.deg2rad(list(degrees))

    # normalize central axis to a 3D vector
    central_axis = np.asarray(central_axis)
    central_axis /= np.sqrt(np.sum(central_axis**2))  # normalize
    if central_axis.shape != (3,):
        raise ValueError(f"Central axis must be a 3D vector")

    # construct Affine transform matrices
    height = int(np.ceil(np.linalg.norm(self.shape)))
    output_shape = (height, self.shape[1], self.shape[2])
    params = _get_rotation_matrices_for_radon_3d(
        radians, central_axis, self.shape, output_shape
    )

    return params, output_shape


@delayed
def radon_single(img: np.ndarray, mtx: np.ndarray, order: int = 3, output_shape=None):
    """Radon transform of 2D image."""
    img_rot = ndi.affine_transform(
        img, mtx, order=order, output_shape=output_shape, prefilter=False
    )
    return np.sum(img_rot, axis=0)


def _get_rotation_matrices_for_radon_3d(
    radians: NDArray[np.float32],
    central_axis: np.ndarray,
    in_shape: tuple[int, int, int],
    out_shape: tuple[int, int, int],
) -> Iterable[np.ndarray]:
    vec = np.stack([central_axis * rad for rad in radians], axis=0)
    in_center = (np.array(in_shape) - 1.0) / 2.0
    out_center = (np.array(out_shape) - 1.0) / 2.0
    return _utils.compose_matrices(in_center, Rotation.from_rotvec(vec), out_center)


# This function is mostly ported from `skimage.transform`.
# The most important difference is that this implementation support arbitrary
# output shape.
@delayed
def iradon(
    img: np.ndarray,
    degrees: np.ndarray,
    output_shape: tuple[int, int],
    interpolation: str = "cubic",
):
    angles_count = len(degrees)
    dtype = img.dtype
    img_shape = img.shape[0]

    # Apply filter in Fourier domain
    projection = fftn(img, axes=(0,)) * get_hamming_filter(img.shape[0])
    radon_filtered = np.real(ifftn(projection, axes=(0,))[:img_shape, :])

    # Reconstruct image by interpolation
    reconstructed = np.zeros(output_shape, dtype=dtype)
    npr, ypr = np.indices(output_shape)
    npr -= output_shape[0] // 2
    ypr -= output_shape[1] // 2

    from scipy.interpolate import interp1d

    x = np.arange(img_shape) - img_shape // 2
    for col, angle in zip(radon_filtered.T, np.deg2rad(degrees)):
        t = ypr * np.cos(angle) - npr * np.sin(angle)
        interpolant = interp1d(
            x, col, kind=interpolation, bounds_error=False, fill_value=0
        )
        reconstructed += interpolant(t)

    return reconstructed * np.pi / (2 * angles_count)


# This function is almost ported from `skimage.transform`.
def get_hamming_filter(size: int):
    n = np.concatenate(
        [
            np.arange(1, size / 2 + 1, 2, dtype=int),
            np.arange(size / 2 - 1, 0, -2, dtype=int),
        ]
    )
    f = np.zeros(size)
    f[0] = 0.25
    f[1::2] = -1 / (np.pi * n[: len(f[1::2])]) ** 2
    fourier_filter = 2 * np.real(np.fft.fft(f))  # ramp filter
    fourier_filter *= np.fft.fftshift(np.hamming(size))
    return fourier_filter[:, np.newaxis]
