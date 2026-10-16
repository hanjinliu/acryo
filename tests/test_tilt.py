import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy.spatial.transform import Rotation
import pytest

from acryo import SubtomogramLoader, Molecules, _utils
from acryo.backend import Backend
from acryo.testing import TomogramGenerator
from acryo.tilt import single_axis, dual_axis
from acryo.tilt import _utils as _tilt_utils

SHAPES = [
    (20, 20, 20),
    (21, 21, 21),
    (32, 48, 64),
    (51, 31, 21),
    (24, 25, 26),
]
ROTATIONS = [Rotation.identity()] + list(Rotation.random(3, random_state=12))
TILT_RANGES = [(-60.0, 60.0), (-70.0, 50.0), (-90.0, 90.0)]


def _ground_truth(
    rotator: Rotation, tilt_range: tuple[float, float], shape, axis: str = "y"
) -> tuple[np.ndarray, np.ndarray]:
    """The measured region of the Fourier space, computed from the tilt angles.

    A subvolume ``f(r) = g(c + R r)`` has the spectrum ``F(k) = G(R k)``. The image of
    tilt angle ``t`` is the central slice that contains the tilt axis and the vector
    ``sin(t) * e_z + cos(t) * e_x`` (y-axis tilt) or ``sin(t) * e_z + cos(t) * e_y``
    (x-axis tilt).

    Returns the measured region and the region too close to the wedge boundaries to
    be compared under floating point errors.
    """
    freqs = [np.fft.fftfreq(s) for s in shape]  # cycles/pixel
    q = np.stack(np.meshgrid(*freqs, indexing="ij"), axis=-1).reshape(-1, 3)
    qz, qy, qx = rotator.apply(q).T
    if axis == "y":
        u, v = qz, qx
    else:
        u, v = qz, qy
    # angle in the plane perpendicular to the tilt axis, folded into [-90, 90)
    ang = np.rad2deg(np.arctan2(u, v))
    ang = (ang + 90) % 180 - 90
    proj = np.hypot(u, v)
    on_axis = proj == 0  # parallel to the tilt axis (in all the slices)
    tmin, tmax = tilt_range
    measured = on_axis | ((tmin <= ang) & (ang <= tmax)) | (tmax - tmin >= 180)
    ambiguous = ~on_axis & (
        (proj < 1e-6 * np.linalg.norm(q, axis=-1))
        | (np.abs((ang - tmin + 90) % 180 - 90) < 1e-4)
        | (np.abs((ang - tmax + 90) % 180 - 90) < 1e-4)
    )
    if tmax - tmin >= 180:
        ambiguous[:] = False
    return measured.reshape(shape), ambiguous.reshape(shape)


def _assert_mask_equal(mask, expected: np.ndarray, ambiguous: np.ndarray):
    mask = np.asarray(mask)
    assert mask.shape == expected.shape
    assert ambiguous.mean() < 1e-3
    assert_array_equal(mask[~ambiguous], expected[~ambiguous])


@pytest.mark.parametrize("s", [6, 7, 8, 9, 49])
def test_fft_indices(s: int):
    shape = (s, s + 1, s + 2)
    idx_tilt = _tilt_utils.get_indices(shape)
    idx_utils = _utils._get_indices(shape)
    for i, size in enumerate(shape):
        expected = np.fft.fftfreq(size, 1 / size)
        sl = [0, 0, 0, i]
        sl[i] = slice(None)  # type: ignore
        assert_allclose(idx_tilt[tuple(sl)], expected)
        assert_allclose(idx_utils[tuple(sl)], expected)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("rot_idx", range(len(ROTATIONS)))
@pytest.mark.parametrize("tilt_range", TILT_RANGES)
@pytest.mark.parametrize("axis", ["y", "x"])
def test_single_axis_mask_ground_truth(shape, rot_idx, tilt_range, axis):
    rot = ROTATIONS[rot_idx]
    mask = single_axis(tilt_range, axis=axis).create_mask(rot, shape)
    _assert_mask_equal(mask, *_ground_truth(rot, tilt_range, shape, axis=axis))


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("rot_idx", range(len(ROTATIONS)))
def test_dual_axis_mask_ground_truth(shape, rot_idx):
    rot = ROTATIONS[rot_idx]
    mask = dual_axis((-60, 50), (-55, 65)).create_mask(rot, shape)
    expected_y, ambiguous_y = _ground_truth(rot, (-60, 50), shape, axis="y")
    expected_x, ambiguous_x = _ground_truth(rot, (-55, 65), shape, axis="x")
    _assert_mask_equal(mask, expected_y | expected_x, ambiguous_y | ambiguous_x)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("rot_idx", range(len(ROTATIONS)))
def test_missing_wedge_mask_functions(shape, rot_idx):
    rot = ROTATIONS[rot_idx]
    tilt_range = (-70.0, 50.0)
    expected, ambiguous = _ground_truth(rot, tilt_range, shape, axis="y")
    mask_utils = _utils.missing_wedge_mask(rot, tilt_range, shape)
    mask_backend = Backend().missing_wedge_mask(rot, tilt_range, shape)
    _assert_mask_equal(mask_utils, expected, ambiguous)
    _assert_mask_equal(mask_backend, expected, ambiguous)


@pytest.mark.parametrize("shape", [(32, 32, 32), (31, 31, 31), (24, 32, 40)])
@pytest.mark.parametrize("rot_idx", range(len(ROTATIONS)))
def test_mask_matches_loaded_subvolume(shape, rot_idx):
    """Missing wedge of a subvolume loaded from a tomogram must match the mask."""
    rot = ROTATIONS[rot_idx]
    tilt = single_axis((-70, 50), axis="y")
    rng = np.random.default_rng(0)
    tomo_shape = (80, 80, 80)
    noise = rng.standard_normal(tomo_shape).astype(np.float32)
    wedge = tilt.create_mask(Rotation.identity(), tomo_shape)
    tomo = np.fft.ifftn(np.fft.fftn(noise) * wedge).real.astype(np.float32)

    center = np.array(tomo_shape) / 2 - 0.5
    mole = Molecules(center[np.newaxis], Rotation.from_quat([rot.as_quat()]))
    loader = SubtomogramLoader(tomo, mole, order=3, output_shape=shape)
    subvol = np.asarray(loader.construct_dask())[0]

    window = np.ones(shape)
    for i, s in enumerate(shape):
        _shape = [1, 1, 1]
        _shape[i] = s
        window = window * np.hanning(s + 2)[1:-1].reshape(_shape)
    power = np.abs(np.fft.fftn((subvol - subvol.mean()) * window)) ** 2

    # Compare the regions apart from the wedge boundaries (spectral leakage of the
    # window) and the high frequencies (aliasing of the rotated corners).
    freqs = np.meshgrid(*[np.fft.fftfreq(s) for s in shape], indexing="ij")
    radius = np.sqrt(sum(f**2 for f in freqs))
    inside = (0 < radius) & (radius < 0.4)
    missing = ~single_axis((-80, 60)).create_mask(rot, shape) & inside
    measured = single_axis((-60, 40)).create_mask(rot, shape) & inside
    # this ratio is > 0.13 if the mask is rotated in the opposite direction.
    leak = power[missing].mean() / power[measured].mean()
    assert leak < 0.1


def _iter_blocks(tomo: np.ndarray, shape, grid_shape):
    _, sy, sx = shape
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            yield tomo[:, i * sy : (i + 1) * sy, j * sx : (j + 1) * sx]


@pytest.mark.parametrize("shape", [(20, 20, 20), (21, 24, 27)])
@pytest.mark.parametrize("tilt_range", [(-60.0, 60.0), (-70.0, 50.0)])
def test_tomogram_generator_missing_wedge(shape, tilt_range):
    rng = np.random.default_rng(0)
    template = rng.random(shape).astype(np.float32)
    grid_shape = (2, 2)
    measured, ambiguous = _ground_truth(Rotation.identity(), tilt_range, shape)

    # without noise, each block is exactly the template filtered by the wedge
    gen = TomogramGenerator(template, grid_shape=grid_shape, noise_sigma=0.0)
    tomo = gen.get_tomogram(tilt_range=tilt_range)
    expected = np.fft.ifftn(np.fft.fftn(template) * measured).real
    for block in _iter_blocks(tomo, shape, grid_shape):
        assert_allclose(block, expected, atol=1e-5)

    # noise is also filtered
    gen = TomogramGenerator(template, grid_shape=grid_shape, noise_sigma=1.0)
    tomo = gen.get_tomogram(tilt_range=tilt_range)
    # Nyquist frequency of an even axis stands for both +0.5 and -0.5, so it is mixed
    # by taking the real part.
    nyquist = np.zeros(shape, dtype=bool)
    for axis, s in enumerate(shape):
        if s % 2 == 0:
            nyquist |= np.indices(shape)[axis] == s // 2
    missing = ~measured & ~ambiguous & ~nyquist
    for block in _iter_blocks(tomo, shape, grid_shape):
        power = np.abs(np.fft.fftn(block)) ** 2
        assert power[missing].max() < 1e-8 * power[measured].mean()


def test_tomogram_generator_no_wedge():
    rng = np.random.default_rng(0)
    template = rng.random((20, 20, 20)).astype(np.float32)
    gen = TomogramGenerator(template, grid_shape=(2, 2), noise_sigma=0.0)
    tomo = gen.get_tomogram(tilt_range=(-90, 90))
    for block in _iter_blocks(tomo, template.shape, (2, 2)):
        assert_allclose(block, template)
