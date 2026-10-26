import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import fft as sfft

from acryo.projection import (
    CentralSliceInsertion,
    DirectFourierReconstructor,
    LocalSliceInsertion,
    VolumeProjector,
    centered_patches_rfft,
    read_patch,
)

TILT_ANGLES = np.deg2rad(np.arange(-90.0, 90.0, 2.0))
IMAGE_SHAPE = (96, 104)


def _tilt_matrix(angle: float) -> np.ndarray:
    """Rotation around the y axis, from the volume to each tilt image (zyx)."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, -s], [0.0, 1.0, 0.0], [s, 0.0, c]])


PROJECTIONS = np.stack([_tilt_matrix(a) for a in TILT_ANGLES])
ROTATIONS = np.linalg.inv(PROJECTIONS)  # from the tilt images to the volume


def _gaussian_tilt_images(
    center, sigma: float, voxel_size: float, mags=None
) -> np.ndarray:
    """Projections of a 3D Gaussian of amplitude 1 at center (pixels of the images).

    The projection is the sum over the voxels of the given size. If given, each image
    is transformed by the (2, 2) matrix of ``mags`` around the center of the image.
    """
    yy, xx = np.indices(IMAGE_SHAPE, dtype=np.float64)
    yy -= IMAGE_SHAPE[0] // 2
    xx -= IMAGE_SHAPE[1] // 2
    if mags is None:
        mags = np.tile(np.eye(2), (len(PROJECTIONS), 1, 1))
    images = []
    for mat, mag in zip(PROJECTIONS, mags):
        _, py, px = mat @ center
        # the pixels in the coordinates of the untransformed image
        (a, b), (c, d) = np.linalg.inv(mag)
        r2 = (a * yy + b * xx - py) ** 2 + (c * yy + d * xx - px) ** 2
        amp = np.sqrt(2 * np.pi) * sigma / voxel_size
        images.append(amp * np.exp(-r2 / (2 * sigma**2)))
    return np.stack(images).astype(np.float32)


def _projected_positions(point: np.ndarray) -> np.ndarray:
    return (
        np.stack([(mat @ point)[1:] for mat in PROJECTIONS])
        + np.array(IMAGE_SHAPE) // 2
    )


@pytest.mark.parametrize("output_spacing", [1.0, 1.6, 2.0])
@pytest.mark.parametrize("sidelength", [24, 25])
def test_reconstruct_gaussian(output_spacing, sidelength):
    # NOTE: the error of interpolation is larger if the Gaussian is larger relative to
    # the cube.
    sigma = 2.5
    center = np.array([2.3, -1.7, 3.1])
    images = _gaussian_tilt_images(center, sigma, output_spacing)
    reconstructor = DirectFourierReconstructor(
        images,
        ROTATIONS,
        input_spacing=1.0,
        output_spacing=output_spacing,
        sidelength=sidelength,
    )
    point = np.array([1.0, 0.5, -0.5]) * output_spacing
    cube = reconstructor(_projected_positions(point))
    assert cube.shape == (sidelength,) * 3
    assert cube.dtype == np.float32
    # the point is at the center of the cube
    coords = np.moveaxis(np.indices(cube.shape), 0, -1) - sidelength // 2
    coords = coords * output_spacing + point
    expected = np.exp(-np.sum((coords - center) ** 2, axis=-1) / (2 * sigma**2))
    assert_allclose(cube, expected, atol=0.04)


@pytest.mark.parametrize("output_spacing", [1.0, 1.6])
def test_reconstruct_magnified_gaussian(output_spacing):
    # each tilt image is magnified (and sheared) around its center
    rng = np.random.default_rng(0)
    mags = 1.04 * np.eye(2) + rng.uniform(-0.04, 0.04, size=(len(PROJECTIONS), 2, 2))
    sigma = 2.5
    center = np.array([2.3, -5.7, 6.1])
    point = np.array([1.0, 0.5, -0.5]) * output_spacing
    # reconstructed from the images without magnification
    expected = DirectFourierReconstructor(
        _gaussian_tilt_images(center, sigma, output_spacing),
        ROTATIONS,
        1.0,
        output_spacing,
        24,
    )(_projected_positions(point))

    images = _gaussian_tilt_images(center, sigma, output_spacing, mags)
    inverse = np.tile(np.eye(3), (len(mags), 1, 1))
    inverse[:, 1:, 1:] = np.linalg.inv(mags)
    origin = np.array(IMAGE_SHAPE) // 2
    offsets = _projected_positions(point) - origin
    positions = np.einsum("tij,tj->ti", mags, offsets) + origin
    reconstructor = DirectFourierReconstructor(
        images, ROTATIONS @ inverse, 1.0, output_spacing, 24
    )
    rigid = DirectFourierReconstructor(images, ROTATIONS, 1.0, output_spacing, 24)
    # the patches are larger in the magnified images
    assert reconstructor.patch_size > rigid.patch_size
    error = np.abs(reconstructor(positions) - expected).max()
    assert error < 0.05
    # the error is much larger if the magnification is ignored
    assert np.abs(rigid(positions) - expected).max() > 2 * error


@pytest.mark.parametrize("padding", [2.0, 3.0])
def test_padding(padding):
    images = _gaussian_tilt_images(np.zeros(3), 2.0, 1.0)
    reconstructor = DirectFourierReconstructor(
        images, ROTATIONS, 1.0, 1.0, 10, padding=padding
    )
    assert reconstructor.insertion.volume_shape[0] >= 10 * padding
    cube = reconstructor(_projected_positions(np.zeros(3)))
    assert cube.shape == (10, 10, 10)
    assert np.unravel_index(np.argmax(cube), cube.shape) == (5, 5, 5)


def test_reconstruct_batch():
    images = _gaussian_tilt_images(np.zeros(3), 2.0, 1.0)
    reconstructor = DirectFourierReconstructor(images, ROTATIONS, 1.0, 1.0, 12)
    points = np.array([[0.0, 0.0, 0.0], [1.5, -2.0, 0.7], [-3.0, 0.2, 2.4]])
    positions = np.stack([_projected_positions(p) for p in points])
    cubes = reconstructor(positions)
    assert cubes.shape == (3, 12, 12, 12)
    for pos, cube in zip(positions, cubes):
        assert_allclose(reconstructor(pos), cube, rtol=1e-5, atol=1e-5)
    nested = reconstructor(positions.reshape(1, 3, *positions.shape[1:]))
    assert_allclose(nested[0], cubes, rtol=1e-5, atol=1e-5)


def test_single_precision():
    from acryo.projection._insertion import _gridding_correction

    images = _gaussian_tilt_images(np.zeros(3), 2.0, 1.0)
    reconstructor = DirectFourierReconstructor(images, ROTATIONS, 1.0, 1.0, 12)
    positions = _projected_positions(np.zeros(3))
    slices = centered_patches_rfft(images, positions, reconstructor.patch_size)
    assert slices.dtype == np.complex64
    insertion = reconstructor.insertion
    assert insertion.insert(slices).dtype == np.complex64
    assert insertion.reconstruct(slices, 12).dtype == np.float32
    assert reconstructor(positions).dtype == np.float32
    assert _gridding_correction((24, 24, 24), (12, 12, 12)).dtype == np.float32


def test_insertion_non_cubic():
    sigma = 2.5
    center = np.array([1.2, 0.3, -2.1])
    images = _gaussian_tilt_images(center, sigma, 1.0)
    volume_shape = (24, 32, 40)
    insertion = CentralSliceInsertion(ROTATIONS, (36, 44), volume_shape)
    origin = np.array(IMAGE_SHAPE) // 2
    positions = np.broadcast_to(origin.astype(np.float64), (len(images), 2))
    slices = centered_patches_rfft(images, positions, (36, 44))
    volume = insertion.reconstruct(slices, output_shape=(12, 16, 20))
    assert volume.shape == (12, 16, 20)
    coords = np.moveaxis(np.indices(volume.shape), 0, -1) - np.array([6, 8, 10])
    expected = np.exp(-np.sum((coords - center) ** 2, axis=-1) / (2 * sigma**2))
    assert_allclose(volume, expected, atol=0.05)
    full = insertion.reconstruct(slices)
    assert full.shape == volume_shape


def test_insertion_invalid_input():
    with pytest.raises(ValueError):
        CentralSliceInsertion(np.eye(3), 8, 8)
    with pytest.raises(ValueError):
        CentralSliceInsertion(ROTATIONS, (8, 8, 8), 8)
    insertion = CentralSliceInsertion(ROTATIONS[:3], 8, 8)
    with pytest.raises(ValueError):
        insertion.insert(np.zeros((2, 8, 5), dtype=np.complex64))
    with pytest.raises(ValueError):
        DirectFourierReconstructor(np.zeros((2, 8, 8)), ROTATIONS[:3], 1.0, 1.0, 4)
    with pytest.raises(ValueError):
        DirectFourierReconstructor(np.zeros((3, 8, 8)), ROTATIONS[:3], 1.0, 1.0, 0)
    with pytest.raises(ValueError):
        DirectFourierReconstructor(
            np.zeros((3, 8, 8)), ROTATIONS[:3], 1.0, 1.0, 4, padding=0.5
        )


def _magnified(seed: int, mag: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Random (T, 2, 2) magnifications and the rotations of the magnified images."""
    rng = np.random.default_rng(seed)
    mags = mag * np.eye(2) + rng.uniform(-0.04, 0.04, size=(len(ROTATIONS), 2, 2))
    return mags, _magnified_rotations(mags)


def _magnified_rotations(mags: np.ndarray) -> np.ndarray:
    inverse = np.tile(np.eye(3), (len(mags), 1, 1))
    inverse[:, 1:, 1:] = np.linalg.inv(mags)
    return ROTATIONS @ inverse


@pytest.mark.parametrize("volume_shape", [24, 25, (20, 24, 27)])
def test_local_insertion_same_as_central(volume_shape):
    rng = np.random.default_rng(0)
    _, magnified = _magnified(1)
    shape = (2, len(ROTATIONS), 20, 11)
    slices = (rng.normal(size=shape) + 1j * rng.normal(size=shape)).astype(np.complex64)
    local = LocalSliceInsertion(20, volume_shape, scale=0.9)
    for rotations in [ROTATIONS, magnified]:
        central = CentralSliceInsertion(rotations, 20, volume_shape, scale=0.9)
        expected = central.reconstruct(slices, 12)
        each = np.broadcast_to(rotations, (2,) + rotations.shape)
        out = local.reconstruct(slices, each, 12)
        assert out.shape == (2, 12, 12, 12)
        assert out.dtype == np.float32
        assert_allclose(out, expected, atol=1e-5 * np.abs(expected).max())
    # the slices of each volume are inserted by its own matrices
    out = local.insert(slices, np.stack([ROTATIONS, magnified]))
    for dft, rotations, sl in zip(out, [ROTATIONS, magnified], slices):
        central = CentralSliceInsertion(rotations, 20, volume_shape, scale=0.9)
        assert_allclose(dft, central.insert(sl), atol=1e-5 * np.abs(dft).max())
    with pytest.raises(ValueError):
        local.insert(slices, ROTATIONS)
    with pytest.raises(ValueError):
        local.insert(slices[..., :5], np.stack([ROTATIONS] * 2))


def test_reconstruct_with_rotations_of_each_cube():
    # The reconstructor is created for the magnification of the whole images, but the
    # images are magnified differently around the cube, which is given for each cube.
    mags, magnified = _magnified(2, mag=1.04)
    rng = np.random.default_rng(3)
    local_mags = mags + rng.uniform(-0.02, 0.02, size=mags.shape)
    local = _magnified_rotations(local_mags)
    images = _gaussian_tilt_images(np.array([2.3, -5.7, 6.1]), 2.5, 1.0, local_mags)
    origin = np.array(IMAGE_SHAPE) // 2
    offsets = _projected_positions(np.array([1.0, 0.5, -0.5])) - origin
    positions = np.einsum("tij,tj->ti", local_mags, offsets) + origin
    # reconstructed by the local magnification for all the cubes
    expected = DirectFourierReconstructor(images, local, 1.0, 1.0, 24)(positions)
    reconstructor = DirectFourierReconstructor(images, magnified, 1.0, 1.0, 24)
    # The first cube is inserted separately, and the next ones by the insertion
    # shared by the cubes of almost the same matrices.
    assert reconstructor.nbytes == 0
    cubes = [reconstructor(positions, local) for _ in range(3)]
    assert reconstructor.nbytes > 0
    error = max(np.abs(cube - expected).max() for cube in cubes)
    assert error < 0.01
    assert_allclose(cubes[1], cubes[0], atol=1e-4)
    assert_allclose(cubes[2], cubes[1])
    # the error is much larger if the local magnification is ignored
    assert np.abs(reconstructor(positions) - expected).max() > 3 * error
    # cubes of different matrices in a batch
    batch = reconstructor(np.stack([positions] * 2), np.stack([local, magnified]))
    assert_allclose(batch[0], cubes[0], atol=1e-4)
    assert_allclose(batch[1], reconstructor(positions), atol=1e-4)
    with pytest.raises(ValueError):
        reconstructor(positions, ROTATIONS[:3])


def test_read_patch():
    image = np.arange(30, dtype=np.float32).reshape(5, 6)
    assert_allclose(read_patch(image, (1, 2), (2, 3)), image[1:3, 2:5])
    patch = read_patch(image, (-1, 4), (3, 4))
    expected = np.zeros((3, 4), dtype=np.float32)
    expected[1:, :2] = image[:2, 4:]
    assert_allclose(patch, expected)
    assert_allclose(read_patch(image, (10, 10), (2, 2)), 0)


@pytest.mark.parametrize("shift", [(0.0, 0.0), (0.3, -0.6), (1.5, 2.25)])
def test_centered_patches_rfft(shift):
    yy, xx = np.indices(IMAGE_SHAPE, dtype=np.float64)
    point = np.array([40.0, 51.0]) + np.array(shift)
    image = np.exp(-((yy - point[0]) ** 2 + (xx - point[1]) ** 2) / 8)
    images = image[np.newaxis].astype(np.float32)
    ft = centered_patches_rfft(images, point[np.newaxis], 24)
    assert ft.shape == (1, 24, 13)
    # the point is moved to the origin
    patch = sfft.irfft2(ft[0], s=(24, 24))
    expected = np.roll(
        np.exp(-((np.arange(24)[:, None] - 12) ** 2 + (np.arange(24) - 12) ** 2) / 8),
        (-12, -12),
        axis=(0, 1),
    )
    assert_allclose(patch, expected, atol=1e-3)


def _blobs(size: int, offsets, sigma: float = 1.5) -> np.ndarray:
    """Gaussian blobs around the center (size // 2) of a cube."""
    inds = np.indices((size,) * 3, dtype=np.float64)
    c = size // 2
    vol = sum(
        np.exp(-sum((i - (c + o)) ** 2 for i, o in zip(inds, off)) / (2 * sigma**2))
        for off in offsets
    )
    return vol.astype(np.float32)


def _realspace_projection(vol: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Sum along z of the volume sampled at ``matrix @ (z, y, x)`` about the center."""
    from scipy import ndimage as ndi

    size = vol.shape[0]
    c = size // 2
    zz, yy, xx = np.indices((2 * size, size, size), dtype=np.float64)
    coords = np.stack([zz - size, yy - c, xx - c]).reshape(3, -1)
    values = ndi.map_coordinates(vol.astype(np.float64), matrix @ coords + c, order=3)
    return values.reshape(2 * size, size, size).sum(axis=0)


BLOB_OFFSETS = [(0, 0, 0), (2, -3, 1), (-2, 2, 3)]


@pytest.mark.parametrize("size", [16, 17, 20, 21])
def test_volume_projector(size):
    from scipy.spatial.transform import Rotation

    vol = _blobs(size, BLOB_OFFSETS)
    rotations = Rotation.random(5, random_state=3).as_matrix()
    projector = VolumeProjector(vol)
    out = projector(rotations)
    assert out.shape == (5, size, size)
    ref = np.stack([_realspace_projection(vol, m) for m in rotations])
    assert np.abs(out - ref).max() < 0.04 * np.abs(ref).max()
    # batch dimensions, and the reflection along the y axis of the projections
    flip = np.diag([1.0, -1.0, 1.0])
    out = projector(np.stack([rotations[:2], rotations[:2] @ flip]))
    assert out.shape == (2, 2, size, size)
    # the row 0 of even-sized projections is not mirrored inside the projections
    start = 1 - size % 2
    assert_allclose(out[1, :, start:], out[0, :, start:][:, ::-1], atol=1e-5)


def test_volume_projector_magnified():
    from scipy.spatial.transform import Rotation

    # a Gaussian, whose projection along any ray is known
    size, sigma = 32, 2.0
    center = np.array([2.0, -3.0, 4.0])
    grid = np.arange(size) - size // 2
    coords = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), axis=-1)
    vol = np.exp(-np.sum((coords - center) ** 2, axis=-1) / (2 * sigma**2))
    projector = VolumeProjector(vol)
    rotations = Rotation.random(3, random_state=1).as_matrix()
    # the projection planes are magnified and sheared
    rng = np.random.default_rng(2)
    planes = np.tile(np.eye(3), (3, 1, 1))
    planes[:, 1:, 1:] += rng.uniform(-0.06, 0.06, size=(3, 2, 2))
    matrices = rotations @ np.linalg.inv(planes)
    out = projector(matrices)
    uy, ux = np.meshgrid(grid, grid, indexing="ij")
    for img, mat in zip(out, matrices):
        # line integrals along the rays through the points of the projection plane
        direction = mat[:, 0] / np.linalg.norm(mat[:, 0])
        rel = np.stack([uy, ux], axis=-1) @ mat[:, 1:].T - center
        perp = rel - (rel @ direction)[..., np.newaxis] * direction
        dist2 = np.sum(perp**2, axis=-1)
        expected = np.sqrt(2 * np.pi) * sigma * np.exp(-dist2 / (2 * sigma**2))
        # as accurate as the projections of rotations (0.2-0.5%)
        assert np.abs(img - expected).max() < 0.01 * expected.max()
    # the background is the same in a magnified projection
    out = VolumeProjector(np.full((10, 10, 10), 2.0))(np.diag([1.0, 0.8, 0.8]))
    assert_allclose(out, 20.0, rtol=1e-5)


def test_volume_projector_background():
    # the average near the faces is projected exactly
    out = VolumeProjector(np.full((10, 10, 10), 2.0))(np.eye(3)[np.newaxis])
    assert_allclose(out, 20.0, rtol=1e-5)
    with pytest.raises(ValueError):
        VolumeProjector(np.zeros((8, 8, 9)))
    with pytest.raises(ValueError):
        VolumeProjector(np.zeros((8, 8, 8)), padding=0.5)
    with pytest.raises(ValueError):
        VolumeProjector(np.zeros((8, 8, 8)))(np.eye(4))
