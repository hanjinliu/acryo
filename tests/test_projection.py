import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy import fft as sfft

from acryo.projection import (
    CentralSliceInsertion,
    DirectFourierReconstructor,
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


def _gaussian_tilt_images(center, sigma: float, voxel_size: float) -> np.ndarray:
    """Projections of a 3D Gaussian of amplitude 1 at center (pixels of the images).

    The projection is the sum over the voxels of the given size.
    """
    yy, xx = np.indices(IMAGE_SHAPE, dtype=np.float64)
    images = []
    for mat in PROJECTIONS:
        _, py, px = mat @ center + np.array([0, *IMAGE_SHAPE]) // 2
        r2 = (yy - py) ** 2 + (xx - px) ** 2
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
    # the device memory is estimated for each batch
    assert insertion._device_nbytes(2, 12) == 2 * insertion._device_nbytes(1, 12)


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
