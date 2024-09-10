import numpy as np
from acryo import pipe
import tempfile
from pathlib import Path


def test_rescale():
    @pipe.provider_function
    def fn(scale: float, value=1.0):
        n = int(10 * scale)
        img = np.full((n, n, n), value)
        return img

    out = fn(2.0)(1.0)
    assert out.shape == (10, 10, 10)
    out = fn(2.0)(2.0)
    assert out.shape == (20, 20, 20)


def test_basic_calculation():
    @pipe.provider_function
    def f0(scale: float, value=0):
        return np.full((2, 2, 2), value)

    assert np.all((f0(1) + f0(2))(0.1) == 3)
    assert np.all((f0(1) - f0(2))(0.1) == -1)
    assert np.all((f0(1) * f0(2))(0.1) == 2)
    assert np.all((f0(1) / f0(2))(0.1) == 0.5)


def test_readers():
    import tifffile
    import mrcfile

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tifffile.imwrite(tmpdir / "test.tif", np.ones((10, 10, 10), dtype=np.float32))
        mrcfile.new(tmpdir / "test.mrc", np.ones((10, 10, 10), dtype=np.float32))
        pipe.from_file(tmpdir / "test.tif")(0.5)
        pipe.from_file(tmpdir / "test.mrc")(0.6)
        pipe.from_array(np.ones((10, 10, 10)), original_scale=0.6)(0.7)
        pipe.from_arrays(
            [np.ones((10, 10, 10)), np.ones((8, 8, 8))], original_scale=0.6
        )(0.7)
        pipe.from_atoms(np.array([[0, 0, 1], [1, 0, 0]]))(0.4)
        pipe.from_gaussian((9, 9, 9))(0.5)


def test_converters():
    pipe.center_by_mass()(np.ones((10, 10, 10)), scale=0.5)
    pipe.closing(1.0)(np.ones((10, 10, 10)), scale=0.5)
    pipe.dilation(1.0)(np.ones((10, 10, 10)), scale=0.5)
    pipe.gaussian_smooth(0.9)(np.ones((10, 10, 10)), scale=0.5)
    pipe.lowpass_filter(0.2)(np.ones((10, 10, 10)), scale=0.5)
    pipe.highpass_filter(0.2)(np.ones((10, 10, 10)), scale=0.5)
    pipe.gaussian_filter(sigma=1.0)(np.ones((10, 10, 10)), scale=0.5)
    pipe.soft_otsu()(np.ones((10, 10, 10)), scale=0.5)
