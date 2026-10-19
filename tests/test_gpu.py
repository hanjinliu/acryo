from concurrent.futures import ThreadPoolExecutor
import threading
import time

import numpy as np
from numpy.testing import assert_allclose
import pytest

from acryo.backend import NUMPY_BACKEND, Backend
from acryo.backend._gpu import DeviceMemoryBudget, backend_of, run_on_device


class _FakeOOM(Exception):
    pass


class _FakeDevice:
    """A device whose memory is allocated by the tasks, to test the budget."""

    def __init__(self, total: int):
        self.total = total
        self.used = 0
        self.peak = 0
        self.released = 0
        self.lock = threading.Lock()

    def free_bytes(self) -> int:
        return self.total - self.used

    def release_cache(self) -> None:
        self.released += 1

    def task(self, nbytes: int, duration: float = 0.01):
        with self.lock:
            if self.used + nbytes > self.total:
                raise _FakeOOM()
            self.used += nbytes
            self.peak = max(self.peak, self.used)
        time.sleep(duration)
        with self.lock:
            self.used -= nbytes
        return nbytes

    def budget(self, fraction: float = 0.7) -> DeviceMemoryBudget:
        return DeviceMemoryBudget(
            self.free_bytes,
            self.release_cache,
            lambda e: isinstance(e, _FakeOOM),
            fraction=fraction,
        )


def test_budget_limits_parallel_tasks():
    device = _FakeDevice(1000)
    budget = device.budget()
    with ThreadPoolExecutor(16) as executor:
        results = list(
            executor.map(lambda _: budget.run(200, device.task, 200), range(40))
        )
    assert results == [200] * 40
    assert budget.capacity == 700
    # at most three tasks (600 bytes) at a time, within the capacity
    assert device.peak <= 600
    assert device.peak > 200  # but in parallel
    assert budget.reserved == 0
    assert device.released == 0


def test_budget_runs_large_task_alone():
    device = _FakeDevice(1000)
    budget = device.budget()
    sizes = [100, 900, 100, 100, 900, 100]
    with ThreadPoolExecutor(6) as executor:
        results = list(executor.map(lambda n: budget.run(n, device.task, n), sizes))
    assert results == sizes
    assert device.peak <= 1000


def test_budget_retries_out_of_memory_alone():
    device = _FakeDevice(1000)
    budget = device.budget(fraction=1.0)
    calls = []

    def _task(n: int):
        # the memory is underestimated by the budget
        calls.append(n)
        return device.task(2 * n, duration=0.02)

    with ThreadPoolExecutor(8) as executor:
        results = list(executor.map(lambda _: budget.run(300, _task, 300), range(8)))
    assert results == [600] * 8
    assert device.peak <= 1000
    assert device.released > 0  # the cache is released before the retry
    assert len(calls) > 8


def test_budget_raises_memory_error():
    device = _FakeDevice(1000)
    budget = device.budget()
    with pytest.raises(MemoryError):
        budget.run(100, device.task, 2000)
    # other errors are not retried
    calls = []

    def _fail():
        calls.append(None)
        raise ValueError("not OOM")

    with pytest.raises(ValueError):
        budget.run(100, _fail)
    assert len(calls) == 1
    assert budget.reserved == 0


def test_run_on_numpy():
    assert run_on_device(NUMPY_BACKEND, 10**15, lambda x, y=1: x + y, 1, y=2) == 3
    assert backend_of(np.zeros(3)) is NUMPY_BACKEND
    assert not NUMPY_BACKEND.is_gpu
    assert NUMPY_BACKEND.xp is np


@pytest.fixture
def cupy_backend():
    cupy = pytest.importorskip("cupy")
    try:
        cupy.cuda.runtime.getDeviceCount()
    except Exception:
        pytest.skip("No GPU is available.")
    return Backend("cupy")


def test_preprocess_on_gpu(cupy_backend):
    from acryo.ctf import CTFCorrection, CTFModel, TiltSeriesCTF
    from acryo.projection import TiltImagePreprocessor, TiltSeries

    rng = np.random.default_rng(0)
    image = rng.normal(size=(300, 340)).astype(np.float32)
    ts = TiltSeries(
        tilt_angles=[40.0],
        tilt_axis_angle=80.0,
        ctf=TiltSeriesCTF(defocus_u=[-2.0], defocus_v=[-2.3], astigmatism_angle=20),
    )
    preprocessor = TiltImagePreprocessor(low=0.01, falloff=0.02).with_filters(
        ts.ctf_correction(), CTFCorrection(CTFModel.from_kv(300, 2.7))
    )
    for binsize in [1, 2, 3]:
        ref = preprocessor(image, scale=0.3, binsize=binsize, backend=NUMPY_BACKEND)
        out = preprocessor(image, scale=0.3, binsize=binsize, backend=cupy_backend)
        assert isinstance(out, np.ndarray)
        assert_allclose(out, ref, atol=1e-3)


def test_reconstruct_tomogram_on_gpu(cupy_backend):
    from acryo.projection import TiltSeries, reconstruct_tomogram

    rng = np.random.default_rng(0)
    ts = TiltSeries(
        tilt_angles=np.linspace(-60, 60, 21), tilt_axis_angle=85.0, scale=0.5
    )
    images = rng.normal(size=(21, 80, 90)).astype(np.float32)
    kwargs = dict(patch_size=16, dtype=np.float32)
    ref = reconstruct_tomogram(
        ts, images, (20, 40, 44), backend=NUMPY_BACKEND, **kwargs
    )
    out = reconstruct_tomogram(ts, images, (20, 40, 44), backend=cupy_backend, **kwargs)
    assert_allclose(out, ref, atol=1e-4 * np.abs(ref).max())


def test_gpu_threads_within_memory(cupy_backend, monkeypatch):
    """Many threads share a small budget of the GPU memory."""
    import cupy
    from acryo.backend import _gpu
    from acryo.projection import TiltSeries, reconstruct_tomogram

    budget = _gpu._cupy_budget()
    # only one patch fits in the budget at a time
    monkeypatch.setattr(budget, "_fraction", 1e-6)
    ts = TiltSeries(
        tilt_angles=np.linspace(-60, 60, 11), tilt_axis_angle=85.0, scale=0.5
    )
    images = np.random.default_rng(1).normal(size=(11, 60, 60)).astype(np.float32)
    out = reconstruct_tomogram(
        ts,
        images,
        (16, 30, 30),
        patch_size=8,
        num_workers=16,
        backend=cupy_backend,
        dtype=np.float32,
    )
    assert np.all(np.isfinite(out))
    assert budget.reserved == 0
    cupy.get_default_memory_pool().free_all_blocks()
