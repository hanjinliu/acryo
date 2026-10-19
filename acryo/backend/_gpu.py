"""Running GPU tasks in parallel threads within the device memory."""

from __future__ import annotations

from contextlib import contextmanager
import threading
from typing import TYPE_CHECKING, Any, Callable, Iterator, TypeVar

if TYPE_CHECKING:
    from acryo.backend._api import Backend

_R = TypeVar("_R")

# fraction of the free device memory that the tasks can reserve
_DEVICE_MEMORY_FRACTION = 0.7


class DeviceMemoryBudget:
    """Budget of the device memory shared by the tasks running in parallel threads.

    Each task reserves its estimated peak memory before running, and waits while the
    memory reserved by the running tasks plus its own exceeds the capacity. Therefore,
    the device memory is not exhausted however many threads run the tasks. The
    capacity is a fraction of the free memory (including the cache of the memory
    pool), which is updated whenever no task is running. A task larger than the
    capacity runs alone. If a task fails by out-of-memory nevertheless (such as by an
    underestimate or other processes), it is retried once alone after the cache is
    released.

    Tasks must not run another task of the same budget inside, which may deadlock.

    Parameters
    ----------
    free_bytes : callable
        Function that returns the free device memory in bytes.
    release_cache : callable
        Function that releases the cached device memory.
    is_out_of_memory : callable
        Function that returns whether an exception is caused by out-of-memory.
    fraction : float, default is 0.7
        Fraction of the free memory that the tasks can reserve.
    """

    def __init__(
        self,
        free_bytes: Callable[[], int],
        release_cache: Callable[[], None],
        is_out_of_memory: Callable[[BaseException], bool],
        fraction: float = _DEVICE_MEMORY_FRACTION,
    ):
        self._free_bytes = free_bytes
        self._release_cache = release_cache
        self._is_out_of_memory = is_out_of_memory
        self._fraction = fraction
        self._cond = threading.Condition()
        self._capacity = 0
        self._reserved = 0
        self._running = 0
        self._exclusive = False  # a task is running alone
        self._waiting_exclusive = 0

    @property
    def capacity(self) -> int:
        """The capacity in bytes, updated when no task is running."""
        return self._capacity

    @property
    def reserved(self) -> int:
        """Memory in bytes reserved by the running tasks."""
        return self._reserved

    def run(self, nbytes: int, func: Callable[..., _R], /, *args, **kwargs) -> _R:
        """Run ``func`` with ``nbytes`` of the device memory reserved."""
        with self._reserve(nbytes, exclusive=False):
            try:
                return func(*args, **kwargs)
            except BaseException as e:
                if not self._is_out_of_memory(e):
                    raise
        # retry alone after releasing the memory cached by the finished tasks
        with self._reserve(nbytes, exclusive=True):
            self._release_cache()
            try:
                return func(*args, **kwargs)
            except BaseException as e:
                if not self._is_out_of_memory(e):
                    raise
                raise MemoryError(
                    f"Out of device memory even with only one task running (estimated "
                    f"{nbytes / 1e6:.1f} MB, {self._free_bytes() / 1e6:.1f} MB free). "
                    "Use a smaller tilt series (larger binning) or patches."
                ) from e

    @contextmanager
    def _reserve(self, nbytes: int, exclusive: bool) -> Iterator[None]:
        with self._cond:
            if exclusive:
                self._waiting_exclusive += 1
            try:
                while not self._can_start(nbytes, exclusive):
                    self._cond.wait()
            finally:
                if exclusive:
                    self._waiting_exclusive -= 1
            if self._running == 0:
                self._capacity = int(self._fraction * self._free_bytes())
            self._running += 1
            self._reserved += nbytes
            self._exclusive = exclusive
        try:
            yield
        finally:
            with self._cond:
                self._running -= 1
                self._reserved -= nbytes
                if exclusive:
                    self._exclusive = False
                self._cond.notify_all()

    def _can_start(self, nbytes: int, exclusive: bool) -> bool:
        if self._exclusive:
            return False
        if self._running == 0:
            return True
        if exclusive or self._waiting_exclusive > 0:
            # an exclusive task waits for the running tasks, which are not added
            return False
        return self._reserved + nbytes <= self._capacity


_BUDGETS: dict[int, DeviceMemoryBudget] = {}
_BUDGETS_LOCK = threading.Lock()
_LOCAL = threading.local()


def _cupy_budget() -> DeviceMemoryBudget:
    """The budget of the current device (cupy), shared in the process."""
    import cupy

    device = cupy.cuda.Device().id
    with _BUDGETS_LOCK:
        if (budget := _BUDGETS.get(device)) is None:
            budget = _BUDGETS[device] = DeviceMemoryBudget(
                _cupy_free_bytes, _cupy_release_cache, _cupy_is_out_of_memory
            )
    return budget


def _cupy_free_bytes() -> int:
    import cupy

    free = cupy.cuda.runtime.memGetInfo()[0]
    return int(free + cupy.get_default_memory_pool().free_bytes())


def _cupy_release_cache() -> None:
    import cupy

    cupy.fft.config.get_plan_cache().clear()  # cuFFT plans of this thread
    cupy.get_default_memory_pool().free_all_blocks()


def _cupy_is_out_of_memory(exc: BaseException) -> bool:
    import cupy

    if isinstance(exc, cupy.cuda.memory.OutOfMemoryError):
        return True
    # cuFFT and cuSPARSE fail to allocate their workspace
    return "ALLOC_FAILED" in str(exc)


@contextmanager
def _thread_stream() -> Iterator[Any]:
    """Use a non-blocking stream of this thread, so that threads run concurrently."""
    import cupy

    streams: dict[int, Any] = _LOCAL.__dict__.setdefault("streams", {})
    device = cupy.cuda.Device().id
    if (stream := streams.get(device)) is None:
        stream = streams[device] = cupy.cuda.Stream(non_blocking=True)
    with stream:
        yield stream


def run_on_device(
    backend: Backend, nbytes: int, func: Callable[..., _R], /, *args, **kwargs
) -> _R:
    """Run ``func`` that computes on the device of the backend.

    For numpy, ``func`` is just called. For cupy, ``func`` runs on the stream of this
    thread, with ``nbytes`` of the device memory reserved in the budget of the device
    (see :class:`DeviceMemoryBudget`), so many threads can run ``func`` in parallel
    without running out of the device memory. ``func`` should return host arrays (or
    small device arrays), because the device memory is not reserved after it returns.
    """
    if not backend.is_gpu:
        return func(*args, **kwargs)

    def _func():
        with _thread_stream() as stream:
            out = func(*args, **kwargs)
            stream.synchronize()
        return out

    return _cupy_budget().run(nbytes, _func)


_CUPY_BACKEND: Backend | None = None


def backend_of(arr: Any) -> Backend:
    """The backend of the array (numpy or cupy)."""
    global _CUPY_BACKEND
    from acryo.backend._api import NUMPY_BACKEND, Backend

    if type(arr).__module__.split(".")[0] != "cupy":
        return NUMPY_BACKEND
    if _CUPY_BACKEND is None:
        _CUPY_BACKEND = Backend("cupy")
    return _CUPY_BACKEND


def synchronize(backend: Backend) -> None:
    """Wait for the device, such as after preparing the data shared by threads."""
    if backend.is_gpu:
        import cupy

        cupy.cuda.Device().synchronize()
