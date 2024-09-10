from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._base import TomographyInput
from acryo._types import pixel
from acryo.backend import Backend, AnyArray
from acryo.backend._pcc import subpixel_pcc, pcc_landscape
from acryo.backend._zncc import (
    subpixel_ncc,
    subpixel_zncc,
    ncc_landscape_with_crop,
    zncc_landscape_with_crop,
)
from acryo.backend._fsc import subpixel_fsc, fsc_landscape
from acryo.backend._zncc import subpixel_zncc, zncc_landscape_with_crop, ncc, zncc
from acryo.backend._fsc import subpixel_fsc, fsc_landscape, fsc


class PCCAlignment(TomographyInput):
    """Alignment model using phase cross correlation."""

    def _optimize(
        self,
        subvolume: AnyArray[np.complex64],
        template: AnyArray[np.complex64],
        max_shifts: tuple[pixel, pixel, pixel],
        quaternion: NDArray[np.float32],
        pos: NDArray[np.float32],
        backend: Backend,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
        """Optimize."""
        mw = self._get_missing_wedge_mask(quaternion, backend)
        shift, pcc = subpixel_pcc(
            subvolume * mw,
            template * mw,
            upsample_factor=20,
            max_shifts=max_shifts,
            backend=backend,
        )
        return shift, self._DUMMY_QUAT, pcc

    def _score(
        self,
        subvolume: AnyArray[np.complex64],
        template: AnyArray[np.complex64],
        quaternion: NDArray[np.float32],
        pos: NDArray[np.float32],
        backend: Backend,
    ) -> float:
        """Score."""
        mw = self._get_missing_wedge_mask(quaternion, backend)
        return subpixel_pcc(
            subvolume * mw,
            template * mw,
            upsample_factor=1,
            max_shifts=(0, 0, 0),
            backend=backend,
        )[1]

    def _landscape(
        self,
        subvolume: AnyArray[np.complex64],
        template: AnyArray[np.complex64],
        max_shifts: tuple[float, float, float],
        quaternion: NDArray[np.float32],
        pos: NDArray[np.float32],
        backend: Backend,
    ) -> AnyArray[np.float32]:
        """Compute landscape."""
        mw = self._get_missing_wedge_mask(quaternion, backend)
        return pcc_landscape(
            subvolume * mw,
            template * mw,
            max_shifts=max_shifts,
            backend=backend,
        )


class NCCAlignment(TomographyInput):
    """Alignment model using zero-mean cross correlation."""

    def _optimize(
        self,
        subvolume: AnyArray[np.complex64],
        template: AnyArray[np.complex64],
        max_shifts: tuple[pixel, pixel, pixel],
        quaternion: NDArray[np.float32],
        pos: NDArray[np.float32],
        backend: Backend,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
        """Optimize."""
        mw = self._get_missing_wedge_mask(quaternion, backend)
        shift, zncc = subpixel_ncc(
            backend.ifftn(subvolume * mw).real,
            backend.ifftn(template * mw).real,
            max_shifts=max_shifts,
            backend=backend,
        )
        return shift, self._DUMMY_QUAT, zncc

    def _score(
        self,
        subvolume: AnyArray[np.complex64],
        template: AnyArray[np.complex64],
        quaternion: NDArray[np.float32],
        pos: NDArray[np.float32],
        backend: Backend,
    ) -> float:
        mw = self._get_missing_wedge_mask(quaternion, backend)
        return ncc(
            backend.ifftn(subvolume * mw).real,
            backend.ifftn(template * mw).real,
            backend=backend,
        )

    def _landscape(
        self,
        subvolume: AnyArray[np.complex64],
        template: AnyArray[np.complex64],
        max_shifts: tuple[float, float, float],
        quaternion: NDArray[np.float32],
        pos: NDArray[np.float32],
        backend: Backend,
    ) -> AnyArray[np.float32]:
        """Compute landscape."""
        mw = self._get_missing_wedge_mask(quaternion, backend)
        return ncc_landscape_with_crop(
            backend.ifftn(subvolume * mw).real,
            backend.ifftn(template * mw).real,
            max_shifts=max_shifts,
            backend=backend,
        )


class ZNCCAlignment(TomographyInput):
    """Alignment model using zero-mean normalized cross correlation."""

    def _optimize(
        self,
        subvolume: AnyArray[np.complex64],
        template: AnyArray[np.complex64],
        max_shifts: tuple[pixel, pixel, pixel],
        quaternion: NDArray[np.float32],
        pos: NDArray[np.float32],
        backend: Backend,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
        """Optimize."""
        mw = self._get_missing_wedge_mask(quaternion, backend)
        shift, zncc = subpixel_zncc(
            backend.ifftn(subvolume * mw).real,
            backend.ifftn(template * mw).real,
            max_shifts=max_shifts,
            backend=backend,
        )
        return shift, self._DUMMY_QUAT, zncc

    def _score(
        self,
        subvolume: AnyArray[np.complex64],
        template: AnyArray[np.complex64],
        quaternion: NDArray[np.float32],
        pos: NDArray[np.float32],
        backend: Backend,
    ) -> float:
        mw = self._get_missing_wedge_mask(quaternion, backend)
        return zncc(
            backend.ifftn(subvolume * mw).real,
            backend.ifftn(template * mw).real,
            backend=backend,
        )

    def _landscape(
        self,
        subvolume: AnyArray[np.complex64],
        template: AnyArray[np.complex64],
        max_shifts: tuple[float, float, float],
        quaternion: NDArray[np.float32],
        pos: NDArray[np.float32],
        backend: Backend,
    ) -> AnyArray[np.float32]:
        """Compute landscape."""
        mw = self._get_missing_wedge_mask(quaternion, backend)
        return zncc_landscape_with_crop(
            backend.ifftn(subvolume * mw).real,
            backend.ifftn(template * mw).real,
            max_shifts=max_shifts,
            backend=backend,
        )


class FSCAlignment(TomographyInput):
    """Alignment model using Fourier shell correlation."""

    def _optimize(
        self,
        subvolume: AnyArray[np.complex64],
        template: AnyArray[np.complex64],
        max_shifts: tuple[pixel, pixel, pixel],
        quaternion: NDArray[np.float32],
        pos: NDArray[np.float32],
        backend: Backend,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], float]:
        """Optimize."""
        mw = self._get_missing_wedge_mask(quaternion, backend)
        shift, fsc = subpixel_fsc(
            subvolume * mw,
            template * mw,
            max_shifts=max_shifts,
            backend=backend,
        )
        return shift, self._DUMMY_QUAT, fsc

    def _score(
        self,
        subvolume: AnyArray[np.complex64],
        template: AnyArray[np.complex64],
        quaternion: NDArray[np.float32],
        pos: NDArray[np.float32],
        backend: Backend,
    ) -> float:
        """Score."""
        mw = self._get_missing_wedge_mask(quaternion, backend)
        return fsc(subvolume * mw, template * mw, backend=backend)

    def _landscape(
        self,
        subvolume: AnyArray[np.complex64],
        template: AnyArray[np.complex64],
        max_shifts: tuple[float, float, float],
        quaternion: NDArray[np.float32],
        pos: NDArray[np.float32],
        backend: Backend,
    ) -> AnyArray[np.float32]:
        """Compute landscape."""
        mw = self._get_missing_wedge_mask(quaternion, backend)
        return fsc_landscape(
            subvolume * mw,
            template * mw,
            max_shifts=max_shifts,
            backend=backend,
        )
