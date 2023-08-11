# pyright: reportPrivateImportUsage=false
from __future__ import annotations
from typing import Any

import numpy as np
from numpy.typing import NDArray
from dask import array as da
from scipy import ndimage as ndi

from acryo.pick._base import BasePickerModel, BaseTemplateMatcher
from acryo.molecules import Molecules
from acryo.backend import NUMPY_BACKEND
from acryo.backend._zncc import ncc_landscape_no_pad
from acryo._types import nm


class ZNCCTemplateMatcher(BaseTemplateMatcher):
    """
    Particle picking based on ZNCC template matching.

    Parameters
    ----------
    template : 3D array or ImageProvider
        Template image.
    rotation : range-like
        3D rotation of template image in degrees.
    tilt_range: tuple of float, default is (-60, 60)
        Tilt range in degrees.
    order : int, default is 1
        Order of interpolation.
    """

    def pick_molecules(
        self,
        image: da.Array,
        scale: nm = 1.0,
        *,
        min_distance: nm = 1.0,
        min_score: float = 0.02,
    ) -> Molecules:
        return super().pick_molecules(
            image, scale, min_distance=min_distance, min_score=min_score
        )

    def pick_in_chunk(
        self,
        image: NDArray[np.float32],
        templates: list[NDArray[np.float32]],
        min_distance: float,
        min_score: float,
    ):
        all_landscapes = np.stack(
            [
                ncc_landscape_no_pad(
                    image - np.mean(image),
                    template - np.mean(template),
                    NUMPY_BACKEND,
                )
                for template in templates
            ],
            axis=0,
        )

        img_argmax = np.argmax(all_landscapes, axis=0)
        landscale_max = np.max(all_landscapes, axis=0)

        max_indices = find_maxima(landscale_max, min_distance, min_score)
        argmax_indices = img_argmax[max_indices]
        score = landscale_max[max_indices]

        pos = np.stack(max_indices, axis=1).astype(np.float32)
        quats = self._index_to_quaternions(argmax_indices)
        return pos, quats, {"score": score}


class LoGPicker(BasePickerModel):
    """Particle picking based on Laplacian of Gaussian."""

    def __init__(self, sigma: nm = 3.5) -> None:
        self._sigma = sigma

    def pick_in_chunk(
        self,
        image: NDArray[np.float32],
        sigma: float,
    ) -> tuple[NDArray[np.float32], NDArray[np.uint16], Any]:
        img_filt = -ndi.gaussian_laplace(image, sigma)
        max_indices = find_maxima(img_filt, sigma, 0.0)
        score = img_filt[max_indices]
        pos = np.stack(max_indices, axis=1).astype(np.float32)
        quats = np.zeros((pos.shape[0], 4), dtype=np.float32)
        quats[:, 3] = 1.0
        return pos, quats, {"score": score}

    def get_params_and_depth(self, scale: nm):
        sigma_px = self._sigma / scale
        depth = int(np.ceil(sigma_px * 2))
        return {"sigma": sigma_px}, depth


class DoGPicker(BasePickerModel):
    """Particle picking based on Difference of Gaussian."""

    def __init__(self, sigma_low: nm = 3.5, sigma_high: nm = 5.0) -> None:
        if sigma_low >= sigma_high:
            raise ValueError("sigma_low must be smaller than sigma_high")
        self._sigma_low = sigma_low
        self._sigma_high = sigma_high

    def pick_in_chunk(
        self,
        image: NDArray[np.float32],
        sigma_low: float,
        sigma_high: float,
    ) -> tuple[NDArray[np.float32], NDArray[np.uint16], Any]:
        img_filt = ndi.gaussian_filter(image, sigma_low) - ndi.gaussian_filter(
            image, sigma_high
        )
        max_indices = find_maxima(img_filt, sigma_low, 0.0)
        score = img_filt[max_indices]
        pos = np.stack(max_indices, axis=1).astype(np.float32)
        quats = np.zeros((pos.shape[0], 4), dtype=np.float32)
        quats[:, 3] = 1.0
        return pos, quats, {"score": score}

    def get_params_and_depth(self, scale: nm):
        sigma1_px = self._sigma_low / scale
        sigma2_px = self._sigma_high / scale
        depth = int(np.ceil(sigma1_px * 2))
        return {"sigma_low": sigma1_px, "sigma_high": sigma2_px}, depth


def maximum_filter(image: da.Array, radius: float) -> NDArray[np.float32]:
    if radius < 1:
        return image
    r_int = int(np.ceil(radius))
    size = 2 * r_int + 1
    zz, yy, xx = np.indices((size,) * 3)
    foot = (zz - r_int) ** 2 + (yy - r_int) ** 2 + (xx - r_int) ** 2 <= radius**2
    return ndi.maximum_filter(image, footprint=foot, mode="nearest")  # type: ignore


def find_maxima(img, min_distance, min_intensity):
    img_max_maxfilt = maximum_filter(img, min_distance)
    return np.where((img_max_maxfilt == img) & (img > min_intensity))
