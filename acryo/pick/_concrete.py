# pyright: reportPrivateImportUsage=false
from __future__ import annotations
from typing import Any

import numpy as np
from numpy.typing import NDArray
from dask import array as da
from scipy import ndimage as ndi

from acryo.pick._base import BasePickerModel, BaseTemplateMatcher
from acryo.backend import NUMPY_BACKEND
from acryo.backend._zncc import ncc_landscape_no_pad
from acryo.molecules import Molecules
from acryo._types import nm


class ZNCCTemplateMatcher(BaseTemplateMatcher):
    """
    Particle picking based on ZNCC template matching.

    Parameters
    ----------
    image : da.Array
        The input image.
    scale : float
        The scale of the image.
    min_distance : float
        The minimum distance between the picked particles.
    min_score : float
        The minimum score of the picked particles.
    boundary : str
        The boundary condition for the template matching.
    """

    def pick_molecules(
        self,
        image: da.Array,
        scale: nm = 1.0,
        *,
        min_distance: nm = 1.0,
        min_score: float = 0.02,
        boundary="nearest",
    ) -> Molecules:
        return super().pick_molecules(
            image,
            scale,
            boundary=boundary,
            min_distance=min_distance / scale,
            min_score=min_score,
        )

    def pick_in_chunk(
        self,
        image: NDArray[np.float32],
        templates: list[NDArray[np.float32]],
        min_distance: float,  # pixel
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

        pos = find_maxima(landscale_max, min_distance, min_score)
        argmax_indices = np.array(
            [img_argmax[tuple(np.round(p).astype(np.int32))] for p in pos]
        )
        score = _sample_score(landscale_max, pos)
        quats = self._index_to_quaternions(argmax_indices)
        offset = (np.array(templates[0].shape) + 1) / 2
        return pos + offset, quats, {"score": score}


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
        pos = find_maxima(img_filt, sigma, 0.0)
        return simple_pick(img_filt, pos)

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
        img_filt = _differece_of_gaussian(image, sigma_low, sigma_high)
        pos = find_maxima(img_filt, sigma_low, 0.0)
        return simple_pick(img_filt, pos)

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


def find_maxima(img, min_distance: float, min_intensity: float):
    img_max_maxfilt = maximum_filter(img, min_distance)
    is_maxima = (img_max_maxfilt == img) & (img > min_intensity)
    s0 = [[0, 0, 0], [0, 1, 0], [0, 0, 0]]
    s1 = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
    structure = np.stack([s0, s1, s0])
    label_img, nfeat = ndi.label(is_maxima, structure=structure)
    centers = ndi.center_of_mass(img, label_img, range(1, nfeat + 1))
    return np.array(centers, dtype=np.float32)


def simple_pick(img: NDArray[np.float32], pos: NDArray[np.float32]):
    score = _sample_score(img, pos)
    quats = np.zeros((pos.shape[0], 4), dtype=np.float32)
    quats[:, 3] = 1.0
    return pos, quats, {"score": score}


def _sample_score(img, pos: NDArray[np.float32]) -> NDArray[np.float32]:
    return ndi.map_coordinates(img, pos.T, order=3, mode="reflect")


def _differece_of_gaussian(
    image: NDArray[np.float32],
    sigma_low: float,
    sigma_high: float,
) -> NDArray[np.float32]:
    img_l = ndi.gaussian_filter(image, sigma_low)
    img_h = ndi.gaussian_filter(image, sigma_high)
    return img_l - img_h
