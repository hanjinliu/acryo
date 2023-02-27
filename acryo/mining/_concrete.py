from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dask import array as da
from dask import delayed
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation

from acryo.mining._base import MinerBase
from acryo.molecules import Molecules
from acryo.pipe._classes import ImageProvider
from acryo._correlation import ncc_landscape_no_pad
from acryo._rotation import normalize_rotations
from acryo._types import nm
from acryo._utils import compose_matrices, missing_wedge_mask


class TemplateMatcher(MinerBase):
    """Tomogram mining based on template matching."""

    def __init__(
        self,
        template: NDArray[np.float32] | ImageProvider,
        rotation,
        tilt_range: tuple[float, float] = (-60, 60),
        order: int = 1,
    ) -> None:
        super().__init__(order=order)
        self._quaternions = normalize_rotations(rotation)
        self._tilt_range = tilt_range
        self._template = template

    def _get_template_input(self, scale: nm) -> list[NDArray[np.float32]]:
        if isinstance(self._template, ImageProvider):
            template = self._template(scale)
        else:
            template = self._template

        rotators = [Rotation.from_quat(r).inv() for r in self._quaternions]
        _center = np.array(template.shape) / 2 - 0.5
        matrices = compose_matrices(_center, rotators)

        # TODO: spline filter
        tasks = [
            delayed_affine(template, mtx, order=self.order, prefilter=False)
            for mtx in matrices
        ]
        mask = missing_wedge_mask(
            Rotation.from_quat([0, 0, 0, 1]), self._tilt_range, template.shape
        )
        out = da.compute(tasks)[0]  # rotated templates
        return [o * mask for o in out]

    def find_molecules(
        self, image: da.Array, scale: nm = 1.0, radius: nm = 1.0
    ) -> Molecules:
        templates = self._get_template_input(scale)
        depth = tuple(np.ceil(np.array(templates[0].shape) / 2).astype(np.uint16))

        all_landscapes = da.stack(
            [
                image.map_overlap(
                    ncc_landscape_no_pad,
                    template=template,
                    depth=depth,
                    boundary="nearest",
                    dtype=np.float32,
                )
                for template in templates
            ],
            axis=0,
        )

        img_argmax = da.argmax(all_landscapes, axis=0)
        img_max = da.choose(img_argmax, all_landscapes)  # 3D array of maxima

        img_max_maxfilt = dask_maximum_filter(img_max, radius / scale)
        max_indices = da.where(img_max_maxfilt == img_max)
        argmax_indices = img_argmax[max_indices]
        argmax_indices_result, max_indices_result = da.compute(
            argmax_indices, max_indices
        )

        pos = np.stack(max_indices_result, axis=1).astype(np.float32) * scale
        quats = np.choose(argmax_indices_result, self._quaternions)

        return Molecules.from_quat(pos, quats)


delayed_affine = delayed(ndi.affine_transform)


def dask_maximum_filter(image: da.Array, radius: float) -> da.Array:
    r_int = int(np.ceil(radius))
    size = 2 * r_int + 1
    zz, yy, xx = np.indices((size,) * 3)
    foot = (zz - r_int) ** 2 + (yy - r_int) ** 2 + (xx - r_int) ** 2 <= radius**2
    return image.map_overlap(
        ndi.maximum_filter, footprint=foot, depth=size, mode="nearest"
    )
