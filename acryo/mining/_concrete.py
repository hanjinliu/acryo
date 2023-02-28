# pyright: reportPrivateImportUsage=false
from __future__ import annotations
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from dask import array as da
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation

from acryo.mining._base import MinerBase
from acryo.molecules import Molecules
from acryo.pipe._classes import ImageProvider
from acryo._correlation import ncc_landscape_no_pad
from acryo._rotation import normalize_rotations
from acryo._types import nm
from acryo._utils import compose_matrices, missing_wedge_mask, delayed_affine


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
        self,
        image: da.Array,
        scale: nm = 1.0,
        radius: nm = 1.0,
        threshold: float = 0.8,
    ) -> Molecules:
        templates = self._get_template_input(scale)
        depth = tuple(np.ceil(np.array(templates[0].shape) / 2).astype(np.uint16))

        boxes: Sequence[MoleculesBox] = (
            da.map_overlap(
                self._find_molecules_in_block,
                image=image,
                templates=templates,
                radius=radius,
                scale=scale,
                threshold=threshold,
                # dask parameters
                depth=depth,
                boundary="nearest",
                dtype=object,
                meta=np.array([], dtype=object),
            )
            .compute()
            .ravel()
        )

        return Molecules.concat([box.to_molecules() for box in boxes])

    def _find_molecules_in_block(
        self,
        image: NDArray[np.float32],
        templates: list[NDArray[np.float32]],
        radius: nm,
        scale: nm,
        threshold: float,
        block_info: dict,
    ) -> NDArray[np.object_]:
        all_landscapes = np.stack(
            [
                ncc_landscape_no_pad(
                    image - np.mean(image), template - np.mean(template)
                )
                for template in templates
            ],
            axis=0,
        )

        img_argmax = np.argmax(all_landscapes, axis=0)
        # img_max = np.choose(img_argmax, all_landscapes)  # 3D array of maxima
        img_max = np.max(all_landscapes, axis=0)

        img_max_maxfilt = maximum_filter(img_max, radius / scale)
        max_indices = np.where((img_max_maxfilt == img_max) & (img_max > threshold))
        argmax_indices = img_argmax[max_indices]

        locs: list[tuple[int, int]] = block_info[None]["array-location"]

        for i, (start, _) in enumerate(locs):
            max_indices[i][:] += start
        pos = np.stack(max_indices, axis=1).astype(np.float32) * scale
        quats = np.take_along_axis(
            self._quaternions, argmax_indices[:, np.newaxis], axis=0
        )
        return np.array([[MoleculesBox(pos, quats)]], dtype=object)


def maximum_filter(image: da.Array, radius: float) -> NDArray[np.float32]:
    r_int = int(np.ceil(radius))
    size = 2 * r_int + 1
    zz, yy, xx = np.indices((size,) * 3)
    foot = (zz - r_int) ** 2 + (yy - r_int) ** 2 + (xx - r_int) ** 2 <= radius**2
    return ndi.maximum_filter(image, footprint=foot, mode="nearest")  # type: ignore


# NOTE: np.array(mole) is dangerous.
class MoleculesBox:
    def __init__(self, pos, quats) -> None:
        self._pos = pos
        self._quats = quats

    def to_molecules(self) -> Molecules:
        return Molecules.from_quat(self._pos, self._quats)
