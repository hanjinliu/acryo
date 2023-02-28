# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from typing import Any, Sequence, SupportsIndex
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation
from dask import array as da

from acryo.pipe._classes import ImageProvider
from acryo.molecules import Molecules
from acryo._rotation import normalize_rotations
from acryo._utils import compose_matrices, missing_wedge_mask, delayed_affine
from acryo._types import nm


class BasePickerModel(ABC):
    def pick_molecules(
        self,
        image: da.Array,
        scale: nm = 1.0,
        **kwargs,
    ) -> Molecules:
        params, depth = self.get_params_and_depth(scale)
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        task: da.Array = image.map_overlap(
            self._pick_in_chunk_wrapped,
            **params,
            **kwargs,
            # dask parameters
            depth=depth,
            trim=False,
            boundary="nearest",
            dtype=object,
            meta=np.array([[[]]]),
        )
        boxes: Sequence[MoleculesBox] = task.compute().ravel()
        mole = Molecules.concat([box.to_molecules() for box in boxes])
        mole._pos *= scale
        return mole

    def _pick_in_chunk_wrapped(
        self,
        image: NDArray[np.float32],
        block_info: dict,
        **kwargs,
    ) -> NDArray[np.object_]:
        pos, quats, features = self.pick_in_chunk(image, **kwargs)

        locs: list[tuple[int, int]] = block_info[None]["array-location"]
        for i, (start, _) in enumerate(locs):
            pos[:, i] += start

        return np.array([[[MoleculesBox(pos, quats, features)]]], dtype=object)

    @abstractmethod
    def pick_in_chunk(
        self, image: NDArray[np.float32], **kwargs
    ) -> tuple[NDArray[np.float32], NDArray[np.uint16], Any]:
        ...

    @abstractmethod
    def get_params_and_depth(
        self, scale: nm
    ) -> tuple[dict[str, Any], int | Sequence[SupportsIndex]]:
        ...


class BaseTemplateMatcher(BasePickerModel):
    """Particle picking based on parallel template matching."""

    def __init__(
        self,
        template: NDArray[np.float32] | ImageProvider,
        rotation,
        tilt_range: tuple[float, float] = (-60, 60),
        order: int = 1,
    ) -> None:
        self._order = order
        self._quaternions = normalize_rotations(rotation)
        self._tilt_range = tilt_range
        self._template = template

    @property
    def order(self) -> int:
        """Interpolation order."""
        return self._order

    def get_params_and_depth(self, scale: nm):
        if isinstance(self._template, ImageProvider):
            template = self._template(scale)
        else:
            template = self._template

        rotators = [Rotation.from_quat(r).inv() for r in self._quaternions]
        _center = np.array(template.shape) / 2 - 0.5
        matrices = compose_matrices(_center, rotators)

        if self.order > 2:
            template = ndi.spline_filter(
                template,
                order=self.order,
                output=np.float32,  # type: ignore
                mode="constant",
            )
        tasks = [
            delayed_affine(template, mtx, order=self.order, prefilter=False)
            for mtx in matrices
        ]
        mask = missing_wedge_mask(
            Rotation.from_quat([0, 0, 0, 1]), self._tilt_range, template.shape
        )
        out: list[NDArray[np.float32]] = da.compute(tasks)[0]  # rotated templates
        templates = [o * mask for o in out]
        depth = tuple(np.ceil(np.array(templates[0].shape) / 2).astype(np.uint16))
        return {"templates": templates}, depth

    def _index_to_quaternions(self, argmax_indices):
        return np.take_along_axis(
            self._quaternions, argmax_indices[:, np.newaxis], axis=0
        )

    @abstractmethod
    def pick_in_chunk(
        self, image: NDArray[np.float32], templates: list[NDArray[np.float32]], **kwargs
    ) -> tuple[NDArray[np.float32], NDArray[np.float32], Any]:
        ...


# NOTE: np.array(mole) is dangerous.
class MoleculesBox:
    def __init__(self, pos, quats, features) -> None:
        self._pos = pos
        self._quats = quats
        self._features = features

    def to_molecules(self) -> Molecules:
        return Molecules.from_quat(self._pos, self._quats, features=self._features)
