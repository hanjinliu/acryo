# pyright: reportPrivateImportUsage=false
from __future__ import annotations

from typing import Any, Sequence, SupportsIndex
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation
from dask import array as da

from acryo.pipe._classes import ImageProvider
from acryo.tilt import TiltSeriesModel, no_wedge
from acryo.molecules import Molecules
from acryo._rotation import normalize_rotations
from acryo._utils import compose_matrices
from acryo._types import nm, RotationType
from acryo._typed_scipy import spline_filter, affine_transform
from acryo._dask import DaskTaskPool


class BasePickerModel(ABC):
    def pick_molecules(
        self,
        image: NDArray[np.number] | da.Array,
        scale: nm = 1.0,
        *,
        boundary="nearest",
        **kwargs,
    ) -> Molecules:
        """Pick molecules from image."""
        if isinstance(image, np.ndarray):
            image = da.asarray(image)
        if image.dtype.kind not in "fiub":
            raise ValueError("Image must be a numeric array.")
        if image.dtype != np.float32:
            image = image.astype(np.float32)
        params, depth = self.get_params_and_depth(scale)
        # if depth is too large
        if isinstance(depth, (int, np.integer)):
            depth = (depth, depth, depth)
        task: da.Array = image.map_overlap(
            self._pick_in_chunk_wrapped,
            **params,
            **kwargs,
            # dask parameters
            depth=[min(s, d) for s, d in zip(image.shape, depth)],
            trim=False,
            boundary=boundary,
            dtype=object,
            meta=np.array([]),
        )
        boxes: Sequence[MoleculesBox] = task.compute().ravel()
        mole = Molecules.concat([box.to_molecules() for box in boxes])
        mole._pos = (mole._pos - depth) * scale
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
        """Pick molecules inside a chunk of image."""

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
        rotation: RotationType | None = None,
        tilt: TiltSeriesModel | None = None,
        order: int = 1,
    ) -> None:
        self._order = order
        self._quaternions = normalize_rotations(rotation)
        if tilt is None:
            tilt = no_wedge()
        self._tilt_model = tilt
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
            template = spline_filter(
                template,
                order=self.order,
                output=np.float32,
                mode="constant",
            )

        pool = DaskTaskPool.from_func(affine_transform)
        for mtx in matrices:
            pool.add_task(template, mtx, order=self.order, prefilter=False)
        mask = self._tilt_model.create_mask(shape=template.shape)
        out = pool.compute()  # rotated templates
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
