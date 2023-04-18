# pyright: reportPrivateImportUsage=false

from __future__ import annotations
from typing import Generic, TypeVar, TYPE_CHECKING
import inspect
import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:
    from ._base import BaseAlignmentModel, TemplateType

_T = TypeVar("_T", bound="BaseAlignmentModel")

class ParametrizedModel(Generic[_T]):
    def __init__(self, cls: type[_T], **params):
        self._cls = cls
        self._params = params
    
    def create_model(self, template: TemplateType, mask: NDArray[np.float32] | None = None) -> _T:
        bound = inspect.signature(self._cls).bind_partial(**self._params)
        return self._cls(template, mask, *bound.args, **bound.kwargs)

    def __call__(self, template: TemplateType, mask: NDArray[np.float32] | None = None) -> _T:
        return self.create_model(template, mask)

    def __repr__(self) -> str:
        _args = ", ".join(f"{k}={v!r}" for k, v in self._params.items())
        return f"ParametrizedModel[{self._cls.__name__}]{_args})"

    @property
    def quaternions(self) -> NDArray[np.float32]:
        from acryo._rotation import normalize_rotations
        
        rotations = self._params.get("rotations", None)
        return normalize_rotations(rotations)

    @property
    def has_rotation(self) -> bool:
        """If the alignment model to be created has rotation optimization."""
        return self.quaternions.shape[0] > 1