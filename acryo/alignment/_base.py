from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, NamedTuple, Sequence
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial.transform import Rotation

from ._utils import normalize_rotations, lowpass_filter, lowpass_filter_ft
from .._types import Ranges
from .._utils import compose_matrices


class AlignmentResult(NamedTuple):
    """The optimal alignment result."""

    label: int
    shift: np.ndarray
    quat: np.ndarray
    corr: float


class BaseAlignmentModel(ABC):
    """
    The base class to implement alignment method.

    Must implement ``optimize`` and ``pre_transform``.
    """

    def __init__(
        self,
        template: np.ndarray | Sequence[np.ndarray],
        mask: np.ndarray | None = None,
    ):
        if isinstance(template, np.ndarray):
            if template.ndim in (3, 4):
                self._template = np.asarray(template)
                if template.ndim == 3:
                    self._n_templates = 1
                else:
                    self._n_templates = template.shape[0]
            else:
                raise TypeError("ndim must be 3 or 4.")
        else:
            self._template: np.ndarray = np.stack(template, axis=0)
            self._n_templates = self._template.shape[0]

        if mask is None:
            self.mask = 1
        else:
            if self._template.shape[-3:] != mask.shape:
                raise ValueError(
                    "Shape mismatch in zyx axes between tempalte image "
                    f"{self._template.shape} and mask image {mask.shape})."
                )
            self.mask = mask

        self._align_func = self._get_alignment_function()
        self.template_input = self._get_template_input()

    @abstractmethod
    def optimize(
        self,
        subvolume: np.ndarray,
        template: np.ndarray,
        max_shifts: tuple[float, ...],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Optimization of shift and rotation of subvolume.

        This method uses a subvolume and a template image to find the optimal
        shift/rotation to fit subvolume to template under the restriction of
        maximum shifts.

        Parameters
        ----------
        subvolume : np.ndarray
            Input subvolume. This array has the same dimensionality as the template.
        template : np.ndarray
            Template image.
        max_shifts : tuple of float
            Maximum shifts in each axis. This value should be float pixel. If
            optimization requires integer input, it is better to convert this
            parameter to tuple of integers.

        Returns
        -------
        (3,) np.ndarray, (4,) np.ndarray and float
            Local shift, local rotation and score. "Local" means you don't have
            to consider the position and orientation of molecules.
            - shift ... Shift in pixel.
            - rotation ... Rotation in quaternion. If this cannot be optimized by the
              implemented algorithm, this value can be ``[0., 0., 0., 0.]``.
            - score ... This value Must be a float and larger value should represent
              better results.
        """

    @abstractmethod
    def pre_transform(self, img: np.ndarray) -> np.ndarray:
        """Pre-transformation applied to input images (including template)."""

    def _get_template_input(self) -> np.ndarray:
        """
        Returns proper template image for alignment.

        Template dimensionality will be dispatched according to the input
        parameters. Returned template should be used in line of the
        :func:`get_alignment_function`.

        Returns
        -------
        ip.ImgArray
            Template image(s). Its axes varies depending on the input.

            - single template image ... 3D
            - many template images ... 4D
        """
        if self.is_multi_templates:
            template_input = np.stack(
                [self.pre_transform(tmp * self.mask) for tmp in self._template],
                axis=0,
            )
        else:
            template_input = self.pre_transform(self._template)
        return template_input

    def align(
        self,
        img: np.ndarray,
        max_shifts: tuple[float, float, float],
    ) -> AlignmentResult:
        """
        Align an image using current alignment parameters.

        Parameters
        ----------
        img : np.ndarray
            Subvolume to be aligned
        max_shifts : tuple[float, float, float]
            Maximum shifts along z, y, x axis in pixel.

        Returns
        -------
        AlignmentResult
            Result of alignment.
        """
        img_masked = img * self.mask
        return self._align_func(
            self.pre_transform(img_masked),
            self.template_input,
            max_shifts,
        )

    def fit(
        self,
        img: np.ndarray,
        max_shifts: tuple[float, float, float],
        cval: float | None = None,
    ) -> tuple[np.ndarray, AlignmentResult]:
        """
        Fit image to template based on the alignment model.

        Parameters
        ----------
        img : np.ndarray
            Input image that will be transformed.
        max_shifts : tuple[float, float, float]
            Maximum shifts along z, y, x axis in pixel.
        cval : float, optional
            Constant value for padding.

        Returns
        -------
        np.ndarray, AlignmentResult
            Transformed input image and the alignment result.
        """
        result = self.align(img, max_shifts=max_shifts)
        rotator = Rotation.from_quat(result.quat)
        matrix = compose_matrices(np.array(img.shape) / 2 - 0.5, [rotator])[0]  #
        if cval is None:
            _cval = np.percentile(img, 1)
        else:
            _cval = cval
        img_shifted = ndi.shift(img, result.shift, cval=_cval)
        img_trans = ndi.affine_transform(img_shifted, matrix, cval=_cval)
        return img_trans, result

    def _optimize_single(
        self,
        subvol: np.ndarray,
        template: np.ndarray,
        max_shifts: tuple[float, float, float],
    ) -> AlignmentResult:
        out = self.optimize(subvol, template, max_shifts)
        return AlignmentResult(0, *out)

    def _optimize_multiple(
        self,
        subvol: np.ndarray,
        template_list: Iterable[np.ndarray],
        max_shifts: tuple[float, float, float],
    ) -> AlignmentResult:
        all_shifts: list[np.ndarray] = []
        all_quat: list[np.ndarray] = []
        all_score: list[float] = []
        for template in template_list:
            shift, quat, score = self.optimize(
                subvol,
                template,
                max_shifts,
            )
            all_shifts.append(shift)
            all_quat.append(quat)
            all_score.append(score)

        iopt = int(np.argmax(all_score))
        return AlignmentResult(iopt, all_shifts[iopt], all_quat[iopt], all_score[iopt])

    def _get_alignment_function(self):
        if self.is_multi_templates:
            return self._optimize_multiple
        else:
            return self._optimize_single

    @property
    def is_multi_templates(self) -> bool:
        """
        Whether alignment parameters requires multi-templates.
        "Multi-template" includes alignment with subvolume rotation.
        """
        return self._n_templates > 1


class SupportRotation(BaseAlignmentModel):
    def __init__(
        self,
        template: np.ndarray | Sequence[np.ndarray],
        mask: np.ndarray | None = None,
        rotations: Ranges | None = None,
    ):
        self.quaternions = normalize_rotations(rotations)
        self._n_rotations = self.quaternions.shape[0]
        super().__init__(template=template, mask=mask)

    def align(
        self,
        img: np.ndarray,
        max_shifts: tuple[float, float, float],
    ) -> AlignmentResult:
        """
        Align an image using current alignment parameters.

        Parameters
        ----------
        img : np.ndarray
            Subvolume to be aligned
        max_shifts : tuple[float, float, float]
            Maximum shifts along z, y, x axis in pixel.

        Returns
        -------
        AlignmentResult
            Result of alignment.
        """
        iopt, shift, _, corr = super().align(img, max_shifts)
        quat = self.quaternions[iopt % self._n_rotations]
        return AlignmentResult(label=iopt, shift=shift, quat=quat, corr=corr)

    def _get_template_input(self) -> np.ndarray:
        """
        Returns proper template image for alignment.

        Template dimensionality will be dispatched according to the input
        parameters. Returned template should be used in line of the
        :func:`get_alignment_function`.

        Returns
        -------
        np.ndarray
            Template image(s). Its axes varies depending on the input.

            - no rotation, single template image ... 3D
            - has rotation, single template image ... 4D
            - no rotation, many template images ... 4D
            - has rotation, many template images ... 4D and when iterated over
              the first axis yielded images will be (rot0, temp0),
              (rot0, temp1), ...
        """
        if self._n_rotations > 1:
            rotators = [Rotation.from_quat(r).inv() for r in self.quaternions]
            matrices = compose_matrices(
                np.array(self._template.shape[-3:]) / 2 - 0.5, rotators
            )
            cval = np.percentile(self._template, 1)
            if self.is_multi_templates:
                all_templates: list[np.ndarray] = []
                for mat in matrices:
                    for tmp in self._template:
                        tmp: np.ndarray
                        cval = np.percentile(tmp, 1)
                        all_templates.append(
                            self.pre_transform(
                                ndi.affine_transform(tmp, mat, cval=cval)
                            )
                        )
                template_input: np.ndarray = np.stack(all_templates, axis=0)

            else:
                template_masked = self._template * self.mask
                template_input: np.ndarray = np.stack(
                    [
                        self.pre_transform(
                            ndi.affine_transform(template_masked, mat, cval=cval)
                        )
                        for mat in matrices
                    ],
                    axis=0,
                )
        else:
            if self.is_multi_templates:
                template_input: np.ndarray = np.stack(
                    [self.pre_transform(tmp * self.mask) for tmp in self._template],
                    axis=0,
                )
            else:
                template_input = self.pre_transform(self._template)

        return template_input

    def _get_alignment_function(self):
        if self.is_multi_templates or self._n_rotations > 1:
            return self._optimize_multiple
        else:
            return self._optimize_single


class FrequencyCutoffInput(SupportRotation):
    """
    An alignment model that supports frequency-based pre-filtering.

    This class can be used for implementing such as low-pass filter or high-
    pass filter before alignment.
    """

    def __init__(
        self,
        template: np.ndarray | Sequence[np.ndarray],
        mask: np.ndarray | None = None,
        rotations: Ranges | None = None,
        cutoff: float | None = None,
    ):
        self._cutoff = cutoff or 1.0
        super().__init__(template, mask, rotations)


class FourierLowpassInput(FrequencyCutoffInput):
    """Abstract model that uses low-pass-filtrated Fourier images as inputs."""

    def pre_transform(self, img: np.ndarray) -> np.ndarray:
        """Apply low-pass filter without IFFT."""
        return lowpass_filter_ft(img, cutoff=self._cutoff)


class RealLowpassInput(FrequencyCutoffInput):
    """Abstract model that uses low-pass-filtrated real images as inputs."""

    def pre_transform(self, img: np.ndarray) -> np.ndarray:
        """Apply low-pass filter."""
        return lowpass_filter(img, cutoff=self._cutoff)
