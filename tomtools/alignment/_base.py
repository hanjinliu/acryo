from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Iterable, NamedTuple, Sequence
import numpy as np
from scipy.spatial.transform import Rotation
import impy as ip

from ._utils import normalize_rotations
from ._types import Ranges
from .._utils import compose_matrices


class AlignmentResult(NamedTuple):
    """The optimal alignment result."""

    label: int | tuple[int, int]
    shift: np.ndarray
    quat: np.ndarray
    corr: float


class BaseAlignmentModel(ABC):
    def __init__(
        self,
        template: ip.ImgArray | Sequence[ip.ImgArray],
        mask: ip.ImgArray | None = None,
    ):
        if isinstance(template, ip.ImgArray):
            if template.ndim == 3:
                self._template = template
                self._n_templates = 1
            elif template.ndim == 4:
                self._template = template
                self._n_templates = template.shape[0]
            else:
                raise TypeError("ndim must be 3 or 4.")
        else:
            self._template: ip.ImgArray = np.stack(template, axis="p")
            self._n_templates = self._template.shape[0]

        if mask is None:
            self.mask = 1
        else:
            if template.sizesof("zyx") != mask.shape:
                raise ValueError(
                    "Shape mismatch in zyx axes between tempalte image "
                    f"({tuple(template.shape)}) and mask image ({tuple(mask.shape)})."
                )
            self.mask = mask

        self._align_func = self._get_alignment_function()
        self.template_input = self._get_template_input()

    @abstractmethod
    def optimize(
        self,
        subvolume: ip.ImgArray,
        template: ip.ImgArray,
        max_shifts: tuple[float, float, float],
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """Optimize."""

    @abstractmethod
    def pre_transform(self, img: ip.ImgArray) -> ip.ImgArray:
        """Pre-transformation applied to input images (including template)."""

    def _get_template_input(self) -> ip.ImgArray:
        """
        Returns proper template image for alignment.

        Template dimensionality will be dispatched according to the input parameters.
        Returned template should be used in line of the :func:`get_alignment_function`.

        Returns
        -------
        ip.ImgArray
            Template image(s). Its axes varies depending on the input.

            - single template image ... "zyx"
            - many template images ... "pzyx"

        """
        if self.is_multi_templates:
            template_input = np.stack(
                [
                    self.pre_transform(tmp * self.mask)
                    for tmp in self._template
                ],
                axis="p",
            )
        else:
            template_input = self.pre_transform(template_input)
        return template_input

    def align(
        self,
        img: ip.ImgArray,
        max_shifts: tuple[float, float, float],
    ) -> AlignmentResult:
        """
        Align an image using current alignment parameters.

        Parameters
        ----------
        img : ip.ImgArray
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
        img: ip.ImgArray,
        max_shifts: tuple[float, float, float],
        cval: float | None = None,
    ) -> tuple[ip.ImgArray, AlignmentResult]:
        """
        Fit image to template based on the alignment model.

        Parameters
        ----------
        img : ip.ImgArray
            Input image that will be transformed.
        max_shifts : tuple[float, float, float]
            Maximum shifts along z, y, x axis in pixel.
        cval : float, optional
            Constant value for padding.

        Returns
        -------
        ip.ImgArray, AlignmentResult
            Transformed input image and the alignment result.
        """
        result = self.align(img, max_shifts=max_shifts)
        rotator = Rotation.from_quat(result.quat)
        matrix = compose_matrices(img.shape, [rotator])[0]
        if cval is None:
            cval = np.percentile(img, 1)
        img_trans = img.affine(translation=result.shift, cval=cval).affine(
            matrix=matrix, cval=cval
        )
        return img_trans, result

    def _optimize_single(
        self,
        subvol: ip.ImgArray,
        template: ip.ImgArray,
        max_shifts: tuple[float, float, float],
    ) -> AlignmentResult:
        out = self.optimize(subvol, template, max_shifts)
        return AlignmentResult(0, *out)

    def _optimize_multiple(
        self,
        subvol: ip.ImgArray,
        template_list: Iterable[ip.ImgArray],
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
        template: ip.ImgArray | Sequence[ip.ImgArray],
        mask: ip.ImgArray | None = None,
        rotations: Ranges | None = None,
    ):
        self.quaternions = normalize_rotations(rotations)
        self._n_rotations = self.quaternions.shape[0]
        super().__init__(template=template, mask=mask)

    def align(
        self,
        img: ip.ImgArray,
        max_shifts: tuple[float, float, float],
    ) -> AlignmentResult:
        """
        Align an image using current alignment parameters.

        Parameters
        ----------
        img : ip.ImgArray
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

    def _get_template_input(self) -> ip.ImgArray:
        """
        Returns proper template image for alignment.

        Template dimensionality will be dispatched according to the input parameters.
        Returned template should be used in line of the :func:`get_alignment_function`.

        Returns
        -------
        ip.ImgArray
            Template image(s). Its axes varies depending on the input.

            - no rotation, single template image ... "zyx"
            - has rotation, single template image ... "pzyx"
            - no rotation, many template images ... "pzyx"
            - has rotation, many template images ... "pzyx" and when iterated over the
              first axis yielded images will be (rot0, temp0), (rot0, temp1), ...
        """
        if self._n_rotations > 1:
            rotators = [Rotation.from_quat(r).inv() for r in self.quaternions]
            matrices = compose_matrices(self._template.sizesof("zyx"), rotators)
            cval = np.percentile(self._template, 1)
            if self.is_multi_templates:
                all_templates: list[ip.ImgArray] = []
                for mat in matrices:
                    for tmp in self._template:
                        tmp: ip.ImgArray
                        cval = np.percentile(tmp, 1)
                        all_templates.append(
                            self.pre_transform(tmp.affine(mat, cval=cval))
                        )
                template_input: ip.ImgArray = np.stack(all_templates, axis="p")

            else:
                template_masked = self._template * self.mask
                template_input: ip.ImgArray = np.stack(
                    [
                        self.pre_transform(template_masked.affine(mat, cval=cval))
                        for mat in matrices
                    ],
                    axis="p",
                )
        else:
            if self.is_multi_templates:
                template_input: ip.ImgArray = np.stack(
                    [
                        self.pre_transform(tmp * self.mask)
                        for tmp in self._template
                    ],
                    axis="p",
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
    An alignment model that supports frequency-based pre-filtering
    
    This class can be used for implementing such as low-pass filter or high-pass
    filter before alignment.
    """
    def __init__(
        self,
        template: ip.ImgArray | Sequence[ip.ImgArray],
        mask: ip.ImgArray | None = None,
        rotations: Ranges | None = None,
        cutoff: float | None = None,
    ):
        self._cutoff = cutoff or 1.0
        super().__init__(template, mask, rotations)
        
class FourierLowpassInput(FrequencyCutoffInput):
    def pre_transform(self, img: ip.ImgArray) -> ip.ImgArray:
        """Apply low-pass filter and FFT."""
        return img.lowpass_filter(cutoff=self._cutoff).fft()  # TODO: do not fft twice


class RealLowpassInput(FrequencyCutoffInput):
    def pre_transform(self, img: ip.ImgArray) -> ip.ImgArray:
        """Apply low-pass filter."""
        return img.lowpass_filter(cutoff=self._cutoff)

