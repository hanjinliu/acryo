===============
Alignment Model
===============

An alignment model defines the protocols for pre-transformation and alignment scoring.

.. mermaid::

    graph TD
        vol(sub-volume\nat <i>i</i>-th molecule)
        tmp[[template image]]
        vol_t(transformed\nsub volume)
        tmp_t[[transformed\ntemplate]]
        aln{alignment}
        result[alignment results]

        vol--masking &<br>pre-transformation-->vol_t-->aln
        tmp--masking &<br>pre-transformation-->tmp_t-->aln
        aln-->result

.. mermaid::

    graph TD

        subgraph Subvolumes
            vol(sub-volume\nat <i>i</i>-th molecule)
            vol_t(transformed\nsub volume)
            vol--masking &<br>pre-transformation-->vol_t
        end

        subgraph Templates
            tmp0[[template image A]]
            tmp1[[template image B]]
            tmp2[[template image C]]
            tmp0_t[[transformed\ntemplate A]]
            tmp1_t[[transformed\ntemplate B]]
            tmp2_t[[transformed\ntemplate C]]
            tmp0--masking &<br>pre-transformation-->tmp0_t
            tmp1--masking &<br>pre-transformation-->tmp1_t
            tmp2--masking &<br>pre-transformation-->tmp2_t
        end

        subgraph Alignment
            aln0{alignment}
            aln1{alignment}
            aln2{alignment}
        end
        result[best alignment results]

        vol_t-->aln0
        vol_t-->aln1
        vol_t-->aln2

        tmp0_t-->aln0
        tmp1_t-->aln1
        tmp2_t-->aln2

        aln0-->result
        aln1-->result
        aln2-->result

.. mermaid::

    graph TD

        subgraph Subvolumes
            vol(sub-volume\nat <i>i</i>-th molecule)
            vol_t(transformed\nsub volume)
            vol--masking &<br>pre-transformation-->vol_t
        end

        subgraph Templates
            tmp[[template image]]
            tmp0[[template image A]]
            tmp1[[template image B]]
            tmp2[[template image C]]
            rot{image rotation}
            tmp-->rot
            rot-->tmp0
            rot-->tmp1
            rot-->tmp2
            tmp0_t[[transformed\ntemplate A]]
            tmp1_t[[transformed\ntemplate B]]
            tmp2_t[[transformed\ntemplate C]]
            tmp0--masking &<br>pre-transformation-->tmp0_t
            tmp1--masking &<br>pre-transformation-->tmp1_t
            tmp2--masking &<br>pre-transformation-->tmp2_t
        end

        subgraph Alignment
            aln0{alignment}
            aln1{alignment}
            aln2{alignment}
        end
        result[best alignment results]

        vol_t-->aln0
        vol_t-->aln1
        vol_t-->aln2

        tmp0_t-->aln0
        tmp1_t-->aln1
        tmp2_t-->aln2

        aln0-->result
        aln1-->result
        aln2-->result

Ready-to-use Models
===================

There are two alignment models now.

1. :class:`ZNCCAlignment`
    Model that align subvolumes using ZNCC (Zero-mean Normalized Cross Correlation) score.
2. :class:`PCCAlignment`
    Model that align subvolumes using PCC (Phase Cross Correlation) score.

Both models are implemented with low-pass filtering, template rotation and missing
wedge masking so that they can easily be used in analysis of tomographic images.

.. code-block:: python

    from acryo.alignment import ZNCCAlignment

    model = ZNCCAlignment(
        template,  # template image
        mask,      # mask image
        rotations=[(10, 5), (4, 2), (10, 5)],
        cutoff=0.5,
        tilt_range=(-60, 60),
    )

- Shape of ``mask`` must be the same as ``template``. ``template * mask`` and
  ``subvolume * mask`` will be used for alignment.

Define Custom Alignment Model
=============================

:class:`acryo.alignment.BaseAlignmentModel` is the abstract base class that provides the
minimum interface.

.. code-block:: python

    def optimize(self, subvolume, reference, max_shifts, quaternion):
        ...

    def pre_transform(self, image):
        ...
