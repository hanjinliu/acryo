==================
Subtomogram Loader
==================

:mod:`acryo` aims at efficient analysis of tomographic data. Here, we show how to use
:class:`Molecules` with actual images to perform subtomogram averaging, alignment and
many other tasks.

.. contents:: Contents
    :local:
    :depth: 1

Subtomogram Loader
==================

A :class:`SubtomogramLoader` is a pair of a 3D tomogram image and a
:class:`Molecules` object, with some additional parameters.

.. code-block:: python

    def __init__(self, image, molecules, order=3, scale=1.0, output_shape=Unset(), corner_safe=False): ...

1. ``image`` (`numpy.ndarray` or `dask.Array`) ... the tomogram image.
2. ``molecules`` (`Molecules`) ... molecules in the tomogram.
3. ``order`` (`int`) ... order of the spline interpolation. 0=nearest, 1=linear,
    3=cubic.
4. ``scale`` (`float`) ... scale (physical/pixel) of the tomogram image. This
    parameter must match the positions of ``molecules``.
5. ``output_shape`` (`tuple`) ... shape of the output subtomograms, which will be
    used to determine the subtomogram shape during subtomogram averaging.
6. ``corner_safe`` (`bool`) ... if true, the subtomogram loader will ensure that
    the volume inside the given output shape will not be affected after rotation,
    otherwise the corners of the subtomograms will be dimmer.

:class:`SubtomogramLoader` can be constructed from a image file using the :meth:`imread` method
or the public :func:`imread` function.

.. code-block:: python

    from acryo import Molecules, imread

    loader = imread("path/to/image.mrc", Molecules.from_csv("path/to/molecules.csv"))

Subtomogram averaging
=====================

:meth:`average` crops all the subtomograms around the molecules and
average them. This method always returns a 3D :class:`numpy.ndarray` object.

.. code-block:: python

    from dask import array as da
    from acryo import SubtomogramLoader, Molecules

    image = da.random.random((100, 100, 100))
    molecules = Molecules([[40, 40, 60], [60, 60, 40]])

    # give output shape beforehand
    loader = SubtomogramLoader(image, molecules, output_shape=(64, 64, 64))
    avg = loader.average()

    # or give output shape after construction
    loader = SubtomogramLoader(image, molecules)
    avg = loader.average(output_shape=(64, 64, 64))

Subtomogram Alignment
=====================

Templated alignment
-------------------

:meth:`align` crops all the subtomograms around the molecules and
align them to the given template image (reference image). This method will return
a new :class:`SubtomogramLoader` object with the updated :class:`Molecules` object.

You have to provide a template image, optionally a mask image, maximum shifts and
an alignment model. The default alignment model is :class:`ZNCCAlignment`.

.. code-block:: python

    from dask import array as da
    from acryo import SubtomogramLoader, Molecules

    image = da.random.random((100, 100, 100))
    template = np.random.random((20, 20, 20))
    molecules = Molecules([[40, 40, 60], [60, 60, 40]])

    loader = SubtomogramLoader(image, molecules)
    out = loader.align(template, max_shifts=(5, 5, 5))

If you want to give parameters to the alignment model, you can use the :meth:`with_params`
method of alignment model classes. It returns a factory function for the parametrized model.

.. code-block:: python

    loader = SubtomogramLoader(image, molecules)
    out = loader.align(
        template,
        max_shifts=(5, 5, 5),
        alignment_model=ZNCCAlignment.with_params(cutoff=0.5),
    )

Template-free alignment
-----------------------

If no a priori information is available for the template image, you'll use the subtomogram
averaging result as the template image. During this task, each subtomogram will be loaded
twice so it is not efficient to call :meth:`average` and :meth:`align` separately.

:meth:`align_no_template` creates a local cache of subtomograms so that alignment will be
faster.

.. code-block:: python

    loader = SubtomogramLoader(image, molecules)
    out = loader.align_no_template(max_shifts=(5, 5, 5), output_shape=(20, 20, 20))

Multi-template alignment
------------------------

If a tomogram is composed of heterogeneous molecules, you can use multiple templates to
align the molecules and determine the best template for each molecule.

.. code-block:: python

    loader = SubtomogramLoader(image, molecules)
    out = loader.align_multi_templates(
        [template0, template1, template2],
        max_shifts=(5, 5, 5),
        alignment_model=ZNCCAlignment.with_params(cutoff=0.5),
        label_name="template_id",
    )
    out.molecules.features["template_id"]  # get the best template id for each molecule

Here, input templates must be given as a list of :class:`numpy.ndarray` objects of the
same shape. ``label_name`` is the name used for the feature colummn of the best template.

Filtering Loader
================

:meth:`filter` is the method quite similar to that in :class:`Molecules` or :class:`DataFrame`.
It returns a new :class:`SubtomogramLoader` object with the filtered molecules.

.. code-block:: python

    loader = SubtomogramLoader(image, molecules)
    out = loader.filter(pl.col("score") > 0.5)
    assert (out.molecules.features["score"] > 0.5).all()

Grouping Loader
===============

:meth:`groupby` is a method that returns a :class:`LoaderGroup` object. An :class:`LoaderGroup`
object is very similar to those returned by :meth:`groupby` methods of :class:`polars.DataFrame`,
:class:`Molecules` or :class:`pandas.DataFrame`.

.. code-block:: python

    loader = SubtomogramLoader(image, molecules)
    for cluster, ldr in loader.groupby("cluster_id"):
        assert (out.molecules.features["cluster_id"] == cluster).all()

:class:`LoaderGroup` has many methods of the same name as those in :class:`SubtomogramLoader`.

Group wise averaging
--------------------

:class:`LoaderGroup` supports all the averaging methods.

- :meth:`average`
- :meth:`average_split`

In :class:`LoaderGroup` version, result is returned as a ``dict``
of group key and the averages.


Group wise alignment
--------------------

:class:`LoaderGroup` also supports all the alignment methods

- :meth:`align`
- :meth:`align_no_template`
- :meth:`align_multi_templates`

In :class:`LoaderGroup` version, result is returned as an updated :class:`LoaderGroup`.

If you want to collect aligned :class:`Molecules` objects, following codes are
essentially equivalent.

.. code-block:: python

    # call align() for each loader
    aligned = []
    for cluster, ldr in loader.groupby("cluster_id"):
        out = ldr.align(template)
        aligned.append(out.molecules)

    # call align() of the LoaderGroup object.
    aligned = []
    for cluster, ldr in loader.groupby("cluster_id").align(template):
        aligned.append(out.molecules)

Since each group does not necessarily composed of the same molecules, you can use a mapping
of templates for alignment functions.

.. code-block:: python

    templates = {
        0: template0,
        1: template1,
        2: template2,
    }
    aligned = loader.groupby("cluster_id").align_multi_templates(templates)

Loading from Collection of Tomograms
====================================

Cryo-ET image analysis is usually performed on a collection of tomograms. Data management
becomes very complicated in this case.

:mod:`acryo` provides a :class:`TomogramCollection` class for this purpose. :class:`TomogramCollection`
shares the same interface with :class:`SubtomogramLoader`. It is constructed using the same parameters.

.. code-block:: python

    def __init__(self, order=3, scale=1.0, output_shape=Unset(), corner_safe=False): ...

:class:`TomogramCollection` can be constructed from a list of :class:`SubtomogramLoader` objects.

.. code-block:: python

    from acryo import Molecules, imread, TomogramCollection

    collection = TomogramCollection.from_loaders(
        [
            imread("path/to/image-0.mrc", Molecules.from_csv("path/to/molecules-0.csv")),
            imread("path/to/image-1.mrc", Molecules.from_csv("path/to/molecules-1.csv")),
            imread("path/to/image-2.mrc", Molecules.from_csv("path/to/molecules-2.csv")),
        ],
    )

.. code-block:: python

    avg = collection.average(output_shape=(20, 20, 20))
    out = collection.align(template, max_shifts=(5, 5, 5))
    group = collection.groupby("cluster_id")
