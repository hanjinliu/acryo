===========================
Piping Images to the Loader
===========================

Subtomogram alignment usually requires a template and mask images, with
appropriate pre-processing. In :mod:`acryo`, this kind of workflow is
provided as "Image Provider" and "Image Converter", and they can directly
passed to the ``template`` or ``mask`` arguments in alignment functions.

.. contents:: Contents
    :local:
    :depth: 1

Image Provider
==============

An image provider is an object that provide an image when called. Currently,
all the image providers are named starting with ``from_``.

Provide an image from a file
----------------------------

The most commonly useful image provider is the image reading pipeline
:func:`from_file`.

.. code-block:: python

    from acryo import SubtomogramLoader
    from acryo.pipe import from_file

    loader = SubtomogramLoader(image, molecules, scale=0.27)  # create a loader

    aligned = loader.align(
        template=from_file("path/to/template.mrc"),
    )

Image scale (nm/pixel) of the template image will be extracted from the image metadata.
If you want to provide it manually, you can pass the second argument.

.. code-block::

    aligned = loader.align(
        template=from_file("path/to/template.mrc", 0.18)
    )

.. note::

    The reason why we use a pipeline here is that the template image has to be
    rescaled differently for different images. Indeed, an image provider is a
    function that takes a float value as an input and returns an image.

    .. code-block:: python

        provider = from_file("path/to/template.mrc")
        type(provider(0.27))  # -> numpy.ndarray

Provid an image from an array
-----------------------------

If you already have an image array, you can use :func:`from_array` to create a
provider. The input array will be properly rescaled considering the ``scale``
argument.

.. code-block::

    from acryo.pipe import from_array
    arr = np.zeros((10, 10, 10))
    aligned = loader.align(
        template=from_array(arr, scale=0.18)
    )


Image Converter
===============

An image converter is an object that convert an image to another. This pipeline
is usually used for the ``mask`` argument in alignment functions. When an image
converter is passed, mask images will be generated from the template image
using the converter.

.. code-block:: python

    from acryo import SubtomogramLoader
    from acryo.pipe import from_file, soft_otsu

    loader = SubtomogramLoader(image, molecules, scale=0.27)  # create a loader

    aligned = loader.align(
        template=from_file("path/to/template.mrc"),
        mask=soft_otsu(sigma=2.0, radius=1.0),
    )

What is actually happening here is,

.. code-block:: python

    # created by user
    reader_function = from_file("path/to/template.mrc")
    soft_otsu_function = soft_otsu(sigma=2.0, radius=1.0)

    # images are generated inside the alignment method
    template = reader_function(0.27)
    mask = soft_otsu_function(template, 0.27)

Custom Pipelines
================

To define custom pipelines, you can use decorators :func:`provider_function` and
:func:`converter_function`.

.. code-block:: python

    from acryo.pipe import provider_function, converter_function

    # the first argument of a provider function must be a float
    @provider_function
    def my_provider_function(scale: float, arg0, arg1=0):
        # do something
        return image

    # the first and the second argument of a converter function must be
    # an array and a float respectively
    @converter_function
    def my_converter_function(image: np.ndarray, scale: float, arg0, arg1=0):
        # do something
        return image

In both cases, the first one or two arguments are to be provided inside a loader.
You can create a pipeline by calling these function without the first argument(s).

.. code-block:: python

    from acryo import SubtomogramLoader

    loader = SubtomogramLoader(image, molecules, scale=0.27)  # create a loader

    aligned = loader.align(
        template=my_provider_function(arg0, arg1=0),
        mask=my_converter_function(arg0, arg1=0),
    )

.. note::

    These decorators are similar to :func:`toolz.curry`.

Composing Pipelines
===================

Pipelines can be composed by ``@`` or :meth:`compose`.

.. code-block:: python

    from acryo.pipe import gaussian_filter, from_file, soft_otsu

    # `converter * provider` is a provider
    # Functions will be called in "reading image -> filtering" order
    composed = gaussian_filter(2.0) @ from_file("path/to/template.mrc")
    composed(0.27)  # -> numpy.ndarray

    # `converter * converter` is a converter
    # Functions will be called in "soft Otsu -> filtering" order
    composed = gaussian_filter(2.0) @ soft_otsu(sigma=2.0, radius=1.0)
    composed(np.zeros((4, 4, 4)))  # -> numpy.ndarray
