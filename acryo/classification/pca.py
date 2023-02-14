# pyright: reportPrivateImportUsage=false
from __future__ import annotations
from typing import TYPE_CHECKING, Iterable, SupportsIndex
import numpy as np
from dask import array as da

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from ._dask_pca import DaskPCA as PCA
    from sklearn.cluster import KMeans
    from typing_extensions import Self
    from numpy.typing import NDArray


class PcaClassifier:
    """A PCA (Principal component analysis) and k-means based image classifier."""

    def __init__(
        self,
        image_stack: NDArray[np.float32] | da.Array,
        mask_image: NDArray[np.float32] | None = None,
        n_components: int = 2,
        n_clusters: int = 2,
        seed: int | None = 0,
    ):
        from ._dask_pca import DaskPCA as PCA
        from sklearn.cluster import KMeans

        if mask_image is None:
            self._mask = 1
        else:
            self._mask = mask_image

        if isinstance(image_stack, np.ndarray):
            image_stack = da.from_array(image_stack)
        if not isinstance(image_stack, da.Array):
            raise TypeError("image_stack must be a numpy array or a dask array.")
        self._image = image_stack
        self._n_image = image_stack.shape[0]
        self._shape = image_stack.shape[1:]  # shape of a single image
        self.n_components = n_components
        self.n_clusters = n_clusters

        self._pca = PCA(n_components=n_components)
        self._kmeans = KMeans(n_clusters=n_clusters, random_state=seed)

    @property
    def pca(self) -> PCA:
        """The PCA object."""
        return self._pca

    @property
    def kmeans(self) -> KMeans:
        """The k-means object."""
        return self._kmeans

    @property
    def labels(self) -> NDArray[np.int32] | None:
        return self._labels

    def run(self) -> Self:
        """Run PCA and k-means clustering."""
        self._pca.fit(self._image_flat(mask=True))
        self._labels = self._kmeans.fit_predict(self.get_transform())
        return self

    def transform(self, input: da.Array, mask: bool = True) -> NDArray[np.float32]:
        """Transform the input image into the principal component space."""
        if mask:
            input = input * self._mask
        flat = input.reshape(input.shape[0], -1)
        return self._pca.transform(flat).compute()

    def predict(self, input: da.Array) -> NDArray[np.int32]:
        """Predict which labels the input images belong to."""
        transformed = self.transform(input)
        labels = self._kmeans.predict(transformed)
        return labels

    def _image_flat(self, mask: bool = False) -> da.Array:
        if mask:
            _input = self._image * self._mask
        else:
            _input = self._image
        return _input.reshape(self._n_image, -1)

    def get_transform(self, labels: Iterable[int] | None = None) -> NDArray[np.float32]:
        """
        Get the transformed vectors from the input images.

        Returns
        -------
        da.Array
            Transormed vectors. If input image stack P images, then
            (P, n_components) array will be returned.
        """
        if labels is None:
            flat = self._image_flat(mask=True)
        else:
            if not isinstance(labels, list):
                labels = list(labels)
                if not isinstance(labels[0], SupportsIndex):
                    raise TypeError("labels must be a list of integers.")
            flat = self._image_flat(mask=True)[labels]
        return self._pca.transform(flat).compute()

    def plot_transform(
        self,
        labels: Iterable[int] | None = None,
        bases: tuple[int, int] = (0, 1),
        ax: Axes | None = None,  # type: ignore
    ) -> Axes:
        ax0, ax1 = bases

        transformed = self.get_transform(labels)
        if ax is None:
            import matplotlib.pyplot as plt

            ax: Axes = plt.gca()
        ax.scatter(transformed[:, ax0], transformed[:, ax1])
        return ax

    def get_bases(self) -> np.ndarray:
        """
        Get base images (principal axes) as image stack.

        Returns
        -------
        np.ndarray
            Same axes as input image stack, while the axis "p" corresponds to
            the identifier of bases.
        """
        return self.pca.components_.reshape(self.n_components, *self._shape)  # type: ignore

    def split_clusters(self) -> list[da.Array]:
        """
        Split input image stack into list of image stacks according to the labels.

        This method must be called after k-means clustering is conducted, otherwise
        only one cluster will be returned. If input image stack has ``"pzyx"`` axes,
        list of ``"pzyx"`` images will be returned.
        """
        output: list[da.Array] = []
        for i in range(self.n_clusters):
            img0 = self._image[self._labels == i]
            output.append(img0)
        return output
