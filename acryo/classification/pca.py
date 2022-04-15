from __future__ import annotations
from typing import TYPE_CHECKING, Iterable
import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans


class PcaClassifier:
    """A PCA (Principal component analysis) and k-means based image classifier."""

    def __init__(
        self,
        image_stack: np.ndarray,
        mask_image: np.ndarray | None = None,
        n_components: int = 2,
        n_clusters: int = 2,
        seed: int | None = 0,
    ):
        from sklearn.decomposition import PCA
        from sklearn.cluster import KMeans

        if mask_image is None:
            self._mask = 1
        else:
            self._mask = mask_image
        self._image = image_stack
        self._n_image = image_stack.shape[0]
        self._shape = image_stack.shape[1:]  # shape of a single image
        self.n_components = n_components
        self.n_clusters = n_clusters

        self._pca = PCA(n_components=n_components)
        self._kmeans = KMeans(n_clusters=n_clusters, random_state=seed)

    @property
    def pca(self) -> PCA:
        return self._pca

    @property
    def kmeans(self) -> KMeans:
        return self._kmeans

    def run(self):
        self._pca.fit(self._image_flat(mask=True))
        self._labels = self._kmeans.fit_predict(self.get_transform())
        return self

    def transform(self, input: np.ndarray, mask: bool = True) -> np.ndarray:
        if mask:
            input = input * mask
        flat = input.reshape(input.shape[0], -1)
        return self._pca.transform(flat)

    def predict(self, input) -> np.ndarray:
        transformed = self.transform(input)
        labels = self._kmeans.predict(transformed)
        return labels

    def _image_flat(self, mask: bool = False) -> np.ndarray:
        if mask:
            _input = self._image * self._mask
        else:
            _input = self._image
        return _input.reshape(self._n_image, -1)

    def get_transform(self, labels: Iterable[int] | None = None) -> np.ndarray:
        """
        Get the transformed vectors from the input images.

        Returns
        -------
        np.ndarray
            Transormed vectors. If input image stack P images, then
            (P, n_components) array will be returned.
        """
        if labels is None:
            flat = self._image_flat(mask=True)
        else:
            if not isinstance(labels, list):
                labels = list(labels)
            flat = self._image_flat(mask=True)[labels]
        transformed = self._pca.transform(flat)
        return transformed

    def plot_transform(
        self,
        labels: Iterable[int] | None = None,
        bases: tuple[int, int] = (0, 1),
        ax=None,
    ) -> Axes:
        ax0, ax1 = bases
        import matplotlib.pyplot as plt

        transformed = self.get_transform(labels)
        if ax is None:
            ax = plt.gca()
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

    def split_clusters(self) -> list[np.ndarray]:
        """
        Split input image stack into list of image stacks according to the labels.

        This method must be called after k-means clustering is conducted, otherwise
        only one cluster will be returned. If input image stack has ``"pzyx"`` axes,
        list of ``"pzyx"`` images will be returned.
        """
        output: list[np.ndarray] = []
        for i in range(self.n_clusters):
            img0 = self._image[self._labels == i]
            output.append(img0)
        return output
