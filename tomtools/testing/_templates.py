from __future__ import annotations
import numpy as np


def gaussian_blob(indices, center: tuple[float, float, float], sigma: float):
    zz, yy, xx = indices
    return np.exp(
        -((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2)
        / sigma ** 2
        / 2
    )


def blobs() -> np.ndarray:
    shape = (40, 40, 40)
    centers = [(17, 17, 17), (17, 17, 26), (24, 17, 18)]
    sigmas = [5.4, 3.6, 2.7]
    img = np.zeros(shape, dtype=np.float32)
    inds = np.indices(shape, dtype=np.float32)
    for center, sigma in zip(centers, sigmas):
        img += gaussian_blob(inds, center, sigma)

    return img


def spiral(radius=4, freq=1) -> np.ndarray:
    shape = (40, 40, 40)
    img = np.zeros(shape, dtype=np.float32)
    inds = np.indices(shape, dtype=np.float32)
    ys = np.linspace(5, 35, 100)
    zs = radius * np.sin(ys * freq) + (shape[0] - 1) / 2
    xs = radius * np.cos(ys * freq) + (shape[2] - 1) / 2
    for center in np.stack([zs, ys, xs], axis=1):
        img += gaussian_blob(inds, center, 1.0)
    img /= img.max()

    return img