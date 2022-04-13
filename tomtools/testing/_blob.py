from __future__ import annotations
import numpy as np
import impy as ip
from ..molecules import Molecules
from .._utils import compose_matrices

def blob() -> ip.ImgArray:
    shape = (40, 40, 40)
    centers = [(17, 17, 17), (17, 17, 26), (24, 17, 18)]
    sigmas = [5.4, 3.6, 2.7]
    img = np.zeros(shape, dtype=np.float32)
    zz, yy, xx = np.indices(shape, dtype=np.float32)
    for center, sigma in zip(centers, sigmas):
        img += (
            np.exp(-(
                (zz - center[0]) ** 2 + 
                (yy - center[1]) ** 2 +
                (xx - center[2]) ** 2
                ) / sigma**2 / 2)
        )
    img = ip.asarray(img, axes="zyx", name="blob")
    img.set_scale(xyz=0.25)
    img.scale_unit = "nm"
    return img

class TestSet:
    def __init__(
        self,
        template: ip.ImgArray,
        grid_shape=(10, 10),
        max_rotation: float = 0,
        noise_sigma: float = 1,
        seed: int = 0,
    ) -> None:
        self._template = template
        self._grid_shape = grid_shape
        self._max_rotation = max_rotation
        self._noise_sigma = noise_sigma
        self._seed = seed
    
    @property
    def template(self):
        return self._template
    
    @property
    def grid_shape(self):
        return self._grid_shape
    
    @property
    def max_rotation(self):
        return self._max_rotation
    
    @property
    def noise_sigma(self):
        return self._noise_sigma
    
    @property
    def scale(self):
        return self.template.scale.x
    
    def get_matrices(self):
        gy, gx = self.grid_shape
        rotvec = (
            np.random.random(size=(gy*gx, 3)) 
            / np.sqrt(3) 
            * np.deg2rad(self.max_rotation)
        )
        from scipy.spatial.transform import Rotation
        rotator = Rotation.from_rotvec(rotvec).as_matrix()
        rotators = [rotator[i] for i in range(len(rotator))]
        return compose_matrices(self.template.shape, rotators)

    def get_tomogram(self):
        np.random.seed(self._seed)
        template = self.template
        gy, gx = self.grid_shape
        mols: list[list[ip.ImgArray]] = [
            [template + 
             np.random.normal(scale=self.noise_sigma, size=template.shape) 
             for _ in range(gy)]
            for _ in range(gx)
        ]
        if self.max_rotation > 0.:
            matrices = self.get_matrices()
            mtx_iterator = iter(matrices)
            for i in range(gy):
                for j in range(gx):
                    mtx = next(mtx_iterator)
                    mols[i][j].affine(mtx, update=True)
        
        tomogram: ip.ImgArray = np.block(mols)
        np.random.seed(None)
        tomogram.set_scale(template)
        return tomogram
    
    def sample_molecules(self, max_distance: float = 3.0):
        gy, gx = self.grid_shape
        shape_vec = np.array(self.template.shape)
        offset = (shape_vec - 1) / 2 * self.scale
        vy, vx = shape_vec[1:] * self.scale
        centers = []
        for i in range(gy):
            for j in range(gx):
                centers.append(offset + np.array([0., vy*i, vx*j]))
        centers = np.stack(centers, axis=0)
        return Molecules(centers).translate_random(
            max_distance=max_distance, seed=self._seed
        )