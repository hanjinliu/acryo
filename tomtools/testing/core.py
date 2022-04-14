from __future__ import annotations
import numpy as np
import impy as ip
from ..molecules import Molecules
from .._utils import compose_matrices
from ..alignment._utils import normalize_rotations

class TomogramGenerator:
    def __init__(
        self,
        template: ip.ImgArray,
        grid_shape=(10, 10),
        rotations=None,
        noise_sigma: float = 1,
        seed: int = 0,
    ) -> None:
        self._template = template
        self._grid_shape = grid_shape
        self._quaternions = normalize_rotations(rotations)
        self._noise_sigma = noise_sigma
        self._seed = seed
    
    @property
    def template(self):
        return self._template
    
    @property
    def grid_shape(self):
        return self._grid_shape
    
    @property
    def quaternions(self):
        return self._quaternions
    
    @property
    def noise_sigma(self):
        return self._noise_sigma
    
    @property
    def scale(self):
        return self.template.scale.x
    
    def get_matrices(self):
        gy, gx = self.grid_shape
        quat_idx = np.random.choice(self.quaternions.shape[0], size=(gy*gx)) 
        
        from scipy.spatial.transform import Rotation
        rotators = [Rotation.from_quat(self.quaternions[idx]) for idx in quat_idx]
        return compose_matrices(self.template.shape, rotators)

    def get_tomogram(self, tilt_range=(-90, 90)):
        np.random.seed(self._seed)
        template = self.template
        if tilt_range != (-90, 90):
            raise NotImplementedError()
        gy, gx = self.grid_shape
        mols = template
        mols: list[list[ip.ImgArray]] = [
            [template.copy() for _ in range(gy)] for _ in range(gx)
        ]
        if self.quaternions.shape[0] > 0:
            matrices = self.get_matrices()
            mtx_iterator = iter(matrices)
            for i in range(gy):
                for j in range(gx):
                    mtx = next(mtx_iterator)
                    mols[i][j].affine(mtx, update=True)
        for i in range(gy):
            for j in range(gx):
                mols[i][j] += np.random.normal(scale=self.noise_sigma, size=template.shape) 
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