
import numpy as np

from scene import file_import
from scene.octree import Octree

class Model:

    def __init__(self, file, conf):
        
        self.file = file
        self.conf = conf
        self.num_points = None
        self.octree = None
        self.kdtree = None

    def build(self, norm_params=None):
        
        points, colors = file_import.read(self.file)

        # swap columns due to blender
        points[:,[1, 2]] = points[:,[2, 1]]

        self.num_points = points.shape[0]

        if norm_params is not None:
            min_coords, max_coords, midpoint = norm_params
        else: # if normalize_data is True:
            # Step 1: Compute the bounding box
            min_coords = points.min(axis=0)
            max_coords = points.max(axis=0)
            # Step 2: Translate points to center at origin
            midpoint = (min_coords + max_coords) / 2
        points = points - midpoint
        # Step 3: Normalize to range [0, 1]
        scale = max(max_coords - min_coords)
        points = points / scale

        self.octree = Octree(points, self.conf["octree_node_size"], colors=colors, autograd=self.conf["autograd"])
    
        return min_coords, max_coords, midpoint
    
class Evaluation_model:

    def __init__(self, file, conf):
        
        self.file = file
        self.conf = conf
        self.num_points = None
        self.octree = None
        self.kdtree = None

    def build(self, norm_params=None, num_samples=None):
        
        points, colors = file_import.read(self.file)

        # swap columns due to blender
        points[:,[1, 2]] = points[:,[2, 1]]

        # subsample points
        if num_samples is not None and num_samples < points.shape[0]:
            rng = np.random.default_rng(seed=42)
            indices = rng.choice(points.shape[0], num_samples, replace=False)
            points = points[indices]
            colors = colors[indices]

        self.num_points = points.shape[0]

        if norm_params is not None:
            min_coords, max_coords, midpoint = norm_params
        else:
            # Step 1: Compute the bounding box
            min_coords = points.min(axis=0)
            max_coords = points.max(axis=0)
            # Step 2: Translate points to center at origin
            midpoint = (min_coords + max_coords) / 2
        points = points - midpoint
        # Step 3: Normalize to range [0, 1]
        scale = max(max_coords - min_coords)
        points = points / scale

        self.octree = Octree(points, self.conf["octree_node_size"], colors=colors, autograd=self.conf["autograd"])
    
        return min_coords, max_coords, midpoint

class Uniform_reference_model:

    def __init__(self, n, conf):
        self.file = "uniform_reference"
        self.conf = conf
        self.num_points = n
        self.octree = None
        self.kdtree = None

    def build(self, norm_params=None):
        if True:
            vec = np.random.normal(size=(self.num_points, 3))
            vec /= np.linalg.norm(vec, axis=1, keepdims=True)  # Normalize to unit sphere
            radius = np.random.uniform(0.3, 0.5, size=(self.num_points, 1)) ** (1/3)
            #radius = 0.25
            points = vec * radius 
        else:
            points = np.random.uniform(low=-1.0, high=1.0, size=(self.num_points, 3))
        #colors = np.random.uniform(low=0, high=1, size=(self.num_points, 3))
        colors = np.full((self.num_points, 3), 0.5)
        colors = np.hstack((colors, np.ones((self.num_points, 1))))
        self.octree = Octree(points, self.conf["octree_node_size"], colors=colors, autograd=self.conf["autograd"])
        return None, None, None