
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

    def build(self, mean=None):
        
        points, colors = file_import.read(self.file)

        self.num_points = points.shape[0]

        # mean is used to normalize all point clouds to the same origin and scale
        if mean is None:
            mean = np.mean(points, axis=0)
        points -= mean
        normalized_points = points/np.max(np.abs(points))

        self.octree = Octree(normalized_points, self.conf["octree_node_size"], colors=colors, normalize=self.conf["normalize_data"], autograd=self.conf["autograd"])

        return mean 