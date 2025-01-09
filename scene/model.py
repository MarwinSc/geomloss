
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

    def build(self):
        
        points, colors = file_import.read(self.file)

        # swap columns due to blender
        points[:,[1, 2]] = points[:,[2, 1]]

        self.num_points = points.shape[0]

        # Step 1: Compute the bounding box
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)

        # Step 2: Translate points to center at origin
        midpoint = (min_coords + max_coords) / 2
        points_centered = points - midpoint

        # Step 3: Normalize to range [0, 1]
        scale = max(max_coords - min_coords)
        points_normalized = points_centered / scale

        self.octree = Octree(points_normalized, self.conf["octree_node_size"], colors=colors, autograd=self.conf["autograd"])
    
