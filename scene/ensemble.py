
import numpy as np

from scene.model import Model
from optimal_transport.__main__ import otot, ot_with_reference

class Ensemble:

    conf = {
            "octree_node_size": 1000,
            "normalize_data": True,
            "autograd": True
        }

    def __init__(self, filelist):
        
        self.filelist = filelist
        self.models = []
        self.correspondences = []
        self.matching_colors = []

        self.idx = 0

        # legacy
        self.num_points = []
        #

    def increment(self):
        """
        Used to increment the index of the currently displayed model.
        """
        if self.idx < len(self.filelist['files']) - 1:
            self.idx += 1

    def get_num_points(self):
        """
        The number of points for each model is needed to exeute the compute shader.
        """
        return [model.num_points for model in self.models]

    def build(self):
        """
        Create a model for each file in the filelist.
        """
        # mean is used to normalize all point clouds to the same origin and scale
        mean = None
        for file in self.filelist['files']:
            model = Model(file, self.conf)
            mean = model.build(mean)
            self.models.append(model)

            self.num_points.append(model.num_points)

    def ot_reference(self, conf):
        """
        Calls OT with the first file as reference.
        """
        #largest = 0
        smallest = 1000000000
        idx = -1
        for i, m in enumerate(self.models):
            #if m.num_points > largest:
            if m.num_points < smallest:
                smallest = m.num_points
                idx = i

        octrees = [model.octree for i, model in enumerate(self.models) if i != idx]
        self.correspondences, self.matching_colors = ot_with_reference(self.models[idx].octree, octrees, conf)
        # reorder models
        self.models = [self.models[idx]] + [model for i, model in enumerate(self.models) if i != idx]

    def get_compute_data(self):
        """
        Get the compute data as needed for the compute shader.
        """
        # current points

        oct = self.models[0].octree

        positions = oct.revoke_normalization(oct.points).detach().cpu().numpy()
        colors = oct.colors

        positions[:,[1, 2]] = positions[:,[2, 1]]
        positions = np.c_[positions, np.ones(positions.shape[0])]

        compute_data = np.empty((positions.shape[0] + colors.shape[0], 4), dtype="f4")
        compute_data[0::2,:] = positions
        compute_data[1::2,:] = colors
        
        # next points

        assignment_positions = self.correspondences[self.idx]
        # swap columns due to blender
        assignment_positions[:,[1, 2]] = assignment_positions[:,[2, 1]]
        # todo 
        #positions = octrees[i].points
        #positions[:,[1, 2]] = positions[:,[2, 1]]
        assignment_distances = np.linalg.norm(assignment_positions - positions[:, :3], axis=1)
        # todo
        max_distance = np.max(assignment_distances)
        assignment_distances = assignment_distances / max_distance 

        assignment = np.empty((len(assignment_positions) * 2, 4), dtype="f4")
        assignment[0::2,:] = np.c_[assignment_positions, assignment_distances]
        assignment[1::2,:] = self.matching_colors[self.idx]
        #assignment[1::2,:] = colors
        assignment = assignment.astype("f4")

        return compute_data, assignment

    def ot_sequential(self):
        """
        Legacy Optimal Transport.
        Calls OT in the sequence of the files given.
        """
        octrees = [model.octree for model in self.models]
        self.correspondences, self.matching_colors = otot(octrees)

    def get_legacy_compute_data(self):
        """
        Get the compute data as needed for the compute shader.
        """
        # current points

        oct = self.models[self.idx + 1].octree

        positions = oct.revoke_normalization(oct.points).detach().cpu().numpy()
        colors = oct.colors

        self.num_points.append(positions.shape[0])

        positions[:,[1, 2]] = positions[:,[2, 1]]
        positions = np.c_[positions, np.ones(positions.shape[0])]

        compute_data = np.empty((positions.shape[0] + colors.shape[0], 4), dtype="f4")
        compute_data[0::2,:] = positions
        compute_data[1::2,:] = colors
        
        # next points

        assignment_positions = self.correspondences[self.idx]
        # swap columns due to blender
        assignment_positions[:,[1, 2]] = assignment_positions[:,[2, 1]]
        # todo 
        #positions = octrees[i].points
        #positions[:,[1, 2]] = positions[:,[2, 1]]
        assignment_distances = np.linalg.norm(assignment_positions - positions[:, :3], axis=1)
        # todo
        max_distance = np.max(assignment_distances)
        assignment_distances = assignment_distances / max_distance 

        assignment = np.empty((len(assignment_positions) * 2, 4), dtype="f4")
        assignment[0::2,:] = np.c_[assignment_positions, assignment_distances]
        assignment[1::2,:] = self.matching_colors[self.idx]
        #assignment[1::2,:] = colors
        assignment = assignment.astype("f4")

        return compute_data, assignment

    @property
    def compute_data(self):
        return self.get_compute_data()
    