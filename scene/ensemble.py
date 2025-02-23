
import numpy as np

from scene.model import Model, Uniform_reference_model
from optimal_transport.__main__ import otot, ot_with_reference

class Ensemble:

    def __init__(self, filelist, conf):
        
        self.filelist = filelist
        self.conf = conf
        self.models = []
        self.correspondences = []
        self.matching_colors = []
        self.emd_matrix = None

        self.idx = 0
        self.idx_lut = None

        # legacy
        self.num_points = []
        #

    def increment(self):
        """
        Used to increment the index of the currently displayed model.
        """
        if self.idx < len(self.filelist['files']) - 1:
            self.idx += 1
        return self.compute_data

    def decrement(self):
        """
        Decrease the index of the currently displayed model.
        """
        if self.idx > 0:
            self.idx -= 1
        return self.compute_data

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
        norm_params = None
        for file in self.filelist['files']:
            model = Model(file, self.conf)
            norm_params = model.build(norm_params)
            if self.conf["normalize_data"]:
                norm_params = None
            self.models.append(model)
            self.num_points.append(model.num_points)

        if self.conf['uniform_reference']:
            n = self.conf['reference_n']
            model = Uniform_reference_model(n, self.conf)
            model.build()
            self.models.insert(0, model)
            self.num_points.insert(0, n)
        
        self.idx_lut = list(range(len(self.models)))

    def swap(self, i, j):
        self.idx_lut[i], self.idx_lut[j] = self.idx_lut[j], self.idx_lut[i]
        if j == 0: # if sort based on emd is true resort if the reference model is swapped
            self.idx_lut = np.argsort(self.emd_matrix[np.argwhere(self.idx_lut == 0), :]).ravel()
        return self.idx_lut

    def ot_reference(self, conf):
        """
        Calls OT with the first file as reference.
        """
        idx = 0
        print(f"Reference model: {self.models[idx].file}")
        octrees = [model.octree for i, model in enumerate(self.models) if i != idx]
        self.correspondences, self.matching_colors = ot_with_reference(self.models[idx].octree, octrees, conf, sort=self.conf["sort_emd"])
        # reorder models
        self.models = [self.models[idx]] + [model for i, model in enumerate(self.models) if i != idx]

        self.post_ot_reference()
        self.idx_lut = np.argsort(self.emd_matrix[0, :])

    def post_ot_reference(self):
        """
        Post process the correspondences.
        Compute EMD matrix for the ensemble.
        """

        emd_matrix = np.zeros((len(self.models), len(self.models)))

        # first row, distance to reference model
        distances = np.r_[[np.mean(np.linalg.norm(self.models[0].octree.points_np - corres, axis=1)) for corres in self.correspondences]][:, None]
        emd_matrix[0, :] = np.c_[np.zeros(1), distances.T]
        emd_matrix[1:, 0] = distances.ravel()
        
        for i, corres in enumerate(self.correspondences):
            # TODO make efficient don'T compute all cells 
            distances = [np.mean(np.linalg.norm(corres - self.correspondences[ii], axis=1)) for ii in range(len(self.correspondences))] 
            emd_matrix[i + 1, 1:] = np.r_[distances]
        
        self.emd_matrix = emd_matrix

    def get_compute_data(self):
        """
        Get the compute data as needed for the compute shader.
        """

        def get_data_reference():
            oct = self.models[0].octree
            positions = oct.points_np
            colors = oct.colors[:, :3]
            positions = np.c_[positions, np.ones(positions.shape[0])]
            compute_data = np.empty((positions.shape[0] + colors.shape[0], 4), dtype="f4")
            compute_data[0::2,:] = positions
            compute_data[1::2,:] = np.c_[colors, np.zeros(colors.shape[0])]
            return compute_data
        
        def get_data_assignment(idx, positions):
            assignment_positions = self.correspondences[idx]

            assignment_distances = np.linalg.norm(assignment_positions - positions[:, :3], axis=1)
            max_distance = np.max(assignment_distances)
            assignment_distances = assignment_distances / max_distance if max_distance > 0 else np.zeros(positions.shape[0])
            colors = self.matching_colors[idx][:, :3]

            compute_data = np.empty((len(assignment_positions) * 2, 4), dtype="f4")
            # could potentially encode a scalar in position.w
            compute_data[0::2,:] = np.c_[assignment_positions, np.ones(positions.shape[0])]
            compute_data[1::2,:] = np.c_[colors, assignment_distances]
            compute_data = compute_data.astype("f4")
            return compute_data

        # always get the reference data        
        reference_data = get_data_reference()

        # respect user choosen sorting of models
        idx = self.idx_lut[self.idx]
        next_idx = self.idx_lut[self.idx + 1]
            
        # the reference model or the idx-1th correspondence is used as data
        source_data = reference_data if idx == 0 else get_data_assignment(idx - 1, reference_data[0::2,:])
        target_data = reference_data if next_idx == 0 else get_data_assignment(next_idx - 1, reference_data[0::2,:])

        return source_data, target_data
    
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
    