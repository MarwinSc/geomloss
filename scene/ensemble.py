import pathlib
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
        self._selected_attribute = 0

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
        
        #self.idx_lut = np.arange(len(self.models))

    def ot_reference(self, conf):
        """
        Calls OT with the first file as reference.
        """
        idx = self.idx_lut[0]
        print(f"Reference model: {self.models[idx].file}")
        octrees = [model.octree for i, model in enumerate(self.models) if i != idx]
        self.correspondences, self.matching_colors, self.processing_times = ot_with_reference(self.models[idx].octree, octrees, conf, sort=self.conf["sort_emd"])
        # insert reference model to correspondences
        self.correspondences.insert(idx, self.models[idx].octree.points_np)
        self.matching_colors.insert(idx, self.models[idx].octree.colors)

        self.post_ot_reference()
        self.idx_lut = np.argsort(self.emd_matrix[self.frechet_mean, :])

    def post_ot_reference(self):
        """
        Post process the correspondences.
        Compute EMD matrix for the ensemble.
        Compute the variance of the points.
        And the total distance between points.
        """

        # emd matrix

        emd_matrix = np.zeros((len(self.models), len(self.models)))
        # save the index of the mean model 
        self.frechet_mean = 0
        min_dist = np.inf

        # iterate over correspondences
        for i, corres in enumerate(self.correspondences):
            # TODO make efficient don'T compute all cells 
            distances = [np.mean(np.linalg.norm(corres - self.correspondences[ii], axis=1)) for ii in range(len(self.correspondences))] 
            emd_matrix[i , :] = np.r_[distances]

            distance_sum = np.sum(distances)
            print(f"Distance sum for {pathlib.Path(self.models[i].file).stem}: {distance_sum}")
            if distance_sum < min_dist:
                min_dist = distance_sum
                self.frechet_mean = i
        
        print(f"Frechet mean: {self.models[self.frechet_mean].file} with distance {min_dist}")
        emd_mat_for_printing = np.array2string(emd_matrix, formatter={'float_kind': lambda x: "%.3f" % x})
        print(f"EMD matrix: \n {emd_mat_for_printing}")

        self.emd_matrix = emd_matrix

        # variance 
        self.variance = np.sum(np.var(np.dstack(self.correspondences), axis=(2)), axis=1)
        self.variance = self.variance / np.max(self.variance) if np.max(self.variance) > 0 else np.zeros(self.variance.shape[0])

        # total 
        self.cummultive = np.sum(np.c_[[np.linalg.norm(self.correspondences[i] - self.correspondences[i + 1], axis=1) for i in range(len(self.correspondences) - 1)]], axis=0)
        self.cummultive = self.cummultive / np.max(self.cummultive) if np.max(self.cummultive) > 0 else np.zeros(self.cummultive.shape[0])

    def get_compute_data(self):
        """
        Get the compute data as needed for the compute shader.
        """

        def get_data_assignment(idx, pos_other=None):
            '''
            Get the compute data for one model.
            Optionally use the position of another model to compute the assignment distances.
            '''
            positions = self.correspondences[idx]
            colors = self.matching_colors[idx][:, :3]
            compute_data = np.empty((len(positions) * 2, 4), dtype="f4")
            # could potentially encode a scalar in position.w
            compute_data[0::2,:] = np.c_[positions, np.ones(positions.shape[0])]
            # color.w is used for exen
            if pos_other is None: # reference (no previous model)
                compute_data[1::2,:] = np.c_[colors, np.zeros(colors.shape[0])]
            else:
                assignment_distances = np.linalg.norm(positions - pos_other, axis=1)
                max_distance = np.max(assignment_distances)
                assignment_distances = assignment_distances / max_distance if max_distance > 0 else np.zeros(assignment_distances.shape[0])
                compute_data[1::2,:] = np.c_[colors, assignment_distances]
            return compute_data

        # respect user choosen sorting of models
        idx = self.idx_lut[self.idx]
        next_idx = self.idx_lut[self.idx + 1]
            
        if self._selected_attribute == 0: # get total
            source_data = get_data_assignment(idx, None)
            target_data = get_data_assignment(next_idx, None)
            source_data[1::2, 3] = self.cummultive
            target_data[1::2, 3] = self.cummultive
        elif self._selected_attribute == 1: # get variance
            source_data = get_data_assignment(idx, None)
            target_data = get_data_assignment(next_idx, None)
            source_data[1::2, 3] = self.variance
            target_data[1::2, 3] = self.variance
        elif self._selected_attribute == 2: # get color reference
            reference_data = get_data_assignment(self.idx_lut[0], None)
            source_data = reference_data if idx == self.idx_lut[0] else get_data_assignment(idx, reference_data[0::2,:3])
            target_data = reference_data if next_idx == self.idx_lut[0] else get_data_assignment(next_idx, reference_data[0::2,:3])
        else: # get color pairwise
            source_data = get_data_assignment(idx, self.correspondences[next_idx])
            target_data = get_data_assignment(next_idx, self.correspondences[idx])

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
    
    @property
    def selected_attribute(self):
        return self._selected_attribute
    
    @selected_attribute.setter
    def selected_attribute(self, value):
        self._selected_attribute = value