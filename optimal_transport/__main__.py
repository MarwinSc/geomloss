import numpy as np
import torch
import os
import sys
import util.file_import as file_import
import optimal_transport.ot as ot
from util.timer import Timer
from scipy.spatial.kdtree import KDTree

import argparse
import logging
import pathlib
import json

log = logging.getLogger(__name__)

def get_data(filepath, mode):
    # todo remove alpha channel from color?
    if mode == 'position':
        points, colors = file_import.read(filepath)
        weights = np.ones(points.shape[0])
        weights = torch.tensor(weights / np.sum(weights), dtype=torch.float32, device='cuda')
        points = torch.tensor(points, dtype=torch.float32, device='cuda')
        return ot.normalize((weights, points), n=None)
    elif mode == 'combined':
        points, colors = file_import.read(filepath)
        points = np.c_[points, colors]
        weights = np.ones(points.shape[0])
        weights = torch.tensor(weights / np.sum(weights), dtype=torch.float32, device='cuda')
        points = torch.tensor(points, dtype=torch.float32, device='cuda')
        return ot.normalize((weights, points), n=None)
    elif mode == 'contrast':
        points, colors = file_import.read(filepath)
        contrast = np.mean(colors, axis=1)
        points = np.c_[points, contrast]
        weights = np.ones(points.shape[0])
        weights = torch.tensor(weights / np.sum(weights), dtype=torch.float32, device='cuda')
        points = torch.tensor(points, dtype=torch.float32, device='cuda')
        return ot.normalize((weights, points), n=None)


def run(filelist_json, output, tmp_directory, mode="position"):

    use_cuda = torch.cuda.is_available()
    numpy = lambda x: x.detach().cpu().numpy()

    with open(filelist_json, 'r') as infile:
        filelist_dict = json.load(infile)
        filelist = filelist_dict['files']
        assignment_list = []

        correspondences = None
        for i in range(1, len(filelist)):  
            logging.info(f"File: {filelist[i]}")
            logging.info("importing files")

            if correspondences is None:
                reference = get_data(filelist[i-1], mode)
            else:
                reference = (reference[0], torch.tensor(correspondences, dtype=torch.float32, device='cuda'))
            
            target = get_data(filelist[i], mode)

            matching = ot.OT_registration(reference, target)

            correspondences = numpy(matching)
            outpath = os.path.join(tmp_directory, output + f'_{i}.npy')

            with open(outpath, "wb") as outfile:
                np.save(outfile, correspondences)
                logging.info(f"wrote file to disk @ {os.path.abspath(outpath)}")

            assignment_list.append(os.path.abspath(outpath))

        filelist_dict['assignments'] = assignment_list


    with open(filelist_json, 'w') as outfile:
        json.dump(filelist_dict, outfile)


def main(raw_args=None):
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser()

    parser.add_argument('filelist_json')
    parser.add_argument('output')

    parser.add_argument('-m', '--mode', choices=['position', 'combined', 'contrast'], default='position')

    # could be that when using the CLI it has to be [1:]
    args = parser.parse_args(raw_args)

    logging.info(f"Mode: {args.mode}")

    tmp_directory = (pathlib.Path('.') / 'tmp').resolve()
    os.makedirs(tmp_directory, exist_ok=True)

    run(args.filelist_json, args.output, tmp_directory, args.mode)


def get_data_direct(points, mode, colors=None, dbg=False):
    # todo remove alpha channel from color?
    if mode == 'position':
        weights = np.ones(points.shape[0])
        if dbg:
            num = int(weights.shape[0] - (weights.shape[0]/8))
            random_indices = np.random.randint(0, weights.shape[0], size=num)
            weights[random_indices] = 0.0
        weights = torch.tensor(weights / np.sum(weights), dtype=torch.float32, device='cuda')
        points = torch.tensor(points, dtype=torch.float32, device='cuda')
        return weights, points
        #(weights, locations), mean, std = ot.normalize((weights, points), n=None)
        #return (weights, locations), mean, std
    
    elif mode == 'combined':
        points = np.c_[points, colors]
        weights = np.ones(points.shape[0])
        weights = torch.tensor(weights / np.sum(weights), dtype=torch.float32, device='cuda')
        points = torch.tensor(points, dtype=torch.float32, device='cuda')
        return ot.normalize((weights, points), n=None)
    elif mode == 'contrast':
        contrast = np.mean(colors, axis=1)
        points = np.c_[points, contrast]
        weights = np.ones(points.shape[0])
        weights = torch.tensor(weights / np.sum(weights), dtype=torch.float32, device='cuda')
        points = torch.tensor(points, dtype=torch.float32, device='cuda')
        return ot.normalize((weights, points), n=None)


def direct_run(octrees, level=1):
    use_cuda = torch.cuda.is_available()
    numpy = lambda x: x.detach().cpu().numpy()

    timer = Timer("Optimal Transport")

    correspondences_list = []
    correspondences = None
    for i in range(1, len(octrees)):  
        log.info(f"Octree: {i}")
        log.info(f"level: {level}")

        #matching = ot.OT_registration(reference, target)
        #centroids_s = numpy(reference[1])
        #centroids_t = numpy(target[1])

        # source

        hierarchy_s, bounds_s, metadata_s, points_s, colors_s = octrees[i - 1].to_list()
        #level_mask_s = metadata_s[:, 4] == level
        level_mask_s = metadata_s[:, 1] == 1
        
        tmp_count = np.count_nonzero(level_mask_s) 
        selection_s = bounds_s[level_mask_s]
        centroids_s = np.r_[[selection_s[:, 0] + ((selection_s[:, 3] - selection_s[:, 0]) / 2.0),
                                selection_s[:, 1] + ((selection_s[:, 4] - selection_s[:, 1]) / 2.0),
                                selection_s[:, 2] + ((selection_s[:, 5] - selection_s[:, 2]) / 2.0)]].T
        
        diagonal_s = np.linalg.norm(selection_s[:, 3:] - selection_s[:, :3], axis=1)

        # todo support mor than two again
        if correspondences is None:
            reference = get_data_direct(centroids_s, 'position', dbg=False)
            correspondences = np.zeros(points_s.shape)
        else:
            reference = (reference[0], torch.tensor(correspondences, dtype=torch.float32, device='cuda'))

        # target

        hierarchy_t, bounds_t, metadata_t, points_t, colors_t = octrees[i].to_list()

        #level_mask_t = metadata_t[:, 4] == level
        level_mask_t = metadata_t[:, 1] == 1
        
        tmp_count = np.count_nonzero(level_mask_t) 
        selection_t = bounds_t[level_mask_t]
        centroids_t = np.r_[[selection_t[:, 0] + ((selection_t[:, 3] - selection_t[:, 0]) / 2.0),
                                selection_t[:, 1] + ((selection_t[:, 4] - selection_t[:, 1]) / 2.0),
                                selection_t[:, 2] + ((selection_t[:, 5] - selection_t[:, 2]) / 2.0)]].T
        
        diagonal_t = np.linalg.norm(selection_t[:, 3:] - selection_t[:, :3], axis=1)

        target = get_data_direct(centroids_t, 'position', dbg=False)

        #matching = ot.OT_registration(reference, target, nits=1)
        matching = ot.OT_transport_plan(reference, target, diameter=max(diagonal_s, diagonal_t))
        
        centroids_s = numpy(reference[1])
        centroids_t = numpy(target[1])

        correspondence_outer = numpy(matching)


        #matching = numpy(matching)
        #matching_dbg = numpy(matching_dbg)
        
        # normalize so it matches the ot data
        #weights, locations = reference
        #weights, locations = weights.contiguous(), locations.contiguous()
        #mean = (weights.view(-1, 1) * locations).sum(dim=0)
        #locations -= mean
        #std = (weights.view(-1) * (locations ** 2).sum(dim=1).view(-1)).sum().sqrt()
        #locations /= std
        #centroids_s = numpy(locations)
#
        #weights, locations = target
        #weights, locations = weights.contiguous(), locations.contiguous()
        #mean = (weights.view(-1, 1) * locations).sum(dim=0)
        #locations -= mean
        #std = (weights.view(-1) * (locations ** 2).sum(dim=1).view(-1)).sum().sqrt()
        #locations /= std
        #centroids_t = numpy(locations)

        kd_tree_target = KDTree(centroids_t)
        for ii in range(len(correspondence_outer)):
                # todo add reasonable distance_upper_bound
            d, neighbour = kd_tree_target.query(correspondence_outer[ii], k=1)
                # check that the neighbour is legit

            point_range_from_s = metadata_s[np.flatnonzero(level_mask_s)[ii], 3]
            point_range_to_s   = metadata_s[np.flatnonzero(level_mask_s)[ii], 0] + point_range_from_s
            points_source = points_s[point_range_from_s:point_range_to_s, :]

            point_range_from = metadata_t[np.flatnonzero(level_mask_t)[neighbour], 3]
            point_range_to   = metadata_t[np.flatnonzero(level_mask_t)[neighbour], 0] + point_range_from
            points_target = points_t[point_range_from:point_range_to, :]
            #points_target = points_t

            reference = get_data_direct(points_source, 'position',dbg=False)
            target = get_data_direct(points_target, 'position', dbg=False)
            print(ii)

            #matching = ot.OT_registration(reference, target, nits=1)
            matching = ot.OT_transport_plan(reference, target)

            matching = numpy(matching)

            # centroids_S need to be normalized like the ot data 
            # todo add an offset because it is cool
            correspondences[point_range_from_s:point_range_to_s, :] = matching# + centroids_s[ii]
        
        correspondences_list.append(correspondences)

    timer.toc()
    
    return correspondences_list


def naive_direct_run(octrees, level=1):
    use_cuda = torch.cuda.is_available()
    numpy = lambda x: x.detach().cpu().numpy()

    timer = Timer("Optimal Transport")

    correspondences_list = []
    for i in range(1, len(octrees)):  
        log.info(f"Octree: {i}")
        log.info(f"level: {level}")

        # source

        hierarchy_s, bounds_s, metadata_s, points_s, colors_s = octrees[i - 1].to_list()
        # todo support mor than two again
        if len(correspondences_list) == 0:
            reference = get_data_direct(points_s, 'position', dbg=False)
        else:
            reference = (reference[0], torch.tensor(correspondences_list[-1], dtype=torch.float32, device='cuda'))

        # target

        hierarchy_t, bounds_t, metadata_t, points_t, colors_t = octrees[i].to_list()
        target = get_data_direct(points_t, 'position', dbg=False)

        matching = ot.OT_registration(reference, target)
        centroids_s = numpy(reference[1])
        centroids_t = numpy(target[1])

        correspondence_outer = numpy(matching)
        correspondences_list.append(correspondence_outer)

    timer.toc()
    
    return correspondences_list


def barycenter_run(octrees, level=1):

    use_cuda = torch.cuda.is_available()
    numpy = lambda x: x.detach().cpu().numpy()

    timer = Timer("Optimal Transport")

    tensors = []

    correspondences_list = []
    for i in range(1, len(octrees)):  
        log.info(f"Octree: {i}")
        log.info(f"level: {level}")

        # source

        hierarchy_s, bounds_s, metadata_s, points_s, colors_s = octrees[i - 1].to_list()
        # todo support mor than two again
        if len(correspondences_list) == 0:
            reference = get_data_direct(points_s, 'position', dbg=False)
        else:
            reference = (reference[0], torch.tensor(correspondences_list[-1], dtype=torch.float32, device='cuda'))

        # target

        hierarchy_t, bounds_t, metadata_t, points_t, colors_t = octrees[i].to_list()
        target = get_data_direct(points_t, 'position', dbg=False)

        # Stack tensors along a new dimension (e.g., dimension 0)
        measures = torch.stack([reference[0], target[0]], dim=0)   
        weights = torch.stack([reference[1] / 2], target[1] / 2, dim=0)

        matching = ot.barycenter(measures, weights)

        #matching = ot.OT_registration(reference, target)
        #centroids_s = numpy(reference[1])
        #centroids_t = numpy(target[1])

        correspondence_outer = numpy(matching)
        correspondences_list.append(correspondence_outer)

    timer.toc()
    
    return correspondences_list


def get_data_with_labels(points, labels):
    weights = np.ones(points.shape[0])
    weights = torch.tensor(weights / np.sum(weights), dtype=torch.float32, device='cuda')
    locations = torch.tensor(points, dtype=torch.float32, device='cuda')
    labels = torch.tensor(labels, dtype=torch.int32, device='cuda')
    #return weights, points
    #(weights, locations), mean, std = ot.normalize((weights, points), n=None)
    #return (weights, locations, labels), mean, std
    return (weights, locations, labels), None, None



def run_with_labels(octrees):
        
    def octree_to_data_coarse(octree):
        hierarchy, bounds, metadata, points, colors = octree.to_list()
        level_mask = metadata[:, 1] == 1

        tmp_count = np.count_nonzero(level_mask) 
        selection = bounds[level_mask]
        centroids = np.r_[[selection[:, 0] + ((selection[:, 3] - selection[:, 0]) / 2.0),
                                selection[:, 1] + ((selection[:, 4] - selection[:, 1]) / 2.0),
                                selection[:, 2] + ((selection[:, 5] - selection[:, 2]) / 2.0)]].T
        pointcount = metadata[level_mask, 0]
        return centroids, pointcount, metadata, points, colors
    
    use_cuda = torch.cuda.is_available()
    numpy = lambda x: x.detach().cpu().numpy()

    timer = Timer("Optimal Transport")

    correspondences_list = []
    colors_list = []
    correspondences = None
    colors_list = []
    colors = None
    for i in range(1, len(octrees)):  
        log.info(f"Octree: {i}")

        # source
        #centroids_s, pointcount_s, metadata_s, points_s, colors_s = octree_to_data_coarse(octrees[i])
        centroids_s, pointcount_s, metadata_s, points_s, colors_s = octree_to_data_coarse(octrees[i])
        level_mask_s = metadata_s[:, 1] == 1
        labels_s = np.zeros(len(points_s), dtype=np.int32)
        for j in range(len(metadata_s)):
            if metadata_s[j, 1] == 1:
                point_range_from_s = metadata_s[j, 3]
                point_range_to_s   = metadata_s[j, 0] + point_range_from_s
                labels_s[point_range_from_s:point_range_to_s] = j

        dbg, dbg_counts = np.unique(labels_s, return_counts=True)

        # target, previous point cloud
        #centroids_t, pointcount_t, metadata_t, points_t, colors_t = octree_to_data_coarse(octrees[i-1])
        centroids_t, pointcount_t, metadata_t, points_t, colors_t = octree_to_data_coarse(octrees[i-1])
        level_mask_t = metadata_t[:, 1] == 1
        labels_t = np.zeros(len(points_t), dtype=np.int32)

        for j in range(len(metadata_t)):
            if metadata_t[j, 1] == 1:
                point_range_from_t = metadata_t[j, 3]
                point_range_to_t   = metadata_t[j, 0] + point_range_from_t
                labels_t[point_range_from_t:point_range_to_t] = j

        dbg_, dbg_counts_ = np.unique(labels_t, return_counts=True)

        reference, mean, std = get_data_with_labels(points_s, labels_s)
        target, mean, std = get_data_with_labels(points_t, labels_t)

        colors_target = torch.tensor(colors_t, dtype=torch.float32, device='cuda')

        matching = ot.ot_labels(reference, target, colors_target)

        correspondence_outer = numpy(matching)
        #colors = numpy(colors)
        correspondences_list.append(correspondence_outer)
        #colors_list.append(colors)


    return correspondences_list#, colors_list


def get_data_coarse(centroids, pointcount, colors=None):
    weights = torch.tensor(pointcount, dtype=torch.float32, device='cuda')
    centroids = torch.tensor(centroids, dtype=torch.float32, device='cuda')
    return ot.normalize((weights, centroids), n=None)
    #return weights, centroids
    #return ot.normalize((weights, points), n=None)

def direct_run_(octrees, level=1):

    def octree_to_data_coarse(octree):
        hierarchy, bounds, metadata, points, colors = octree.to_list()
        level_mask = metadata[:, 1] == 1

        tmp_count = np.count_nonzero(level_mask) 
        selection = bounds[level_mask]
        centroids = np.r_[[selection[:, 0] + ((selection[:, 3] - selection[:, 0]) / 2.0),
                                selection[:, 1] + ((selection[:, 4] - selection[:, 1]) / 2.0),
                                selection[:, 2] + ((selection[:, 5] - selection[:, 2]) / 2.0)]].T
        pointcount = metadata[level_mask, 0]
        return centroids, pointcount, metadata, points, colors
    
    def octree_to_data_coarse_more_attributes(octree):
        hierarchy, bounds, metadata, points, colors = octree.to_list()
        level_mask = metadata[:, 1] == 1

        tmp_count = np.count_nonzero(level_mask) 
        selection = bounds[level_mask]
        centroids = np.r_[[selection[:, 0] + ((selection[:, 3] - selection[:, 0]) / 2.0),
                                selection[:, 1] + ((selection[:, 4] - selection[:, 1]) / 2.0),
                                selection[:, 2] + ((selection[:, 5] - selection[:, 2]) / 2.0)]].T
        
        mean_colors = np.zeros((len(centroids), 4))
        for meta in metadata[level_mask]:
            point_range_from = meta[3]
            point_range_to   = meta[0] + point_range_from
            colors_target = colors[point_range_from:point_range_to, :]
            mean_colors[i] = np.mean(colors_target, axis=0)

        centroids = np.c_[centroids, mean_colors]

        pointcount = metadata[level_mask, 0]
        return centroids, pointcount, metadata, points, colors

    use_cuda = torch.cuda.is_available()
    numpy = lambda x: x.detach().cpu().numpy()

    timer = Timer("Optimal Transport")

    correspondences_list = []
    correspondences = None
    colors_list = []
    colors = None
    for i in range(1, len(octrees)):  
        log.info(f"Octree: {i}")
        log.info(f"level: {level}")

        # source
        centroids_s, pointcount_s, metadata_s, points_s, colors_s = octree_to_data_coarse(octrees[i])
        #centroids_s, pointcount_s, metadata_s, points_s, colors_s = octree_to_data_coarse_more_attributes(octrees[i])
        level_mask_s = metadata_s[:, 1] == 1

        # target, previous point cloud
        centroids_t, pointcount_t, metadata_t, points_t, colors_t = octree_to_data_coarse(octrees[i-1])
        #centroids_t, pointcount_t, metadata_t, points_t, colors_t = octree_to_data_coarse_more_attributes(octrees[i-1])
        level_mask_t = metadata_t[:, 1] == 1

        # balance the assignment 
        if len(points_t) < len(points_s):
            print(f"need to remove {len(points_s) - len(points_t)} points")
            # iterate through metadata 
            # collect range and point count
            # with probability point count / sum point count each node receives points, n have to be distributed in total
        else:
            print(f"need to create {len(points_t) - len(points_s)} points")

        reference, mean, std = get_data_coarse(centroids_s, pointcount_s)
        target, mean, std = get_data_coarse(centroids_t, pointcount_t)
        
        correspondences = np.zeros(points_s.shape)
        colors = np.zeros(colors_s.shape)

        match_indices, match_values = ot.ot_coarse(reference, target)
        #match_indices, match_values = ot.ot_coarse_iterations(reference, target)

        target_indices_unique = np.unique(match_indices[:, 1])
        moved_sources = np.zeros(len(points_s), dtype=bool)

        dbg_received_mass = np.zeros(len(centroids_t))

        # iterate over all targets and collect the points from the sources
        for ii in target_indices_unique:
            rows = np.argwhere(match_indices[:, 1] == ii)
            sources = match_indices[rows, 0]
            weights = match_values[rows] # how much of a source maps to the corresponding target

            num_points_all_nodes = int(np.floor(np.sum(pointcount_s[sources] * weights)))
            points_source = np.zeros((num_points_all_nodes, 3))
            pointer = 0

            correspondences_indices = np.zeros((num_points_all_nodes), dtype=int)

            # go through the corresponding sources and pick the number of points 
            for iii, s in enumerate(sources):
                
                # todo ignores how much the actual target node has
                num_points = int(np.floor(weights[iii] * pointcount_s[s]))

                point_range_from_s = metadata_s[np.flatnonzero(level_mask_s)[s], 3]
                point_range_to_s   = metadata_s[np.flatnonzero(level_mask_s)[s], 0] + point_range_from_s
                point_range_from_s = point_range_from_s.item()
                point_range_to_s = point_range_to_s.item()

                mask_selected_sources = np.invert(moved_sources[point_range_from_s:point_range_to_s])

                # todo holes due to replace     
                # pick random points
                if True:
                    selected = np.random.choice(np.argwhere(mask_selected_sources).ravel(), num_points, replace=False)
                else:
                # pick closest points, early target nodes are prefered 
                    indices = np.flatnonzero(mask_selected_sources)
                    distance = np.linalg.norm(points_s[indices + point_range_from_s] - np.full((len(indices), 3), centroids_t[ii]), axis=1)
                    # is it lowest first
                    selected = indices[np.argsort(distance)][:num_points] 

                #moved_sources_before = np.count_nonzero(moved_sources)
                moved_sources[selected + point_range_from_s] = True
                #moved_sources_after = np.count_nonzero(moved_sources)
                #if moved_sources_before == moved_sources_after and num_points > 0:
                #    print("error")

                #dbg_available = np.count_nonzero(np.invert(moved_sources))

                points_source[pointer: pointer+num_points, :] = points_s[selected + point_range_from_s, :]
                correspondences_indices[pointer:pointer+num_points] = selected + point_range_from_s

                # todo we don't guarantee that all points of a source node move...ss
                # could do another round of ot assigning the points to the target centroids and weights
                #distance = np.linalg.norm(source_points - np.full((point_range_to_s - point_range_from_s, 3), centroids_t[ii]), axis=1)
                #selected = np.argsort(distance)[:num_points]

                #points_source[pointer: pointer+num_points, :] = points_s[point_range_from_s:point_range_to_s, :][selected]
                #correspondences_indices[pointer:pointer+num_points] = selected + point_range_from_s

                pointer += num_points

            # target points we can select straight away
            point_range_from = metadata_t[np.flatnonzero(level_mask_t)[ii], 3]
            point_range_to   = metadata_t[np.flatnonzero(level_mask_t)[ii], 0] + point_range_from
            points_target = points_t[point_range_from:point_range_to, :]

            colors_target = colors_t[point_range_from:point_range_to, :]
            colors_target = torch.tensor(colors_target, dtype=torch.float32, device='cuda')

            dbg_received_mass[ii] += num_points_all_nodes/len(points_target)

            # todo improve handeling
            if len(points_source) == 0 or len(points_target) == 0:
                continue

            reference, mean, std = get_data_direct(points_source, 'position',dbg=False)
            target, mean, std = get_data_direct(points_target, 'position', dbg=False)

            matching, matching_color = ot.OT_transport_plan(reference, target, colors_target)

            matching = numpy(matching)
            matching = matching * numpy(std) + numpy(mean)

            matching_color = numpy(matching_color)

            # centroids_S need to be normalized like the ot data 
            # todo add an offset because it is cool
            #correspondences[point_range_from_s:point_range_to_s, :] = matching# + centroids_s[ii]
            correspondences[correspondences_indices, :] = matching# + centroids_s[ii]
            colors[correspondences_indices, :] = matching_color

        print(f"received mass: {dbg_received_mass}")

        target_row = np.array([0.0, 0.0, 0.0])
        matches = np.all(correspondences == target_row, axis=1)
        dbg_empty = np.argwhere(matches)
        count = np.sum(matches)
        print(f"empty correspondences: {count}")

        print(f"empty based on moved sources: {np.count_nonzero(~moved_sources)}")

        correspondences_list.append(correspondences)
        colors_list.append(colors)

    timer.toc()
    
    return correspondences_list, colors_list


def direct_run_color(octrees, level=1):

    def octree_to_data_coarse(octree):
        hierarchy, bounds, metadata, points, colors = octree.to_list()
        level_mask = metadata[:, 1] == 1

        tmp_count = np.count_nonzero(level_mask) 
        selection = bounds[level_mask]
        centroids = np.r_[[selection[:, 0] + ((selection[:, 3] - selection[:, 0]) / 2.0),
                                selection[:, 1] + ((selection[:, 4] - selection[:, 1]) / 2.0),
                                selection[:, 2] + ((selection[:, 5] - selection[:, 2]) / 2.0)]].T
        pointcount = metadata[level_mask, 0]
        return centroids, pointcount, metadata, points, colors

    use_cuda = torch.cuda.is_available()
    numpy = lambda x: x.detach().cpu().numpy()

    timer = Timer("Optimal Transport")

    correspondences_list = []
    correspondences = None
    colors_list = []
    colors = None
    for i in range(1, len(octrees)):  
        log.info(f"Octree: {i}")
        log.info(f"level: {level}")

        # source
        centroids_s, pointcount_s, metadata_s, points_s, colors_s = octree_to_data_coarse(octrees[i - 1])
        level_mask_s = metadata_s[:, 1] == 1

        if correspondences is None:
            reference, mean, std = get_data_coarse(centroids_s, pointcount_s)
            #reference = get_data_direct(centroids_s, 'position', dbg=False)
            
            correspondences = np.zeros(points_s.shape)
            colors = np.zeros(colors_s.shape)
        else:
            reference = (reference[0], torch.tensor(correspondences, dtype=torch.float32, device='cuda'))

        # target

        centroids_t, pointcount_t, metadata_t, points_t, colors_t = octree_to_data_coarse(octrees[i])
        level_mask_t = metadata_t[:, 1] == 1

        target, mean, std = get_data_coarse(centroids_t, pointcount_t)
        #target = get_data_direct(centroids_t, 'position', dbg=False)

        match_indices, match_values = ot.ot_coarse_iterations(reference, target)
        indices_unique = np.unique(match_indices[:, 1])

        dbg_source_indices = np.unique(match_indices[:, 0])
        dbg_sum_assigned = 0

        moved_sources = np.zeros(len(points_s), dtype=bool)

        dbg_moved_mass = np.zeros(len(centroids_s))
        dbg_received_mass = np.zeros(len(centroids_t))

        # iterate over all targets and collect the points from the sources
        for ii in indices_unique:
            rows = np.argwhere(match_indices[:, 1] == ii)
            sources = match_indices[rows, 0]
            weights = match_values[rows] # how much of a source maps to the corresponding target

            num_points_all_nodes = int(np.floor(np.sum(pointcount_s[sources] * weights)))
            points_source = np.zeros((num_points_all_nodes, 3))
            pointer = 0

            correspondences_indices = np.zeros((num_points_all_nodes), dtype=int)

            # go through the corresponding sources and pick the number of points 
            for iii, s in enumerate(sources):

                num_points = int(np.floor(weights[iii] * pointcount_s[s]))

                dbg_moved_mass[s] += weights[iii]

                point_range_from_s = metadata_s[np.flatnonzero(level_mask_s)[s], 3]
                point_range_to_s   = metadata_s[np.flatnonzero(level_mask_s)[s], 0] + point_range_from_s
                point_range_from_s = point_range_from_s.item()
                point_range_to_s = point_range_to_s.item()

                #source_points = points_s[point_range_from_s:point_range_to_s, :]

                # cluster the points based on the mean point and the target centroids
                #corresponding_targets = np.argwhere(match_indices[:, 0] == ii)
                #corresponding_targets_centroids = centroids_t[corresponding_targets]
                #dot_products = np.zeros((len(source_points), len(corresponding_targets)))
                #for i in range(len(corresponding_targets)):
                #    dot_products[:, i] = np.abs(np.sum(from_mean_to_points * np.full((len(source_points), 3), corresponding_targets_centroids[i, :]), axis=1))

                mask_selected_sources = moved_sources[point_range_from_s:point_range_to_s]
                dbg_num_available = np.count_nonzero(~mask_selected_sources)
                selected = np.random.choice(np.argwhere(~mask_selected_sources).ravel(), num_points, replace=False)
                moved_sources[selected + point_range_from_s] = True

                points_source[pointer: pointer+num_points, :] = points_s[point_range_from_s:point_range_to_s, :][selected]
                correspondences_indices[pointer:pointer+num_points] = selected + point_range_from_s

                # todo we don't guarantee that all points of a source node move...
                # could do another round of ot assigning the points to the target centroids and weights
                #distance = np.linalg.norm(source_points - np.full((point_range_to_s - point_range_from_s, 3), centroids_t[ii]), axis=1)
                #selected = np.argsort(distance)[:num_points]

                #points_source[pointer: pointer+num_points, :] = points_s[point_range_from_s:point_range_to_s, :][selected]
                #correspondences_indices[pointer:pointer+num_points] = selected + point_range_from_s

                pointer += num_points

            dbg = np.count_nonzero(correspondences_indices == 0)
            dbg_sum_assigned += len(correspondences_indices)

            # target points we can select straight away
            point_range_from = metadata_t[np.flatnonzero(level_mask_t)[ii], 3]
            point_range_to   = metadata_t[np.flatnonzero(level_mask_t)[ii], 0] + point_range_from
            points_target = points_t[point_range_from:point_range_to, :]

            colors_target = colors_t[point_range_from:point_range_to, :]
            colors_target = torch.tensor(colors_target, dtype=torch.float32, device='cuda')

            dbg_received_mass[ii] += num_points_all_nodes/len(points_target)

            # todo improve handeling
            if len(points_source) == 0 or len(points_target) == 0:
                continue

            reference, mean, std = get_data_direct(points_source, 'position',dbg=False)
            target, mean, std = get_data_direct(points_target, 'position', dbg=False)

            matching, matching_color = ot.OT_transport_plan(reference, target, colors_target)

            matching = numpy(matching)
            matching = matching * numpy(std) + numpy(mean)

            matching_color = numpy(matching_color)

            # centroids_S need to be normalized like the ot data 
            # todo add an offset because it is cool
            #correspondences[point_range_from_s:point_range_to_s, :] = matching# + centroids_s[ii]
            correspondences[correspondences_indices, :] = matching# + centroids_s[ii]
            colors[correspondences_indices, :] = matching_color

        target_row = np.array([0.0, 0.0, 0.0])
        matches = np.all(correspondences == target_row, axis=1)
        dbg_empty = np.argwhere(matches)
        count = np.sum(matches)
        print(f"empty correspondences: {count}")

        print(f"empty based on moved sources: {np.count_nonzero(~moved_sources)}")

        correspondences_list.append(correspondences)
        colors_list.append(colors)

    timer.toc()
    
    return correspondences_list, colors_list


def otot(octrees):
    
    numpy = lambda x: x.detach().cpu().numpy()

    correspondences_list = []
    colors_list = []
    for i in range(1, len(octrees)): 
        correspondence, colors_matching = ot.ot_octree(octrees[i], octrees[i-1]) 
        correspondences_list.append(numpy(correspondence))
        colors_list.append(numpy(colors_matching))
    return correspondences_list, colors_list

    use_cuda = torch.cuda.is_available()
    numpy = lambda x: x.detach().cpu().numpy()

    timer = Timer("Optimal Transport")

    correspondences_list = []
    for i in range(1, len(octrees)):  
        log.info(f"Octree: {i}")

        # source

        hierarchy_s, bounds_s, metadata_s, points_s, colors_s = octrees[i - 1].to_list()
        # todo support mor than two again
        if len(correspondences_list) == 0:
            reference = get_data_direct(points_s, 'position', dbg=False)
        else:
            reference = (reference[0], torch.tensor(correspondences_list[-1], dtype=torch.float32, device='cuda'))

        # target

        hierarchy_t, bounds_t, metadata_t, points_t, colors_t = octrees[i].to_list()
        target = get_data_direct(points_t, 'position', dbg=False)

        matching = ot.ot_octree(reference, target)
        centroids_s = numpy(reference[1])
        centroids_t = numpy(target[1])

        correspondence_outer = numpy(matching)
        correspondences_list.append(correspondence_outer)

    timer.toc()
    
    return correspondences_list

if __name__ == "__main__":
    main()


