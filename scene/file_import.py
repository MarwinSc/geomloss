import os
import e57
import meshio
import laspy
import numpy as np
import logging
from sklearn import cluster
import torch
import SimpleITK as sitk
from skimage.measure import marching_cubes
import robust_laplacian
import point_cloud_utils as pcu
import scene.octree as octree
from util.timer import Timer

log = logging.getLogger(__name__)

def read(filename, octree_node_size=1000, normalize_data=True):

    timer = Timer(f"Import {filename}")

    name, extension = os.path.splitext(filename)

    if extension == ".obj":
        mesh = meshio.read(filename, file_format="obj")
        points = mesh.points
        colors = np.ones((points.shape[0], 4), dtype=np.float32)

    elif extension == ".ply":

        timer_inner = Timer("meshio ply import")

        mesh = meshio.read(filename, file_format="ply")

        timer_inner.toc()
        
        points = mesh.points
        
        if True:
        # todo
            colors = np.ones((points.shape[0], 4), dtype=np.float32)
        else:
            red = mesh.point_data['red'].astype(np.ubyte)
            green = mesh.point_data['green'].astype(np.ubyte)
            blue = mesh.point_data['blue'].astype(np.ubyte)
            colors = np.hstack((red.reshape(red.shape[0], 1),
                                green.reshape(green.shape[0], 1),
                                blue.reshape(blue.shape[0], 1)))
            colors = colors.astype(np.float32)
            #  todo change min max to 0 and 255
            colors = (1.0 / (np.max(colors) - np.min(colors))) * (colors - np.min(colors))  # map to range 0, 1
            colors = np.hstack((colors, np.ones((colors.shape[0], 1), dtype=np.float32)))

    elif extension == ".e57":
        pc = e57.read_points(filename)
        points = pc.points
        colors = pc.color
        colors = np.hstack((colors, np.ones((colors.shape[0], 1))))

    elif extension == ".laz":
        points_out = []
        count = 0
        with laspy.open(filename) as input_las:
            for points in input_las.chunk_iterator(200_000):

                points_out.extend(np.vstack((points.x, points.y, points.z)).transpose())

        points = np.r_[points_out]
        colors = None

    # todo
    # could also use niibabel?
    elif extension == "nii":
        """Uses the marching cube algorithm to turn a .nii binary mask into a surface weighted point cloud."""
        tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

        mask = sitk.GetArrayFromImage(sitk.ReadImage(fname))
        # mask = skimage.transform.downscale_local_mean(mask, (4,4,4))
        points, triangles, normals, values = marching_cubes(mask, 0.5)

        # Our mesh is given as a collection of ABC triangles:
        A, B, C = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]

        # Locations and weights of our Dirac atoms:
        X = (A + B + C) / 3  # centers of the faces
        S = np.sqrt(np.sum(np.cross(B - A, C - A) ** 2, 1)) / 2  # areas of the faces

        # We return a (normalized) vector of weights + a "list" of points
        weigths = tensor(S / np.sum(S)) 
        points = tensor(X)

    print(f"Number of Points: {points.shape[0]}")
    
    timer.toc()

    if False:
        colors = change_colors_for_evaluation(points)

    return points, colors
    
    #timer_laplacian = Timer("laplacian")
    #L, M = robust_laplacian.point_cloud_laplacian(normalized_points)
    #timer_laplacian.toc()

    #idx = np.random.randint(0, len(points), 100000)
    #idx = pcu.downsample_point_cloud_poisson_disk(normalized_points, 0.04)
    #points = points[idx]
    #colors = colors[idx]
    
    #bbox_size = normalized_points.max(0) - normalized_points.min(0)
    #size_of_voxel = bbox_size / 128
    #points, colors = pcu.downsample_point_cloud_on_voxel_grid(size_of_voxel, normalized_points, colors)
    
    #kmeans = cluster.KMeans(n_clusters=8000, max_iter=10).fit(points)

    #weights = np.ones(points.shape[0])
    #weights = torch.tensor(weights / np.sum(weights), dtype=torch.float32, device='cuda')
    #points = torch.tensor(points, dtype=torch.float32, device='cuda')
    #weights, locations = weights.contiguous(), points.contiguous()
    #mean = (weights.view(-1, 1) * locations).sum(dim=0)
    #locations -= mean
    #std = (weights.view(-1) * (locations ** 2).sum(dim=1).view(-1)).sum().sqrt()
    #locations /= std
    #normalized_points = locations.detach().cpu().numpy()

def change_colors_for_evaluation(points, mode_rgb):
    
    if mode_rgb:
        # Step 1: Compute the bounding box
        min_coords = points.min(axis=0)
        max_coords = points.max(axis=0)
        # Step 2: Translate points to center at origin
        #midpoint = (min_coords + max_coords) / 2
        points = points - min_coords
        # Step 3: Normalize to range [0, 1]
        scale = max(max_coords - min_coords)
        points = points / scale
        return np.hstack((points, np.ones((points.shape[0], 1), dtype=np.float32)))
    else:
        x = points[:, 0]
        # Normalize x to a 0-1 range for the pattern
        x_min, x_max = np.min(x), np.max(x)
        x_norm = (x - x_min) / (x_max - x_min)  # Avoid division by zero

        # One full cycle: red -> white -> blue -> white
        n_cycles = 7
        scaled_x = x_norm * n_cycles

        #color1 = (252/255, 253/255, 191/255)
        color1 = (1.0, 0.0, 0.0)
        #color2 = (183/255, 55/255, 121/255)
        color2 = (0.0, 0.0, 1.0)
        #mid_color = (0/255, 0/255, 4/255)
        mid_color = (1.0, 1.0, 1.0)

        # Fractional part of scaled_x for the pattern
        frac = scaled_x % 1

        colors = np.zeros((len(x), 3))

        # color1 to mid_color (0 <= frac < 0.25)
        mask1 = frac < 0.25
        t = frac[mask1] / 0.25
        colors[mask1, 0] = color1[0] + (mid_color[0] - color1[0]) * t
        colors[mask1, 1] = color1[1] + (mid_color[1] - color1[1]) * t
        colors[mask1, 2] = color1[2] + (mid_color[2] - color1[2]) * t

        # mid_color to color2 (0.25 <= frac < 0.5)
        mask2 = (frac >= 0.25) & (frac < 0.5)
        t = (frac[mask2] - 0.25) / 0.25
        colors[mask2, 0] = mid_color[0] + (color2[0] - mid_color[0]) * t
        colors[mask2, 1] = mid_color[1] + (color2[1] - mid_color[1]) * t
        colors[mask2, 2] = mid_color[2] + (color2[2] - mid_color[2]) * t

        # color2 to mid_color (0.5 <= frac < 0.75)
        mask3 = (frac >= 0.5) & (frac < 0.75)
        t = (frac[mask3] - 0.5) / 0.25
        colors[mask3, 0] = color2[0] + (mid_color[0] - color2[0]) * t
        colors[mask3, 1] = color2[1] + (mid_color[1] - color2[1]) * t
        colors[mask3, 2] = color2[2] + (mid_color[2] - color2[2]) * t

        # mid_color to color1 (0.75 <= frac < 1)
        mask4 = frac >= 0.75
        t = (frac[mask4] - 0.75) / 0.25
        colors[mask4, 0] = mid_color[0] + (color1[0] - mid_color[0]) * t
        colors[mask4, 1] = mid_color[1] + (color1[1] - mid_color[1]) * t
        colors[mask4, 2] = mid_color[2] + (color1[2] - mid_color[2]) * t
        return np.hstack((colors, np.ones((colors.shape[0], 1), dtype=np.float32)))

def load_assignment(filepath):
    return np.load(filepath)

def write_assignment(coordinates, colors, filepath, mode="e57"):

    if mode == "e57":
        import pye57

        # Create E57 file
        e57_file = pye57.E57(str(filepath.with_suffix(".e57")), mode='w')

        # Prepare data dictionary
        data = {
            "cartesianX": coordinates[:, 0],
            # flipped: 
            "cartesianZ": coordinates[:, 1],
            "cartesianY": coordinates[:, 2],
            "colorRed": colors[:, 0],
            "colorGreen": colors[:, 1],
            "colorBlue": colors[:, 2],
        }

        # Write the scan
        e57_file.write_scan_raw(data)
        e57_file.close()

    elif mode == "npy":
        np.save("output/" + filepath.stem + "_coordinates.npy", coordinates)
        np.save("output/" + filepath.stem + "_colors.npy", colors)
