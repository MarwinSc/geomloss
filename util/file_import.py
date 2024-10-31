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
import util.octree as octree
from util.timer import Timer

log = logging.getLogger(__name__)

def read(filename, mean):

    timer = Timer(f"Import {filename}")

    name, extension = os.path.splitext(filename)

    if extension == ".obj":
        mesh = meshio.read(filename, file_format="obj")
        points = mesh.points
        colors = None

    elif extension == ".ply":

        timer_inner = Timer("meshio ply import")

        mesh = meshio.read(filename, file_format="ply")

        timer_inner.toc()
        
        points = mesh.points

        # todo
        #return points, None

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
    
    if mean is None:
        mean = np.mean(points, axis=0)
    points -= mean
    normalized_points = points/np.max(np.abs(points))
    
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
    
    timer.toc()

    oct = octree.Octree(normalized_points, 5000, colors=colors)

    return oct, mean


def load_assignment(filepath):
    return np.load(filepath)