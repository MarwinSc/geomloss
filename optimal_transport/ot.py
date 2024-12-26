import torch
import time
#from geomloss import SamplesLoss
#from geomloss import ImagesBarycenter
from geomloss.samples_loss import SamplesLoss, Samplesloss_octree
import numpy as np
import logging
from pykeops.torch import LazyTensor

from scipy.spatial import KDTree

use_cuda = torch.cuda.is_available()

def normalize(measure, n=None):
    """Reduce a point cloud to at most n points and normalize the weights and point cloud."""
    weights, locations = measure
    N = len(weights)

    if n is not None and n < N:
        n = int(n)
        indices = torch.randperm(N)
        indices = indices[:n]
        weights, locations = weights[indices], locations[indices]

    weights = weights / weights.sum()
    weights, locations = weights.contiguous(), locations.contiguous()

    # Center, normalize the point cloud
    mean = (weights.view(-1, 1) * locations).sum(dim=0)
    locations -= mean
    std = (weights.view(-1) * (locations ** 2).sum(dim=1).view(-1)).sum().sqrt()
    locations /= std

    return (weights, locations), mean, std


def OT_registration(source, target, nits=1):
    a, x = source  # weights, locations
    b, y = target  # weights, locations

    x.requires_grad = True
    z = x.clone()  # Moving point cloud

    if use_cuda:
        torch.cuda.synchronize()

    #Loss = SamplesLoss("sinkhorn", p=2, blur=0.01, scaling=0.5, truncate=1)
    #Loss = SamplesLoss("sinkhorn", p=2, blur=0.02, scaling=0.4, truncate=1, backend="multiscale", diameter=1.0, cluster_scale=0.01)
    #Loss = SamplesLoss("sinkhorn", p=2, blur=0.02, scaling=0.4, truncate=1, backend="online")

    Loss = SamplesLoss("sinkhorn", p=2, blur=0.003, scaling=0.7, truncate=0.1, backend="multiscale", verbose=True)

    for it in range(nits):
        wasserstein_zy = Loss(a, z, b, y)

        start = time.time()

        [grad_z] = torch.autograd.grad(wasserstein_zy, [z])
        z -= grad_z / a[:, None]  # Apply the regularized Brenier map

        # save_vtk(f"matching_{name}_it_{it}.vtk", numpy(z), colors)

        end = time.time()
        print("Autodiff gradient in {:.3f}s.".format(end - start))

    return z


def barycenter(measures, weights):

    a, x = source  # weights, locations
    b, y = target  # weights, locations

    #N, M, D = x.shape[0], y.shape[0], x.shape[1]
    x.requires_grad = True
    z = x.clone()  # Moving point cloud

    if use_cuda:
        torch.cuda.synchronize()

    Loss = SamplesLoss("sinkhorn", p=2, blur=0.001, scaling=0.7, truncate=1, backend="multiscale")
    wasserstein_zy = Loss(a, z, b, y)

    [grad_z] = torch.autograd.grad(wasserstein_zy, [z])
    z -= grad_z / a[:, None] 

    model = x - z / a.view(N, 1, 1)

    barycenter = 1 * 1 * model

    return model


def ot_labels(source, target, colors_target):
    a, x, lx = source  # weights, locations
    b, y, ly = target  # weights, locations

    N, M, D = x.shape[0], y.shape[0], x.shape[1]

    x.requires_grad = True
    z = x.clone()  # Moving point cloud

    p = 2
    blur = 0.001
    OT_solver = SamplesLoss(loss = "sinkhorn", p = p, blur = blur, reach=None, cluster_scale=1/16, scaling=0.9, truncate=None, backend="multiscale", debias = True, potentials = False, verbose=True) # 
    
    wasserstein_zy = OT_solver(lx, a, z, ly, b, y)  # Dual potentials
    
    [grad_z] = torch.autograd.grad(wasserstein_zy, [z])
    z -= grad_z / a[:, None]  # Apply the regularized Brenier map

    return z

    #F, G = OT_solver(lx, a, x, ly, b, y)  # Dual potentials

    F_i = LazyTensor(F.view(N,1,1))
    G_j = LazyTensor(G.view(1,M,1))
    a_i = LazyTensor(a.view(N,1,1))
    b_j = LazyTensor(b.view(1,M,1))
    x_i = LazyTensor(x.view(N,1,D))
    y_j = LazyTensor(y.view(1,M,D))
    
    # ...
    # C_ij is computed using e.g. some points x_i, y_j
    C_ij = (1/p) * ((x_i - y_j)**p).sum(-1)
    # ...
    eps = blur**p  # temperature epsilon
    # The line below defines a (N, M, 1) symbolic LazyTensor:
    P_ij = ((F_i + G_j - C_ij) / eps).exp() * (a_i * b_j)

    #P_ij = P_ij.normalize()
    
    #[grad_z] = torch.autograd.grad(wasserstein_zy, [z])
    #z -= grad_z / a[:, None]  # Apply the regularized Brenier map

    divisor = LazyTensor(P_ij.sum(1).view(N, 1, 1))

    P_ij = P_ij / divisor#LazyTensor(divisor.view(N, 1, 1))

    #dbg = LazyTensor.cat((P_ij, P_ij, P_ij), dim=2)
    #dbg = dbg @ y_j

    # todo could probably be optimized
    dbg = torch.stack((P_ij @ y[:, 0].contiguous(), P_ij @ y[:, 1].contiguous(), P_ij @ y[:, 2].contiguous()), dim=1)
    color_matching = torch.stack((P_ij @ colors_target[:, 0].contiguous(), P_ij @ colors_target[:, 1].contiguous(), P_ij @ colors_target[:, 2].contiguous(), P_ij @ colors_target[:, 3].contiguous()), dim=1)
    
    return dbg, color_matching



def OT_transport_plan(source, target, colors_target, diameter=np.sqrt(3)):
    a, x = source  # weights, locations
    b, y = target  # weights, locations

    N, M, D = x.shape[0], y.shape[0], x.shape[1]

    x.requires_grad = True
    z = x.clone()  # Moving point cloud

    p = 2
    blur = 0.001
    OT_solver = SamplesLoss(loss = "sinkhorn", p = p, blur = blur, reach=None, diameter=diameter, scaling=0.9, truncate=70, backend="multiscale", debias = True, potentials = True, verbose=True) # 
    F, G = OT_solver(a, x, b, y)  # Dual potentials

    F_i = LazyTensor(F.view(N,1,1))
    G_j = LazyTensor(G.view(1,M,1))
    a_i = LazyTensor(a.view(N,1,1))
    b_j = LazyTensor(b.view(1,M,1))
    x_i = LazyTensor(x.view(N,1,D))
    y_j = LazyTensor(y.view(1,M,D))
    
    # ...
    # C_ij is computed using e.g. some points x_i, y_j
    C_ij = (1/p) * ((x_i - y_j)**p).sum(-1)
    # ...
    eps = blur**p  # temperature epsilon
    # The line below defines a (N, M, 1) symbolic LazyTensor:
    P_ij = ((F_i + G_j - C_ij) / eps).exp() * (a_i * b_j)

    #P_ij = P_ij.normalize()
    
    #[grad_z] = torch.autograd.grad(wasserstein_zy, [z])
    #z -= grad_z / a[:, None]  # Apply the regularized Brenier map

    divisor = LazyTensor(P_ij.sum(1).view(N, 1, 1))

    P_ij = P_ij / divisor#LazyTensor(divisor.view(N, 1, 1))

    #dbg = LazyTensor.cat((P_ij, P_ij, P_ij), dim=2)
    #dbg = dbg @ y_j

    # todo could probably be optimized
    dbg = torch.stack((P_ij @ y[:, 0].contiguous(), P_ij @ y[:, 1].contiguous(), P_ij @ y[:, 2].contiguous()), dim=1)
    color_matching = torch.stack((P_ij @ colors_target[:, 0].contiguous(), P_ij @ colors_target[:, 1].contiguous(), P_ij @ colors_target[:, 2].contiguous(), P_ij @ colors_target[:, 3].contiguous()), dim=1)
    
    return dbg, color_matching


def ot_coarse(source, target):
    a, x = source  # weights, locations
    b, y = target  # weights, locations

    N, M, D = x.shape[0], y.shape[0], x.shape[1]

    p = 2
    blur = 0.02
    OT_solver = SamplesLoss(loss = "sinkhorn", p = p, blur = blur, reach=None, diameter=np.sqrt(3), scaling=0.9, truncate=None, backend="tensorized", debias = True, potentials = True) # 
    F, G = OT_solver(a, x, b, y)  # Dual potentials
    
    a_i, x_i = a.view(N,1), x.view(N,1,D)
    b_j, y_j = b.view(1,M), y.view(1,M,D)
    F_i, G_j = F.view(N,1), G.view(1,M)
    C_ij = (1/p) * ((x_i - y_j)**p).sum(-1)  # (N,M) cost matrix
    eps = blur**p  # temperature epsilon
    P_ij = ((F_i + G_j - C_ij) / eps).exp() * (a_i * b_j)  # (N,M) transport plan
    transport_plan = P_ij.detach().cpu().numpy()
    transport_plan = transport_plan / np.expand_dims(np.sum(transport_plan, axis=1), axis=1)

    indices = np.argwhere(transport_plan)
    values = transport_plan[indices[:, 0], indices[:, 1]]
    return indices, values
    


def ot_coarse_iterations(source, target):
    a, x = source  # weights, locations
    b, y = target  # weights, locations

    N, M, D = x.shape[0], y.shape[0], x.shape[1]

    x.requires_grad = True
    z = x.clone()  # Moving point cloud

    p = 2
    blur = 0.3

    Loss = SamplesLoss("sinkhorn", p=p, blur=blur, scaling=0.9, truncate=1)

    for it in range(10):
        wasserstein_zy = Loss(a, z, b, y)
        [grad_z] = torch.autograd.grad(wasserstein_zy, [z])
        z -= grad_z / a[:, None]  # Apply the regularized Brenier map


    OT_solver = SamplesLoss(loss = "sinkhorn", p = p, blur = blur, reach=None, diameter=np.sqrt(3), scaling=0.9, truncate=1, backend="multiscale", debias=True, potentials=True) # 
    F, G = OT_solver(a, z, b, y)  # Dual potentials
    
    a_i, x_i = a.view(N,1), z.view(N,1,D)
    b_j, y_j = b.view(1,M), y.view(1,M,D)
    F_i, G_j = F.view(N,1), G.view(1,M)
    C_ij = (1/p) * ((x_i - y_j)**p).sum(-1)  # (N,M) cost matrix
    eps = blur**p  # temperature epsilon
    P_ij = ((F_i + G_j - C_ij) / eps).exp() * (a_i * b_j)  # (N,M) transport plan
    transport_plan = P_ij.detach().cpu().numpy()
    transport_plan = transport_plan / np.expand_dims(np.sum(transport_plan, axis=1), axis=1)

    indices = np.argwhere(transport_plan)
    values = transport_plan[indices[:, 0], indices[:, 1]]
    return indices, values


def OT_transport_plan_full(source, target):
    a, x = source  # weights, locations
    b, y = target  # weights, locations

    N, M, D = x.shape[0], y.shape[0], x.shape[1]

    x.requires_grad = True
    z = x.clone()  # Moving point cloud

    p = 2
    blur = 0.01
    OT_solver = SamplesLoss(loss = "sinkhorn", p = p, blur = blur, scaling=0.4, truncate=1, backend="multiscale", debias = True, potentials = True) # 
    F, G = OT_solver(a, x, b, y)  # Dual potentials
    
    a_i, x_i = a.view(N,1), x.view(N,1,D)
    b_j, y_j = b.view(1,M), y.view(1,M,D)
    F_i, G_j = F.view(N,1), G.view(1,M)
    C_ij = (1/p) * ((x_i - y_j)**p).sum(-1)  # (N,M) cost matrix
    eps = blur**p  # temperature epsilon
    P_ij = ((F_i + G_j - C_ij) / eps).exp() * (a_i * b_j)  # (N,M) transport plan
    transport_plan = P_ij.detach().cpu().numpy()
    transport_plan = transport_plan / np.expand_dims(np.sum(transport_plan, axis=1), axis=1)
    #transport_plan = transport_plan / np.expand_dims(a.detach().cpu().numpy(), axis=1)

    #cost_dbg = C_ij.detach().cpu().numpy()

    #cost_sum = np.sum(cost_dbg * transport_plan)

    temp_x = y_j.detach().cpu().numpy()
    temp_x = np.full((N, M, 3), temp_x)

    transport_plan = np.dstack((transport_plan, transport_plan, transport_plan))
    tmp = transport_plan * temp_x
    return np.sum(tmp, axis=1)

def ot_octree(source, target):

    if use_cuda:
        torch.cuda.synchronize()
    start = time.time()

    blur = 0.001
    p = 2

    #x.requires_grad = True
    #z = x.clone()  # Moving point cloud
#
    #wasserstein_zy = Samplesloss_octree("sinkhorn", p=p, blur=blur, scaling=0.7, truncate=None, backend="multiscale", potentials=True)
#
    #[grad_z] = torch.autograd.grad(wasserstein_zy, [z])
    #z -= grad_z / a[:, None]  # Apply the regularized Brenier map
    #
    #return z


    Loss = Samplesloss_octree("sinkhorn", p=p, blur=blur, scaling=0.7, truncate=2, backend="multiscale", potentials=True)#, reach=1)

    F, G, x, y = Loss(source, target)

    end = time.time()   
    print("Registered shape in {:.3f}s.".format(end - start))
    start = time.time()

    debug_F = F.detach().cpu().numpy()
    debug_G = G.detach().cpu().numpy()
    print(f"f real: {np.count_nonzero(np.isfinite(debug_F))}, min {np.min(debug_F)}, max {np.max(debug_F)}")
    print(f"g real: {np.count_nonzero(np.isfinite(debug_G))}, min {np.min(debug_G)}, max {np.max(debug_G)}")

    N, M, D = source.point_count, target.point_count, 3
    
    weights = np.ones(source.points.shape[0])
    a = torch.tensor(weights / np.sum(weights), dtype=torch.float32, device='cuda')

    weights = np.ones(target.points.shape[0])
    b = torch.tensor(weights / np.sum(weights), dtype=torch.float32, device='cuda')

    #a, x, b, y = a.contiguous(), x.contiguous(), b.contiguous(), y.contiguous()

    F_i = LazyTensor(F.view(N,1,1))
    G_j = LazyTensor(G.view(1,M,1))
    a_i = LazyTensor(a.view(N,1,1))
    b_j = LazyTensor(b.view(1,M,1))
    x_i = LazyTensor(x.view(N,1,D))
    y_j = LazyTensor(y.view(1,M,D))

    end = time.time()   
    print("Set up Lazytensors in {:.3f}s.".format(end - start))
    start = time.time()
    
    # ...
    # C_ij is computed using e.g. some points x_i, y_j
    C_ij = (1/p) * ((x_i - y_j)**p).sum(-1)

    end = time.time()   
    print("Form the cost matrix in {:.3f}s.".format(end - start))
    start = time.time()

    # ...
    eps = blur**p  # temperature epsilon
    # The line below defines a (N, M, 1) symbolic LazyTensor:
    P_ij = ((F_i + G_j - C_ij) / eps).exp()# * (a_i * b_j)

    end = time.time()   
    print("Form the Transport matrix in {:.3f}s.".format(end - start))
    start = time.time()

    #P_ij = P_ij.normalize()
    
    #[grad_z] = torch.autograd.grad(wasserstein_zy, [z])
    #z -= grad_z / a[:, None]  # Apply the regularized Brenier map

    #divisor = LazyTensor(P_ij.sum(1).view(N, 1, 1))

    #debug = P_ij.sum(1).detach().cpu().numpy()

    P_ij = P_ij / P_ij.sum(1).view(N, 1, 1)

    #dbg = LazyTensor.cat((P_ij, P_ij, P_ij), dim=2)
    #dbg = dbg @ y_j

    end = time.time()   
    print("Normalize the Transport matrix in {:.3f}s.".format(end - start))
    start = time.time()

    # todo could probably be optimized
    #dbg = torch.stack((P_ij @ y[:, 0].contiguous(), P_ij @ y[:, 1].contiguous(), P_ij @ y[:, 2].contiguous()), dim=1)

    dbg = P_ij @ y

    end = time.time()   
    print("Compute coordinates in {:.3f}s.".format(end - start))
    start = time.time()

    colors_target = torch.tensor(target.colors, dtype=torch.float32, device='cuda')
    #color_matching = torch.stack((P_ij @ colors_target[:, 0].contiguous(), P_ij @ colors_target[:, 1].contiguous(), P_ij @ colors_target[:, 2].contiguous(), P_ij @ colors_target[:, 3].contiguous()), dim=1)
    color_matching = P_ij @ colors_target

    end = time.time()   
    print("Compute Colors {:.3f}s.".format(end - start))

    return dbg, color_matching


def ot_octree_autodiff(source, target):

    if use_cuda:
        torch.cuda.synchronize()
    start = time.time()

    blur = 0.001
    p = 2

    Loss = Samplesloss_octree("sinkhorn", p=p, blur=blur, scaling=0.7, truncate=5, backend="multiscale", potentials=False)#, reach=1)

    emd = Loss(source, target)
    [grad_source] = torch.autograd.grad(emd, [source.points])

    # todo is using clone() here the most efficient way?
    points = source.points.clone() - grad_source / source.weights[:, None]  # Apply the regularized Brenier map

    #source.points -= grad_source / source.weights[:, None]  # Apply the regularized Brenier map

    end = time.time()   
    print("Registered shape in {:.3f}s.".format(end - start))

    if True:
        start = time.time()
        kd = KDTree(target.points.detach().cpu().numpy())
        d, i = kd.query(points.detach().cpu().numpy(), k=1, distance_upper_bound=0.3)
        color = torch.tensor(target.colors[i], dtype=torch.float32, device='cuda')
        end = time.time()   
        print("KD color query in {:.3f}s.".format(end - start))
    return points, color


