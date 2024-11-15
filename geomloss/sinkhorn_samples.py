"""Implements the (debiased) Sinkhorn divergence between sampled measures."""

import numpy as np
import torch
from functools import partial

try:  # Import the keops library, www.kernel-operations.io
    from pykeops.torch import generic_logsumexp
    from pykeops.torch.cluster import grid_cluster, cluster_ranges_centroids
    from pykeops.torch.cluster import sort_clusters, from_matrix, swap_axes
    from pykeops.torch import LazyTensor, Vi, Vj, Pm

    keops_available = True
except:
    keops_available = False

from .utils import scal, squared_distances, distances

from .sinkhorn_divergence import epsilon_schedule, scaling_parameters
from .sinkhorn_divergence import dampening, log_weights, sinkhorn_cost, sinkhorn_loop


# ==============================================================================
#                          backend == "tensorized"
# ==============================================================================

cost_routines = {
    1: (lambda x, y: distances(x, y)),
    2: (lambda x, y: squared_distances(x, y) / 2),
}


def softmin_tensorized(eps, C_xy, h_y):
    r"""Soft-C-transform, implemented using dense torch Tensors.

    This routine implements the (soft-)C-transform
    between dual vectors, which is the core computation for
    Auction- and Sinkhorn-like optimal transport solvers.

    If `eps` is a float number, `C_xy` is a (batched) cost matrix :math:`C(x_i,y_j)`
    and `h_y` encodes a dual potential :math:`h_j` that is supported by the points
    :math:`y_j`'s, then `softmin_tensorized(eps, C_xy, h_y)` returns a dual potential
    `f` for ":math:`f_i`", supported by the :math:`x_i`'s, that is equal to:

    .. math::
        f_i \gets - \varepsilon \log \sum_{j=1}^{\text{M}} \exp
        \big[ h_j - C(x_i, y_j) / \varepsilon \big]~.

    For more detail, see e.g. Section 3.3 and Eq. (3.186) in Jean Feydy's PhD thesis.

    Args:
        eps (float, positive): Temperature :math:`\varepsilon` for the Gibbs kernel
            :math:`K_{i,j} = \exp(-C(x_i, y_j) / \varepsilon)`.

        C_xy ((B, N, M) Tensor): Cost matrix :math:`C(x_i,y_j)`, with a batch dimension.

        h_y ((B, M) Tensor): Vector of logarithmic "dual" values, with a batch dimension.
            Most often, this vector will be computed as `h_y = b_log + g_j / eps`,
            where `b_log` is a vector of log-weights :math:`\log(\beta_j)`
            for the :math:`y_j`'s and :math:`g_j` is a dual vector
            in the Sinkhorn algorithm, so that:

            .. math::
                f_i \gets - \varepsilon \log \sum_{j=1}^{\text{M}} \beta_j
                \exp \tfrac{1}{\varepsilon} \big[ g_j - C(x_i, y_j) \big]~.

    Returns:
        (B, N) Tensor: Dual potential `f` of values :math:`f_i`, supported
            by the points :math:`x_i`.
    """
    B = C_xy.shape[0]
    return -eps * (h_y.view(B, 1, -1) - C_xy / eps).logsumexp(2).view(B, -1)


def sinkhorn_tensorized(
    a,
    x,
    b,
    y,
    p=2,
    blur=0.05,
    reach=None,
    diameter=None,
    scaling=0.5,
    cost=None,
    debias=True,
    potentials=False,
    **kwargs,
):
    r"""Vanilla PyTorch implementation of the Sinkhorn divergence.

    Args:
        a ((B, N) Tensor): Weights :math:`\alpha_i` for the first measure,
            with a batch dimension.

        x ((B, N, D) Tensor): Sampling locations :math:`x_i` for the first measure,
            with a batch dimension.

        b ((B, M) Tensor): Weights :math:`\beta_j` for the second measure,
            with a batch dimension.

        y ((B, M, D) Tensor): Sampling locations :math:`y_j` for the second measure,
            with a batch dimension.

        p (int, optional): Exponent of the ground cost function
            :math:`C(x_i,y_j)`, which is equal to
            :math:`\tfrac{1}{p}\|x_i-y_j\|^p` if it is not provided
            explicitly through the `cost` optional argument.
            Defaults to 2.

        blur (float, optional): Target value for the blurring scale
            of the Gibbs kernel
            :math:`K_{i,j} = \exp(-C(x_i,y_j)/\varepsilon) = \exp(-\|x_i-y_j\|^p / p \text{blur}^p).
            In the Sinkhorn algorithm, the temperature :math:`\varepsilon`
            is computed as :math:`\text{blur}^p`.
            Defaults to 0.05.

        reach (float or None (= +infty), optional): Typical scale for the
            maximum displacement between any two points :math:`x_i` and :math:`y_j`
            in the optimal transport model.
            In the unbalanced Sinkhorn divergence,
            the strength :math:`\rho` of the soft marginal constraints
            is computed as :math:`\rho = \text{reach}^p`.
            Defaults to None.

        diameter (float or None, optional): Upper bound on the value
            of the distance :math:`\|x_i-y_j\|` between any two samples.
            This will be used as a first value of the `blur` radius
            in the epsilon-scaling annealing descent.
            Defaults to None: an upper bound will be estimated on the fly.

        scaling (float in (0, 1), optional): Ratio between two successive
            values of the blur radius in the epsilon-scaling annealing descent.
            Defaults to 0.5.

        cost (function, optional): Cost function :math:`C(x_i,y_j)`.
            It should take as input two point clouds `x` and `y`
            with a batch dimension, encoded as `(B, N, D)`, `(B, M, D)`
            torch Tensors and return a `(B, N, M)` torch Tensor.
            Defaults to None: we use a Euclidean cost
            :math:`C(x_i,y_j) = \tfrac{1}{p}\|x_i-y_j\|^p`.

        debias (bool, optional): Should we used the "de-biased" Sinkhorn divergence
            :math:`\text{S}_{\varepsilon, \rho}(\al,\be)` instead
            of the "raw" entropic OT cost
            :math:`\text{OT}_{\varepsilon, \rho}(\al,\be)`?
            This slows down the OT solver but guarantees that our approximation
            of the Wasserstein distance will be positive and definite
            - up to convergence of the Sinkhorn loop.
            For a detailed discussion of the influence of this parameter,
            see e.g. Fig. 3.21 in Jean Feydy's PhD thesis.
            Defaults to True.

        potentials (bool, optional): Should we return the optimal dual potentials
            instead of the cost value?
            Defaults to False.

    Returns:
        (B,) Tensor or pair of (B, N), (B, M) Tensors: if `potentials` is True,
            we return a pair of (B, N), (B, M) Tensors that encode the optimal dual vectors,
            respectively supported by :math:`x_i` and :math:`y_j`.
            Otherwise, we return a (B,) Tensor of values for the Sinkhorn divergence.
    """

    # Retrieve the batch size B, the numbers of samples N, M
    # and the size of the ambient space D:
    B, N, D = x.shape
    _, M, _ = y.shape

    # By default, our cost function :math:`C(x_i,y_j)` is a halved,
    # squared Euclidean distance (p=2) or a simple Euclidean distance (p=1):
    if cost is None:
        cost = cost_routines[p]

    # Compute the relevant cost matrices C(x_i, y_j), C(y_j, x_i), etc.
    # Note that we "detach" the gradients of the "right-hand sides":
    # this is coherent with the way we compute our gradients
    # in the `sinkhorn_loop(...)` routine, in the `sinkhorn_divergence.py` file.
    # Please refer to the comments in this file for more details.
    C_xy = cost(x, y.detach())  # (B,N,M) torch Tensor
    C_yx = cost(y, x.detach())  # (B,M,N) torch Tensor

    # N.B.: The "auto-correlation" matrices C(x_i, x_j) and C(y_i, y_j)
    #       are only used by the "debiased" Sinkhorn algorithm.
    C_xx = cost(x, x.detach()) if debias else None  # (B,N,N) torch Tensor
    C_yy = cost(y, y.detach()) if debias else None  # (B,M,M) torch Tensor

    # Compute the relevant values of the diameter of the configuration,
    # target temperature epsilon, temperature schedule across itereations
    # and strength of the marginal constraints:
    diameter, eps, eps_list, rho = scaling_parameters(
        x, y, p, blur, reach, diameter, scaling
    )

    # Use an optimal transport solver to retrieve the dual potentials:
    f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(
        softmin_tensorized,
        log_weights(a),
        log_weights(b),
        C_xx,
        C_yy,
        C_xy,
        C_yx,
        eps_list,
        rho,
        debias=debias,
    )

    # Optimal transport cost:
    return sinkhorn_cost(
        eps,
        rho,
        a,
        b,
        f_aa,
        g_bb,
        g_ab,
        f_ba,
        batch=True,
        debias=debias,
        potentials=potentials,
    )


# ==============================================================================
#                          backend == "online"
# ==============================================================================


def softmin_online_lazytensor(eps, C_xy, h_y, p=2):
    r"""Soft-C-transform, implemented using symbolic KeOps LazyTensors.

    This routine implements the (soft-)C-transform
    between dual vectors, which is the core computation for
    Auction- and Sinkhorn-like optimal transport solvers.

    If `eps` is a float number, `C_xy = (x, y)` is a pair of (batched)
    point clouds, encoded as (B, N, D) and (B, M, D) Tensors
    and `h_y` encodes a dual potential :math:`h_j` that is supported by the points
    :math:`y_j`'s, then `softmin_tensorized(eps, C_xy, h_y)` returns a dual potential
    `f` for ":math:`f_i`", supported by the :math:`x_i`'s, that is equal to:

    .. math::
        f_i \gets - \varepsilon \log \sum_{j=1}^{\text{M}} \exp
        \big[ h_j - \|x_i - y_j\|^p / p \varepsilon \big]~.

    For more detail, see e.g. Section 3.3 and Eq. (3.186) in Jean Feydy's PhD thesis.

    Args:
        eps (float, positive): Temperature :math:`\varepsilon` for the Gibbs kernel
            :math:`K_{i,j} = \exp(- \|x_i - y_j\|^p / p \varepsilon)`.

        C_xy (pair of (B, N, D), (B, M, D) Tensors): Point clouds :math:`x_i`
            and :math:`y_j`, with a batch dimension.

        h_y ((B, M) Tensor): Vector of logarithmic "dual" values, with a batch dimension.
            Most often, this vector will be computed as `h_y = b_log + g_j / eps`,
            where `b_log` is a vector of log-weights :math:`\log(\beta_j)`
            for the :math:`y_j`'s and :math:`g_j` is a dual vector
            in the Sinkhorn algorithm, so that:

            .. math::
                f_i \gets - \varepsilon \log \sum_{j=1}^{\text{M}} \beta_j
                \exp \tfrac{1}{\varepsilon} \big[ g_j - \|x_i - y_j\|^p / p \big]~.

    Returns:
        (B, N) Tensor: Dual potential `f` of values :math:`f_i`, supported
            by the points :math:`x_i`.
    """
    x, y = C_xy  # Retrieve our point clouds
    B = x.shape[0]  # Batch dimension

    # Encoding as batched KeOps LazyTensors:
    x_i = LazyTensor(x[:, :, None, :])  # (B, N, 1, D)
    y_j = LazyTensor(y[:, None, :, :])  # (B, 1, M, D)
    h_j = LazyTensor(h_y[:, None, :, None])  # (B, 1, M, 1)

    # Cost matrix:
    if p == 2:  # Halved, squared Euclidean distance
        C_ij = ((x_i - y_j) ** 2).sum(-1) / 2  # (B, N, M, 1)

    elif p == 1:  # Simple Euclidean distance
        C_ij = ((x_i - y_j) ** 2).sum(-1).sqrt()  # (B, N, M, 1)

    else:
        raise NotImplementedError()

    # KeOps log-sum-exp reduction over the "M" dimension:
    smin = (h_j - C_ij * torch.Tensor([1 / eps]).type_as(x)).logsumexp(2).view(B, -1)

    return -eps * smin


def lse_lazytensor(p, D, batchdims=(1,)):
    """This implementation is currently disabled."""

    x_i = Vi(0, D)
    y_j = Vj(1, D)
    f_j = Vj(2, 1)
    epsinv = Pm(3, 1)

    x_i.batchdims = batchdims
    y_j.batchdims = batchdims
    f_j.batchdims = batchdims
    epsinv.batchdims = batchdims

    if p == 2:
        D_ij = ((x_i - y_j) ** 2).sum(-1) / 2
    elif p == 1:
        D_ij = ((x_i - y_j) ** 2).sum(-1).sqrt()

    smin = (f_j - epsinv * D_ij).logsumexp(2)
    return smin


# Low-level KeOps formulas for the ground cost:
cost_formulas = {
    1: "Norm2(X-Y)",
    2: "(SqDist(X,Y) / IntCst(2))",
}


def lse_genred(cost, D, dtype="float32"):
    """Legacy "Genred" implementation, with low-level KeOps formulas."""

    log_conv = generic_logsumexp(
        "( B - (P * " + cost + " ) )",
        "A = Vi(1)",
        f"X = Vi({D})",
        f"Y = Vj({D})",
        "B = Vj(1)",
        "P = Pm(1)",
        # dtype=dtype,
    )
    return log_conv


def softmin_online(eps, C_xy, h_y, log_conv=None):
    x, y = C_xy
    # KeOps is pretty picky on the input shapes...
    batch = x.dim() > 2
    B = x.shape[0]
    h = h_y.view(B, -1, 1) if batch else h_y.view(-1, 1)

    out = -eps * log_conv(x, y, h, torch.Tensor([1 / eps]).type_as(x))

    return out.view(B, -1) if batch else out.view(1, -1)


def sinkhorn_online(
    a,
    x,
    b,
    y,
    p=2,
    blur=0.05,
    reach=None,
    diameter=None,
    scaling=0.5,
    cost=None,
    debias=True,
    potentials=False,
    **kwargs,
):
    B, N, D = x.shape
    B, M, _ = y.shape

    if cost is None and B > 1:
        if True:
            # raise ValueError("Not expected in this benchmark!")
            softmin = partial(softmin_online_lazytensor, p=p)
        else:
            my_lse = lse_lazytensor(p, D, batchdims=(B,))
            softmin = partial(softmin_online, log_conv=my_lse)

    else:
        if B > 1:
            raise ValueError(
                "Custom cost functions are not yet supported with batches." ""
            )

        x = x.squeeze(0)  # (1, N, D) -> (N, D)
        y = y.squeeze(0)  # (1, M, D) -> (M, D)

        if cost is None:
            cost = cost_formulas[p]

        my_lse = lse_genred(cost, D, dtype=str(x.dtype)[6:])
        softmin = partial(softmin_online, log_conv=my_lse)

    # The "cost matrices" are implicitly encoded in the point clouds,
    # and re-computed on-the-fly:
    C_xx, C_yy = ((x, x.detach()), (y, y.detach())) if debias else (None, None)
    C_xy, C_yx = ((x, y.detach()), (y, x.detach()))

    diameter, eps, eps_list, rho = scaling_parameters(
        x, y, p, blur, reach, diameter, scaling
    )

    f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(
        softmin,
        log_weights(a),
        log_weights(b),
        C_xx,
        C_yy,
        C_xy,
        C_yx,
        eps_list,
        rho,
        debias=debias,
    )

    return sinkhorn_cost(
        eps,
        rho,
        a,
        b,
        f_aa,
        g_bb,
        g_ab,
        f_ba,
        batch=True,
        debias=debias,
        potentials=potentials,
    )


# ==============================================================================
#                          backend == "multiscale"
# ==============================================================================


def keops_lse(cost, D, dtype="float32"):
    log_conv = generic_logsumexp(
        "( B - (P * " + cost + " ) )",
        "A = Vi(1)",
        "X = Vi({})".format(D),
        "Y = Vj({})".format(D),
        "B = Vj(1)",
        "P = Pm(1)",
        # dtype=dtype,
    )
    return log_conv


def softmin_multiscale(eps, C_xy, f_y, log_conv=None):
    x, y, ranges_x, ranges_y, ranges_xy = C_xy
    # KeOps is pretty picky on the input shapes...
    return -eps * log_conv(
        x, y, f_y.view(-1, 1), torch.Tensor([1 / eps]).type_as(x), ranges=ranges_xy
    ).view(-1)


def clusterize(a, x, scale=None, labels=None):
    """
    Performs a simple 'voxelgrid' clustering on the input measure,
    putting points into cubic bins of size 'scale' = σ_c.
    The weights are summed, and the centroid position is that of the bin's center of mass.
    Most importantly, the "fine" lists of weights and points are *sorted*
    so that clusters are *contiguous in memory*: this allows us to perform
    kernel truncation efficiently on the GPU.

    If
        [a_c, a], [x_c, x], [x_ranges] = clusterize(a, x, σ_c),
    then
        a_c[k], x_c[k] correspond to
        a[x_ranges[k,0]:x_ranges[k,1]], x[x_ranges[k,0]:x_ranges[k,1],:]
    """
    perm = None  # did we sort the point cloud at some point? Here's the permutation.

    if (
        labels is None and scale is None
    ):  # No clustering, single-scale Sinkhorn on the way...
        return [a], [x], []

    else:  # As of today, only two-scale Sinkhorn is implemented:
        # Compute simple (voxel-like) class labels:
        x_lab = grid_cluster(x, scale) if labels is None else labels
        # Compute centroids and weights:
        ranges_x, x_c, a_c = cluster_ranges_centroids(x, x_lab, weights=a)
        # Make clusters contiguous in memory:
        x_labels, perm = torch.sort(x_lab.view(-1))
        a, x = a[perm], x[perm]

        # N.B.: the lines above were return to replace a call to
        #       'sort_clusters' which does not return the permutation,
        #       an information that is needed to de-permute the dual potentials
        #       if they are required by the user.
        # (a, x), x_labels = sort_clusters( (a,x), x_lab)

        return [a_c, a], [x_c, x], [ranges_x], perm


def kernel_truncation(
    C_xy, C_yx, C_xy_, C_yx_, f_ba, g_ab, eps, truncate=None, cost=None, verbose=False
):
    """Prunes out useless parts of the (block-sparse) cost matrices for finer scales.

    This is where our approximation takes place.
    To be mathematically rigorous, we should make several coarse-to-fine passes,
    making sure that we're not forgetting anyone. A good reference here is
    Bernhard Schmitzer's work: "Stabilized Sparse Scaling Algorithms for
    Entropy Regularized Transport Problems, (2016)".
    """
    if truncate is None:
        return C_xy_, C_yx_
    else:
        x, yd, ranges_x, ranges_y, _ = C_xy
        y, xd, _, _, _ = C_yx
        x_, yd_, ranges_x_, ranges_y_, _ = C_xy_
        y_, xd_, _, _, _ = C_yx_

        with torch.no_grad():
            C = cost(x, y)
            keep = f_ba.view(-1, 1) + g_ab.view(1, -1) > C - truncate * eps
            ranges_xy_ = from_matrix(ranges_x, ranges_y, keep)
            if verbose:
                ks, Cs = keep.sum(), C.shape[0] * C.shape[1]
                print(
                    "Keep {}/{} = {:2.1f}% of the coarse cost matrix.".format(
                        ks, Cs, 100 * float(ks) / Cs
                    )
                )

        return (x_, yd_, ranges_x_, ranges_y_, ranges_xy_), (
            y_,
            xd_,
            ranges_y_,
            ranges_x_,
            swap_axes(ranges_xy_),
        )


def extrapolate_samples(f_ba, g_ab, eps, damping, C_xy, b_log, C_xy_, softmin=None):
    yd = C_xy[1]  # Source points (coarse)
    x_ = C_xy_[0]  # Target points (fine)

    C = (
        x_,
        yd,
        None,
        None,
        None,
    )  # "Rectangular" cost matrix, don't bother with ranges
    return damping * softmin(eps, C, (b_log + g_ab / eps).detach())


def sinkhorn_multiscale(
    a,
    x,
    b,
    y,
    p=2,
    blur=0.05,
    reach=None,
    diameter=None,
    scaling=0.5,
    truncate=5,
    cost=None,
    cluster_scale=None,
    debias=True,
    potentials=False,
    labels_x=None,
    labels_y=None,
    verbose=False,
    **kwargs,
):
    N, D = x.shape
    M, _ = y.shape

    if cost is None:
        cost = cost_formulas[p], cost_routines[p]
    cost_formula, cost_routine = cost[0], cost[1]

    softmin = partial(
        softmin_multiscale, log_conv=keops_lse(cost_formula, D, dtype=str(x.dtype)[6:])
    )
    extrapolate = partial(extrapolate_samples, softmin=softmin)

    diameter, eps, eps_list, rho = scaling_parameters(
        x, y, p, blur, reach, diameter, scaling
    )

    # Clusterize and sort our point clouds:
    if cluster_scale is None:
        cluster_scale = diameter / (np.sqrt(D) * 2000 ** (1 / D))
    [a_c, a], [x_c, x], [ranges_x], perm_x = clusterize(
        a, x, scale=cluster_scale, labels=labels_x
    )
    [b_c, b], [y_c, y], [ranges_y], perm_y = clusterize(
        b, y, scale=cluster_scale, labels=labels_y
    )

    debug = ranges_x.detach().cpu().numpy()

    jumps = [len(eps_list) - 1]
    for i, eps in enumerate(eps_list[2:]):
        if cluster_scale**p > eps:
            jumps = [i + 1]
            break

    if verbose:
        print(
            "{}x{} clusters, computed at scale = {:2.3f}".format(
                len(x_c), len(y_c), cluster_scale
            )
        )
        print(
            "Successive scales : ",
            ", ".join(["{:.3f}".format(x ** (1 / p)) for x in eps_list]),
        )
        if jumps[0] >= len(eps_list) - 1:
            print("Extrapolate from coarse to fine after the last iteration.")
        else:
            print(
                "Jump from coarse to fine between indices {} (σ={:2.3f}) and {} (σ={:2.3f}).".format(
                    jumps[0],
                    eps_list[jumps[0]] ** (1 / p),
                    jumps[0] + 1,
                    eps_list[jumps[0] + 1] ** (1 / p),
                )
            )

    # The input measures are stored at two levels: coarse and fine
    a_logs = [log_weights(a_c), log_weights(a)]
    b_logs = [log_weights(b_c), log_weights(b)]

    # We do the same [ coarse, fine ] decomposition for "cost matrices",
    # which are implicitely encoded as point clouds
    # + integer summation ranges, and re-computed on-the-fly:
    C_xxs = (
        [
            (x_c, x_c.detach(), ranges_x, ranges_x, None),
            (x, x.detach(), None, None, None),
        ]
        if debias
        else None
    )
    C_yys = (
        [
            (y_c, y_c.detach(), ranges_y, ranges_y, None),
            (y, y.detach(), None, None, None),
        ]
        if debias
        else None
    )
    C_xys = [
        (x_c, y_c.detach(), ranges_x, ranges_y, None),
        (x, y.detach(), None, None, None),
    ]
    C_yxs = [
        (y_c, x_c.detach(), ranges_y, ranges_x, None),
        (y, x.detach(), None, None, None),
    ]

    f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(
        softmin,
        a_logs,
        b_logs,
        C_xxs,
        C_yys,
        C_xys,
        C_yxs,
        eps_list,
        rho,
        jumps=jumps,
        cost=cost_routine,
        kernel_truncation=partial(kernel_truncation, verbose=verbose),
        truncate=truncate,
        extrapolate=extrapolate,
        debias=debias,
    )

    cost = sinkhorn_cost(
        eps, rho, a, b, f_aa, g_bb, g_ab, f_ba, debias=debias, potentials=potentials
    )

    if potentials:  # we should de-sort the vectors of potential values
        F_x, G_y = cost
        f_x, g_y = F_x.clone(), G_y.clone()
        f_x[perm_x], g_y[perm_y] = F_x, G_y
        return f_x, g_y
    else:
        return cost
    


# ==============================================================================
#                          backend == "octree"
# ==============================================================================


def softmin_octree(eps, C_xy, f_y, log_conv=None):
    x, y, ranges_x, ranges_y, ranges_xy = C_xy
    # KeOps is pretty picky on the input shapes...
    return -eps * log_conv(
        x, y, f_y.view(-1, 1), torch.Tensor([1 / eps]).type_as(x), ranges=ranges_xy
    ).view(-1)


def clusterize_octree(a, x, scale=None, labels=None):
    """
    Performs a simple 'voxelgrid' clustering on the input measure,
    putting points into cubic bins of size 'scale' = σ_c.
    The weights are summed, and the centroid position is that of the bin's center of mass.
    Most importantly, the "fine" lists of weights and points are *sorted*
    so that clusters are *contiguous in memory*: this allows us to perform
    kernel truncation efficiently on the GPU.

    If
        [a_c, a], [x_c, x], [x_ranges] = clusterize(a, x, σ_c),
    then
        a_c[k], x_c[k] correspond to
        a[x_ranges[k,0]:x_ranges[k,1]], x[x_ranges[k,0]:x_ranges[k,1],:]
    """
    perm = None  # did we sort the point cloud at some point? Here's the permutation.

    if (
        labels is None and scale is None
    ):  # No clustering, single-scale Sinkhorn on the way...
        return [a], [x], []

    else:  # As of today, only two-scale Sinkhorn is implemented:
        # Compute simple (voxel-like) class labels:
        x_lab = grid_cluster(x, scale) if labels is None else labels
        # Compute centroids and weights:
        ranges_x, x_c, a_c = cluster_ranges_centroids(x, x_lab, weights=a)
        # Make clusters contiguous in memory:
        x_labels, perm = torch.sort(x_lab.view(-1))
        a, x = a[perm], x[perm]

        # N.B.: the lines above were return to replace a call to
        #       'sort_clusters' which does not return the permutation,
        #       an information that is needed to de-permute the dual potentials
        #       if they are required by the user.
        # (a, x), x_labels = sort_clusters( (a,x), x_lab)

        return [a_c, a], [x_c, x], [ranges_x], perm


def kernel_truncation_octree(
    C_xy, C_yx, C_xy_, C_yx_, f_ba, g_ab, eps, truncate=None, cost=None, verbose=False
):
    """Prunes out useless parts of the (block-sparse) cost matrices for finer scales.

    This is where our approximation takes place.
    To be mathematically rigorous, we should make several coarse-to-fine passes,
    making sure that we're not forgetting anyone. A good reference here is
    Bernhard Schmitzer's work: "Stabilized Sparse Scaling Algorithms for
    Entropy Regularized Transport Problems, (2016)".
    """
    if truncate is None:
        return C_xy_, C_yx_
    else:
        x, yd, ranges_x, ranges_y, _ = C_xy
        y, xd, _, _, _ = C_yx
        x_, yd_, ranges_x_, ranges_y_, _ = C_xy_
        y_, xd_, _, _, _ = C_yx_

        with torch.no_grad():
            C = cost(x, y)
            keep = f_ba.view(-1, 1) + g_ab.view(1, -1) > C - truncate * eps
            ranges_xy_ = from_matrix(ranges_x, ranges_y, keep)
            if verbose:
                ks, Cs = keep.sum(), C.shape[0] * C.shape[1]
                print(
                    "Keep {}/{} = {:2.1f}% of the coarse cost matrix.".format(
                        ks, Cs, 100 * float(ks) / Cs
                    )
                )

        return (x_, yd_, ranges_x_, ranges_y_, ranges_xy_), (
            y_,
            xd_,
            ranges_y_,
            ranges_x_,
            swap_axes(ranges_xy_),
        )


def extrapolate_samples_octree(f_ba, g_ab, eps, damping, C_xy, b_log, C_xy_, softmin=None):
    yd = C_xy[1]  # Source points (coarse)
    x_ = C_xy_[0]  # Target points (fine)

    C = (
        x_,
        yd,
        None,
        None,
        None,
    )  # "Rectangular" cost matrix, don't bother with ranges
    return damping * softmin(eps, C, (b_log + g_ab / eps).detach())


def get_octree_clusters(octree, jumps=1, elongate=0, verbose=False):

    depth = len(octree.node_count)
    hierarchy, bounds, metadata, points, colors = octree.to_list()

    # coordinates
    c = []
    # weights
    w = []
    # ranges
    r = []

    # initialize with first eight nodes
    next_metadata_ref = np.argwhere(metadata[:, -1] == 1).ravel()

    # save a counter on how often we hold back nodes for elongation
    elongate_counter = 0
    
    while True:
        
        elongated_this_iteration = False
        # don't elongate in the first iteration
        if len(next_metadata_ref) > 8:
            if elongate > 0 and elongate_counter < elongate:
                # take half
                # alternate which half to take
                if elongate_counter % 2 == 0:
                    elongate_metadata_ref = next_metadata_ref[:len(next_metadata_ref) // 2]
                    next_metadata_ref = next_metadata_ref[len(next_metadata_ref) // 2:]
                else:
                    elongate_metadata_ref = next_metadata_ref[len(next_metadata_ref) // 2:]
                    next_metadata_ref = next_metadata_ref[:len(next_metadata_ref) // 2]
                elongated_this_iteration = True
                elongate_counter += 1

        # to index metadata
        current_metadata_ref = next_metadata_ref

        # handle leafs
        is_leaf = metadata[current_metadata_ref, 1] == 0

        # to index hierarchy
        current_hierarchy_ref = metadata[current_metadata_ref, 1]

        # the next hierarchy ref, for non leafs, is according to the hierarchy array
        next_metadata_ref = np.zeros((len(current_metadata_ref), 8), dtype=np.int64)
        next_metadata_ref[~is_leaf] = np.r_[hierarchy[current_hierarchy_ref[~is_leaf]]]
        # for leafs we need the same node again, so we add a row with the index and then 7 zeros 
        carry_on = np.zeros((np.count_nonzero(is_leaf), 8))
        carry_on[:, 0] = current_metadata_ref[is_leaf]
        next_metadata_ref[is_leaf] = carry_on

        # compute ranges from the nonzero childs per node
        childs_per_node = np.count_nonzero(next_metadata_ref, axis=1)
        ranges = np.stack((np.insert(np.cumsum(childs_per_node[:-1]), 0, 0), np.cumsum(childs_per_node)), axis=1)
        # todo ?
        #ranges[-1, 1] -= 1
        # remove empty leafs
        next_metadata_ref = next_metadata_ref[next_metadata_ref != 0]

        # if we elongated we have to add the second half of the hierarchy_ref again
        if elongated_this_iteration:
            if elongate_counter % 2 == 0:
                next_metadata_ref = np.r_[elongate_metadata_ref, next_metadata_ref]
            else:
                next_metadata_ref = np.r_[next_metadata_ref, elongate_metadata_ref]


        ##### Finally add the centroids, weights and computed ranges

        # get centroids and weights
        # get centroids 
        selection = bounds[current_metadata_ref]
        # todo change to center of mass 
        centroids = np.r_[[selection[:, 0] + ((selection[:, 3] - selection[:, 0]) / 2.0),
                        selection[:, 1] + ((selection[:, 4] - selection[:, 1]) / 2.0),
                       selection[:, 2] + ((selection[:, 5] - selection[:, 2]) / 2.0)]].T

        # get weights
        pointcount = metadata[current_metadata_ref, 0]

        # to torch 
        centroids = torch.tensor(centroids, dtype=torch.float32, device='cuda')
        weights = torch.tensor(pointcount, dtype=torch.float32, device='cuda')
        weights = weights / weights.sum()
        ranges = torch.tensor(ranges, dtype=torch.int32, device='cuda')

        weights, centroids, ranges = weights.contiguous(), centroids.contiguous(), ranges.contiguous()
        
        c.append(centroids)
        w.append(weights)
        r.append(ranges)

        if np.count_nonzero(~is_leaf) == 0 and not elongated_this_iteration:
            # extrapolate to the points
            break

    # change the last ranges where all nodes are leafs to the point ranges
    point_ref = metadata[next_metadata_ref, 2]
    point_count = metadata[next_metadata_ref, 0]
    point_range = np.stack((point_ref, point_ref + point_count), axis=1)
    r[-1] = torch.tensor(point_range, dtype=torch.int32, device='cuda').contiguous()

    # append the last level e.g. the points themselves
    centroids = torch.tensor(points, dtype=torch.float32, device='cuda')
    weights = torch.tensor(np.ones(len(centroids)), dtype=torch.float32, device='cuda')
    weights = weights / weights.sum()
    weights, centroids = weights.contiguous(), centroids.contiguous()
    c.append(centroids)
    w.append(weights)
    r.append(None)

    if verbose:
        print(f"Number of clusters per scale: {[len(cluster) for cluster in c]}")

    return c, w, r 


def sinkhorn_octree(
    octree_x,
    octree_y,
    p=2,
    blur=0.05,
    reach=None,
    diameter=None,
    scaling=0.5,
    truncate=5,
    cost=None,
    cluster_scale=None,
    debias=True,
    potentials=False,
    labels_x=None,
    labels_y=None,
    verbose=False,
    **kwargs,
):
    #todo 
    D = 3
    _type = torch.float32
    #N, D = x.shape
    #M, _ = y.shape

    if cost is None:
        cost = cost_formulas[p], cost_routines[p]
    cost_formula, cost_routine = cost[0], cost[1]

    softmin = partial(
        softmin_octree, log_conv=keops_lse(cost_formula, D, dtype=str(_type)[6:])
    )
    extrapolate = partial(extrapolate_samples_octree, softmin=softmin)

    #diameter, eps, eps_list, rho = scaling_parameters(
    #    x, y, p, blur, reach, diameter, scaling
    #)
    # todo 
    diameter = np.sqrt(3)
    eps = blur**p
    rho = None if reach is None else reach**p
    eps_list = epsilon_schedule(p, diameter, blur, scaling)

    # Clusterize and sort our point clouds:
    # ensure that these have the same length 
    depth_x = len(octree_x.node_count)
    depth_y = len(octree_y.node_count)
    verbose = True
    jumps = 1
    if depth_x != depth_y:
        diff = abs(depth_x - depth_y)
        if depth_x > depth_y:
            c_x, w_x, r_x = get_octree_clusters(octree_x, jumps=jumps, verbose=verbose)
            c_y, w_y, r_y = get_octree_clusters(octree_y, jumps=jumps, elongate = diff, verbose=verbose)
        else:
            c_x, w_x, r_x = get_octree_clusters(octree_x, jumps=jumps, elongate = diff, verbose=verbose)
            c_y, w_y, r_y = get_octree_clusters(octree_y, jumps=jumps, verbose=verbose)
    else:
        c_x, w_x, r_x = get_octree_clusters(octree_x, jumps=jumps, verbose=verbose)
        c_y, w_y, r_y = get_octree_clusters(octree_y, jumps=jumps, verbose=verbose)

    # jumps
    #if len(c_x) == 2:
    #    jumps = [len(eps_list) - (len(eps_list) // 3)]
    #else:
    #    sublist_size = len(eps_list) // (len(c_x) - 1)
    #    remainder = len(eps_list) % (len(c_x) - 1) # to handle the case where the division isn't perfect
    #    jumps = np.arange(sublist_size, len(eps_list), sublist_size)
    #    jumps[0] += remainder
    
    #min_scale = lambda x, n, m: n - (x**2 / 0.5**m)
#
    #cluster_scales = np.logspace(0.0001, 0.2, 100) - 1
    #dbgscales = [min_scale(x, eps_list[-5], len(c_x)) for x in cluster_scales]
#
    #possible_scales = np.argwhere(np.r_[dbgscales] > 0).ravel()
    #cluster_scale = cluster_scales[possible_scales[-1]]
#
    #jumps = []
    #for i, eps in enumerate(eps_list[2:]):
    #    if cluster_scale**p > eps:
    #        jumps.append(i + 1)
    #        cluster_scale /= 2

    if verbose:
        print(f"eps: {len(eps_list)}")

    if True:
        jumps = []
        jump_count = 0  # Keep track of the number of jumps
        i = 0           # Start from the first index in eps_list[2:]
        cluster_scale = 0.5
        # Loop through eps_list starting from the 3rd element (index 2)
        while jump_count < len(c_x)-1 and i < len(eps_list) - 5:
            eps = eps_list[i + 2]
            
            # Check the condition for a jump
            if cluster_scale**p > eps:
                jumps.append(i + 1)  # Append the jump index (i + 1 because we're using eps_list[2:])
                cluster_scale /= 2   # Update cluster scale
                jump_count += 1      # Increment the jump counter
            
            i += 1  # Move to the next index

        # If there are still fewer than n jumps and we have more elements, append remaining jumps
        # But only append as many as there are available in eps_list
        while jump_count < len(c_x)-1 and i < len(eps_list) - 2:
            jumps.append(i + 1)
            i += 1
            jump_count += 1

        # todo dont forget!
    #    jumps[-1] = len(eps_list) - 1
        #jumps[-1] = jumps[-2] + 10

        if verbose:
            print(f"Jumps: {jumps}")

    else:
        #jumps = [3,6,15,33,40]
        #jumps = [3,6,15,33,len(eps_list) - 2]

        jumps = [6, 13, 19, 26, 33, 39, 46, 53]



    #if verbose:
    #    print(
    #        "{}x{} clusters, computed at scale = {:2.3f}".format(
    #            len(x_c), len(y_c), cluster_scale
    #        )
    #    )
    #    print(
    #        "Successive scales : ",
    #        ", ".join(["{:.3f}".format(x ** (1 / p)) for x in eps_list]),
    #    )
    #    if jumps[0] >= len(eps_list) - 1:
    #        print("Extrapolate from coarse to fine after the last iteration.")
    #    else:
    #        print(
    #            "Jump from coarse to fine between indices {} (σ={:2.3f}) and {} (σ={:2.3f}).".format(
    #                jumps[0],
    #                eps_list[jumps[0]] ** (1 / p),
    #                jumps[0] + 1,
    #                eps_list[jumps[0] + 1] ** (1 / p),
    #            )
    #        )

    # The input measures are stored at multiple levels
    a_logs = [log_weights(a) for a in w_x]
    b_logs = [log_weights(b) for b in w_y]

    # We do the same [ coarse, fine ] decomposition for "cost matrices",
    # which are implicitely encoded as point clouds
    # + integer summation ranges, and re-computed on-the-fly:
    if debias:
        C_xxs = []
        for i in range(0, len(c_x) - 1):
            C_xxs.append((c_x[i], c_x[i].detach(), r_x[i], r_x[i], None))
        C_xxs.append((c_x[-1], c_x[-1].detach(), None, None, None))
    else:
        C_xxs = None

    if debias:
        C_yys = []
        for i in range(0, len(c_y) - 1):
            C_yys.append((c_y[i], c_y[i].detach(), r_y[i], r_y[i], None))
        C_yys.append((c_y[-1], c_y[-1].detach(), None, None, None))
    else:
        C_yys = None

    #C_xxs = (
    #    [
    #        (x_c, x_c.detach(), ranges_x, ranges_x, None),
    #        (x, x.detach(), None, None, None),
    #    ]
    #    if debias
    #    else None
    #)

    C_xys = []
    for i in range(0, len(c_x) - 1):
        C_xys.append((c_x[i], c_y[i].detach(), r_x[i], r_y[i], None))
    C_xys.append((c_x[-1], c_y[-1].detach(), None, None, None))

    #C_xys = [
    #    (x_c, y_c.detach(), ranges_x, ranges_y, None),
    #    (x, y.detach(), None, None, None),
    #]

    C_yxs = []
    for i in range(0, len(c_y) - 1):
        C_yxs.append((c_y[i], c_x[i].detach(), r_y[i], r_x[i], None))
    C_yxs.append((c_y[-1], c_x[-1].detach(), None, None, None))

    #C_yxs = [
    #    (y_c, x_c.detach(), ranges_y, ranges_x, None),
    #    (y, x.detach(), None, None, None),
    #]

    f_aa, g_bb, g_ab, f_ba = sinkhorn_loop(
        softmin,
        a_logs,
        b_logs,
        C_xxs,
        C_yys,
        C_xys,
        C_yxs,
        eps_list,
        rho,
        jumps=jumps,
        cost=cost_routine,
        kernel_truncation=partial(kernel_truncation, verbose=verbose),
        truncate=truncate,
        extrapolate=extrapolate,
        debias=debias,
    )

    # todo check 
    a = w_x[-1]
    b = w_y[-1]

    cost = sinkhorn_cost(
        eps, rho, a, b, f_aa, g_bb, g_ab, f_ba, debias=debias, potentials=potentials
    )

    if potentials:  # we should de-sort the vectors of potential values
        F_x, G_y = cost
        f_x, g_y = F_x.clone(), G_y.clone()

        print(f"f_x non finite:  {len(f_x) - torch.count_nonzero(torch.isfinite(f_x))}")
        print(f"g_y non finite:  {len(g_y) - torch.count_nonzero(torch.isfinite(g_y))}")

        # todo check 
        #f_x[perm_x], g_y[perm_y] = F_x, G_y
        return f_x, g_y, c_x[-1], c_y[-1]
    else:
        return cost
