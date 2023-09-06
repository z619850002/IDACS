import torch
from ipdb import set_trace as S
import numpy as np

def sample_uniform(z_values):
    # For example: near=0, far=4, n=5
    # z_values: [0, 1, 2, 3, 4]
    bins = 0.5 *(z_values[:,:-1] + z_values[:,1:]) # (B, N-1) interval mid points
    # bins: [0.5, 1.5, 2.5, 3.5]
    lower = torch.cat([z_values[:,:1], bins], -1) # (B,N)
    # lower: [0, 0.5, 1.5, 2.5, 3.5]
    upper = torch.cat([bins, z_values[:,-1:]], -1) # (B,N)
    # upper: [0.5, 1.5, 2.5, 3.5, 4]
    z_values = lower + (upper - lower) * torch.rand(z_values.shape, device=z_values.device)
    # ranges of each number
    # [0, 0.5)
    # [0.5, 1.5)
    # [1.5, 2.5)
    # [2.5, 3.5)
    # [3.5, 4)
    return z_values

# from nerf_pl
def sample_inverse(z_values, weight, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weight.

    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weight: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero

    Outputs:
        samples: the sampled samples
    """
    bins = 0.5 * (z_values[: ,:-1] + z_values[: ,1:])
    N_rays, N_samples_ = weight.shape
    weight = weight + eps # prevent division by zero (don't do inplace op!)
    pdf = weight / torch.sum(weight, -1, keepdim=True) # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, side='right')
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    # stack instead of cat because shape after catting is (2*N_importacne)
    inds_sampled = torch.stack([below, above], -1).view(N_rays, N_importance*2)
    cdf_g = torch.gather(cdf, 1, inds_sampled).view(N_rays, N_importance, 2)
    bins_g = torch.gather(bins, 1, inds_sampled).view(N_rays, N_importance, 2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples

    get_ray_directions(300,400,1)