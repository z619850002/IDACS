import torch
from torch import nn
from ipdb import set_trace as S
from model import *
from utils import *
from time import time
import original_nerf
from torchsearchsorted import searchsorted




def volume_rendering(z_values, sigma, norm=None, function="sigma", model=None):
    # sigma: (B,N)
    delta = z_values[:, 1:] - z_values[:, :-1] #(B,N-1)
    delta = torch.cat((delta, torch.full_like(delta[:, 0:1], 1e10)), 1)#(B,N)
    # Multiply each distance by the norm of its corresponding direction ray
    # to convert to real world distance (accounts for non-unit directions).
    if norm is not None:
        delta = delta * norm #(B,N)
    if function == "SDF":
        sigma = sdf_to_sigma(sigma, *model.forward_ab())
    alpha = 1-torch.exp(-sigma*delta) #(B,N)
    # T_i = a_1 * ... * a_(i-1)
    # and T_1 = e^0 = 1
    alpha_shift = torch.cat([torch.ones_like(alpha[:, :1]), 1-alpha[:, :-1]+1e-10], -1) #(B,N)
    T = torch.cumprod(alpha_shift, -1) #(B,N)
    weight = T*alpha #(B,N)

    return weight

    
class VanillaSampling(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.coarse = eval(f"{hparams.Model}RadianceField")(hparams)
        self.fine = eval(f"{hparams.Model}RadianceField")(hparams)
        self.N_samples = hparams.N_samples
        self.N_importance_samples = hparams.N_importance_samples
        # self.white_background = hparams.white_background
        self.noise = hparams.noise
        self.function = hparams.Function
        self.near, self.far = hparams.near, hparams.far
        
    def forward(self, rays, **kwargs):
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (B, 3)
        norm = torch.norm(rays_d, dim=-1, keepdim=True) #(B,3)->(B,1) all 1 when unit direction
        
        result = {}
        ######## coarse sampling, yielding weight for fine (no color-rending when test mode)
        z_steps = torch.linspace(0, 1, self.N_samples, device=rays.device) # (N)
        z_values_coarse = (self.near + (self.far-self.near)*z_steps).unsqueeze(0).expand(N_rays, -1) # (B, N)
        z_values_coarse = sample_uniform(z_values_coarse) # (B, N)
        # (B,1,3) * (B,N,1) -> B,N,3
        xyz_samples_coarse = (rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_values_coarse.unsqueeze(2)).reshape(-1, 3) # (B*N, 3)
        if self.training:
            dir_samples_coarse = torch.repeat_interleave(rays_d, repeats=self.N_samples, dim=0) # (B*N, 3)
            color, sigma = self.coarse(xyz_samples_coarse, dir_samples_coarse)
            sigma = sigma.reshape(N_rays, self.N_samples)
            color = color.reshape(N_rays, self.N_samples, 3)
        else: # level=="coarse" and mode=="test"
            sigma = self.coarse(xyz_samples_coarse).reshape(N_rays, self.N_samples)
        noise = torch.randn(sigma.shape, device=sigma.device) if self.noise else 0
        sigma = torch.relu(sigma + noise)
        weight = volume_rendering(z_values_coarse, sigma, norm, self.function, self.coarse)
        if self.training:
            result["colors_coarse"] = (weight.unsqueeze(-1) * color).sum(1) # (B, 3)
            # if self.white_background:
            #     result["colors_coarse"] = result["colors_coarse"] + 1-weight.sum(1, keepdim=True)
        ######## fine sampling
        if self.N_importance_samples > 0:
            z_steps = torch.linspace(0, 1, self.N_importance_samples, device=rays.device) # (N)
            z_values_fine = (self.near + (self.far-self.near)* z_steps).unsqueeze(0).expand(N_rays, -1) # (B, N)
            z_values_fine = sample_inverse(z_values_fine, weight[:, 1:-1], self.N_importance_samples).detach() #B, N
            z_values_fine = torch.sort(torch.cat([z_values_coarse, z_values_fine], -1), -1)[0]
            xyz_samples_fine = (rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_values_fine.unsqueeze(2)).reshape(-1, 3) # (B*N, 3)
            dir_samples_fine = torch.repeat_interleave(rays_d, repeats=self.N_importance_samples+self.N_samples, dim=0) # (B*N, 3)
            color, sigma = self.fine(xyz_samples_fine, dir_samples_fine)
            sigma = sigma.reshape(N_rays, self.N_importance_samples+self.N_samples)
            color = color.reshape(N_rays, self.N_importance_samples+self.N_samples, 3)
            noise = torch.randn(sigma.shape, device=sigma.device) if self.noise else 0
            sigma = torch.relu(sigma + noise)
            weight = volume_rendering(z_values_fine, sigma, norm, self.function, self.fine)
            result["colors"] = (weight.unsqueeze(-1) * color).sum(1) # (B, 3)
            result['depths'] = (weight * z_values_fine).sum(1) # (B, 1)
            # if self.white_background:
            #     result["colors"] = result["colors"] + 1-weight.sum(1, keepdim=True)
        
        return result


class VanillaSamplingCanonical2(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.coarse = CanonicalMLPRadianceField(hparams)
        # self.fine = CanonicalMLPRadianceField(hparams)
        self.N_samples = hparams.N_samples
        self.N_importance_samples = hparams.N_importance_samples
        # self.white_background = hparams.white_background
        self.noise = hparams.noise
        self.function = hparams.Function
        self.near, self.far = hparams.near, hparams.far



    def forward(self, rays, indices, max_index, **kwargs):
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (B, 3)
        ts = indices.float() / max_index

        norm = torch.norm(rays_d, dim=-1, keepdim=True)  # (B,3)->(B,1) all 1 when unit direction

        result = {}
        ######## coarse sampling, yielding weight for fine (no color-rending when test mode)
        z_steps = torch.linspace(0, 1, self.N_samples, device=rays.device)  # (N)
        z_values_coarse = (self.near + (self.far - self.near) * z_steps).unsqueeze(0).expand(N_rays, -1)  # (B, N)
        z_values_coarse = sample_uniform(z_values_coarse)  # (B, N)
        # (B,1,3) * (B,N,1) -> B,N,3
        xyz_samples_coarse = (rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_values_coarse.unsqueeze(2)).reshape(-1,
                                                                              3)  # (B*N, 3)

        train_times = torch.ones_like(xyz_samples_coarse[:,:1])  * ts[0]
        xyz_samples_coarse = torch.cat([xyz_samples_coarse, train_times], dim=-1)

        with torch.no_grad():
            if self.training:
                dir_samples_coarse = torch.repeat_interleave(rays_d, repeats=self.N_samples, dim=0)  # (B*N, 3)
                color, sigma = self.coarse(xyz_samples_coarse, dir_samples_coarse)
                sigma = sigma.reshape(N_rays, self.N_samples)
                color = color.reshape(N_rays, self.N_samples, 3)
            else:  # level=="coarse" and mode=="test"
                sigma = self.coarse(xyz_samples_coarse).reshape(N_rays, self.N_samples)
            noise = torch.randn(sigma.shape, device=sigma.device) if self.noise else 0
            sigma = torch.relu(sigma + noise)
            weight = volume_rendering(z_values_coarse, sigma, norm, self.function, self.coarse)
            # if self.training:
            #     result["colors_coarse"] = (weight.unsqueeze(-1) * color).sum(1)  # (B, 3)
        # if self.white_background:
        #     result["colors_coarse"] = result["colors_coarse"] + 1-weight.sum(1, keepdim=True)
        ######## fine sampling
        if self.N_importance_samples > 0:
            z_steps = torch.linspace(0, 1, self.N_importance_samples, device=rays.device)  # (N)
            z_values_fine = (self.near + (self.far - self.near) * z_steps).unsqueeze(0).expand(N_rays, -1)  # (B, N)
            z_values_fine = sample_inverse(z_values_fine, weight[:, 1:-1], self.N_importance_samples).detach()  # B, N
            z_values_fine = torch.sort(torch.cat([z_values_coarse, z_values_fine], -1), -1)[0]
            xyz_samples_fine = (rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_values_fine.unsqueeze(2)).reshape(-1,
                                                                                                                3)  # (B*N, 3)
            dir_samples_fine = torch.repeat_interleave(rays_d, repeats=self.N_importance_samples + self.N_samples,
                                                       dim=0)  # (B*N, 3)
            xyz_samples_fine = torch.cat([xyz_samples_fine, torch.ones_like(xyz_samples_fine[:, :1]) * ts[0]],dim=-1)
            color, sigma = self.coarse(xyz_samples_fine, dir_samples_fine)
            # color, sigma = self.coarse(xyz_samples_fine, dir_samples_fine)

            sigma = sigma.reshape(N_rays, self.N_importance_samples + self.N_samples)
            color = color.reshape(N_rays, self.N_importance_samples + self.N_samples, 3)
            noise = torch.randn(sigma.shape, device=sigma.device) if self.noise else 0
            sigma = torch.relu(sigma + noise)
            weight = volume_rendering(z_values_fine, sigma, norm, self.function, self.coarse)
            result["colors"] = (weight.unsqueeze(-1) * color).sum(1)  # (B, 3)
            result['depths'] = (weight * z_values_fine).sum(1)  # (B, 1)

            # if self.training:
            #     # compute TV loss
            #
            #     positions = xyz_samples_fine[:,:3]
            #     times = xyz_samples_fine[:,-1:]
            #     result['space'] = self.coarse.query_time(positions,  times)
            #
            #     time_interval = 0.02
            #     previous_times = times - time_interval
            #     previous_times[previous_times < 0.0] = 0.0
            #     result['pre_space'] = self.coarse.query_time(positions, previous_times)
            #     del previous_times
            #
            #     next_time = times + time_interval
            #     del times
            #     next_time[next_time > 1.0] = 1.0
            #     result['next_spaces'] = self.coarse.query_time(positions, next_time)
            #     del positions
            #     del next_time

            # if self.white_background:
            #     result["colors"] = result["colors"] + 1-weight.sum(1, keepdim=True)

        return result


# Hierarchical sampling (section 5.2)
def sample_pdf(bins, weights, N_samples, device, det=False, pytest=False):
    # Get pdf
    weights = weights + 1e-5 # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[...,:1]), cdf], -1)  # (batch, len(bins))

    # Take uniform samples
    if det:
        u = torch.linspace(0., 1., steps=N_samples)
        u = u.expand(list(cdf.shape[:-1]) + [N_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [N_samples])

    u = u.to(device)

    # Invert CDF
    u = u.contiguous()
    inds = searchsorted(cdf, u, side='right')
    below = torch.max(torch.zeros_like(inds-1), inds-1)
    above = torch.min((cdf.shape[-1]-1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    # cdf_g = tf.gather(cdf, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    # bins_g = tf.gather(bins, inds_g, axis=-1, batch_dims=len(inds_g.shape)-2)
    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[...,1]-cdf_g[...,0])
    denom = torch.where(denom<1e-5, torch.ones_like(denom), denom)
    t = (u-cdf_g[...,0])/denom
    samples = bins_g[...,0] + t * (bins_g[...,1]-bins_g[...,0])

    return samples




def raw2outputs(raw, z_vals, rays_d, device, raw_noise_std=0, white_bkgd=False, pytest=False):
    """Transforms model's predictions to semantically meaningful values.
    Args:
        raw: [num_rays, num_samples along ray, 4]. Prediction from model.
        z_vals: [num_rays, num_samples along ray]. Integration time.
        rays_d: [num_rays, 3]. Direction of each ray.
    Returns:
        rgb_map: [num_rays, 3]. Estimated RGB color of a ray.
        disp_map: [num_rays]. Disparity map. Inverse of depth map.
        acc_map: [num_rays]. Sum of weights along each ray.
        weights: [num_rays, num_samples]. Weights assigned to each sampled color.
        depth_map: [num_rays]. Estimated distance to object.
    """
    # torch.set_default_tensor_type('torch.cuda.FloatTensor')
    raw2alpha = lambda raw, dists, act_fn=F.relu: 1.-torch.exp(-act_fn(raw)*dists)

    dists = z_vals[...,1:] - z_vals[...,:-1]
    dists = torch.cat([dists, torch.Tensor([1e10]).to(device).expand(dists[...,:1].shape)], -1)  # [N_rays, N_samples]

    dists = dists * torch.norm(rays_d[...,None,:], dim=-1)

    rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
    noise = 0.
    if raw_noise_std > 0.:
        noise = torch.randn(raw[...,3].shape) * raw_noise_std

        # Overwrite randomly sampled data if pytest
        # if pytest:
        #     np.random.seed(0)
        #     noise = np.random.rand(*list(raw[...,3].shape)) * raw_noise_std
        #     noise = torch.Tensor(noise)

    alpha = raw2alpha(raw[...,3] + noise, dists)  # [N_rays, N_samples]
    # weights = alpha * tf.math.cumprod(1.-alpha + 1e-10, -1, exclusive=True)
    weights = alpha * torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1)).to(device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3]

    depth_map = torch.sum(weights * z_vals, -1)
    disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1))
    acc_map = torch.sum(weights, -1)

    if white_bkgd:
        rgb_map = rgb_map + (1.-acc_map[...,None])
        # rgb_map = rgb_map + torch.cat([acc_map[..., None] * 0, acc_map[..., None] * 0, (1. - acc_map[..., None])], -1)

    return rgb_map, disp_map, acc_map, weights, depth_map
# rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(raw, z_vals, rays_d, raw_noise_std, white_bkgd, pytest=pytest)








class VanillaSamplingCanonical(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.coarse, self.embedtime_fn, self.embed_fn, self.embeddirs_fn, self.input_ch, self.input_ch_time, self.input_ch_view = original_nerf.create_nerf(self.device)
        # self.fine = CanonicalMLPRadianceField(hparams)
        self.N_samples = hparams.N_samples
        self.N_importance_samples = hparams.N_importance_samples
        # self.white_background = hparams.white_background
        self.noise = hparams.noise
        self.function = hparams.Function
        self.near, self.far = hparams.near, hparams.far

    def forward(self, rays, indices, max_index, **kwargs):
        N_rays = rays.shape[0]
        B = len(rays)
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (B, 3)
        ts = indices.float() / max_index

        norm = torch.norm(rays_d, dim=-1, keepdim=True)  # (B,3)->(B,1) all 1 when unit direction

        result = {}
        ######## coarse sampling, yielding weight for fine (no color-rending when test mode)
        z_steps = torch.linspace(0, 1, self.N_samples, device=rays.device)  # (N)
        z_values_coarse = (self.near + (self.far - self.near) * z_steps).unsqueeze(0).expand(N_rays, -1)  # (B, N)
        z_values_coarse = sample_uniform(z_values_coarse)  # (B, N)
        # (B,1,3) * (B,N,1) -> B,N,3
        xyz_samples_coarse = (rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_values_coarse.unsqueeze(2)).reshape(-1,
                                                                                                                3)  # (B*N, 3)

        train_times = torch.ones_like(xyz_samples_coarse[:, :1]) * ts[0]

        embed_xyz_samples_coarse = self.embed_fn(xyz_samples_coarse)
        embed_train_times = self.embedtime_fn(train_times)

        with torch.no_grad():
            if self.training:
                dir_samples_coarse = torch.repeat_interleave(rays_d, repeats=self.N_samples, dim=0)  # (B*N, 3)
                embed_dir_samples_coarse = self.embeddirs_fn(dir_samples_coarse)
                embed_x = torch.cat([embed_xyz_samples_coarse, embed_dir_samples_coarse], dim = -1)
                color_and_sigma = self.coarse(embed_x, embed_train_times)
                sigma = color_and_sigma[:,3:]
                color = color_and_sigma[:, :3]
                sigma = sigma.reshape(N_rays, self.N_samples)
                color = color.reshape(N_rays, self.N_samples, 3)
                color = torch.sigmoid(color)
            else:  # level=="coarse" and mode=="test"
                dir_samples_coarse = torch.repeat_interleave(rays_d, repeats=self.N_samples, dim=0)  # (B*N, 3)
                embed_dir_samples_coarse = self.embeddirs_fn(dir_samples_coarse)
                embed_x = torch.cat([embed_xyz_samples_coarse, embed_dir_samples_coarse], dim=-1)
                color_and_sigma = self.coarse(embed_x, embed_train_times)
                sigma = color_and_sigma[:, 3:]
                # color = color_and_sigma[:, :3]
                # color = torch.sigmoid(color)
                sigma = sigma.reshape(N_rays, self.N_samples)
            noise = torch.randn(sigma.shape, device=sigma.device) if self.noise else 0
            sigma = torch.relu(sigma + noise)
            weight = volume_rendering(z_values_coarse, sigma, norm, self.function, self.coarse)
            # color_and_sigma_reshaped = color_and_sigma.reshape([N_rays, self.N_samples, -1])
            # _, _, _, weights, _ = raw2outputs(color_and_sigma_reshaped, z_values_coarse, rays_d, rays.device, 0.0, False, pytest=False)
            #
            # z_vals_mid = .5 * (z_values_coarse[...,1:] + z_values_coarse[...,:-1])
            # z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], self.N_importance_samples, device=rays.device, det=False, pytest=False)
            # z_samples = z_samples.detach()
            # z_vals, _ = torch.sort(torch.cat([z_values_coarse, z_samples], -1), -1)



            # if self.training:
            #     result["colors_coarse"] = (weight.unsqueeze(-1) * color).sum(1)  # (B, 3)
        # if self.white_background:
        #     result["colors_coarse"] = result["colors_coarse"] + 1-weight.sum(1, keepdim=True)

        del embed_xyz_samples_coarse
        del embed_train_times
        del embed_dir_samples_coarse


        ######## fine sampling
        if self.N_importance_samples > 0:
            z_steps = torch.linspace(0, 1, self.N_importance_samples, device=rays.device)  # (N)
            z_values_fine = (self.near + (self.far - self.near) * z_steps).unsqueeze(0).expand(N_rays, -1)  # (B, N)
            z_values_fine = sample_inverse(z_values_fine, weight[:, 1:-1], self.N_importance_samples).detach()  # B, N

            # z_values_fine = sample_inverse(z_values_coarse, weight[:, 1:-1], self.N_importance_samples).detach()



            z_values_fine = torch.sort(torch.cat([z_values_coarse, z_values_fine], -1), -1)[0]
            xyz_samples_fine = (rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_values_fine.unsqueeze(2)).reshape(-1,
                                                                                                                3)  # (B*N, 3)
            dir_samples_fine = torch.repeat_interleave(rays_d, repeats=self.N_importance_samples + self.N_samples,
                                                       dim=0)  # (B*N, 3)

            #embed
            embed_x_fine = self.embed_fn(xyz_samples_fine)
            embed_dir_samples_fine = self.embeddirs_fn(dir_samples_fine)
            # xyz_samples_fine = torch.cat([xyz_samples_fine, torch.ones_like(xyz_samples_fine[:, :1]) * ts[0]], dim=-1)
            embed_x_fine = torch.cat([embed_x_fine, embed_dir_samples_fine], dim=-1)
            del embed_dir_samples_fine
            train_times_fine = torch.ones_like(xyz_samples_fine[:, :1]) * ts[0]
            embed_train_times_fine = self.embedtime_fn(train_times_fine)
            color_and_sigma = self.coarse(embed_x_fine, embed_train_times_fine)
            # color, sigma = self.coarse(xyz_samples_fine, dir_samples_fine)
            # color, sigma = self.coarse(xyz_samples_fine, dir_samples_fine)
            color_and_sigma_reshaped = color_and_sigma.reshape([N_rays, self.N_samples + self.N_importance_samples,-1])
            rgb_map, disp_map, acc_map, weights, depth_map = raw2outputs(color_and_sigma_reshaped, z_values_fine, rays_d, rays.device, 0.0, False,
                                                                         pytest=False)

            # sigma = color_and_sigma[:, 3:]
            # color = color_and_sigma[:, :3]
            #
            # sigma = sigma.reshape(N_rays, self.N_importance_samples + self.N_samples)
            # color = color.reshape(N_rays, self.N_importance_samples + self.N_samples, 3)
            # color = torch.sigmoid(color)
            # noise = torch.randn(sigma.shape, device=sigma.device) if self.noise else 0
            # sigma = torch.relu(sigma + noise)
            # weight = volume_rendering(z_values_fine, sigma, norm, self.function, self.coarse)
            # result["colors"] = (weight.unsqueeze(-1) * color).sum(1)  # (B, 3)
            # result['depths'] = (weight * z_values_fine).sum(1)  # (B, 1)

            result["colors"] = rgb_map
            result['depths'] = depth_map


        return result