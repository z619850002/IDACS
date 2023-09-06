from einops import rearrange
import vren
import numpy as np
import torch
from torch.cuda.amp import custom_fwd, custom_bwd
from torch_scatter import segment_csr
from torch import nn
import tinycudann as tcnn
from kornia.utils.grid import create_meshgrid3d
from model import batchify

class RayAABBIntersector(torch.autograd.Function):
    """
    Computes the intersections of rays and axis-aligned voxels.

    Inputs:
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions
        centers: (N_voxels, 3) voxel centers
        half_sizes: (N_voxels, 3) voxel half sizes
        max_hits: maximum number of intersected voxels to keep for one ray
                  (for a cubic scene, this is at most 3*N_voxels^(1/3)-2)

    Outputs:
        hits_cnt: (N_rays) number of hits for each ray
        (followings are from near to far)
        hits_t: (N_rays, max_hits, 2) hit t's (-1 if no hit)
        hits_voxel_idx: (N_rays, max_hits) hit voxel indices (-1 if no hit)
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, center, half_size, max_hits):
        return vren.ray_aabb_intersect(rays_o, rays_d, center, half_size, max_hits)


class RaySphereIntersector(torch.autograd.Function):
    """
    Computes the intersections of rays and spheres.

    Inputs:
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) ray directions
        centers: (N_spheres, 3) sphere centers
        radii: (N_spheres, 3) radii
        max_hits: maximum number of intersected spheres to keep for one ray

    Outputs:
        hits_cnt: (N_rays) number of hits for each ray
        (followings are from near to far)
        hits_t: (N_rays, max_hits, 2) hit t's (-1 if no hit)
        hits_sphere_idx: (N_rays, max_hits) hit sphere indices (-1 if no hit)
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, center, radii, max_hits):
        return vren.ray_sphere_intersect(rays_o, rays_d, center, radii, max_hits)


class RayMarcher(torch.autograd.Function):
    """
    March the rays to get sample point positions and directions.

    Inputs:
        rays_o: (N_rays, 3) ray origins
        rays_d: (N_rays, 3) normalized ray directions
        hits_t: (N_rays, 2) near and far bounds from aabb intersection
        density_bitfield: (C*G**3//8)
        cascades: int
        scale: float
        exp_step_factor: the exponential factor to scale the steps
        grid_size: int
        max_samples: int

    Outputs:
        rays_a: (N_rays) ray_idx, start_idx, N_samples
        xyzs: (N, 3) sample positions
        dirs: (N, 3) sample view directions
        deltas: (N) dt for integration
        ts: (N) sample ts
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, rays_o, rays_d, hits_t,
                density_bitfield, cascades, scale, exp_step_factor,
                grid_size, max_samples):
        # noise to perturb the first sample of each ray
        noise = torch.rand_like(rays_o[:, 0])

        rays_a, xyzs, dirs, deltas, ts, counter = \
            vren.raymarching_train(
                rays_o, rays_d, hits_t,
                density_bitfield, cascades, scale,
                exp_step_factor, noise, grid_size, max_samples)

        total_samples = counter[0] # total samples for all rays
        # remove redundant output
        xyzs = xyzs[:total_samples]
        dirs = dirs[:total_samples]
        deltas = deltas[:total_samples]
        ts = ts[:total_samples]

        ctx.save_for_backward(rays_a, ts)

        return rays_a, xyzs, dirs, deltas, ts, total_samples

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_drays_a, dL_dxyzs, dL_ddirs,
                 dL_ddeltas, dL_dts, dL_dtotal_samples):
        rays_a, ts = ctx.saved_tensors
        segments = torch.cat([rays_a[:, 1], rays_a[-1:, 1]+rays_a[-1:, 2]])
        dL_drays_o = segment_csr(dL_dxyzs, segments)
        dL_drays_d = \
            segment_csr(dL_dxyzs*rearrange(ts, 'n -> n 1')+dL_ddirs, segments)

        return dL_drays_o, dL_drays_d, None, None, None, None, None, None, None


class VolumeRenderer(torch.autograd.Function):
    """
    Volume rendering with different number of samples per ray
    Used in training only

    Inputs:
        sigmas: (N)
        rgbs: (N, 3)
        deltas: (N)
        ts: (N)
        rays_a: (N_rays, 3) ray_idx, start_idx, N_samples
                meaning each entry corresponds to the @ray_idx th ray,
                whose samples are [start_idx:start_idx+N_samples]
        T_threshold: float, stop the ray if the transmittance is below it

    Outputs:
        total_samples: int, total effective samples
        opacity: (N_rays)
        depth: (N_rays)
        rgb: (N_rays, 3)
        ws: (N) sample point weights
    """
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, sigmas, rgbs, deltas, ts, rays_a, T_threshold):
        rgbs = rgbs.contiguous()
        total_samples, opacity, depth, rgb, ws = \
            vren.composite_train_fw(sigmas, rgbs, deltas, ts,
                                    rays_a, T_threshold)
        ctx.save_for_backward(sigmas, rgbs, deltas, ts, rays_a,
                              opacity, depth, rgb, ws)
        ctx.T_threshold = T_threshold
        return total_samples.sum(), opacity, depth, rgb, ws

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dtotal_samples, dL_dopacity, dL_ddepth, dL_drgb, dL_dws):
        sigmas, rgbs, deltas, ts, rays_a, \
        opacity, depth, rgb, ws = ctx.saved_tensors
        dL_dsigmas, dL_drgbs = \
            vren.composite_train_bw(dL_dopacity, dL_ddepth, dL_drgb, dL_dws,
                                    sigmas, rgbs, ws, deltas, ts,
                                    rays_a,
                                    opacity, depth, rgb,
                                    ctx.T_threshold)
        return dL_dsigmas, dL_drgbs, None, None, None, None

# Implementation from torch-ngp:
# https://github.com/ashawkey/torch-ngp/blob/93b08a0d4ec1cc6e69d85df7f0acdfb99603b628/activation.py
class TruncExp(torch.autograd.Function):
    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.exp(x)

    @staticmethod
    @custom_bwd
    def backward(ctx, dL_dout):
        x = ctx.saved_tensors[0]
        return dL_dout * torch.exp(x.clamp(-15, 15))


class HashRadianceField(nn.Module):
    def forward_ab(self):
        beta = torch.exp(self.ln_beta * self.speed_factor)
        return 1./beta, beta

    def __init__(self, hparams):
        super().__init__()
        self.chunk_size = hparams.chunk_size
        if hparams.Function == 'SDF':
            self.speed_factor = hparams.speed_factor
            ln_beta_init = np.log(hparams.beta_init) / self.speed_factor
            self.ln_beta = nn.Parameter(data=torch.Tensor([ln_beta_init]), requires_grad=True)
            
        # scene bounding box
        self.aabb = hparams.get('aabb', 0.5)
        self.register_buffer('center', torch.zeros(1, 3))
        self.register_buffer('xyz_min', -torch.ones(1, 3)*self.aabb)
        self.register_buffer('xyz_max', torch.ones(1, 3)*self.aabb)
        self.register_buffer('half_size', (self.xyz_max-self.xyz_min)/2)
        # each density grid covers [-2^(k-1), 2^(k-1)]^3 for k in [0, C-1]
        self.cascades = hparams.levels
        # self.cascades = max(1+int(np.ceil(np.log2(2*self.aabb))), 1)
        self.grid_size = 128
        self.register_buffer('density_bitfield', torch.zeros(self.cascades*self.grid_size**3//8, dtype=torch.uint8))
        self.register_buffer('density_grid', torch.zeros(self.cascades, self.grid_size**3))
        self.register_buffer('grid_coords', create_meshgrid3d(self.grid_size, self.grid_size, self.grid_size, False, dtype=torch.int32).reshape(-1, 3))
        self.NEAR_DISTANCE = hparams.get('NEAR_DISTANCE', None)

        # constants
        L = 16; F = 2; log2_T = 19; N_min = 16
        b = np.exp(np.log(2048*self.aabb/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.xyz_encoder = \
            tcnn.NetworkWithInputEncoding(
                n_input_dims=3, n_output_dims=16,
                encoding_config={
                    "otype": "Grid",
	                "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                },
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 1,
                }
            )

        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )

        self.rgb_net = \
            tcnn.Network(
                n_input_dims=32, n_output_dims=3,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": 'Sigmoid',
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                }
            )



    @batchify
    def forward(self, xyzs, dirs=None, **kwargs):
        """
        Inputs:
            xyzs: (N, 3) xyzs in [-aabb, aabb]
            dirs: (N, 3) directions

        Outputs:
            sigmas: (N)
            rgbs: (N, 3)
        """
        xyzs = (xyzs-self.xyz_min)/(self.xyz_max-self.xyz_min) # (0,1)
        h = self.xyz_encoder(xyzs)
        sigmas = TruncExp.apply(h[:, 0])
        if dirs is None:
            return sigmas
        dirs = dirs/torch.norm(dirs, dim=1, keepdim=True)
        dirs = self.dir_encoder((dirs+1)/2)
        rgbs = self.rgb_net(torch.cat([dirs, h], 1))
        return sigmas, rgbs

    @torch.no_grad()
    def get_all_cells(self):
        """
        Get all cells from the density grid.
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        indices = vren.morton3D(self.grid_coords).long()
        cells = [(indices, self.grid_coords)] * self.cascades

        return cells

    @torch.no_grad()
    def sample_uniform_and_occupied_cells(self, M, density_threshold):
        """
        Sample both M uniform and occupied cells (per cascade)
        occupied cells are sample from cells with density > @density_threshold
        
        Outputs:
            cells: list (of length self.cascades) of indices and coords
                   selected at each cascade
        """
        cells = []
        for c in range(self.cascades):
            # uniform cells
            coords1 = torch.randint(self.grid_size, (M, 3), dtype=torch.int32,
                                    device=self.density_grid.device)
            indices1 = vren.morton3D(coords1).long()
            # occupied cells
            indices2 = torch.nonzero(self.density_grid[c]>density_threshold)[:, 0]
            if len(indices2)>0:
                rand_idx = torch.randint(len(indices2), (M,),
                                         device=self.density_grid.device)
                indices2 = indices2[rand_idx]
            coords2 = vren.morton3D_invert(indices2.int())
            # concatenate
            cells += [(torch.cat([indices1, indices2]), torch.cat([coords1, coords2]))]

        return cells

    @torch.no_grad()
    def mark_invisible_cells(self, K, poses, img_wh, chunk=64**3):
        """
        mark the cells that aren't covered by the cameras with density -1
        only executed once before training starts

        Inputs:
            K: (3, 3) camera intrinsics
            poses: (N, 3, 4) camera to world poses
            img_wh: image width and height
            chunk: the chunk size to split the cells (to avoid OOM)
        """
        N_cams = poses.shape[0]
        self.count_grid = torch.zeros_like(self.density_grid)
        w2c_R = rearrange(poses[:, :3, :3], 'n a b -> n b a') # (N_cams, 3, 3)
        w2c_T = -w2c_R@poses[:, :3, 3:] # (N_cams, 3, 1)
        cells = self.get_all_cells()
        for c in range(self.cascades):
            indices, coords = cells[c]
            for i in range(0, len(indices), chunk):
                xyzs = coords[i:i+chunk]/(self.grid_size-1)*2-1
                s = min(2**(c-1), self.aabb)
                half_grid_size = s/self.grid_size
                xyzs_w = (xyzs*(s-half_grid_size)).T # (3, chunk)
                xyzs_c = w2c_R @ xyzs_w + w2c_T # (N_cams, 3, chunk)
                uvd = K @ xyzs_c # (N_cams, 3, chunk)
                uv = uvd[:, :2]/uvd[:, 2:] # (N_cams, 2, chunk)
                in_image = (uvd[:, 2]>=0)& \
                           (uv[:, 0]>=0)&(uv[:, 0]<img_wh[0])& \
                           (uv[:, 1]>=0)&(uv[:, 1]<img_wh[1])
                covered_by_cam = (uvd[:, 2]>=self.NEAR_DISTANCE)&in_image # (N_cams, chunk)
                # if the cell is visible by at least one camera
                self.count_grid[c, indices[i:i+chunk]] = count = covered_by_cam.sum(0)/N_cams

                too_near_to_cam = (uvd[:, 2]<self.NEAR_DISTANCE)&in_image # (N, chunk)
                # if the cell is too close (in front) to any camera
                too_near_to_any_cam = too_near_to_cam.any(0)
                # a valid cell should be visible by at least one camera and not too close to any camera
                valid_mask = (count>0)&(~too_near_to_any_cam)
                self.density_grid[c, indices[i:i+chunk]] = \
                    torch.where(valid_mask, 0., -1.)

    @torch.no_grad()
    def update_density_grid(self, density_threshold, warmup=False, decay=0.95, erode=False):
        density_grid_tmp = torch.zeros_like(self.density_grid)
        if warmup: # during the first steps
            cells = self.get_all_cells()
        else:
            cells = self.sample_uniform_and_occupied_cells(self.grid_size**3//4,
                                                           density_threshold)
        # infer sigmas
        for c in range(self.cascades):
            indices, coords = cells[c]
            s = min(2**(c-1), self.aabb)
            half_grid_size = s/self.grid_size
            xyzs_w = (coords/(self.grid_size-1)*2-1)*(s-half_grid_size)
            # pick random position in the cell by adding noise in [-hgs, hgs]
            xyzs_w += (torch.rand_like(xyzs_w)*2-1) * half_grid_size
            density_grid_tmp[c, indices] = self.forward(xyzs_w)

        if erode:
            # My own logic. decay more the cells that are visible to few cameras
            decay = torch.clamp(decay**(1/self.count_grid), 0.1, 0.95)
        self.density_grid = \
            torch.where(self.density_grid<0,
                        self.density_grid,
                        torch.maximum(self.density_grid*decay, density_grid_tmp))

        mean_density = self.density_grid[self.density_grid>0].mean().item()

        vren.packbits(self.density_grid, min(mean_density, density_threshold),
                      self.density_bitfield)


class NGPPlSampling(torch.nn.Module):
    def __init__(self, hparams, dataset_hparams):
        super().__init__()
        self.function = hparams.Function
        # self.random_bg = hparams.random_bg
        if hparams.aabb > 0.5:
            self.exp_step_factor = 1/256
        else:
            self.exp_step_factor = 0.
        self.nerf = eval(f"{hparams.Model}RadianceField")(hparams) 
        self.update_interval = 16
        self.erode = hparams.get('erode', False)
        self.MAX_SAMPLES = hparams.MAX_SAMPLES
        self.NEAR_DISTANCE = hparams.NEAR_DISTANCE
        self.dataset_hparams = dataset_hparams
        
        
    def mark_invisible_cells(self):
        self.nerf.mark_invisible_cells(*self.dataset_hparams)
        
        
    def update(self, warmup):
        self.nerf.update_density_grid(0.01*self.MAX_SAMPLES/3**0.5, warmup=warmup, erode=self.erode)
        

    @torch.cuda.amp.autocast()
    def forward(self, rays, step):
        """
        Render rays by
        1. Compute the intersection of the rays with the scene bounding box
        2. Follow the process in @render_func (different for train/test)

        Inputs:
            rays_o: (N_rays, 3) ray origins
            rays_d: (N_rays, 3) ray directions

        Outputs:
            result: dictionary containing final rgb and depth
        """
        if step%16 == 0:
            warmup = step<256
            self.update(warmup)
        rays_o, rays_d = rays[:, 0:3].contiguous(), rays[:, 3:6].contiguous() # both (B, 3)
        _, hits_t, _ = RayAABBIntersector.apply(rays_o, rays_d, self.nerf.center, self.nerf.half_size, 1)
        hits_t[(hits_t[:, 0, 0]>=0)&(hits_t[:, 0, 0]<self.NEAR_DISTANCE), 0, 0] = self.NEAR_DISTANCE

        mode = 'train' if self.training else 'test'
        results = eval(f'self.render_{mode}')(rays_o, rays_d, hits_t)
        return results

    @torch.no_grad()
    def render_test(self, rays_o, rays_d, hits_t):
        """
        Render rays by

        while (a ray hasn't converged)
            1. Move each ray to its next occupied @N_samples (initially 1) samples 
            and evaluate the properties (sigmas, rgbs) there
            2. Composite the result to output; if a ray has transmittance lower
            than a threshold, mark this ray as converged and stop marching it.
            When more rays are dead, we can increase the number of samples
            of each marching (the variable @N_samples)
        """
        results = {}

        # output tensors to be filled in
        N_rays = len(rays_o)
        device = rays_o.device
        opacity = torch.zeros(N_rays, device=device)
        depth = torch.zeros(N_rays, device=device)
        color = torch.zeros(N_rays, 3, device=device)

        samples = total_samples = 0
        alive_indices = torch.arange(N_rays, device=device)
        # if it's synthetic data, bg is majority so min_samples=1 effectively covers the bg
        # otherwise, 4 is more efficient empirically
        min_samples = 1 if self.exp_step_factor==0 else 4

        while samples < self.MAX_SAMPLES:
            N_alive = len(alive_indices)
            if N_alive==0: 
                break

            # the number of samples to add on each ray
            N_samples = max(min(N_rays//N_alive, 64), min_samples)
            samples += N_samples

            xyzs, dirs, deltas, ts, N_eff_samples = \
                vren.raymarching_test(rays_o, 
                                      rays_d, 
                                      hits_t[:, 0], 
                                      alive_indices,
                                      self.nerf.density_bitfield, 
                                      self.nerf.cascades,
                                      self.nerf.aabb, 
                                      self.exp_step_factor,
                                      self.nerf.grid_size, 
                                      self.MAX_SAMPLES, 
                                      N_samples
                                    )
            total_samples += N_eff_samples.sum()
            xyzs = rearrange(xyzs, 'n1 n2 c -> (n1 n2) c')
            dirs = rearrange(dirs, 'n1 n2 c -> (n1 n2) c')
            valid_mask = ~torch.all(dirs==0, dim=1)
            if valid_mask.sum()==0: 
                break

            sigmas = torch.zeros(len(xyzs), device=device)
            rgbs = torch.zeros(len(xyzs), 3, device=device)
            sigmas[valid_mask], _rgbs = self.nerf(xyzs[valid_mask], dirs[valid_mask])
            rgbs[valid_mask] = _rgbs.float()
            
            if self.function == 'SDF':
                sigmas = sdf_to_sigma(sigmas, *self.nerf.forward_ab())
            
            sigmas = rearrange(sigmas, '(n1 n2) -> n1 n2', n2=N_samples)
            rgbs = rearrange(rgbs, '(n1 n2) c -> n1 n2 c', n2=N_samples)

            vren.composite_test_fw(
                sigmas, 
                rgbs, 
                deltas, 
                ts,
                hits_t[:, 0], 
                alive_indices, 
                1e-4,
                N_eff_samples, 
                opacity, 
                depth, 
                color
            )
            alive_indices = alive_indices[alive_indices>=0] # remove converged rays

        results['opacity'] = opacity
        results['depths'] = depth
        results['colors'] = color
        results['total_samples'] = total_samples # total samples for all rays

        if self.exp_step_factor==0: # synthetic
            rgb_bg = torch.ones(3, device=device)
        else: # real
            rgb_bg = torch.zeros(3, device=device)
        results['colors'] += rgb_bg*rearrange(1-opacity, 'n -> n 1')

        return results


    def render_train(self, rays_o, rays_d, hits_t):
        """
        Render rays by
        1. March the rays along their directions, querying @density_bitfield
        to skip empty space, and get the effective sample points (where there is object)
        2. Infer the NN at these positions and view directions to get properties (currently sigmas and rgbs)
        3. Use volume rendering to combine the result (front to back compositing
        and early stop the ray if its transmittance is below a threshold)
        """
        results = {}

        ret = RayMarcher.apply(
                rays_o, 
                rays_d, 
                hits_t[:, 0], 
                self.nerf.density_bitfield,
                self.nerf.cascades, 
                self.nerf.aabb,
                self.exp_step_factor, 
                self.nerf.grid_size, 
                self.MAX_SAMPLES
            )
        rays_a, xyzs, dirs, results['deltas'], results['ts'], results['rm_samples'] = ret
        sigmas, rgbs = self.nerf(xyzs, dirs)
        if self.function == 'SDF':
            sigmas = sdf_to_sigma(sigmas, *self.nerf.forward_ab())
            
        ret = VolumeRenderer.apply(sigmas, rgbs, results['deltas'], results['ts'], rays_a, 1e-4)
        results['vr_samples'], results['opacity'], results['depths'], results['colors'], results['ws'] = ret
        results['rays_a'] = rays_a

        # if self.exp_step_factor==0: # synthetic
        #     rgb_bg = torch.ones(3, device=rays_o.device)
        # else: # real
        #     if self.random_bg:
        #         rgb_bg = torch.rand(3, device=rays_o.device)
        #     else:
        #         rgb_bg = torch.zeros(3, device=rays_o.device)
        # results['colors'] = results['colors'] + rgb_bg*rearrange(1-results['opacity'], 'n -> n 1')

        return results