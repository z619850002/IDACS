from model import *
import time as tim


import nerfacc
from nerfacc.estimators.occ_grid import OccGridEstimator
from nerfacc.estimators.prop_net import PropNetEstimator, get_proposal_requires_grad_fn
from utils import *


def sdf_to_sigma(sdf: torch.Tensor, alpha, beta):
    exp = 0.5 * torch.exp(-torch.abs(sdf) / beta)
    psi = torch.where(sdf >= 0, exp, 1 - exp)
    return alpha * psi


class NGPSampling(nn.Module):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.nerf = eval(f"{hparams.Model}RadianceField")(hparams)
        # self.nerf = MLPRadianceField(hparams)
        self.near = hparams.near
        self.far = hparams.far
        aabb = hparams.aabb
        levels = hparams.levels
        self.aabb = torch.tensor([-aabb, -aabb, -aabb, aabb, aabb, aabb])
        self.estimator = OccGridEstimator(roi_aabb=self.aabb, resolution=128, levels=1)

        self.render_step_size = 5e-3
        
    def forward(self, rays, **kwargs):
        if self.training:
            def occ_eval_fn(x):
                density = self.nerf(x)
                return density * self.render_step_size
            self.estimator.update_every_n_steps(
                step=kwargs['step'],
                occ_eval_fn=occ_eval_fn,
            )
        
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (B, 3)
        
        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            sigmas = self.nerf(positions)
            return sigmas.squeeze(-1)

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            rgbs, sigmas = self.nerf(positions, t_dirs)
            return rgbs, sigmas.squeeze(-1)
        
        ray_indices, t_starts, t_ends = self.estimator.sampling(
            rays_o,
            rays_d,
            sigma_fn=sigma_fn,
            near_plane = self.near,
            far_plane = self.far,
            render_step_size = self.render_step_size,
            stratified=self.training,
        )


        if (len(ray_indices)==0):
            result = None
            return result

        rgb, opacity, depth, extras = nerfacc.rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=N_rays,
            rgb_sigma_fn=rgb_sigma_fn,
        )
        result = {}
        result['colors'] = rgb
        result['depths'] = depth
        return result



class NGPSamplingCanonical(nn.Module):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.nerf = CanonicalMLPRadianceField(hparams)

        # self.nerf = MLPRadianceField(hparams)
        self.near = hparams.near
        self.far = hparams.far
        aabb = hparams.aabb
        levels = hparams.levels
        self.aabb = torch.tensor([-aabb, -aabb, -aabb, aabb, aabb, aabb])
        self.estimator = OccGridEstimator(roi_aabb=self.aabb, resolution=128, levels=1)

        self.render_step_size = 5e-3

    # def forward(self, rays, indices, max_index, **kwargs):
    #     if self.training:
    #         def occ_eval_fn(x):
    #             density = self.nerf(x)
    #             return density * self.render_step_size
    #
    #         self.estimator.update_every_n_steps(
    #             step=kwargs['step'],
    #             occ_eval_fn=occ_eval_fn,
    #         )
    #
    #     N_rays = rays.shape[0]
    #     rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (B, 3)
    #
    #     def sigma_fn(t_starts, t_ends, ray_indices):
    #         t_origins = rays_o[ray_indices]
    #         t_dirs = rays_d[ray_indices]
    #         positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
    #         sigmas = self.nerf(positions)
    #         return sigmas.squeeze(-1)
    #
    #     def rgb_sigma_fn(t_starts, t_ends, ray_indices):
    #         t_origins = rays_o[ray_indices]
    #         t_dirs = rays_d[ray_indices]
    #         positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
    #         rgbs, sigmas = self.nerf(positions, t_dirs)
    #         return rgbs, sigmas.squeeze(-1)
    #
    #     ray_indices, t_starts, t_ends = self.estimator.sampling(
    #         rays_o,
    #         rays_d,
    #         sigma_fn=sigma_fn,
    #         near_plane=self.near,
    #         far_plane=self.far,
    #         render_step_size=self.render_step_size,
    #         stratified=self.training,
    #     )
    #
    #
    #     rgb, opacity, depth, extras = nerfacc.rendering(
    #         t_starts,
    #         t_ends,
    #         ray_indices,
    #         n_rays=N_rays,
    #         rgb_sigma_fn=rgb_sigma_fn,
    #     )
    #     result = {}
    #     result['colors'] = rgb
    #     result['depths'] = depth
    #     return result


    def forward(self, rays, indices, max_index, **kwargs):
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (B, 3)
        ts = indices.float() / max_index

        if self.training:
            def occ_eval_fn(x):
                times = torch.ones_like(x)[:, 0:1]
                times *= ts[0]
                input = torch.cat((x, times), dim=1)
                density = self.nerf(input)
                occupied = torch.ones_like(density)
                return density * self.render_step_size

            self.estimator.update_every_n_steps(
                step=kwargs['step'],
                occ_eval_fn=occ_eval_fn,
            )
        # if 'test' in kwargs:
        #     if kwargs['test']:
        #         ts = torch.ones_like(ts) * 0.5

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            times = ts[ray_indices].unsqueeze(1).to(positions.device)
            inputs = torch.cat((positions, times), dim = 1)
            sigmas = self.nerf(inputs)
            return sigmas.squeeze(-1)

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            times = ts[ray_indices].unsqueeze(1).to(positions.device)
            inputs = torch.cat((positions, times), dim=1)
            rgbs, sigmas = self.nerf(inputs, t_dirs)

            return rgbs, sigmas.squeeze(-1)

        ray_indices, t_starts, t_ends = self.estimator.sampling(
            rays_o,
            rays_d,
            sigma_fn=sigma_fn,
            near_plane=self.near,
            far_plane=self.far,
            render_step_size=self.render_step_size,
            stratified=self.training,
        )

        # if (len(ray_indices)==0):
        #     result = None
        #     del ray_indices
        #     del t_starts
        #     del t_ends
        #     del rays_o
        #     del rays_d
        #     return result

        rgb, opacity, depth, extras = nerfacc.rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=N_rays,
            rgb_sigma_fn=rgb_sigma_fn,
        )




        result = {}
        result['colors'] = rgb
        result['depths'] = depth

        if self.training:
            # compute TV loss

            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            del t_origins
            del t_dirs
            del t_starts
            del t_ends
            del rays_o
            del rays_d
            times = ts[ray_indices].unsqueeze(1).to(positions.device)
            del ray_indices
            if len(times) > 300000:
                times = times[:300000]
                positions = positions[:300000]
            result['space']  = self.nerf.query_time(positions, times)

            time_interval = 0.02
            previous_times = times - time_interval
            previous_times[previous_times < 0.0] = 0.0
            result['pre_space'] = self.nerf.query_time(positions, previous_times)
            del previous_times

            next_time = times + time_interval
            del times
            next_time[next_time > 1.0] = 1.0
            result['next_spaces'] = self.nerf.query_time(positions, next_time)
            del positions
            del next_time

        return result




class NGPSamplingTime(nn.Module):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.nerf = MultiHashTimeRadianceField(hparams)


        # self.nerf = MultiMLPRadianceField(hparams)
        self.near = hparams.near
        self.far = hparams.far
        aabb = hparams.aabb
        levels = hparams.levels
        self.aabb = torch.tensor([-aabb, -aabb, -aabb, aabb, aabb, aabb])
        self.estimator = OccGridEstimator(roi_aabb=self.aabb, resolution=128, levels=1)

        self.render_step_size = 5e-3

    def compute_feature_diff(self, rays_o, rays_d, ts, t_starts, t_ends, ray_indices):
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        times = ts[ray_indices].unsqueeze(1).to(positions.device)
        inputs = torch.cat((positions, times), dim = 1)
        time_diff = 0.005
        h, h_prev = self.nerf.get_feature_diff(inputs, time_diff)
        return h, h_prev


    def forward(self, rays, indices, max_index, **kwargs):
        # T1 = tim.time()

        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (B, 3)
        ts = indices.float()/max_index
        ts = torch.ones_like(indices.float()/max_index) * 0.0
        if self.training:
            def occ_eval_fn(x):
                times = torch.ones_like(x)[:,0:1]
                times *= ts[0]
                input = torch.cat((x, times), dim=1)
                density = self.nerf(input)
                return density * self.render_step_size

            self.estimator.update_every_n_steps(
                step=kwargs['step'],
                occ_eval_fn=occ_eval_fn,
            )


        # T2 = tim.time()
        # print('occ运行时间:%s毫秒' % ((T2 - T1) * 1000))

        # if 'test' in kwargs:
        #     if kwargs['test']:
        #         ts = torch.ones_like(ts) * 0.5

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            times = ts[ray_indices].unsqueeze(1).to(positions.device)
            inputs = torch.cat((positions, times), dim = 1)
            sigmas = self.nerf(inputs)
            return sigmas.squeeze(-1)




        ray_indices, t_starts, t_ends = self.estimator.sampling(
            rays_o,
            rays_d,
            sigma_fn=sigma_fn,
            near_plane=self.near,
            far_plane=self.far,
            render_step_size=self.render_step_size,
            stratified=self.training,
        )

        if (len(ray_indices) == 0):
            result = None
            return result


        # T2 = tim.time()
        # print('sampling运行时间:%s毫秒' % ((T2 - T1) * 1000))
        if self.training:
            def rgb_sigma_fn(t_starts, t_ends, ray_indices):
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
                times = ts[ray_indices].unsqueeze(1).to(positions.device)
                inputs = torch.cat((positions, times), dim=1)
                rgbs, sigmas, feature1, feature2 = self.nerf(inputs, t_dirs, True)
                return rgbs, sigmas.squeeze(-1), feature1, feature2

            rgb, opacity, depth, extras = nerfacc.rendering_features(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=N_rays,
                rgb_sigma_fn=rgb_sigma_fn,
            )
        else:
            def rgb_sigma_fn(t_starts, t_ends, ray_indices):
                t_origins = rays_o[ray_indices]
                t_dirs = rays_d[ray_indices]
                positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
                times = ts[ray_indices].unsqueeze(1).to(positions.device)
                inputs = torch.cat((positions, times), dim=1)
                rgbs, sigmas = self.nerf(inputs, t_dirs, False)
                return rgbs, sigmas.squeeze(-1)

            rgb, opacity, depth, extras = nerfacc.rendering(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=N_rays,
                rgb_sigma_fn=rgb_sigma_fn,
            )

        # h, h_prev = self.compute_feature_diff(rays_o, rays_d, ts, t_starts, t_ends, ray_indices)
        # h = h.detach().cpu()
        # h_prev = h_prev.detach().cpu()

        # T2 = tim.time()
        # print('render运行时间:%s毫秒' % ((T2 - T1) * 1000))


        result = {}
        result['colors'] = rgb
        result['depths'] = depth
        if self.training:
            result['hidden_feature'] = extras['features1']
            result['prev_feature'] = extras['features2']
        # result['prev_feature'] = h_prev
        return result




class NGPSamplingTime2(nn.Module):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.nerf = HashTimeRadianceField(hparams)
        self.near = hparams.near
        self.far = hparams.far
        aabb = hparams.aabb
        levels = hparams.levels
        self.aabb = torch.tensor([-aabb, -aabb, -aabb, aabb, aabb, aabb])
        self.estimator = OccGridEstimator(roi_aabb=self.aabb, resolution=128, levels=1)

        self.render_step_size = 5e-3

    def compute_feature_diff(self, rays_o, rays_d, ts, t_starts, t_ends, ray_indices):
        t_origins = rays_o[ray_indices]
        t_dirs = rays_d[ray_indices]
        positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
        times = ts[ray_indices].unsqueeze(1).to(positions.device)
        inputs = torch.cat((positions, times), dim = 1)
        time_diff = 0.02
        h, h_prev = self.nerf.get_feature_diff(inputs, time_diff)
        return h, h_prev


    def forward(self, rays, indices, max_index, **kwargs):
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (B, 3)
        ts = indices.float() / max_index

        if self.training:
            def occ_eval_fn(x):
                times = torch.ones_like(x)[:, 0:1]
                times *= ts[0]
                input = torch.cat((x, times), dim=1)
                density = self.nerf(input)
                occupied = torch.ones_like(density)
                return density * self.render_step_size

            self.estimator.update_every_n_steps(
                step=kwargs['step'],
                occ_eval_fn=occ_eval_fn,
            )
        # if 'test' in kwargs:
        #     if kwargs['test']:
        #         ts = torch.ones_like(ts) * 0.5

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            times = ts[ray_indices].unsqueeze(1).to(positions.device)
            inputs = torch.cat((positions, times), dim = 1)
            sigmas = self.nerf(inputs)
            return sigmas.squeeze(-1)

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            times = ts[ray_indices].unsqueeze(1).to(positions.device)
            inputs = torch.cat((positions, times), dim=1)
            rgbs, sigmas = self.nerf(inputs, t_dirs)

            return rgbs, sigmas.squeeze(-1)

        ray_indices, t_starts, t_ends = self.estimator.sampling(
            rays_o,
            rays_d,
            sigma_fn=sigma_fn,
            near_plane=self.near,
            far_plane=self.far,
            render_step_size=self.render_step_size,
            stratified=self.training,
        )

        if (len(ray_indices)==0):
            result = None
            return result

        rgb, opacity, depth, extras = nerfacc.rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=N_rays,
            rgb_sigma_fn=rgb_sigma_fn,
        )

        result = {}
        result['colors'] = rgb
        result['depths'] = depth
        if self.training:
            h, h_prev = self.compute_feature_diff(rays_o, rays_d, ts, t_starts, t_ends, ray_indices)
            result['hidden_feature'] = h
            result['prev_feature'] = h_prev
        # result['hidden_feature'] = h
        # result['prev_feature'] = h_prev
        return result




class CollaborativeNGPSampling(nn.Module):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.node_num = hparams.node_num
        self.nerf_list = torch.nn.ModuleList([eval(f"{hparams.Model}RadianceField")(hparams)  for i in range(self.node_num)])
        # self.nerf = eval(f"{hparams.Model}RadianceField")(hparams)
        self.near = hparams.near
        self.far = hparams.far
        aabb = hparams.aabb
        levels = hparams.levels
        self.aabb = torch.tensor([-aabb, -aabb, -aabb, aabb, aabb, aabb])
        self.estimator_list = torch.nn.ModuleList([OccGridEstimator(roi_aabb=self.aabb, resolution=128, levels=1)  for i in range(self.node_num)])
        self.render_step_size = 5e-3
        self.step_list = [0 for i in range(self.node_num)]

    def determine_nerf_index(self):
        nerf_index = 0

        return nerf_index

    def forward(self, rays, **kwargs):
        indices = kwargs['indices']
        max_index = kwargs['max_index']
        nerf_index = int(float(indices[0].item())/float(max_index[0].item()) * self.node_num)
        if nerf_index >= self.node_num:
            nerf_index = self.node_num-1
        if self.training:
            def occ_eval_fn(x):
                density = self.nerf_list[nerf_index](x)
                return density * self.render_step_size

            step = self.step_list[nerf_index]
            self.estimator_list[nerf_index].update_every_n_steps(
                step=step,
                occ_eval_fn=occ_eval_fn,
            )
            self.step_list[nerf_index] = self.step_list[nerf_index]+1

        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (B, 3)

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            sigmas = self.nerf_list[nerf_index](positions)
            return sigmas.squeeze(-1)

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            rgbs, sigmas = self.nerf_list[nerf_index](positions, t_dirs)

            return rgbs, sigmas.squeeze(-1)

        ray_indices, t_starts, t_ends = self.estimator_list[nerf_index].sampling(
            rays_o,
            rays_d,
            sigma_fn=sigma_fn,
            near_plane=self.near,
            far_plane=self.far,
            render_step_size=self.render_step_size,
            stratified=self.training,
        )

        if ray_indices.shape[0] == 0:
            return None

        rgb, opacity, depth, extras = nerfacc.rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=N_rays,
            rgb_sigma_fn=rgb_sigma_fn,
        )
        result = {}
        result['colors'] = rgb
        result['depths'] = depth
        return result



class DoubleSampling(nn.Module):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        self.nerf = eval(f"{hparams.Model}RadianceField")(hparams)
        # S()
        # self.near = hparams.near
        # self.far = hparams.far
        self.estimator = PropNetEstimator()

        # self.N_samples = hparams.N_samples
        # self.N_importance_samples = hparams.N_importance_samples
        # self.white_background = hparams.white_background
        # self.noise = hparams.noise
        # self.function = hparams.Function
        # self.near, self.far = hparams.near, hparams.far
        self.render_step_size = 5e-3
        
    def forward(self, rays, **kwargs):
        if self.training:
            def occ_eval_fn(x):
                density = self.nerf(x)
                return density * self.render_step_size
            self.estimator.update_every_n_steps(
                step=kwargs['step'],
                occ_eval_fn=occ_eval_fn,
            )
        
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (B, 3)
        
        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            sigmas = self.nerf(positions)
            return sigmas.squeeze(-1)

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            rgbs, sigmas = self.nerf(positions, t_dirs)
            return rgbs, sigmas.squeeze(-1)
        
        ray_indices, t_starts, t_ends = self.estimator.sampling(
            rays_o,
            rays_d,
            sigma_fn=sigma_fn,
            # near_plane = self.near,
            # far_plane = self.far,
            render_step_size = self.render_step_size,
            stratified=self.training,
        )
        rgb, opacity, depth, extras = nerfacc.rendering(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=N_rays,
            rgb_sigma_fn=rgb_sigma_fn,
        )
        result = {}
        result['colors'] = rgb
        result['depths'] = depth
        return result

