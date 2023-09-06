import torch
import math
from torch import nn
import tinycudann as tcnn
from ipdb import set_trace as S
import numpy as np
from kornia.utils.grid import create_meshgrid3d
from einops import rearrange
import torch.nn.functional as F
from torch.cuda.amp import custom_fwd, custom_bwd
import time as ti2


def batchify_gt_color(forward):
    def decorator(self, xyz_samples, dir_samples=None, gt_color = None):
        chunk_size = self.chunk_size
        if dir_samples is not None:
            if len(xyz_samples) == 0:
                return torch.empty(0, device=xyz_samples.device), torch.empty(0, device=xyz_samples.device)
            sigma_ls, color_ls = [], []
            color2_ls = []
            for i in range(0, xyz_samples.shape[0], chunk_size):
                if gt_color is None:
                    color, color2, sigma = forward(self, xyz_samples[i:i + chunk_size], dir_samples[i:i + chunk_size])  # (B*N,1), (B*N,3)
                else:
                    color, color2, sigma = forward(self, xyz_samples[i:i+chunk_size], dir_samples[i:i+chunk_size], gt_color[i:i+chunk_size]) # (B*N,1), (B*N,3)
                sigma_ls.append(sigma)
                color_ls.append(color)
                color2_ls.append(color2)
            sigma = torch.cat(sigma_ls, 0)
            color = torch.cat(color_ls, 0)
            color2 = torch.cat(color2_ls, 0)
            return color, color2, sigma
        else:
            if len(xyz_samples) == 0:
                return torch.empty(0, device=xyz_samples.device)
            sigma_ls= []
            for i in range(0, xyz_samples.shape[0], chunk_size):
                sigma_ls.append(forward(self, xyz_samples[i:i+chunk_size]))
            sigma = torch.cat(sigma_ls, 0)
            return sigma

    return decorator

def batchify_index(forward):
    def decorator(self, xyz_samples, frame_index, dir_samples=None):
        chunk_size = self.chunk_size
        if dir_samples is not None:
            if len(xyz_samples) == 0:
                return torch.empty(0, device=xyz_samples.device), torch.empty(0, device=xyz_samples.device)
            sigma_ls, color_ls = [], []
            for i in range(0, xyz_samples.shape[0], chunk_size):
                sigma, color = forward(self, xyz_samples[i:i+chunk_size], frame_index, dir_samples[i:i+chunk_size]) # (B*N,1), (B*N,3)
                sigma_ls.append(sigma)
                color_ls.append(color)
            sigma = torch.cat(sigma_ls, 0)
            color = torch.cat(color_ls, 0)
            return sigma, color
        else:
            if len(xyz_samples) == 0:
                return torch.empty(0, device=xyz_samples.device)
            sigma_ls= []
            for i in range(0, xyz_samples.shape[0], chunk_size):
                sigma_ls.append(forward(self, xyz_samples[i:i+chunk_size], frame_index))
            sigma = torch.cat(sigma_ls, 0)
            return sigma

    return decorator

def batchify(forward):
    def decorator(self, xyz_samples, dir_samples=None):
        chunk_size = self.chunk_size
        if dir_samples is not None:
            if len(xyz_samples) == 0:
                return torch.empty(0, device=xyz_samples.device), torch.empty(0, device=xyz_samples.device)
            sigma_ls, color_ls = [], []
            for i in range(0, xyz_samples.shape[0], chunk_size):
                sigma, color = forward(self, xyz_samples[i:i+chunk_size], dir_samples[i:i+chunk_size]) # (B*N,1), (B*N,3)
                sigma_ls.append(sigma)
                color_ls.append(color)
            sigma = torch.cat(sigma_ls, 0)
            color = torch.cat(color_ls, 0)
            return sigma, color
        else:
            if len(xyz_samples) == 0:
                return torch.empty(0, device=xyz_samples.device)
            sigma_ls= []
            for i in range(0, xyz_samples.shape[0], chunk_size):
                sigma_ls.append(forward(self, xyz_samples[i:i+chunk_size]))
            sigma = torch.cat(sigma_ls, 0)
            return sigma

    return decorator

def batchify_features(forward):
    def decorator(self, xyz_samples, dir_samples=None, maintain_features = True):
        chunk_size = self.chunk_size
        if dir_samples is not None:
            if len(xyz_samples) == 0:
                return torch.empty(0, device=xyz_samples.device), torch.empty(0, device=xyz_samples.device), None, None
            sigma_ls, color_ls = [], []
            if maintain_features:
                features1_ls, features2_ls = [], []
            for i in range(0, xyz_samples.shape[0], chunk_size):
                sigma, color, feature1, feature2 = forward(self, xyz_samples[i:i+chunk_size], dir_samples[i:i+chunk_size]) # (B*N,1), (B*N,3)
                sigma_ls.append(sigma)
                color_ls.append(color)
                if maintain_features:
                    features1_ls.append(feature1)
                    features2_ls.append(feature2)
            sigma = torch.cat(sigma_ls, 0)
            color = torch.cat(color_ls, 0)
            if maintain_features:
                features1 = torch.cat(features1_ls, 0)
                features2 = torch.cat(features2_ls, 0)
            if maintain_features:
                return sigma, color, features1, features2
            else:
                return sigma, color
        else:
            if len(xyz_samples) == 0:
                return torch.empty(0, device=xyz_samples.device)
            sigma_ls= []
            for i in range(0, xyz_samples.shape[0], chunk_size):
                sigma_ls.append(forward(self, xyz_samples[i:i+chunk_size]))
            sigma = torch.cat(sigma_ls, 0)
            return sigma

    return decorator

def diffbatchify(get_diff):
    def decorator(self, original_xyzs, time_diff):
        chunk_size = self.chunk_size
        if len(original_xyzs) == 0:
            return torch.empty(0, device=original_xyzs.device)
        hiddens0 = []
        hiddens1 = []
        for i in range(0, original_xyzs.shape[0], chunk_size):
            h0, h1 = get_diff(self, original_xyzs[i:i + chunk_size], time_diff)
            hiddens0.append(h0)
            hiddens1.append(h1)
        all_h0 = torch.cat(hiddens0, 0)
        all_h1 = torch.cat(hiddens1, 0)
        return all_h0, all_h1

    return decorator

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
 
 
class RadianceField(nn.Module):
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


class FrequenceEncoder(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.funcs = [torch.sin, torch.cos]
        self.freqs = 2**torch.linspace(0,L-1,L)
    def forward(self, xyzs): # (B,c)
        encodings = [xyzs]
        for freq in self.freqs:
            for func in self.funcs:
                encodings.append(func(freq*xyzs))
        return torch.cat(encodings, -1)



class LinearActivation(nn.Module):
    def __init__(self, inp, out, activation=None):
        super().__init__()
        model = [nn.Linear(inp, out)]
        if activation == "relu":
            model.append(nn.ReLU(True))
        elif activation == "sigmoid":
            model.append(nn.Sigmoid())
        self.model = nn.Sequential(*model)

    def forward(self, xyzs):
        return self.model(xyzs)
            
        
class MLPRadianceField(RadianceField):
    def __init__(self, hparams, xyz_encoder=None, dir_encoder=None):
        super().__init__(hparams)
        if xyz_encoder is not None:
            self.xyz_encoder, self.dir_encoder = xyz_encoder, dir_encoder
        else:
            self.xyz_encoder, self.dir_encoder = FrequenceEncoder(10), FrequenceEncoder(4)
        self.backbone1 = nn.Sequential(
            LinearActivation(63, 256, "relu"),
            LinearActivation(256, 256, "relu"),
            LinearActivation(256, 256, "relu"),
            LinearActivation(256, 256, "relu"),
            LinearActivation(256, 256, "relu"),
        )
        self.backbone2 = nn.Sequential(
            LinearActivation(256+63, 256, "relu"),
            LinearActivation(256, 256, "relu"),
            LinearActivation(256, 256, "relu"),
        )
        self.sigma_head = nn.Linear(256, 1) # no relu-activated here since maybe noise is needed before activated (activated outside in rendering)
        self.backbone3 = LinearActivation(256, 256)
        self.color_head = nn.Sequential(
            LinearActivation(256+27, 128, "relu"),
            LinearActivation(128, 3, "sigmoid")
        )
    
    @batchify
    def forward(self, xyzs, dirs=None, **kwargs): # (B*N,3)  (B*N,3)
        xyz_encoding = self.xyz_encoder(xyzs) # (B*N,63)
        feature = self.backbone1(xyz_encoding) # (B*N,256)
        feature = self.backbone2(torch.cat((feature, xyz_encoding), 1)) # (B*N,256)
        sigma = self.sigma_head(feature) # (B*N,1)
        if dirs is None:
            return sigma
        dir_encoding = self.dir_encoder(dirs) # (B*N,27)
        feature = self.backbone3(feature)
        feature = torch.cat((feature, dir_encoding), 1)
        color = self.color_head(feature)# (B*N,3)
        return color, sigma



class DensityRadianceField(RadianceField):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.aabb = hparams.aabb
        # constants
        L = 16
        F = 2
        log2_T = 19
        N_min = 16
        N_max = 4096
        b = np.exp(np.log(N_max / N_min) / (L - 1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')
        self.use_color_net = False

        self.xyz_encoder_sigma_net = \
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


        self.color_encoder_sigma_net = \
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

        self.color_net = \
            tcnn.Network(
                n_input_dims=32, n_output_dims=3,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
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

    def activate_color_net(self):
        self.use_color_net = True

    def fix_density(self):
        for parameter in self.xyz_encoder_sigma_net.parameters():
            parameter.requires_grad = False
            parameter.grad = None

    @torch.cuda.amp.autocast()
    @batchify
    def forward(self, xyzs, dirs=None, **kwargs):
        """
        Inputs:
            xyzs: (N, 3) xyzs in [-aabb, aabb]
            dirs: (N, 3) directions
        Outputs:
            sigma: (N)
            color: (N, 3)
        """
        xyzs = (xyzs + self.aabb) / self.aabb / 2  # (0,1)
        h = self.xyz_encoder_sigma_net(xyzs)
        sigma = TruncExp.apply(h[:, 0])
        if dirs is None:
            return sigma
        else:
            # dirs = dirs/torch.norm(dirs, dim=1, keepdim=True)
            # dirs = self.dir_encoder((dirs+1)/2)
            rgbs = dirs[:,3:]
            t_dirs = dirs[:,:3]
            if self.use_color_net:

                hc = self.color_encoder_sigma_net(xyzs)
                t_dirs = self.dir_encoder(t_dirs)
                color = self.color_net(torch.cat([t_dirs, hc], 1))
                color = torch.sigmoid(color)
            else:
                color = rgbs
            # color = self.color_net(torch.cat([t_dirs, h], 1))
            # rgbs[rgbs<=0.0] = 1e-5
            # rgbs[rgbs>=1.0] = 1-(1e-5)
            # color = color + torch.logit(rgbs)

            # color = torch.sigmoid(color)
            # color_ratio = F.softmax(color, dim=1)
        return color, sigma



class SampleRadianceField(RadianceField):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.aabb = hparams.aabb
        # constants
        L = 16
        F = 2
        log2_T = 10
        N_min = 16
        N_max = 4096
        b = np.exp(np.log(N_max / N_min) / (L - 1))
        self.frame_num = 100
        self.fix_grad = False
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.xyz_hashes = torch.nn.ModuleList([tcnn.Encoding(n_input_dims=3,
                                        encoding_config={
                                            "otype": "Grid",
                                            "type": "Hash",
                                            "n_levels": L,
                                            "n_features_per_level": F,
                                            "log2_hashmap_size": log2_T,
                                            "base_resolution": N_min,
                                            "per_level_scale": b,
                                            "interpolation": "Linear"
                                        }) for i in range(self.frame_num)])


        self.xyz_encoder = tcnn.Network(
            n_input_dims=L * F,
            n_output_dims=16,
            network_config={
                "otype": "CutlassMLP",
                "activation": "SoftPlus",
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

    def get_table_and_ref_index(self, frame_index):
        table_index = int(frame_index/3)
        ref_index = table_index*3+1
        if ref_index >= len(self.imgs)-1:
            ref_index = len(self.imgs)-2
        ref_indices = [ref_index-1, ref_index, ref_index+1]
        return table_index, ref_indices


    def clamp_positions(self, positions):
        positions = torch.clamp(positions, torch.Tensor([0,0]).to(positions.device), torch.Tensor([self.w-2, self.h-2]).to(positions.device))
        return positions

    def bilinear_sample(self, img, pixel_positions):
        #Get four corners
        pixel_positions_clamp = pixel_positions[:,:2]
        pixel_positions_clamp = self.clamp_positions(pixel_positions_clamp)
        x_min_y_min = pixel_positions_clamp.int()
        x_min_y_max = pixel_positions_clamp.int()
        x_min_y_max[:,1] +=1

        x_max_y_min = pixel_positions_clamp.int()
        x_max_y_min[:,0] +=1
        x_max_y_max = pixel_positions_clamp.int()
        x_max_y_max[:,0] +=1
        x_max_y_max[:,1] +=1


        def get_indices(x_y):
            indices = (x_y[:, 1] * self.w + x_y[:, 0]).long()
            outlier_mask = indices < 0
            outlier_mask2 = indices >= self.w * self.h
            outlier_mask = torch.logical_or(outlier_mask, outlier_mask2)
            return outlier_mask, indices

        outlier_mask_1, x_min_y_min_ind = get_indices(x_min_y_min)
        outlier_mask_2, x_min_y_max_ind = get_indices(x_min_y_max)
        outlier_mask_3, x_max_y_min_ind = get_indices(x_max_y_min)
        outlier_mask_4, x_max_y_max_ind = get_indices(x_max_y_max)

        outlier_masks = torch.logical_or(torch.logical_or(torch.logical_or(outlier_mask_1, outlier_mask_2), outlier_mask_3), outlier_mask_4)

        x_min_y_min_ind[outlier_mask_1] = 0
        x_min_y_max_ind[outlier_mask_2] = 0
        x_max_y_min_ind[outlier_mask_3] = 0
        x_max_y_max_ind[outlier_mask_4] = 0

        img_view = img.view(-1,3)
        color_x_min_y_min = img_view[x_min_y_min_ind]
        color_x_min_y_max = img_view[x_min_y_max_ind]
        color_x_max_y_min = img_view[x_max_y_min_ind]
        color_x_max_y_max = img_view[x_max_y_max_ind]

        x_diff = (pixel_positions_clamp - x_min_y_min)[:, 0]
        y_diff = (pixel_positions_clamp - x_min_y_min)[:, 1]

        color_y_min =  x_diff.unsqueeze(1) * color_x_max_y_min + (1-x_diff.unsqueeze(1)) * color_x_min_y_min
        color_y_max =  x_diff.unsqueeze(1) * color_x_max_y_max + (1-x_diff.unsqueeze(1)) * color_x_min_y_max

        color = y_diff.unsqueeze(1) * color_y_max + (1-y_diff.unsqueeze(1)) * color_y_min
        color[outlier_masks] *=0.0
        return color

    def fix_network(self):
        self.fix_grad = True
        for parameter in self.xyz_encoder.parameters():
            parameter.requires_grad = False

    def bind_images_and_poses(self, imgs, K, T_wcs):
        self.imgs = imgs
        self.h = self.imgs[0].shape[0]
        self.w = self.imgs[0].shape[1]
        self.K = K
        self.T_wcs = T_wcs

    # @torch.cuda.amp.autocast()
    @batchify_index
    def forward(self, xyzs, index, dirs=None, **kwargs):
        table_index, ref_indices = self.get_table_and_ref_index(index)

        if dirs is not None:
            # all_pixel_positions = []
            # for i in range(len(ref_indices)):
            ref_index = ref_indices[1]
            img = self.imgs[ref_index]
            T_wc = self.T_wcs[ref_index]
            R_wc = T_wc[:3, :3]
            t_wc = T_wc[:3, 3]
            # T_wc = torch.cat([T_wc, torch.FloatTensor([0., 0., 0., 1.]).unsqueeze(0).to(T_wc.device)], dim=0)
            R_cw = R_wc.t()
            t_cw = torch.matmul(-R_cw, t_wc)

            pixel_positions = (xyzs @ R_cw.T + t_cw)
            pixel_positions[:, 2:][torch.abs(pixel_positions[:, 2:]) < 1e-3] = 1e-3
            pixel_positions = (pixel_positions / pixel_positions[:, 2:]) @ self.K.T
                # all_pixel_positions.append(pixel_positions)

        with torch.cuda.amp.autocast():
            xyzs = (xyzs + self.aabb) / self.aabb / 2  # (0,1)
            features = self.xyz_hashes[table_index](xyzs)

            h = self.xyz_encoder(features)
            sigma = TruncExp.apply(h[:, 0])
            if dirs is None:
                return sigma
            else:
                # dirs = dirs/torch.norm(dirs, dim=1, keepdim=True)
                # dirs = self.dir_encoder((dirs+1)/2)
                # dirs = self.dir_encoder(dirs)
                # color = self.color_net(torch.cat([dirs, h], 1))
                #Sample color
                # for i in range(len())
                color = self.bilinear_sample(img, pixel_positions)


        return color, sigma





class HashRadianceField(RadianceField):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.aabb = hparams.aabb
        # constants
        L = 16
        F = 2
        log2_T = 19
        N_min = 16
        N_max = 4096
        b = np.exp(np.log(N_max / N_min) / (L - 1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.xyz_encoder_sigma_net = \
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

        self.color_net = \
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

    @torch.cuda.amp.autocast()
    @batchify
    def forward(self, xyzs, dirs=None, **kwargs):
        """
        Inputs:
            xyzs: (N, 3) xyzs in [-aabb, aabb]
            dirs: (N, 3) directions
        Outputs:
            sigma: (N)
            color: (N, 3)
        """
        xyzs = (xyzs + self.aabb) / self.aabb / 2  # (0,1)
        h = self.xyz_encoder_sigma_net(xyzs)
        sigma = TruncExp.apply(h[:, 0])
        if dirs is None:
            return sigma
        else:
            # dirs = dirs/torch.norm(dirs, dim=1, keepdim=True)
            # dirs = self.dir_encoder((dirs+1)/2)
            dirs = self.dir_encoder(dirs)
            color = self.color_net(torch.cat([dirs, h], 1))

        return color, sigma




class CanonicalHashRadianceField(RadianceField):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.aabb = hparams.aabb
        # constants
        L = 16
        F = 2
        log2_T = 19
        N_min = 16
        N_max = 4096
        b = np.exp(np.log(N_max / N_min) / (L - 1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.xyz_hashes = tcnn.Encoding(n_input_dims=3,
               encoding_config={
                   "otype": "Grid",
                   "type": "Hash",
                   "n_levels": L,
                   "n_features_per_level": F,
                   "log2_hashmap_size": log2_T,
                   "base_resolution": N_min,
                   "per_level_scale": b,
                   "interpolation": "Linear"
               })

        self.xyz_encoder = tcnn.Network(
            n_input_dims=L * F,
            n_output_dims=16,
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

        self.color_net = \
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

        self.W = 256
        self.D = 8
        self.input_ch = 3
        self.input_ch_time = 1
        self.skips = [4]
        self.time_encoder, self.time_encoder_final = self.create_time_net()


    def create_time_net(self):
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 1):

            layer = nn.Linear

            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch

            layers += [layer(in_channels, self.W)]
        return nn.ModuleList(layers), nn.Linear(self.W, 3)

    @torch.cuda.amp.autocast()
    def query_time(self, new_pts, t):
        h = torch.cat([new_pts, t], dim=-1)
        for i, l in enumerate(self.time_encoder):
            h = self.time_encoder[i](h)
            h =  F.relu(h)
            if i in self.skips:
                h = torch.cat([new_pts, h], -1)
        # for i in range(len(self.time_encoder)):
        #     if self.time_encoder[i].weight.grad is not None:
        #         if math.isnan(self.time_encoder[i].weight.grad.norm()) or math.isinf(self.time_encoder[i].weight.grad.norm()):
        #             print('Wrong!')
        #
        # if math.isnan(h.norm()) or math.isinf(h.norm()):
        #     print('Wrong!')

        return self.time_encoder_final(h)



    @torch.cuda.amp.autocast()
    @batchify
    def forward(self, original_xyzs, dirs=None, **kwargs):
        """
        Inputs:
            xyzs: (N, 3) xyzs in [-aabb, aabb]
            dirs: (N, 3) directions
        Outputs:
            sigma: (N)
            color: (N, 3)
        """

        ts = original_xyzs[:, -1:]
        xyzs = original_xyzs[:, :3]
        xyzs = (xyzs + self.aabb) / self.aabb / 2  # (0,1)

        canonical_space = self.query_time(xyzs, ts)

        # self.canonical_space = canonical_space

        # xyzs = (xyzs + self.aabb) / self.aabb / 2  # (0,1)
        xyz_hash = self.xyz_hashes(canonical_space + xyzs)
        h = self.xyz_encoder(xyz_hash)
        sigma = TruncExp.apply(h[:, 0])
        if dirs is None:
            return sigma
        else:
            # dirs = dirs/torch.norm(dirs, dim=1, keepdim=True)
            # dirs = self.dir_encoder((dirs+1)/2)
            dirs = self.dir_encoder(dirs)
            color = self.color_net(torch.cat([dirs, h], 1))

        return color, sigma

    def __init__(self, hparams, xyz_encoder=None, dir_encoder=None):
        super().__init__(hparams)
        if xyz_encoder is not None:
            self.xyz_encoder, self.dir_encoder = xyz_encoder, dir_encoder
        else:
            self.xyz_encoder, self.dir_encoder = FrequenceEncoder(10), FrequenceEncoder(4)
        self.backbone1 = nn.Sequential(
            LinearActivation(63, 256, "relu"),
            LinearActivation(256, 256, "relu"),
            LinearActivation(256, 256, "relu"),
            LinearActivation(256, 256, "relu"),
            LinearActivation(256, 256, "relu"),
        )
        self.backbone2 = nn.Sequential(
            LinearActivation(256 + 63, 256, "relu"),
            LinearActivation(256, 256, "relu"),
            LinearActivation(256, 256, "relu"),
        )
        self.sigma_head = nn.Linear(256,
                                    1)  # no relu-activated here since maybe noise is needed before activated (activated outside in rendering)
        self.backbone3 = LinearActivation(256, 256)
        self.color_head = nn.Sequential(
            LinearActivation(256 + 27, 128, "relu"),
            LinearActivation(128, 3, "sigmoid")
        )

    @batchify
    def forward(self, xyzs, dirs=None, **kwargs):  # (B*N,3)  (B*N,3)
        xyz_encoding = self.xyz_encoder(xyzs)  # (B*N,63)
        feature = self.backbone1(xyz_encoding)  # (B*N,256)
        feature = self.backbone2(torch.cat((feature, xyz_encoding), 1))  # (B*N,256)
        sigma = self.sigma_head(feature)  # (B*N,1)
        if dirs is None:
            return sigma
        dir_encoding = self.dir_encoder(dirs)  # (B*N,27)
        feature = self.backbone3(feature)
        feature = torch.cat((feature, dir_encoding), 1)
        color = self.color_head(feature)  # (B*N,3)
        return color, sigma



class NeRF:
    @staticmethod
    def get_by_name(type,  *args, **kwargs):
        print ("NeRF type selected: %s" % type)

        if type == "original":
            model = NeRFOriginal(*args, **kwargs)
        elif type == "direct_temporal":
            model = DirectTemporalNeRF(*args, **kwargs)
        else:
            raise ValueError("Type %s not recognized." % type)
        return model

class NeRFOriginal(nn.Module):
    def __init__(self, D=8, W=256, input_ch=3, input_ch_views=3, input_ch_time=1, output_ch=4, skips=[4],
                 use_viewdirs=False, memory=[], embed_fn=None, output_color_ch=3, zero_canonical=True):
        super(NeRFOriginal, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_views = input_ch_views
        self.skips = skips
        self.use_viewdirs = use_viewdirs

        # self.pts_linears = nn.ModuleList(
        #     [nn.Linear(input_ch, W)] +
        #     [nn.Linear(W, W) if i not in self.skips else nn.Linear(W + input_ch, W) for i in range(D-1)])

        layers = [nn.Linear(input_ch, W)]
        for i in range(D - 1):
            if i in memory:
                raise NotImplementedError
            else:
                layer = nn.Linear

            in_channels = W
            if i in self.skips:
                in_channels += input_ch

            layers += [layer(in_channels, W)]

        self.pts_linears = nn.ModuleList(layers)

        ### Implementation according to the official code release (https://github.com/bmild/nerf/blob/master/run_nerf_helpers.py#L104-L105)
        self.views_linears = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])

        ### Implementation according to the paper
        # self.views_linears = nn.ModuleList(
        #     [nn.Linear(input_ch_views + W, W//2)] + [nn.Linear(W//2, W//2) for i in range(D//2)])

        if use_viewdirs:
            self.feature_linear = nn.Linear(W, W)
            self.alpha_linear = nn.Linear(W, 1)
            self.rgb_linear = nn.Linear(W//2, output_color_ch)
        else:
            self.output_linear = nn.Linear(W, output_ch)

    def forward(self, x, ts):
        input_pts, input_views = torch.split(x, [self.input_ch, self.input_ch_views], dim=-1)
        h = input_pts
        for i, l in enumerate(self.pts_linears):
            h = self.pts_linears[i](h)
            h = F.relu(h)
            if i in self.skips:
                h = torch.cat([input_pts, h], -1)

        if self.use_viewdirs:
            alpha = self.alpha_linear(h)
            feature = self.feature_linear(h)
            h = torch.cat([feature, input_views], -1)

            for i, l in enumerate(self.views_linears):
                h = self.views_linears[i](h)
                h = F.relu(h)

            rgb = self.rgb_linear(h)
            outputs = torch.cat([rgb, alpha], -1)
        else:
            outputs = self.output_linear(h)

        return outputs, torch.zeros_like(input_pts[:, :3])

    def load_weights_from_keras(self, weights):
        assert self.use_viewdirs, "Not implemented if use_viewdirs=False"

        # Load pts_linears
        for i in range(self.D):
            idx_pts_linears = 2 * i
            self.pts_linears[i].weight.data = torch.from_numpy(np.transpose(weights[idx_pts_linears]))
            self.pts_linears[i].bias.data = torch.from_numpy(np.transpose(weights[idx_pts_linears+1]))

        # Load feature_linear
        idx_feature_linear = 2 * self.D
        self.feature_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_feature_linear]))
        self.feature_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_feature_linear+1]))

        # Load views_linears
        idx_views_linears = 2 * self.D + 2
        self.views_linears[0].weight.data = torch.from_numpy(np.transpose(weights[idx_views_linears]))
        self.views_linears[0].bias.data = torch.from_numpy(np.transpose(weights[idx_views_linears+1]))

        # Load rgb_linear
        idx_rbg_linear = 2 * self.D + 4
        self.rgb_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear]))
        self.rgb_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_rbg_linear+1]))

        # Load alpha_linear
        idx_alpha_linear = 2 * self.D + 6
        self.alpha_linear.weight.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear]))
        self.alpha_linear.bias.data = torch.from_numpy(np.transpose(weights[idx_alpha_linear+1]))





class CanonicalMLPRadianceField(RadianceField):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.aabb = hparams.aabb
        # constants
        L = 16
        F = 2
        log2_T = 19
        N_min = 16
        N_max = 4096
        b = np.exp(np.log(N_max / N_min) / (L - 1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.W = 256
        self.D = 8
        self.level_xyz_time = 10
        self.level_dirs = 4
        self.dim_xyz_freqs = 2 * len(2 ** torch.linspace(0, self.level_xyz_time - 1, self.level_xyz_time))+1
        self.dim_dir_freqs = 2 * len(2 ** torch.linspace(0, self.level_dirs - 1, self.level_dirs)) + 1
        self.input_ch = 3*self.dim_xyz_freqs
        self.input_ch_time = self.dim_xyz_freqs
        self.skips = [4]
        self.canonical_encoder, self.canonical_encoder_final = self.create_time_net()

        self.xyz_encoder, self.dir_encoder, self.time_encoder = FrequenceEncoder(self.level_xyz_time), FrequenceEncoder(self.level_dirs), FrequenceEncoder(self.level_xyz_time)
        self.backbone1 = nn.Sequential(
            LinearActivation(63, 256, "relu"),
            LinearActivation(256, 256, "relu"),
            LinearActivation(256, 256, "relu"),
            LinearActivation(256, 256, "relu"),
            LinearActivation(256, 256, "relu"),
        )
        self.backbone2 = nn.Sequential(
            LinearActivation(256 + 63, 256, "relu"),
            LinearActivation(256, 256, "relu"),
            LinearActivation(256, 256, "relu"),
        )
        self.sigma_head = nn.Linear(256,
                                    1)  # no relu-activated here since maybe noise is needed before activated (activated outside in rendering)
        self.backbone3 = LinearActivation(256, 256)
        self.color_head = nn.Sequential(
            LinearActivation(256 + 27, 128, "relu"),
            LinearActivation(128, 3, "sigmoid")
        )


    def create_time_net(self):
        layers = [nn.Linear(self.input_ch + self.input_ch_time, self.W)]
        for i in range(self.D - 1):

            layer = nn.Linear

            in_channels = self.W
            if i in self.skips:
                in_channels += self.input_ch

            layers += [layer(in_channels, self.W)]

        return nn.ModuleList(layers), nn.Linear(self.W, 3)

    # @torch.cuda.amp.autocast()
    def query_time(self, original_pts, original_ts):
        if original_ts[0,0] == 0.0:
            return torch.zeros_like(original_pts)
        new_pts = self.xyz_encoder(original_pts)
        t = self.time_encoder(original_ts)
        h = torch.cat([new_pts, t], dim=-1)
        del t
        for i, l in enumerate(self.canonical_encoder):
            h = self.canonical_encoder[i](h)
            h =  F.relu(h)
            if i in self.skips:
                h = torch.cat([new_pts, h], -1)


        del new_pts
        return self.canonical_encoder_final(h)



    # @torch.cuda.amp.autocast()
    @batchify
    def forward(self, original_xyzs, dirs=None, **kwargs):
        """
        Inputs:
            xyzs: (N, 3) xyzs in [-aabb, aabb]
            dirs: (N, 3) directions
        Outputs:
            sigma: (N)
            color: (N, 3)
        """

        ts = original_xyzs[:, -1:]
        xyzs = original_xyzs[:, :3]

        dx = self.query_time(xyzs, ts)
        # xyzs = xyzs+dx

        # aabb = self.aabb

        # xyzs = (xyzs + self.aabb) / self.aabb / 2  # (0,1)


        # self.canonical_space = canonical_space

        # xyzs = (xyzs + self.aabb) / self.aabb / 2  # (0,1)
        # xyz_hash = self.xyz_hashes(canonical_space + xyzs)
        # h = self.xyz_encoder(xyz_hash)
        # sigma = TruncExp.apply(h[:, 0])
        # if dirs is None:
        #     return sigma
        # else:
        #     # dirs = dirs/torch.norm(dirs, dim=1, keepdim=True)
        #     # dirs = self.dir_encoder((dirs+1)/2)
        #     dirs = self.dir_encoder(dirs)
        #     color = self.color_net(torch.cat([dirs, h], 1))

        xyz_encoding = self.xyz_encoder(xyzs+dx)  # (B*N,63)
        feature = self.backbone1(xyz_encoding)  # (B*N,256)
        feature = self.backbone2(torch.cat((feature, xyz_encoding), 1))  # (B*N,256)
        sigma = self.sigma_head(feature)  # (B*N,1)
        if dirs is None:
            return sigma
        dir_encoding = self.dir_encoder(dirs)  # (B*N,27)
        feature = self.backbone3(feature)
        feature = torch.cat((feature, dir_encoding), 1)
        color = self.color_head(feature)  # (B*N,3)

        return color, sigma





class HashTimeRadianceField(RadianceField):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.aabb = hparams.aabb
        # constants
        L = 16
        F = 2
        log2_T = 19
        N_min = 8
        N_max = 2048
        b = np.exp(np.log(N_max/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.xyz_hash = tcnn.Encoding(n_input_dims=4,
                encoding_config={
                    "otype": "Grid",
	                "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                })

        self.xyz_encoder = tcnn.Network(
            n_input_dims=L*F,
            n_output_dims=16,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 1,
            }
        )
        #
        # self.xyz_encoder_sigma_net = \
        #     tcnn.NetworkWithInputEncoding(
        #         n_input_dims=4, n_output_dims=16,
        #         encoding_config={
        #             "otype": "Grid",
	    #             "type": "Hash",
        #             "n_levels": L,
        #             "n_features_per_level": F,
        #             "log2_hashmap_size": log2_T,
        #             "base_resolution": N_min,
        #             "per_level_scale": b,
        #             "interpolation": "Linear"
        #         },
        #         network_config={
        #             "otype": "CutlassMLP",
        #             "activation": "ReLU",
        #             "output_activation": "None",
        #             "n_neurons": 64,
        #             "n_hidden_layers": 1,
        #         }
        #     )

        self.dir_encoder = \
            tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "SphericalHarmonics",
                    "degree": 4,
                },
            )

        self.color_net = \
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

    @torch.cuda.amp.autocast()
    @diffbatchify
    def get_feature_diff(self, original_xyzs, time_diff):
        """
        Inputs:
            xyzs: (N, 3) xyzs in [-aabb, aabb]
            dirs: (N, 3) directions
        Outputs:
            sigma: (N)
            color: (N, 3)
        """
        ts = original_xyzs[:, -1:]
        prev_ts = torch.max(ts-time_diff, torch.zeros_like(ts))
        xyzs = original_xyzs[:, :3]
        xyzs = (xyzs + self.aabb) / self.aabb / 2  # (0,1)
        input = torch.cat((xyzs, ts), dim=1)
        input_prev = torch.cat((xyzs, prev_ts), dim=1)
        h = self.xyz_hash(input)
        h_prev = self.xyz_hash(input_prev)
        return h, h_prev


    @torch.cuda.amp.autocast()
    @batchify
    def forward(self, original_xyzs, dirs=None, **kwargs):
        """
        Inputs:
            xyzs: (N, 3) xyzs in [-aabb, aabb]
            dirs: (N, 3) directions
        Outputs:
            sigma: (N)
            color: (N, 3)
        """
        ts = original_xyzs[:,-1:]
        xyzs = original_xyzs[:,:3]
        xyzs = (xyzs+self.aabb)/self.aabb/2 # (0,1)
        input = torch.cat((xyzs, ts), dim=1)
        hash_res = self.xyz_hash(input)
        h = self.xyz_encoder(hash_res)
        sigma = TruncExp.apply(h[:, 0])
        if dirs is None:
            return sigma
        else:
            # dirs = dirs/torch.norm(dirs, dim=1, keepdim=True)
            # dirs = self.dir_encoder((dirs+1)/2)
            dirs = self.dir_encoder(dirs)
            color = self.color_net(torch.cat([dirs, h], 1))

        return color, sigma


class MultiMLPRadianceField(RadianceField):
    def __init__(self, hparams, xyz_encoder=None, dir_encoder=None):
        super().__init__(hparams)
        if xyz_encoder is not None:
            self.xyz_encoder, self.dir_encoder = xyz_encoder, dir_encoder
        else:
            self.xyz_encoder, self.dir_encoder = FrequenceEncoder(10), FrequenceEncoder(4)

        self.time_resolution = 50

        self.backbones1 = nn.ModuleList(
            nn.Sequential(
                LinearActivation(63, 256, "relu"),
                LinearActivation(256, 256, "relu"),
                LinearActivation(256, 256, "relu"),
                LinearActivation(256, 256, "relu"),
                LinearActivation(256, 256, "relu"),
            ) for i in range(self.time_resolution)
        )


        self.backbones2 = nn.ModuleList(
            nn.Sequential(
                LinearActivation(256 + 63, 256, "relu"),
                LinearActivation(256, 256, "relu"),
                LinearActivation(256, 256, "relu"),
            ) for i in range(self.time_resolution)
        )


        self.sigma_head = nn.Linear(256,
                                    1)  # no relu-activated here since maybe noise is needed before activated (activated outside in rendering)
        self.backbone3 = LinearActivation(256, 256)
        self.color_head = nn.Sequential(
            LinearActivation(256 + 27, 128, "relu"),
            LinearActivation(128, 3, "sigmoid")
        )

    def get_feature_stream(self, index, input):
        feature = self.backbones1[index](input)
        feature = self.backbones2[index](torch.cat((feature, input), 1))  # (B*N,256)
        if math.isnan(feature.norm()) or math.isinf(feature.norm()):
            print('Wrong!')
        return feature

    def get_features(self, index, input, prev_ratio, next_ratio):

        feature1 = self.get_feature_stream(index, input)
        next_index = min(index + 1, len(self.backbones1) - 1)
        feature2 = self.get_feature_stream(next_index, input)
        sum = prev_ratio + next_ratio
        prev_ratio = prev_ratio / sum
        next_ratio = next_ratio / sum
        # result = feature1 * prev_ratio + feature2 * next_ratio
        result = feature1 * prev_ratio + feature2 * next_ratio
        return result, feature1, feature2


    @batchify_features
    def forward(self, original_xyzs, dirs=None, **kwargs):  # (B*N,3)  (B*N,3)
        ts = original_xyzs[:, -1:]
        xyzs = original_xyzs[:, :3]
        # xyzs = (xyzs + self.aabb) / self.aabb / 2  # (0,1)
        index = int(ts[0] * self.time_resolution)
        prev_ratio = 1 - (ts[0] * self.time_resolution - float(index))
        next_ratio = 1 - prev_ratio
        if index >= len(self.backbones1):
            index = len(self.backbones1) - 1
        xyz_encoding = self.xyz_encoder(xyzs)  # (B*N,63)
        feature, prev_feature, next_feature = self.get_features(index, xyz_encoding, prev_ratio, next_ratio)
        sigma = self.sigma_head(feature)  # (B*N,1)
        if dirs is None:
            return sigma
        dir_encoding = self.dir_encoder(dirs)  # (B*N,27)
        feature = self.backbone3(feature)
        feature = torch.cat((feature, dir_encoding), 1)
        color = self.color_head(feature)  # (B*N,3)
        return color, sigma, prev_feature, next_feature


class MultiHashTimeRadianceField(RadianceField):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.aabb = hparams.aabb
        # constants
        # L = 16
        # F = 2
        # log2_T = 19
        # N_min = 16
        # N_max = 4096

        L = 16
        F = 2
        log2_T = 19
        N_min = 16
        N_max = 4096
        b = np.exp(np.log(N_max/N_min)/(L-1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.time_resolution = 16
        self.time_resolution2 = 20
        self.time_grid_size = 1/float(self.time_resolution)
        self.time_grid_size2 = 1/float(self.time_resolution2)
        self.static_xyz_hashes = tcnn.Encoding(n_input_dims=3,
                encoding_config={
                    "otype": "Grid",
	                "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                })


        self.xyz_hashes = torch.nn.ModuleList([tcnn.Encoding(n_input_dims=3,
                encoding_config={
                    "otype": "Grid",
	                "type": "Hash",
                    "n_levels": L,
                    "n_features_per_level": F,
                    "log2_hashmap_size": log2_T,
                    "base_resolution": N_min,
                    "per_level_scale": b,
                    "interpolation": "Linear"
                }) for i in range(self.time_resolution)])

        self.xyz_hashes2 = torch.nn.ModuleList([tcnn.Encoding(n_input_dims=3,
            encoding_config={
                "otype": "Grid",
                "type": "Hash",
                "n_levels": L,
                "n_features_per_level": F,
                "log2_hashmap_size": log2_T,
                "base_resolution": N_min,
                "per_level_scale": b,
                "interpolation": "Linear"}) for i in range(self.time_resolution2)])


        self.xyz_encoder = tcnn.Network(
            n_input_dims=L*F*3,
            n_output_dims=16,
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

        self.color_net = \
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

    @torch.cuda.amp.autocast()
    def hashes(self, index, input, prev_ratio, next_ratio):
        feature1 = self.xyz_hashes[index](input)
        next_index = min(index+1, len(self.xyz_hashes)-1)
        feature2  =self.xyz_hashes[next_index](input)
        sum = prev_ratio + next_ratio
        prev_ratio = prev_ratio/sum
        next_ratio = next_ratio/sum
        # result = feature1 * prev_ratio + feature2 * next_ratio
        result = feature1 * prev_ratio + feature2 * next_ratio
        aligned_feature1 = feature1[:,-8:]
        aligned_feature2 = feature2[:, -8:]
        return result, aligned_feature1, aligned_feature2

    @torch.cuda.amp.autocast()
    def hashes2(self, index, input, prev_ratio, next_ratio):
        feature1 = self.xyz_hashes2[index](input)
        next_index = min(index + 1, len(self.xyz_hashes2) - 1)
        feature2 = self.xyz_hashes2[next_index](input)
        sum = prev_ratio + next_ratio
        prev_ratio = prev_ratio / sum
        next_ratio = next_ratio / sum
        result = feature1 * prev_ratio + feature2 * next_ratio
        aligned_feature1 = feature1[:, -8:]
        aligned_feature2 = feature2[:, -8:]
        return result, aligned_feature1, aligned_feature2


    @torch.cuda.amp.autocast()
    @diffbatchify
    def get_feature_diff(self, original_xyzs, time_diff):
        """
        Inputs:
            xyzs: (N, 3) xyzs in [-aabb, aabb]
            dirs: (N, 3) directions
        Outputs:
            sigma: (N)
            color: (N, 3)
        """
        ts = original_xyzs[:, -1:]
        xyzs = original_xyzs[:, :3]
        xyzs = (xyzs + self.aabb) / self.aabb / 2  # (0,1)
        # input = torch.cat((xyzs, ts), dim=1)
        # input_prev = torch.cat((xyzs, prev_ts), dim=1)
        index = int(ts[0] * self.time_resolution)+1
        if index >= len(self.xyz_hashes):
            index = len(self.xyz_hashes) - 1
        prev_index = index-1
        if prev_index <0:
            prev_index = 0

        h = self.xyz_hashes[index](xyzs)
        h_prev = self.xyz_hashes[prev_index](xyzs)
        return h, h_prev


    @torch.cuda.amp.autocast()
    @batchify_features
    def forward(self, original_xyzs, dirs=None, maintain_features = True, **kwargs):
        """
        Inputs:
            xyzs: (N, 3) xyzs in [-aabb, aabb]
            dirs: (N, 3) directions
        Outputs:
            sigma: (N)
            color: (N, 3)
        """
        ts = original_xyzs[:,-1:]
        xyzs = original_xyzs[:,:3]
        xyzs = (xyzs+self.aabb)/self.aabb/2 # (0,1)

        #index in the coarse level
        index = int(ts[0] * self.time_resolution)
        prev_ratio =  1-(ts[0] * self.time_resolution - float(index))
        next_ratio = 1 - prev_ratio
        if index >= len(self.xyz_hashes):
            index = len(self.xyz_hashes)-1

        #index in the fine level
        index2 = int(ts[0] * self.time_resolution2)
        prev_ratio2 = 1 - (ts[0] * self.time_resolution2 - float(index2))
        next_ratio2 = 1 - prev_ratio2
        if index2 >= len(self.xyz_hashes2):
            index2 = len(self.xyz_hashes2) - 1

        hash_res, feature1, feature2 = self.hashes(index,xyzs, prev_ratio, next_ratio)
        hash_res_2, feature1_2, feature2_2 = self.hashes2(index2,xyzs, prev_ratio2, next_ratio2)

        hash_res = torch.cat([hash_res, hash_res_2],dim=1)
        feature1 = torch.cat([feature1, feature1_2], dim=1)
        feature2 = torch.cat([feature2, feature2_2], dim=1)

        static_hash_res = self.static_xyz_hashes(xyzs)
        hash_res = torch.cat([hash_res, static_hash_res],dim=1)
        h = self.xyz_encoder(hash_res)
        sigma = TruncExp.apply(h[:, 0])
        if dirs is None:
            return sigma
        else:
            dirs = self.dir_encoder(dirs)
            color = self.color_net(torch.cat([dirs, h], 1))
        if maintain_features:
            return color, sigma, feature1, feature2
        else:
            return color, sigma


class FrameFeatureRadianceField(RadianceField):
    def __init__(self, hparams, xyz_encoder=None, dir_encoder=None):
        super().__init__(hparams)

        if xyz_encoder is not None:
            self.xyz_encoder, self.dir_encoder = xyz_encoder, dir_encoder
        else:
            self.xyz_encoder, self.dir_encoder = FrequenceEncoder(10), FrequenceEncoder(4)

        self.remove_outliers = False

        self.aabb = hparams.aabb
        # constants
        L = 16
        F = 2
        log2_T = 19
        N_min = 16
        N_max = 4096
        b = np.exp(np.log(N_max / N_min) / (L - 1))
        print(f'GridEncoding: Nmin={N_min} b={b:.5f} F={F} T=2^{log2_T} L={L}')

        self.xyz_encoder_sigma_net = \
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

        self.xyz_hashes = tcnn.Encoding(n_input_dims=3,
                                               encoding_config={
                                                   "otype": "Grid",
                                                   "type": "Hash",
                                                   "n_levels": L,
                                                   "n_features_per_level": F,
                                                   "log2_hashmap_size": log2_T,
                                                   "base_resolution": N_min,
                                                   "per_level_scale": b,
                                                   "interpolation": "Linear"
                                               })

        self.color_and_sigma_encoder = tcnn.Network(
            n_input_dims=6,
            n_output_dims=16,
            network_config={
                "otype": "CutlassMLP",
                "activation": "ReLU",
                "output_activation": "None",
                "n_neurons": 64,
                "n_hidden_layers": 2,
            }
        )




        self.xyz_encoder = tcnn.Network(
            n_input_dims= L*F,
            n_output_dims=16,
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

        self.color_net = \
            tcnn.Network(
                n_input_dims=32, n_output_dims=4,
                network_config={
                    "otype": "CutlassMLP",
                    "activation": "Relu",
                    "output_activation": 'Sigmoid',
                    "n_neurons": 64,
                    "n_hidden_layers": 2,
                }
            )

        # self.output_layer = \
        #     tcnn.Network(
        #         n_input_dims=10, n_output_dims=3,
        #         network_config={
        #             "otype": "CutlassMLP",
        #             "activation": "ReLU",
        #             "output_activation": 'Sigmoid',
        #             "n_neurons": 64,
        #             "n_hidden_layers": 1,
        #         }
        #     )

    def set_remove_outliers(self, remove=True):
        self.remove_outliers = remove

    def bind_images_and_poses(self, imgs, depths, K, T_wcs, features, indices):

        self.imgs = imgs
        self.device = self.imgs.device
        self.depths = depths
        self.depths[depths<1e-3] = torch.max(self.depths)
        self.indices = indices
        self.h = self.imgs.shape[1]
        self.w = self.imgs.shape[2]
        self.K = K
        self.T_wcs = T_wcs
        self.origins = self.T_wcs[:, :3, 3]
        self.directions = torch.FloatTensor([0., 0., 1.]).to(T_wcs.device) @ self.T_wcs[:, :3, :3].transpose(1, 2)
        self.directions /= self.directions.norm(dim=1).unsqueeze(1)
        self.indices = indices
        self.R_cws = self.T_wcs[:, :3, :3].transpose(1, 2)
        self.t_cws = (-torch.einsum("ijk,ik->ij", [self.R_cws, self.origins]))

        #compute max fovs
        # fov1 = torch.sum(((torch.FloatTensor([0., 0., 1.]).to(T_wcs.device) @ torch.inverse(self.K).T)/(torch.FloatTensor([0, self.h, 1.]).to(T_wcs.device) @ torch.inverse(self.K).T).norm()) * torch.FloatTensor([0., 0., 1.]).to(T_wcs.device))
        # fov2 = torch.sum(((torch.FloatTensor([self.w, 0., 1.]).to(T_wcs.device) @ torch.inverse(self.K).T)/(torch.FloatTensor([0, self.h, 1.]).to(T_wcs.device) @ torch.inverse(self.K).T).norm()) * torch.FloatTensor([0., 0., 1.]).to(T_wcs.device))
        # fov3 = torch.sum(((torch.FloatTensor([0., self.h, 1.]).to(T_wcs.device) @ torch.inverse(self.K).T)/(torch.FloatTensor([0, self.h, 1.]).to(T_wcs.device) @ torch.inverse(self.K).T).norm()) * torch.FloatTensor([0., 0., 1.]).to(T_wcs.device))
        # fov4 = torch.sum(((torch.FloatTensor([self.w, self.h, 1.]).to(T_wcs.device) @ torch.inverse(self.K).T)/(torch.FloatTensor([0, self.h, 1.]).to(T_wcs.device) @ torch.inverse(self.K).T).norm()) * torch.FloatTensor([0., 0., 1.]).to(T_wcs.device))
        # self.max_fov = max(fov1.item(), fov2.item(), fov3.item(), fov4.item())
        # self.min_fov = max(fov1.item(), fov2.item(), fov3.item(), fov4.item())
        self.bind_grids()


    def locate_grids(self, xyzs):
        def get_index(bounding, indices):
            index = indices[:, 2] + indices[:, 1] * bounding[2] + indices[:, 0] * bounding[1] * bounding[2]
            return index

        grid_positions = (xyzs - self.grid_aabb_min) / self.grid_size
        left_bottom_indices = grid_positions.long()


        grid_indices_list = []
        ratios_list = None
        for shift in self.indices_shift_list:
            indices = left_bottom_indices + shift
            grid_indices = get_index(self.grid_num, indices)
            grid_indices_list.append(grid_indices)
            if ratios_list is None:
                ratios_list =(grid_positions - indices)

        return grid_indices_list, ratios_list



    def bind_grids(self):
        self.indices_shift_list = [
            torch.LongTensor([0, 0, 0]).to(self.device),
            torch.LongTensor([0, 0, 1]).to(self.device),
            torch.LongTensor([0, 1, 0]).to(self.device),
            torch.LongTensor([0, 1, 1]).to(self.device),
            torch.LongTensor([1, 0, 0]).to(self.device),
            torch.LongTensor([1, 0, 1]).to(self.device),
            torch.LongTensor([1, 1, 0]).to(self.device),
            torch.LongTensor([1, 1, 1]).to(self.device),
        ]
        self.grid_resolution = 128*2
        self.best_num = 15
        self.grid_size = (self.aabb*2)/self.grid_resolution
        self.grid_aabb_min = torch.Tensor([-self.aabb, -self.aabb, -self.aabb]).to(self.device)
        self.grid_aabb_max = self.grid_aabb_min + self.aabb*2
        self.grid_num = torch.IntTensor([self.grid_resolution, self.grid_resolution, self.grid_resolution]).to(self.device)

        grid = torch.linspace(self.grid_aabb_min[0], self.grid_aabb_max[0], self.grid_resolution)
        grids = torch.meshgrid(grid, grid, grid)
        grids_coords = torch.cat(
            [grids[0].contiguous().view(-1, 1), grids[1].contiguous().view(-1, 1), grids[2].contiguous().view(-1, 1)],
            dim=1)
        self.grids_coords = grids_coords.to(self.device)

        self.grids_indices = self.get_best_indices(self.grids_coords, 1)

        #get colors of these grids
        # for i in range(len(self.grids_indices)):
        #     self.grids_indices[i] = self.grids_indices[0]

        all_rgbs = []
        all_counts = []
        for choose_ind in self.grids_indices:
            # rgbs = torch.ones_like(xyzs) * -1
            rgbs = torch.zeros([len(self.grids_coords), 3]).to(self.imgs.device)
            counts = torch.zeros_like(self.grids_coords)[:, 0]
            valid_indices = choose_ind[choose_ind >= 0]
            R_cw = self.R_cws[valid_indices]
            t_cw = self.t_cws[valid_indices]
            step_xyzs = self.grids_coords[choose_ind >= 0]
            step_xyzs = (torch.einsum("ijk,ik->ij", [R_cw, step_xyzs]) + t_cw)
            # dists = (step_xyzs[:, 2].unsqueeze(1))
            step_xyzs = step_xyzs / (step_xyzs[:, 2].unsqueeze(1))
            step_xyzs = step_xyzs @ self.K.T
            colors = self.bilinear_sample_rgbs(step_xyzs, valid_indices)
            rgbs[choose_ind >= 0] = rgbs[choose_ind >= 0] * 0.0 + colors
            counts[choose_ind >= 0] = 1.
            rgbs = torch.cat([rgbs, torch.zeros_like(rgbs)], dim=1)
            all_rgbs.append(rgbs)
            all_counts.append(counts)
            break

        for i in range(self.best_num-1):
            all_rgbs.append(torch.zeros_like(rgbs))
            all_counts.append(torch.zeros_like(counts))

        all_rgbs = torch.stack(all_rgbs)
        # all_counts = torch.stack(all_counts)


        self.update_indices = torch.empty([(all_rgbs).shape[1]]).to(all_rgbs.device)

        # comput_counts = torch.sum(all_counts, dim=0).unsqueeze(1)
        # no_data_mask = comput_counts < (1 - (1e-3))
        # comput_counts[comput_counts < 1e-3] = 1e-3
        # mean_rgbs = torch.sum(all_rgbs[:,:,:3] * all_counts.unsqueeze(2), dim=0) / comput_counts
        # var_rgbs = torch.sum(((all_rgbs[:,:,:3] - mean_rgbs) * all_counts.unsqueeze(2)) ** 2, dim=0) / comput_counts
        # var_rgbs = torch.sqrt(var_rgbs)
        # compensation = torch.zeros_like(var_rgbs)
        # compensation[no_data_mask.squeeze(1)] = 1.0
        # var_rgbs = var_rgbs + compensation
        # mean_rgbs[mean_rgbs<1e-3] = 0.0
        self.all_rgbs = all_rgbs.contiguous()
        # self.mean_rgbs = mean_rgbs
        # self.var_rgbs = var_rgbs
        # self.sum_vars = torch.sum(var_rgbs, dim=1)
        # self.mean_rgbs *=0.0
        # self.mean_rgbs[torch.sum(var_rgbs, dim=1).unsqueeze(1).repeat(1, 3) < 1e-5] = 0.0
        # self.mean_rgbs[torch.sum(var_rgbs, dim=1).unsqueeze(1).repeat(1, 3) > 0.2] = 0.0




    def get_best_indices(self, xyzs_all, best_num):
        batch_size = 500000
        batch_num = int(len(xyzs_all)/batch_size)+1

        choose_indices = []
        choose_dists = []
        for i in range(batch_num):
            start = i*batch_size
            finish = min(start+batch_size, len(xyzs_all))
            xyzs = xyzs_all[start:finish]

            xyz_projections = torch.einsum("ijk,lk->lij", [self.R_cws, xyzs]) + self.t_cws
            dists = xyz_projections.norm(dim=2)
            mask1 = xyz_projections[:, :, 2] < 0.1
            xyz_projections = xyz_projections / xyz_projections[:, :, 2].unsqueeze(2)
            xyz_projections = xyz_projections @ self.K.T
            # get projections inner images
            mask2 = xyz_projections[:, :, 0] < 0
            mask3 = xyz_projections[:, :, 0] > self.w - 2
            mask4 = xyz_projections[:, :, 1] < 0
            mask5 = xyz_projections[:, :, 1] > self.h - 2
            mask_outer = torch.logical_or(mask1, mask2)
            mask_outer = torch.logical_or(mask_outer, mask3)
            mask_outer = torch.logical_or(mask_outer, mask4)
            mask_outer = torch.logical_or(mask_outer, mask5)
            dists[mask_outer] = 2e6

            k = best_num
            for i in range(k):
                num_points = dists.shape[1]
                min_indices = (torch.rand(len(dists)) * num_points).long().to(dists.device)
                min_dists = torch.gather(dists, 1, min_indices.unsqueeze(1))
                # min_dists, min_indices = torch.min(dists, dim=1)
                # dists.view(-1)[torch.IntTensor([m for m in range(len(xyzs))]).to(xyzs.device) * len(
                #     self.indices) + min_indices] = 2e6 + 1
                # min_indices[min_dists > 1e6] = -1
                if len(choose_dists)<=i:
                    choose_dists.append(min_dists)
                    choose_indices.append(min_indices)
                else:
                    choose_dists[i] = torch.cat([choose_dists[i], min_dists], dim=0)
                    choose_indices[i] = torch.cat([choose_indices[i], min_indices], dim=0)
        return choose_indices


    def clamp_positions(self, positions):
        positions = torch.clamp(positions, torch.Tensor([0,0]).to(positions.device), torch.Tensor([self.w-2, self.h-2]).to(positions.device))
        return positions


    def bilinear_sample_rgbs(self, pixel_positions, valid_indices):
        # img = imgs[0]
        #Get four corners
        pixel_positions_clamp = pixel_positions[:,:2]
        pixel_positions_clamp = self.clamp_positions(pixel_positions_clamp)
        x_min_y_min = pixel_positions_clamp.int()
        x_min_y_max = pixel_positions_clamp.int()
        x_min_y_max[:,1] +=1

        x_max_y_min = pixel_positions_clamp.int()
        x_max_y_min[:,0] +=1
        x_max_y_max = pixel_positions_clamp.int()
        x_max_y_max[:,0] +=1
        x_max_y_max[:,1] +=1


        def get_indices(x_y, valid_indices):
            img_indices = (x_y[:, 1] * self.w + x_y[:, 0]).long()
            outlier_mask = img_indices < 0
            outlier_mask2 = img_indices >= self.w * self.h
            outlier_mask = torch.logical_or(outlier_mask, outlier_mask2)
            indices = valid_indices * self.h * self.w + img_indices
            return outlier_mask, indices

        outlier_mask_1, x_min_y_min_ind = get_indices(x_min_y_min, valid_indices)
        outlier_mask_2, x_min_y_max_ind = get_indices(x_min_y_max, valid_indices)
        outlier_mask_3, x_max_y_min_ind = get_indices(x_max_y_min, valid_indices)
        outlier_mask_4, x_max_y_max_ind = get_indices(x_max_y_max, valid_indices)

        outlier_masks = torch.logical_or(torch.logical_or(torch.logical_or(outlier_mask_1, outlier_mask_2), outlier_mask_3), outlier_mask_4)

        x_min_y_min_ind[outlier_mask_1] = 0
        x_min_y_max_ind[outlier_mask_2] = 0
        x_max_y_min_ind[outlier_mask_3] = 0
        x_max_y_max_ind[outlier_mask_4] = 0

        # img_view = self.imgs.view(-1,3)
        # feature_view = self.features.view(-1,self.features.shape[-1])
        color_x_min_y_min = self.imgs.view(-1, self.imgs.shape[-1])[x_min_y_min_ind]
        color_x_min_y_max = self.imgs.view(-1, self.imgs.shape[-1])[x_min_y_max_ind]
        color_x_max_y_min = self.imgs.view(-1, self.imgs.shape[-1])[x_max_y_min_ind]
        color_x_max_y_max = self.imgs.view(-1, self.imgs.shape[-1])[x_max_y_max_ind]

        x_diff = (pixel_positions_clamp - x_min_y_min)[:, 0]
        y_diff = (pixel_positions_clamp - x_min_y_min)[:, 1]

        color_y_min =  x_diff.unsqueeze(1) * color_x_max_y_min + (1-x_diff.unsqueeze(1)) * color_x_min_y_min
        color_y_max =  x_diff.unsqueeze(1) * color_x_max_y_max + (1-x_diff.unsqueeze(1)) * color_x_min_y_max

        color = y_diff.unsqueeze(1) * color_y_max + (1-y_diff.unsqueeze(1)) * color_y_min
        color[outlier_masks] *=0.0
        return color


    def bilinear_sample_imgs(self, pixel_positions, valid_indices):
        # img = imgs[0]
        #Get four corners
        pixel_positions_clamp = pixel_positions[:,:2]
        pixel_positions_clamp = self.clamp_positions(pixel_positions_clamp)
        x_min_y_min = pixel_positions_clamp.int()
        x_min_y_max = pixel_positions_clamp.int()
        x_min_y_max[:,1] +=1

        x_max_y_min = pixel_positions_clamp.int()
        x_max_y_min[:,0] +=1
        x_max_y_max = pixel_positions_clamp.int()
        x_max_y_max[:,0] +=1
        x_max_y_max[:,1] +=1


        def get_indices(x_y, valid_indices):
            img_indices = (x_y[:, 1] * self.w + x_y[:, 0]).long()
            outlier_mask = img_indices < 0
            outlier_mask2 = img_indices >= self.w * self.h
            outlier_mask = torch.logical_or(outlier_mask, outlier_mask2)
            indices = valid_indices * self.h * self.w + img_indices
            return outlier_mask, indices

        outlier_mask_1, x_min_y_min_ind = get_indices(x_min_y_min, valid_indices)
        outlier_mask_2, x_min_y_max_ind = get_indices(x_min_y_max, valid_indices)
        outlier_mask_3, x_max_y_min_ind = get_indices(x_max_y_min, valid_indices)
        outlier_mask_4, x_max_y_max_ind = get_indices(x_max_y_max, valid_indices)

        outlier_masks = torch.logical_or(torch.logical_or(torch.logical_or(outlier_mask_1, outlier_mask_2), outlier_mask_3), outlier_mask_4)

        x_min_y_min_ind[outlier_mask_1] = 0
        x_min_y_max_ind[outlier_mask_2] = 0
        x_max_y_min_ind[outlier_mask_3] = 0
        x_max_y_max_ind[outlier_mask_4] = 0

        # img_view = self.imgs.view(-1,3)
        # feature_view = self.features.view(-1,self.features.shape[-1])
        color_x_min_y_min = self.depths.view(-1,self.depths.shape[-1])[x_min_y_min_ind]
        color_x_min_y_max = self.depths.view(-1,self.depths.shape[-1])[x_min_y_max_ind]
        color_x_max_y_min = self.depths.view(-1,self.depths.shape[-1])[x_max_y_min_ind]
        color_x_max_y_max = self.depths.view(-1,self.depths.shape[-1])[x_max_y_max_ind]

        color2_x_min_y_min = self.imgs.view(-1, self.imgs.shape[-1])[x_min_y_min_ind]
        color2_x_min_y_max = self.imgs.view(-1, self.imgs.shape[-1])[x_min_y_max_ind]
        color2_x_max_y_min = self.imgs.view(-1, self.imgs.shape[-1])[x_max_y_min_ind]
        color2_x_max_y_max = self.imgs.view(-1, self.imgs.shape[-1])[x_max_y_max_ind]

        color_x_min_y_min = torch.cat([color_x_min_y_min, color2_x_min_y_min], dim=1)
        color_x_min_y_max = torch.cat([color_x_min_y_max, color2_x_min_y_max], dim=1)
        color_x_max_y_min = torch.cat([color_x_max_y_min, color2_x_max_y_min], dim=1)
        color_x_max_y_max = torch.cat([color_x_max_y_max, color2_x_max_y_max], dim=1)


        x_diff = (pixel_positions_clamp - x_min_y_min)[:, 0]
        y_diff = (pixel_positions_clamp - x_min_y_min)[:, 1]

        color_y_min =  x_diff.unsqueeze(1) * color_x_max_y_min + (1-x_diff.unsqueeze(1)) * color_x_min_y_min
        color_y_max =  x_diff.unsqueeze(1) * color_x_max_y_max + (1-x_diff.unsqueeze(1)) * color_x_min_y_max

        color = y_diff.unsqueeze(1) * color_y_max + (1-y_diff.unsqueeze(1)) * color_y_min
        color[outlier_masks] *=0.0
        return color


    # @torch.cuda.amp.autocast()
    @batchify_gt_color
    def forward(self, xyzs, dirs=None, gt_colors = None,  **kwargs):  # (B*N,3)  (B*N,3)
        grid_indice_list, ratio_list = self.locate_grids(xyzs)
        for i in range(len(grid_indice_list)):
            grid_indice_list[i][grid_indice_list[i]<0]=0
            grid_indice_list[i][grid_indice_list[i]>=len(self.grids_coords)]=len(self.grids_coords)-1

        # sample_rgbs_list = []
        # sample_vars_list = []
        # for i in range(len(grid_indice_list)):
        #     sample_rgbs_list.append(self.mean_rgbs[grid_indice_list[i]])
        #     sample_vars_list.append(self.var_rgbs[grid_indice_list[i]])

        #Trilinear interpolation
        # rgb_x_min_y_min = sample_rgbs_list[0] * ratio_list[:,2:] + sample_rgbs_list[1] * (1-ratio_list[:,2:])
        # rgb_x_min_y_max = sample_rgbs_list[2] * ratio_list[:,2:] + sample_rgbs_list[3] * (1-ratio_list[:,2:])
        # rgb_x_max_y_min = sample_rgbs_list[4] * ratio_list[:,2:] + sample_rgbs_list[5] * (1-ratio_list[:,2:])
        # rgb_x_max_y_max = sample_rgbs_list[6] * ratio_list[:,2:] + sample_rgbs_list[7] * (1-ratio_list[:,2:])
        # rgb_x_min = rgb_x_min_y_min * ratio_list[:,1:2] + rgb_x_min_y_max * (1-ratio_list[:,1:2])
        # rgb_x_max = rgb_x_max_y_min * ratio_list[:, 1:2] + rgb_x_max_y_max * (1 - ratio_list[:, 1:2])
        # interpolate_rgbs = rgb_x_min * ratio_list[:,0:1] + rgb_x_max * (1-ratio_list[:,0:1])
        vertex_distance = torch.min(torch.cat([ratio_list.unsqueeze(0), 1 - ratio_list.unsqueeze(0)], dim=0), dim=0)[0]
        ratio_list[ratio_list < 0.5] = 0
        ratio_list[ratio_list > 0.5] = 1
        ratio_list = ratio_list.long()
        ratio_indices = ratio_list[:, 2] + ratio_list[:, 1] * 2 + ratio_list[:, 0] * 4
        vertex_indices = torch.stack(grid_indice_list).permute(1, 0).gather(1, ratio_indices.unsqueeze(1)).squeeze(1)
        interpolate_rgbs = torch.gather(self.all_rgbs, 1, vertex_indices.unsqueeze(0).unsqueeze(2).repeat(3,1,6))

        distance = torch.norm((vertex_distance - interpolate_rgbs[:,:,3:]), dim=2)
        used_indices = torch.min(distance, dim=0)[1]
        interpolate_rgbs = torch.gather(interpolate_rgbs, 0, used_indices.unsqueeze(0).unsqueeze(2).repeat(1,1,6)).squeeze(0)[:,:3]
        # print('Finish selection')
        # interpolate_rgbs = (sample_rgbs_list.permute(1, 2, 0).gather(2, ratio_indices.unsqueeze(1).repeat(1, 3).unsqueeze(2))).squeeze(2)
        # interpolate_vars = (sample_var_list.permute(1, 2, 0).gather(2, ratio_indices.unsqueeze(1).repeat(1, 3).unsqueeze(2))).squeeze(2)

        # xyz_projections = torch.einsum("ijk,lk->lij", [self.R_cws, xyzs]) + self.t_cws
        # dists = xyz_projections.norm(dim=2)
        # mask1 = xyz_projections[:,:,2] < 0.1
        # xyz_projections = xyz_projections/xyz_projections[:,:,2].unsqueeze(2)
        # xyz_projections = xyz_projections @ self.K.T
        # #get projections inner images
        # mask2 = xyz_projections[:, :, 0] < 0
        # mask3 = xyz_projections[:, :, 0] > self.w-2
        # mask4 = xyz_projections[:, :, 1] < 0
        # mask5 = xyz_projections[:, :, 1] > self.h-2
        # mask_outer = torch.logical_or(mask1, mask2)
        # mask_outer = torch.logical_or(mask_outer, mask3)
        # mask_outer = torch.logical_or(mask_outer, mask4)
        # mask_outer = torch.logical_or(mask_outer, mask5)
        # dists[mask_outer] = 2e6
        # 
        # k=5
        # choose_indices = []
        # choose_dists = []
        # for i in range(k):
        #     min_dists, min_indices = torch.min(dists, dim=1)
        #     dists.view(-1)[torch.IntTensor([m for m in range(len(xyzs))]).to(xyzs.device) * len(self.indices) + min_indices] = 2e6+1
        #     min_indices[min_dists > 1e6] = -1
        #     choose_dists.append(min_dists)
        #     choose_indices.append(min_indices)
        # 
        # all_rgbs = []
        # all_counts = []
        # for choose_ind in choose_indices:
        #     # rgbs = torch.ones_like(xyzs) * -1
        #     rgbs = torch.ones([len(xyzs), self.depths.shape[-1] * 2 + self.imgs.shape[-1]]).to(self.imgs.device)
        #     counts = torch.zeros_like(xyzs)[:,0]
        #     valid_indices = choose_ind[choose_ind >= 0]
        #     R_cw = self.R_cws[valid_indices]
        #     t_cw = self.t_cws[valid_indices]
        #     step_xyzs = xyzs[choose_ind >= 0]
        #     step_xyzs = (torch.einsum("ijk,ik->ij", [R_cw, step_xyzs]) + t_cw)
        #     dists = (step_xyzs[:, 2].unsqueeze(1))
        #     step_xyzs = step_xyzs/ (step_xyzs[:, 2].unsqueeze(1))
        #     step_xyzs = step_xyzs @ self.K.T
        #     colors = self.bilinear_sample_imgs(step_xyzs, valid_indices)
        #     dists = (dists - colors[:,:1]) * (dists - colors[:,:1])
        #     colors = torch.cat([colors, dists], dim=1)
        #     rgbs[choose_ind >= 0] = rgbs[choose_ind >= 0] * 0.0 + colors
        #     counts[choose_ind>=0] = 1.
        #     all_rgbs.append(rgbs)
        #     all_counts.append(counts)
        # 
        # 
        # all_rgbs = torch.stack(all_rgbs)
        # all_counts = torch.stack(all_counts)
        # comput_counts = torch.sum(all_counts, dim=0).unsqueeze(1)
        # no_data_mask = comput_counts<(1-(1e-3))
        # comput_counts[comput_counts<1e-3] = 1e-3
        # mean_rgbs = torch.sum(all_rgbs*all_counts.unsqueeze(2), dim=0)/comput_counts
        # var_rgbs = torch.sum(((all_rgbs - mean_rgbs) * all_counts.unsqueeze(2))**2, dim=0)/comput_counts
        # var_rgbs = torch.sqrt(var_rgbs)
        # compensation = torch.zeros_like(var_rgbs)
        # compensation[no_data_mask.squeeze(1)] = 1.0
        # var_rgbs = var_rgbs + compensation



        # with torch.cuda.amp.autocast():
        # mean_rgbs = mean_rgbs[:,1:4]
        # var_rgbs = var_rgbs[:,1:4]
        original_xyzs = xyzs
        xyzs = (xyzs + self.aabb) / self.aabb / 2  # (0,1)

        feat = self.xyz_hashes(xyzs)
        #cat color
        # feat = torch.cat([feat, mean_rgbs, var_rgbs], dim=1)
        #cat depth
        # feat = torch.cat([feat, mean_rgbs[:, 0:1], mean_rgbs[:, 4:], var_rgbs[:, 0:1], var_rgbs[:, 4:]], dim=1)

        with torch.cuda.amp.autocast():

            h = self.xyz_encoder(feat)
            # h = self.xyz_encoder_sigma_net(xyzs)
            # sigma = TruncExp.apply(h[:, 0] + ((1.5)*1e-3)/(mean_rgbs[:,4]/(mean_rgbs[:,0]**2+1e-5) +1e-5))
            sigma = TruncExp.apply(h[:, 0])
            if dirs is None:
                return sigma
            else:

                # dirs = dirs/torch.norm(dirs, dim=1, keepdim=True)
                # dirs = self.dir_encoder((dirs+1)/2)
                dirs = self.dir_encoder(dirs)
                predicted_color = self.color_net(torch.cat([dirs, h], 1))
                # color = predicted_color[:,:3]
                # color = interpolate_rgbs
                color = predicted_color[:,:3]
                # color2 = interpolate_rgbs
                color2 = predicted_color[:, 3:] * predicted_color[:,:3] + (1 - predicted_color[:, 3:]) * interpolate_rgbs

                if self.remove_outliers and gt_colors is not None:
                    # mean_dist = torch.mean((color2 - gt_colors).norm(dim=1))
                    mean_dist = 0.05
                    vertex_indices = torch.stack(grid_indice_list).permute(1,0).gather(1, ratio_indices.unsqueeze(1)).squeeze(1)
                    update_vertex_indices = vertex_indices[(color2 - gt_colors).norm(dim=1) > mean_dist]
                    self.update_indices[update_vertex_indices] = self.update_indices[update_vertex_indices]+1
                    self.update_indices[self.update_indices>=self.all_rgbs.shape[0]] = 0
                    update_index = self.update_indices[update_vertex_indices]
                    start_vertices_indices = (update_vertex_indices * 6 + update_index.long() * self.all_rgbs.shape[1] * self.all_rgbs.shape[2]).long()
                    all_color_vertices_indices = torch.cat([start_vertices_indices.unsqueeze(1),(start_vertices_indices+1).unsqueeze(1), (start_vertices_indices+2).unsqueeze(1)], dim=1)
                    self.all_rgbs.view(-1)[all_color_vertices_indices] = gt_colors[:,:3][(color2 - gt_colors).norm(dim=1) > mean_dist].float()
                    self.all_rgbs.view(-1)[all_color_vertices_indices+3] = vertex_distance[(color2 - gt_colors).norm(dim=1) > mean_dist]



                    # self.mean_rgbs[vertex_indices[(color - gt_colors).norm(dim=1) > 0.2]] = gt_colors[
                    #     (color - gt_colors).norm(dim=1) > 0.2]
                    # self.mean_rgbs[vertex_indices[torch.exp(-sigma) < 0.1]] = self.mean_rgbs[vertex_indices[torch.exp(-sigma) < 0.1]] * 0.0

                # color = torch.tanh(predicted_color/100.) +  interpolate_rgbs
                # color = torch.clamp(color, 0., 1.)


                # diff = color - interpolate_rgbs
                # predicted_color[:, :1] * predicted_color[:, 1:] + (1 - predicted_color[:, :1]) *
                # interpolate_rgbs[interpolate_rgbs<(1e-5)] = 1e-5
                # interpolate_vars[interpolate_vars<(1e-5)] = 1e-5
                # interpolate_rgbs[interpolate_rgbs>(1-(1e-5))] = 1-(1e-5)
                # interpolate_vars[interpolate_vars>(1-(1e-5))] = 1-(1e-5)
                # features = torch.cat([color, torch.logit(interpolate_rgbs), torch.logit(interpolate_vars)], dim=1)


                # color = color[:,:3] * color[:,3:] + mean_rgbs[:,1:4] * (1- color[:,3:])
                # color = torch.sigmoid(color)
                # + mean_rgbs[:,1:4]


                # mean_rgbs[:, 1:4]

                # color = mean_rgbs[:,1:4]


                # color = mean_rgbs[:,1:4]




        return color2, color, sigma



#backup
 # time_start = ti2.time()
 # 
 #        # torch.unique
 #        xyz_projections = torch.einsum("ijk,lk->lij", [self.R_cws, xyzs]) + self.t_cws
 #        dists = xyz_projections.norm(dim=2)
 #        mask1 = xyz_projections[:,:,2] < 0.1
 #        xyz_projections = xyz_projections/xyz_projections[:,:,2].unsqueeze(2)
 #        xyz_projections = xyz_projections @ self.K.T
 #        #get projections inner images
 #        mask2 = xyz_projections[:, :, 0] < 0
 #        mask3 = xyz_projections[:, :, 0] > self.w-2
 #        mask4 = xyz_projections[:, :, 1] < 0
 #        mask5 = xyz_projections[:, :, 1] > self.h-2
 #        mask_outer = torch.logical_or(mask1, mask2)
 #        mask_outer = torch.logical_or(mask_outer, mask3)
 #        mask_outer = torch.logical_or(mask_outer, mask4)
 #        mask_outer = torch.logical_or(mask_outer, mask5)
 #        dists[mask_outer] = 2e6
 # 
 # 
 #        time_1 = ti2.time()
 # 
 #        k=5
 #        choose_indices = []
 #        choose_dists = []
 #        for i in range(k):
 #            min_dists, min_indices = torch.min(dists, dim=1)
 #            dists.view(-1)[torch.IntTensor([m for m in range(len(xyzs))]).to(xyzs.device) * len(self.indices) + min_indices] = 2e6+1
 #            min_indices[min_dists > 1e6] = -1
 #            choose_dists.append(min_dists)
 #            choose_indices.append(min_indices)
 # 
 #        time_2 = ti2.time()
 # 
 #        all_rgbs = []
 #        all_counts = []
 #        for choose_ind in choose_indices:
 #            # rgbs = torch.ones_like(xyzs) * -1
 #            rgbs = torch.ones([len(xyzs), self.depths.shape[-1] * 2 + self.imgs.shape[-1]]).to(self.imgs.device)
 #            counts = torch.zeros_like(xyzs)[:,0]
 #            valid_indices = choose_ind[choose_ind >= 0]
 #            R_cw = self.R_cws[valid_indices]
 #            t_cw = self.t_cws[valid_indices]
 #            step_xyzs = xyzs[choose_ind >= 0]
 #            step_xyzs = (torch.einsum("ijk,ik->ij", [R_cw, step_xyzs]) + t_cw)
 #            dists = (step_xyzs[:, 2].unsqueeze(1))
 #            step_xyzs = step_xyzs/ (step_xyzs[:, 2].unsqueeze(1))
 #            # step_xyzs /= step_xyzs[:,2].unsqueeze(1)
 #            step_xyzs = step_xyzs @ self.K.T
 #            colors = self.bilinear_sample_imgs(step_xyzs, valid_indices)
 #            dists = (dists - colors[:,:1]) * (dists - colors[:,:1])
 #            colors = torch.cat([colors, dists], dim=1)
 #            rgbs[choose_ind >= 0] = rgbs[choose_ind >= 0] * 0.0 + colors
 #            counts[choose_ind>=0] = 1.
 #            all_rgbs.append(rgbs)
 #            all_counts.append(counts)
 # 
 # 
 #        time_3 = ti2.time()
 # 
 #        all_rgbs = torch.stack(all_rgbs)
 #        all_counts = torch.stack(all_counts)
 #        comput_counts = torch.sum(all_counts, dim=0).unsqueeze(1)
 #        no_data_mask = comput_counts<(1-(1e-3))
 #        comput_counts[comput_counts<1e-3] = 1e-3
 #        mean_rgbs = torch.sum(all_rgbs*all_counts.unsqueeze(2), dim=0)/comput_counts
 #        var_rgbs = torch.sum(((all_rgbs - mean_rgbs) * all_counts.unsqueeze(2))**2, dim=0)/comput_counts
 #        var_rgbs = torch.sqrt(var_rgbs)
 #        compensation = torch.zeros_like(var_rgbs)
 #        compensation[no_data_mask.squeeze(1)] = 1.0
 #        var_rgbs = var_rgbs + compensation
 # 
 # 
 #        time_4 = ti2.time()
 #        print('Time cost 1 is: ', time_1-time_start)
 #        print('Time cost 2 is: ', time_2-time_start)
 #        print('Time cost 3 is: ', time_3-time_start)
 #        print('Time cost 4 is: ', time_4-time_start)