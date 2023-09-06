import sys
import os
from os.path import join
import json
import numpy as np
import torch
from torch import FloatTensor
from torch.utils.data import Dataset
from ipdb import set_trace as S
from utils import *
import glob
from tqdm import tqdm
import cv2
import pytorch3d.transforms

class BaseDataset(Dataset):
    
    def __init__(self, hparams, split, **kwargs):
        assert split in ['train', 'test_traj', 'test', 'train_all', 'test_all'], "Undefined split."
        self.split = split
        self.data_name = hparams.data_name
        self.root_dir = join(hparams.data_root, hparams.data_name)
        self.downsample = hparams.downsample
        self.loss = hparams.Loss
        self.sampling = hparams.Sampling
        if self.sampling in ['Vanilla']:
            self.near, self.far = hparams.near, hparams.far
        if 'vis' in kwargs.keys():
            self.vis = kwargs['vis']
        else:
            self.vis = None
        self.check_pose = kwargs.get("check_pose", False)
        self.crop = False
        self.crop_size = 15
        self.depth_replica = False
        self.npz = False
    
    def __len__(self):
        return len(self.all_rays)
    
    def __getitem__(self, idx):
        sample = {"rays": self.all_rays[idx]} # train: (8)->(B, 8); test: (h*w, 8)->(1, h*w, 8)
        if 'idx_ls' in self.__dict__:
            sample['indices'] = self.indices[idx]
            sample['max_index'] = self.max_index
        if 'all_colors' in self.__dict__: 
            sample["colors"] = self.all_colors[idx] # train: (3)->(B, 3); test: (h, w, 3)->(1, h,w, 3)
        if 'all_depths' in self.__dict__: 
            sample["depths"] = self.all_depths[idx] # train: (3)->(B, 3); test: (h, w, 3)->(1, h,w, 3)
        if 'all_semantics' in self.__dict__: 
            sample["semantics"] = self.all_semantics[idx] # train: (3)->(B, 3); test: (h, w, 3)->(1, h,w, 3)
        if 'times' in self.__dict__:
            sample["times"] = self.times[idx]
        return sample
    
    '''before calling center, make sure that you have self.poses: (N,3,4) FloatTensor
    function: center self.poses
    '''
    def center(self):
        print('\nbefore center: ', self.poses[:,:,3].min(), self.poses[:,:,3].max())
        if 'pts3d' in self.__dict__:  
            self.poses, self.pts3d = center_poses(self.poses, self.pts3d)
            self.poses = FloatTensor(self.poses)
        else:    
            self.poses = FloatTensor(center_poses(self.poses))
        print('after center: ', self.poses[:,:,3].min(), self.poses[:,:,3].max())
    
    '''before calling scale_to, make sure that you have self.poses: (N,3,4) FloatTensor
    function: scale self.poses to [-scale_to, scale_to]
    '''
    def scale_to(self, scale_to=None, scale=None):
        print('\nbefore scale: ', self.poses[:,:,3].min(), self.poses[:,:,3].max())
        if scale_to is None:
            if scale is None:
                self.scale = np.linalg.norm(self.poses[..., 3], axis=-1).min()
            else:
                self.scale = scale
            # scale = max(self.poses.min().abs(), self.poses.max())/1.5
        elif scale_to == -1:
            self.scale = 1
        else:
            self.scale = max(self.poses.min().abs(), self.poses.max())/scale_to
        print(f'scale = {self.scale}')
        self.poses[..., 3] /= self.scale
        if 'pts3d' in self.__dict__:  
            self.pts3d /= self.scale
        print('after scale: ', self.poses[:,:,3].min(), self.poses[:,:,3].max(), '\n')
    '''before calling buffer, make sure that you have self.poses: (N,3,4) FloatTensor
    function: creat buffer:  
                    self.all_rays (N, 6) 
                    self.all_colors if has self.img_paths (N)
                    self.all_depths if has self.depth_paths (N)
                    self.all_semantics if has self.semantics_paths (N)
    '''

    def feature_grids_dim(self, h, w):
        return h*w

    def buffer(self, blend_a, background=None):


        self.all_rays = torch.empty((len(self.poses), self.h*self.w, 6), dtype=torch.float)

        self.camera_rays = torch.empty((self.h*self.w, 6), dtype=torch.float)
        #Generate Features
        grids_dim = self.feature_grids_dim(self.h, self.w)
        feature_dim = 13
        self.all_features = torch.empty((len(self.poses), grids_dim, feature_dim), dtype=torch.float)

        # def reset_parameters(self):
        #     std = 1e-4
        #     self.embeddings.data.uniform_(-std, std)

        if "idx_ls" in self.__dict__:
            self.indices = torch.empty((len(self.poses), self.h, self.w, 1))
            self.max_index = max(self.idx_ls)
        if "img_paths" in self.__dict__:
            self.all_colors = torch.empty((len(self.img_paths), self.h, self.w, 3))
            print(f'Loading {len(self.img_paths)} {self.split} images of shape ({self.h},{self.w})')
        if "depth_paths" in self.__dict__:
            self.all_depths = torch.empty((len(self.depth_paths), self.h, self.w))
            print(f'Loading {len(self.depth_paths)} {self.split} depth map of shape ({self.h},{self.w})')
        if "semantics_paths" in self.__dict__:
            self.all_semantics = np.empty((len(self.semantics_paths), self.h, self.w))
            print(f'Loading {len(self.semantics_paths)} {self.split} semantics map of shape ({self.h},{self.w})')

        camera_rays_o, camera_rays_d = get_camera_rays(self.directions)
        self.camera_rays[:,:3] = camera_rays_o
        self.camera_rays[:, 3:6] = camera_rays_d

        # #TODO: 删除这些测试代码
        # depth_scales = torch.load('/home/kyrie/Documents/Machine_learning/DeepSLAM/ModuleNerf2/ModuleNeRF-main/scales/scale_family.pt')
        # depth_shifts = torch.load('/home/kyrie/Documents/Machine_learning/DeepSLAM/ModuleNerf2/ModuleNeRF-main/scales/shift_family.pt')


        for i in tqdm(range(len(self.poses)), desc='Preparing rays buffer'):
            c2w = self.poses[i]
            rays_o, rays_d = get_rays(self.directions, c2w)
            # if 'is_spheric' in self.__dict__ and not self.is_spheric: # llff
            #     rays_o, rays_d = get_ndc_rays(self.h, self.w, self.focal, 1.0, rays_o, rays_d)
            self.all_rays[i,:,:3] = rays_o
            self.all_rays[i,:,3:6] = rays_d
            if "idx_ls" in self.__dict__:
                self.indices[i] = self.idx_ls[i]
            if "img_paths" in self.__dict__:
                self.all_colors[i] = read_resize_img_01tensor_hwc(self.img_paths[i], (self.h,self.w), blend_a=blend_a, background=background, crop = self.crop, crop_size=self.crop_size)
            if "depth_paths" in self.__dict__:
                if self.npz:
                    self.all_depths[i] = read_resize_scale_depth_tensor_npz(self.depth_paths[i], (self.h, self.w),
                                                                        self.scale, crop=self.crop,
                                                                        crop_size=self.crop_size,
                                                                        depth_replica=self.depth_replica)
                    # local_scale = depth_scales[int(i/2)]
                    # local_shift = depth_shifts[int(i/2)]
                    # next_index = int(i/2)+1
                    # if next_index >= len(depth_scales):
                    #     next_index = len(depth_scales)-1
                    # next_scale = depth_scales[next_index]
                    # next_shift = depth_shifts[next_index]
                    # next_ratio = (i-int(i/2)*2)/2
                    # current_scale = local_scale*(1-next_ratio) + next_scale * next_ratio
                    # current_shift = local_shift*(1-next_ratio) + next_shift * next_ratio
                    # self.all_depths[i] = self.all_depths[i] * current_scale.item() + current_shift.item() / self.scale
                else:
                    self.all_depths[i] = read_resize_scale_depth_tensor(self.depth_paths[i], (self.h,self.w), self.scale, crop = self.crop, crop_size=self.crop_size, depth_replica=self.depth_replica)
            if "semantics_paths" in self.__dict__:
                self.all_semantics[i] = read_resize_semantic_tensor(self.semantics_paths[i], (self.h,self.w))

        if self.split.startswith('trains'):
            self.all_rays = self.all_rays.reshape(-1,6) # (N_images*h*w, 8)
            if "indices" in self.__dict__:
                self.indices = self.indices.reshape(-1)
            if "img_paths" in self.__dict__:
                self.all_colors = self.all_colors.reshape(-1,3) # (N_images*h*w, 3)
            if "depth_paths" in self.__dict__:
                self.all_depths = self.all_depths.reshape(-1)
                print("depths min:", self.all_depths.min().item(), "depths max:", self.all_depths.max().item())
            if "semantics_paths" in self.__dict__:
                self.all_semantics = self.all_semantics.reshape(-1)

        if self.check_pose:
            self.visualize_poses()

    def visualize_poses(self):
        if self.vis is None:
            self.vis = create_vis()
        if 'pts3d' in self.__dict__:
            view_pts3d(self.vis, self.pts3d, self.pts3d_color)
        if 'mesh_path' in self.__dict__:
            view_mesh(self.vis, self.mesh_path)
        view_poses(self.vis, self.K, self.h, self.w, self.poses)
        color = [0,0,1] if self.split=='train' else [1,0,0]
        if self.sampling in ['Vanilla']:
            view_rays(self.vis, self.all_rays.reshape(-1, self.h, self.w, 6), color, self.near, self.far)
        # elif self.sampling in ['NGP']:
        #     view_rays(self.vis, self.all_rays.reshape(-1, self.h, self.w, 6), color, 0, 1)
        if not self.split.startswith('train'):
            show_vis(self.vis)
            # sys.exit()


class Blender(BaseDataset):
    def __init__(self, hparams, split="train", **kwargs):
        super().__init__(hparams, split, **kwargs)
        self.h = self.w = int(800*self.downsample)
        
        if split.endswith("all"):
            with open(join(self.root_dir, f"transforms_train.json")) as file:
                meta = json.load(file)
        else:
            with open(join(self.root_dir, f"transforms_{split}.json")) as file:
                meta = json.load(file)


        self.focal = (0.5*self.h)/np.tan(0.5*meta['camera_angle_x'])
        self.K = FloatTensor([[self.focal, 0, self.w/2],
                            [0, self.focal, self.h/2],
                            [0,  0,  1]])

        crop_frac = 0.5
        crop_w = int(self.w//2 * crop_frac)
        crop_h = int(self.h//2 * crop_frac)
        self.directions = get_ray_directions(self.h, self.w, self.K) # (h,w,3)
        self.directions_grid = create_meshgrid(self.h, self.w, normalized_coordinates=False)[0]
        self.directions_mask =  (self.directions_grid[:, :, 0] >= crop_w) & (self.directions_grid[:, :, 1] >= crop_w)
        self.directions_mask = (self.directions_mask) & (self.directions_grid[:, :, 0] < self.w - crop_w)
        self.directions_mask = (self.directions_mask) & (self.directions_grid[:, :, 1] < self.h - crop_h)
        self.directions_mask = self.directions_mask.contiguous().view(-1)

        self.idx_ls = [i for i in range(len(meta['frames']))]

        # meta['frames'] = meta['frames'][:10]
        self.poses = torch.empty(len(meta['frames']), 3, 4)
        self.times = torch.empty(len(meta['frames']))
        for i, frame in enumerate(meta['frames']):
            self.poses[i] = FloatTensor(frame['transform_matrix'])[:3,:4]
            self.poses[i][:, 1:3] *= -1 # [right up back] to [right down front]
            # self.times[i] = frame['time']

        # if self.sampling in ["NGP"]:
        #     self.scale_to(scale=2.7)
        
        self.img_paths = [join(self.root_dir, f"{frame['file_path']}.png") for frame in meta['frames']]
        background = 'random' if 'train' in self.split else 'white'
        self.buffer(blend_a=True, background='black') 
#
#
# class Colmap(BaseDataset):
#     def __init__(self, hparams, split='train', **kwargs):
#         super().__init__(hparams, split, **kwargs)
#
#         camera_data = read_cameras_binary(join(self.root_dir, 'sparse/0/cameras.bin'))
#         self.h = int(round(camera_data[1].height*self.downsample))
#         self.w = int(round(camera_data[1].width*self.downsample))
#         if camera_data[1].model == 'SIMPLE_RADIAL':
#             fx = fy = camera_data[1].params[0]*self.downsample
#             cx = camera_data[1].params[1]*self.downsample
#             cy = camera_data[1].params[2]*self.downsample
#         elif camera_data[1].model in ['PINHOLE', 'OPENCV']:
#             fx = camera_data[1].params[0]*self.downsample
#             fy = camera_data[1].params[1]*self.downsample
#             cx = camera_data[1].params[2]*self.downsample
#             cy = camera_data[1].params[3]*self.downsample
#         else:
#             raise ValueError(f"Please parse the intrinsics for camera model {camera_data[1].model}!")
#         self.K = FloatTensor([[fx, 0, cx],
#                             [0, fy, cy],
#                             [0,  0,  1]])
#         self.directions = get_ray_directions(self.h, self.w, self.K)
#
#         img_data = read_images_binary(join(self.root_dir, 'sparse/0/images.bin'))
#         img_names = [img_data[k].name for k in img_data.keys()]
#         # img_names = img_names[::int(len(img_names)/100)+1]
#         orders = np.argsort(img_names) # for matching the order of self.poses with sorted(self.img_paths)
#         img_names = sorted(img_names)
#
#         if os.path.exists(join(self.root_dir, f'images_{int(1/self.downsample)}')):
#             folder = f'images_{int(1/self.downsample)}'
#         else:
#             folder = 'images'
#         print('img_folder: ', folder)
#
#         img_paths = [join(self.root_dir, folder, name) for name in img_names]
#         w2c_mats = []
#         bottom = np.array([[0, 0, 0, 1.]])
#         for k in img_data.keys():
#             im = img_data[k]
#             R = im.qvec2rotmat()
#             t = im.tvec.reshape(3, 1)
#             w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
#         w2c_mats = np.stack(w2c_mats, 0)
#         self.poses = np.linalg.inv(w2c_mats)
#         self.poses = FloatTensor(self.poses[orders, :3]) # (N_images, 3, 4) cam2world matrices
#
#         self.pts3d = read_points3d_binary(join(self.root_dir, 'sparse/0/points3D.bin'))
#         self.pts3d_color = np.array([self.pts3d[k].rgb/255 for k in self.pts3d.keys()]) # (N, 3)
#         self.pts3d = np.array([self.pts3d[k].xyz for k in self.pts3d.keys()]) # (N, 3)
#
#         self.center()
#         self.scale_to(scale_to=0.4)
#
#         if split =='train':
#             self.img_paths = [x for i, x in enumerate(img_paths) if i%8!=0]
#             self.poses = self.poses[[i for i in range(len(img_paths)) if i%8!=0]]
#         elif split =='test':
#             self.img_paths = [x for i, x in enumerate(img_paths) if i%8==0]
#             self.poses = self.poses[[i for i in range(len(img_paths)) if i%8==0]]
#         elif split.endswith('all'):
#             self.img_paths = [x for i, x in enumerate(img_paths)]
#         elif split == 'test_traj':
#             self.poses = create_spheric_poses(1.2, self.poses[:, 1, 3].mean())
#             self.poses = FloatTensor(self.poses)
#         else:
#             raise
#
#         self.buffer(blend_a=False)


class Colmap(BaseDataset):
    def __init__(self, hparams, split='train', **kwargs):
        super().__init__(hparams, split, **kwargs)

        camera_data = read_cameras_binary(join(self.root_dir, 'sparse/0/cameras.bin'))
        self.h = int(round(camera_data[1].height * self.downsample))
        self.w = int(round(camera_data[1].width * self.downsample))
        if camera_data[1].model == 'SIMPLE_RADIAL':
            fx = fy = camera_data[1].params[0] * self.downsample
            cx = camera_data[1].params[1] * self.downsample
            cy = camera_data[1].params[2] * self.downsample
        elif camera_data[1].model in ['PINHOLE', 'OPENCV']:
            fx = camera_data[1].params[0] * self.downsample
            fy = camera_data[1].params[1] * self.downsample
            cx = camera_data[1].params[2] * self.downsample
            cy = camera_data[1].tparams[3] * self.downsample
        elif camera_data[1].model in ['SIMPLE_PINHOLE']:
            fx = camera_data[1].params[0] * self.downsample
            fy = camera_data[1].params[0] * self.downsample
            cx = camera_data[1].params[1] * self.downsample
            cy = camera_data[1].params[2] * self.downsample
        else:
            raise ValueError(f"Please parse the intrinsics for camera model {camera_data[1].model}!")
        self.K = FloatTensor([[fx, 0, cx],
                              [0, fy, cy],
                              [0, 0, 1]])
        self.directions = get_ray_directions(self.h, self.w, self.K)

        img_data = read_images_binary(join(self.root_dir, 'sparse/0/images.bin'))
        img_names = [img_data[k].name for k in img_data.keys()]
        # img_names = img_names[::int(len(img_names)/100)+1]
        orders = np.argsort(img_names)  # for matching the order of self.poses with sorted(self.img_paths)
        img_names = sorted(img_names)

        if os.path.exists(join(self.root_dir, f'images_{int(1 / self.downsample)}')):
            folder = f'images_{int(1 / self.downsample)}'
        else:
            folder = 'images'
        print('img_folder: ', folder)

        img_paths = [join(self.root_dir, folder, name) for name in img_names]
        w2c_mats = []
        bottom = np.array([[0, 0, 0, 1.]])
        for k in img_data.keys():
            im = img_data[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)
        self.poses = np.linalg.inv(w2c_mats)
        self.poses = FloatTensor(self.poses[orders, :3])  # (N_images, 3, 4) cam2world matrices
        #
        # self.pts3d = read_points3d_binary(join(self.root_dir, 'sparse/0/points3D.bin'))
        # self.pts3d_color = np.array([self.pts3d[k].rgb / 255 for k in self.pts3d.keys()])  # (N, 3)
        # self.pts3d = np.array([self.pts3d[k].xyz for k in self.pts3d.keys()])  # (N, 3)

        self.center()
        self.scale_to(scale_to=0.4)
        #
        # poses_bounds = np.load(join(self.root_dir, 'poses_bounds.npy'))  # (N_images, 17)
        # if split in ['train', 'test']:
        #     img_paths = sorted(glob.glob(join(self.root_dir, 'images/*')))  # load full resolution image then resize
        #     assert len(poses_bounds) == len(
        #         img_paths), 'Mismatch between number of images and number of poses! Please rerun COLMAP!'
        # bounds = poses_bounds[:, 15:]  # (N_images, 2)
        # poses_bounds = poses_bounds[:, :15].reshape(-1, 3, 5)
        # poses = poses_bounds[:, :, :4]  # (N_images, 3, 4)
        # # original intrinsics same for all images, rescale focal length according to training resolution
        # self.h, self.w, self.focal = poses_bounds[0, :, 4] * self.downsample
        # self.h, self.w = int(self.h), int(self.w)
        # self.K = FloatTensor([[self.focal, 0, self.w / 2],
        #                       [0, self.focal, self.h / 2],
        #                       [0, 0, 1]])
        # # ray directions for all pixels, same for all images (same H, W, focal)
        # self.directions = get_ray_directions(self.h, self.w, self.K)  # (H, W, 3)
        # '''
        # Step 2: correct poses
        # Original poses has rotation in form "down right back", change to "right down front"
        # See https://github.com/bmild/nerf/issues/34
        # '''
        # poses = np.concatenate([poses[..., 1:2], poses[..., 0:1], -poses[..., 2:4]],
        #                        -1)  # (N_images, 3, 4) exclude H, W, focal
        # poses = center_poses(poses)
        # '''
        # Step 3: correct scale so that the nearest depth is at a little more than 1.0
        # See https://github.com/bmild/nerf/issues/34
        # '''
        # near_original = bounds.min()
        # scale_factor = near_original * 0.75  # 0.75 is the default parameter, the nearest depth is at 1/0.75=1.33
        # bounds /= scale_factor
        # poses[..., 3] /= scale_factor

        # if split == 'train':
        #     self.img_paths = [x for i, x in enumerate(img_paths) if i % 8 != 0]
        #     self.poses = self.poses[[i for i in range(len(img_paths)) if i % 8 != 0]]
        # elif split == 'test':
        #     self.img_paths = [x for i, x in enumerate(img_paths) if i % 8 == 0]
        #     self.poses = self.poses[[i for i in range(len(img_paths)) if i % 8 == 0]]
        # elif split.endswith('all'):
        #     self.img_paths = [x for i, x in enumerate(img_paths)]
        # elif split == 'test_traj':
        #     self.poses = create_spheric_poses(1.2, self.poses[:, 1, 3].mean())
        #     self.poses = FloatTensor(self.poses)
        # else:
        #     raise

        self.img_paths = [x for i, x in enumerate(img_paths)]
        self.poses = self.poses[[i for i in range(len(img_paths))]]

        if 'Depth' in self.loss:
            depth_dir_name = 'npz'
            depth_list = os.listdir(join(self.root_dir, depth_dir_name))
            depth_list.sort(key=lambda x: int(x.split('.')[0][7:]))
            self.depth_paths = [join(self.root_dir, depth_dir_name, img) for i, img in enumerate(depth_list)]
            self.npz = True


        # self.img_paths = self.img_paths[::4]
        # self.poses = self.poses[::4]
        # self.depth_paths = self.depth_paths[::4]
        if split.startswith('train'):
            idx_ls = [i for i in range(len(self.img_paths)) if i%hparams['tracking_skip_interval']==0]
            self.img_paths = [self.img_paths[ind] for ind in idx_ls]
            self.poses = self.poses[idx_ls]
            self.depth_paths = [self.depth_paths[ind] for ind in idx_ls]





        self.buffer(blend_a=False)

        def sobel_2d(im):
            # sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')  #

            sobel_kernel_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype='float32')  #
            sobel_kernel_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype='float32')  #

            sobel_kernel_x = sobel_kernel_x.reshape((1, 1, 3, 3))
            sobel_kernel_y = sobel_kernel_y.reshape((1, 1, 3, 3))
            weight_x = torch.from_numpy(sobel_kernel_x).to(im.device)
            weight_y = torch.from_numpy(sobel_kernel_y).to(im.device)
            edge_detect_x = torch.nn.functional.conv2d(im.clone().unsqueeze(1), weight_x, padding=1)
            edge_detect_x = torch.abs(edge_detect_x.squeeze(1))/im

            edge_detect_y = torch.nn.functional.conv2d(im.clone().unsqueeze(1), weight_y, padding=1)
            edge_detect_y = torch.abs(edge_detect_y.squeeze(1)) / im

            return edge_detect_x, edge_detect_y


        depths = self.all_depths
        edges_x, edges_y = sobel_2d(depths)
        #TODO: Francis
        self.all_depths = (self.all_depths + hparams['aabb_shift']) * hparams['aabb_scale']
        self.all_depths[self.all_depths<3e-3] = 3e-3
        # cv2.imshow('depth', depths[0].detach().cpu().numpy()*3)
        # cv2.waitKey(0)
        self.all_depths[edges_x > 8 * (edges_x.mean(dim=1).mean(dim=1).unsqueeze(1).unsqueeze(2))] = 0.
        self.all_depths[edges_y > 8 *  (edges_y.mean(dim=1).mean(dim=1).unsqueeze(1).unsqueeze(2))] = 0.
        #
        # cv2.imshow('depth2', depths[0].detach().cpu().numpy()*3)
        # cv2.waitKey(0)



        
class Scannet(BaseDataset):
    def __init__(self, hparams, split='train', **kwargs):
        super().__init__(hparams, split, **kwargs)
        assert os.path.exists(self.root_dir)
        try:
            self.K = np.loadtxt(join(self.root_dir, 'intrinsic.txt'))[:3, :3]
        except:
            self.K = np.loadtxt(join(self.root_dir, 'intrinsic', 'intrinsic_color.txt'))[:3, :3]
        self.K[:2] *= self.downsample
        self.K = FloatTensor(self.K)
        img_dir_name = "images"
        start_frame = 0
        img_list = os.listdir(join(self.root_dir, img_dir_name))
        img_list.sort(key=lambda x: int(x.split('.')[0]))
        img_list = img_list[start_frame:]
        N_used = 100

        self.poses = torch.stack([FloatTensor(np.loadtxt(join(self.root_dir, 'pose', img.split('.')[0]+'.txt'))[:3,:4]) for img in img_list], 0)



        self.center()
        # self.scale_to(scale_to=0.7)
        self.scale_to(scale_to=0.4)
        # self.scale_to(scale_to=1)


        # load_poses = torch.load("estimated_poses.pth").detach().cpu()
        # load_poses_test = torch.load("estimated_poses_test.pth")
        #
        # self.poses[:200] = load_poses[:200,:3]


        # if split != 'test':
        #     for i in range(2,self.poses.shape[0]):
        #         self.poses[i,:,3] += torch.randn_like(self.poses[i,:,3]) * 0.02 * 0.2
        #         self.poses[i,:,:3] = torch.matmul(pytorch3d.transforms.axis_angle_to_matrix(torch.randn_like(self.poses[i,:,3]) * 0.01 * 0.2), self.poses[i,:,:3])
        #

        # diff = int(len(img_list)/N_used)+1
        # diff  = 2
        # if split == 'test':
        #     img_list = img_list[1:diff*N_used:diff]
        #     self.poses = self.poses[1:diff*N_used:diff]
        # else:
        diff = 1
        img_list = img_list[:diff*N_used:diff]
        self.poses = self.poses[:diff*N_used:diff]






        
        if split == 'train':
            idx_ls = [i for i in range(len(img_list)) ]
        elif split == 'test':
            idx_ls = [i for i in range(len(img_list)) ]
        elif split.endswith('all'):
            idx_ls = [i for i in range(len(img_list))]
        else:
            raise
        
        self.poses = self.poses[idx_ls]





        self.idx_ls = idx_ls
        self.img_paths = [join(self.root_dir, img_dir_name, img) for i, img in enumerate(img_list) if i in idx_ls]
        # if 'Depth' in self.loss:
        #     depth_dir_name = 'depth'
        #     depth_list = os.listdir(join(self.root_dir, depth_dir_name))
        #     depth_list.sort(key=lambda x: int(x.split('.')[0]))
        #     depth_list = depth_list[start_frame:]
        #     # if split == 'test':
        #     #     depth_list = depth_list[1:diff*N_used:diff]
        #     # else:
        #     depth_list = depth_list[:diff*N_used:diff]
        #     self.depth_paths = [join(self.root_dir, depth_dir_name, img) for i, img in enumerate(depth_list) if i in idx_ls]
        #

        if 'Depth' in self.loss:
            depth_dir_name = 'npz'
            depth_list = os.listdir(join(self.root_dir, depth_dir_name))
            depth_list.sort(key=lambda x: int(x.split('.')[0][6:]))
            self.depth_paths = [join(self.root_dir, depth_dir_name, img) for i, img in enumerate(depth_list)]
            self.npz = True


        if 'Semantics' in self.loss:
            semantics_dir_name = 'semantic_deeplab'
            semantics_list = os.listdir(join(self.root_dir, semantics_dir_name))
            semantics_list.sort(key=lambda x: int(x.split('.')[0]))
            semantics_list = semantics_list[::int(len(semantics_list)/N_used)+1]
            self.semantics_paths = [join(self.root_dir, semantics_dir_name, img) for i, img in enumerate(semantics_list) if i in idx_ls]

        h, w = cv2.imread(self.img_paths[0]).shape[:2]
        self.h, self.w = int(h*self.downsample), int(w*self.downsample)

        self.crop = True
        if self.crop:
            self.h -= self.crop_size * 2
            self.w -= self.crop_size * 2
            self.K[:2, 2] -= self.crop_size

        self.directions = get_ray_directions(self.h, self.w, self.K)
        self.mesh_path = join(self.root_dir, f"{self.data_name}_vh_clean_2.ply")


        self.buffer(blend_a=False)

        def sobel_2d(im):
            # sobel_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype='float32')  #

            sobel_kernel_x = np.array([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype='float32')  #
            sobel_kernel_y = np.array([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype='float32')  #

            sobel_kernel_x = sobel_kernel_x.reshape((1, 1, 3, 3))
            sobel_kernel_y = sobel_kernel_y.reshape((1, 1, 3, 3))
            weight_x = torch.from_numpy(sobel_kernel_x).to(im.device)
            weight_y = torch.from_numpy(sobel_kernel_y).to(im.device)
            edge_detect_x = torch.nn.functional.conv2d(im.clone().unsqueeze(1), weight_x, padding=1)
            edge_detect_x = torch.abs(edge_detect_x.squeeze(1))/im

            edge_detect_y = torch.nn.functional.conv2d(im.clone().unsqueeze(1), weight_y, padding=1)
            edge_detect_y = torch.abs(edge_detect_y.squeeze(1)) / im

            return edge_detect_x, edge_detect_y


        depths = self.all_depths
        edges_x, edges_y = sobel_2d(depths)
        #TODO: Francis
        self.all_depths = (self.all_depths + hparams['aabb_shift']) * hparams['aabb_scale']
        self.all_depths[self.all_depths<3e-3] = 3e-3
        # cv2.imshow('depth', depths[0].detach().cpu().numpy()*3)
        # cv2.waitKey(0)
        self.all_depths[edges_x > 8 * (edges_x.mean(dim=1).mean(dim=1).unsqueeze(1).unsqueeze(2))] = 0.
        self.all_depths[edges_y > 8 *  (edges_y.mean(dim=1).mean(dim=1).unsqueeze(1).unsqueeze(2))] = 0.


class Replica(BaseDataset):
    def __init__(self, hparams, split='train', **kwargs):
        super().__init__(hparams, split, **kwargs)



        with open(join(self.root_dir, f"transforms.json")) as file:
            meta = json.load(file)
        self.depth_replica = True
        self.focal = meta['fl_x']
        self.K = FloatTensor([[self.focal, 0, meta['cx']],
                              [0, self.focal, meta['cy']],
                              [0, 0, 1]])

        self.K[:2] *= self.downsample




        if split == 'train':
            idx_ls = [i for i in range(len(meta['frames']))]
        elif split == 'test':
            idx_ls = [i for i in range(len(meta['frames']))]
        elif split.endswith('all'):
            idx_ls = [i for i in range(len(meta['frames']))]
        else:
            raise

        self.idx_ls = idx_ls

        self.scale = 1.0

        # meta['frames'] = meta['frames'][:10]
        self.poses = torch.empty(len(meta['frames']), 3, 4)
        self.times = torch.empty(len(meta['frames']))

        def replica_matrix_to_ngp(new_pose, scale=1.0, offset=[0, 0, 0]):


            tmp = new_pose[2, :].clone()
            new_pose[2, :] = new_pose[1, :]
            new_pose[1, :] = new_pose[0, :]
            new_pose[0, :] = tmp


            return new_pose

        for i, frame in enumerate(meta['frames']):
            self.poses[i] = FloatTensor(frame['transform_matrix'])[:3, :4]
            self.poses[i] = replica_matrix_to_ngp(self.poses[i])
            # self.poses[i][:3, 1] *= -1
            # self.poses[i][:3, 2] *= -1
            # self.poses[i][:, 1:3] *= -1  # [right up back] to [right down front]
            # self.times[i] = frame['time']

        # if self.sampling in ["NGP"]:
        #     self.scale_to(scale=2.7)

        self.img_paths = [join(self.root_dir, f"{frame['file_path']}") for frame in meta['frames']]
        self.depth_paths = [join(self.root_dir, f"{frame['depth_path']}") for frame in meta['frames']]

        h, w = cv2.imread(self.img_paths[0]).shape[:2]
        self.h, self.w = int(h * self.downsample), int(w * self.downsample)

        self.directions = get_ray_directions(self.h, self.w, self.K)  # (h,w,3)

        self.idx_ls = [i for i in range(len(meta['frames']))]

        self.center()
        # self.scale_to(scale_to=0.7)
        self.scale_to(scale_to=0.4)

        N_used = 200
        diff = 4
        if split == 'test':
            self.img_paths = self.img_paths[2:diff * N_used:diff]
            self.depth_paths = self.depth_paths[2:diff * N_used:diff]
            self.poses = self.poses[2:diff * N_used:diff]
        else:
            self.img_paths = self.img_paths[:diff * N_used:diff]
            self.depth_paths = self.depth_paths[:diff * N_used:diff]
            self.poses = self.poses[:diff * N_used:diff]

        # background = 'random' if 'train' in self.split else 'white'
        self.buffer(blend_a=False)


class LLFF(BaseDataset):
    def __init__(self, hparams, split='train', **kwargs):
        """
        is_spheric: 
            True: the images are taken in a is_spheric inward-facing manner
            False: forward-facing (default)
        """
        super().__init__(hparams, split, **kwargs)
        print("spheric data form" if self.is_spheric else "llff data form")
        
        poses_bounds = np.load(join(self.root_dir, 'poses_bounds.npy')) # (N_images, 17)
        if split in ['train', 'test']:
            img_paths = sorted(glob.glob(join(self.root_dir, 'images/*'))) # load full resolution image then resize
            assert len(poses_bounds) == len(img_paths), 'Mismatch between number of images and number of poses! Please rerun COLMAP!'
        bounds = poses_bounds[:, 15:] # (N_images, 2)
        poses_bounds = poses_bounds[:, :15].reshape(-1, 3, 5)
        poses = poses_bounds[:,:,:4] # (N_images, 3, 4)
        # original intrinsics same for all images, rescale focal length according to training resolution
        self.h, self.w, self.focal = poses_bounds[0,:,4]*self.downsample
        self.h, self.w = int(self.h), int(self.w) 
        self.K = FloatTensor([[self.focal, 0, self.w/2],
                            [0, self.focal, self.h/2],
                            [0,  0,  1]])
        # ray directions for all pixels, same for all images (same H, W, focal)
        self.directions = get_ray_directions(self.h, self.w, self.K) # (H, W, 3)
        '''
        Step 2: correct poses
        Original poses has rotation in form "down right back", change to "right down front"
        See https://github.com/bmild/nerf/issues/34
        '''
        poses = np.concatenate([poses[..., 1:2], poses[..., 0:1], -poses[..., 2:4]], -1) # (N_images, 3, 4) exclude H, W, focal
        poses = center_poses(poses)
        '''
        Step 3: correct scale so that the nearest depth is at a little more than 1.0
        See https://github.com/bmild/nerf/issues/34
        '''
        near_original = bounds.min()
        scale_factor = near_original*0.75 # 0.75 is the default parameter, the nearest depth is at 1/0.75=1.33
        bounds /= scale_factor
        poses[..., 3] /= scale_factor

        if not self.is_spheric: #llff
            self.near, self.far = 0, 1
        else:
            self.near = bounds.min()
            self.far = min(8 * self.near, bounds.max())
        
        if split == 'train':
            self.img_paths = [x for i, x in enumerate(img_paths) if i%8!=0]
            self.poses = FloatTensor(np.float32([x for i, x in enumerate(poses) if i%8!=0]))
        elif split == 'test':
            self.img_paths = [x for i, x in enumerate(img_paths) if i%8==0]
            self.poses = FloatTensor(np.float32([x for i, x in enumerate(poses) if i%8==0]))
        elif split.endswith('all'):
            self.img_paths = [x for i, x in enumerate(img_paths)]
            self.poses = FloatTensor(np.float32([x for i, x in enumerate(poses)]))
        elif split == 'test_traj': # create a parametric rendering path
            if not is_spheric: # llff
                '''
                hardcoded, this is numerically close to the formula given in the original repo. 
                Mathematically if near=1 and far=infinity, then this number will converge to 4.
                '''
                focus_depth = 3.5 
                radii = np.percentile(np.abs(poses[..., 3]), 90, axis=0)
                self.poses = FloatTensor(create_spiral_poses(radii, focus_depth), dtype=torch.float)
            else:
                radius = 1.1 * bounds.min()
                self.poses = FloatTensor(create_spheric_poses(radius))

        self.buffer(blend_a=False)
        # pl_rays = torch.load("/home/fy/nerf_pl/llff_all_rays.pth")
        # pl_rgbs = torch.load("/home/fy/nerf_pl/llff_all_rgbs.pth")
        # print(torch.allclose(pl_rays, self.all_rays))

