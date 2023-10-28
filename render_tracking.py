from model import *
import time as tim
import random
import nerfacc
from nerfacc.estimators.occ_grid import OccGridEstimator
from utils import *
import pytorch3d.transforms
import open3d as o3d
from RAFT import flow_wrapper

def sdf_to_sigma(sdf: torch.Tensor, alpha, beta):
    exp = 0.5 * torch.exp(-torch.abs(sdf) / beta)
    psi = torch.where(sdf >= 0, exp, 1 - exp)
    return alpha * psi



def comp_closest_pts_idx_with_split(pts_src, pts_des):
    """
    :param pts_src:     (3, S)
    :param pts_des:     (3, D)
    :param num_split:
    :return:
    """
    pts_src_list = torch.split(pts_src, 500000, dim=1)
    idx_list = []
    for pts_src_sec in pts_src_list:
        diff = pts_src_sec[:, :, np.newaxis] - pts_des[:, np.newaxis, :]  # (3, S, 1) - (3, 1, D) -> (3, S, D)
        dist = torch.linalg.norm(diff, dim=0)  # (S, D)
        closest_idx = torch.argmin(dist, dim=1)  # (S,)
        idx_list.append(closest_idx)
    closest_idx = torch.cat(idx_list)
    return closest_idx


def comp_point_point_error(Xt, Yt):
    closest_idx = comp_closest_pts_idx_with_split(Xt, Yt)
    pt_pt_vec = Xt - Yt[:, closest_idx]  # (3, S) - (3, S) -> (3, S)
    pt_pt_dist = torch.linalg.norm(pt_pt_vec, dim=0)
    eng = torch.mean(pt_pt_dist)
    return eng





def estimate_normal(pcd, radius=0.06, max_nn=30):
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))

def visualization(src_pcd, tgt_pcd):
    if not src_pcd.has_normals():
        estimate_normal(src_pcd)
        estimate_normal(tgt_pcd)
    src_pcd.paint_uniform_color([1, 0.706, 0])
    tgt_pcd.paint_uniform_color([0, 0.651, 0.929])
    o3d.visualization.draw_geometries([src_pcd, tgt_pcd])

    # src_pcd.transform(pred_trans)
    o3d.visualization.draw_geometries([src_pcd, tgt_pcd])






class Learn_Distortion_List(nn.Module):
    def __init__(self, num_cams, learn_scale = False, learn_shift = False, fix_scaleN = True):
        """depth distortion parameters

        Args:
            num_cams (int): number of cameras
            learn_scale (bool): whether to update scale
            learn_shift (bool): whether to update shift
            cfg (dict): argument options
        """
        super(Learn_Distortion_List, self).__init__()

        self.global_scales = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.FloatTensor([1]), requires_grad=learn_scale) for i in range(num_cams)])

        self.global_shifts = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.FloatTensor([0]), requires_grad=learn_shift) for i in range(num_cams)])


        self.fix_scaleN = fix_scaleN
        self.fix_num=0
        self.num_cams = num_cams

    def set_scale_and_shift(self, frame_index, scale, shift):
        with torch.autograd.no_grad():
            self.global_scales[frame_index].data = self.global_scales[frame_index].data * 0.0 + scale
            self.global_shifts[frame_index].data = self.global_shifts[frame_index].data * 0.0 + shift

    def shift_scales(self, frame_index):
        with torch.autograd.no_grad():
            for i in range(frame_index+1, len(self.global_scales)):
                self.global_scales[i].data = self.global_scales[frame_index].data * 1.0
                self.global_shifts[i].data = self.global_shifts[frame_index].data * 1.0

    def fix_scale_and_shift(self, frame_index):
        self.global_scales[frame_index].requires_grad = False
        self.global_scales[frame_index].grad = None
        self.global_shifts[frame_index].requires_grad = False
        self.global_shifts[frame_index].grad = None

    def fix_other_scale_and_shift(self, frame_index):
        for i in range(len(self.global_scales)):
            if i != frame_index:
                self.global_scales[i].requires_grad = False
                self.global_scales[i].grad = None
                self.global_shifts[i].requires_grad = False
                self.global_shifts[i].grad = None

    def free_scale_and_shift(self):
        for i in range(len(self.global_scales)):
            self.global_scales[i].requires_grad = True
            self.global_shifts[i].requires_grad = True


    def fix_previous_frame(self, frame_ind, window_size):
        for i in range(frame_ind-window_size):
            self.global_scales[i].requires_grad = False
            self.global_scales[i].grad = None
            self.global_shifts[i].requires_grad = False
            self.global_shifts[i].grad = None

    def forward(self, cam_id):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        scale = self.global_scales[cam_id]
        if scale < 0.01:
            scale = torch.tensor(0.01).to(device)
        if self.fix_scaleN and cam_id <= self.fix_num:
            scale = torch.tensor(1).to(device)
        shift = self.global_shifts[cam_id]

        return scale, shift


class Learn_Distortion(nn.Module):
    def __init__(self, num_cams, learn_scale = False, learn_shift = False, fix_scaleN = True):
        """depth distortion parameters

        Args:
            num_cams (int): number of cameras
            learn_scale (bool): whether to update scale
            learn_shift (bool): whether to update shift
            cfg (dict): argument options
        """
        super(Learn_Distortion, self).__init__()
        self.global_scales = nn.Parameter(torch.ones(size=(num_cams, 1), dtype=torch.float32),
                                          requires_grad=learn_scale)
        self.global_shifts = nn.Parameter(0.0 * torch.ones(size=(num_cams, 1), dtype=torch.float32),
                                          requires_grad=learn_shift)
        self.fix_scaleN = fix_scaleN
        self.fix_num=0
        self.num_cams = num_cams

    def shift_scales(self, frame_index):
        print('Scale and shift are: ', self.global_scales[frame_index].data * 1.0, ' ', self.global_shifts[frame_index].data * 1.0)
        with torch.autograd.no_grad():
            for i in range(frame_index+1, len(self.global_scales)):
                self.global_scales[i].data = self.global_scales[frame_index].data * 1.0
                self.global_shifts[i].data = self.global_shifts[frame_index].data * 1.0



    def forward(self, cam_id):
        scale = self.global_scales[cam_id]
        if scale < 0.01:
            scale = torch.tensor(0.01, device=self.global_scales.device)
        if self.fix_scaleN and cam_id <= self.fix_num:
            scale = torch.tensor(1, device=self.global_scales.device)
        shift = self.global_shifts[cam_id]

        return scale, shift



class LiePose(nn.Module):
    def __init__(self, frame_num, window_size):
        super().__init__()
        self.frame_num = frame_num
        self.window_size = window_size
        #rotations in lie algebra form
        self.shift_rotations = torch.nn.ParameterList(
            [torch.nn.Parameter((torch.zeros(3))) for i in range(self.frame_num)])

        self.shift_translations = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(3)) for i in range(self.frame_num)])

        self.shift_rotations[0].requires_grad = False
        self.shift_translations[0].requires_grad = False


    def predict_pose_with_motion_model(self, frame_ind):
        with torch.autograd.no_grad():
            if frame_ind > 0:
                if frame_ind >1:
                    previous_rotation = Exp(self.shift_rotations[frame_ind - 1].data)
                    former_previous_rotation = Exp(self.shift_rotations[frame_ind - 2].data)
                    self.shift_rotations[frame_ind].data = Log(torch.matmul(previous_rotation, torch.matmul(former_previous_rotation.T, previous_rotation)))
                    relative_translation = torch.matmul(former_previous_rotation.T, self.shift_translations[frame_ind-1] - self.shift_translations[frame_ind-2])
                    self.shift_translations[frame_ind].data = self.shift_translations[frame_ind - 1].data * 1.0 + torch.matmul(previous_rotation, relative_translation)

                else:
                    self.shift_rotations[frame_ind].data = self.shift_rotations[frame_ind-1].data * 1.0
                    self.shift_translations[frame_ind].data = self.shift_translations[frame_ind - 1].data * 1.0

    def get_rotation_matrix(self, frame_ind):
        return Exp(self.shift_rotations[frame_ind])


    def get_translation_vector(self, frame_ind):
        return self.shift_translations[frame_ind]


    def fix_other_frames(self, frame_ind):
        for i in range(len(self.shift_rotations)):
            if i != frame_ind:
                self.shift_rotations[i].requires_grad = False
                self.shift_rotations[i].grad = None
                self.shift_translations[i].requires_grad = False
                self.shift_translations[i].grad = None

    def fix_all_frames(self):
        for i in range(len(self.shift_rotations)):
                self.shift_rotations[i].requires_grad = False
                self.shift_translations[i].requires_grad = False
                self.shift_rotations[i].grad = None
                self.shift_translations[i].grad = None

    def fix_first_frame(self):
        i = self.current_ref_frame_indices[0]
        self.shift_rotations[i].requires_grad = False
        self.shift_translations[i].requires_grad = False
        self.shift_rotations[i].grad = None
        self.shift_translations[i].grad = None

    def fix_previous_frame(self, frame_ind, window_size):
        for i in range(frame_ind-window_size):
            self.shift_rotations[i].requires_grad = False
            self.shift_translations[i].requires_grad = False
            self.shift_rotations[i].grad = None
            self.shift_translations[i].grad = None


    def finish_fix_frames(self):
        for i in range(len(self.shift_rotations)):
            self.shift_rotations[i].requires_grad = True
            self.shift_translations[i].requires_grad = True

    def bind_gt_pose(self, T_wcs):
        self.T_wcs = T_wcs

    def copy_pose(self, index, copy_num):
        with torch.autograd.no_grad():
            self.shift_rotations[index].data = Log(self.T_wcs[copy_num][:3, :3]) * 1.0
            self.shift_translations[index].data = self.T_wcs[copy_num][:3, 3] * 1.0





class QuaternionPose(nn.Module):
    def __init__(self, frame_num, window_size):
        super().__init__()
        self.frame_num = frame_num
        self.window_size = window_size
        self.shift_rotations = torch.nn.ParameterList(
            [torch.nn.Parameter(pytorch3d.transforms.matrix_to_quaternion(
                pytorch3d.transforms.axis_angle_to_matrix(torch.zeros(3)))) for i in range(self.frame_num)])

        self.shift_translations = torch.nn.ParameterList(
            [torch.nn.Parameter(torch.zeros(3)) for i in range(self.frame_num)])

        self.shift_rotations[0].requires_grad = False
        self.shift_translations[0].requires_grad = False

    def get_motion_model_predicted_pose(self, frame_ind):
        with torch.autograd.no_grad():
            if frame_ind > 0:
                if frame_ind >1:
                    previous_rotation = pytorch3d.transforms.quaternion_to_matrix(self.shift_rotations[frame_ind - 1].data)
                    former_previous_rotation = pytorch3d.transforms.quaternion_to_matrix(self.shift_rotations[frame_ind - 2].data)
                    predicted_rotation = pytorch3d.transforms.matrix_to_quaternion(torch.matmul(previous_rotation, torch.matmul(former_previous_rotation.T, previous_rotation)))
                    relative_translation = torch.matmul(former_previous_rotation.T, self.shift_translations[frame_ind-1] - self.shift_translations[frame_ind-2])
                    predicted_translation = self.shift_translations[frame_ind - 1].data * 1.0 + torch.matmul(previous_rotation, relative_translation)

                else:
                    predicted_rotation = self.shift_rotations[frame_ind-1].data * 1.0
                    predicted_translation = self.shift_translations[frame_ind - 1].data * 1.0
        return predicted_rotation, predicted_translation

    def predict_pose_with_motion_model(self, frame_ind):
        with torch.autograd.no_grad():
            if frame_ind > 0:
                if frame_ind >1:
                    previous_rotation = pytorch3d.transforms.quaternion_to_matrix(self.shift_rotations[frame_ind - 1].data)
                    former_previous_rotation = pytorch3d.transforms.quaternion_to_matrix(self.shift_rotations[frame_ind - 2].data)
                    self.shift_rotations[frame_ind].data = pytorch3d.transforms.matrix_to_quaternion(torch.matmul(previous_rotation, torch.matmul(former_previous_rotation.T, previous_rotation)))
                    relative_translation = torch.matmul(former_previous_rotation.T, self.shift_translations[frame_ind-1] - self.shift_translations[frame_ind-2])
                    self.shift_translations[frame_ind].data = self.shift_translations[frame_ind - 1].data * 1.0 + torch.matmul(previous_rotation, relative_translation)

                else:
                    self.shift_rotations[frame_ind].data = self.shift_rotations[frame_ind-1].data * 1.0
                    self.shift_translations[frame_ind].data = self.shift_translations[frame_ind - 1].data * 1.0

    def get_rotation_matrix(self, frame_ind):
        return pytorch3d.transforms.quaternion_to_matrix(self.shift_rotations[frame_ind])


    def get_translation_vector(self, frame_ind):
        return self.shift_translations[frame_ind]


    def fix_other_frames(self, frame_ind):
        for i in range(len(self.shift_rotations)):
            if i != frame_ind:
                self.shift_rotations[i].requires_grad = False
                self.shift_rotations[i].grad = None
                self.shift_translations[i].requires_grad = False
                self.shift_translations[i].grad = None

    def fix_all_frames(self):
        for i in range(len(self.shift_rotations)):
                self.shift_rotations[i].requires_grad = False
                self.shift_translations[i].requires_grad = False
                self.shift_rotations[i].grad = None
                self.shift_translations[i].grad = None

    def fix_first_frame(self):
        i = self.current_ref_frame_indices[0]
        self.shift_rotations[i].requires_grad = False
        self.shift_translations[i].requires_grad = False
        self.shift_rotations[i].grad = None
        self.shift_translations[i].grad = None

    def fix_previous_frame(self, frame_ind, window_size):
        for i in range(frame_ind-window_size):
            self.shift_rotations[i].requires_grad = False
            self.shift_translations[i].requires_grad = False
            self.shift_rotations[i].grad = None
            self.shift_translations[i].grad = None


    def finish_fix_frames(self):
        for i in range(len(self.shift_rotations)):
            self.shift_rotations[i].requires_grad = True
            self.shift_translations[i].requires_grad = True

    def bind_gt_pose(self, T_wcs):
        self.T_wcs = T_wcs

    def copy_pose(self, index, copy_num):
        with torch.autograd.no_grad():
            self.shift_rotations[index].data = pytorch3d.transforms.matrix_to_quaternion(self.T_wcs[copy_num][:3, :3]) * 1.0
            self.shift_translations[index].data = self.T_wcs[copy_num][:3, 3] * 1.0







class TrackingAndMappingSampling(nn.Module):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        # self.nerf_list = torch.nn.ModuleList([DensityRadianceField(hparams) for i in range(2)])
        self.nerf = DensityRadianceField(hparams)
        self.near = hparams.near
        self.far = hparams.far
        aabb = hparams.aabb
        levels = hparams.levels
        self.aabb = torch.tensor([-aabb, -aabb, -aabb, aabb, aabb, aabb])
        # self.estimators = torch.nn.ModuleList([OccGridEstimator(roi_aabb=self.aabb, resolution=128, levels=1) for i in range(100)])
        self.estimator = OccGridEstimator(roi_aabb=self.aabb, resolution=128, levels=1)
        self.render_step_size = 5e-3
        self.fix_network = False
        self.frame_num = 1000

        self.distance_thres = hparams['distance_thres']
        self.unknown_distance = hparams['unknown_distance']

        # self.depth_scale = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor([1.])) for i in range(self.frame_num)])
        # self.depth_shift = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor([0.])) for i in range(self.frame_num)])
        self.depth_distortion = Learn_Distortion_List(self.frame_num, learn_scale=hparams['learn_scale'], learn_shift = hparams['learn_shift'])

        self.current_index = 0
        self.window_size = 1


        self.learned_poses = QuaternionPose(self.frame_num, self.window_size)

        # self.learned_poses = LiePose(self.frame_num, self.window_size)

        self.flow_model = flow_wrapper.generate_model()
        self.flows = []
        self.inverse_flows = []
        # self.optical_flows = torch.load('/media/kyrie/000A8561000BB32A/Flow/Tanks/flows_family.pth')
        # self.optical_flows_inverse = torch.load('/media/kyrie/000A8561000BB32A/Flow/flows_inverse.pth')

    def get_flow(self, frame_ind):
        if frame_ind > 0:
            img_prev = self.imgs[frame_ind-1].permute(2,0,1).unsqueeze(0) * 255
            img_curr = self.imgs[frame_ind].permute(2,0,1).unsqueeze(0)* 255
            flow = flow_wrapper.get_flow(self.flow_model, img_prev, img_curr)
            # flow_wrapper.vis_correspondences(img_prev, img_curr, flow)
            pad_y = int((flow.shape[2]- img_curr.shape[2])/2)
            pad_x = int((flow.shape[3]- img_curr.shape[3])/2)
            if pad_y > 0:
                flow = flow[0,:,pad_y:-pad_y,:]
            if pad_x > 0:
                flow = flow[:,:,pad_x:-pad_x]
            self.flows.append(flow)
            #Get inverse flow
            inverse_flow = flow_wrapper.get_flow(self.flow_model, img_curr, img_prev)
            if pad_y > 0:
                inverse_flow = inverse_flow[0,:,pad_y:-pad_y,:]
            if pad_x > 0:
                inverse_flow = inverse_flow[:,:,pad_x:-pad_x]
            self.inverse_flows.append(inverse_flow)


    def visuallize_pcd(self, frame_index1, frame_index2):
        depths_1 = self.get_depth_by_index(frame_index1)
        pts_3d_1 = depths_1.unsqueeze(2) * self.camera_pts
        current_rotation = self.learned_poses.get_rotation_matrix(frame_index1)
        pts_3d_w_1 = pts_3d_1 @ current_rotation.T + self.learned_poses.get_translation_vector(frame_index1)
        pts_3d_w_1 = pts_3d_w_1.detach().cpu().view(-1,3).numpy()

        depths_2 = self.get_depth_by_index(frame_index2)
        pts_3d_2 = depths_2.unsqueeze(2) * self.camera_pts
        current_rotation = self.learned_poses.get_rotation_matrix(frame_index2)
        pts_3d_w_2 = pts_3d_2 @ current_rotation.T + self.learned_poses.get_translation_vector(frame_index2)
        pts_3d_w_2 = pts_3d_w_2.detach().cpu().view(-1,3).numpy()

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pts_3d_w_1[::10,:])
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pts_3d_w_2[::10,:])
        visualization(pcd1, pcd2)



    def get_flow_loss(self, frame_index, depths, indices, forward = True):

        # flow = self.optical_flows[frame_index]
        # flow = flow[:, 2:-2, :]
        if forward:
            flow = self.flows[frame_index]
            match_frame_index = frame_index + 1
        else:
            flow = self.inverse_flows[frame_index-1]
            frame_index = frame_index
            match_frame_index = frame_index-1
        pts_3d = depths * self.camera_pts.view(-1,3)[indices]
        current_rotation = self.learned_poses.get_rotation_matrix(frame_index)
        match_rotation = self.learned_poses.get_rotation_matrix(match_frame_index)
        relative_rotation = torch.matmul(match_rotation.T, current_rotation)
        relative_translation = torch.matmul(match_rotation.T, self.learned_poses.get_translation_vector(frame_index) - self.learned_poses.get_translation_vector(match_frame_index))
        pts_3d = pts_3d
        pts_3d = pts_3d @ relative_rotation.T + relative_translation
        projections = (pts_3d/pts_3d[:,2:])  @ self.K.T
        estimate_flow = projections - self.pixel_pts.view(-1,3)[indices]
        gt_flow = flow.permute(1,2,0).view(-1,2)[indices]
        return gt_flow, estimate_flow[:,:2]

    def get_pose_constraint_loss(self, frame_index, forward = True):

        # flow = self.optical_flows[frame_index]
        # flow = flow[:, 2:-2, :]
        if forward:
            flow = self.flows[frame_index]
            match_frame_index = frame_index + 1
        else:
            flow = self.inverse_flows[frame_index]
            frame_index = frame_index+1
            match_frame_index = frame_index-1
        depths = self.get_depth_by_index(frame_index)
        # depths = rendered_depth
        pts_3d = depths.unsqueeze(2) * self.camera_pts
        current_rotation = self.learned_poses.get_rotation_matrix(frame_index)
        match_rotation = self.learned_poses.get_rotation_matrix(match_frame_index)
        relative_rotation = torch.matmul(match_rotation.T, current_rotation)
        relative_translation = torch.matmul(match_rotation.T, self.learned_poses.get_translation_vector(frame_index) - self.learned_poses.get_translation_vector(match_frame_index))
        pts_3d = pts_3d[depths>1e-3][depths[depths>1e-3] < torch.mean(depths)]
        pts_3d = pts_3d @ relative_rotation.T + relative_translation
        projections = (pts_3d/pts_3d[:,2:])  @ self.K.T
        estimate_flow = projections - self.pixel_pts[depths>1e-3][depths[depths>1e-3] < torch.mean(depths)]
        gt_flow = flow.permute(1,2,0)[depths>1e-3][depths[depths>1e-3] < torch.mean(depths)]
        return gt_flow, estimate_flow[:,:2]

    def get_icp_constraint_loss(self, frame_index, forward = True):
        point_cloud_size = 5000
        # flow = self.optical_flows[frame_index]
        # flow = flow[:, 2:-2, :]
        if forward:
            flow = self.flows[frame_index]
            match_frame_index = frame_index + 1
        else:
            flow = self.inverse_flows[frame_index]
            frame_index = frame_index+1
            match_frame_index = frame_index-1
        depths = self.get_depth_by_index(frame_index)
        # depths = rendered_depth
        #point cloud of current frame
        pts_3d = depths[depths>1e-3].unsqueeze(1) * self.camera_pts[depths>1e-3]
        pts_3d = pts_3d[::int(len(pts_3d) / point_cloud_size)]
        current_rotation = self.learned_poses.get_rotation_matrix(frame_index)
        pts_3d = pts_3d @ current_rotation.T + self.learned_poses.get_translation_vector(frame_index)

        #point cloud of match frame
        match_depth = self.get_depth_by_index(match_frame_index)
        match_pts_3d = match_depth[match_depth>1e-3].unsqueeze(1) * self.camera_pts[match_depth>1e-3]
        match_rotation = self.learned_poses.get_rotation_matrix(match_frame_index)
        match_pts_3d = match_pts_3d @ match_rotation.T + self.learned_poses.get_translation_vector(match_frame_index)
        match_pts_3d = match_pts_3d[::int(len(match_pts_3d) / point_cloud_size)]

        loss = comp_point_point_error(pts_3d.permute(1, 0),
                               match_pts_3d.permute(1, 0))
        loss += comp_point_point_error(match_pts_3d.permute(1, 0),
                               pts_3d.permute(1, 0))
        return loss

    def get_epipolar_constraint_loss(self, frame_index, forward=True):
        if forward:
            flow = self.flows[frame_index]
            match_frame_index = frame_index + 1
        else:
            flow = self.inverse_flows[frame_index]
            frame_index = frame_index + 1
            match_frame_index = frame_index - 1

        depths = self.get_depth_by_index(frame_index)
        #Relative Pose
        current_rotation = self.learned_poses.get_rotation_matrix(frame_index)
        match_rotation = self.learned_poses.get_rotation_matrix(match_frame_index)
        relative_rotation = torch.matmul(match_rotation.T, current_rotation)
        relative_translation = torch.matmul(match_rotation.T, self.learned_poses.get_translation_vector(
            frame_index) - self.learned_poses.get_translation_vector(match_frame_index))
        relative_translation = relative_translation/((relative_translation+1e-5).norm())
        gt_flow = flow.permute(1, 2, 0)[depths > 1e-3][depths[depths > 1e-3] < torch.mean(depths)]
        current_pixels = self.pixel_pts[depths > 1e-3][depths[depths > 1e-3] < torch.mean(depths)]
        match_pixels = current_pixels + torch.cat([gt_flow, torch.zeros_like(gt_flow)[:,:1]], dim=1)
        def vec2skew(v):
            """
            :param v:  (3, ) torch tensor
            :return:   (3, 3)
            """
            zero = torch.zeros(1, dtype=torch.float32, device=v.device)
            skew_v0 = torch.cat([ zero,    -v[2:3],   v[1:2]])  # (3, 1)
            skew_v1 = torch.cat([ v[2:3],   zero,    -v[0:1]])
            skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
            skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
            return skew_v  # (3, 3)
        essential_matrix = torch.matmul(vec2skew(relative_translation), relative_rotation)
        inv_K = torch.linalg.inv(self.K)
        fundumental_matrix = torch.matmul(torch.matmul(inv_K.T, essential_matrix), inv_K)
        loss = torch.mean(torch.sum(torch.abs(torch.mul(torch.matmul(current_pixels , fundumental_matrix), match_pixels)), dim=1))/100
        return loss





    def predict_pose_with_motion_model(self, frame_ind):
        self.learned_poses.predict_pose_with_motion_model(frame_ind)


    def fix_other_frames(self, frame_ind):
        self.learned_poses.fix_other_frames(frame_ind)

    def fix_all_frames(self):
        self.learned_poses.fix_all_frames()

    def fix_first_frame(self):
        self.learned_poses.fix_first_frame()

    def fix_previous_frame(self, frame_ind, window_size):
        self.learned_poses.fix_previous_frame(frame_ind, window_size)


    def finish_fix_frames(self):
        self.learned_poses.finish_fix_frames()


    #
    # def initialize_pose(self, keyframe_interval):
    #     self.learned_poses.initialize_pose(keyframe_interval)
    #
    #
    # def initialize_pose_gt(self, keyframe_interval):
    #     self.learned_poses.initialize_pose_gt(keyframe_interval)


    def initialize_pose(self, keyframe_interval):
        #0-initialize_num
        initialize_num = (self.window_size*2) * keyframe_interval
        initialize_num = 4
        #Initialize pose
        self.keyframe_list = []
        with torch.autograd.no_grad():
            for i in range(len(self.T_wcs)):
                if i<= initialize_num:
                    copy_num = i
                    #Generate a new keyframe
                    if i >= 1:
                        self.get_flow(i)
                    if i%keyframe_interval == 0:
                        self.keyframe_list.append(i)

                else:
                    copy_num = initialize_num
                copy_num = 0
                self.learned_poses.copy_pose(i, copy_num)
        self.initialize_num = initialize_num
        self.current_ref_frame_indices = [item for item in self.keyframe_list]


    def initialize_pose_gt(self, keyframe_interval):
        #0-initialize_num
        initialize_num = (self.window_size*2) * keyframe_interval
        initialize_num = 0
        #Initialize pose
        self.keyframe_list = []
        with torch.autograd.no_grad():
            for i in range(len(self.T_wcs)):
                copy_num = i
                if i<= initialize_num:
                    #Generate a new keyframe
                    if i%keyframe_interval == 0:
                        self.keyframe_list.append(i)
                self.learned_poses.copy_pose(i, copy_num)
        self.initialize_num = initialize_num
        self.current_ref_frame_indices = [item for item in self.keyframe_list]




    def bind_predict_depths(self, predict_depths, all_depths):
        self.gt_depths = all_depths
        self.predict_depths = predict_depths


    def get_depth_by_index(self, frame_index):
        scale, shift = self.depth_distortion(frame_index)
        depth = self.gt_depths[frame_index] * 1.0
        depth[depth>1e-3] = scale * depth[depth>1e-3] + shift
        return depth


    def bind_images_and_poses(self, imgs, K, T_wcs, all_rays, camera_rays):
        self.imgs = imgs
        self.h = self.imgs[0].shape[0]
        self.w = self.imgs[0].shape[1]
        self.K = K
        self.T_wcs = T_wcs
        self.learned_poses.bind_gt_pose(T_wcs)
        self.all_rays = all_rays
        self.camera_rays = camera_rays.view(self.h, self.w, 6)
        self.camera_pts = self.camera_rays[:,:,:3] + (1./self.camera_rays[:,:,5:]) * self.camera_rays[:,:,3:]
        self.pixel_pts = self.camera_pts @ self.K.T




    def create_new_keyframe(self, frame_ind):
        self.keyframe_list.append(frame_ind)
        self.current_ref_frame_indices.append(frame_ind)
        if len(self.current_ref_frame_indices) > self.window_size*2+1:
            self.current_ref_frame_indices.pop(0)


    def clamp_positions(self, positions):
        clamp_positions = torch.clamp(positions, torch.Tensor([0,0]).to(positions.device), torch.Tensor([self.w-2, self.h-2]).to(positions.device))
        clamp_mask = (positions-clamp_positions).norm(dim=1) > 3
        return clamp_positions, clamp_mask

    def bilinear_sample(self, img,  pixel_positions, dirs = None, depths = None):
        #Get four corners
        pixel_positions_clamp = pixel_positions[:,:2]
        pixel_positions_clamp, clamp_mask = self.clamp_positions(pixel_positions_clamp)
        # if len(clamp_mask[clamp_mask == True]) > 0:
        #     print('Clamp pixels!')
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
        if dirs is not None:
            dir_view = dirs.view(-1,3)
            img_view = torch.cat([img_view, dir_view], dim=1)
        if depths is not None:
            depths_view = depths.view(-1,1)
            img_view = torch.cat([img_view, depths_view], dim=1)

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

        if dirs is None:
            return color, clamp_mask
        else:
            return color[:,:3], color[:,3:] , clamp_mask

    #multi_frame
    def forward(self, rays, frame_index, **kwargs):
        nerf_index = 0
        #Get nearest three indices
        first_index = 0
        last_index = len(self.keyframe_list)-1

        for i in range(len(self.keyframe_list)):
            if self.keyframe_list[i] >= frame_index:
                if (i-1) >= first_index:
                    first_index = i-1
                if (i+1) <= last_index:
                    last_index = i+1
                break
        if last_index-first_index>3:
            first_index = last_index-3
            first_index = max(first_index, 0)

        #Long term connections
        long_connection_index1 = first_index-5
        if long_connection_index1 < 0:
            long_connection_index1 = 0


        long_connection_index2 = first_index-15
        if long_connection_index2 < 0:
            long_connection_index2 = 0


        long_connection_index3 = first_index-30
        if long_connection_index3 < 0:
            long_connection_index3 = 0


        used_indices = self.keyframe_list[first_index:last_index+1]
        if self.training:
            if long_connection_index1 < first_index:
                random_conn1 = random.randint(long_connection_index1, first_index-1)
                used_indices.append(random_conn1)
            if long_connection_index2 < long_connection_index1:
                random_conn2 = random.randint(long_connection_index2, long_connection_index1-1)
                used_indices.append(random_conn2)
            if long_connection_index3 < long_connection_index2:
                random_conn3 = random.randint(long_connection_index3, long_connection_index2-1)
                used_indices.append(random_conn3)
            # if long_connection_index4 < long_connection_index3:
            #     used_indices.append(long_connection_index4)
        ref_frame_indices = torch.LongTensor(used_indices).to(self.imgs.device)
        imgs = self.imgs[ref_frame_indices]
        # with torch.autograd.no_grad():
        ref_depths = [self.gt_depths[ind] for ind in ref_frame_indices]
        dirs = self.all_rays[ref_frame_indices,:,3:6].contiguous().view(len(ref_frame_indices),self.h,self.w,3)

        #Used for sample color
        shift_rotations = [self.learned_poses.get_rotation_matrix(ind) for ind in ref_frame_indices]
        shift_translations = [self.learned_poses.get_translation_vector(ind) for ind in ref_frame_indices]
        #Used for current frame
        current_shift_rotation = self.learned_poses.get_rotation_matrix(frame_index)
        current_shift_translation = self.learned_poses.get_translation_vector(frame_index)
        # print('Cost1 is: ', tim.time()-start_time)

        if self.training:
            def occ_eval_fn(x):
                density = self.nerf(x)
                return density * self.render_step_size

            self.estimator.update_every_n_steps(
                step=kwargs['step'],
                occ_eval_fn=occ_eval_fn,
            )

        # print('Cost2 is: ', tim.time() - start_time)
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (B, 3)
        rays_o = rays_o + current_shift_translation
        rays_d = rays_d @ current_shift_rotation.T
        # print('Cost3 is: ', tim.time() - start_time)

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            sigmas = self.nerf(positions)
            return sigmas.squeeze(-1)

        with torch.autograd.no_grad():
            ray_indices, t_starts, t_ends = self.estimator.sampling(
                rays_o,
                rays_d,
                sigma_fn=sigma_fn,
                near_plane=self.near,
                far_plane=self.far,
                render_step_size=self.render_step_size,
                stratified=self.training,
            )

        # print('Cost4 is: ', tim.time() - start_time)
        if (len(ray_indices) == 0):
            result = None
            return result

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0

            rgbs_list = []
            sample_dir_list = []
            for ii in range(len(imgs)):
                img = imgs[ii]
                dir = dirs[ii]
                ref_depth = ref_depths[ii]
                local_rotation = shift_rotations[ii].T
                local_translation = -torch.matmul(local_rotation, shift_translations[ii])
                local_positions = positions @ local_rotation.T + local_translation

                local_positions[:, 2:][local_positions[:, 2:] < 1e-4] = 1.
                local_positions[:, :2][local_positions[:, 2] < 1e-4] = -1000.0
                # local_positions[:, 2:][torch.abs(local_positions[:, 2:]) < 1e-3] = 1e-3
                pixel_positions = (local_positions / local_positions[:, 2:]) @ self.K.T
                rgbs, sample_dirs, clamp_mask = self.bilinear_sample(img, pixel_positions, dir, ref_depth)
                depths = sample_dirs[:,-1:].detach()
                sample_dirs = sample_dirs[:,:-1]
                rgbs_list.append(rgbs)

                distances = ((local_positions[:,2:] - depths)/depths).detach()
                distance_thres = self.distance_thres
                unknown_distance = self.unknown_distance
                distances[depths<1e-3] = unknown_distance
                rgb_weight = 1 / ((sample_dirs - t_dirs).norm(dim=1) ** 2 + 1e-7)
                rgb_weight[distances.squeeze(1) > distance_thres] = rgb_weight[distances.squeeze(1) > distance_thres] * 0.0 + ((distance_thres/distances.squeeze(1)[distances.squeeze(1) > distance_thres])**2)
                rgb_weight[clamp_mask] = 1e-7
                sample_dir_list.append(rgb_weight)


            if self.training and len(rgbs_list) > 1:
                for i in range(len(ref_frame_indices)):
                    if ref_frame_indices[i] == frame_index:
                        sample_dir_list.pop(i)
                        rgbs_list.pop(i)

            for k in range(0,len(sample_dir_list)):
                if self.training:
                    sample_dir_list[k][sample_dir_list[k] > 1] = 1.

            sum_weight = sample_dir_list[0]
            for k in range(1,len(sample_dir_list)):
                sum_weight = sum_weight + sample_dir_list[k]

            rgbs = (sample_dir_list[0]/sum_weight).unsqueeze(1) * rgbs_list[0]
            for k in range(1,len(sample_dir_list)):
                rgbs = rgbs + (sample_dir_list[k]/sum_weight).unsqueeze(1) * rgbs_list[k]


            # sum_weight = (sample_dir_list[0] + sample_dir_list[1] + sample_dir_list[2])
            # rgbs = (sample_dir_list[0] / sum_weight).unsqueeze(1) * rgbs_list[0] + (
            #             sample_dir_list[1] / sum_weight).unsqueeze(1) * rgbs_list[1] + (
            #                    sample_dir_list[2] / sum_weight).unsqueeze(1) * rgbs_list[2]



            t_dirs = torch.cat([t_dirs, rgbs], dim=1)
            ratios, sigmas = self.nerf(positions, t_dirs)

            # if self.training:
            #     num = random.randint(0,len(rgbs_list)-1)
            #     return rgbs_list[num], sigmas.squeeze(-1), rgbs_list

            return ratios, sigmas.squeeze(-1), rgbs_list

        rgb, opacity, depth, extras = nerfacc.rendering_color_list(
            t_starts,
            t_ends,
            ray_indices,
            n_rays=N_rays,
            rgb_sigma_fn=rgb_sigma_fn,
        )
        result = {}
        result['colors'] = rgb
        result['depths'] = depth
        result['color_list'] = extras['color_list']

        return result



class TrackingAndMappingSamplingSkip(nn.Module):
    def __init__(self, hparams, *args, **kwargs):
        super().__init__()
        # self.nerf_list = torch.nn.ModuleList([DensityRadianceField(hparams) for i in range(2)])
        self.nerf = DensityRadianceField(hparams)
        self.initialize_num = hparams.initialize_num
        self.distillation = False
        self.near = hparams.near
        self.far = hparams.far
        aabb = hparams.aabb
        levels = hparams.levels
        self.aabb = torch.tensor([-aabb, -aabb, -aabb, aabb, aabb, aabb])
        # self.estimators = torch.nn.ModuleList([OccGridEstimator(roi_aabb=self.aabb, resolution=128, levels=1) for i in range(100)])
        self.estimator = OccGridEstimator(roi_aabb=self.aabb, resolution=128, levels=1)
        self.render_step_size = 5e-3
        self.fix_network = False
        self.frame_num = 1000

        self.distance_thres = hparams['distance_thres']
        self.unknown_distance = hparams['unknown_distance']

        # self.depth_scale = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor([1.])) for i in range(self.frame_num)])
        # self.depth_shift = torch.nn.ParameterList([torch.nn.Parameter(torch.FloatTensor([0.])) for i in range(self.frame_num)])
        self.depth_distortion = Learn_Distortion_List(self.frame_num, learn_scale=hparams['learn_scale'], learn_shift = hparams['learn_shift'])

        self.current_index = 0
        self.window_size = 1


        self.learned_poses = QuaternionPose(self.frame_num, self.window_size)

        # self.learned_poses = LiePose(self.frame_num, self.window_size)

        self.flow_model = flow_wrapper.generate_model()
        self.flows = []
        self.inverse_flows = []

        self.tracking_mode = False

    def deactivate_mapping_network(self):
        for parameter in self.nerf.parameters():
            parameter.requires_grad = False
            parameter.grad = None

    def activate_mapping_network(self):
        for parameter in self.nerf.parameters():
            parameter.requires_grad = True



    def set_tracking_mode(self, tracking_mode):
        self.tracking_mode = tracking_mode


    def get_flow(self, frame_ind):
        if frame_ind > 0:
            img_prev = self.imgs[frame_ind-1].permute(2,0,1).unsqueeze(0) * 255
            img_curr = self.imgs[frame_ind].permute(2,0,1).unsqueeze(0)* 255
            flow = flow_wrapper.get_flow(self.flow_model, img_prev, img_curr)
            # flow_wrapper.vis_correspondences(img_prev, img_curr, flow)
            pad_y = int((flow.shape[2]- img_curr.shape[2])/2)
            pad_x = int((flow.shape[3]- img_curr.shape[3])/2)
            if pad_y > 0:
                flow = flow[0,:,pad_y:-pad_y,:]
            if pad_x > 0:
                flow = flow[:,:,pad_x:-pad_x]
            self.flows.append(flow)
            #Get inverse flow
            inverse_flow = flow_wrapper.get_flow(self.flow_model, img_curr, img_prev)
            if pad_y > 0:
                inverse_flow = inverse_flow[0,:,pad_y:-pad_y,:]
            if pad_x > 0:
                inverse_flow = inverse_flow[:,:,pad_x:-pad_x]
            self.inverse_flows.append(inverse_flow)


    def visuallize_pcd(self, frame_index1, frame_index2):
        depths_1 = self.get_depth_by_index(frame_index1)
        pts_3d_1 = depths_1.unsqueeze(2) * self.camera_pts
        current_rotation = self.learned_poses.get_rotation_matrix(frame_index1)
        pts_3d_w_1 = pts_3d_1 @ current_rotation.T + self.learned_poses.get_translation_vector(frame_index1)
        pts_3d_w_1 = pts_3d_w_1.detach().cpu().view(-1,3).numpy()

        depths_2 = self.get_depth_by_index(frame_index2)
        pts_3d_2 = depths_2.unsqueeze(2) * self.camera_pts
        current_rotation = self.learned_poses.get_rotation_matrix(frame_index2)
        pts_3d_w_2 = pts_3d_2 @ current_rotation.T + self.learned_poses.get_translation_vector(frame_index2)
        pts_3d_w_2 = pts_3d_w_2.detach().cpu().view(-1,3).numpy()

        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pts_3d_w_1[::10,:])
        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pts_3d_w_2[::10,:])
        visualization(pcd1, pcd2)



    def get_flow_loss(self, frame_index, depths, indices, forward = True):

        # flow = self.optical_flows[frame_index]
        # flow = flow[:, 2:-2, :]
        if forward:
            flow = self.flows[frame_index]
            match_frame_index = frame_index + 1
        else:
            flow = self.inverse_flows[frame_index-1]
            frame_index = frame_index
            match_frame_index = frame_index-1
        pts_3d = depths * self.camera_pts.view(-1,3)[indices]
        current_rotation = self.learned_poses.get_rotation_matrix(frame_index)
        match_rotation = self.learned_poses.get_rotation_matrix(match_frame_index)
        relative_rotation = torch.matmul(match_rotation.T, current_rotation)
        relative_translation = torch.matmul(match_rotation.T, self.learned_poses.get_translation_vector(frame_index) - self.learned_poses.get_translation_vector(match_frame_index))
        pts_3d = pts_3d
        pts_3d = pts_3d @ relative_rotation.T + relative_translation
        projections = (pts_3d/pts_3d[:,2:])  @ self.K.T
        estimate_flow = projections - self.pixel_pts.view(-1,3)[indices]
        gt_flow = flow.permute(1,2,0).view(-1,2)[indices]
        return gt_flow, estimate_flow[:,:2]

    def get_pose_constraint_loss(self, frame_index, forward = True):

        # flow = self.optical_flows[frame_index]
        # flow = flow[:, 2:-2, :]
        if forward:
            flow = self.flows[frame_index]
            match_frame_index = frame_index + 1
        else:
            flow = self.inverse_flows[frame_index]
            frame_index = frame_index+1
            match_frame_index = frame_index-1
        depths = self.get_depth_by_index(frame_index)
        # depths = rendered_depth
        pts_3d = depths.unsqueeze(2) * self.camera_pts
        current_rotation = self.learned_poses.get_rotation_matrix(frame_index)
        match_rotation = self.learned_poses.get_rotation_matrix(match_frame_index)
        relative_rotation = torch.matmul(match_rotation.T, current_rotation)
        relative_translation = torch.matmul(match_rotation.T, self.learned_poses.get_translation_vector(frame_index) - self.learned_poses.get_translation_vector(match_frame_index))
        pts_3d = pts_3d[depths>1e-3][depths[depths>1e-3] < torch.mean(depths)]
        pts_3d = pts_3d @ relative_rotation.T + relative_translation
        projections = (pts_3d/pts_3d[:,2:])  @ self.K.T
        estimate_flow = projections - self.pixel_pts[depths>1e-3][depths[depths>1e-3] < torch.mean(depths)]
        gt_flow = flow.permute(1,2,0)[depths>1e-3][depths[depths>1e-3] < torch.mean(depths)]
        return gt_flow, estimate_flow[:,:2]

    def get_icp_constraint_loss(self, frame_index, forward = True):
        point_cloud_size = 5000
        # flow = self.optical_flows[frame_index]
        # flow = flow[:, 2:-2, :]
        if forward:
            # flow = self.flows[frame_index]
            match_frame_index = frame_index + 1
        else:
            # flow = self.inverse_flows[frame_index]
            frame_index = frame_index+1
            match_frame_index = frame_index-1
        depths = self.get_depth_by_index(frame_index)
        # depths = rendered_depth
        #point cloud of current frame
        pts_3d = depths[depths>1e-3].unsqueeze(1) * self.camera_pts[depths>1e-3]
        pts_3d = pts_3d[::int(len(pts_3d) / point_cloud_size)]
        current_rotation = self.learned_poses.get_rotation_matrix(frame_index)
        pts_3d = pts_3d @ current_rotation.T + self.learned_poses.get_translation_vector(frame_index)

        #point cloud of match frame
        match_depth = self.get_depth_by_index(match_frame_index)
        match_pts_3d = match_depth[match_depth>1e-3].unsqueeze(1) * self.camera_pts[match_depth>1e-3]
        match_rotation = self.learned_poses.get_rotation_matrix(match_frame_index)
        match_pts_3d = match_pts_3d @ match_rotation.T + self.learned_poses.get_translation_vector(match_frame_index)
        match_pts_3d = match_pts_3d[::int(len(match_pts_3d) / point_cloud_size)]

        loss = comp_point_point_error(pts_3d.permute(1, 0),
                               match_pts_3d.permute(1, 0))
        loss += comp_point_point_error(match_pts_3d.permute(1, 0),
                               pts_3d.permute(1, 0))
        return loss

    def get_pose_prior_loss(self, frame_ind):
        predicted_rotation, predicted_translation = self.learned_poses.get_motion_model_predicted_pose(frame_ind)
        current_rotation = self.learned_poses.get_rotation_matrix(frame_ind)
        current_translation = self.learned_poses.get_translation_vector(frame_ind)
        relative_predicted_rotation = pytorch3d.transforms.matrix_to_quaternion(torch.matmul(pytorch3d.transforms.quaternion_to_matrix(predicted_rotation).T, current_rotation))
        relative_predicted_translation = torch.matmul(pytorch3d.transforms.quaternion_to_matrix(predicted_rotation).T, current_translation-predicted_translation)
        relative_rotation_error = F.smooth_l1_loss(relative_predicted_rotation, torch.Tensor([1., 0., 0., 0.]).to(relative_predicted_translation.device))
        relative_translation_error = F.smooth_l1_loss(relative_predicted_translation, torch.zeros_like(relative_predicted_translation))

        distortion_error = 0.
        if frame_ind > 0:
            current_scale = self.depth_distortion.global_scales[frame_ind]
            current_shift = self.depth_distortion.global_shifts[frame_ind]
            last_scale = self.depth_distortion.global_scales[frame_ind-1].data.detach()
            last_shift = self.depth_distortion.global_shifts[frame_ind-1].data.detach()
            distortion_error += F.smooth_l1_loss(current_scale, last_scale)
            distortion_error += F.smooth_l1_loss(current_shift, last_shift)

        return relative_rotation_error + relative_translation_error + distortion_error

    def get_epipolar_constraint_loss(self, frame_index, forward=True):
        if forward:
            flow = self.flows[frame_index]
            match_frame_index = frame_index + 1
        else:
            flow = self.inverse_flows[frame_index]
            frame_index = frame_index + 1
            match_frame_index = frame_index - 1

        depths = self.get_depth_by_index(frame_index)
        #Relative Pose
        current_rotation = self.learned_poses.get_rotation_matrix(frame_index)
        match_rotation = self.learned_poses.get_rotation_matrix(match_frame_index)
        relative_rotation = torch.matmul(match_rotation.T, current_rotation)
        relative_translation = torch.matmul(match_rotation.T, self.learned_poses.get_translation_vector(
            frame_index) - self.learned_poses.get_translation_vector(match_frame_index))
        relative_translation = relative_translation/((relative_translation+1e-5).norm())
        gt_flow = flow.permute(1, 2, 0)[depths > 1e-3][depths[depths > 1e-3] < torch.mean(depths)]
        current_pixels = self.pixel_pts[depths > 1e-3][depths[depths > 1e-3] < torch.mean(depths)]
        match_pixels = current_pixels + torch.cat([gt_flow, torch.zeros_like(gt_flow)[:,:1]], dim=1)
        def vec2skew(v):
            """
            :param v:  (3, ) torch tensor
            :return:   (3, 3)
            """
            zero = torch.zeros(1, dtype=torch.float32, device=v.device)
            skew_v0 = torch.cat([ zero,    -v[2:3],   v[1:2]])  # (3, 1)
            skew_v1 = torch.cat([ v[2:3],   zero,    -v[0:1]])
            skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
            skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
            return skew_v  # (3, 3)
        essential_matrix = torch.matmul(vec2skew(relative_translation), relative_rotation)
        inv_K = torch.linalg.inv(self.K)
        fundumental_matrix = torch.matmul(torch.matmul(inv_K.T, essential_matrix), inv_K)
        loss = torch.mean(torch.sum(torch.abs(torch.mul(torch.matmul(current_pixels , fundumental_matrix), match_pixels)), dim=1))/100
        return loss





    def predict_pose_with_motion_model(self, frame_ind):
        self.learned_poses.predict_pose_with_motion_model(frame_ind)


    def fix_other_frames(self, frame_ind):
        self.learned_poses.fix_other_frames(frame_ind)

    def fix_all_frames(self):
        self.learned_poses.fix_all_frames()

    def fix_first_frame(self):
        self.learned_poses.fix_first_frame()

    def fix_previous_frame(self, frame_ind, window_size):
        self.learned_poses.fix_previous_frame(frame_ind, window_size)


    def finish_fix_frames(self):
        self.learned_poses.finish_fix_frames()




    def initialize_pose(self, keyframe_interval):
        #0-initialize_num
        initialize_num = (self.window_size*2) * keyframe_interval
        initialize_num = self.initialize_num
        #Initialize pose
        self.keyframe_list = []
        with torch.autograd.no_grad():
            for i in range(len(self.T_wcs)):
                if i<= initialize_num:
                    copy_num = i
                    #Generate a new keyframe
                    if i >= 1:
                        self.get_flow(i)
                    if i%keyframe_interval == 0:
                        self.keyframe_list.append(i)

                else:
                    copy_num = initialize_num
                copy_num = 0
                self.learned_poses.copy_pose(i, copy_num)
        self.initialize_num = initialize_num
        self.current_ref_frame_indices = [item for item in self.keyframe_list]


    def initialize_pose_gt(self, keyframe_interval):
        #0-initialize_num
        initialize_num = (self.window_size*2) * keyframe_interval
        initialize_num = self.initialize_num
        #Initialize pose
        self.keyframe_list = []
        with torch.autograd.no_grad():
            for i in range(len(self.T_wcs)):
                copy_num = i
                if i<= initialize_num:
                    #Generate a new keyframe
                    if i%keyframe_interval == 0:
                        self.keyframe_list.append(i)
                self.learned_poses.copy_pose(i, copy_num)
        self.initialize_num = initialize_num
        self.current_ref_frame_indices = [item for item in self.keyframe_list]




    def bind_predict_depths(self, predict_depths, all_depths):
        self.gt_depths = all_depths
        self.predict_depths = predict_depths


    def get_depth_by_index(self, frame_index):
        scale, shift = self.depth_distortion(frame_index)
        depth = self.gt_depths[frame_index] * 1.0
        depth[depth>1e-3] = scale * depth[depth>1e-3] + shift
        return depth

    def get_tracking_depth_by_index(self, depth, frame_index):
        scale, shift = self.depth_distortion(frame_index)
        depth[depth>1e-3] = scale * depth[depth>1e-3] + shift
        return depth


    def bind_images_and_poses(self, imgs, K, T_wcs, all_rays, camera_rays):
        self.imgs = imgs
        self.h = self.imgs[0].shape[0]
        self.w = self.imgs[0].shape[1]
        self.K = K
        self.T_wcs = T_wcs
        self.learned_poses.bind_gt_pose(T_wcs)
        self.all_rays = all_rays
        self.camera_rays = camera_rays.view(self.h, self.w, 6)
        self.camera_pts = self.camera_rays[:,:,:3] + (1./self.camera_rays[:,:,5:]) * self.camera_rays[:,:,3:]
        self.pixel_pts = self.camera_pts @ self.K.T




    def create_new_keyframe(self, frame_ind):
        self.keyframe_list.append(frame_ind)
        self.current_ref_frame_indices.append(frame_ind)
        if len(self.current_ref_frame_indices) > self.window_size*2+1:
            self.current_ref_frame_indices.pop(0)


    def clamp_positions(self, positions):
        clamp_positions = torch.clamp(positions, torch.Tensor([0,0]).to(positions.device), torch.Tensor([self.w-2, self.h-2]).to(positions.device))
        clamp_mask = (positions-clamp_positions).norm(dim=1) > 3
        return clamp_positions, clamp_mask


    def distillation_mode(self):
        self.nerf.activate_color_net()
        self.nerf.fix_density()
        #Fix scales and distortions
        for parameter in self.learned_poses.parameters():
            parameter.requires_grad = False
            parameter.grad = None

        for parameter in self.depth_distortion.parameters():
            parameter.requires_grad = False
            parameter.grad = None

    #TODO:Fast implementation
    def bilinear_sample(self, img,  pixel_positions, dirs = None, depths = None):
        #Get four corners
        start = tim.time()
        pixel_positions_clamp = pixel_positions[:,:2]
        pixel_positions_clamp, clamp_mask = self.clamp_positions(pixel_positions_clamp)

        x_min_y_min = pixel_positions_clamp.int()
        x_min_y_max = pixel_positions_clamp.int()
        x_min_y_max[:,1] +=1

        x_max_y_min = pixel_positions_clamp.int()
        x_max_y_min[:,0] +=1
        x_max_y_max = pixel_positions_clamp.int()
        x_max_y_max[:,0] +=1
        x_max_y_max[:,1] +=1


        #Fast implementation
        def get_indices(x_y):
            indices = (x_y[:, 1] * self.w + x_y[:, 0]).long()
            return indices

        x_min_y_min_ind = get_indices(x_min_y_min)
        x_min_y_max_ind = x_min_y_min_ind + self.w
        x_max_y_min_ind = x_min_y_min_ind + 1
        x_max_y_max_ind = x_min_y_max_ind + 1


        img_view = img.view(-1,3)
        if dirs is not None:
            dir_view = dirs.view(-1, 3)
            img_view = torch.cat([img_view, dir_view], dim=1)
        if depths is not None:
            depths_view = depths.view(-1, 1)
            img_view = torch.cat([img_view, depths_view], dim=1)
        color_x_min_y_min = img_view[x_min_y_min_ind]
        color_x_min_y_max = img_view[x_min_y_max_ind]
        color_x_max_y_min = img_view[x_max_y_min_ind]
        color_x_max_y_max = img_view[x_max_y_max_ind]


        #FIXME:Fast
        x_diff = (pixel_positions_clamp - x_min_y_min)[:, 0]
        y_diff = (pixel_positions_clamp - x_min_y_min)[:, 1]
        color_y_min = x_diff.unsqueeze(1) * (color_x_max_y_min - color_x_min_y_min) + color_x_min_y_min
        color_y_max = x_diff.unsqueeze(1) * (color_x_max_y_max - color_x_min_y_max) + color_x_min_y_max
        color = y_diff.unsqueeze(1) * (color_y_max - color_y_min) + color_y_min


        if dirs is None:
            return color, clamp_mask
        else:
            return color[:,:3], color[:,3:] , clamp_mask


    #multi_frame
    def forward(self, rays, frame_index, **kwargs):
        nerf_index = 0
        #Get nearest three indices
        first_index = 0
        last_index = len(self.keyframe_list)-1
        for i in range(len(self.keyframe_list)):
            if self.keyframe_list[i] >= frame_index:
                if (i-1) >= first_index:
                    first_index = i-1
                if (i+1) <= last_index:
                    last_index = i+1
                break
        if last_index-first_index>3:
            first_index = last_index-3
            first_index = max(first_index, 0)

        #Long term connections
        long_connection_index1 = first_index-5
        if long_connection_index1 < 0:
            long_connection_index1 = 0


        long_connection_index2 = first_index-15
        if long_connection_index2 < 0:
            long_connection_index2 = 0


        long_connection_index3 = first_index-30
        if long_connection_index3 < 0:
            long_connection_index3 = 0



        used_indices = self.keyframe_list[first_index:last_index+1]
        if self.training and not self.tracking_mode:
            if long_connection_index1 < first_index:
                random_conn1 = random.randint(long_connection_index1, first_index-1)
                used_indices.append(random_conn1)
            if long_connection_index2 < long_connection_index1:
                random_conn2 = random.randint(long_connection_index2, long_connection_index1-1)
                used_indices.append(random_conn2)
            if long_connection_index3 < long_connection_index2:
                random_conn3 = random.randint(long_connection_index3, long_connection_index2-1)
                used_indices.append(random_conn3)
            # if long_connection_index4 < long_connection_index3:
            #     used_indices.append(long_connection_index4)
        ref_frame_indices = torch.LongTensor(used_indices).to(self.imgs.device)
        imgs = self.imgs[ref_frame_indices]
        # with torch.autograd.no_grad():
        ref_depths = [self.gt_depths[ind] for ind in ref_frame_indices]
        # ref_depths = [self.get_depth_by_index(ind).detach() for ind in ref_frame_indices]
        dirs = self.all_rays[ref_frame_indices,:,3:6].contiguous().view(len(ref_frame_indices),self.h,self.w,3)

        #Used for sample color
        shift_rotations = [self.learned_poses.get_rotation_matrix(ind) for ind in ref_frame_indices]
        shift_translations = [self.learned_poses.get_translation_vector(ind) for ind in ref_frame_indices]
        #Used for current frame
        if self.tracking_mode:
            current_shift_rotation = pytorch3d.transforms.quaternion_to_matrix(kwargs['current_shift_rotation'])
            current_shift_translation = kwargs['current_shift_translation']
        else:
            current_shift_rotation = self.learned_poses.get_rotation_matrix(frame_index)
            current_shift_translation = self.learned_poses.get_translation_vector(frame_index)
        # print('Cost1 is: ', tim.time()-start_time)

        if self.training:
            def occ_eval_fn(x):
                density = self.nerf(x)
                return density * self.render_step_size

            self.estimator.update_every_n_steps(
                step=kwargs['step'],
                occ_eval_fn=occ_eval_fn,
            )

        # print('Cost2 is: ', tim.time() - start_time)
        N_rays = rays.shape[0]
        rays_o, rays_d = rays[:, 0:3], rays[:, 3:6]  # both (B, 3)
        rays_o = rays_o + current_shift_translation
        rays_d = rays_d @ current_shift_rotation.T
        # print('Cost3 is: ', tim.time() - start_time)

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0
            sigmas = self.nerf(positions)
            return sigmas.squeeze(-1)

        with torch.autograd.no_grad():
            ray_indices, t_starts, t_ends = self.estimator.sampling(
                rays_o,
                rays_d,
                sigma_fn=sigma_fn,
                near_plane=self.near,
                far_plane=self.far,
                render_step_size=self.render_step_size,
                stratified=self.training,
            )

        # print('Cost4 is: ', tim.time() - start_time)
        if (len(ray_indices) == 0):
            result = None
            return result

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = rays_o[ray_indices]
            t_dirs = rays_d[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends)[:, None] / 2.0

            rgbs_list = []
            sample_dir_list = []
            for ii in range(len(imgs)):
                if self.training and len(imgs) > 1:
                    if ref_frame_indices[ii] == frame_index:
                        continue
                img = imgs[ii]
                dir = dirs[ii]
                ref_depth = ref_depths[ii]
                local_rotation = shift_rotations[ii].T
                local_translation = -torch.matmul(local_rotation, shift_translations[ii])
                local_positions = positions @ local_rotation.T + local_translation
                # if len(local_positions[:, 2:][torch.abs(local_positions[:, 2:]) < 1e-3]) > 0:
                #     print('Wrong positions!')
                local_positions[:, 2:][local_positions[:, 2:] < 1e-4] = 1.
                local_positions[:, :2][local_positions[:, 2] < 1e-4] = -1000.0
                # local_positions[:, 2:][torch.abs(local_positions[:, 2:]) < 1e-3] = 1e-3
                pixel_positions = (local_positions / local_positions[:, 2:]) @ self.K.T
                # print('Cost rotate is: ', tim.time() - start)

                #Sample
                rgbs, sample_dirs, clamp_mask = self.bilinear_sample(img, pixel_positions, dir, ref_depth)
                # print('Cost sample is: ', tim.time() - start)

                #Interpolate
                depths = sample_dirs[:,-1:].detach()
                sample_dirs = sample_dirs[:,:-1]
                rgbs_list.append(rgbs)

                distances = ((local_positions[:,2:] - depths)/(depths+1e-9)).detach()
                distance_thres = self.distance_thres
                unknown_distance = self.unknown_distance
                distances[depths<1e-3] = unknown_distance

                distance_ratio = torch.relu(distances)
                distance_ratio[distance_ratio<distance_thres] = distance_thres
                distance_ratio = ((distance_thres/distance_ratio)**2).squeeze(1)
                # distance_ratio[distance_ratio>2] = 2
                # distance_ratio = torch.pow((1e-3), distance_ratio).squeeze(1)

                rgb_weight = 1 / ((sample_dirs - t_dirs).norm(dim=1) ** 2 + 1e-7)
                # rgb_weight[distances.squeeze(1) > distance_thres] = rgb_weight[distances.squeeze(1) > distance_thres] * 0.0 + ((distance_thres/distances.squeeze(1)[distances.squeeze(1) > distance_thres])**2)
                if self.training:
                    rgb_weight[rgb_weight>1.] = 1.
                    rgb_weight = rgb_weight * distance_ratio
                else:
                    rgb_weight[distances.squeeze(1) > distance_thres] = rgb_weight[distances.squeeze(
                        1) > distance_thres] * 0.0 + ((distance_thres / distances.squeeze(1)[
                        distances.squeeze(1) > distance_thres]) ** 2)

                rgb_weight[clamp_mask] = 1e-7
                sample_dir_list.append(rgb_weight)
                # print('Cost others is: ', tim.time() - start)



            sum_weight = sample_dir_list[0]
            for k in range(1,len(sample_dir_list)):
                sum_weight = sum_weight + sample_dir_list[k]

            rgbs = (sample_dir_list[0]/sum_weight).unsqueeze(1) * rgbs_list[0]
            for k in range(1,len(sample_dir_list)):
                rgbs = rgbs + (sample_dir_list[k]/sum_weight).unsqueeze(1) * rgbs_list[k]



            t_dirs = torch.cat([t_dirs, rgbs], dim=1)
            ratios, sigmas = self.nerf(positions, t_dirs)

            # if self.training:
            #     num = random.randint(0,len(rgbs_list)-1)
            #     return rgbs_list[num], sigmas.squeeze(-1), rgbs_list

            return ratios, sigmas.squeeze(-1)

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
