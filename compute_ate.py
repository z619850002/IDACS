import os
import sys
import logging
import time
import argparse

import numpy as np
import torch

from utils_poses.comp_ate import compute_ATE, compute_rpe
from utils_poses.align_traj import align_ate_c2b_use_a2b
import pytorch3d.transforms

import open3d

def create_vis():
    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(open3d.geometry.TriangleMesh.create_coordinate_frame())
    radius = 4
    # vis.add_geometry(open3d.geometry.TriangleMesh.create_box(2*radius,2*radius,2*radius).translate([-radius, -radius, -radius]))
    return vis



def show_vis(vis, close=True):
    vis.poll_events()
    vis.update_renderer()
    vis.run()
    if close:
        vis.destroy_window()


def create_camera(K_tensor, R, t, w, h, scale=1, color=[0.8, 0.8, 0.8], plane=True):
    """Create axis, plane and pyramed geometries in Open3D format.
    :param K: calibration matrix (camera intrinsics)
    :param R: rotation matrix
    :param t: translation
    :param w: image width
    :param h: image height
    :param scale: camera model scale
    :param color: color of the image plane and pyramid lines
    :return: camera model geometries (axis, plane and pyramid)
    """

    # intrinsics
    K = K_tensor / scale
    Kinv = np.linalg.inv(K)

    # R[:,1] *= -1
    # R[:,2] *= -1

    # 4x4 transformation
    T = np.column_stack((R, t))
    T = np.vstack((T, (0, 0, 0, 1)))
    # axis
    axis = open3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5 * scale)
    axis.transform(T)
    # R_my = axis.get_rotation_matrix_from_zyx((-np.pi/2, np.pi/2, 0))
    # axis.rotate(R_my)
    # points in pixel

    if not plane:
        return [axis]
    else:
        points_pixel = [
            [0, 0, 0],
            [0, 0, 1],
            [w, 0, 1],
            [0, h, 1],
            [w, h, 1],
        ]
        # pixel to camera coordinate system
        points = [Kinv @ p for p in points_pixel]

        # image plane
        width = abs(points[1][0]) + abs(points[3][0])
        height = abs(points[1][1]) + abs(points[3][1])
        plane = open3d.geometry.TriangleMesh.create_box(width, height, depth=1e-7)
        # plane.paint_uniform_color([0.8,0.2,0.2])
        plane.translate([points[1][0], points[1][1], scale])
        plane.transform(T)
        # plane.rotate(R_my)

        # pyramid
        points_in_world = [(R @ p + t) for p in points]
        lines = [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
        ]
        colors = [color for i in range(len(lines))]
        line_set = open3d.geometry.LineSet()
        line_set.points=open3d.utility.Vector3dVector(points_in_world)
        line_set.lines=open3d.utility.Vector2iVector(lines)
        line_set.colors = open3d.utility.Vector3dVector(colors)
        # line_set.rotate(R_my)
        return [axis, plane, line_set]

    # return as list in Open3D format

def view_poses(vis, K, H, W, poses, traj_ind, plane=True):
    color_ls = [[0.2, 0.8, 0.2], [0.8, 0.2, 0.2], [0.2, 0.2, 0.8]]

    frames = []
    for pose in poses[::1]:
        R = pose[:3, :3]
        t = pose[:3, 3]
        cam_model = create_camera(K, R, t, W, H, color=color_ls[traj_ind], plane=plane)
        frames.extend(cam_model)
    for frame in frames:
        vis.add_geometry(frame)

def visualize_poses(poses1, poses2):
    h = 540
    w = 960
    K = torch.Tensor([[587.94, 0., 480.0], [0., 587.94, 270.0], [0.,0.,1.]]).numpy()
    vis = create_vis()
    view_poses(vis, K, h, w, poses1, 0)
    view_poses(vis, K, h, w, poses2, 1)
    color = [0, 0, 1]
    show_vis(vis)



def Exp(data):
    # batch_dims = data.shape[:-1]

    theta = torch.norm(data, dim=-1, keepdim=True)
    omega = data / theta

    omega1 = omega[...,0:1]
    omega2 = omega[...,1:2]
    omega3 = omega[...,2:3]
    zeros = torch.zeros_like(omega1)

    K = torch.concat([torch.concat([zeros, -omega3, omega2], dim=-1)[...,None,:],
                      torch.concat([omega3, zeros, -omega1], dim=-1)[...,None,:],
                      torch.concat([-omega2, omega1, zeros], dim=-1)[...,None,:]], dim=-2)
    I = torch.eye(3).expand(3,3).to(data)

    return I + torch.sin(theta).unsqueeze(-1) * K + (1. - torch.cos(theta).unsqueeze(-1)) * (K @ K)

def Log(rot):
    """
    :param rot: [N, 3, 3]
    :return:
    """
    return pytorch3d.transforms.quaternion_to_axis_angle(pytorch3d.transforms.matrix_to_quaternion(rot))



if __name__=="__main__":
    gt_path = sys.argv[1]
    estimated_path = sys.argv[2]
    estimated_rotations = torch.load(os.path.join(estimated_path, "tracking_rotation.pt"))
    estimated_translations = torch.load(os.path.join(estimated_path + "tracking_translation.pt"))
    gt_poses = torch.load(gt_path)
    gt_poses = gt_poses[:,:3,:]
    gt_poses[:, :,1:3] *= -1  # [right up back] to [right down front]
    estimated_poses = []
    for i in range(len(estimated_rotations)):
        estimated_rotation = pytorch3d.transforms.quaternion_to_matrix(estimated_rotations[i])
        estimated_translation = estimated_translations[i]
        estimated_pose = torch.cat([estimated_rotation, estimated_translation.unsqueeze(1)], dim=1)

        estimated_pose = torch.cat([estimated_pose, torch.Tensor([0., 0., 0., 1.]).unsqueeze(0).to(estimated_pose.device)], dim=0)
        estimated_poses.append(estimated_pose)
    estimated_poses = torch.stack(estimated_poses)
    estimated_poses = estimated_poses[0:len(gt_poses)].detach()
    gt_poses = gt_poses[0:len(gt_poses)].detach()
    gt_poses = torch.cat([gt_poses, torch.Tensor([0.,0.,0.,1.]).unsqueeze(0).unsqueeze(1).repeat(len(gt_poses),1,1).to(gt_poses.device)], dim=1)

    #Scale family
    scale = 1.


    gt_poses[:,:3,3] =  gt_poses[:,:3,3] * scale
    estimated_poses[:,:3,3] =  estimated_poses[:,:3,3]


    c2ws_est_aligned = align_ate_c2b_use_a2b(estimated_poses, gt_poses)


    ate = compute_ATE(gt_poses.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
    rpe_trans, rpe_rot = compute_rpe(gt_poses.cpu().numpy(), c2ws_est_aligned.cpu().numpy())
    # visualize_poses(gt_poses.cpu().numpy(), c2ws_est_aligned.cpu().numpy())

    print('ATE is: ', ate)
    print('RPE trans is: ', rpe_trans * 100)
    print('RPE rot is: ', rpe_rot * 180 / 3.1416)
