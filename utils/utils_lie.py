import torch
import numpy as np
import logging
from matplotlib import pyplot as plt
import os
import shutil
from scipy.spatial.transform import Slerp

from scipy.spatial.transform import Rotation as R
import pytorch3d.transforms
# def vec2skew(v):
#     """
#     :param v:  (3, ) torch tensor
#     :return:   (3, 3)
#     """
#     zero = torch.zeros(1, dtype=torch.float32, device=v.device)
#     skew_v0 = torch.cat([ zero,    -v[2:3],   v[1:2]])  # (3, 1)
#     skew_v1 = torch.cat([ v[2:3],   zero,    -v[0:1]])
#     skew_v2 = torch.cat([-v[1:2],   v[0:1],   zero])
#     skew_v = torch.stack([skew_v0, skew_v1, skew_v2], dim=0)  # (3, 3)
#     return skew_v  # (3, 3)
#
#
# def Exp(r):
#     """so(3) vector to SO(3) matrix
#     :param r: (3, ) axis-angle, torch tensor
#     :return:  (3, 3)
#     """
#     skew_r = vec2skew(r)  # (3, 3)
#     norm_r = r.norm() + 1e-15
#     eye = torch.eye(3, dtype=torch.float32, device=r.device)
#     R = eye + (torch.sin(norm_r) / norm_r) * skew_r + ((1 - torch.cos(norm_r)) / norm_r**2) * (skew_r @ skew_r)
#     return R

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