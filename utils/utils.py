import torch
from ipdb import set_trace as S
import numpy as np
from PIL import Image
import json
import open3d
from plyfile import PlyData, PlyElement
import cv2
#Uncrop
#
# def read_resize_img_01tensor_hwc(img_path, hw, blend_a=True, background='white', crop = False, crop_size = -1):
#     img = Image.open(img_path) # img.size: (w,h)
#     # img = cv2.imread(img_path)
#     # assert img.size[0] * hw[0] == img.size[1] * hw[1], "HW resized ratio!"
#     w = hw[1]
#     h = hw[0]
#     if crop and crop_size > 0:
#         w += 2*crop_size
#         h += 2*crop_size
#     # img = img.resize((w, h), Image.NEAREST) # resize: (w,h)
#
#     img = (np.array(img) / 255.).astype(np.float32) # (h,w,c)
#     img =  cv2.resize(img, (h, w), interpolation=cv2.INTER_AREA)
#     if crop and crop_size > 0:
#         img = img[crop_size:-crop_size, crop_size:-crop_size]
#     if img.shape[2] == 4 and blend_a:
#         color, alpha = img[:,:,:3], img[:,:,3:]
#         if background == 'random':
#             bg_color = np.random.rand(3)
#         elif background == 'white':
#             bg_color = np.ones(3)
#         elif background == 'black':
#             bg_color = np.zeros(3)
#             alpha = np.ones_like(alpha)
#         img = color * alpha + bg_color * (1-alpha) # blend A to RGB
#
#
#
#     return torch.from_numpy(img)


def read_resize_img_01tensor_hwc(img_path, hw, blend_a=True, background='white', crop = False, crop_size = -1):
    img = Image.open(img_path) # img.size: (w,h)
    # assert img.size[0] * hw[0] == img.size[1] * hw[1], "HW resized ratio!"
    w = hw[1]
    h = hw[0]
    if crop and crop_size > 0:
        w += 2*crop_size
        h += 2*crop_size
    img = img.resize((w, h), Image.NEAREST) # resize: (w,h)
    img = np.array(img)/255 # (h,w,c)
    if crop and crop_size > 0:
        img = img[crop_size:-crop_size, crop_size:-crop_size]
    if img.shape[2] == 4 and blend_a:
        color, alpha = img[:,:,:3], img[:,:,3:] 
        if background == 'random':
            bg_color = np.random.rand(3)
        elif background == 'white':
            bg_color = np.ones(3)
        elif background == 'black':
            bg_color = np.zeros(3)
        img = color * alpha + bg_color * (1-alpha) # blend A to RGB
    return torch.from_numpy(img)

depth_scale = 1000
depth_scale_replica = 1./0.00015259021896696422

def read_resize_scale_depth_tensor_npz(depth_path, hw, scale, crop = False, crop_size = -1, depth_replica = False):
    used_depth_scale = 1.
    if depth_replica:
        used_depth_scale = depth_scale_replica
    w = hw[1]
    h = hw[0]
    if crop and crop_size > 0:
        w += 2 * crop_size
        h += 2 * crop_size

    img = np.load(depth_path)['pred']
    if img.shape[0] == 1:
        img = img[0]
    img = cv2.resize(img, (w, h), cv2.INTER_NEAREST) / used_depth_scale / scale
    if crop and crop_size > 0:
        img = img[crop_size:-crop_size, crop_size:-crop_size]

    return torch.as_tensor(img)




def read_resize_scale_depth_tensor(depth_path, hw, scale, crop = False, crop_size = -1, depth_replica = False):
    used_depth_scale = depth_scale
    if depth_replica:
        used_depth_scale = depth_scale_replica
    w = hw[1]
    h = hw[0]
    if crop and crop_size > 0:
        w += 2 * crop_size
        h += 2 * crop_size


    img = cv2.imread(depth_path, -1)
    img = cv2.resize(img, (w, h), cv2.INTER_NEAREST)/used_depth_scale/scale
    if crop and crop_size > 0:
        img = img[crop_size:-crop_size, crop_size:-crop_size]
    # print(img.shape, img.min(), img.max())
    # S()
    return torch.as_tensor(img)

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
        plane = open3d.geometry.TriangleMesh.create_box(width, height, depth=1e-6, create_uv_map=True)
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
        colors = [[0.2,0.2,0.8] for i in range(len(lines))]
        line_set = open3d.geometry.LineSet(
            points=open3d.utility.Vector3dVector(points_in_world),
            lines=open3d.utility.Vector2iVector(lines))
        line_set.colors = open3d.utility.Vector3dVector(colors)
        # line_set.rotate(R_my)
        return [axis, plane, line_set]
        
    # return as list in Open3D format
    
def view_directions(vis, origns, directions, color=[0.5,0.5,0.5], near=None, far=None, view_all=False):
    if view_all:
        points = torch.stack((origns, origns + directions*near), 0).reshape(-1,3)
        lines = [[i, i+origns.shape[0]] for i in range(origns.shape[0])]
        line_set = open3d.geometry.LineSet(
            points=open3d.utility.Vector3dVector(points),
            lines=open3d.utility.Vector2iVector(lines))
        line_set.colors = open3d.utility.Vector3dVector([np.array(color)/2 for i in range(len(lines))])
        vis.add_geometry(line_set)
    else:
        orign = origns[0,0:1,:]
        directions = [
            orign + directions[0:1,0,:]*near,
            orign + directions[-1:,0,:]*near,
            orign + directions[-1:,-1,:]*near,
            orign + directions[0:1,-1,:]*near,

            orign + directions[0:1,0,:]*far,
            orign + directions[-1:,0,:]*far,
            orign + directions[-1:,-1,:]*far,
            orign + directions[0:1,-1,:]*far,
        ]
        # S()
        points = torch.cat([orign] + directions, 0)
        lines = [
            [0, 1],
            [0, 2],
            [0, 3],
            [0, 4],
            
            [1, 5],
            [2, 6],
            [3, 7],
            [4, 8],
            
            [5, 6],
            [6, 7],
            [7, 8],
            [8, 5],
        ]
        line_set = open3d.geometry.LineSet(
            points=open3d.utility.Vector3dVector(points),
            lines=open3d.utility.Vector2iVector(lines[:4]))
        line_set.colors = open3d.utility.Vector3dVector([np.array(color)/3+0.5 for i in range(len(lines[:4]))])
        vis.add_geometry(line_set)
        line_set = open3d.geometry.LineSet(
            points=open3d.utility.Vector3dVector(points),
            lines=open3d.utility.Vector2iVector(lines[4:]))
        # color = np.random.rand(3)
        colors = [color for i in range(len(lines[4:]))]
        line_set.colors = open3d.utility.Vector3dVector(colors)
        vis.add_geometry(line_set)
        
def view_rays(vis, all_rays, color, near=None, far=None):
    for img_ray in all_rays[::8]:
        rays_o, rays_d = img_ray[:,:,:3], img_ray[:,:,3:6]
        view_directions(vis, rays_o, rays_d, color, near, far)

def view_poses(vis, K, H, W, poses, plane=True):
    color_ls = [[0.8, 0.2, 0.2], [0.2, 0.8, 0.2], [0.2, 0.2, 0.8]]
    frames = []
    for pose in poses[::8]:
        R = pose[:3, :3]
        t = pose[:3, 3]
        cam_model = create_camera(K, R, t, W, H, color=color_ls[0], plane=plane)
        frames.extend(cam_model)
    for frame in frames:
        vis.add_geometry(frame)

def view_imgs(vis, all_imgs):
    for path in all_imgs:
        img = open3d.io.read_image(path)
        S()
        vis.add_geometry(img)
        break
      
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

def view_pts3d(vis, pts3d, pts3d_color):
    pcd = open3d.geometry.PointCloud()
    # S()
    vis.get_render_option().point_size = 1.5
    pcd.points= open3d.utility.Vector3dVector(pts3d)
    pcd.colors = open3d.utility.Vector3dVector(pts3d_color)
    vis.add_geometry(pcd)

def view_mesh(vis, mesh_path):
    mesh = open3d.io.read_triangle_mesh(mesh_path)
    vis.add_geometry(mesh)
    

if __name__=="__main__":
    get_ray_directions(300,400,1)