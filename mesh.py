import os
import time
import torch
from render import *
from utils import *
from ipdb import set_trace as S
import argparse
import numpy as np
from kornia.color import grayscale_to_rgb
import mcubes
import trimesh
from plyfile import PlyData, PlyElement
import open3d as o3d
from dataset import *
import skimage
import skimage.measure

np.set_printoptions(suppress=True)
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def extract_mesh(density_func, radius=1, render_func='Vanilla'):
    chunk_size=1024*64
    ### Tune these parameters until the whole object lies tightly in range with little noise ###
    N = 256 # controls the resolution, set this number small here because we're only finding
            # good ranges here, not yet for mesh reconstruction; we can set this number high
            # when it comes to final reconstruction.
    xmin, xmax = -radius, radius # left/right range
    ymin, ymax = -radius, radius # forward/backward range
    zmin, zmax = -radius, radius # up/down range
    ## Attention! the ranges MUST have the same length!
    sigma_threshold = 50. # controls the noise (lower=maybe more noise; higher=some mesh might be missing)
    ############################################################################################

    x = torch.linspace(xmin, xmax, N)
    y = torch.linspace(ymin, ymax, N)
    z = torch.linspace(zmin, zmax, N)
    x, y, z = torch.meshgrid(x, y, z, indexing='xy') # both (N,N,N)
    xyz_samples = torch.stack((x,y,z), -1).reshape(-1, 3).cuda()
    # dir_ = torch.zeros_like(xyz_).cuda()


    sigma_ls = []
    with torch.no_grad():
        for i in range(0, xyz_samples.shape[0], chunk_size):
            sigma = density_func(xyz_samples[i:i+chunk_size])
            sigma_ls.append(sigma)
    sigma = torch.cat(sigma_ls, 0).reshape(N, N, N)
    sigma = sigma.cpu().numpy()
    sigma = np.maximum(sigma, 0)
    print(sigma.min(), sigma.max())

    if render_func == "Vanilla":
        vertices, triangles = mcubes.marching_cubes(sigma, sigma_threshold)
        # mcubes.export_mesh(vertices, triangles, "lego.dae")
        mesh = trimesh.Trimesh(vertices/N, triangles)
    else:
        sdf = sigma
        vertices, faces, normals, values = skimage.measure.marching_cubes(
            sdf, level=0.0, spacing=[radius, radius, radius]
        )
        # vertices += [-radius/2., -radius/2., -radius/2.]
        # mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # return mesh
    mesh.show()
    S()

    vertices_ = (vertices/N).astype(np.float32)
    ## invert x and y coordinates (WHY? maybe because of the marching cubes algo)
    x_ = (ymax-ymin) * vertices_[:, 1] + ymin
    y_ = (xmax-xmin) * vertices_[:, 0] + xmin
    vertices_[:, 0] = x_
    vertices_[:, 1] = y_
    vertices_[:, 2] = (zmax-zmin) * vertices_[:, 2] + zmin
    vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]

    face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = triangles

    PlyData([PlyElement.describe(vertices_[:,0],'vertex'), PlyElement.describe(face,'face')]).write(mesh_path)

    # remove noise in the mesh by keeping only the biggest cluster
    print('Removing noise ...')
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    idxs, count, _ = mesh.cluster_connected_triangles()
    max_cluster_idx = np.argmax(count)
    triangles_to_remove = [i for i in range(len(face)) if idxs[i] != max_cluster_idx]
    mesh.remove_triangles_by_index(triangles_to_remove)
    mesh.remove_unreferenced_vertices()
    print(f'Mesh has {len(mesh.vertices)/1e6:.2f} M vertices and {len(mesh.triangles)/1e6:.2f} M faces.')
    S()
    vertices_ = np.asarray(mesh.vertices).astype(np.float32)
    triangles = np.asarray(mesh.triangles)

def f():
    # perform color prediction
    # Step 0. define constants (image width, height and intrinsics)
    W, H = args.img_wh
    K = np.array([[dataset.focal, 0, W/2],
                  [0, dataset.focal, H/2],
                  [0,             0,   1]]).astype(np.float32)

    # Step 1. transform vertices into world coordinate
    N_vertices = len(vertices_)
    vertices_homo = np.concatenate([vertices_, np.ones((N_vertices, 1))], 1) # (N, 4)

    if args.use_vertex_normal: ## use normal vector method as suggested by the author.
                               ## see https://github.com/bmild/nerf/issues/44
        mesh.compute_vertex_normals()
        rays_d = torch.FloatTensor(np.asarray(mesh.vertex_normals))
        near = dataset.bounds.min() * torch.ones_like(rays_d[:, :1])
        far = dataset.bounds.max() * torch.ones_like(rays_d[:, :1])
        rays_o = torch.FloatTensor(vertices_) - rays_d * near * args.near_t

        nerf_coarse = NeRF()
        load_ckpt(nerf_coarse, args.ckpt_path, model_name='nerf_coarse')
        nerf_coarse.cuda().eval()

        results = f([nerf_coarse, nerf_fine], embeddings,
                    torch.cat([rays_o, rays_d, near, far], 1).cuda(),
                    args.N_samples,
                    args.N_importance,
                    args.chunk,
                    dataset.white_back)

    else: ## use my color average method. see README_mesh.md
        ## buffers to store the final averaged color
        non_occluded_sum = np.zeros((N_vertices, 1))
        v_color_sum = np.zeros((N_vertices, 3))

        # Step 2. project the vertices onto each training image to infer the color
        print('Fusing colors ...')
        for idx in tqdm(range(len(dataset.image_paths))):
            ## read image of this pose
            image = Image.open(dataset.image_paths[idx]).convert('RGB')
            image = image.resize(tuple(args.img_wh), Image.LANCZOS)
            image = np.array(image)

            ## read the camera to world relative pose
            P_c2w = np.concatenate([dataset.poses[idx], np.array([0, 0, 0, 1]).reshape(1, 4)], 0)
            P_w2c = np.linalg.inv(P_c2w)[:3] # (3, 4)
            ## project vertices from world coordinate to camera coordinate
            vertices_cam = (P_w2c @ vertices_homo.T) # (3, N) in "right up back"
            vertices_cam[1:] *= -1 # (3, N) in "right down forward"
            ## project vertices from camera coordinate to pixel coordinate
            vertices_image = (K @ vertices_cam).T # (N, 3)
            depth = vertices_image[:, -1:]+1e-5 # the depth of the vertices, used as far plane
            vertices_image = vertices_image[:, :2]/depth
            vertices_image = vertices_image.astype(np.float32)
            vertices_image[:, 0] = np.clip(vertices_image[:, 0], 0, W-1)
            vertices_image[:, 1] = np.clip(vertices_image[:, 1], 0, H-1)

            ## compute the color on these projected pixel coordinates
            ## using bilinear interpolation.
            ## NOTE: opencv's implementation has a size limit of 32768 pixels per side,
            ## so we split the input into chunks.
            colors = []
            remap_chunk = int(3e4)
            for i in range(0, N_vertices, remap_chunk):
                colors += [cv2.remap(image, 
                                    vertices_image[i:i+remap_chunk, 0],
                                    vertices_image[i:i+remap_chunk, 1],
                                    interpolation=cv2.INTER_LINEAR)[:, 0]]
            colors = np.vstack(colors) # (N_vertices, 3)
            
            ## predict occlusion of each vertex
            ## we leverage the concept of NeRF by constructing rays coming out from the camera
            ## and hitting each vertex; by computing the accumulated opacity along this path,
            ## we can know if the vertex is occluded or not.
            ## for vertices that appear to be occluded from every input view, we make the
            ## assumption that its color is the same as its neighbors that are facing our side.
            ## (think of a surface with one side facing us: we assume the other side has the same color)

            ## ray's origin is camera origin
            rays_o = torch.FloatTensor(dataset.poses[idx][:, -1]).expand(N_vertices, 3)
            ## ray's direction is the vector pointing from camera origin to the vertices
            rays_d = torch.FloatTensor(vertices_) - rays_o # (N_vertices, 3)
            rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
            near = dataset.bounds.min() * torch.ones_like(rays_o[:, :1])
            ## the far plane is the depth of the vertices, since what we want is the accumulated
            ## opacity along the path from camera origin to the vertices
            far = torch.FloatTensor(depth) * torch.ones_like(rays_o[:, :1])
            results = f([nerf_fine], embeddings,
                        torch.cat([rays_o, rays_d, near, far], 1).cuda(),
                        args.N_samples,
                        0,
                        args.chunk,
                        dataset.white_back)
            opacity = results['opacity_coarse'].cpu().numpy()[:, np.newaxis] # (N_vertices, 1)
            opacity = np.nan_to_num(opacity, 1)

            non_occluded = np.ones_like(non_occluded_sum) * 0.1/depth # weight by inverse depth
                                                                    # near=more confident in color
            non_occluded += opacity < args.occ_threshold
            
            v_color_sum += colors * non_occluded
            non_occluded_sum += non_occluded

    # Step 3. combine the output and write to file
    if args.use_vertex_normal:
        v_colors = results['rgb_fine'].cpu().numpy() * 255.0
    else: ## the combined color is the average color among all views
        v_colors = v_color_sum/non_occluded_sum
    v_colors = v_colors.astype(np.uint8)
    v_colors.dtype = [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    vertices_.dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    vertex_all = np.empty(N_vertices, vertices_.dtype.descr+v_colors.dtype.descr)
    for prop in vertices_.dtype.names:
        vertex_all[prop] = vertices_[prop][:, 0]
    for prop in v_colors.dtype.names:
        vertex_all[prop] = v_colors[prop][:, 0]
        
    face = np.empty(len(triangles), dtype=[('vertex_indices', 'i4', (3,))])
    face['vertex_indices'] = triangles

    PlyData([PlyElement.describe(vertex_all, 'vertex'), 
             PlyElement.describe(face, 'face')]).write(f'{args.scene_name}.ply')

    print('Done!')


from yacs.config import CfgNode as CN

def mesh_hnv():
    ckpt_path = "save/hnv_Color/308_huge/epoch=30.pth"
    load = torch.load(ckpt_path)['model']
    dic = {}
    for k in load.keys():
        if 'nerf' in k and 'ln_beta' not in k:
            dic[k.replace('nerf.', '')] = load[k]     
    hparams = CN(new_allowed=True)
    hparams.Function = 'Vanilla'
    hparams.scale = 1.
    hparams.NEAR_DISTANCE = 0.01
    model = HashRadianceField(hparams).to(device)
    model.load_state_dict(dic)
    print(f"Successfully loading from {ckpt_path}\n")

    mesh = extract_mesh(model.density, radius=hparams.scale, render_func='Vanilla')

def mesh_mvv():
    
    ckpt_path = "save/venilla/lego/epoch=18.pth"
    load = torch.load(ckpt_path)['model']
    dic = {}
    for k in load.keys():
        if 'fine' in k and 'ln_beta' not in k:
            dic[k.replace('fine.', '')] = load[k]     
    hparams = CN(new_allowed=True)
    hparams.Function = 'Vanilla'
    model = MLPRadianceField(hparams).to(device)
    model.load_state_dict(dic)
    print(f"Successfully loading from {ckpt_path}\n")

    mesh = extract_mesh(model.forward, radius=1, render_func='Vanilla')
    
def mesh_mvs():
    ckpt_path = "save/mvs_Color/lego/epoch=30.pth"
    load = torch.load(ckpt_path)['model']
    dic = {}
    for k in load.keys():
        if 'fine' in k:
            dic[k.replace('fine.', '')] = load[k]     
    hparams = CN(new_allowed=True)
    hparams.Function = 'SDF'
    hparams.speed_factor = 10
    hparams.beta_init = 0.1
    model = MLPRadianceField(hparams).to(device)
    model.load_state_dict(dic)
    print(f"Successfully loading from {ckpt_path}\n")

    mesh = extract_mesh(model.forward, radius=1, render_func='SDF')
    
    
if __name__=="__main__":
    mesh_hnv()