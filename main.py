# import math
import os
# import time
from os.path import join
import sys
import torch
import numpy as np
np.set_printoptions(suppress=True)
from time import time
# from ipdb import set_trace as S
# from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
# from apex.optimizers import FusedAdam

from parameter import get_hparams
from dataset import *
from render import *
from render_tracking import TrackingAndMappingSampling, TrackingAndMappingSamplingSkip
from test import test_one_frame_tracking
from metric import *
import random


global all_global_iters
all_global_iters = 0

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


img2mse = lambda x, y : torch.mean((x - y) ** 2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def relative_depth_error_func(gt, result):
    gt_median, index = torch.median(gt, dim=0)
    gt_scale = torch.mean(torch.abs(gt-gt_median))
    result_median = result[index]
    result_scale = torch.mean(torch.abs(result - result_median))
    scaled_depth_gt = (gt-gt_median)/gt_scale
    scaled_depth_result = (result-result_median)/result_scale
    return F.smooth_l1_loss(scaled_depth_gt, scaled_depth_result) * result_scale/10



def train_iter(train_dataset, hparams, model, optimizer, all_ray_choices, iter, global_iters, all_iters, frame_ind, tracking_frames = [], stop_update_grid = False, relative_depth_error = False, with_motion_model = False):
    start = tim.time()
    global all_global_iters
    all_global_iters = all_global_iters + 1
    loss_func = F.smooth_l1_loss
    if relative_depth_error:
        depth_loss_func = relative_depth_error_func
    else:
        depth_loss_func = F.smooth_l1_loss
    i = iter
    batch_choice = frame_ind
    batch = train_dataset[batch_choice]
    rays_camera = train_dataset.camera_rays.view(-1, 6).to(device)
    # rays_all = batch['rays'].squeeze().view(-1, 6).to(device)  # (1, h*w, 8) -> h*w, 8
    if "Color" in hparams.Loss:
        all_color_target = batch['colors'].squeeze().view(-1, 3).to(device)
    if "Depth" in hparams.Loss:
        # all_depth_target = batch['depths'].view(-1, 1).to(device)
        all_depth_target = model.get_depth_by_index(batch_choice).view(-1, 1)

    mean_depth = torch.mean(all_depth_target)


    loss_ratio = (all_color_target.norm() / all_depth_target[all_depth_target > 1e-3].norm()).item()  * hparams['depth_loss_ratio']

    # loss_ratio = 0

    ray_choices = all_ray_choices
    # choose_indices = np.random.choice(ray_choices, size=[hparams.batch_size], replace=False)
    indices = torch.randperm(len(ray_choices))[:hparams.batch_size]
    # if stop_update_grid:
    #     indices = torch.LongTensor([i * int((len(ray_choices) - 1) / 1024) for i in range(1024)]).to(ray_choices.device)
    choose_indices = ray_choices[indices]

    used_rays_camera = rays_camera[choose_indices]


    result = model(used_rays_camera, frame_ind, step=global_iters, stop_update_grid = stop_update_grid)

    optimizer.zero_grad()

    end3 = tim.time()-start
    loss = 0
    if result is None:
        torch.cuda.empty_cache()
        return

    color_target = all_color_target[choose_indices]
    depth_target = all_depth_target[choose_indices]
    depth_mask = depth_target > 1e-3
    color_loss = loss_func(result["colors"], color_target)

    depth_loss = depth_loss_func(result["depths"][depth_mask], depth_target[depth_mask])

    depth_loss_test = F.smooth_l1_loss(result["depths"][depth_mask], depth_target[depth_mask])

    # depth_loss = loss_func(result["depths"][depth_mask], depth_target[depth_mask])
    #Relative Depth Loss
    # depth_target = depth_target[depth_mask]
    # depth_result = result["depths"][depth_mask]
    # median_depth = torch.median(depth_target)

    if hparams['use_inv_depth_loss']:
        depth_loss += depth_loss_func(1./(result["depths"][depth_mask]*hparams['inv_depth_ratio']), 1./(depth_target[depth_mask] * hparams['inv_depth_ratio']))
        depth_loss_test += F.smooth_l1_loss(1./(result["depths"][depth_mask]*hparams['inv_depth_ratio']), 1./(depth_target[depth_mask] * hparams['inv_depth_ratio']))

    # if relative_depth_error:
    #     print('depth scale: ', depth_loss_test/depth_loss)
    flow_loss = 0.
    compute_flow = False
    num = 0

    if hparams['use_flow_initialize']:
        num = 0
        for ind in tracking_frames:
            if ind > 0:
                num += 2
                #
                gt_flow, estimate_flow = model.get_pose_constraint_loss(ind - 1)
                flow_loss += loss_func(estimate_flow, gt_flow)
                # epipolar_loss = model.get_epipolar_constraint_loss(ind - 1)
                # flow_loss += epipolar_loss

                # gt_flow_inv, estimate_flow_inv = model.get_flow_loss(ind, result["depths"], indices, forward=False)
                gt_flow_inv, estimate_flow_inv = model.get_pose_constraint_loss(ind - 1, forward=False)
                # epopolar_loss_inv = model.get_epipolar_constraint_loss(ind-1, forward=False)
                flow_loss += loss_func(estimate_flow_inv, gt_flow_inv)
                # flow_loss += epopolar_loss_inv

                # icp_loss += model.get_icp_constraint_loss(ind - 1) * 5

                compute_flow = True
        if num > 0:
            flow_loss /= num




    loss += (color_loss + depth_loss * loss_ratio) * 10
    loss += flow_loss * 0.5


    if with_motion_model:
        pose_loss = model.get_pose_prior_loss(frame_ind)
        loss += hparams['prior_loss_ratio'] * pose_loss

    #
    # if len(result['color_list']) > 1:
    #     for i in range(len(result['color_list'])-1):
    #         pose_loss = loss_func(result['color_list'][i], result['color_list'][i+1])


    loss.backward()

    end4 = tim.time()-start

    optimizer.step()

    print_interval = 1
    if i % print_interval == 0:
        log = f"\rF {frame_ind:>2d} I {i}/{all_iters} Loss:"
        log += f" color={color_loss.item() * 10:>0.3f}"
        log += f" depth={depth_loss.item() * 10:>0.3f}"
        if compute_flow:
            log += f" flow={flow_loss.item():>0.3f}"

        if with_motion_model:
            log += f" pose={pose_loss.item() * 10:>0.3f}"
        log += f" time_for={end3:>0.3f} time_back={end4:>0.3f}"
        sys.stdout.write(log)
        torch.cuda.synchronize()
    del result
    optimizer.zero_grad()





#No depth loss
def train_iter_tracking(train_dataset, hparams, model, optimizer, all_ray_choices, iter, global_iters, all_iters, frame_ind, ref_ind, current_shift_rotation, current_shift_translation, stop_update_grid = False):
    # start = tim.time()

    loss_func = F.smooth_l1_loss
    i = iter
    batch_choice = frame_ind
    batch = train_dataset[batch_choice]
    rays_camera = train_dataset.camera_rays.view(-1, 6).to(device)
    # rays_all = batch['rays'].squeeze().view(-1, 6).to(device)  # (1, h*w, 8) -> h*w, 8
    if "Color" in hparams.Loss:
        all_color_target = batch['colors'].squeeze().view(-1, 3).to(device)
    if "Depth" in hparams.Loss:
        all_depth_target = batch['depths'].view(-1, 1).to(device)
        all_depth_target = model.get_tracking_depth_by_index(all_depth_target, batch_choice).view(-1, 1)

    ray_choices = all_ray_choices
    # choose_indices = np.random.choice(ray_choices, size=[hparams.batch_size], replace=False)
    indices = torch.randperm(len(ray_choices))[:hparams.batch_size]
    # if stop_update_grid:
    #     indices = torch.LongTensor([i * int((len(ray_choices) - 1) / 1024) for i in range(1024)]).to(ray_choices.device)
    choose_indices = ray_choices[indices]

    loss_ratio = (all_color_target.norm() / all_depth_target[all_depth_target > 1e-3].norm()).item()/4
    used_rays_camera = rays_camera[choose_indices]


    result = model(used_rays_camera, ref_ind, step=global_iters, stop_update_grid = stop_update_grid, current_shift_rotation = current_shift_rotation, current_shift_translation = current_shift_translation)

    optimizer.zero_grad()

    loss = 0
    if result is None:
        torch.cuda.empty_cache()
        return

    color_target = all_color_target[choose_indices]
    depth_target = all_depth_target[choose_indices]
    depth_mask = depth_target > 1e-3
    color_loss = loss_func(result["colors"], color_target)
    depth_loss = loss_func(result["depths"][depth_mask], depth_target[depth_mask])
    if hparams['use_inv_depth_loss']:
        depth_loss += depth_loss_func(1./(result["depths"][depth_mask]*hparams['inv_depth_ratio']), 1./(depth_target[depth_mask] * hparams['inv_depth_ratio']))
     #Relative Depth Loss

    loss += (color_loss) * 10

    loss.backward()
    optimizer.step()

    print_interval = 1
    if i % print_interval == 0:
        log = f"\rF {frame_ind:>2d} I {i}/{all_iters} Loss:"
        log += f" color={color_loss.item() * 10:>0.3f}"
        log += f" depth={depth_loss.item() * 10:>0.3f}"
        sys.stdout.write(log)
        torch.cuda.synchronize()
    del result
    optimizer.zero_grad()


def save_state(model, frame_ind, iters):
    state_dict = {
        'rotations': model.learned_poses.shift_rotations,
        'translations': model.learned_poses.shift_translations,
        'frame_ind': frame_ind,
        'iters': iters,
    }
    str_iters = str(iters)
    while len(str_iters) < 6:
        str_iters = '0' + str_iters
    filename = './trajectory/pose_' + str_iters  + '.pt'
    torch.save(state_dict, filename)
    # torch.save(model.learned_poses.shift_translations, filename)


if __name__=="__main__":
    setup_seed(0)
    hparams = get_hparams()
    print(hparams)
    os.makedirs(hparams.ckpt_path, exist_ok=True)
    pose_path = os.path.join(hparams.ckpt_path, 'poses')

    os.makedirs(pose_path, exist_ok=True)

    train_dataset = eval(hparams.data_class)(hparams, "train_all", check_pose=hparams.check_pose)
    batch_size = hparams.batch_size

    global_iters = 0
    data_length = hparams['data_length']

    batch_size = 1
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=False,
        num_workers=hparams.N_workers,
        batch_size=batch_size,
        pin_memory=True
    )

    test_dataloader = DataLoader(
        eval(hparams.data_class)(hparams, "test", check_pose=hparams.check_pose),
        shuffle=False,
        num_workers=hparams.N_workers,
        batch_size=1,
        pin_memory=True
    )

    model = TrackingAndMappingSamplingSkip(hparams).to(device)


    #Only depth loss here
    depth_loss_func = F.smooth_l1_loss

    print_interval = 1
    global_step = 0

    all_ray_choices = None

    all_ray_choices = torch.LongTensor([ind for ind in range(train_dataset.camera_rays.view(-1, 6).shape[0])]).to(device)

    frame_inices = [ind for ind in range(len(train_dataloader))]

    # depth prediction for all frames

    #Skip here
    K = train_dataset.K.to(device)
    all_imgs = train_dataset.all_colors.to(device)
    all_poses = train_dataset.poses.to(device)
    all_rays = train_dataset.all_rays.to(device)
    all_depths = train_dataset.all_depths.to(device)
    camera_rays = train_dataset.camera_rays.to(device)



    model.bind_images_and_poses(all_imgs, K, all_poses, all_rays, camera_rays)
    predicted_depths = all_depths
    model.bind_predict_depths(predicted_depths, all_depths)



    params_dict = [{'params': model.nerf.parameters(), 'lr': hparams.lr},
                   {'params': model.learned_poses.shift_rotations, 'lr': hparams.lr * hparams['rotation_lr_ratio']},
                   {'params': model.learned_poses.shift_translations, 'lr': hparams.lr * hparams['translation_lr_ratio']},
                   {'params': model.depth_distortion.parameters(), 'lr': hparams.lr * hparams['distortion_lr_ratio']}]

    # params_dict = [{'params': model.nerf.parameters(), 'lr': hparams.lr}]


    #
    pose_params_dict = [{'params': model.learned_poses.shift_rotations, 'lr': hparams.lr * hparams['rotation_lr_ratio']},
                        {'params': model.learned_poses.shift_translations, 'lr': hparams.lr * hparams['translation_lr_ratio']}]



    # params_dict = [{'params': model.nerf.parameters(), 'lr': hparams.lr}]


    optimizer = Adam(params_dict)
    pose_optimizer = Adam(pose_params_dict)
    scheduler = CosineAnnealingLR(optimizer, T_max=hparams.N_epochs * 50 * 50,
                                  eta_min=hparams.lr / hparams.shrink, verbose=True)



    epoch = 0
    fix_encoder_grad = False
    hparams.img_path = join(hparams.ckpt_path, 'images')
    os.makedirs(hparams.img_path, exist_ok=True)


    #Initialize the model
    keyframe_interval = hparams['keyframe_interval']
    model.initialize_pose(keyframe_interval)
    initialize_iters = hparams['initialize_iters']
    # model.enter_mapping_mode()
    # model.depth_distortion.global_shifts.requires_grad = False
    model.train()



    for i in range(initialize_iters*3):
        keyframe_num = len(model.keyframe_list)
        #Choose a frame randomly
        max_frame = model.keyframe_list[-1]
        tracking_frame_list = [tid for tid in range(1,max_frame+1)]
        frame_ind = random.randint(0, max_frame)
        # frame_ind = model.keyframe_list[frame_ind]
        if i < initialize_iters*2:
            train_iter(train_dataset, hparams, model, optimizer, all_ray_choices, i, global_iters, initialize_iters*3, frame_ind, tracking_frames=tracking_frame_list)
        else:
            train_iter(train_dataset, hparams, model, optimizer, all_ray_choices, i, global_iters, initialize_iters*3,
                       frame_ind, tracking_frames=[])
        global_iters +=1


    current_index = model.keyframe_list[keyframe_num-1]+1

    for frame_ind in range(current_index, data_length):
        model.depth_distortion.shift_scales(frame_ind-1)
        model.train()
        #Get flow
        if hparams['use_flow_initialize']:
            model.get_flow(frame_ind)

        create_keyframe =  False
        if frame_ind % keyframe_interval==0:
            #create a new keyframe
            create_keyframe = True
        #Track the pose of current frame
        model.predict_pose_with_motion_model(frame_ind)
        model.fix_other_frames(frame_ind)
        localize_iters = hparams['localize_iters']
        mapping_iters = hparams['mapping_iters']

        # model.depth_distortion.fix_other_scale_and_shift(frame_ind)
        # model.deactivate_mapping_network()
        for i in range(localize_iters):
            # train_iter(train_dataset, hparams, model, pose_optimizer, all_ray_choices, i, global_iters, localize_iters, frame_ind, tracking_frames=[frame_ind])
            with_motion_model = False
            if i<localize_iters/2:
                with_motion_model = True
            train_iter(train_dataset, hparams, model, pose_optimizer, all_ray_choices, i, global_iters, localize_iters, frame_ind, tracking_frames=[], with_motion_model=with_motion_model)
        # model.activate_mapping_network()
        # model.depth_distortion.free_scale_and_shift()

        #Create a new keyframe
        model.finish_fix_frames()

        if create_keyframe:
            #update map
            model.create_new_keyframe(frame_ind)
            # model.fix_first_frame()
            model.fix_previous_frame(frame_ind, keyframe_interval)
            for i in range(mapping_iters):
                #Only key-frames, no testing frames
                random_frame_indices = model.current_ref_frame_indices
                # print('Random indices are: ', random_frame_indices)
                all_ba_indices = [i for i in range(random_frame_indices[0], random_frame_indices[-1]+1)]
                random_frame_index = np.random.choice(random_frame_indices)
                train_iter(train_dataset, hparams, model, optimizer, all_ray_choices, i, global_iters, mapping_iters,random_frame_index, tracking_frames=[])
                global_iters += 1
            # test_one_frame_tracking(hparams, model, train_dataloader, frame_ind)
            # model.depth_distortion.free_scale_and_shift()
            model.finish_fix_frames()

        # test_one_frame_tracking(hparams, model, train_dataloader, frame_ind)

        if frame_ind % hparams['global_ba_interval'] == 0:
            print('Global BA')
            for i in range(hparams['global_ba_iters']*2):
                random_frame_indices = model.keyframe_list
                #Testing frames won't be optimized
                random_frame_indices = [k for k in range(random_frame_indices[0], random_frame_indices[-1] + 1) if
                                        (k + 1) % 8 != 0]

                # random_frame_indices = [k for k in range(random_frame_indices[0], random_frame_indices[-1] + 1)]
                # random_frame_indices = [k for k in range(random_frame_indices[0], random_frame_indices[-1] + 1)]
                # print('Random indices are: ', random_frame_indices)
                random_frame_index = np.random.choice(random_frame_indices)
                all_ba_indices = [k for k in range(random_frame_index - 1, random_frame_index + 1)]
                model.fix_other_frames(random_frame_index)
                train_iter(train_dataset, hparams, model, optimizer, all_ray_choices, i, global_iters * 2,
                           hparams['global_ba_iters'], random_frame_index, tracking_frames=[])
                global_iters += 1
                model.finish_fix_frames()

        torch.save(model.learned_poses.shift_rotations, pose_path + '/ba_rotation.pt')
        torch.save(model.learned_poses.shift_translations, pose_path + '/ba_translation.pt')



    print('Global BA')
    for i in range(hparams['global_ba_iters'] * 10 ):
        random_frame_indices = model.keyframe_list
        #Skip testing frames
        random_frame_indices = [k for k in range(random_frame_indices[0], random_frame_indices[-1]+1) if (k+1)%8!=0]
        # random_frame_indices = [k for k in range(random_frame_indices[0], random_frame_indices[-1]+1)]

        # print('Random indices are: ', random_frame_indices)
        random_frame_index = np.random.choice(random_frame_indices)
        all_ba_indices = [k for k in range(random_frame_index - 1, random_frame_index + 1)]
        model.fix_other_frames(random_frame_index)
        # model.depth_distortion.fix_other_scale_and_shift(random_frame_index)
        train_iter(train_dataset, hparams, model, optimizer, all_ray_choices, i, global_iters * 10,
                   hparams['global_ba_iters'], random_frame_index, tracking_frames=[])
        global_iters += 1
        # model.depth_distortion.free_scale_and_shift()
        model.finish_fix_frames()
        # if (i+1)% (hparams['global_ba_iters'] * 50) == 0:
        #     decay_rate = 0.9
        #     print('Adjust learning rate')
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = param_group['lr'] * decay_rate



    torch.save(model.learned_poses.shift_rotations, pose_path + '/ba_rotation.pt')
    torch.save(model.learned_poses.shift_translations, pose_path + '/ba_translation.pt')
    #Align test frames
    # model.fix_first_frame()
    for frame_ind in range(1, data_length):
        if frame_ind % 8 == 0:
            model.fix_other_frames(frame_ind-1)
            for i in range(hparams['test_localize_iters']):
                train_iter(train_dataset, hparams, model, pose_optimizer, all_ray_choices, i, global_iters,
                           hparams['test_localize_iters'], frame_ind-1, tracking_frames=[i for i in range(frame_ind-2, frame_ind+1)],
                           stop_update_grid=True)

            test_one_frame_tracking(hparams, model, train_dataloader, frame_ind - 1)
            model.finish_fix_frames()



    #tracking all test frames
    model.set_tracking_mode(True)
    tracking_length = hparams['tracking_skip_interval'] * data_length
    #initialize
    ref_shift_rotation = model.learned_poses.shift_rotations[0].data.detach().clone()
    ref_shift_translation = model.learned_poses.shift_translations[0].data.detach().clone()
    shift_rotations = torch.nn.Parameter(ref_shift_rotation)
    shift_translations = torch.nn.Parameter(ref_shift_translation)
    all_rotations = []
    all_translations = []
    for i in range(tracking_length):
        ref_index = int(i/hparams['tracking_skip_interval'])
        if ref_index*hparams['tracking_skip_interval'] == i:
            ref_shift_rotation = model.learned_poses.shift_rotations[ref_index].data.detach().clone()
            ref_shift_translation = model.learned_poses.shift_translations[ref_index].data.detach().clone()
            shift_rotations = torch.nn.Parameter(ref_shift_rotation)
            shift_translations = torch.nn.Parameter(ref_shift_translation)
            shift_rotations.requires_grad = True
            shift_translations.requires_grad = True
        else:
            tracking_params_dict = [
                {'params': shift_rotations, 'lr': hparams.lr * hparams['rotation_lr_ratio']},
                {'params': shift_translations, 'lr': hparams.lr * hparams['translation_lr_ratio']}]
            optimizer = Adam(tracking_params_dict)
            for it in range(100):
                train_iter_tracking(test_dataloader.dataset, hparams, model, optimizer, all_ray_choices, it, global_iters, 100, i, ref_index, shift_rotations, shift_translations, True)
        all_rotations.append(shift_rotations.data.detach().clone())
        all_translations.append(shift_translations.data.detach().clone())

    all_rotations = torch.stack(all_rotations)
    all_translations = torch.stack(all_translations)

    torch.save(all_rotations, pose_path + '/tracking_rotation.pt')
    torch.save(all_translations, pose_path + '/tracking_translation.pt')

