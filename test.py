import time
import numpy as np
np.set_printoptions(suppress=True)
from kornia.color import grayscale_to_rgb

from torch.utils.data import DataLoader 
from torchvision.utils import save_image

from utils import *
from dataset import *
from render import *
from metric import *
from parameter import get_hparams
from eval import pytorch_ssim
import cv2

import lpips as lpips_lib
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lpips_metric = lpips_lib.LPIPS(net='vgg').to(device)

@torch.no_grad()
def test(hparams, model, test_dataloader, N_test=-1):
    model.eval()
    avg_psnr = 0
    avg_cost = 0
    if N_test < 0 or N_test > len(test_dataloader):
        N_test = len(test_dataloader)
    hw = (test_dataloader.dataset.h, test_dataloader.dataset.w)
    for i, batch in enumerate(test_dataloader):
        rays = batch['rays'].squeeze().to(device)  # (1, h*w, 8) -> h*w, 8

        prev_time = time.time()
        result = model(rays, step=i)
        torch.cuda.synchronize()
        cost_time = time.time() - prev_time
        avg_cost += cost_time
        
        save_ls = []
        if result is None:
            continue
        colors = result['colors'].reshape(*hw, 3).cpu()
        save_ls.append(colors.permute(2,0,1))
        ncol = 1
        if 'colors' in batch.keys():
            target = batch['colors'].squeeze() # h, w, 3
            psnr_result = psnr(target, colors).item()
            avg_psnr += psnr_result
            print(f"\rImg {i+1:>3d}/{len(test_dataloader)} psnr={psnr_result:>0.2f} cost={cost_time:>0.2f}s")
            save_path = join(hparams.img_path, f"{i:0>3}_{psnr_result:>0.2f}.jpg")
            save_ls.append(target.permute(2,0,1))
        else:
            print(f"\rImg {i+1:>3d}/{len(test_dataloader)} cost={cost_time:>0.2f}s")
            save_path = join(hparams.img_path, f"{i:0>3}.jpg")
        
        if 'depths' in result.keys():
            ncol = 2
            depths = result['depths'].reshape(1, *hw).cpu()
            depths = torch.nan_to_num(depths) # change nan to 0
            depths = depths/3
            # depths = (depths-depths.min())/(depths.max()-depths.min()+1e-8) # normalize to 0~1
            depths = grayscale_to_rgb(depths)
            save_ls.append(depths)
        if 'depths' in batch.keys():
            ncol = 2
            target = batch['depths']/3 # h, w
            target = grayscale_to_rgb(target)
            save_ls.append(target)

        save_image(save_ls, save_path, nrow=2, ncol=ncol)
        torch.cuda.empty_cache()
        if i == N_test-1:
            break 
    avg_psnr /= N_test
    avg_cost /= N_test
    os.rename(hparams.img_path, f"{hparams.img_path}_{avg_psnr:>0.2f}_{cost_time:>0.2f}s")
    return avg_psnr


@torch.no_grad()
def test_one_frame_tracking(hparams, model, test_dataloader, test_index):
    model.eval()
    avg_psnr = 0
    avg_cost = 0
    hw = (test_dataloader.dataset.h, test_dataloader.dataset.w)

    batch = test_dataloader.dataset[test_index]
    i = test_index

    rays = test_dataloader.dataset.camera_rays.view(-1, 6).to(device)
    # rays = batch['rays'].squeeze().to(device)  # (1, h*w, 8) -> h*w, 8

    # indices = batch['indices'].squeeze().view(-1)
    prev_time = time.time()
    results = []
    chunksize = 4990
    for k in range(int(len(rays) / chunksize) + 1):
        start = k * chunksize
        end = min(start + chunksize, len(rays))
        result = model(rays[start:end], i, step=k, test=True)
        results.append(result)
        torch.cuda.synchronize()

    cost_time = time.time() - prev_time
    avg_cost += cost_time

    save_ls = []
    # if result is None:
    #     continue
    colors_ls = []
    depths_ls = []
    for result in results:
        colors_ls.append(result['colors'])
        depths_ls.append(result['depths'])

    result = {}
    result['colors'] = torch.cat(colors_ls, dim=0)
    result['depths'] = torch.cat(depths_ls, dim=0)

    colors = result['colors'].reshape(*hw, 3).cpu()


    save_ls.append(colors.permute(2, 0, 1))
    ncol = 1
    if 'colors' in batch.keys():

        color_mask = colors.norm(dim=2) > 1e-2
        target = batch['colors'].squeeze()  # h, w, 3
        psnr_result = psnr(target[color_mask], colors[color_mask]).item()

        ssim_result = pytorch_ssim.ssim(target.permute(2, 0, 1).unsqueeze(0), colors.permute(2, 0, 1).unsqueeze(0)).item()

        lpips_loss = lpips_metric(target.permute(2, 0, 1).unsqueeze(0).contiguous().to(device),
                                  colors.permute(2, 0, 1).unsqueeze(0).contiguous().to(device), normalize=True).item()

        avg_psnr += psnr_result
        print(f"\rImg {i + 1:>3d}/{len(test_dataloader)} psnr={psnr_result:>0.2f}  ssim={ssim_result:>0.2f} lpips_loss={lpips_loss:>0.2f} cost={cost_time:>0.2f}s")
        save_path = join(hparams.img_path, f"{i:0>3}_{psnr_result:>0.2f}_{ssim_result:>0.2f}_{lpips_loss:>0.2f}.jpg")
        save_ls.append(target.permute(2, 0, 1))
    else:
        print(f"\rImg {i + 1:>3d}/{len(test_dataloader)} cost={cost_time:>0.2f}s")
        save_path = join(hparams.img_path, f"{i:0>3}.jpg")

    if 'depths' in result.keys():
        ncol = 2
        depths = result['depths'].reshape(1, *hw).cpu()
        depths = torch.nan_to_num(depths)  # change nan to 0
        depths = depths * 1.5
        # depths = (depths-depths.min())/(depths.max()-depths.min()+1e-8) # normalize to 0~1
        depths = grayscale_to_rgb(depths)
        save_ls.append(depths)
    if 'depths' in batch.keys():
        ncol = 2
        target = (batch['depths'] ).reshape(1, *hw).cpu() # h, w
        target = grayscale_to_rgb(target * 1.5)
        save_ls.append(target)

    save_image(save_ls, save_path, nrow=2, ncol=ncol)
    del result
    torch.cuda.empty_cache()
    model.train()
    return avg_psnr





@torch.no_grad()
def test_one_frame(hparams, model, test_dataloader, test_index):
    model.eval()
    avg_psnr = 0
    avg_cost = 0
    hw = (test_dataloader.dataset.h, test_dataloader.dataset.w)

    batch = test_dataloader.dataset[test_index]
    i = test_index

    rays = batch['rays'].squeeze().to(device)  # (1, h*w, 8) -> h*w, 8

    indices = batch['indices'].squeeze().view(-1)
    prev_time = time.time()
    results = []
    chunksize = 499
    for k in range(int(len(rays) / chunksize) + 1):
        start = k * chunksize
        end = min(start + chunksize, len(rays))
        result = model(rays[start:end], i, i, step=i, test=True)
        results.append(result)
        torch.cuda.synchronize()

    cost_time = time.time() - prev_time
    avg_cost += cost_time

    save_ls = []
    # if result is None:
    #     continue
    colors_ls = []
    depths_ls = []
    for result in results:
        colors_ls.append(result['colors'])
        depths_ls.append(result['depths'])

    result = {}
    result['colors'] = torch.cat(colors_ls, dim=0)
    result['depths'] = torch.cat(depths_ls, dim=0)

    colors = result['colors'].reshape(*hw, 3).cpu()
    save_ls.append(colors.permute(2, 0, 1))
    ncol = 1
    if 'colors' in batch.keys():

        color_mask = colors.norm(dim=2) > 1e-3
        target = batch['colors'].squeeze()  # h, w, 3
        psnr_result = psnr(target[color_mask], colors[color_mask]).item()
        avg_psnr += psnr_result
        print(f"\rImg {i + 1:>3d}/{len(test_dataloader)} psnr={psnr_result:>0.2f} cost={cost_time:>0.2f}s")
        save_path = join(hparams.img_path, f"{i:0>3}_{psnr_result:>0.2f}.jpg")
        save_ls.append(target.permute(2, 0, 1))
    else:
        print(f"\rImg {i + 1:>3d}/{len(test_dataloader)} cost={cost_time:>0.2f}s")
        save_path = join(hparams.img_path, f"{i:0>3}.jpg")

    if 'depths' in result.keys():
        ncol = 2
        depths = result['depths'].reshape(1, *hw).cpu()
        depths = torch.nan_to_num(depths)  # change nan to 0
        depths = depths
        # depths = (depths-depths.min())/(depths.max()-depths.min()+1e-8) # normalize to 0~1
        depths = grayscale_to_rgb(depths)
        save_ls.append(depths)
    if 'depths' in batch.keys():
        ncol = 2
        target = (batch['depths'] ).reshape(1, *hw).cpu() # h, w
        target = grayscale_to_rgb(target)
        save_ls.append(target)

    save_image(save_ls, save_path, nrow=2, ncol=ncol)
    del result
    torch.cuda.empty_cache()
    return avg_psnr



if __name__=="__main__":
    hparams = get_hparams()
    hparams.img_path = f"{hparams.ckpt_path}/epoch={hparams.test_epoch:0>2}_testing"
    print(hparams)
    os.makedirs(hparams.img_path, exist_ok=True)
    
    test_dataset = eval(hparams.data_class)(hparams, "test")
    test_dataloader = DataLoader(
        test_dataset,
        shuffle=False,
        num_workers=hparams.N_workers,
        batch_size=1,
        pin_memory=True
    )
    if hparams.Sampling in ["Vanilla", "NGP"]:
        model = eval(f'{hparams.Sampling}Sampling')(hparams).to(device)
    elif hparams.Sampling == "NGPPl":
        dataset_hparams = test_dataset.K.to(device), test_dataset.poses.to(device), (test_dataset.h, test_dataset.w) 
        model = NGPPlSampling(hparams, dataset_hparams).to(device)
        model.mark_invisible_cells()
    else:
        raise
    ckpt_path = join(hparams.ckpt_path, f"epoch={hparams.test_epoch:0>2}.pth")
    model.load_state_dict(torch.load(ckpt_path)['model'])
    print(f"Successfully loading from {ckpt_path}\n")
    
    print(f" avg_psnr={test(hparams, model, test_dataloader):>0.4f}")
    