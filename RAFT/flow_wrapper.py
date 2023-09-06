import sys


import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
import time


from RAFT.core.raft import RAFT
from RAFT.core.utils import flow_viz
from RAFT.core.utils.utils import InputPadder

DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = cv2.resize(img, (int(img.shape[1]), int(img.shape[0])))
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2, 1, 0]] / 255.0)
    cv2.waitKey()


def vis_correspondences(img1, img2, flo):
    img1 = img1[0].permute(1, 2, 0).cpu().numpy()[:, :, [2, 1, 0]].astype('uint8')
    img2 = img2[0].permute(1, 2, 0).cpu().numpy()[:, :, [2, 1, 0]].astype('uint8')

    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    height = img1.shape[0]
    width = img1.shape[1]
    kp1 = []
    kp2 = []

    # gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    # halisi = cv2.cornerHarris(src=gray, blockSize=2, ksize=3, k=0.03)
    # halisi_point = halisi>0.01*halisi.max()
    for h in range(10, height, 100):
        for w in range(10, width, 100):
            # w = 500
            # h = 200
            kp1.append(cv2.KeyPoint(x=w, y=h, size=3))
            flow = flo[h, w]
            kp2.append(cv2.KeyPoint(x=w + flow[0], y=h + flow[1], size=3))

    matches = []
    for i in range(len(kp1)):
        dmatch = cv2.DMatch()
        dmatch.distance = 1.0
        dmatch.imgIdx = 1
        dmatch.queryIdx = i
        dmatch.trainIdx = i
        matches.append(dmatch)

    # img_out = []
    img_match = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
    cv2.imshow('img', img_match)
    cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    flows = []
    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))

        images = sorted(images)
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile2)
            image2 = load_image(imfile1)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            start = time.time()
            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            flows.append(flow_up)
            print('Time cost: ', time.time() - start)
            # viz(image1, flow_up)
            # vis_correspondences(image1, image2, flow_up)
    flows = torch.cat(flows, dim=0)
    torch.save(flows, '/media/kyrie/000A8561000BB32A/Flow/Scannet/flows_079.pth')


def generate_model():
    dic = {
        'model': './RAFT/models/raft-kitti.pth',
        'path': '/media/kyrie/000A8561000BB32A/Flow/Scannet/img',
        'small': False,
        'mixed_precision': False,
        'alternate_corr': False,
    }
    args = argparse.Namespace(**dic)
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()
    return model

def get_flow(model, img1, img2):
    with torch.no_grad():
        padder = InputPadder(img1.shape)
        image1, image2 = padder.pad(img1, img2)
        start = time.time()
        flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
        return flow_up


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')


    dic = {
        'model': './RAFT/models/raft-kitti.pth',
        'path': '/media/kyrie/000A8561000BB32A/Flow/Scannet/img',
        'small': False,
        'mixed_precision': False,
        'alternate_corr': False,
    }
    args1 = argparse.Namespace(**dic)

    # args2 = parser.parse_args()
    demo(args1)