import os
from os import path

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchvision import transforms

import numpy as np
import cv2
import math
from scipy.optimize import linear_sum_assignment


from dinov2.models import vision_transformer as vits
import dinov2.utils.utils as dinov2_utils
from dinov2.data.transforms import MaybeToTensor, make_normalize_transform

from dinov2.dino_matcher import Dino_Matcher

import torch


def pad_bottom_right(inp, pad_size, ret_mask=False):
    assert isinstance(pad_size, int) and pad_size >= max(inp.shape[-2:]), f"{pad_size} < {max(inp.shape[-2:])}"
    mask = None
    if inp.ndim == 2:
        padded = np.zeros((pad_size, pad_size), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1]] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    elif inp.ndim == 3:
        padded = np.zeros((pad_size, pad_size, inp.shape[2]), dtype=inp.dtype)
        padded[:inp.shape[0], :inp.shape[1], :] = inp
        if ret_mask:
            mask = np.zeros((pad_size, pad_size), dtype=bool)
            mask[:inp.shape[0], :inp.shape[1]] = True
    else:
        raise NotImplementedError()
    return padded, mask


def resize_im_(wo, ho, imsize=None, dfactor=1, value_to_scale=max):
    wt, ht = wo, ho
    if imsize is not None and imsize > 0:
        scale = imsize / value_to_scale(wo, ho)
        ht, wt = int(round(ho * scale)), int(round(wo * scale))

    # Make sure new sizes are divisible by the given factor
    wt, ht = map(lambda x: int(x // dfactor * dfactor), [wt, ht])
    scale = (wo / wt, ho / ht)
    return wt, ht, scale


from torchvision.transforms import functional


def load_im_padding(im_path0, im_path1, imsize=1024, max_img_size=1344, device="cuda", dfactor=32):
    im0 = cv2.imread(im_path0)[:, :, ::-1]
    origin_im0 = im0.copy()
    ho, wo, _ = im0.shape
    if imsize > 0:
        posible_max = imsize / min(ho, wo) * max(ho, wo)
    else:
        posible_max = -1
    if posible_max > max_img_size:
        wt0, ht0, scale0 = resize_im_(wo, ho, imsize=max_img_size, dfactor=dfactor, value_to_scale=max)
    else:
        wt0, ht0, scale0 = resize_im_(wo, ho, imsize=imsize, dfactor=dfactor, value_to_scale=min)
    im0 = cv2.resize(im0, (wt0, ht0))

    im1 = cv2.imread(im_path1)[:, :, ::-1]
    origin_im1 = im1.copy()
    ho, wo, _ = im1.shape
    wt1, ht1, scale1 = resize_im_(wo, ho, imsize=imsize, dfactor=dfactor, value_to_scale=min)
    im1 = cv2.resize(im1, (wt1, ht1))

    if im0.shape != im1.shape:
        pad_to = max(ht0, wt0)
        im0, mask0 = pad_bottom_right(im0, pad_to, ret_mask=True)
        wt1, ht1, scale1_ = resize_im_(wt1, ht1, imsize=pad_to, dfactor=dfactor, value_to_scale=max)
        scale1 = (scale1[0] * scale1_[0], scale1[1] * scale1_[1])
        im1 = cv2.resize(im1, (wt1, ht1))
        pad_to = max(ht1, wt1)
        im1, mask1 = pad_bottom_right(im1, pad_to, ret_mask=True)
        mask0 = torch.from_numpy(mask0).unsqueeze(0)
        mask1 = torch.from_numpy(mask1).unsqueeze(0)
        if device:
            mask0, mask1 = mask0.to(device), mask1.to(device)
    else:
        mask0, mask1 = None, None
    # im0 = F.to_tensor(im0).unsqueeze(0).to(device)
    # im1 = F.to_tensor(im1).unsqueeze(0).to(device)

    im0 = functional.to_tensor(im0).unsqueeze(0)
    im1 = functional.to_tensor(im1).unsqueeze(0)
    if device:
        im0, im1 = im0.to(device), im1.to(device)

    # print(im0.shape, im1.shape)

    return origin_im0, origin_im1, im0, im1, mask0, mask1, scale0, scale1




class Dino_Eval:
    def __init__(self, *args, args_in=None, gpuid=0, **kwargs):

        self.imsize = 518
        self.imsize = 518
        self.max_imsize = self.imsize
        self.patch = 14
        self.no_match_upscale = True

        self.model = Dino_Matcher(img_size=self.imsize, device='cuda')
        # self.model.eval()
        # self.model.to('cuda')
        # self.model_config = model_config

        # Name the method
        self.name = f'Dino_v2_'
        # if self.no_match_upscale:
        #     self.name += '_noms'
        # print(f'Initialize {self.name}')

    def match_inputs_(self, gray1, gray2, mask1=None, mask2=None):

        batch = {'image0': gray1, 'image1': gray2}
        if mask1 is not None:
            batch['mask0_origin'] = mask1
        if mask2 is not None:
            batch['mask1_origin'] = mask2
        with torch.no_grad():
            self.model.forward(batch)

        kpts1 = batch["mkpts0_c"].cpu().numpy()
        kpts2 = batch["mkpts1_c"].cpu().numpy()
        scores = batch["mconf"].cpu().numpy()
        del batch

        matches = np.concatenate([kpts1, kpts2], axis=1)
        return matches, kpts1, kpts2, scores

    def match_pairs(self, im1_path, im2_path, ):

        ori_rgb1, ori_rgb2, rgb1, rgb2, mask1, mask2, sc1, sc2 = load_im_padding(im1_path, im2_path, self.imsize,
                                                                                 dfactor=self.patch,
                                                                                 max_img_size=self.max_imsize)

        upscale = np.array([sc1 + sc2])
        matches, kpts1, kpts2, scores = self.match_inputs_(rgb1, rgb2, mask1=mask1, mask2=mask2)

        return matches, kpts1, kpts2, scores, upscale.squeeze(0)


if __name__ == '__main__':
    obj = Dino_Eval()
    imgs = ['1.ppm', '2.ppm']
    imgs = [os.path.join('/data/hqx/myprojects/CasMTR/data/datasets/hpatches-sequences-release/v_yard', x) for x in
            imgs]
    matches, kpts1, kpts2, scores, upscale = obj.match_pairs(*imgs)
    print('len match:', len(matches))
