
import copy
import torch
import torch.nn as nn

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
# from scipy.optimize import linear_sum_assignment

from dinov2.models import vision_transformer as vits
import dinov2.utils.utils as dinov2_utils
from dinov2.data.transforms import MaybeToTensor, make_normalize_transform

import torch

INF = 1e9



def mask_border(m, b: int, v):
    """ Mask borders with value
    Args:
        m (torch.Tensor): [N, H0, W0, H1, W1]
        b (int)
        v (m.dtype)
    """
    if b <= 0:
        return

    m[:, :b] = v
    m[:, :, :b] = v
    m[:, :, :, :b] = v
    m[:, :, :, :, :b] = v
    m[:, -b:] = v
    m[:, :, -b:] = v
    m[:, :, :, -b:] = v
    m[:, :, :, :, -b:] = v


def mask_border_with_padding(m, bd, v, p_m0, p_m1):
    if bd <= 0:
        return

    m[:, :bd] = v
    m[:, :, :bd] = v
    m[:, :, :, :bd] = v
    m[:, :, :, :, :bd] = v

    h0s, w0s = p_m0.sum(1).max(-1)[0].int(), p_m0.sum(-1).max(-1)[0].int()
    h1s, w1s = p_m1.sum(1).max(-1)[0].int(), p_m1.sum(-1).max(-1)[0].int()
    for b_idx, (h0, w0, h1, w1) in enumerate(zip(h0s, w0s, h1s, w1s)):
        m[b_idx, h0 - bd:] = v
        m[b_idx, :, w0 - bd:] = v
        m[b_idx, :, :, h1 - bd:] = v
        m[b_idx, :, :, :, w1 - bd:] = v


import torch
import torch.nn.functional as F


def interpolate_match_confidence(m1, new_dims, device='cpu', mode='bilinear'):
    """
    Interpolates a match confidence tensor from original dimensions to new dimensions.

    Args:
        m1: Original match confidence tensor of shape (bs, h0, w0, h1, w1).
        orig_dims: Tuple of original dimensions (h0, w0, h1, w1).
        new_dims: Tuple of new dimensions (new_h0, new_w0, new_h1, new_w1).
        device: Device on which the computations are performed.

    Returns:
        Interpolated match confidence tensor of shape (bs, new_h0, new_w0, new_h1, new_w1).
    """
    # Move the tensor to the specified device
    m1 = m1.to(device)
    bs, h0, w0, h1, w1 = m1.shape
    new_h0, new_w0, new_h1, new_w1 = new_dims

    # Reshape and interpolate for the first pair of dimensions (h0, w0)
    m1_interpolated_h0w0 = F.interpolate(
        m1.view(bs, h0, w0, h1 * w1).permute(0, 3, 1, 2),  # Reshape to (bs, h1*w1, h0, w0)
        size=(new_h0, new_w0),  # New size for the first pair of dimensions
        mode=mode, # align_corners=True
    ).permute(0, 2, 3, 1).view(bs, new_h0, new_w0, h1, w1)  # Reshape back to original format

    # Reshape and interpolate for the second pair of dimensions (h1, w1)
    m1_interpolated_h1w1 = F.interpolate(
        m1_interpolated_h0w0.view(bs, new_h0 * new_w0, h1, w1),
        # Reshape to (bs, h1, w1, new_h0*new_w0)
        size=(new_h1, new_w1),  # New size for the second pair of dimensions
        mode=mode,  # align_corners=True
    ).view(bs, new_h0, new_w0, new_h1, new_w1)  # Reshape back to (bs, new_h0, new_w0, new_h1, new_w1)

    ret = m1_interpolated_h1w1.view(bs, new_h0 * new_w0, new_h1 * new_w1)
    ret = F.normalize(ret, p=1, dim=1) * F.normalize(ret, p=1, dim=2)
    return ret


# class Dino_Matcher:
class Dino_Matcher(nn.Module):
    def __init__(self, img_size=518, patch_size=14, device=None):
        super(Dino_Matcher, self).__init__()
        self.device = device
        self.patch_size = patch_size
        self.img_size = img_size

        dinov2_kwargs = dict(
            img_size=img_size,
            patch_size=patch_size,
            init_values=1e-5,
            ffn_layer='mlp',
            block_chunks=0,
            qkv_bias=True,
            proj_bias=True,
            ffn_bias=True,
        )

        self.encoder = torch.hub.load('./dinov2', 'dinov2_vits14_reg', source='local')
        # self.encoder = torch.hub.load('./dinov2', 'dinov2_vitb14_reg', source='local')
        # self.encoder = torch.hub.load('./dinov2', 'dinov2_vitl14_reg', source='local')
        # self.encoder = torch.hub.load('./dinov2', 'dinov2_vitg14_reg', source='local')

        self.temperature = 0.1
        self.border_rm = 0
        self.thr = 0.2
        self.embed_dim = 256
        self.linear_proj = nn.Linear(self.encoder.embed_dim, self.embed_dim)
        self.num_patch = self.encoder.patch_embed.num_patches
        self.patch_HW = (80, 80)
        self.num_patch = self.patch_HW[0] * self.patch_HW[1]

        # transforms for image encoder
        self.encoder_transform = transforms.Compose([
            MaybeToTensor(),
            make_normalize_transform(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        self.layer_feat = None


    def extract_img_feats1(self, ref_imgs, tar_img, device, data):
        self.ref_imgs = ref_imgs  # (1, 3, 518, 518)
        self.tar_img = tar_img  # (1, 3, 518, 518)
        b1, c1, h1, w1 = ref_imgs.shape
        b2, c2, h2, w2 = tar_img.shape
        h1, w1 = h1 // 14, w1 // 14
        h2, w2 = h2 // 14, w2 // 14

        ref_imgs, tar_img = ref_imgs.transpose(2, 3), tar_img.transpose(2, 3)

        LAYER_IDX = 9
        ref_feats = self.encoder.forward_features(ref_imgs.to(device), layer_feat_idx=[LAYER_IDX])["layer_feat"][LAYER_IDX][:, -w1 * h1:]
        tar_feat = self.encoder.forward_features(tar_img.to(device), layer_feat_idx=[LAYER_IDX])["layer_feat"][LAYER_IDX][:, -w2 * h2:]

        fea0ori = ref_feats.reshape(b1, w1, h1, -1).permute(0, 3, 2, 1)
        fea1ori = tar_feat.reshape(b2, w2, h2, -1).permute(0, 3, 2, 1)
        data.update({
            'fea0ori': fea0ori,
            'fea1ori': fea1ori,
        })
        ref_feats = self.linear_proj(ref_feats).reshape(b1, w1, h1, -1).permute(0, 2, 1, 3)
        tar_feat = self.linear_proj(tar_feat).reshape(b2, w2, h2, -1).permute(0, 2, 1, 3)

        ref_feats = ref_feats.reshape(-1, self.embed_dim)  # ns*N, c
        tar_feat = tar_feat.reshape(-1, self.embed_dim)  # N, c; (518//14 * 518 // 14 = 1369, 1024)

        ref_feats_ = ref_feats / ref_feats.shape[-1] ** .5
        tar_feat_ = tar_feat / tar_feat.shape[-1] ** .5

        S = ref_feats_ @ tar_feat_.t() / self.temperature # ns*N, N
        return ref_feats.unsqueeze(0), tar_feat.unsqueeze(0), S.unsqueeze(0)

    def extract_img_feats(self, ref_imgs, tar_img, device):

        ref_imgs, tar_img = ref_imgs.transpose(2, 3), tar_img.transpose(2, 3)
        ref_feats = self.encoder.forward_features(ref_imgs.to(device))["x_prenorm"][:, -self.num_patch:]
        tar_feat = self.encoder.forward_features(tar_img.to(device))["x_prenorm"][:, -self.num_patch:]

        ref_feats = self.linear_proj(ref_feats)
        tar_feat = self.linear_proj(tar_feat)


        ref_feats_ = ref_feats.reshape(-1, self.embed_dim)  # ns*N, c
        tar_feat_ = tar_feat.reshape(-1, self.embed_dim)  # N, c; (518//14 * 518 // 14 = 1369, 1024)

        ref_feats_ = ref_feats_ / ref_feats_.shape[-1] ** .5
        tar_feat_ = tar_feat_ / tar_feat_.shape[-1] ** .5


        S = ref_feats_ @ tar_feat_.t() / self.temperature  # ns*N, N
        return ref_feats_.unsqueeze(0), tar_feat_.unsqueeze(0), S.unsqueeze(0)

    def cal_conf_mat(self, data, level='8c'):
        # get feature
        feat_c0, feat_c1, conf_matrix = self.extract_img_feats1(data['image0'], data['image1'], device=data['image0'].device,
                                                      data=data)

        # match
        data.update({
            'fea0': feat_c0,
            'fea1': feat_c1,
            'conf': conf_matrix,
        })
        return {}

    @torch.no_grad()
    def get_coarse_match(self, conf_matrix, data, level, scale=14):
        """
        Args:
            conf_matrix (torch.Tensor): [N, L, S]
            data (dict): with keys ['hw0_i', 'hw1_i', 'hw0_c', 'hw1_c']
        Returns:
            coarse_matches (dict): {
                'b_ids' (torch.Tensor): [M'],
                'i_ids' (torch.Tensor): [M'],
                'j_ids' (torch.Tensor): [M'],
                'gt_mask' (torch.Tensor): [M'],
                'm_bids' (torch.Tensor): [M],
                'mkpts0_c' (torch.Tensor): [M, 2],
                'mkpts1_c' (torch.Tensor): [M, 2],
                'mconf' (torch.Tensor): [M]}
        """
        axes_lengths = {
            'h0c': data[f'hw0_{level}'][0],
            'w0c': data[f'hw0_{level}'][1],
            'h1c': data[f'hw1_{level}'][0],
            'w1c': data[f'hw1_{level}'][1]
        }

        mask = conf_matrix > self.thr
        mask = rearrange(mask, 'b (h0c w0c) (h1c w1c) -> b h0c w0c h1c w1c', **axes_lengths)
        if f'mask_{level}0' not in data:
            mask_border(mask, self.border_rm, False)
        else:
            mask_border_with_padding(mask, self.border_rm, False, data[f'mask_{level}0'], data[f'mask_{level}1'])
        mask = rearrange(mask, 'b h0c w0c h1c w1c -> b (h0c w0c) (h1c w1c)', **axes_lengths)

        # 2. mutual nearest 保证行列最大值是同一个idx [B, HW, HW]
        mask = mask * (conf_matrix == conf_matrix.max(dim=2, keepdim=True)[0]) * (
                conf_matrix == conf_matrix.max(dim=1, keepdim=True)[0])
        if mask.sum() == 0:
            mask[:, 0] = True  # 保底一个

        # 3. find all valid coarse matches
        # this only works when at most one `True` in each row
        mask_v, all_j_ids = mask.max(dim=2)  # [B, HW]
        b_ids, i_ids = torch.where(mask_v)
        j_ids = all_j_ids[b_ids, i_ids]  # j_ids: image0 target at which idx of image1
        mconf = conf_matrix[b_ids, i_ids, j_ids]

        # These matches select patches that feed into fine-level network
        coarse_matches = {'b_ids': b_ids, 'i_ids': i_ids, 'j_ids': j_ids}

        # 4. Update with matches in original image resolution
        scale0 = scale * data['scale0'][b_ids] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][b_ids] if 'scale1' in data else scale
        mkpts0_c = torch.stack(
            [i_ids % data[f'hw0_{level}'][1], torch.div(i_ids, data[f'hw0_{level}'][1], rounding_mode='trunc')],
            dim=1) * scale0
        mkpts1_c = torch.stack(
            [j_ids % data[f'hw1_{level}'][1], torch.div(j_ids, data[f'hw1_{level}'][1], rounding_mode='trunc')],
            dim=1) * scale1

        # These matches is the current prediction (for visualization)
        coarse_matches.update({
            'gt_mask': mconf == 0,
            'm_bids': b_ids[mconf != 0],  # mconf == 0 => gt matches
            'mkpts0_c': mkpts0_c[mconf != 0],
            'mkpts1_c': mkpts1_c[mconf != 0],
            'mconf': mconf[mconf != 0]
        })

        return coarse_matches

    def process_img(self, ref_imgs ,tar_img):
        def process_(img):
            b, c, h, w = img.shape
            if img.shape[1] == 1:
                img = torch.repeat_interleave(img, 3, dim=1)
            resize = transforms.Resize((h // 8 * 14, w // 8 * 14), interpolation=transforms.InterpolationMode.BICUBIC, antialias=True)
            return torch.cat([self.encoder_transform(resize(im_))[None, ...] for im_ in img], dim=0)

        return process_(ref_imgs), process_(tar_img)

    def forward(self, data):
        data['image0'], data['image1'] = self.process_img(data['image0'], data['image1'])

        patch_size = self.patch_size
        data.update({
            'bs': data['image0'].size(0),
            'hw0_i': data['image0'].shape[2:], 'hw1_i': data['image1'].shape[2:],
            'hw0_8c': torch.Size(torch.tensor(data['image0'].shape[2:]) // patch_size),
            'hw1_8c': torch.Size(torch.tensor(data['image1'].shape[2:]) // patch_size),
        })

        res = self.cal_conf_mat(data, )
        sim_matrix = data['conf']
        data['mask_sim'] = sim_matrix
        data['mask_s8'] = F.softmax(sim_matrix, 1) * F.softmax(sim_matrix, 2)
        data.update(res)
