# -*- coding: utf-8 -*-
import os
import random
import time
import kornia
# import imgaug.augmenters as iaa
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
import sys


from warnings import filterwarnings

# from src.utils.preprocess_utils import resize_aspect_ratio, get_perspective_mat, scale_homography
# from src.utils.dataset import pad_bottom_right
filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')

import glob
# from src.utils.misc import lower_config, load_im_padding

import pydegensac
import cv2


class HpatchesDataset(Dataset):
    def __init__(self,
        data_root = './data/datasets/hpatches-sequences-release',
        rank=0,
        world_size=None,
        device='cpu',
        **kwargs):
        self.data_root = data_root
        self.device = device
        self.max_img_size = 1200
        self.imsize = 480

        seq_dirs = sorted(glob.glob('{}/*'.format(data_root)))
        self.data = [dict(
            name=os.path.basename(p) + "/1_{}".format(im_idx),
            img1=os.path.join(p, '1.ppm'),
            img2=os.path.join(p, '{}.ppm'.format(im_idx)),
            H_gt=os.path.join(p, 'H_1_{}'.format(im_idx)),  )
            for p in seq_dirs for im_idx in range(2, 7)]

        for i, _ in enumerate(self.data):
            _['index'] = i

        self.data = self.data[::-1]
        if rank == 0:
            print('Init Hpatches datasets finished: length={}'.format(len(self.data)))

        world_size = None
        if world_size is not None:
            # bz = int(len(self.data) // world_size)

            bz = int((len(self.data) + world_size - 1) // world_size)

            start = rank * bz

            if world_size * bz > len(self.data):
                self.data += [None for _ in range(world_size * bz - len(self.data))]

            end = start + bz

            assert len(self.data) == bz * world_size, \
                'len(self.data) != bz * world_size, len(self.data)={}, bz={}, world_size={}'.format(len(self.data), bz, world_size)

            self.data = self.data[start:end]

        self.togray = transforms.Grayscale()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        if self.data[index] is None:
            return {}

        item = self.data[index]

        device = self.device
        from eval_tool.immatch.utils.data_io import load_gray_scale_tensor_cv
        gray1, sc1 = load_gray_scale_tensor_cv(
            item['img1'], device, imsize=self.imsize, dfactor=8, value_to_scale=min
        )
        gray2, sc2 = load_gray_scale_tensor_cv(
            item['img2'], device, imsize=self.imsize, dfactor=8, value_to_scale=min
        )

        name = item['name']
        H_gt0 = np.loadtxt(item['H_gt'])


        scale = sc1 + sc2
        H_scale_im1 = scale_homography(scale[0], scale[1])
        H_scale_im2 = scale_homography(scale[2], scale[3])
        H_gt = np.linalg.inv(H_scale_im2) @ H_gt0 @ H_scale_im1

        H_gt = torch.from_numpy(H_gt).float().reshape(1, 3, 3)

        H_gt0 = torch.from_numpy(H_gt0).float().reshape(1, 3, 3)

        data = {
            'name': name,
            'index': item['index'],
            'dataset_name': 'Oxford',
            "image0": gray1,
            "image1": gray2,
            'sc1': sc1,
            'sc2': sc2,
            'H_gt': H_gt0.reshape(3, 3),
            'pair_id': item['index'],
            'pair_names': name,
            'H_0to1': H_gt,  # (1, 3, 3)
            'H_1to0': torch.inverse(H_gt)[0],
        }
        return data

def scale_homography(sw, sh):
    return np.array([[sw,  0, 0],
                     [ 0, sh, 0],
                     [ 0,  0, 1]])

class Hpatches_Eval:
    # def __init__(self, matcher, loftr_cfg, plmodel: PLMacher = None):
    def __init__(self, matcher, loftr_cfg, plmodel: None):
        self.plmodel = plmodel
        self.model = getattr(self.plmodel, 'matcher', None)
        self.loftr_cfg = loftr_cfg
        # self.model = matcher

        self.eval_step_results = []
        self.no_match_upscale = True

    def match_inputs_(self, gray1, gray2, mask1=None, mask2=None):

        batch = {'image0': gray1, 'image1': gray2}
        with torch.no_grad():
            batch = self.model(batch)
        kpts1 = batch['mkpts0_f'].cpu().numpy()
        kpts2 = batch['mkpts1_f'].cpu().numpy()

        scores = batch['mconf'].cpu().numpy()
        matches = np.concatenate([kpts1, kpts2], axis=1)
        # print(len(scores))
        return matches, kpts1, kpts2, scores

    def match_pairs(self, image0, image1, sc1, sc2, mask1=None, mask2=None, ori_rgb1=None, ori_rgb2=None, **kwargs):
        torch.cuda.empty_cache()

        upscale = np.array([sc1 + sc2])
        matches, kpts1, kpts2, scores = self.match_inputs_(image0, image1, mask1=mask1, mask2=mask2)
        # matches, kpts1, kpts2, scores = self.match_inputs_(gray1, gray2)

        if self.no_match_upscale:
            return matches, kpts1, kpts2, scores, upscale.squeeze(0)

        # Upscale matches &  kpts
        matches = upscale * matches
        kpts1 = sc1 * kpts1
        kpts2 = sc2 * kpts2

        return matches, kpts1, kpts2, scores

    def cal_one(self, name, H_gt, isprint=False, **kwargs):
        task = 'homography'
        scale_H = self.no_match_upscale  # True when hpatches
        h_solver = 'cv'
        ransac_thres = 3
        match_res = self.match_pairs(H_gt=H_gt, **kwargs)
        sc = np.array(match_res[4])  # scale

        scale = np.ones(4)

        image0 = kwargs['image0']
        im1_size_wh = np.array([image0.shape[-1], image0.shape[-2]]) * sc[:2]

        # Predict matches
        try:
            t0 = time.time()

            matches, p1s, p2s = match_res[0:3]
            H_gt_raw = H_gt
            if scale_H:
                # scale = (wo / wt, ho / ht) for im1 & im2
                scale = match_res[4]

                # Scale gt homoragphies
                H_scale_im1 = scale_homography(scale[0], scale[1])
                H_scale_im2 = scale_homography(scale[2], scale[3])
                H_gt = np.linalg.inv(H_scale_im2) @ H_gt @ H_scale_im1
        except Exception as e:
            # print(e)
            import traceback
            traceback.print_exc()
            p1s = p2s = matches = []
            # match_failed += 1
        # n_matches.append(len(matches))


        if 'homography' in task:
            try:
                if 'cv' in h_solver:
                    H_pred, inliers = cv2.findHomography(
                        matches[:, :2], matches[:, 2:4], cv2.RANSAC, ransac_thres,
                        maxIters=8000, confidence=0.99995,
                    )
                else:
                    H_pred, inliers = pydegensac.findHomography(matches[:, :2], matches[:, 2:4], ransac_thres)
            except:
                H_pred = None
                if isprint:
                    import traceback
                    traceback.print_exc()

            if H_pred is None:
                corner_dist = np.nan
                irat = 0
                # h_failed += 1
                inliers = []
            else:
                w, h = im1_size_wh
                w, h = w / scale[0], h / scale[1]
                corners = np.array([[0, 0, 1],
                                    [0, h - 1, 1],
                                    [w - 1, 0, 1],
                                    [w - 1, h - 1, 1]])
                real_warped_corners = np.dot(corners, np.transpose(H_gt))
                real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
                warped_corners = np.dot(corners, np.transpose(H_pred))
                warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
                corner_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
                irat = np.mean(inliers)

            if isprint:
                print(f'Scene {name}, pair:1-{""} matches:{len(matches)}')
                if 'homography' in task:
                    print(f'Corner dist:{corner_dist:.2f} inliers:{np.sum(inliers)}')

        Image_Class_Map = {'i': 1, 'v': 2}
        ret = dict(corner_dist=corner_dist, irat=irat, cid=Image_Class_Map.get(name[0], 0))
        return ret

    def cal_scores(self, eval_step_results=None):
        eval_step_results = eval_step_results if eval_step_results is not None else self.eval_step_results
        if isinstance(eval_step_results, list):
            dists_sa, dists_si, dists_sv = [
                [_['corner_dist'] for _ in eval_step_results if query in {0, _['cid']}]
                for query in [0, 1, 2]
            ]
        elif isinstance(eval_step_results, dict):
            corner_dist = eval_step_results['corner_dist']
            cid = eval_step_results['cid']
            tonumpy_func = lambda tensor: tensor.cpu().numpy() if torch.is_tensor(tensor) else np.array(tensor)
            corner_dist, cid = list(map(tonumpy_func, [corner_dist, cid]))
            dists_sa, dists_si, dists_sv = [
                corner_dist, corner_dist[cid == 1], corner_dist[cid == 2]
            ]
        else:
            assert 0, 'input error'

        from eval_tool.immatch.utils.hpatches_helper import eval_summary_homography
        _, auc = eval_summary_homography(dists_sa, dists_si, dists_sv, thres=[1, 3, 5, 10], lprint=lambda *args: None)
        self.eval_step_results = []
        return auc


import os
import torch
from pytorch_lightning.plugins.precision.deepspeed import DeepSpeedPrecisionPlugin
import math
import sys
import argparse
import pprint
from distutils.util import strtobool
from pathlib import Path
from loguru import logger as loguru_logger
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.strategies import DDPStrategy, DeepSpeedStrategy

from model.loftr_src.lightning.data import HomoDataModule
from model.loftr_src.config.default import get_cfg_defaults
from model.loftr_src.utils.misc import get_rank_zero_only_logger, setup_gpus
from model.loftr_src.utils.profiler import build_profiler
from lightning.lightning_homo_srmatcher import PL_H_SRMatcher

loguru_logger = get_rank_zero_only_logger(loguru_logger)


import torch
torch.set_float32_matmul_precision('high')
def load_model(ckpt_path, **kwargs):
    from lightning.train_homo_srmatcher import parse_args
    args = parse_args()
    args.num_nodes = 1
    args.gpus = CUDA_VISIBLE_DEVICES
    args.gpus = 1 if ',' not in args.gpus else args.gpus
    rank_zero_only(pprint.pprint)(vars(args))

    # init default-cfg and merge it with the main- and data-cfg
    default_config = get_cfg_defaults()
    # model_config = get_cfg_model()
    default_config.merge_from_file(args.loftr_cfg_path)
    default_config.merge_from_file(args.data_cfg_path)
    # pl.seed_everything(default_config.TRAINER.SEED)  # reproducibility
    # TODO: Use different seeds for each dataloader workers
    # This is needed for data augmentation

    # scale lr and warmup-step automatically
    args.gpus = _n_gpus = setup_gpus(args.gpus)
    default_config.TRAINER.WORLD_SIZE = _n_gpus * args.num_nodes
    default_config.TRAINER.TRUE_BATCH_SIZE = default_config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = default_config.TRAINER.TRUE_BATCH_SIZE / default_config.TRAINER.CANONICAL_BS
    default_config.TRAINER.SCALING = _scaling
    default_config.TRAINER.TRUE_LR = default_config.TRAINER.CANONICAL_LR * _scaling * 1  # set 0.5 to finetune
    default_config.TRAINER.WARMUP_STEP = math.floor(default_config.TRAINER.WARMUP_STEP / _scaling)
    # lightning module
    profiler = build_profiler(args.profiler_name)

    # TensorBoard Logger
    logger = TensorBoardLogger(
        save_dir='logs/tb_logs',
        name=args.exp_name,
        version='hpatches',
        default_hp_metric=False)
    ckpt_dir = Path(logger.log_dir) / 'checkpoints'
    # print(ckpt_dir)

    cktp_dir = str(Path(logger.log_dir) / 'checkpoints')

    os.makedirs(logger.log_dir, exist_ok=True)
    if ckpt_path is None:
        ckpt_path = args.ckpt_path
    model = PL_H_SRMatcher(default_config, profiler=profiler, pretrained_ckpt=ckpt_path, log_dir=logger.log_dir)

    return model

# def load_matcher(ckpt, config_path='configs/model_configs/outdoor/loftr_ds_quadtree_cas_twins_large_stage3.py', stage=1):
#     from configs.default import get_cfg_defaults
#     config = get_cfg_defaults()
#     config.merge_from_file(config_path)
#     # config.LOFTR.TRAINING_STAGE = getattr(args, 'training_stage', config.LOFTR.TRAINING_STAGE)
#     config.LOFTR.TRAINING_STAGE = stage
#
#     config = lower_config(config)
#     model_config = config["loftr"]
#
#     from src.model.cascade_model_stage3_geo import CasMTRG
#     matcher = CasMTRG(config=model_config)
#     state_dict = torch.load(ckpt, map_location="cpu")["state_dict"]
#     matcher.load_state_dict(state_dict, strict=True)
#
#     matcher.eval()
#     matcher.to("cuda")
#     return matcher, model_config
#
# from tqdm import tqdm
#
if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    model = load_model()
    model = model.to('cuda')


    obj = HpatchesDataset()
    _ = obj[0]
    # ckpt = 'logs/tb_logs/CasMTR-4c-stage2-oxford-size640-epochs8to25/geo_stage2/check_points/epoch=24-step=60118.ckpt'
    ckpt = 'pretrained_weights/trained_ckpt/stage2-4c-geo-epoch25.ckpt'
    # Hest AUC: a=[0.18673951 0.51113241 0.66311925 0.80178877]

    # batch = obj[0]
    #
    #
    # batch['dataset_name'] = (batch['dataset_name'], )
    # for key in ['image0', 'image1', 'mask1', 'mask2', 'H_0to1', 'H_1to0', 'H_gt']:
    #     if key not in batch: continue
    #     batch[key] = batch[key].unsqueeze(0).to('cuda')
    # model.validation_step(batch, 0, dataloader_idx=0)

    # model_config['coarse2']['post_config']['method'] = "maxpool_nms"
    # model_config['coarse2']['post_config']['window_size'] = 5
    # # model_config['training_stage'] = 2
    # hpatches = Hpatches_Eval(matcher, model_config,)
    # res = []
    # # for idx, batch in enumerate(tqdm(obj)):
    # for idx in range(2):
    #     batch = obj[idx]
    #     batch['rgb1'] = batch['rgb1'].to('cuda')
    #     batch['rgb2'] = batch['rgb2'].to('cuda')
    #     for key in batch:
    #         if torch.is_tensor(batch[key]):
    #             batch[key] = batch[key].to('cuda')
    #     res.append(hpatches.cal_one(isprint=True, **batch))
    # #
    # auc = hpatches.cal_scores(res)

    # print(auc)
