import os.path
from argparse import Namespace
import torch
import numpy as np
import cv2

from model.loftr_src.loftr.utils.cvpr_ds_config import default_cfg
from model.full_model import GeoFormer as GeoFormer_
from .base import Matching
from eval_tool.immatch.utils.data_io import load_gray_scale_tensor_cv
from model.geo_config import default_cfg as geoformer_cfg

class GeoFormer(Matching):
    def __init__(self, args, gpuid=0):
        super().__init__(gpuid)
        if type(args) == dict:
            args = Namespace(**args)

        self.imsize = args.imsize
        self.match_threshold = args.match_threshold
        self.no_match_upscale = args.no_match_upscale

        # Load model
        conf = dict(default_cfg)
        conf['match_coarse']['thr'] = self.match_threshold
        geoformer_cfg['coarse_thr'] = self.match_threshold
        self.model = GeoFormer_(conf)
        ckpt_dict = torch.load(args.ckpt, map_location=torch.device('cpu'))
        if 'state_dict' in ckpt_dict:
            ckpt_dict = ckpt_dict['state_dict']
        self.model.load_state_dict(ckpt_dict, strict=False)
        self.model = self.model.eval().to(self.device)

        # Name the method
        self.ckpt_name = args.ckpt.split('/')[-1].split('.')[0]
        self.name = f'GeoFormer_{self.ckpt_name}'
        if self.no_match_upscale:
            self.name += '_noms'
        print(f'Initialize {self.name}')

    def change_deivce(self, device):
        self.device = device
        self.model.to(device)
    def load_im(self, im_path, enhanced=False):
        return load_gray_scale_tensor_cv(
            im_path, self.device, imsize=self.imsize, dfactor=8, enhanced=enhanced, value_to_scale=min
        )

    def cal_match_dist(self, H_pred):
        from PIL import Image
        im = Image.open(self.im1_path)
        w1, h1 = im.size
        im = Image.open(self.im2_path)
        w2, h2 = im.size
        points_gd = np.loadtxt(self.gd)
        raw = np.zeros([len(points_gd), 2])
        dst = np.zeros([len(points_gd), 2])
        raw[:, 0] = points_gd[:, 0] * w1
        raw[:, 1] = points_gd[:, 1] * h1
        dst[:, 0] = points_gd[:, 2] * w2
        dst[:, 1] = points_gd[:, 3] * h2
        # if scale_H:
        #     # scale = (wo / wt, ho / ht) for im1 & im2
        #     scale = match_res[4]
        #
        #     # Scale gt homoragphies
        #     H_scale_im1 = scale_homography(scale[1], scale[0])
        #     H_scale_im2 = scale_homography(scale[3], scale[2])
        #     H_pred = np.linalg.inv(H_scale_im2) @ H_pred @ H_scale_im1
        dst_pred = cv2.perspectiveTransform(raw.reshape(-1, 1, 2), H_pred).squeeze()
        dis = (dst - dst_pred) ** 2
        dis = np.sqrt(dis[:, 0] + dis[:, 1])
        avg_dist = dis.mean()
        return {'mean': dis.mean(), 'points_gd': points_gd.tolist(), 'dis': dis.tolist()}
        # irat = np.mean(inliers)
        # mae = dis.max()
        # mee = np.median(dis)

    def match_inputs_(self, gray1, gray2):
        def cal_corner_dist(h, w, H_gt, H_pred):
            corners = np.array([[0, 0, 1],
                                [0, h - 1, 1],
                                [w - 1, 0, 1],
                                [w - 1, h - 1, 1]])
            real_warped_corners = np.dot(corners, np.transpose(H_gt))
            real_warped_corners = real_warped_corners[:, :2] / real_warped_corners[:, 2:]
            warped_corners = np.dot(corners, np.transpose(H_pred))
            warped_corners = warped_corners[:, :2] / warped_corners[:, 2:]
            corner_dist = np.mean(np.linalg.norm(real_warped_corners - warped_corners, axis=1))
            return corner_dist

        def cal_batch(batch):
            m0, m1 = batch['mkpts0_f'].cpu().numpy(), batch['mkpts1_f'].cpu().numpy()
            kwargs = dict(maxIters=8000, confidence=0.99995,)
            H_pred, inliers = cv2.findHomography(
                m0.astype('int32'), m1.astype('int32'), cv2.RANSAC, 3,
                **kwargs
            )

            scale = self.upscale
            # print(scale)
            # Scale gt homoragphies
            from eval_tool.immatch.utils.my_helper import scale_homography
            H_scale_im1 = scale_homography(1 / scale[0], 1 / scale[1])
            H_scale_im2 = scale_homography(1 / scale[2], 1 / scale[3])
            H_pred = np.linalg.inv(H_scale_im2) @ H_pred @ H_scale_im1

            basename = lambda p: os.path.basename(p).split('.')[0]
            batch['image_path1'] = self.im1_path
            batch['image_path2'] = self.im2_path
            batch['name'] = '{}_{}'.format(basename(self.im1_path), basename(self.im2_path))

            # H_gt = batch['H_0to1'].to(torch.float32).squeeze(0).cpu().numpy()
            # dist = cal_corner_dist(*batch['hw0_i'].tolist(), H_gt, H_pred)
            dist = self.cal_match_dist(H_pred)
            print('dist={}'.format(dist))
            ret = {'corner_dist': dist['mean']}
            ret.update(dist)
            return ret


        batch = {'image0': gray1, 'image1': gray2}
        with torch.no_grad():
            batch = self.model(batch)
        kpts1 = batch['mkpts0_f'].cpu().numpy()
        kpts2 = batch['mkpts1_f'].cpu().numpy()

        # import matplotlib.pyplot as plt
        # import cv2
        # import numpy as np
        # plt.figure(dpi=200)
        # kp0 = kpts1
        # kp1 = kpts2
        # # if len(kp0) > 0:
        # kp0 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp0]
        # kp1 = [cv2.KeyPoint(int(k[0]), int(k[1]), 30) for k in kp1]
        # matches = [cv2.DMatch(_trainIdx=i, _queryIdx=i, _distance=1, _imgIdx=-1) for i in
        #            range(len(kp0))]
        #
        # show = cv2.drawMatches((gray1.cpu()[0][0].numpy() * 255).astype(np.uint8), kp0,
        #                        (gray2.cpu()[0][0].numpy() * 255).astype(np.uint8), kp1, matches,
        #                        None)
        # plt.imshow(show)
        # plt.show()
        scores = batch['mconf'].cpu().numpy()
        matches = np.concatenate([kpts1, kpts2], axis=1)

        # save to disk
        # from create_hpatche_cache import cache_to_disk0
        # @cache_to_disk0
        # def cache_func_handle(batch=None, model=None, version=None):
        #     pass
        #     return batch

        # ret_ = cal_batch(batch)
        # batch.update({'ret_dict': ret_})
        # cache_func_handle(batch=batch, version='ours2', cache_folder='./outputs/isc_cache')
        return matches, kpts1, kpts2, scores

    def match_pairs(self, im1_path, im2_path, cpu=False, gd=None):
        self.__dict__.update({"im1_path": im1_path, "im2_path": im2_path, "gd": gd})
        torch.cuda.empty_cache()
        tmp_device = self.device
        if cpu:
            self.change_deivce('cpu')
        gray1, sc1 = self.load_im(im1_path)
        gray2, sc2 = self.load_im(im2_path)

        upscale = np.array([sc1 + sc2])
        self.upscale = upscale.squeeze(0)
        matches, kpts1, kpts2, scores = self.match_inputs_(gray1, gray2)

        if self.no_match_upscale:
            return matches, kpts1, kpts2, scores, upscale.squeeze(0)

        # Upscale matches &  kpts
        matches = upscale * matches
        kpts1 = sc1 * kpts1
        kpts2 = sc2 * kpts2

        if cpu:
            self.change_deivce(tmp_device)

        return matches, kpts1, kpts2, scores


    # def match_pairs(self, im1_path, im2_path, cpu=False):
    #     tmp_device = self.device
    #     if cpu:
    #         self.change_deivce('cpu')
    #     gray1, sc1 = self.load_im(im1_path)
    #     gray2, sc2 = self.load_im(im2_path)
    #
    #
    #     torch.cuda.empty_cache()
    #     upscale = np.array([sc1 + sc2])
    #     data = {'image0': gray1, 'image1': gray2}
    #
    #     with torch.no_grad():
    #         self.model(data)
    #
    #         fine_kps_list1, fine_kps_list2, fine_scores_list = data['fine_kps_list0'], data['fine_kps_list1'], data['fine_scores_list']
    #
    #     kpts1, kpts2, scores = fine_kps_list1[0].cpu().numpy(), fine_kps_list2[0].cpu().numpy(), fine_scores_list[0].cpu().numpy()
    #     matches = np.concatenate([kpts1, kpts2], axis=1)
    #
    #     if self.no_match_upscale:
    #         if 'first_match_num' in data:
    #             return matches, kpts1, kpts2, scores, upscale.squeeze(0), data['first_match_num'], data['first_ransac_num']
    #         return matches, kpts1, kpts2, scores, upscale.squeeze(0)
    #
    #     # Upscale matches &  kpts
    #     matches = upscale * matches
    #     kpts1 = sc1 * kpts1
    #     kpts2 = sc2 * kpts2
    #     if cpu:
    #         self.change_deivce(tmp_device)
    #     return matches, kpts1, kpts2, scores

