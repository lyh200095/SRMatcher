import torch
import torch.nn as nn
from einops import rearrange

from model.geo_transformer.transformer import GeoTransformer, GeoTransformer_inplace, GeoTransformer_Semantic
import cv2

from model.loftr_src.loftr.utils.position_encoding import PositionEncodingSine
from utils.common_utils import get_map_keypoints, generate_window
from utils.homography import warp_points_batch


class GeoModule(nn.Module):
    def __init__(self, config, d_model):
        super().__init__()
        self.d_model = d_model
        self.window_size = config['window_size']
        self.pos_encoding = PositionEncodingSine(d_model)
        self.des_transformer = GeoTransformer_Semantic(config, config['layer_names'], d_model, linear=False)

    def apply_RANSAC(self, desc_map0, desc_map1, kps0, kps1, data):

        bs, cc, hh0, ww0 = desc_map0.shape
        bs, cc, hh1, ww1 = desc_map1.shape

        desc_map0 = rearrange(self.pos_encoding(desc_map0), 'n c h w -> n (h w) c')
        desc_map1 = rearrange(self.pos_encoding(desc_map1), 'n c h w -> n (h w) c')

        H0, W0 = data['image0'].shape[2:]
        H1, W1 = data['image1'].shape[2:]

        kp_cross_list1 = []
        kp_cross_list0 = []
        mask_cross_list1 = []
        mask_cross_list0 = []
        scale = int(data['hw0_i'][0] // data['hw0_c'][0])
        scale0 = scale * data['scale0'][:, None] if 'scale0' in data else scale
        scale1 = scale * data['scale1'][:, None] if 'scale0' in data else scale

        kps0 = [(kp / scale0[b] * scale).long() if torch.is_tensor(scale0) else kp.long() for b, kp in enumerate(kps0)]
        kps1 = [(kp / scale1[b] * scale).long() if torch.is_tensor(scale1) else kp.long() for b, kp in enumerate(kps1)]

        for s, (kp0, kp1) in enumerate(zip(kps0, kps1)):
            M = None
            if len(kp0) > 8:
                M, mask = cv2.findHomography(kp0.cpu().numpy(), kp1.cpu().numpy(), cv2.RANSAC, 4.0, maxIters=8000, confidence=0.9995,)

            if M is not None:
                kps0[s] = kp0[mask[:, 0] == 1].long()
                kps1[s] = kp1[mask[:, 0] == 1].long()

                keypoints = get_map_keypoints(H0, W0, scale).to(desc_map0.device)

                keypoints1 = \
                    warp_points_batch(keypoints.unsqueeze(0),
                                      homographies=torch.from_numpy(M).unsqueeze(0).to(desc_map0))[0]
                kp_windows_list1, masks_list1 = generate_window([keypoints1], (H1, W1),
                                                                window_size=self.window_size, scale=scale1)


                keypoints = get_map_keypoints(H1, W1, scale).to(desc_map0.device)

                keypoints0 = \
                    warp_points_batch(keypoints.unsqueeze(0),
                                      homographies=torch.inverse(torch.from_numpy(M).unsqueeze(0)).to(desc_map0))[0]

                kp_windows_list0, masks_list0 = generate_window([keypoints0], (H0, W0),
                                                                window_size=self.window_size, scale=scale0)

                kp_cross_list0.extend(kp_windows_list0)
                kp_cross_list1.extend(kp_windows_list1)
                mask_cross_list0.extend(masks_list0)
                mask_cross_list1.extend(masks_list1)
            else:
                kp_cross_list1.append(None)
                kp_cross_list0.append(None)
                mask_cross_list1.append(None)
                mask_cross_list0.append(None)

        map0 = torch.zeros([bs, 1, hh0, ww0]).to(
            desc_map0.device)
        map1 = torch.zeros([bs, 1, hh1, ww1]).to(
            desc_map0.device)
        for i in range(bs):
            map0[i, :, kps0[i][:, 1] // scale, kps0[i][:, 0] // scale] = 1
            map1[i, :, kps1[i][:, 1] // scale, kps1[i][:, 0] // scale] = 1

        map0 = map0 > 0.5
        map1 = map1 > 0.5

        map0 = map0.view(bs, -1)
        map1 = map1.view(bs, -1)

        desc_map0_o, desc_map1_o = self.des_transformer(desc_map0, desc_map1, kp_cross_list0,
                                                        kp_cross_list1, hh0, ww0, hh1,
                                                        ww1, scale,
                                                        mask_self0=map0, mask_self1=map1, mask_cross0=mask_cross_list0,
                                                        mask_cross1=mask_cross_list1,
                                                        data=data
                                                        )

        return desc_map0_o, desc_map1_o

    def forward(self, cnn_desc0, cnn_desc1, batch):
        mkpt0_c, mkpt1_c, m_bid = batch['mkpts0_c'],  batch['mkpts1_c'], batch['m_bids']
        bs = batch['image0'].shape[0]
        kpts0 = [mkpt0_c[m_bid == b].long() for b in range(bs)]
        kpts1 = [mkpt1_c[m_bid == b].long() for b in range(bs)]

        desc_map0_o, desc_map1_o = self.apply_RANSAC(cnn_desc0, cnn_desc1, kpts0, kpts1, batch)

        desc_map0, desc_map1 = desc_map0_o, desc_map1_o
        return desc_map0, desc_map1

