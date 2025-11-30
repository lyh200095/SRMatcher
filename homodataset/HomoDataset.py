# -*- coding: utf-8 -*-
import os
import random
import time
import kornia
import imgaug.augmenters as iaa
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2
import numpy as np
from utils.homography import sample_homography, compute_valid_mask

from warnings import filterwarnings

from utils.preprocess_utils import resize_aspect_ratio, get_perspective_mat, scale_homography

filterwarnings(action='ignore', category=DeprecationWarning, message='`np.bool` is a deprecated alias')


class HomoDataset(Dataset):
    def __init__(self, img_dir, size=(640, 640), st=32, rank=0, word_size=None, split_N=1):
        self.img_dir = img_dir
        self.size = size
        imgs = []
        self.st = st
        for curDir, dirs, files in os.walk(img_dir):
            fs = [os.path.join(curDir, x) for x in files]

            for i in fs:
                (path, filename) = os.path.split(i)

                if (i.endswith('.jpg') or i.endswith('.ppm')):
                    imgs.append(i)

        self.data = imgs

        if split_N is not None:
            self.data = self.data[::split_N]

        word_size = 4
        rank=0
        if word_size is not None:
            bz = int(len(self.data) // word_size)
            start = rank * bz
            end = start + bz if start + bz < len(self.data) else len(self.data)
            self.data = self.data[start:end]

        self.model_image_height, self.model_image_width = size
        self.transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

        import albumentations as alb
        self.aug_list = [
            alb.OneOf([alb.RandomBrightness(limit=0.2, p=0.8), alb.RandomContrast(limit=0.3, p=0.6)], p=0.5),
            alb.OneOf([alb.MotionBlur(p=0.5), alb.GaussNoise(p=0.6)], p=0.5),
            ]
        self.aug_func = alb.Compose(self.aug_list, p=0.65)
        self.apply_aug = True
        self.config = {
            'apply_color_aug': True,  # whether to apply photometric distortions
            'image_height': size[0],
            'image_width': size[1],
            'augmentation_params':{
                'patch_ratio': 0.8,
                'translation': 0.2,  # translation component range
            }
        }
        self.aug_params = self.config['augmentation_params']

    def __len__(self):
        return len(self.data)


    def apply_augmentations(self, image1, image2):
        image1_dict = {'image': image1}
        image2_dict = {'image': image2}
        result1, result2 = self.aug_func(**image1_dict), self.aug_func(**image2_dict)
        return result1['image'], result2['image']

    def get_pair(self, file_path):
        resize = True
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        height, width = image.shape[0:2]
        homo_matrix = None
        while homo_matrix is None:
            homo_matrix = get_perspective_mat(self.aug_params['patch_ratio'], width, height,
                                              self.aug_params['translation'])
            try:
                torch.inverse(torch.from_numpy(homo_matrix))
            except:
                homo_matrix = None
        warped_image = cv2.warpPerspective(image.copy(), homo_matrix, (width, height))

        if self.st == 0:
            if height > width:
                self.config['image_width'], self.config['image_height'] = self.size[1], self.size[0]
            else:
                self.config['image_width'], self.config['image_height'] = self.size[0], self.size[1]
        else:
            if height > width:
                self.config['image_height'] = self.size[0]
                self.config['image_width'] = int((self.config['image_height'] / height * width) // self.st * self.st)
            else:
                self.config['image_width'] = self.size[1]
                self.config['image_height'] = int((self.config['image_width'] / width * height) // self.st * self.st)

        if resize:
            orig_resized = cv2.resize(image, (self.config['image_width'], self.config['image_height']))
            warped_resized = cv2.resize(warped_image, (self.config['image_width'], self.config['image_height']))
        else:
            orig_resized = image
            warped_resized = warped_image
        if self.apply_aug:
            orig_resized, warped_resized = self.apply_augmentations(orig_resized, warped_resized)
        homo_matrix = scale_homography(homo_matrix, height, width, self.config['image_height'],
                                       self.config['image_width']).astype(np.float32)
        orig_resized = np.expand_dims(orig_resized, 0).astype(np.float32) / 255.0
        warped_resized = np.expand_dims(warped_resized, 0).astype(np.float32) / 255.0
        return orig_resized, warped_resized, homo_matrix

    def get_images_labels(self, index):
        is_neg = False
        file_path = self.data[index]
        image0, image1, homography = self.get_pair(file_path)
        homography = torch.from_numpy(homography).unsqueeze(0)
        if np.random.uniform(0, 1) < 0.:  # negative
            is_neg = True
            neg_path = self.get_neg_sample(index, False)
            neg0, neg1, _ = self.get_pair(neg_path)
            image1 = neg0
            if np.random.uniform(0, 1) < 0.3:
                image1 = neg1
            valid_mask_left = torch.zeros([1, image0.shape[1], image0.shape[2]])
            valid_mask_right = torch.zeros([1, image1.shape[1], image1.shape[2]])
            homography = torch.eye(3).unsqueeze(0)
            return image0, image1, homography, valid_mask_left, valid_mask_right, is_neg
        valid_mask_right = compute_valid_mask(image0.shape[1:], homography)
        valid_mask_left = kornia.geometry.transform.warp_perspective(valid_mask_right.unsqueeze(0),
                                                                     torch.inverse(homography),
                                                                     image1.shape[1:],
                                                                     align_corners=True)[0]
        if np.random.uniform(0, 1) < 0.5:
            homography = torch.inverse(homography)
            tmp = image1
            image1 = image0
            image0 = tmp
            valid_mask = valid_mask_right
            valid_mask_right = valid_mask_left
            valid_mask_left = valid_mask

        return image0, image1, homography, valid_mask_left, valid_mask_right, is_neg
    def __getitem__(self, index: int):
        image0, image1, homography, valid_mask_left, valid_mask_right, is_neg = self.get_images_labels(index)

        name = self.data[index]
        name = os.path.split(name)[-1]
        data = {
            'image0': image0,  # (1, h, w)
            'image1': image1,
            'H_0to1': homography[0],  # (1, 3, 3)
            'H_1to0': torch.inverse(homography)[0],
            'valid_mask_left': valid_mask_left,
            'valid_mask_right': valid_mask_right,
            'is_negs': is_neg,
            'dataset_name': 'Oxford',
            'pair_id': index,
            'pair_names': (name + '_0', name + '_1'),
        }
        return data
    
    def on_the_fly(self, query, mask=None, mode=0):
        m = mask
        image_shape = query.shape[:2]
        homography = sample_homography(image_shape, 'cpu', mode=mode)
        valid_mask = compute_valid_mask(image_shape, homography)

        r = cv2.warpPerspective(query, homography.squeeze().numpy(), tuple(image_shape[::-1]))
        if mask is not None:
            m = cv2.warpPerspective(mask, homography.squeeze().numpy(), tuple(image_shape[::-1]))
            m[m < 10] = 0
        r[r < 10] = 0

        return r, m, homography, valid_mask
