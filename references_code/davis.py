from __future__ import division
import torch
from torch.utils import data
# general libs
import numpy as np
import math
import time
import os
import random
import argparse
import glob
import json
# image io libs
import cv2
from PIL import Image
from scipy import ndimage, signal


def temporal_transform(frame_indices, sample_range):
    tmp = np.random.randint(0, len(frame_indices) - sample_range)
    return frame_indices[tmp:tmp + sample_range]

#imset - 데이터 레이블 텍스트
#resolution - 화질
class DAVIS(data.Dataset):
    def __init__(self, root, resolution='480p', size=(256, 256), sample_duration=0):
        self.sample_duration = sample_duration
        #경로
        self.root = root
        #마스크 경로, 이미지 경로
        self.mask_dir = os.path.join(root, 'Annotations', resolution)
        self.image_dir = os.path.join(root, 'JPEGImages', resolution)
        _imset_dir = os.path.join(root, 'ImageSets')


        self.size = size
        self.videos = []
        self.videos.append(resolution)

        #videos list에 append
        #num_frames[_video]
        self.img_dir = len(glob.glob(os.path.join(self.image_dir, '*.jpg')))
        _mask = np.array(Image.open(os.path.join(self.mask_dir, '00000.png')).convert("P"))
        # _mask = np.array(mmcv.imread(os.path.join(self.mask_dir, _video, '00000.png'), flag='grayscale'))
        #num_objects[_video]
        self.mk_dir = np.max(_mask)
        self.shape = np.shape(_mask)

    def __len__(self):
        return len(self.videos)

    def __getitem__(self, index):

        images = []
        masks = []
        struct = ndimage.generate_binary_structure(2, 2)

        f_list = list(range(self.img_dir))
        if self.sample_duration > 0:
            f_list = temporal_transform(f_list, self.sample_duration)

        for f in f_list:

            img_file = os.path.join(self.image_dir, '{:05d}.jpg'.format(f))
            image_ = cv2.resize(cv2.imread(img_file), self.size, cv2.INTER_CUBIC)
            # image_ = mmcv.imresize(mmcv.imread(img_file), self.size, 'bicubic')
            image_ = np.float32(image_) / 255.0
            images.append(torch.from_numpy(image_))

            try:
                mask_file = os.path.join(self.mask_dir, '{:05d}.png'.format(f))
            except:
                mask_file = os.path.join(self.mask_dir, '00000.png')
            mask_ = np.array(Image.open(mask_file).convert('P'), np.uint8)
            mask_ = cv2.resize(mask_, self.size, cv2.INTER_NEAREST)
            # mask_ = np.array(mmcv.imread(mask_file, flag='grayscale'), np.uint8)
            # mask_ = mmcv.imresize(mask_, self.size, 'nearest')


            mask_ = (mask_ != 0)
            # else:
            #     select_mask = min(1, mask_.max())
            #     mask_ = (mask_ == select_mask).astype(np.float)

            w_k = np.ones((10, 6))
            mask2 = signal.convolve2d(mask_.astype(np.float), w_k, 'same')
            mask2 = 1 - (mask2 == 0)
            mask_ = np.float32(mask2)
            masks.append(torch.from_numpy(mask_))

        masks = torch.stack(masks)
        masks = (masks == 1).type(torch.FloatTensor).unsqueeze(0)
        images = torch.stack(images).permute(3, 0, 1, 2)

        return images, masks