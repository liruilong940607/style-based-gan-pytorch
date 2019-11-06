from io import BytesIO

import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import torch

import glob
import imageio
import scipy.io
import cv2
import IPython
import os
import random
import torch
import tqdm

import paths_ls
import paths_tri
import paths_lsO

from utils import *


##########################################################################
## new
##########################################################################
class MultiResolutionDataset():
    def __init__(self, resolution=256, exclude_neutral=True):
        self.resolution = resolution
        self.exclude_neutral = exclude_neutral
        
        self.labels, self.images, self.neutral_images = [], [], []
        for subset in ["lightstage", "tripleganger", "lightstage-old"]:
            labels, images, neutral_images = self._get_filenames(subset)
            self.labels += labels
            self.images += images
            self.neutral_images += neutral_images
            print (subset, len(labels), len(images), len(neutral_images))
            
        self.labels_mean = load_exp_mean()
        self.labels_std = load_exp_std()
            
        self.length = len(self.labels)
        

    def _get_filenames(self, subset):
        assert subset in ["lightstage", "tripleganger", "lightstage-old"]
        if subset == "lightstage":
            paths = paths_ls
        elif subset == "tripleganger":
            paths = paths_tri
        elif subset == "lightstage-old":
            paths = paths_lsO    
        
        labels = sorted(glob.glob(paths.file_exp_weights))
        images = sorted(glob.glob(paths.file_pointcloud_ms.format(self.resolution)))
        
        labels, images = self._check(labels, images, paths)
        
        neutral_images = []
        keep_indexs = []
        for i, f in enumerate(images):
            if subset == "lightstage":
                neutral_f = f.split("/")[-1].split("_")[0:2]
                neutral_f = "_".join(neutral_f) + "*_01_pointcloud.exr"
            elif subset == "tripleganger":
                neutral_f = f.split("/")[-1].split("_")[0:1]
                neutral_f = "_".join(neutral_f) + "*_01_*_pointcloud.exr"
            elif subset == "lightstage-old":
                neutral_f = f.split("/")[-1].split("_")[0:3]
                neutral_f = "_".join(neutral_f) + "*_1_pointcloud.exr"
            neutral_f = os.path.join(paths.folder_pointcloud_ms.format(self.resolution), neutral_f)
            neutral_f = glob.glob(neutral_f)
            assert len(neutral_f) <= 1
            if len(neutral_f) == 0:
                continue
            neutral_f = neutral_f[0]            
            assert os.path.exists(neutral_f)
            neutral_images.append(neutral_f)
            keep_indexs.append(i)
        
        labels = [labels[i] for i in keep_indexs]
        images = [images[i] for i in keep_indexs]
        return labels, images, neutral_images
    
        
    def _check(self, labels, images, paths):
        labels_valid, images_valid = [], []
        for label in labels:
            image = os.path.join(paths.folder_pointcloud_ms.format(self.resolution),
                                 label.split('/')[-1].replace('_BSweights.mat', '_pointcloud.exr'))
            if image in images:
                labels_valid.append(label)
                images_valid.append(image)
        return labels_valid, images_valid
        
        
    def __len__(self):
        return 100_000_000
    
    
    def sample_label(self, k=1, randn=True):
        # return [k * label_size]
        mean = torch.from_numpy(self.labels_mean).unsqueeze(0).repeat(k, 1)
        std = torch.from_numpy(self.labels_std).unsqueeze(0).repeat(k, 1) * 5
        return torch.normal(mean=mean, std=std).float()
            
        
    def getitem_neutral(self, index=None, rand=False):
        if rand == True:
            file = random.choice(self.neutral_images)
        else:
            index = index % self.length
            file = self.neutral_images[index]
        img_neutral = To_tensor(load_img(file)) 
        if random.random() < 0.5:
            img_neutral = self.flip_img(img_neutral)
        return img_neutral
    
    
    def flip_img(self, tensor):
        # [c,h,w]
        tensor = tensor.clone()
        inv_idx = torch.arange(tensor.size(2)-1, -1, -1).long()
        tensor = tensor[:, :, inv_idx]
        tensor[0, :, :] *= -1
        return tensor
    
    
    def __getitem__(self, index):
        index = index % self.length
        img = To_tensor(load_img(self.images[index]))
        
        if self.exclude_neutral:
            img_neutral = To_tensor(load_img(self.neutral_images[index]))
            img -= img_neutral
            
        label = torch.from_numpy(load_mat(self.labels[index], "BSweights"))[:, 0]
        return img.float(), label.float()
    
    
    
    
    
if __name__ == "__main__":
    dataset = MultiResolutionDataset()

    print (dataset.sample_label(10).shape)
    
    img, label = dataset[0]
    for i in range(10):
        img_neutral = dataset.getitem_neutral(rand=True)
        
        #save_img(f"test_rand_add_neutral/{i}.exr", To_numpy(img + img_neutral))
        save_mat(f"test_rand_add_neutral/{i}.mat", To_numpy(img + img_neutral), "data")
