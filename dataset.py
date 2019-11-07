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
    def __init__(self, resolution=256):
        self.resolution = resolution
        self.datasets_ls = paths_ls.Dataset(resolution)
        self.datasets_tri = paths_tri.Dataset(resolution)
        self.datasets_lsO = paths_lsO.Dataset(resolution)
        
        self.length = self.datasets_ls.length + self.datasets_tri.length + self.datasets_lsO.length
        
        mask = imageio.imread("./face_mask.png")[:, :, -1]
        self.mask_multires = {}
        for res in [8, 16, 32, 64, 128, 256, 512, 1024]:
            self.mask_multires[res] = torch.from_numpy(
                cv2.resize(mask, (res, res), interpolation=cv2.INTER_NEAREST)[np.newaxis, :, :]
            ).float() / 255.0

        
    def __len__(self):
        return 100_000_000
    
    def __getitem__(self, index):
        index = index % self.length
        
        if index < self.datasets_ls.length:
            dataset = self.datasets_ls
        elif index < self.datasets_ls.length + self.datasets_tri.length:
            index -= self.datasets_ls.length
            dataset = self.datasets_tri
        else:
            index -= self.datasets_ls.length + self.datasets_tri.length
            dataset = self.datasets_lsO
    
        albedo, pointcloud, age, gender = dataset[index]
        
        albedo *= self.mask_multires[self.resolution]
        pointcloud *= self.mask_multires[self.resolution]
        
        image = torch.cat([albedo, pointcloud], dim=0).float()
        
        return image, age.view(-1).float(), gender.long()
    
    def sample_label(self, k=1, randn=True):
        # return [k * label_size]
        ages = torch.randint(low=18, high=80, size=(k, 1))
        genders = torch.randint(low=0, high=2, size=(k, ))
        return ages.float(), genders.long()

    
if __name__ == "__main__":
    dataset = MultiResolutionDataset()

    ages, genders = dataset.sample_label(10)
    print (ages, genders)
    
    for i in range(10):
        albedo, pointcloud, age, gender = dataset[i]
        print (albedo.shape, pointcloud.shape, age, gender)