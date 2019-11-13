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
        for res in [8, 16, 32, 64, 128, 256]:
            self.mask_multires[res] = torch.from_numpy(
                cv2.resize(mask, (res, res), interpolation=cv2.INTER_NEAREST)[np.newaxis, :, :]
            ).float() / 255.0
            
        mean_pc = load_img("/home/ICT2000/rli/mnt/vgldb1/LightStageFaceDB/Datasets/FaceEncoding/generic_neutral_pointcloud.exr")
        self.mean_pc_multires = {}
        for res in [8, 16, 32, 64, 128, 256]:
            self.mean_pc_multires[res] = To_tensor(
                cv2.resize(mean_pc, (res, res), interpolation=cv2.INTER_CUBIC)
            ).float()
            
        self.mean, self.std = self.load_mean()
        
        # gender & age
        self.mean_gender = 0.5
        self.std_gender = 0.5
        self.mean_age = 32.5841
        self.std_age = 13.9430
            
    def calc_mean(self, save=False):
        loader = DataLoader(self, shuffle=False, batch_size=32, num_workers=16)
        data_loader = iter(loader)

        images = []
        for i in tqdm.tqdm(range(200)):
            real_image, age, gender = next(data_loader)
            images.append(real_image)

        images = torch.cat(images, dim=0)
        images = images.permute(1, 0, 2, 3)
        images = images.clone().view(6, -1)
        std = images.std(dim=1).view(-1, 1, 1)
        mean = images.mean(dim=1).view(-1, 1, 1)
        print (mean, std)
        if save:
            torch.save({'mean': mean, 'std': std}, f'./norm_data_subneutral.data')

    def load_mean(self):
        if os.path.exists("./norm_data_subneutral.data"):
            norm_data = torch.load("./norm_data_subneutral.data")
            mean = norm_data["mean"]
            std = norm_data["std"]
        else:
            mean = 0.0
            std = 1.0
        return mean, std
        
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
        pointcloud -= self.mean_pc_multires[self.resolution]
        
        image = torch.cat([albedo, pointcloud], dim=0).float()
        image = (image - self.mean) / self.std
        
        gender = gender.float()
        gender = (gender - self.mean_gender) / self.std_gender
        age = (age - self.mean_age) / self.std_age
        
        return image, age.view(-1).float(), gender.view(-1).float()
    
    def sample_label(self, k=1, randn=True):
        # return [k * label_size]
        ages = torch.randint(low=18, high=80, size=(k, 1))
        genders = torch.randint(low=0, high=2, size=(k, ))
        return ages.float(), genders.long()

    
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms, utils
    from train import exr2rgb, load_region_range
    import tqdm
    
    dataset = MultiResolutionDataset()
    dataset.calc_mean(save=False)
    
#     img_mean, img_min, img_max = load_region_range()
#     print (f"[drange] min: {img_min}; max: {img_max}; mean: {img_mean}")
    
#     dataset = MultiResolutionDataset()

#     loader = DataLoader(dataset, shuffle=True, batch_size=32, num_workers=16)
#     data_loader = iter(loader)
    
#     images = []
    
#     for i in tqdm.tqdm(range(200)):
#         real_image, age, gender = next(data_loader)
# #         real_image = (real_image - img_mean) / (img_max - img_min + 1e-10)
#         images.append(real_image)
        
#     images = torch.cat(images, dim=0)
#     print (images.shape)
#     images = images.permute(1, 0, 2, 3)
#     print (images.shape)
#     images = images.clone().view(6, -1)
#     std = images.std(dim=1)
#     mean = images.mean(dim=1)
    
#     print (mean, std)
    
#     real_image = real_image * (img_max - img_min + 1e-10) + img_mean
    
#     vis_real = exr2rgb(real_image[:16, 3:6])
#     utils.save_image(
#         vis_real.data.cpu(),
#         f'debug/pointcloud-real_image.jpg',
#         nrow=4,
#         normalize=True,
#         range=(0, 1),
#     )
    
#     vis_real = exr2rgb(real_image[:16, 0:3])
#     utils.save_image(
#         vis_real.data.cpu(),
#         f'debug/albedo-real_image.jpg',
#         nrow=4,
#         normalize=True,
#         range=(0, 1),
#     )
