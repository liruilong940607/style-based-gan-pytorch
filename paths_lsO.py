import os
import glob
import imageio
import tqdm
import multiprocessing
import cv2
import random
import scipy.io
from functools import partial
from os.path import join as osp
import pandas as pd

from utils import *

# roots to store data
root_raw = "/home/ICT2000/rli/mnt/vgldb1/OldLightStageDataTransfer/TrainingData/"
root_process = "/home/ICT2000/rli/mnt/glab2/ForRuilong/OldLightStageData_process2/"

# raw 4k data: e.g. 20190521_Mark_29_0_pointcloud.exr
folder_pointcloud_4k = osp(root_raw, "PointCloud_Aligned")
folder_albedo_4k = osp(root_raw, "DiffuseAlbedo")
file_pointcloud_neutral_4k = osp(folder_pointcloud_4k, "*_*_*_1_pointcloud.exr")
file_albedo_neutral_4k = osp(folder_albedo_4k, "*_*_*_1_diffuse_albedo.exr")


# processed multi-scale data:
folder_pointcloud_ms = osp(root_process, "{}", "PointCloud_Aligned")
folder_albedo_ms = osp(root_process, "{}", "DiffuseAlbedo")
file_pointcloud_neutral_ms = osp(folder_pointcloud_ms, "*_*_*_1_pointcloud.exr")
file_albedo_neutral_ms = osp(folder_albedo_ms, "*_*_*_1_diffuse_albedo.exr")


# raw augmentation data: e.g. 20191002_RyanWatson_01_blendshape_iter_18_pointcloud.exr
folder_pointcloud_aug = osp(root_raw, "GeometryAugmentation_PointCloud")
file_pointcloud_aug = osp(folder_pointcloud_aug, "*_*_*_1_01_blendshape_iter_*_pointcloud.exr")


file_gender_age = "./gender_age/Aligned_Point_Cloud_Neutral_OldLightStageData.xlsx"

class Dataset(object):
    def __init__(self, resolution):
        self.resolution = resolution
        assert resolution in [8, 16, 32, 64, 128, 256]
        
        self.albedos = sorted(glob.glob(file_albedo_neutral_ms.format(resolution)))
        self.pointclouds = sorted(glob.glob(file_pointcloud_neutral_ms.format(resolution)))
        self.length = len(self.albedos)
        assert len(self.albedos) == len(self.pointclouds)
        
        self.labels = df = pd.read_excel(file_gender_age, sheet_name="Sheet1")
        
    def __len__(self):
        return 50_000_000
    
    def __getitem__(self, index):
        index = index % self.length
        albedo_file, pointcloud_file, age, gender = self.get_files(index, aug=True)
        
        albedo = load_img(albedo_file)
        pointcloud = load_img(pointcloud_file)
        
        if isnan(albedo):
            return self.__getitem__(index + 1)
        
        resolution = self.resolution
        if albedo.shape[0] != resolution:
            albedo = cv2.resize(albedo, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        if pointcloud.shape[0] != resolution:
            pointcloud = cv2.resize(pointcloud, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        
        if random.random() < 0.5: #flip
            albedo = albedo[:, ::-1, :].copy()
            pointcloud = pointcloud[:, ::-1, :].copy()
            pointcloud[:, :, 0] *= -1
        
        albedo = To_tensor(albedo)
        pointcloud = To_tensor(pointcloud)
        age = torch.tensor(age, dtype=torch.float32)
        gender = torch.tensor(gender, dtype=torch.int32)
        
        return albedo, pointcloud, age, gender
        
        
    def get_files(self, index, aug=True):
        index = index % self.length
        albedo, pointcloud = self.albedos[index], self.pointclouds[index]
        
        labels = self.labels.loc[self.labels['FileName'] == pointcloud.split("/")[-1]]
        age = float(labels["Age"].values)
        gender = 1 if labels["Gender"].values == "M" else 0
        # ethnicity = labels["Ethnicity"]
        
        if aug and random.random() < 0.5:
            name = pointcloud.split("/")[-1].split("_pointcloud.exr")[0]
            pointcloud_augs = glob.glob(osp(folder_pointcloud_aug, 
                                        f"{name}_01_blendshape_iter_*_pointcloud.exr"))
            if len(pointcloud_augs) > 0:
                pointcloud = random.choice(pointcloud_augs)
            
        assert os.path.exists(albedo)
        assert os.path.exists(pointcloud)
        
        return albedo, pointcloud, age, gender
        
    
#############################################################
# Preprocess
#############################################################
def load_and_save(file, resolutions, skip_if_exist=True):
    img = load_img(file)
    for resolution in resolutions:
        img_resized = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        file_new = file.replace(root_raw, "")
        file_new = osp(root_process, f"{resolution}/", file_new)
        save_img(file_new, img_resized, skip_if_exist)
        
    return True

def preprocess_multiscale(n_worker = 8, resolutions = [8, 16, 32, 64, 128, 256, 512, 1024], skip_if_exist=True):
    resize_fn = partial(load_and_save, resolutions=resolutions, skip_if_exist=skip_if_exist)
    
    files = []
    files += glob.glob(file_pointcloud_neutral_4k) +\
             glob.glob(file_albedo_neutral_4k)
    
    with multiprocessing.Pool(n_worker) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(resize_fn, files)):
            pass
    
if __name__ == "__main__":
#     preprocess_multiscale(n_worker=8, resolutions=[8, 16, 32, 64, 128, 256], skip_if_exist=False)
    
    dataset = Dataset(resolution=64)
    albedo_file, pointcloud_file, age, gender = dataset.get_files(0, aug=True)
    print (albedo_file, pointcloud_file)
    albedo, pointcloud, age, gender = dataset[0]
    print (albedo.shape, pointcloud.shape, age, gender)
