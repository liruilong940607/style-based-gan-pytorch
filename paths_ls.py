import os
import glob
import imageio
import tqdm
import multiprocessing
import cv2
import scipy.io
from functools import partial
from os.path import join as osp

from utils import *

# roots to store data
root_raw = "/home/ICT2000/rli/mnt/vgldb1/LightStageFaceDB/Datasets/FaceEncoding/"
root_process = "/home/ICT2000/rli/mnt/glab2/ForRuilong/FaceEncoding_process2/"


# raw 4k data: e.g. 20190521_Mark_29_0_pointcloud.exr
folder_pointcloud_4k = osp(root_raw, "PointCloud_Aligned")
file_pointcloud_4k = osp(folder_pointcloud_4k, "*_*_*_pointcloud.exr")
file_pointcloud_neutral_4k = osp(folder_pointcloud_4k, "*_*_01_pointcloud.exr")


# processed multi-scale data:
folder_pointcloud_ms = osp(root_process, "{}", "PointCloud_Aligned")
file_pointcloud_ms = osp(folder_pointcloud_ms, "*_*_*_pointcloud.exr")
file_pointcloud_neutral_ms = osp(folder_pointcloud_ms, "*_*_01_pointcloud.exr")


# expression weights
folder_exp_weights = osp(root_raw, "BlendingWeights_facewarehouse")
file_exp_weights = osp(folder_exp_weights, "*_*_*_BSweights.mat")


    
#############################################################
# Preprocess
#############################################################
def load_and_save(file, resolutions, skip_if_exist=True):
    img = load_img(file)
    for resolution in resolutions:
        img_resized = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        file = file.replace(root_raw, "")
        file = osp(root_process, f"{resolution}/", file)
        save_img(file, img_resized, skip_if_exist)
        
    return True

def preprocess_multiscale(n_worker = 8, resolutions = [8, 16, 32, 64, 128, 256, 512, 1024], skip_if_exist=True):
    resize_fn = partial(load_and_save, resolutions=resolutions, skip_if_exist=skip_if_exist)
    
    files = []
    files += glob.glob(file_pointcloud_4k)
    
    with multiprocessing.Pool(n_worker) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(resize_fn, files)):
            pass
    
if __name__ == "__main__":
    preprocess_multiscale(n_worker=8, resolutions=[256], skip_if_exist=False)