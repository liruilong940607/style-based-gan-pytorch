import argparse
from io import BytesIO
import multiprocessing
from functools import partial

from PIL import Image
import lmdb
from tqdm import tqdm
from torchvision import datasets
from torchvision.transforms import functional as trans_fn

import cv2
import imageio
import scipy.io
import glob
import os
import numpy as np
import re

def resize_and_convert(img, size, quality=100):
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC)
    return img


def resize_multiple(img, sizes=(8, 16, 32, 64, 128, 256, 512, 1024), quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, quality))
        
    return imgs


def resize_worker(img_file, sizes):
    i, file1, file2 = img_file
    img1 = imageio.imread(file1, format='EXR-FI')
    img2 = imageio.imread(file2, format='EXR-FI')
    out1 = resize_multiple(img1, sizes=sizes)
    out2 = resize_multiple(img2, sizes=sizes)

    out = [np.concatenate([o1, o2], axis=2).tobytes() for o1, o2 in zip(out1, out2)]
    return i, out


def prepare(transaction, dataset1, dataset2, n_worker, sizes=(8, 16, 32, 64, 128, 256, 512, 1024)):
    resize_fn = partial(resize_worker, sizes=sizes)

    files = [(i, file1, file2) for i, (file1, file2) in enumerate(zip(dataset1, dataset2))]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = f'{size}-{str(i).zfill(5)}'.encode('utf-8')
                transaction.put(key, img)
                print 

            total += 1

        transaction.put('length'.encode('utf-8'), str(total).encode('utf-8'))

def _check(pointclouds, albedos):
    pointclouds_valid, albedos_valid = [], []
    for pointcloud in pointclouds:
        dirname, filename = os.path.split(pointcloud)
        dirname = dirname.replace("PointCloud_Aligned", "DiffuseAlbedo")
        filename = filename.replace("_pointcloud.exr", "_diffuse_albedo.exr")
        albedo = os.path.join(dirname, filename)
        assert os.path.exists(albedo)
        assert os.path.exists(pointcloud)
        if albedo in albedos:
            pointclouds_valid.append(pointcloud)
            albedos_valid.append(albedo)
    return pointclouds_valid, albedos_valid
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default="./lmdb")
    parser.add_argument('--n_worker', type=int, default=8)

    args = parser.parse_args()

    pointcloud_ls = "/home/ICT2000/rli/mnt/vgldb1/LightStageFaceDB/Datasets/FaceEncoding/PointCloud_Aligned/*_01_pointcloud.exr"
    albedo_ls = "/home/ICT2000/rli/mnt/vgldb1/LightStageFaceDB/Datasets/FaceEncoding/DiffuseAlbedo/*_01_diffuse_albedo.exr"

    pointcloud_tri = "/home/ICT2000/rli/mnt/vgldb1/InfiniteRealities_Triplegangers/TrainingData/PointCloud_Aligned/Age*-*_01_*_pointcloud.exr"
#     albedo_tri = "/home/ICT2000/rli/mnt/vgldb1/InfiniteRealities_Triplegangers/TrainingData/DiffuseAlbedo/Age*-*_01_*_diffuse_albedo.exr"
    albedo_tri = "/home/ICT2000/rli/mnt/vgldb1/InfiniteRealities_Triplegangers/TrainingDataV2/DiffuseAlbedo_sRGB/Age*-*_01_*_diffuse_albedo.exr"
    
    pointcloud_lso = "/home/ICT2000/rli/mnt/vgldb1/OldLightStageDataTransfer/TrainingData/PointCloud_Aligned/*_1_pointcloud.exr"
    albedo_lso = "/home/ICT2000/rli/mnt/vgldb1/OldLightStageDataTransfer/TrainingData/DiffuseAlbedo/*_1_diffuse_albedo.exr"
    
    pointclout_set = glob.glob(pointcloud_ls) + glob.glob(pointcloud_tri) + glob.glob(pointcloud_lso)
    albedo_set = glob.glob(albedo_ls) + glob.glob(albedo_tri) + glob.glob(albedo_lso)
    pointclout_set, albedo_set = _check(pointclout_set, albedo_set)
    
    print (len(pointclout_set), len(albedo_set))

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        with env.begin(write=True) as txn:
            prepare(txn, pointclout_set, albedo_set, args.n_worker)
