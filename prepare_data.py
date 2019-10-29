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
    return img.tobytes()


def resize_multiple(img, sizes=(8, 16, 32, 64, 128, 256, 512, 1024), quality=100):
    imgs = []

    for size in sizes:
        imgs.append(resize_and_convert(img, size, quality))
        
    return imgs

weights = []

def resize_worker(img_file, sizes, exclude_neutral=False):
    i, file, wfile = img_file
    img = imageio.imread(file, format='EXR-FI')
    if exclude_neutral:
        subs = file.split("_")
        subs[-2] = "01"
        file_neutral = "_".join(subs)
        img_neutral = imageio.imread(file_neutral, format='EXR-FI')
        img = img - img_neutral
        
    out = resize_multiple(img, sizes=sizes)

    weight = scipy.io.loadmat(wfile)["BSweights"][:, 0]
    weight[weight < 1e-6] = 0.0
    weight = np.float32(weight).tobytes()
    
    return i, out, weight, wfile


def prepare(transaction, dataset, wset, args, sizes=(8, 16, 32, 64, 128, 256, 512, 1024)):
    resize_fn = partial(resize_worker, sizes=sizes, exclude_neutral=args.exclude_neutral)
    n_worker = args.n_worker

    files = [(i, file, wfile) for i, (file, wfile) in enumerate(zip(dataset, wset))]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs, weight, wfile in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = f'{size}-{str(i).zfill(5)}'.encode('utf-8')
                transaction.put(key, img)
                key = f'{size}-{str(i).zfill(5)}-weight'.encode('utf-8')
                transaction.put(key, weight)
                key = f'{size}-{str(i).zfill(5)}-file'.encode('utf-8')
                transaction.put(key, wfile.encode('utf-8'))

            total += 1

        transaction.put('length'.encode('utf-8'), str(total).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str)
    parser.add_argument('--n_worker', type=int, default=8)
    parser.add_argument('--exclude_neutral', action='store_true', default=False)

    args = parser.parse_args()

    path_pointcloud = "/mount/ForRuilong/FaceEncoding_process/1024/PointCloud_Aligned/{}_pointcloud.exr"
    
    wholeset = sorted(glob.glob(path_pointcloud.format("*")))
    if args.exclude_neutral:
        wholeset = [f for f in wholeset if "_01_" not in f]
    
    imgset = []
    wset = []
    for imgfile in wholeset:
        wfile = imgfile.replace("PointCloud_Aligned", "BlendingWeights_9").replace("_pointcloud.exr", "_BSweights.mat")
        if os.path.exists(wfile):
            wset.append(wfile)
            imgset.append(imgfile)
            
    print (len(imgset), len(wset))

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        with env.begin(write=True) as txn:
            prepare(txn, imgset, wset, args)