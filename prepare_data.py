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


def resize_worker(img_file, sizes):
    i, file = img_file
    img = imageio.imread(file, format='EXR-FI')
    out = resize_multiple(img, sizes=sizes)

    return i, out


def prepare(transaction, dataset, n_worker, sizes=(8, 16, 32, 64, 128, 256, 512, 1024)):
    resize_fn = partial(resize_worker, sizes=sizes)

    files = [(i, file) for i, file in enumerate(dataset)]
    total = 0

    with multiprocessing.Pool(n_worker) as pool:
        for i, imgs in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = f'{size}-{str(i).zfill(5)}'.encode('utf-8')
                transaction.put(key, img)

            total += 1

        transaction.put('length'.encode('utf-8'), str(total).encode('utf-8'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str)
    parser.add_argument('--n_worker', type=int, default=8)

    args = parser.parse_args()

    path_pointcloud = "/mount/ForRuilong/FaceEncoding_process/1024/PointCloud_Aligned/{}_pointcloud.exr"
    
    imgset = sorted(glob.glob(path_pointcloud.format("*")))
    print (len(imgset))

    with lmdb.open(args.out, map_size=1024 ** 4, readahead=False) as env:
        with env.begin(write=True) as txn:
            prepare(txn, imgset, args.n_worker)
