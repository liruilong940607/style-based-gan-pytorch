from io import BytesIO

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np
import imageio
import cv2

import random

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform=None, resolution=8):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.transform = transform
        
        mask = imageio.imread("face_mask.png")[:, :, -1]
        self.mask_multires = {}
        for res in [8, 16, 32, 64, 128, 256]:
            self.mask_multires[res] = np.float32(cv2.resize(mask, (res, res), interpolation=cv2.INTER_NEAREST)) / 255.0

    def __len__(self):
        return 50_000_000

    def __getitem__(self, index):
        index = index % self.length
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img = txn.get(key)
            img = np.frombuffer(img, dtype=np.float32)
            img = img.reshape(self.resolution, self.resolution, 6) * self.mask_multires[self.resolution][:, :, np.newaxis]
            
        # left right flip
        if random.random() < 0.5:
            img = img[:, ::-1, :]
             # pointcloud
            img[:, :, 0] *= -1
            img = img.copy()
                
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()

        return img
