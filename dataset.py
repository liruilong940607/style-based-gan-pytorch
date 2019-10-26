from io import BytesIO

import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import torch

class MultiResolutionDataset(Dataset):
    def __init__(self, path, transform, resolution=8):
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

    def __len__(self):
        return self.length * 1000

    def __getitem__(self, index):
        index = index % self.length
        
        with self.env.begin(write=False) as txn:
            key = f'{self.resolution}-{str(index).zfill(5)}'.encode('utf-8')
            img = txn.get(key)
            img = np.frombuffer(img, dtype=np.float32)
            img = img.reshape(self.resolution, self.resolution, 3)
            
            key = f'{self.resolution}-{str(index).zfill(5)}-weight'.encode('utf-8')
            weight = txn.get(key)
            weight = np.frombuffer(weight, dtype=np.float32)
            weight = weight.reshape((51,))
            
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        label = torch.from_numpy(weight).float() / 6.8

        return img, label
