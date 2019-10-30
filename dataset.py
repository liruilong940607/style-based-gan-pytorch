from io import BytesIO

import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import torch

import imageio
import cv2

# "cheekPuff", "eyeBlink_L", "eyeBlink_R", "jawForward", "jawOpen", "mouthClose", "mouthFunnel", "mouthSmile_L", "mouthSmile_R"

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
        self.labels = self.load_labels()

    def __len__(self):
        return self.length * 1000
    
    def load_labels(self):
        labels = []
        with self.env.begin(write=False) as txn:
            for index in range(self.length):
                key = f'{self.resolution}-{str(index).zfill(5)}-weight'.encode('utf-8')
                weight = txn.get(key)
                weight = np.frombuffer(weight, dtype=np.float32)
                weight = weight.reshape((9,))
                
                key = f'{self.resolution}-{str(index).zfill(5)}-file'.encode('utf-8')
                file = txn.get(key).decode('utf-8')
            
                label = torch.from_numpy(weight).float() 
                labels.append((file, label))
        return labels

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
            weight = weight.reshape((9,))
            
            key = f'{self.resolution}-{str(index).zfill(5)}-file'.encode('utf-8')
            wfile = txn.get(key).decode('utf-8')
            
            imgfile = wfile.replace("BlendingWeights_9", "PointCloud_Aligned").replace("_BSweights.mat", "_pointcloud.exr").replace("1024", f"{self.resolution}")
            subs = imgfile.split("_")
            subs[-2] = "01"
            file_neutral = "_".join(subs)
            img_neutral = imageio.imread(file_neutral, format='EXR-FI')
#             img_neutral = cv2.resize(img_neutral, (self.resolution, self.resolution), interpolation=cv2.INTER_CUBIC)
            
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        img_neutral = torch.from_numpy(img_neutral.transpose(2, 0, 1)).float()
        label = torch.from_numpy(weight).float() 

        return img, label, img_neutral
