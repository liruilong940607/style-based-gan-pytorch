from io import BytesIO

import lmdb
from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np
import torch

import imageio
import scipy.io
import cv2
import IPython
import os
import random
import torch

def load_img(filename):
    return imageio.imread(filename, format='EXR-FI')

def To_tensor(img):
    return torch.from_numpy(img.transpose(2, 0, 1)).float()

def load_mat(filename, key):
    return scipy.io.loadmat(filename)[key]


class BaseDataset():
    def __init__(self):
        self.root_geo = "/mount/ForRuilong/FaceEncoding_process/{}/PointCloud_Aligned/"
        self.root_expw = "/mount/ForRuilong/FaceEncoding_process/{}/BlendingWeights_9/"
        
        self.identities, self.names= self.extract_filenames()
        self.ids = list(self.identities.keys())
        
    def extract_filenames(self):
        files_geo = os.listdir(self.root_geo.format("1024"))
        files_expw = os.listdir(self.root_expw.format("1024"))
        
        names = [f.replace("_pointcloud.exr", "") for f in files_geo if f.replace("_pointcloud.exr", "_BSweights.mat") in files_expw]
        identities = {}
        for name in names:
            identity = name[:-3]
            if identity not in identities:
                identities[identity] = []
            identities[identity].append(name)
                
        return identities, names
    

class MultiResolutionDataset(BaseDataset):
    def __init__(self, resolution=8, sameID=True):
        super().__init__()
        self.sameID = sameID
        self.resolution = resolution
        self.labels = self.load_labels()
        print ("Number of Labels:", len(self.labels))
        
    def load_labels(self):
        labels = []
        for id, names in self.identities.items():
            for name in names:
                label_path = os.path.join(self.root_expw.format(self.resolution), f"{name}_BSweights.mat")
                label = torch.from_numpy(load_mat(label_path, "BSweights")[:, 0]).float() 
                labels.append(label)
        return labels
        
    def __len__(self):
        return 10_000_000
    
    def __getitem__(self, index):
        if self.sameID:
            id_sample = random.choice(self.ids)
            name1, name2 = random.sample(self.identities[id_sample], k=2)
            
        else:
            id1, id2 = random.sample(self.ids, k=2)
            name1 = random.choice(self.identities[id1])
            name2 = random.choice(self.identities[id2])
            
        img1_path = os.path.join(self.root_geo.format(self.resolution), f"{name1}_pointcloud.exr")
        img2_path = os.path.join(self.root_geo.format(self.resolution), f"{name2}_pointcloud.exr")
            
        img1 = To_tensor(load_img(img1_path))
        img2 = To_tensor(load_img(img2_path))
            
        label1_path = os.path.join(self.root_expw.format(self.resolution), f"{name1}_BSweights.mat")
        label2_path = os.path.join(self.root_expw.format(self.resolution), f"{name2}_BSweights.mat")
            
        label1 = torch.from_numpy(load_mat(label1_path, "BSweights")[:, 0]).float() 
        label2 = torch.from_numpy(load_mat(label2_path, "BSweights")[:, 0]).float() 
            
        return img1, label1, img2, label2
