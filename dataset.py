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

def To_tensor(img):
    return torch.from_numpy(img.transpose(2, 0, 1)).float()

def To_numpy(tensor):
    return tensor.detach().cpu().numpy().transpose(1, 2, 0)


def load_img(filename):
    return imageio.imread(filename, format='EXR-FI')

def save_img(filename_out, img, skip_if_exist=False):    
    if skip_if_exist and os.path.exists(filename_out):
        return
    os.makedirs(os.path.dirname(filename_out), exist_ok=True)
    imageio.imwrite(filename_out, img, format='EXR-FI')

    
def load_mat(filename, key):
    return scipy.io.loadmat(filename)[key]

def save_mat(filename_out, data, key, skip_if_exist=False):
    if skip_if_exist and os.path.exists(filename_out):
        return
    os.makedirs(os.path.dirname(filename_out), exist_ok=True)
    scipy.io.savemat(filename_out, {key: data})


class BaseDataset():
    def __init__(self, exclude_neutral=False):
        self.root_geo = "/mount/ForRuilong/FaceEncoding_process/{}/PointCloud_Aligned/"
        self.root_expw = "/mount/ForRuilong/FaceEncoding_process/{}/BlendingWeights_9/"
        self.root_geo_neutral = self.root_geo
        
        self.identities, self.names= self.extract_filenames(exclude_neutral)
        self.ids = list(self.identities.keys())
        
        # specific attributes
        self.labels = self.load_labels()
    
    def get_img_file(self, name, resolution):
        return os.path.join(self.root_geo.format(resolution), f"{name}_pointcloud.exr")
        
    def get_label_file(self, name):
        return os.path.join(self.root_expw.format(1024), f"{name}_BSweights.mat")
    
    def get_neutral_file(self, name, resolution):
        img_neutral_path = os.path.join(self.root_geo_neutral.format(resolution), 
                                        f"{name[:-3]}_01_pointcloud.exr")
        return img_neutral_path
    
    def load_labels(self):
        labels = []
        for id, names in self.identities.items():
            for name in names:
                label = torch.from_numpy(load_mat(self.get_label_file(name), "BSweights")[:, 0]).float() 
                labels.append(label)
        return labels
        
    def sample_label(self, k=1):
        return torch.stack(random.choices(self.labels, k=k))
    
    def extract_filenames(self, exclude_neutral=False):
        files_geo = os.listdir(self.root_geo.format("1024"))
        files_expw = os.listdir(self.root_expw.format("1024"))
        
        names = [f.replace("_pointcloud.exr", "") for f in files_geo if f.replace("_pointcloud.exr", "_BSweights.mat") in files_expw]
        identities = {}
        for name in names:
            if exclude_neutral and name[-3:] == "_01":
                continue
            identity = name[:-3]
            if identity not in identities:
                identities[identity] = []
            identities[identity].append(name)
                
        return identities, names
    

def _load_and_save(file, root_geo, root_expw):
    img = load_img(os.path.join(root_geo, file))
    wfile_in = file.replace("_pointcloud.exr", ".mat")
    wfile_out = file.replace("_pointcloud.exr", "_BSweights.mat")
    weight = load_mat(os.path.join(root_expw, wfile_in), "BSweights")        
    for resolution in [8, 16, 32, 64, 128, 256, 512, 1024]:
        os.makedirs(f"/mount/ForRuilong/FaceWareHouseExp/{resolution}", exist_ok=True)
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        save_img(f"/mount/ForRuilong/FaceWareHouseExp/{resolution}/PointCloud_Aligned/{file}", img, skip_if_exist=True)
        save_mat(f"/mount/ForRuilong/FaceWareHouseExp/{resolution}/BlendingWeightsFWH/{wfile_out}", weight, "BSweights", skip_if_exist=True)
    
    
class FWHDataset():
    @classmethod
    def process(cls):
        import tqdm
        import multiprocessing
        from functools import partial

        root_geo = "/mount/ForRuilong/FaceWareHouseExp/tempExpressions/"
        root_expw = "/mount/ForRuilong/FaceWareHouseExp/RandmonMeshFacewarehouse/"
        
        files = [f for f in os.listdir(root_geo) if f[-3:] == "exr"]
        
        resize_fn = partial(_load_and_save, root_geo=root_geo, root_expw=root_expw)
        with multiprocessing.Pool(16) as pool:
            for _ in tqdm.tqdm(pool.imap_unordered(resize_fn, files)):
                pass
        '''
        for file in tqdm.tqdm(files):
            img = load_img(os.path.join(root_geo, file))
            wfile_in = file.replace("_pointcloud.exr", ".mat")
            wfile_out = file.replace("_pointcloud.exr", "_BSweights.mat")
            weight = load_mat(os.path.join(root_expw, wfile_in), "BSweights")        
            for resolution in [8, 16, 32, 64, 128, 256, 512, 1024]:
                os.makedirs(f"/mount/ForRuilong/FaceWareHouseExp/{resolution}", exist_ok=True)
                img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
                save_img(f"/mount/ForRuilong/FaceWareHouseExp/{resolution}/PointCloud_Aligned/{file}", img)
                save_mat(f"/mount/ForRuilong/FaceWareHouseExp/{resolution}/BlendingWeightsFWH/{wfile_out}", weight, "BSweights")
        '''
    
    def __init__(self, exclude_neutral=False):
        #self.root_geo = "/mount/ForRuilong/FaceWareHouseExp/tempExpressions/20191002_RyanWatson_01_blendshape_25_iter_15_pointcloud.exr"
        #self.root_expw = "/mount/ForRuilong/FaceWareHouseExp/RandmonMeshFacewarehouse/20191002_RyanWatson_01_blendshape_25_iter_15.mat"
        self.root_geo = "/mount/ForRuilong/FaceWareHouseExp/{}/PointCloud_Aligned/"
        self.root_expw = "/mount/ForRuilong/FaceWareHouseExp/{}/BlendingWeightsFWH/"
        self.root_geo_neutral = "/mount/ForRuilong/FaceEncoding_process/{}/PointCloud_Aligned/"
        
        assert exclude_neutral == True
        
        self.identities, self.names= self.extract_filenames()
        self.ids = list(self.identities.keys())
    
        # specific attributes
        self.labels = self.load_labels()
        # self.labels = []
        self.labels_mean = load_mat("/mount/ForRuilong/FaceWareHouseExp/weightsFacewareMeanStd.mat", "Expression_mean")[0]
        self.labels_std = load_mat("/mount/ForRuilong/FaceWareHouseExp/weightsFacewareMeanStd.mat", "Expression_std")[0]
    
    def get_img_file(self, name, resolution):
        return os.path.join(self.root_geo.format(resolution), f"{name}_pointcloud.exr")
        
    def get_label_file(self, name):
        return os.path.join(self.root_expw.format(1024), f"{name}_BSweights.mat")
    
    def get_neutral_file(self, name, resolution):
        neutral_id = name.split("_01_blendshape_")[0]
        img_neutral_path = os.path.join(self.root_geo_neutral.format(resolution),
                                        f"{neutral_id}_01_pointcloud.exr")
        return img_neutral_path
    
    def load_labels(self):
        labels = []
        print ("loading labels ...")
        for id, names in tqdm.tqdm(self.identities.items()):
            for name in names:
                label = torch.from_numpy(load_mat(self.get_label_file(name), "BSweights")).float() 
                labels.append(label)
        return labels
        
    def sample_label(self, k=1, randn=False):
        # return [k * label_size]
        if randn:
            mean = torch.from_numpy(self.labels_mean).unsqueeze(0).repeat(k, 1)
            std = torch.from_numpy(self.labels_std).unsqueeze(0).repeat(k, 1)
            return torch.normal(mean=mean, std=std)
            
        else:
            return torch.stack(random.choices(self.labels, k=k))
    
    
    def extract_filenames(self):
        files_geo = os.listdir(self.root_geo.format("1024"))
        files_expw = os.listdir(self.root_expw.format("1024"))
        
        names = [f.replace("_pointcloud.exr", "") for f in files_geo if f.replace("_pointcloud.exr", "_BSweights.mat") in files_expw]
        identities = {}
        for name in names:
            identity = name.split("_01_blendshape_")[0]
            if not os.path.exists(self.get_neutral_file(name, 1024)):
                continue
            if identity not in identities:
                identities[identity] = []
            identities[identity].append(name)
                
        return identities, names
    

#class MultiResolutionDataset(BaseDataset):
class MultiResolutionDataset(FWHDataset):
    def __init__(self, resolution=8, return_neutral=False, sameID=True, exclude_neutral=True):
        super().__init__(exclude_neutral=exclude_neutral)
        self.sameID = sameID
        self.return_neutral = return_neutral
        self.resolution = resolution
        self.exclude_neutral = exclude_neutral
        print ("Number of Labels:", len(self.labels))
        print ("Exclude Neutral:", exclude_neutral)
        
    def __len__(self):
        return 10_000_000
    
    def getitem_neutral(self, index=None, rand=False):
        if rand == True:
            id_sample = random.choice(self.ids)
        else:
            index = index % len(self.ids)
            id_sample = self.ids[index]
        
        name = self.identities[id_sample][0]
        img_neutral_path = self.get_neutral_file(name, self.resolution)
        img_neutral = To_tensor(load_img(img_neutral_path)) 
        return img_neutral
    
    def __getitem__(self, index):
        if self.return_neutral:
            return self.getitem_neutral(rand=True)
        
        sample_k = 1
        
        if self.sameID:
            id_sample = random.choice(self.ids)
            names = random.sample(self.identities[id_sample], k=sample_k)
            
        else:
            ids = random.sample(self.ids, k=sample_k)
            names = [random.choice(self.identities[id]) for id in ids]
        
        imgs_path = [self.get_img_file(name, self.resolution) for name in names]
        imgs = [To_tensor(load_img(img_path)) for img_path in imgs_path]
        
        if self.exclude_neutral:
            imgs_neutral_path = [self.get_neutral_file(name, self.resolution) for name in names]
            imgs_neutral = [To_tensor(load_img(img_neutral_path)) for img_neutral_path in imgs_neutral_path]
            imgs = [img-img_neutral for img, img_neutral in zip(imgs, imgs_neutral)]
            
        labels_path = [self.get_label_file(name) for name in names]
        labels = [torch.from_numpy(load_mat(label_path, "BSweights")).float() for label_path in labels_path]
            
        return imgs[0], labels[0]
        # return imgs[0], labels[0], imgs[1], labels[1] 

if __name__ == "__main__":
    #FWHDataset.process()
    
    
    dataset = MultiResolutionDataset(resolution=64, exclude_neutral=True)

    labels = dataset.sample_label(k=10, randn=True)
    print (labels)
    
    img_neutral = dataset.getitem_neutral(rand=True)
    for i in range(10):
        img, label = dataset[0]
        
        save_img(f"test_rand_add_neutral/{i}.exr", To_numpy(img + img_neutral))
#        # save_mat(f"tmp/test.mat", To_numpy(img + img_neutral), "data")