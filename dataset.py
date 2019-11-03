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

import paths

# on server 01-05, use this.
#root_dir = "/mount/ForRuilong/"
# on my local ARYA machine, use this.
root_dir = "/home/ICT2000/rli/mnt/glab2/ForRuilong/"


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
        self.root_geo = root_dir + "/FaceEncoding_process/{}/PointCloud_Aligned/"
        self.root_expw = root_dir + "/FaceEncoding_process/{}/BlendingWeights_9/"
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
        os.makedirs(root_dir + f"/FaceWareHouseExp/{resolution}", exist_ok=True)
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        save_img(root_dir + f"/FaceWareHouseExp/{resolution}/PointCloud_Aligned/{file}", img, skip_if_exist=True)
        save_mat(root_dir + f"/FaceWareHouseExp/{resolution}/BlendingWeightsFWH/{wfile_out}", weight, "BSweights", skip_if_exist=True)
    
    
class FWHDataset():
    @classmethod
    def process(cls):
        import tqdm
        import multiprocessing
        from functools import partial

        root_geo = root_dir + "/FaceWareHouseExp/tempExpressions/"
        root_expw = root_dir + "/FaceWareHouseExp/RandmonMeshFacewarehouse/"
        
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
        self.root_geo = root_dir + "/FaceWareHouseExp/{}/PointCloud_Aligned/"
        self.root_expw = root_dir + "/FaceWareHouseExp/{}/BlendingWeightsFWH/"
        self.root_geo_neutral = root_dir + "/FaceEncoding_process/{}/PointCloud_Aligned/"
        
        assert exclude_neutral == True
        
        self.identities, self.names= self.extract_filenames()
        self.ids = list(self.identities.keys())
    
        # specific attributes
        self.labels = self.load_labels()
        # self.labels = []
        self.labels_mean = load_mat(root_dir + "/FaceWareHouseExp/weightsFacewareMeanStd.mat", "Expression_mean")[0]
        self.labels_std = load_mat(root_dir + "/FaceWareHouseExp/weightsFacewareMeanStd.mat", "Expression_std")[0]
    
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
                label_cls = int(self.get_label_file(name).split("_blendshape_")[1][0:2]) - 1
                label_reg = label[0, label_cls]
                labels.append((label_cls, label_reg))
        return labels
        
    def sample_label(self, k=1, randn=False):
        # return [k * label_size]
        if randn:
            mean = torch.from_numpy(self.labels_mean).unsqueeze(0).repeat(k, 1)
            std = torch.from_numpy(self.labels_std).unsqueeze(0).repeat(k, 1)
            return torch.normal(mean=mean, std=std)
            
        else:
            labels = random.choices(self.labels, k=k)
            labels_cls = torch.stack([label_cls for (label_cls, label_reg) in labels])
            labels_reg = torch.stack([label_reg for (label_cls, label_reg) in labels])
            return labels_cls, labels_reg
    
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
    
def _load_and_save2(file, root_geo, root_expw):
    img = load_img(os.path.join(root_geo, file))
    wfile_in = file.replace("_pointcloud.exr", ".mat")
    wfile_out = file.replace("_pointcloud.exr", "_BSweights.mat")
    weight = load_mat(os.path.join(root_expw, wfile_in), "BSweights")        
    for resolution in [8, 16, 32, 64, 128, 256, 512, 1024]:
        os.makedirs(root_dir + f"/FaceWareHouseRandExp/{resolution}", exist_ok=True)
        img = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        save_img(root_dir + f"/FaceWareHouseRandExp/{resolution}/PointCloud_Aligned/{file}", img, skip_if_exist=True)
        save_mat(root_dir + f"/FaceWareHouseRandExp/{resolution}/BlendingWeightsFWH/{wfile_out}", weight, "BSweights", skip_if_exist=True)
    
    

class FWHRandDataset():
    def __init__(self, exclude_neutral=True):
        self.root_geo = paths.folder_exp_pointcloud_ms
        self.root_expw = paths.folder_exp_weights
        self.root_geo_neutral = paths.folder_pointcloud_ms
        
        assert exclude_neutral == True
        
        self.identities, self.names= self.extract_filenames()
        self.ids = list(self.identities.keys())
    
        # specific attributes
        self.labels = self.load_labels()
        # self.labels = []
        self.labels_mean = paths.load_exp_mean()
        self.labels_std = paths.load_exp_std()
    
    def get_img_file(self, name, resolution):
        return os.path.join(self.root_geo.format(resolution), f"{name}_pointcloud.exr")
        
    def get_label_file(self, name):
        return os.path.join(self.root_expw, f"{name}_BSweights.mat")
    
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
                labels.append((label))
        return labels
        
    def sample_label(self, k=1, randn=False):
        # return [k * label_size]
        if randn:
            mean = torch.from_numpy(self.labels_mean).unsqueeze(0).repeat(k, 1)
            std = torch.from_numpy(self.labels_std).unsqueeze(0).repeat(k, 1)
            return torch.normal(mean=mean, std=std)
            
        else:
            labels = random.choices(self.labels, k=k)
            labels_cls = torch.stack([label_cls for (label_cls, label_reg) in labels])
            labels_reg = torch.stack([label_reg for (label_cls, label_reg) in labels])
            return labels_cls, labels_reg
    
    def extract_filenames(self):
        files_geo = os.listdir(self.root_geo.format("64"))
        files_expw = os.listdir(self.root_expw)
        
        names = [f.replace("_pointcloud.exr", "") for f in files_geo if f.replace("_pointcloud.exr", "_BSweights.mat") in files_expw]
        identities = {}        
        
        for name in names:
            identity = name.split("_01_blendshape_")[0]
            print (self.get_neutral_file(name, 64))
            if not os.path.exists(self.get_neutral_file(name, 64)):
                continue
            if identity not in identities:
                identities[identity] = []
            identities[identity].append(name)
                
        return identities, names
    

# class MultiResolutionDataset(BaseDataset):
# class MultiResolutionDataset(FWHDataset):
class MultiResolutionDataset(FWHRandDataset):
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
        labels = [torch.from_numpy(load_mat(label_path, "BSweights"))[0, ].float() for label_path in labels_path]
#         labels_cls = [int(label_path.split("_blendshape_")[1][0:2]) - 1 for label_path in labels_path]
#         labels_reg = [label[0, label_cls] for label, label_cls in zip(labels, labels_cls)]
        
#         return imgs[0], labels_cls[0], labels_reg[0]
        return imgs[0], labels[0]
        # return imgs[0], labels[0], imgs[1], labels[1] 

    
    
##########################################################################
## new
##########################################################################
class MultiResolutionDataset():
    def __init__(self, resolution=64, exclude_neutral=True):
        self.resolution = resolution
        self.exclude_neutral = exclude_neutral
        
        self.labels = sorted(glob.glob(paths.file_exp_weights))
        self.images = sorted(glob.glob(paths.file_exp_pointcloud_ms.format(resolution)))
        self.labels, self.images = self._check(self.labels, self.images)
        
        self.neutral_images = []
        for f in self.images:
            neutral_f = f.replace(paths.folder_exp_pointcloud_ms.format(resolution),
                                  paths.folder_pointcloud_ms.format(resolution))
            neutral_f = neutral_f.split("_blendshape_")[0]
            neutral_f = neutral_f[:-2] + "01_pointcloud.exr"
            assert os.path.exists(neutral_f)
            self.neutral_images.append(neutral_f)
            
        self.labels_mean = paths.load_exp_mean()
        self.labels_std = paths.load_exp_std()
            
        self.length = len(self.labels)
        print (len(self.labels), len(self.images), len(self.neutral_images))
        
    def __len__(self):
        return 10_000_000
    
    def _check(self, labels, images):
        labels_valid, images_valid = [], []
        for label in labels:
            image = os.path.join(paths.folder_exp_pointcloud_ms.format(self.resolution),
                                 label.split('/')[-1].replace('.mat', '_pointcloud.exr'))
            if image in images:
                labels_valid.append(label)
                images_valid.append(image)
        return labels_valid, images_valid
        
    
    def sample_label(self, k=1, randn=True):
        # return [k * label_size]
        mean = torch.from_numpy(self.labels_mean).unsqueeze(0).repeat(k, 1)
        std = torch.from_numpy(self.labels_std).unsqueeze(0).repeat(k, 1) * 5
        return torch.normal(mean=mean, std=std).float()
            
    def getitem_neutral(self, index=None, rand=False):
        if rand == True:
            neutral_images = glob.glob(paths.file_pointcloud_neutral_ms.format(self.resolution))
            file = random.choice(neutral_images)
        else:
            index = index % self.length
            file = self.neutral_images[index]
        img_neutral = To_tensor(load_img(file)) 
        if random.random() < 0.5:
            img_neutral = self.flip_img(img_neutral)
        return img_neutral
    
    def flip_img(self, tensor):
        # [c,h,w]
        tensor = tensor.clone()
        inv_idx = torch.arange(tensor.size(2)-1, -1, -1).long()
        tensor = tensor[:, :, inv_idx]
        tensor[0, :, :] *= -1
        return tensor
    
    def flip_label(self, tensor):
        # [25,]
        tensor = tensor.clone()
        idxs_left = [22]
        idxs_right = [24]
        tensor_left = tensor[idxs_left].clone()
        tensor[idxs_left] = tensor[idxs_right]
        tensor[idxs_right] = tensor_left
        return tensor
    
    def __getitem__(self, index):
        index = index % self.length
        print (self.images[index])
        img = To_tensor(load_img(self.images[index]))
        
        if self.exclude_neutral:
            img_neutral = To_tensor(load_img(self.neutral_images[index]))
            img -= img_neutral
            
        label = torch.from_numpy(load_mat(self.labels[index], "BSweights"))[0, ]
        return img.float(), label.float()
    
    
    
    
    
if __name__ == "__main__":
    dataset = MultiResolutionDataset()
    
    t = torch.tensor([-0.23,  0.18, -0.09, -0.03, -0.36,  0.63,  0.4,  -0.27, -0.,    0.42, -0.52,  0.13,
                      -0.35, -0.6 ,  0.03 , 0.28 , 0.09 , 0.03 ,-0.28, -0.03,  0.51,  0.58, -0.38,  0.21,
                      -0.43])
    
    for i in range(10000000):
        img, label = dataset[i]
        if (label-t).abs().mean() < 0.03:
            print (label)
            break
        
#     print (dataset.sample_label(10).shape)
    
#     img, label = dataset[0]
#     for i in range(10):
#         img_neutral = dataset.getitem_neutral(rand=True)
        
#         #save_img(f"test_rand_add_neutral/{i}.exr", To_numpy(img + img_neutral))
#         save_mat(f"test_rand_add_neutral/{i}.mat", To_numpy(img + img_neutral), "data")
