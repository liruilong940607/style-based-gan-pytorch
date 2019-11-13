import os
import torch
import imageio
import cv2
import scipy.io
import numpy as np

def exr2rgb(tensor):
    return (tensor*12.92) * (tensor<=0.0031308).float() + (1.055*(tensor**(1.0/2.4))-0.055) * (tensor>0.0031308).float()
    

#############################################################
# Common Utils
#############################################################
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
    if filename_out.split('.')[-1] == "exr":
        imageio.imwrite(filename_out, img, format='EXR-FI')
    else:
        img = np.uint8(exr2rgb(img))
        imageio.imwrite(filename_out, img)

    
def load_mat(filename, key):
    return scipy.io.loadmat(filename)[key]

def save_mat(filename_out, data, key, skip_if_exist=False):
    if skip_if_exist and os.path.exists(filename_out):
        return
    os.makedirs(os.path.dirname(filename_out), exist_ok=True)
    scipy.io.savemat(filename_out, {key: data})

def isnan(x):
    return (x != x).any()

#############################################################
# Specific Utils
#############################################################
def load_exp_mean():
    # return (25,)
    mean = load_mat("/home/ICT2000/rli/mnt/glab2/ForRuilong/FaceEncoding_process2/weightsFacewareMeanStd.mat", 
                    "Expression_mean")[0]
    return mean
    
def load_exp_std():
    # return (25,)
    std = load_mat("/home/ICT2000/rli/mnt/glab2/ForRuilong/FaceEncoding_process2/weightsFacewareMeanStd.mat",
                   "Expression_std")[0]
    return std