import os
import glob
import imageio
import tqdm
import multiprocessing
import cv2
import scipy.io
from functools import partial
from os.path import join as osp


# roots to store data
# root_raw = "/home/ICT2000/rli/mnt/vgldb1/LightStageFaceDB/Datasets/FaceEncoding/"
# root_process = "/home/ICT2000/rli/mnt/glab2/ForRuilong/FaceEncoding_process2/"
# root_yajie = "/home/ICT2000/rli/mnt/glab2/Users/yajie/"

root_raw = ""
root_process = "/mount/ForRuilong/FaceEncoding_process2/"
root_yajie = "/mount/Users/yajie/"

# folders 4k
folder_pointcloud_4k = osp(root_raw, "PointCloud_Aligned")
folder_diffuse_4k = osp(root_raw, "DiffuseAlbedo")
folder_specular_4k = osp(root_raw, "SpecularUnlit")
folder_displacement_4k = osp(root_raw, "Displacement")
# files 4k
file_pointcloud_4k = osp(folder_pointcloud_4k, "*_*_*_pointcloud.exr")
file_pointcloud_neutral_4k = osp(folder_pointcloud_4k, "*_*_01_pointcloud.exr")
file_diffuse_4k = osp(folder_diffuse_4k, "*_*_*_diffuse_albedo.exr")
file_specular_4k = osp(folder_specular_4k, "*_*_*_specular_unlit.exr")
file_displacement_4k = osp(folder_displacement_4k, "*_*_*_displacement.exr")


# folders multi scales [2^3 ... 2^10]
folder_pointcloud_ms = osp(root_process, "{}", "PointCloud_Aligned")
folder_diffuse_ms = osp(root_process, "{}", "DiffuseAlbedo")
folder_specular_ms = osp(root_process, "{}", "SpecularUnlit")
folder_displacement_ms = osp(root_process, "{}", "Displacement")
# files multi scales [2^3 ... 2^10]
file_pointcloud_ms = osp(folder_pointcloud_ms, "*_*_*_pointcloud.exr")
file_pointcloud_neutral_ms = osp(folder_pointcloud_ms, "*_*_01_pointcloud.exr")
file_diffuse_ms = osp(folder_diffuse_ms, "*_*_*_diffuse_albedo.exr")
file_specular_ms = osp(folder_specular_ms, "*_*_*_specular_unlit.exr")
file_displacement_ms = osp(folder_displacement_ms, "*_*_*_displacement.exr")


# facewarehouse expression fitting 1024.
# e.g. 20191002_RyanWatson_01_blendshape_25_iter_15.mat
folder_exp_weights = osp(root_yajie, "RandmonMeshFacewarehouseAllExpressions")
folder_exp_pointcloud = osp(root_yajie, "RandmonMeshFacewarehouseAllExpressions_pointCloud")
folder_exp_pointcloud_ms = osp(root_process, "{}", "RandmonMeshFacewarehouseAllExpressions_pointCloud")

file_exp_weights = osp(folder_exp_weights, "*_*_*_blendshape_*_iter_*.mat")
file_exp_pointcloud = osp(folder_exp_pointcloud, "*_*_*_blendshape_*_iter_*_pointcloud.exr")
file_exp_pointcloud_ms = osp(folder_exp_pointcloud_ms, "*_*_*_blendshape_*_iter_*_pointcloud.exr")

file_exp_meanstd = osp(root_process, "weightsFacewareMeanStd.mat")


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
    imageio.imwrite(filename_out, img, format='EXR-FI')

    
def load_mat(filename, key):
    return scipy.io.loadmat(filename)[key]

def save_mat(filename_out, data, key, skip_if_exist=False):
    if skip_if_exist and os.path.exists(filename_out):
        return
    os.makedirs(os.path.dirname(filename_out), exist_ok=True)
    scipy.io.savemat(filename_out, {key: data})

    
#############################################################
# Specific Utils
#############################################################
def load_exp_mean():
    # return (25,)
    mean = load_mat(file_exp_meanstd, "Expression_mean")[0]
    return mean
    
def load_exp_std():
    # return (25,)
    std = load_mat(file_exp_meanstd, "Expression_std")[0]
    return std
    
    
#############################################################
# Preprocess
#############################################################
def load_and_save(file, resolutions, skip_if_exist=True):
    img = load_img(file)
    for resolution in resolutions:
        img_resized = cv2.resize(img, (resolution, resolution), interpolation=cv2.INTER_CUBIC)
        file = file.replace(root_raw, "")
        file = file.replace(root_yajie, "")
        file = osp(root_process, f"{resolution}/", file)
        save_img(file, img_resized, skip_if_exist)
        
    return True

def preprocess_multiscale(n_worker = 8, resolutions = [8, 16, 32, 64, 128, 256, 512, 1024], skip_if_exist=True):
    resize_fn = partial(load_and_save, resolutions=resolutions, skip_if_exist=skip_if_exist)
    
    files = []
#     files += glob.glob(file_pointcloud_4k)
#     files += glob.glob(file_diffuse_4k)
    files += glob.glob(file_exp_pointcloud)
#     files += glob.glob(file_specular_4k)
#     files += glob.glob(file_displacement_4k)
    
    with multiprocessing.Pool(n_worker) as pool:
        for _ in tqdm.tqdm(pool.imap_unordered(resize_fn, files)):
            pass
    
if __name__ == "__main__":
    preprocess_multiscale(n_worker=8, resolutions=[64], skip_if_exist=False)