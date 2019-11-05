import imageio
import os

from train import calc_region_range
from dataset import MultiResolutionDataset

def load_img(filename):
    return imageio.imread(filename, format='EXR-FI')

def save_img(filename_out, img, skip_if_exist=False):    
    if skip_if_exist and os.path.exists(filename_out):
        return
    os.makedirs(os.path.dirname(filename_out), exist_ok=True)
    imageio.imwrite(filename_out, img, format='EXR-FI')


dataset = MultiResolutionDataset("./lmdb", None, 64)
img_mean, img_min, img_max = calc_region_range(dataset)
img_mean = img_mean.numpy()[0, :, 0, 0]
img_min = img_min.numpy()[0, :, 0, 0]
img_max = img_max.numpy()[0, :, 0, 0]
print (f"[drange] min: {img_min}; max: {img_max}; mean: {img_mean}")

img = (load_img("./sample/024000-pointcloud.exr") + img_mean[0:3]) * (img_max[0:3] - img_min[0:3] + 1e-10)
save_img("./pointcloud.exr", img)


img = (load_img("./sample/024000-albedo.exr") + img_mean[3:6]) * (img_max[3:6] - img_min[3:6] + 1e-10)
save_img("./albedo.exr", img)