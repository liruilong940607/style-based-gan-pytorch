import argparse
import math
import os
import numpy as np
import scipy.io
import imageio
import shutil
import tqdm

import torch
from torchvision import utils

from model import StyledGenerator
from dataset import MultiResolutionDataset

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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256, help='size of the image')
    parser.add_argument('--dim', type=int, default=None)
    parser.add_argument('--N', type=int, default=5)
    parser.add_argument('--ckpt', type=str, default="./256_train_Offset_10000xExp_iter-10999.model")
    
    args = parser.parse_args()
    
    device = 'cuda'
    code_size = 512
    label_size = 25
    step = int(math.log(args.size, 2)) - 2
    
    generator = StyledGenerator(code_size, label_dim=label_size).to(device)
    generator.load_state_dict(torch.load(args.ckpt)['g_running'])
    generator.eval()

    dataset = MultiResolutionDataset(resolution=args.size, exclude_neutral=True)
    neutral = dataset.getitem_neutral(rand=True)
    neutral = neutral.unsqueeze(0).cuda()
    mean = torch.from_numpy(dataset.labels_mean).float()
    
    if args.dim is None:
        dims = list(range(label_size))
    else:
        dims = [args.dim]

    for dim in tqdm.tqdm(dims):
        sample_dir = f"./sample/resolution-{args.size}/dim-{dim}/"
        os.makedirs(sample_dir, exist_ok=True)
    
        std = dataset.labels_std[dim] * 5
        gen_in = torch.ones((1, label_size), dtype=torch.float32) * mean
        values = np.linspace(-std, std, args.N)
        for i in range(args.N):
            gen_in[0, dim] = values[i]
            gen_in = gen_in.to(device)
            print (f"value:{values[i]}, dim:{dim}, gen_in: {gen_in}")

            fake_image = generator(gen_in, step=step, alpha=1.0)
            fake_image = fake_image + neutral

            fake_image = To_numpy(fake_image[0])
            save_mat(os.path.join(sample_dir, f'{i}.mat'), fake_image, "data")
            save_img(os.path.join(sample_dir, f'{i}.exr'), fake_image)
        
    shutil.make_archive(f"./sample/resolution-{args.size}", 'zip', f"./sample/resolution-{args.size}/")