import argparse
import random
import math

from tqdm import tqdm
import numpy as np
from PIL import Image

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from dataset import MultiResolutionDataset
from model import StyledGenerator, Discriminator, Monitor

import imageio
import random
from utils import *

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(dataset, batch_size, image_size=4):
    dataset = MultiResolutionDataset(image_size)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=16)

    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult
    
        
def exr2rgb(tensor):
    return (tensor*12.92) * (tensor<=0.0031308).float() + (1.055*(tensor.abs()**(1.0/2.4))-0.055) * (tensor>0.0031308).float()
    
    
def random_flip(tensor):
    inv_idx = torch.arange(tensor.size(3)-1, -1, -1).long().cuda()
    if random.random() < 0.5:
        albedo = tensor[:, 0:3, :, inv_idx]
    else:
        albedo = tensor[:, 0:3, :, :]
        
#     if random.random() < 0.5:
#         pointcloud = tensor[:, 3:6, :, inv_idx]
#         pointcloud = pointcloud * torch.tensor([[[[-1.0]], [[1.0]], [[1.0]]]], dtype=pointcloud.dtype).cuda()
#     else:
#         pointcloud = tensor[:, 3:6, :, :]
    pointcloud = tensor[:, 3:6, :, :]
    
    tensor = torch.cat([albedo, pointcloud], dim=1)
    return tensor
    
        
def train(args, dataset, monitor):    
    step = int(math.log2(args.init_size)) - 2
    resolution = 4 * 2 ** step
    loader = sample_data(
        dataset, args.batch.get(resolution, args.batch_default), resolution
    )
    data_loader = iter(loader)
    
    #new fix
    adjust_lr(d_optimizerMo, args.lr.get(resolution, 0.001))

    pbar = tqdm(range(3_000_000))

    requires_grad(monitor, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0

    alpha = 0
    used_sample = 0

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = True
    
    mean = dataset.mean.cuda().unsqueeze(0)
    std = dataset.std.cuda().unsqueeze(0)
    
    MSELoss = nn.MSELoss()
    loss_weight = 100
    
    mask = dataset.mask_multires[resolution].unsqueeze(0).cuda()
    for i in pbar:
        monitor.zero_grad()

        alpha = min(1, 1 / args.phase * (used_sample + 1))

        if (resolution == args.init_size and args.ckpt is None) or final_progress:
            alpha = 1

        if used_sample > args.phase * 2:
            used_sample = 0
            step += 1

            if step > max_step:
                step = max_step
                final_progress = True
                ckpt_step = step + 1

            else:
                alpha = 0
                ckpt_step = step

            resolution = 4 * 2 ** step

            loader = sample_data(
                dataset, args.batch.get(resolution, args.batch_default), resolution
            )
            data_loader = iter(loader)

            torch.save(
                {    
                    'monitor': monitor.module.state_dict(),
                    'd_optimizerMo': d_optimizerMo.state_dict(),
                },
                f'checkpoint/train_step-{ckpt_step}.model',
            )
            adjust_lr(d_optimizerMo, args.lr.get(resolution, 0.001))
            mask = dataset.mask_multires[resolution].unsqueeze(0).cuda()

        try:
            real_image, real_label_age, real_label_gender = next(data_loader)
            real_label = torch.cat([real_label_age, real_label_gender.float()], dim=1)

        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image, real_label_age, real_label_gender = next(data_loader)
            real_label = torch.cat([real_label_age, real_label_gender.float()], dim=1)
            
        used_sample += real_image.shape[0]

        b_size = real_image.size(0)
        real_image = real_image.cuda()
        real_image = real_image * mask
        real_image = random_flip(real_image)
        
        real_label_age = real_label_age.cuda()
        real_label_gender = real_label_gender.cuda()
        real_label = real_label.cuda()
        
        # update monitor
        real_age, real_gender = monitor(real_image, step=step, alpha=alpha)
        loss_monitor = MSELoss(real_age, real_label_age) + MSELoss(real_gender, real_label_gender)
        loss_monitor.backward()
        d_optimizerMo.step()
        
            
        if (i + 1) % 1000 == 0:
            torch.save(
                {
                    'monitor': monitor.module.state_dict(),
                    'd_optimizerMo': d_optimizerMo.state_dict(),
                },
                f'checkpoint/train_resolution{resolution}_Monitor_iter-{i}.model',
            )


        state_msg = (
            f'Size: {4 * 2 ** step}; Alpha: {alpha:.2f}; M-Real: {loss_monitor.item():.2f}'
        )

        pbar.set_description(state_msg)


if __name__ == '__main__':
    code_size = 512
    batch_size = 16
    n_critic = 1

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    parser.add_argument(
        '--phase',
        type=int,
        default=600_000,
        help='number of samples used for each training phases',
    )
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--init_size', default=256, type=int, help='initial image size')
    parser.add_argument('--max_size', default=256, type=int, help='max image size')
    parser.add_argument(
        '--ckpt', default=None, type=str, help='load from previous checkpoints'
    )
    parser.add_argument(
        '--ckptExp', default=None, type=str,
    )
    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )
    parser.add_argument(
        '--mixing', action='store_true', help='use mixing regularization'
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='wgan-gp',
        choices=['wgan-gp', 'r1'],
        help='class of gan loss',
    )

    args = parser.parse_args()

    monitor = nn.DataParallel(Monitor(from_rgb_activate=True, in_channel=6)).cuda()
#     ckpt = torch.load(args.ckptExp)
#     monitor.module.load_state_dict(ckpt['model'])
    
    d_optimizerMo = optim.Adam(monitor.parameters(), lr=args.lr, betas=(0.0, 0.99))

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)

        monitor.module.load_state_dict(ckpt['monitor'])
        d_optimizerMo.load_state_dict(ckpt['d_optimizerMo'])

    dataset = MultiResolutionDataset(args.init_size)

    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        # args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32}

#         # 1 GPU
#         args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 16, 128: 8, 256: 4}
#         args.phase = 300_000
        
        # 2 GPU
        args.batch = {4: 1024, 8: 512, 16: 256, 32: 128, 64: 32, 128: 32, 256: 16}
        args.phase = 300_000

#         # 4 GPU
#         args.batch = {4: 2048, 8: 1024, 16: 512, 32: 256, 64: 64, 128: 64, 256: 32}
#         args.phase = 300_000
        
#         # 6 GPU
#         args.batch = {4: 3072, 8: 1536, 16: 768, 32: 384, 64: 192, 128: 192, 256: 192}
#         args.phase = 1200_000
        
#         # 8 GPU
#         args.batch = {4: 4096, 8: 2048, 16: 1024, 32: 512, 64: 128, 128: 64, 256: 32, 512: 16, 1024: 8}
#         args.phase = 1200_000

    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {}

    args.batch_default = 16

    train(args, dataset, monitor)