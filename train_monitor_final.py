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
from model import StyledGenerator, Discriminator

import random
import time
import paths
from collections import deque
import imageio
import cv2

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(1 - decay, par2[k].data)


def sample_data(dataset, batch_size, image_size=4):
    dataset.resolution = image_size
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=16)

    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult

def weighted_mse_loss(input, target, weight):
    return torch.mean(weight * (input - target) ** 2)

def train_monitorExp(model, resolution, batch_size):
    requires_grad(model, True)
    step = int(math.log2(resolution)) - 2
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.0, 0.99))
    L1loss = nn.L1Loss()
    MSEloss = nn.MSELoss()
    CEloss = nn.CrossEntropyLoss()
    
    dataset = MultiResolutionDataset(resolution, exclude_neutral=True)
    
    data_loader = iter(DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=16))
        
    mask = imageio.imread("/home/ICT2000/rli/mnt/glab2/ForRuilong/FaceEncoding_process2/mask_key.png")[:, :, 0]
    mask = cv2.resize(mask, (resolution, resolution), cv2.INTER_NEAREST)
    mask_eye = torch.from_numpy(np.logical_or(mask == 60, mask == 230)).float().unsqueeze(0).unsqueeze(0)
    mask_other = torch.from_numpy(np.logical_and(mask != 60, mask != 230)).float().unsqueeze(0).unsqueeze(0)
        
#     img_mean, img_min, img_max = calc_region_range(resolution)
#     print (img_mean.min(), img_mean.max())
    
    pbar = tqdm(range(40_000))
    errorQue = deque(maxlen=50)
    for i in pbar:        
#         neutral = dataset.getitem_neutral(rand=True)
#         neutral = neutral.unsqueeze(0).cuda()
        
        tick = time.time()
        img, label = next(data_loader)
#         img = (img - img_mean) / (img_max - img_min + 1e-10)
#         img = img * mask_eye * 100 + img * mask_other * 1 
        img_eye = img * mask_eye * 100
        img_other = img * mask_other
        img = torch.cat([img_eye, img_other], dim=1)
        tock = time.time()

        image = img.cuda()
        target = label.cuda()
        predict = model(image, step=step, alpha=1.0)
        
        weight = torch.ones((1, target.size(1)), dtype=torch.float32).cuda()
        if len(errorQue) == errorQue.maxlen:
            errorMean = torch.zeros((1, target.size(1)), dtype=torch.float32).cuda()
            for error in errorQue:
                errorMean += error / errorQue.maxlen
            errorMean = errorMean / errorMean.sum() * target.size(1)
            weight = errorMean

        model.zero_grad()
        loss = MSEloss(predict, target) # weighted_mse_loss(predict, target, weight)
        loss.backward()
        optimizer.step()

        errorQue.append((predict - target).abs().mean(dim=0, keepdim=True).detach())
        state_msg = (
            f'[MonitorExp] Size: {4 * 2 ** step}; Loss: {loss.item():.3f}; Data: {tock-tick:.3f};'
        )

        pbar.set_description(state_msg)
        
        if i%200 == 0:
            np.set_printoptions(precision=2, suppress=True)
            print (' ----------------------------- ')
            print (predict.data.cpu().numpy()[0])
            print (target.data.cpu().numpy()[0])
            print (weight.data.cpu().numpy()[0])
        
        if i%1000 == 0:
            torch.save(
                {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                f'checkpoint/monitorExp/resolution-{resolution}-iter-{i}.model',
            )

    torch.save(
        {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        },
        f'checkpoint/monitorExp/resolution-{resolution}-iter-{i}.model',
    )
    requires_grad(model, False)
    return model


def test_monitorExp(ckpt_path, resolution, batch_size):
    model = nn.DataParallel(Discriminator(from_rgb_activate=True, out_channel=25)).cuda()
    
    step = int(math.log2(resolution)) - 2
    
    ckpt = torch.load(ckpt_path)
    model.module.load_state_dict(ckpt['model'])
    
    MSEloss = nn.MSELoss()
    
    dataset = MultiResolutionDataset(resolution, exclude_neutral=True)
    
    data_loader = iter(DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=1))
    
    pbar = tqdm(range(40_000))
    loss = 0
    total = 0
    np.set_printoptions(precision=2, suppress=True)
    for i in pbar:        
        tick = time.time()
        img, label = next(data_loader)
        tock = time.time()

        image = img.cuda()
        target = label.cuda()
        predict = model(image, step=step, alpha=1.0)
        
        loss += (predict - target).abs().mean(dim=0).detach().data.cpu().numpy()
        total += 1

        print(
            f'[MonitorExp] Size: {4 * 2 ** step}; Accu: {loss/total}; Data: {tock-tick:.3f};'
        )

#         pbar.set_description(state_msg)


def calc_region_range(resolution):
    dataset = MultiResolutionDataset(resolution, exclude_neutral=True)
    data_loader = iter(DataLoader(dataset, shuffle=False, batch_size=2048, num_workers=16))
    img, label = next(data_loader)
    
    img_mean = img.mean(dim=0, keepdim=True)
    img_min = img.min(dim=0, keepdim=True)[0]
    img_max = img.max(dim=0, keepdim=True)[0]
    
    return img_mean, img_min, img_max
    
    
        
    
# python trainxxx.py --trainExp 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Aux monitors')
    parser.add_argument('--trainExp', action='store_true')
    parser.add_argument('--testExp', action='store_true')
    args = parser.parse_args()
    
    args.batch = {64: 32, 128: 16, 256: 8, 512: 4, 1024: 2}
    label_size = 25
    
    if args.trainExp:
        for step in [6]:
            resolution = 4 * 2 ** step
            monitorExp = nn.DataParallel(Discriminator(from_rgb_activate=True, in_channel=6, out_channel=label_size)).cuda()
#             ckpt = torch.load(f'checkpoint/monitorExp/resolution-{2 ** step}-iter-{8000}.model')
#             monitorExp.module.load_state_dict(ckpt['model'])
            batch_size = args.batch.get(resolution, 32) * 16
            monitorID = train_monitorExp(monitorExp, resolution, batch_size)
    
    if args.testExp:
        test_monitorExp("./checkpoint/monitorExp/resolution-256-iter-8000.model", 256, 16)
        test_monitorExp("./checkpoint/monitorExp/resolution-64-iter-19999.model", 64, 16)
    