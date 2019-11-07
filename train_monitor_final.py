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
from model import StyledGenerator, Monitor

import random
import time
import paths
from collections import deque
import imageio
import cv2


def train_monitorExp(model, resolution, batch_size):
    step = int(math.log2(resolution)) - 2
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.0, 0.99))
    L1loss = nn.L1Loss()
    MSEloss = nn.MSELoss()
    CEloss = nn.CrossEntropyLoss()
    
    dataset = MultiResolutionDataset(resolution)
    
    data_loader = iter(DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=16))
        
    img_mean, img_min, img_max = load_region_range()

    pbar = tqdm(range(20_000))
    for i in pbar:        
        tick = time.time()
        img, age, gender = next(data_loader)
        img = (img - img_mean) / (img_max - img_min + 1e-10)
        tock = time.time()

        image = img.cuda()
        age = age.view(-1, 1).cuda()
        gender = gender.cuda()
        age_predict, gender_predict = model(image, step=step, alpha=1.0)

        model.zero_grad()
        loss_age = MSEloss(age_predict, age)
        loss_gender = CEloss(gender_predict, gender)
        loss = loss_age + loss_gender
        loss.backward()
        optimizer.step()

        state_msg = (
            f'[MonitorExp] Size: {4 * 2 ** step}; LossAge: {loss_age.item():.3f}; LossGender: {loss_gender.item():.3f}; Data: {tock-tick:.3f};'
        )

        pbar.set_description(state_msg)
        
        if i%100 == 0:
            np.set_printoptions(precision=2, suppress=True)
            print (' ----------------------------- ')
            gender_predict = F.softmax(gender_predict, dim=1)
            print (f"Age: {age_predict[0, 0].item(): .2f}; "
                   f"GTAge: {age[0, 0].item(): .2f}; "
                   f"Gender: {gender_predict[0, 1].item(): .2f};"
                   f"GTGender: {gender[0]: .2f}")
        
        if i%1000 == 0:
            torch.save(
                {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                f'checkpoint/monitor-neutral/resolution-{resolution}-iter-{i}.model',
            )

    torch.save(
        {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        },
        f'checkpoint/monitor-neutral/resolution-{resolution}-iter-{i}.model',
    )
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


def calc_region_range(dataset):
    data_loader = iter(DataLoader(dataset, shuffle=False, batch_size=16, num_workers=16))
    img = torch.cat([next(data_loader)[0] for _ in tqdm(range(100))], dim=0)
    img_mean = img.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    img_min = img.min(dim=0, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    img_max = img.max(dim=0, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    print (img_max.shape)
    return img_mean, img_min, img_max
        
    
def load_region_range():
    norm_data = torch.load("./norm_data.data")
    img_mean = norm_data["img_mean"]
    img_min = norm_data["img_min"]
    img_max = norm_data["img_max"]
    return img_mean, img_min, img_max
    
    
# python trainxxx.py --trainExp 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Aux monitors')
    parser.add_argument('--trainExp', action='store_true')
    parser.add_argument('--testExp', action='store_true')
    parser.add_argument('--mean', action='store_true')
    args = parser.parse_args()
    
    args.batch = {64: 32, 128: 16, 256: 8, 512: 4, 1024: 2}
    
    if args.mean:
        dataset = MultiResolutionDataset(256)
        img_mean, img_min, img_max = calc_region_range(dataset)
        torch.save(
            {
                'img_mean': img_mean,
                'img_min': img_min,
                'img_max': img_max,
            },
            f'norm_data.data',
        )
        norm_data = torch.load("norm_data.data")
        print (norm_data["img_mean"])
        
    
    if args.trainExp:
        for step in [4]:
            resolution = 4 * 2 ** step
            monitorExp = nn.DataParallel(Monitor(from_rgb_activate=True, in_channel=6)).cuda()
#             ckpt = torch.load(f'checkpoint/monitorExp/resolution-{2 ** step}-iter-{8000}.model')
#             monitorExp.module.load_state_dict(ckpt['model'])
            batch_size = args.batch.get(resolution, 32) * 16
            monitorExp = train_monitorExp(monitorExp, resolution, batch_size)
    
    if args.testExp:
        test_monitorExp("./checkpoint/monitorExp/resolution-256-iter-8000.model", 256, 16)
        test_monitorExp("./checkpoint/monitorExp/resolution-64-iter-19999.model", 64, 16)
    