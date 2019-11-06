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


def train_monitorExp(model, resolution, batch_size):
    requires_grad(model, True)
    step = int(math.log2(resolution)) - 2
    
    optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=(0.0, 0.99))
    L1loss = nn.L1Loss()
    MSEloss = nn.MSELoss()
    CEloss = nn.CrossEntropyLoss()
    
    dataset = MultiResolutionDataset(resolution)
    
    data_loader = iter(DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=12))
        
    pbar = tqdm(range(20_000))
    for i in pbar:        
        tick = time.time()
        img, label = next(data_loader)
        tock = time.time()

        image = img.cuda()
        target = label.cuda().float()
        predict = model(image, step=step, alpha=1.0)
        
        model.zero_grad()
        loss = MSEloss(predict, target) 
        loss.backward()
        optimizer.step()
            
        state_msg = (
            f'[MonitorExp] Size: {4 * 2 ** step}; Loss: {loss.item():.3f}; Data: {tock-tick:.3f};'
        )

        pbar.set_description(state_msg)
        
        if i%200 == 0:
            np.set_printoptions(precision=2, suppress=True)
            print (' ----------------------------- ')
            print (predict.data.cpu().numpy()[0])
            print (target.data.cpu().numpy()[0])
        
        if i%200 == 0:
            torch.save(
                {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                f'checkpoint/monitorExp-25-Rand-step-{step}-iter-{i}.model',
            )

    torch.save(
        {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        },
        f'checkpoint/monitorExp-25-Rand-step-{step}-iter-{i}.model',
    )
    requires_grad(model, False)
    return model


def test_monitorExp(ckpt_path, resolution, batch_size):
    model = nn.DataParallel(Discriminator(from_rgb_activate=True, out_channel=9)).cuda()
    
    step = int(math.log2(resolution)) - 2
    
    ckpt = torch.load(ckpt_path)
    model.module.load_state_dict(ckpt['model'])
    
    MSEloss = nn.MSELoss()
    
    dataset = MultiResolutionDataset(resolution, sameID=False, exclude_neutral=True)
    
    data_loader = iter(DataLoader(dataset, shuffle=True, batch_size=int(batch_size/2), num_workers=1))
    
    pbar = tqdm(range(20_000))
    loss = 0
    total = 0
    for i in pbar:        
        neutral = dataset.getitem_neutral(rand=True)
        neutral = neutral.unsqueeze(0).cuda()

        tick = time.time()
        img, label = next(data_loader)
        tock = time.time()

        image = img.cuda()
        target = label.cuda()
        predict = model(image + neutral, step=step, alpha=1.0)
        
        loss += MSEloss(predict, target).item()
        total += 1

        state_msg = (
            f'[MonitorExp] Size: {4 * 2 ** step}; Accu: {loss/total:.3f}; Data: {tock-tick:.3f};'
        )

        pbar.set_description(state_msg)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Aux monitors')
    parser.add_argument('--trainID', action='store_true')
    parser.add_argument('--testID', action='store_true')
    parser.add_argument('--trainExp', action='store_true')
    parser.add_argument('--testExp', action='store_true')
    parser.add_argument('--resolution', type=int)
    args = parser.parse_args()
    
    args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32}
    
    if args.trainID:
        monitorID = nn.DataParallel(Discriminator(from_rgb_activate=True, 
                                                  in_channel=6, out_channel=2)).cuda()
        monitorID = train_monitorID(monitorID, resolution=64, batch_size=32)
        
    if args.testID:
        test_monitorID("./checkpoint/monitorID-step-4-iter-15000.model", 64, 32)
        
    if args.trainExp:
        label_size = 25
        monitorExp = nn.DataParallel(Discriminator(from_rgb_activate=True, out_channel=label_size)).cuda()
        batch_size = args.batch.get(args.resolution, 32) * 8
        monitorID = train_monitorExp(monitorExp, args.resolution, batch_size)
    
    if args.testExp:
        test_monitorExp("./checkpoint/save-monitorExp-9-MSE0.4-res64.model", 64, 32) # 0.036 # 0.050
#         test_monitorExp("./checkpoint/save-monitorExp-9-MSE0.6-res64-exN.model", 64, 32) # 0.053 # 0.058

    