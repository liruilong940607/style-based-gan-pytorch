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


def train_monitorID(model, resolution, batch_size):
    requires_grad(model, True)
    step = int(math.log2(resolution)) - 2
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.0, 0.99))
    CEloss = nn.CrossEntropyLoss()
    
    dataset_same = MultiResolutionDataset(resolution, sameID=True)
    dataset_diff = MultiResolutionDataset(resolution, sameID=False)
    
    loader_same = iter(DataLoader(dataset_same, shuffle=True, batch_size=batch_size, num_workers=1))
    loader_diff = iter(DataLoader(dataset_diff, shuffle=True, batch_size=batch_size, num_workers=1))
    
    pbar = tqdm(range(20_000))
    for i in pbar:
        flag_same = None
        if random.random() < 0.5:
            data_loader = loader_same
            flag_same = True
        else:
            data_loader = loader_diff
            flag_same = False
        
        tick = time.time()
        img1, _, img2, _ = next(data_loader)
        tock = time.time()

        image = torch.cat([img1, img2], dim=1).cuda()
        predict = model(image, step=step, alpha=1.0)
        if flag_same:
            target = torch.ones(batch_size, dtype=torch.long).cuda()
        else:
            target = torch.zeros(batch_size, dtype=torch.long).cuda()
        
        model.zero_grad()
        loss = CEloss(predict, target)
        loss.backward()
        optimizer.step()
            
        state_msg = (
            f'[MonitorID] Size: {4 * 2 ** step}; Loss: {loss.item():.3f}; Data: {tock-tick:.3f};'
        )

        pbar.set_description(state_msg)
        
        if i%5000 == 0:
            torch.save(
                {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                f'checkpoint/monitorID-step-{step}-iter-{i}.model',
            )

    torch.save(
        {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        },
        f'checkpoint/monitorID-step-{step}-iter-{i}.model',
    )
    requires_grad(model, False)
    return model

def test_monitorID(ckpt_path, resolution, batch_size):
    model = nn.DataParallel(Discriminator(from_rgb_activate=True, 
                                              in_channel=6, out_channel=2)).cuda()
    ckpt = torch.load(ckpt_path)
    model.module.load_state_dict(ckpt['model'])
    
    requires_grad(model, False)
    step = int(math.log2(resolution)) - 2
        
    dataset_same = MultiResolutionDataset(resolution, sameID=True)
    dataset_diff = MultiResolutionDataset(resolution, sameID=False)
    
    loader_same = iter(DataLoader(dataset_same, shuffle=True, batch_size=batch_size, num_workers=1))
    loader_diff = iter(DataLoader(dataset_diff, shuffle=True, batch_size=batch_size, num_workers=1))
    
    pbar = tqdm(range(1000))
    pos = 0
    total = 0
    for i in pbar:
        flag_same = None
        if random.random() < 0.5:
            data_loader = loader_same
            flag_same = True
        else:
            data_loader = loader_diff
            flag_same = False
        
        tick = time.time()
        img1, _, img2, _ = next(data_loader)
        tock = time.time()

        image = torch.cat([img1, img2], dim=1).cuda()
        predict = model(image, step=step, alpha=1.0)
        predict = F.softmax(predict, dim=1)
        predict = predict.argmax(dim=1)
        if flag_same:
            target = torch.ones(batch_size, dtype=torch.long).cuda()
        else:
            target = torch.zeros(batch_size, dtype=torch.long).cuda()
        
        pos += (predict == target).sum().item()
        total += batch_size

        state_msg = (
            f'[MonitorID] Size: {4 * 2 ** step}; Accu: {pos/total:.3f}; Data: {tock-tick:.3f};'
        )

        pbar.set_description(state_msg)

def train_monitorExp(model, resolution, batch_size):
    requires_grad(model, True)
    step = int(math.log2(resolution)) - 2
    
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.0, 0.99))
    MSEloss = nn.MSELoss()
    
    dataset = MultiResolutionDataset(resolution, sameID=False)
    
    data_loader = iter(DataLoader(dataset, shuffle=True, batch_size=int(batch_size/2), num_workers=1))
    
    pbar = tqdm(range(20_000))
    for i in pbar:        
        tick = time.time()
        img1, label1, img2, label2 = next(data_loader)
        tock = time.time()

        image = torch.cat([img1, img2], dim=0).cuda()
        target = torch.cat([label1, label2], dim=0).cuda()
        predict = model(image, step=step, alpha=1.0)
        
        model.zero_grad()
        loss = MSEloss(predict, target)
        loss.backward()
        optimizer.step()
            
        state_msg = (
            f'[MonitorExp] Size: {4 * 2 ** step}; Loss: {loss.item():.3f}; Data: {tock-tick:.3f};'
        )

        pbar.set_description(state_msg)
        
        if i%5000 == 0:
            torch.save(
                {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                },
                f'checkpoint/monitorExp-step-{step}-iter-{i}.model',
            )

    torch.save(
        {
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict(),
        },
        f'checkpoint/monitorExp-step-{step}-iter-{i}.model',
    )
    requires_grad(model, False)
    return model

def test_monitorExp(ckpt_path, resolution, batch_size):
    requires_grad(model, False)
    step = int(math.log2(resolution)) - 2
    
    ckpt = torch.load(ckpt_path)
    model.module.load_state_dict(ckpt['model'])
    
    MSEloss = nn.MSELoss()
    
    dataset = MultiResolutionDataset(resolution, sameID=False)
    
    data_loader = iter(DataLoader(dataset, shuffle=True, batch_size=int(batch_size/2), num_workers=1))
    
    pbar = tqdm(range(20_000))
    loss = 0
    total = 0
    for i in pbar:        
        tick = time.time()
        img1, label1, img2, label2 = next(data_loader)
        tock = time.time()

        image = torch.cat([img1, img2], dim=0).cuda()
        target = torch.cat([label1, label2], dim=0).cuda()
        predict = model(image, step=step, alpha=1.0)
        
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
    args = parser.parse_args()
    
    if args.trainID:
        monitorID = nn.DataParallel(Discriminator(from_rgb_activate=True, 
                                                  in_channel=6, out_channel=2)).cuda()
        monitorID = train_monitorID(monitorID, resolution=64, batch_size=32)
    
    if args.testID:
        test_monitorID("./checkpoint/monitorID-step-4-iter-15000.model", 64, 32)
        
    if args.trainExp:
        label_size = 25
        monitorExp = nn.DataParallel(Discriminator(from_rgb_activate=True, out_channel=label_size)).cuda()
        monitorExp = train_monitorExp(monitorExp, resolution=64, batch_size=32)
        
    if args.testExp:
        test_monitorExp("./checkpoint/monitorExp-step-4-iter-15000.model", 64, 32)

    