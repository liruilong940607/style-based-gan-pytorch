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
import imageio
import numpy as np
import glob
import imageio
import scipy.io
import cv2
import IPython
import os
import random
import torch

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
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)

    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult

        
def train(args, dataset, generator, discriminator, monitorExp):
    requires_grad(monitorExp, False)
    
    step = int(math.log2(args.init_size)) - 2
    resolution = 4 * 2 ** step
    loader = sample_data(
        dataset, args.batch.get(resolution, args.batch_default), resolution
    )
    data_loader = iter(loader)

    adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
    adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))

    pbar = tqdm(range(3_000_000))

    requires_grad(generator, False)
    requires_grad(discriminator, True)

    disc_loss_val = 0
    gen_loss_val = 0
    grad_loss_val = 0
    exp_loss_val = 0

    alpha = 0
    used_sample = 0

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = True
    
    #Exp
    mask = imageio.imread("/home/ICT2000/rli/mnt/glab2/ForRuilong/FaceEncoding_process2/mask_key.png")[:, :, 0]
    mask = cv2.resize(mask, (resolution, resolution), cv2.INTER_NEAREST)
    mask_eye = torch.from_numpy(np.logical_or(mask == 60, mask == 230)).float().unsqueeze(0).unsqueeze(0).cuda()
    mask_other = torch.from_numpy(np.logical_and(mask != 60, mask != 230)).float().unsqueeze(0).unsqueeze(0).cuda()
        
    for i in pbar:
        discriminator.zero_grad()

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
                    'generator': generator.module.state_dict(),
                    'discriminator': discriminator.module.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'g_running': g_running.state_dict(),
                },
                f'checkpoint/{resolution}_train_step-{ckpt_step}.model',
            )

            adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))
                
        try:
            real_image, real_label = next(data_loader)

        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image, real_label = next(data_loader)

        used_sample += real_image.shape[0]

        b_size = real_image.size(0)
        real_image = real_image.cuda()
        neutral = dataset.getitem_neutral(rand=True)
        neutral = neutral.unsqueeze(0).cuda()

        if args.loss == 'wgan-gp':
            real_predict = discriminator(real_image + neutral, step=step, alpha=alpha)
            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
            (-real_predict).backward()

        fake_label1 = dataset.sample_label(k=b_size, randn=True).cuda()
        fake_label2 = dataset.sample_label(k=b_size, randn=True).cuda()
        
#         rand_scale1 = torch.rand_like(fake_label1, device=fake_label1.device) * 0.2 + 0.9 
#         rand_scale2 = torch.rand_like(fake_label2, device=fake_label2.device) * 0.2 + 0.9 
        
#         fake_label1 *= rand_scale1
#         fake_label2 *= rand_scale2
        
        gen_in1 = fake_label1
        gen_in2 = fake_label2
        
        fake_image = generator(gen_in1, step=step, alpha=alpha)
        fake_predict = discriminator(fake_image + neutral, step=step, alpha=alpha)

        if args.loss == 'wgan-gp':
            fake_predict = fake_predict.mean()
            fake_predict.backward()

            eps = torch.rand(b_size, 1, 1, 1).cuda()
            x_hat = eps * real_image.data + (1 - eps) * fake_image.data
            x_hat.requires_grad = True
            hat_predict = discriminator(x_hat + neutral, step=step, alpha=alpha)
            grad_x_hat = grad(
                outputs=hat_predict.sum(), inputs=x_hat, create_graph=True
            )[0]
            grad_penalty = (
                (grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2
            ).mean()
            grad_penalty = 10 * grad_penalty
            grad_penalty.backward()
            grad_loss_val = grad_penalty.item()
            disc_loss_val = (real_predict - fake_predict).item()

        d_optimizer.step()

        if (i + 1) % n_critic == 0:
            generator.zero_grad()

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            fake_image = generator(gen_in2, step=step, alpha=alpha)
            predict = discriminator(fake_image + neutral, step=step, alpha=alpha)
            
            # monitor Exp
            img_eye = fake_image * mask_eye * 100
            img_other = fake_image * mask_other
            fake_image = torch.cat([img_eye, img_other], dim=1)
        
            predict_exp = monitorExp(fake_image, step=step, alpha=1.0)
            loss_exp = nn.MSELoss()(predict_exp, fake_label2.detach())
            
            if args.loss == 'wgan-gp':
                if i < 20000:
                    loss_weight = 10000
                else:
                    loss_weight = 10000
                    
                loss = -predict.mean() + loss_exp.mean() * loss_weight
                    
            elif args.loss == 'r1':
                loss = F.softplus(-predict).mean()

            gen_loss_val = (-predict.mean()).item()
            exp_loss_val = loss_exp.item()

            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator.module, 0)

            requires_grad(generator, False)
            requires_grad(discriminator, True)

        if (i + 1) % 200 == 0:
            nsample = 5
            with torch.no_grad():
                for isample in range(nsample):
                    label_code = real_label[isample:isample+1].cuda()
                    
                    image = g_running(label_code, step=step, alpha=alpha)
                    score = discriminator.module(image + neutral, step=step, alpha=alpha)
                    
                    img_eye = image * mask_eye * 100
                    img_other = image * mask_other
                    image_m = torch.cat([img_eye, img_other], dim=1)
                    weight = monitorExp.module(image_m, step=step, alpha=1.0)
                
                    image = image.data.cpu().numpy()[0].transpose(1, 2, 0)
                    image += neutral.data.cpu().numpy()[0].transpose(1, 2, 0)
                    np.set_printoptions(precision=2, suppress=True)
                    print (f"score: {score.item():.2f}; weight: {label_code.data.cpu().numpy()}; weight_pred: {weight.data.cpu().numpy()}")
                    imageio.imwrite(f'sample/{str(i + 1).zfill(6)}-{isample}--Fake.exr', image, format='EXR-FI')
                    save_mat(f'sample/{str(i + 1).zfill(6)}-{isample}--Fake.mat', image, "data")
                    
                    real = real_image.data.cpu().numpy()[isample].transpose(1, 2, 0)
                    real += neutral.data.cpu().numpy()[0].transpose(1, 2, 0)
                    imageio.imwrite(f'sample/{str(i + 1).zfill(6)}-{isample}--Real.exr', real, format='EXR-FI')
                    save_mat(f'sample/{str(i + 1).zfill(6)}-{isample}--Real.mat', real, "data")
            
        if (i + 1) % 500 == 0:
            torch.save(
                {
                    'generator': generator.module.state_dict(),
                    'discriminator': discriminator.module.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'g_running': g_running.state_dict(),
                },
                f'checkpoint/{resolution}_train_Offset_{loss_weight}xExp_iter-{i}.model',
            )

        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f}; Exp: {exp_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
        )

        pbar.set_description(state_msg)


if __name__ == '__main__':
    code_size = 512
    label_size = 25
    batch_size = 16
    n_critic = 1

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    parser.add_argument(
        '--phase',
        type=int,
        default=600_000,
        help='number of samples used for each training phases',
    )
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', default=True, help='use lr scheduling')
    parser.add_argument('--init_size', default=256, type=int, help='initial image size')
    parser.add_argument('--max_size', default=256, type=int, help='max image size')
    parser.add_argument(
        '--ckpt', default="./256_train_Offset_1000xExp_iter-8999.model", type=str, help='load from previous checkpoints'
    )
    parser.add_argument(
        '--ckptExp', default='./checkpoint/monitorExp/resolution-256-iter-23000.model', type=str,
    )
    parser.add_argument(
        '--no_from_rgb_activate',
        action='store_true',
        help='use activate in from_rgb (original implementation)',
    )
    parser.add_argument(
        '--mixing', action='store_true', default=False, help='use mixing regularization', 
    )
    parser.add_argument(
        '--loss',
        type=str,
        default='wgan-gp',
        choices=['wgan-gp', 'r1'],
        help='class of gan loss',
    )

    args = parser.parse_args()

    generator = nn.DataParallel(StyledGenerator(code_size, label_dim=label_size)).cuda()
    discriminator = nn.DataParallel(Discriminator(from_rgb_activate=not args.no_from_rgb_activate)).cuda()
    monitorExp = nn.DataParallel(Discriminator(from_rgb_activate=True, in_channel=6, out_channel=label_size)).cuda()
    ckpt = torch.load(args.ckptExp)
    monitorExp.module.load_state_dict(ckpt['model'])
    g_running = StyledGenerator(code_size, label_dim=label_size).cuda()
    g_running.train(False)

    class_loss = nn.CrossEntropyLoss()

    g_optimizer = optim.Adam(
        generator.module.generator.parameters(), lr=args.lr, betas=(0.0, 0.99)
    )
    g_optimizer.add_param_group(
        {
            'params': generator.module.style.parameters(),
            'lr': args.lr * 0.01,
            'mult': 0.01,
        }
    )
    d_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.99))

    accumulate(g_running, generator.module, 0)

    if args.ckpt is not None:
        ckpt = torch.load(args.ckpt)
        generator.module.load_state_dict(ckpt['generator'])
        discriminator.module.load_state_dict(ckpt['discriminator'])
        g_running.load_state_dict(ckpt['g_running'])
        g_optimizer.load_state_dict(ckpt['g_optimizer'])
        d_optimizer.load_state_dict(ckpt['d_optimizer'])

    transform = transforms.Compose(
        [
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )

    dataset = MultiResolutionDataset(resolution=args.init_size, exclude_neutral=True)

    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
#         # 1 GPU
#         args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32}
#         args.phase = 1200_000

        # 2 GPU
        args.batch = {4: 1024, 8: 512, 16: 256, 32: 128, 64: 64, 128: 64, 256: 64}
        args.phase = 1200_000

#         # 4 GPU
#         args.batch = {4: 2048, 8: 1024, 16: 512, 32: 256, 64: 128, 128: 128, 256: 128}
#         args.phase = 1200_000
        
#         # 6 GPU
#         args.batch = {4: 3072, 8: 1536, 16: 768, 32: 384, 64: 192, 128: 192, 256: 192}
#         args.phase = 1200_000
        
#         # 8 GPU
#         args.batch = {4: 4096, 8: 2048, 16: 1024, 32: 512, 64: 128, 128: 64, 256: 32, 512: 16, 1024: 8}
#         args.phase = 1200_000
    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    args.batch_default = 32

    train(args, dataset, generator, discriminator, monitorExp)
