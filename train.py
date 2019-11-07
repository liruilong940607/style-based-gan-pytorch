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

        
def load_region_range():
    norm_data = torch.load("./norm_data.data")
    img_mean = norm_data["img_mean"]
    img_min = norm_data["img_min"]
    img_max = norm_data["img_max"]
    return img_mean, img_min, img_max
    
        
def exr2rgb(tensor):
    return (tensor*12.92) * (tensor<=0.0031308).float() + (1.055*(tensor**(1.0/2.4))-0.055) * (tensor>0.0031308).float()
    
        
def train(args, dataset, generator, discriminator, monitor):
    requires_grad(monitor, False)
    
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

    alpha = 0
    used_sample = 0

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = False
    
    img_mean, img_min, img_max = load_region_range()
    print (f"[drange] min: {img_min}; max: {img_max}; mean: {img_mean}")
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
                f'checkpoint/train_step-{ckpt_step}.model',
            )

            adjust_lr(g_optimizer, args.lr.get(resolution, 0.001))
            adjust_lr(d_optimizer, args.lr.get(resolution, 0.001))
            

        try:
            real_image, real_label_age, real_label_gender = next(data_loader)
            real_label = torch.cat([real_label_age, F.one_hot(real_label_gender).float()], dim=1)

        except (OSError, StopIteration):
            data_loader = iter(loader)
            real_image, real_label_age, real_label_gender = next(data_loader)
            real_label = torch.cat([real_label_age, F.one_hot(real_label_gender).float()], dim=1)
            
        real_image = (real_image - img_mean) / (img_max - img_min + 1e-10)

        used_sample += real_image.shape[0]

        b_size = real_image.size(0)
        real_image = real_image.cuda()

        if args.loss == 'wgan-gp':
            real_predict = discriminator(real_image, step=step, alpha=alpha)
            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
            (-real_predict).backward()

        elif args.loss == 'r1':
            real_image.requires_grad = True
            real_scores = discriminator(real_image, step=step, alpha=alpha)
            real_predict = F.softplus(-real_scores).mean()
            real_predict.backward(retain_graph=True)

            grad_real = grad(
                outputs=real_scores.sum(), inputs=real_image, create_graph=True
            )[0]
            grad_penalty = (
                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()
            grad_penalty = 10 / 2 * grad_penalty
            grad_penalty.backward()
            grad_loss_val = grad_penalty.item()

        if args.mixing and random.random() < 0.9:
            gen_in11, gen_in12, gen_in21, gen_in22 = torch.randn(
                4, b_size, code_size, device='cuda'
            ).chunk(4, 0)
            gen_in1 = [gen_in11.squeeze(0), gen_in12.squeeze(0)]
            gen_in2 = [gen_in21.squeeze(0), gen_in22.squeeze(0)]

        else:
            gen_in1, gen_in2 = torch.randn(2, b_size, code_size, device='cuda').chunk(
                2, 0
            )
            gen_in1 = gen_in1.squeeze(0)
            gen_in2 = gen_in2.squeeze(0)

        fake_label_age1, fake_label_gender1 = dataset.sample_label(k=b_size, randn=True)
        fake_label_age1 = fake_label_age1.cuda()
        fake_label_gender1 = fake_label_gender1.cuda()
        fake_label1 = torch.cat([fake_label_age1, F.one_hot(fake_label_gender1).float()], dim=1)
        fake_label_age2, fake_label_gender2 = dataset.sample_label(k=b_size, randn=True)
        fake_label_age2 = fake_label_age2.cuda()
        fake_label_gender2 = fake_label_gender2.cuda()
        fake_label2 = torch.cat([fake_label_age2, F.one_hot(fake_label_gender2).float()], dim=1)
            
        fake_image = generator(gen_in1, fake_label1, step=step, alpha=alpha)
        fake_predict = discriminator(fake_image, step=step, alpha=alpha)

        if args.loss == 'wgan-gp':
            fake_predict = fake_predict.mean()
            fake_predict.backward()

            eps = torch.rand(b_size, 1, 1, 1).cuda()
            x_hat = eps * real_image.data + (1 - eps) * fake_image.data
            x_hat.requires_grad = True
            hat_predict = discriminator(x_hat, step=step, alpha=alpha)
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

        elif args.loss == 'r1':
            fake_predict = F.softplus(fake_predict).mean()
            fake_predict.backward()
            disc_loss_val = (real_predict + fake_predict).item()

        d_optimizer.step()

        if (i + 1) % n_critic == 0:
            generator.zero_grad()

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            fake_image = generator(gen_in2, fake_label2, step=step, alpha=alpha)

            predict = discriminator(fake_image, step=step, alpha=alpha)

#             # monitor
#             age_predict, gender_predict = monitor(fake_image, step=step, alpha=1.0)
#             loss_age = nn.MSELoss()(age_predict, fake_label_age2)
#             loss_gender = nn.CrossEntropyLoss()(gender_predict, fake_label_gender2)

            
            if args.loss == 'wgan-gp':
                loss_weight = 0
                loss = -predict.mean() #+ (loss_age + loss_gender) * loss_weight

            elif args.loss == 'r1':
                loss = F.softplus(-predict).mean()

            gen_loss_val = -predict.mean().item()

            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator.module, 0)

            requires_grad(generator, False)
            requires_grad(discriminator, True)

        if (i + 1) % 200 == 0:
            real_image = real_image * (img_max.cuda() - img_min.cuda() + 1e-10) + img_mean.cuda()
            
            vis_real = exr2rgb(real_image[:16, 0:3])

            utils.save_image(
                vis_real.data.cpu(),
                f'sample/{str(i + 1).zfill(6)}-real_image.jpg',
                nrow=4,
                normalize=True,
                range=(0, 1),
            )
            
            fake_image = fake_image * (img_max.cuda() - img_min.cuda() + 1e-10) + img_mean.cuda()

            vis_fake = exr2rgb(fake_image[:16, 0:3])

            utils.save_image(
                vis_fake.data.cpu(),
                f'sample/{str(i + 1).zfill(6)}-fake_image.jpg',
                nrow=4,
                normalize=True,
                range=(0, 1),
            )
            
            image = fake_image.data.cpu().numpy()[0].transpose(1, 2, 0)
            imageio.imwrite(f'sample/{str(i + 1).zfill(6)}-pointcloud.exr', image[:, :, 3:6], format='EXR-FI')
            imageio.imwrite(f'sample/{str(i + 1).zfill(6)}-albedo.exr', image[:, :, 0:3], format='EXR-FI')
                    

        if (i + 1) % 1000 == 0:
            torch.save(
                {
                    'generator': generator.module.state_dict(),
                    'discriminator': discriminator.module.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'g_running': g_running.state_dict(),
                },
                f'checkpoint/train_resolution{resolution}_{loss_weight}xMonitor_iter-{i}.model',
            )


        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f}; Alpha: {alpha:.5f}'
            #f' Age: {loss_age.item():.3f}; Gender: {loss_gender.item():.3f};'
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
    parser.add_argument('--init_size', default=64, type=int, help='initial image size')
    parser.add_argument('--max_size', default=256, type=int, help='max image size')
    parser.add_argument(
        '--ckpt', default=None, type=str, help='load from previous checkpoints'
    )
    parser.add_argument(
        '--ckptExp', default='./checkpoint/monitor-neutral/resolution-64-iter-8000.model', type=str,
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

    generator = nn.DataParallel(StyledGenerator(code_size, out_channel=6)).cuda()
    discriminator = nn.DataParallel(
        Discriminator(from_rgb_activate=not args.no_from_rgb_activate, in_channel=6)
    ).cuda()
    
    monitor = nn.DataParallel(Monitor(from_rgb_activate=True, in_channel=6)).cuda()
    ckpt = torch.load(args.ckptExp)
    monitor.module.load_state_dict(ckpt['model'])
    
    g_running = StyledGenerator(code_size, out_channel=6).cuda()
    g_running.train(False)

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

    dataset = MultiResolutionDataset(args.init_size)

    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        # args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32}
        
        # 2 GPU
        args.batch = {4: 1024, 8: 512, 16: 256, 32: 128, 64: 64, 128: 32, 256: 16}
        args.phase = 600_000

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

    args.gen_sample = {}

    args.batch_default = 16

    train(args, dataset, generator, discriminator, monitor)