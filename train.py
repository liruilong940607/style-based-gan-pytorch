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
    dataset.resolution = image_size
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=16)

    return loader


def adjust_lr(optimizer, lr):
    for group in optimizer.param_groups:
        mult = group.get('mult', 1)
        group['lr'] = lr * mult

        
def train(args, dataset, generator, discriminator):
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
    label_loss_val = 0

    alpha = 0
    used_sample = 0

    max_step = int(math.log2(args.max_size)) - 2
    final_progress = False

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
            if label_size == 0:
                raise NotImplementedError
                real_image = next(data_loader)
            else:
                real_image, real_label = next(data_loader)

        except (OSError, StopIteration):
            data_loader = iter(loader)
            if label_size == 0:
                raise NotImplementedError
                real_image = next(data_loader)
            else:
                real_image, real_label = next(data_loader)
                
        assert real_label is not None

        used_sample += real_image.shape[0]

        b_size = real_image.size(0)
        real_image = real_image.cuda()
        real_label = real_label.cuda()

        if args.loss == 'wgan-gp':
            real_predict, real_predictL = discriminator(real_image, real_label, step=step, alpha=alpha)
            real_predict = real_predict.mean() - 0.001 * (real_predict ** 2).mean()
            real_predictL = MSELoss(real_predictL, real_label)
            (-real_predict + real_predictL).backward()

        elif args.loss == 'r1': # Can't use. Not Implement Conditional
            raise NotImplementedError
            real_image.requires_grad = True
            real_scores = discriminator(real_image, label, step=step, alpha=alpha)
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

        # hard code. label range is [0, 6.8], centralize at 0.
        fake_label1, fake_label2 = torch.randn(2, b_size, label_size, device='cuda').clamp(0, 1).chunk(
            2, 0
        )
        fake_label1 = fake_label1.squeeze(0)
        fake_label2 = fake_label2.squeeze(0)
            
        fake_image = generator(gen_in1, fake_label1, step=step, alpha=alpha)
        fake_predict, fake_predictL = discriminator(fake_image, fake_label1, step=step, alpha=alpha)

        if args.loss == 'wgan-gp':
            fake_predict = fake_predict.mean()
            fake_predictL = MSELoss(fake_predictL, fake_label1)
            (fake_predict + fake_predictL).backward()

            eps = torch.rand(b_size, 1, 1, 1).cuda()
            x_hat = eps * real_image.data + (1 - eps) * fake_image.data
            x_hat.requires_grad = True
            hat_predict, _ = discriminator(x_hat, fake_label1, step=step, alpha=alpha)
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

        elif args.loss == 'r1': # Can't use. Not Implement Conditional
            raise NotImplementedError
            fake_predict = F.softplus(fake_predict).mean()
            fake_predict.backward()
            disc_loss_val = (real_predict + fake_predict).item()

        d_optimizer.step()

        if (i + 1) % n_critic == 0:
            generator.zero_grad()

            requires_grad(generator, True)
            requires_grad(discriminator, False)

            fake_image = generator(gen_in2, fake_label2, step=step, alpha=alpha)

            predict, predictL = discriminator(fake_image, fake_label2, step=step, alpha=alpha)

            if args.loss == 'wgan-gp':
                predict = predict.mean()
                predictL = MSELoss(predictL, fake_label2)
                loss = (-predict) + predictL

            elif args.loss == 'r1': # Can't use. Not Implement Conditional
                raise NotImplementedError
                loss = F.softplus(-predict).mean()

            gen_loss_val = (-predict).item()

            loss.backward()
            g_optimizer.step()
            accumulate(g_running, generator.module)

            requires_grad(generator, False)
            requires_grad(discriminator, True)

        if (i + 1) % 500 == 0:
            gen_i, gen_j = args.gen_sample.get(resolution, (5, 1))
            latent_code = torch.randn(gen_j, code_size).cuda()
            with torch.no_grad():
                for idx in range(gen_i):
                    label_code = torch.zeros(gen_j, label_size).cuda()
                    label_code[:, 24] = 0.2 * (idx + 1)
                    image = g_running(
                        latent_code, label_code, step=step, alpha=alpha
                    ).data.cpu().numpy()[0].transpose(1, 2, 0)
                    imageio.imwrite(f'sample/{str(i + 1).zfill(6)}-jawOpen-{0.2 * (idx + 1):.2f}.exr', image, format='EXR-FI')
                    
                    label_code = torch.zeros(gen_j, label_size).cuda()
                    label_code[:, 26] = 0.2 * (idx + 1)
                    image = g_running(
                        latent_code, label_code, step=step, alpha=alpha
                    ).data.cpu().numpy()[0].transpose(1, 2, 0)
                    imageio.imwrite(f'sample/{str(i + 1).zfill(6)}-mouthClose-{0.2 * (idx + 1):.2f}.exr', image, format='EXR-FI')

        if (i + 1) % 1000 == 0:
            torch.save(
                {
                    'generator': generator.module.state_dict(),
                    'discriminator': discriminator.module.state_dict(),
                    'g_optimizer': g_optimizer.state_dict(),
                    'd_optimizer': d_optimizer.state_dict(),
                    'g_running': g_running.state_dict(),
                },
                f'checkpoint/train_iter-{i}.model',
            )

        label_loss_val = (real_predictL + fake_predictL + predictL).item() / 3
            
        state_msg = (
            f'Size: {4 * 2 ** step}; G: {gen_loss_val:.3f}; D: {disc_loss_val:.3f};'
            f' Grad: {grad_loss_val:.3f} Label: {label_loss_val:.3f}; Alpha: {alpha:.5f}'
        )

        pbar.set_description(state_msg)


if __name__ == '__main__':
    code_size = 512
    label_size = 51
    batch_size = 16
    n_critic = 1

    parser = argparse.ArgumentParser(description='Progressive Growing of GANs')

    parser.add_argument('path', type=str, help='path of specified dataset')
    parser.add_argument(
        '--phase',
        type=int,
        default=600_000,
        help='number of samples used for each training phases',
    )
    parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('--sched', action='store_true', help='use lr scheduling')
    parser.add_argument('--init_size', default=8, type=int, help='initial image size')
    parser.add_argument('--max_size', default=1024, type=int, help='max image size')
    parser.add_argument(
        '--ckpt', default=None, type=str, help='load from previous checkpoints'
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

    generator = nn.DataParallel(StyledGenerator(code_size, label_size)).cuda()
    discriminator = nn.DataParallel(
        Discriminator(label_size, from_rgb_activate=not args.no_from_rgb_activate)
    ).cuda()
    g_running = StyledGenerator(code_size, label_size).cuda()
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
    
    MSELoss = nn.MSELoss()

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

    dataset = MultiResolutionDataset(args.path, transform)

    if args.sched:
        args.lr = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        args.batch = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32, 128: 32, 256: 32}

    else:
        args.lr = {}
        args.batch = {}

    args.gen_sample = {512: (8, 4), 1024: (4, 2)}

    args.batch_default = 32

    train(args, dataset, generator, discriminator)
