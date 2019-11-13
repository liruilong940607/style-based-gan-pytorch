import argparse
import math
import tqdm

import torch
import torch.nn as nn
from torchvision import utils

from model import StyledGenerator, Monitor

from utils import *
from dataset import MultiResolutionDataset

def save_result(tensor, z, folder, ids, age=None, gender=None):
    os.makedirs(folder, exist_ok=True)
    if z is None:
        z = [None] * len(tensor)
    for img, latent, id in zip(tensor, z, ids):
        if latent is not None:
            torch.save({"z": latent, "age": age, "gender": gender, "noise": None}, 
                       f'{folder}/{id}.pkl')
        utils.save_image(
            exr2rgb(img[3:6]),
            f'{folder}/{id}_pointcloud.png',
            nrow=1,
            normalize=True,
            range=(0, 1),
        )
        utils.save_image(
            exr2rgb(img[0:3]),
            f'{folder}/{id}_albedo.png',
            nrow=1,
            normalize=True,
            range=(0, 1),
        )
        
        save_img(os.path.join(folder, f"{id}_pointcloud.exr"), To_numpy(img[3:6]))
        save_img(os.path.join(folder, f"{id}_albedo.exr"), To_numpy(img[0:3]))

    if len(tensor) > 1:
        utils.save_image(
            exr2rgb(tensor[:16, 0:3]),
            f'./sample_albedo.jpg',
            nrow=4,
            normalize=True,
            range=(0, 1),
        )
        utils.save_image(
            exr2rgb(tensor[:16, 3:6]),
            f'./sample_albedo.jpg',
            nrow=4,
            normalize=True,
            range=(0, 1),
        )

        
@torch.no_grad()
def get_mean_style(generator, device):
    mean_style = None

    for i in range(10):
        style = generator.mean_style(torch.randn(1024, 512).to(device))

        if mean_style is None:
            mean_style = style

        else:
            mean_style += style

    mean_style /= 10
    return mean_style

@torch.no_grad()
def sample(generator, step, mean_style, n_sample, device, style_weight=0.7, z=None, input_style=None):
    if input_style is None and z is None:
            z = torch.randn(n_sample, 512).to(device)
    image = generator(
        z,
        step=step,
        alpha=1,
        mean_style=mean_style,
        style_weight=style_weight,
        input_style=input_style,
    )
    
    return image, z

@torch.no_grad()
def style_mixing(generator, step, mean_style, n_source, n_target, device, processor):
    source_code = torch.randn(n_source, 512).to(device)
    target_code = torch.randn(n_target, 512).to(device)
    
    shape = 4 * 2 ** step
    alpha = 1

    images = [torch.zeros(1, 6, shape, shape).to(device)]

    source_image = processor.forward(generator(
        source_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    ))[0]
    target_image = processor.forward(generator(
        target_code, step=step, alpha=alpha, mean_style=mean_style, style_weight=0.7
    ))[0]

    images.append(source_image)

    for i in range(n_target):
        image = processor.forward(generator(
            [target_code[i].unsqueeze(0).repeat(n_source, 1), source_code],
            step=step,
            alpha=alpha,
            mean_style=mean_style,
            style_weight=1.0,
            mixing_range=(2, 4),
        ))[0]
        images.append(target_image[i].unsqueeze(0))
        images.append(image)

    images = torch.cat(images, 0)
    
    return images


class Processor():
    def __init__(self):
        dataset = MultiResolutionDataset(256)
        self.mean = dataset.mean.to('cuda')
        self.std = dataset.std.to('cuda')
        self.mean_pc = dataset.mean_pc_multires[256].to('cuda')
        self.mask = dataset.mask_multires[256].unsqueeze(0).to('cuda')
        self.mean_gender = dataset.mean_gender
        self.std_gender = dataset.std_gender
        self.mean_age = dataset.mean_age
        self.std_age = dataset.std_age
        
    def forward_age(self, age):
        age = (age * self.std_age) + self.mean_age
        return age
        
    def forward_gender(self, gender):
        gender = (gender * self.std_gender) + self.mean_gender
        return gender
    
    def forward(self, output, z=None, age=None, gender=None):
        output = (output * self.std + self.mean) * self.mask
        output[:, 3:6] += self.mean_pc
        if age is not None:
            age = (age * self.std_age) + self.mean_age
            gender = (gender * self.std_gender) + self.mean_gender
            return output, z, age, gender
        else:
            return output, z

@torch.no_grad()
def get_mean_style_gender(generator, monitor, step, device, processor):
    mean_style_male = []
    mean_style_female = []

    for i in tqdm.tqdm(range(10000)):
        z = torch.randn(1, 512).to(device)
        style = generator.mean_style(z)
        output = generator(
            z,
            step=step,
            alpha=1,
        )
        age, gender = monitor(output, step, alpha=1.0)
        gender = processor.forward_gender(gender)
        
        
        if gender < 0.2:
            mean_style_female.append(style)
        elif gender > 0.8:
            mean_style_male.append(style)
        else:
            continue
        
    mean_style_male = sum(mean_style_male) / len(mean_style_male)
    mean_style_female = sum(mean_style_female) / len(mean_style_female)
    
    return mean_style_male, mean_style_female


@torch.no_grad()
def get_mean_style_age(generator, monitor, step, device, processor):
    mean_style_age = {}
    counts = {}
    age_list = [20, 30, 35, 40, 50, 60]
    for age in age_list:#range(20, 81):
        mean_style_age[age] = []
        counts[age] = 0
    
    for i in tqdm.tqdm(range(200000)):
        z = torch.randn(1, 512).to(device)
        style = generator.mean_style(z)
        output = generator(
            z,
            step=step,
            alpha=1,
        )
        age, gender = monitor(output, step, alpha=1.0)
        age = processor.forward_age(age)
        
        age = age[0].long().item()
        for age_mean in age_list:
            if age < age_mean + 5 and age > age_mean - 5:
                mean_style_age[age_mean].append(style)
                counts[age_mean] += 1
                break
                
        print (counts)
    for age in age_list:
        mean_style_age[age] = sum(mean_style_age[age]) / counts[age]
        
    return mean_style_age
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256, help='size of the image')
    parser.add_argument('--N', type=int, default=10)
    parser.add_argument('--ckpt', type=str, default="checkpoint/train_resolution256_0xMonitor_iter-13999.model",
                        help='path to checkpoint file')
    parser.add_argument('--ckptMo', type=str, default="checkpoint/train_resolution256_Monitor_iter-19999.model",
                        help='path to checkpoint file')
    
    args = parser.parse_args()
    step = int(math.log(args.size, 2)) - 2
    
    device = 'cuda'
    
    monitor = Monitor(from_rgb_activate=True, in_channel=6).cuda()
    monitor.load_state_dict(torch.load(args.ckptMo)['monitor'])

    generator = StyledGenerator(512).to(device)
    #generator.load_state_dict(torch.load(args.path)['generator'])
    generator.load_state_dict(torch.load(args.ckpt)['g_running'])
    generator.eval()
    
    processor = Processor()
    
    # ---- mean_style ----
    mean_style = get_mean_style(generator, device)
#     style_weights = [i/30 for i in range(10)]
#     for style_weight in tqdm.tqdm(style_weights):
#         output = processor.forward(*generator(
#             torch.randn(1, 512).to(device),
#             step=step,
#             alpha=1,
#             mean_style=mean_style,
#             style_weight=style_weight,
#         ))
        
#         save_result(output, None, "./mean_style", [style_weight])
        
    # ---- sample ----
    # ---- age & gender prediction ----
#     for i in tqdm.tqdm(range(args.N)):
#         output, z = sample(generator, step, mean_style, 1, device)
#         age, gender = monitor(output, step, alpha=1.0)
#         output, z, age, gender = processor.forward(output, z, age, gender)
#         save_result(output, z, "./sample", [i], age, gender)
#         print (age, gender, i)

    # ---- transfer ----
#     img = style_mixing(generator, step, mean_style, args.N, args.N, device, processor)
#     utils.save_image(
#         exr2rgb(img[:, 0:3]),
#         f'./transfer.jpg',
#         nrow=args.N+1,
#         normalize=True,
#         range=(0, 1),
#     )
    
    # ---- interplation ----
#     output1, z1 = processor.forward(*sample(generator, step, mean_style, 1, device))
#     save_result(output1, z1, "./interplation", ["start"])
    
#     output2, z2 = processor.forward(*sample(generator, step, mean_style, 1, device))
#     save_result(output2, z2, "./interplation", ["end"])
    
#     delta = z2 - z1
#     for i in tqdm.tqdm(range(121)):
#         if i == 0:
#             continue
#         z = z1 + i/121.0 * delta
#         output, z = processor.forward(*sample(generator, step, mean_style, 1, device, z))
#         save_result(output, z, "./interplation", [i])

    # ---- mean gender ----
#     mean_style_male, mean_style_female = get_mean_style_gender(generator, monitor, step, device, processor)
#     torch.save({"male": mean_style_male, "female": mean_style_female}, "mean_style_gender.pkl")
    
    # ---- gender sample ----
#     mean_male = torch.load("mean_style_gender.pkl")["male"]
#     mean_female = torch.load("mean_style_gender.pkl")["female"]
#     for i in tqdm.tqdm(range(args.N)):
#         output, z = sample(generator, step, mean_male, 1, device, style_weight=0.2)
#         age, gender = monitor(output, step, alpha=1.0)
#         output, z, age, gender = processor.forward(output, z, age, gender)
#         save_result(output, z, "./sample_male", [i], age, gender)
#         print (gender, i)
        
#     for i in tqdm.tqdm(range(args.N)):
#         output, z = sample(generator, step, mean_female, 1, device, style_weight=0.2)
#         age, gender = monitor(output, step, alpha=1.0)
#         output, z, age, gender = processor.forward(output, z, age, gender)
#         save_result(output, z, "./sample_female", [i], age, gender)
#         print (gender, i)
    
    # ---- gender transfer ----
#     mean_male = torch.load("mean_style_gender.pkl")["male"]
#     mean_female = torch.load("mean_style_gender.pkl")["female"]
#     male2female = mean_female - mean_male
    
#     z = torch.randn(1, 512).to(device)
    
#     output1, z1 = processor.forward(*sample(generator, step, mean_male, 1, device, style_weight=0.0, z=z))
#     save_result(output1, z1, "./gender_interpolation", ["start_male"])
    
#     output2, z2 = processor.forward(*sample(generator, step, mean_female, 1, device, style_weight=0.0, z=z))
#     save_result(output2, z2, "./gender_interpolation", ["end_female"])
    
#     for i in tqdm.tqdm(range(121)):
#         if i == 0:
#             continue
#         mean_style_gender = mean_male + i/121.0 * male2female
#         output, _ = sample(generator, step, mean_style_gender, 1, device, style_weight=0.0, z=z)
#         age, gender = monitor(output, step, alpha=1.0)
#         print (gender)
#         output = processor.forward(output)[0]
#         save_result(output, None, "./gender_interpolation", [i])
    
    # ---- age
#     mean_style_age = get_mean_style_age(generator, monitor, step, device, processor)
#     torch.save({"age": mean_style_age}, "mean_style_age.pkl")

    # ---- age transfer ---
    keys =  [20, 60]
    for key in keys:
        mean_age = torch.load("mean_style_age.pkl")["age"][key]
        output1, z1 = processor.forward(*sample(generator, step, mean_age, 1, device, style_weight=0.0))
        save_result(output1, z1, "./age_interpolation", ["mean_{}".format(key)])
       
    z = torch.randn(1, 512).to(device)
    style_z = generator.mean_style(z)
    
    interpolations = []
    for idx in range(len(keys) - 1):
        key1 = keys[idx]
        key2 = keys[idx + 1]
        mean1 = torch.load("mean_style_age.pkl")["age"][key1]
        mean2 = torch.load("mean_style_age.pkl")["age"][key2]
        for i in range(0, 23, 1):
            mean = mean1 + i/50 * (mean2 - mean1)
            interpolations.append(mean)
        for i in range(230, 300, 2):
            i /= 10.0
            mean = mean1 + i/50 * (mean2 - mean1)
            interpolations.append(mean)
        for i in range(30, 50, 1):
            mean = mean1 + i/50 * (mean2 - mean1)
            interpolations.append(mean)
            
    cnt = 0
    for mean_style_age in tqdm.tqdm(interpolations):
        output, _ = sample(generator, step, style_z, 1, device, style_weight=0.3, input_style=mean_style_age)
        age, gender = monitor(output, step, alpha=1.0)
        print (processor.forward_age(age))
        output = processor.forward(output)[0]
        save_result(output, None, "./age_interpolation", [cnt])
        cnt += 1
    