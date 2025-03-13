import os
import sys
import math
import time
import argparse
from tqdm import tqdm
import cv2
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from lib.NTF.NTFnet import NTFNet
import torch.nn.functional as F
from model.metric import get_PSNR
from config.args import get_parser
# from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms as T
import wandb

def sort_key(s):
    # "pair0_x5.896_y45.396_z-24.465_left.jpg"
    pairs = s.split('_')
    index = int(pairs[0][4:])
    direction = int(pairs[-1][0])
    return (index, direction)

def get_dataset(transform, args):
    from data.dataset import SceneDataset

    data_prefix = os.path.join(os.getcwd(), args.data_path)
    images_path = os.listdir(args.data_path)
    images_path = sorted(images_path, key=sort_key)


    ds = SceneDataset(images_path=images_path,
                        data_prefix=data_prefix,
                        transform=transform,
                        train_NTF=args.train_NTF
                       )
    return ds

def get_transform(args):
    transforms = [
        T.Resize((args.image_height, args.image_width)),
        T.ToTensor(),
    ]
    return T.Compose(transforms)

def update_two_views(model:NTFNet, iteration, data, device):
    
    
    images, views = data
    images, views = images.to(device), views.to(device)
    exe_time = 0
    for i in range(iteration):
        start_time = time.time()
        model.update(views, images)
        exe_time += time.time() - start_time

    start_time = time.time()
    results, masks = model.getResults(views, images)

    psnr = get_PSNR(F.mse_loss(results, images))
    ssim = model.ssim_calc(results, images)
    exe_time += time.time() - start_time

    
    return psnr,ssim,exe_time

def train_ntf(args):

    dataset = get_dataset(transform=get_transform(args), args=args)
    device = 'cuda:0'

    view_ratio_ = 1920 / 518.4 * 60
    model = NTFNet(args=args, view_ratio = (view_ratio_, view_ratio_), device=device)
    
    psnr_ls = list()
    ssim_ls = list()
    all_num = args.ntf_num
    iteration = args.ntf_iteration
    pbar = tqdm(range(all_num), desc="Processing", unit="iteration")
    for dataidx in pbar:
        # train
        data = dataset.__getitem__(dataidx)
        psnr, ssim, time = update_two_views(model=model, iteration=iteration, 
                                            data=data, device=device)
        psnr_ls.append(psnr)
        ssim_ls.append(ssim)
        pbar.set_postfix(PSNR=f"{psnr:.4f}", SSIM=f"{ssim:.4f}", Time=f"{time:.4f}s")

    psnr_t = torch.tensor(psnr_ls)
    ssim_t = torch.tensor(ssim_ls)


    return  (torch.mean(psnr_t), torch.std(psnr_t)),\
            (torch.mean(ssim_t), torch.std(ssim_t))


        

from tqdm import trange

if __name__ == '__main__':
    

    

    parser = get_parser()
    args = parser.parse_args()

    args.paper_iteration = 5
    args.paper_all_num = 200
    train_ntf(args)
   