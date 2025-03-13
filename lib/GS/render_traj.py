#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import sys
import imageio
import torch
import numpy as np
from PIL import Image
import cv2

import subprocess
cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

os.system('echo $CUDA_VISIBLE_DEVICES')

from scene import Scene
import json
import time
from gaussian_renderer import render
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.loss_utils import ssim
from utils.image_utils import psnr
from utils.camera_utils import cameraList_from_camInfos
from utils.system_utils import searchForMaxIteration
import math
from scene.dataset_readers import CameraInfo

to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def save_video(frames, save_path):
    if len(frames[0].shape) == 2:
        frames_np = [frame.cpu().numpy() for frame in frames]
    else:
        frames_np = [frame.permute(1, 2, 0).cpu().numpy() for frame in frames]
    imageio.mimwrite(save_path, to8b(frames_np), fps=60, quality=8)

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))



def readCamerasFromTransforms(path):
    cam_infos = []
    with open(os.path.join(path)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = None
        frames = contents["frames"]
        ct = 0
        w = 1920
        h = 1080
        # w=480
        # h=270
        image = Image.new('RGB', (w, h), (255,255,255))
        progress_bar = tqdm(frames, desc="Loading dataset")

        for idx, frame in enumerate(frames):
            # c2w = np.array(frame["transform_matrix"])
            c2w = np.array(frame["rot_mat"])
            
            ct += 1
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1
            if "small_city_img" in path:
                c2w[-1,-1] = 1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)

            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            if fovx is not None:
                fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
                FovY = fovy 
                FovX = fovx
            else:
                # given focal in pixel unit
                FovY = focal2fov(frame["fl_y"], image.size[1])
                FovX = focal2fov(frame["fl_x"], image.size[0])
            

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path="zhita", image_name=None, width=image.size[0], height=image.size[1]))
            
    return cam_infos
    
def render_set(dataset, name, iteration, gaussians, pipeline, background, json_path):
    model_path = dataset.model_path
    render_path = os.path.join(model_path, name, "renders")
    video_path = os.path.join(model_path, name, f"video")
    if not os.path.exists(render_path):
        os.makedirs(render_path)
    # if not os.path.exists(gts_path):
    #     os.makedirs(gts_path)
    if not os.path.exists(video_path):
        os.makedirs(video_path)

    name_list = []
    per_view_dict = {}
    # debug = 0
    t_list = []
    used_gaussian_number = []
    cam_infos = readCamerasFromTransforms(json_path)
    dataset.resolution = 1
    views = cameraList_from_camInfos(cam_infos, 1, dataset)
    rgb_frames = []
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        torch.cuda.synchronize(); t0 = time.time()
        rendering = render(view, gaussians, pipeline, background)["render"]
        torch.cuda.synchronize(); t1 = time.time()
        
        t_list.append(t1-t0)
        
        # vis = visualize(render_pkg, view, have_gt=False)
        # pano_fil_name = "{0:05d}".format(idx) + f"_{idx}.png"
        # vis_path = os.path.join(render_path, pano_fil_name)
        # torchvision.utils.save_image(vis, vis_path, nrow=3)

        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        rgb_frames.append(rendering)
        
    # save used gaussian number
    torch.cuda.empty_cache()
    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

    with open(os.path.join(model_path, name, "per_view_count.json"), 'w') as fp:
        json.dump(per_view_dict, fp, indent=True)     
    
    save_video(rgb_frames,os.path.join(video_path,'rgb.mp4'))
     
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, json_path : str, exp_name: str):
    with torch.no_grad():
        gaussians = GaussianModel(3)
        
        gaussians.load_ply(os.path.join(dataset.model_path,
                                        "point_cloud.ply"))
        
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        render_set(dataset, exp_name, iteration, gaussians, pipeline, background, json_path)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--traj_path", default='traj/zhita_trajectory.json',type=str)
    parser.add_argument("--exp_name", type=str, default='traj')
    # args = get_combined_args(parser)
    args = parser.parse_args(sys.argv[1:])
    print("Rendering " + args.model_path)
    # args.source_path = '/cpfs01/shared/pjlab-lingjun-landmarks/data/matrixcity_origin_version/small_city_img/new_high/all'
    args.data_device = 'cpu'
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.traj_path, args.exp_name)
    
