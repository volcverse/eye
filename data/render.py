import os
import math
import numpy as np
from tqdm import tqdm
from os import makedirs
import torch
import torch.nn as nn
import torchvision
from argparse import ArgumentParser
from typing import NamedTuple

from lib.GS.arguments import ParamGroup, PipelineParams, get_combined_args, ModelParams
from lib.GS.scene.gaussian_model import GaussianModel
from lib.GS.gaussian_renderer import render
from lib.GS.utils.general_utils import safe_state

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getWorld2View2_pytorch(R: torch.Tensor, t, translate=torch.tensor([.0, .0, .0]), scale=1.0):
    Rt = torch.zeros((4, 4), device=R.device)
    Rt[:3, :3] = R.t()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    translate = translate.to(R.device)
    C2W = torch.inverse(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.inverse(C2W)
    return Rt.float()

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class CameraInfo(NamedTuple):
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    width: int
    height: int

class Camera(nn.Module):
    def __init__(self, R, T, FoVx, FoVy, image_width, image_height,
                 trans=torch.Tensor([0.0, 0.0, 0.0])
                 ):
        super(Camera, self).__init__()
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_width = image_width
        self.image_height = image_height
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = trans
        # self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans)).transpose(0, 1).cuda()
        self.world_view_transform = getWorld2View2_pytorch(R, T, trans).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

def cameraList_from_camInfos(cam_infos, args):
    camera_list = []
    for id, cam_info in enumerate(cam_infos):
        camera_list.append(Camera(R=cam_info.R, T=cam_info.T, FoVx=cam_info.FovX, FoVy=cam_info.FovY))
    return camera_list

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class GaussianRender:
    def __init__(self, parser, sh_degree, gaussians_path, white_background, FOV, render_image_size=(1080, 1920)) -> None:
        self.gaussians = GaussianModel(sh_degree)
        self.gaussians.load_ply(gaussians_path)
        self.pipeline = PipelineParams(parser)
        bg_color = [1,1,1] if white_background else [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        # self.render_image_size = (810, 1440)
        # focal_length = 0.5 * 800 / math.tan(0.5 * FOV)
        # self.fovx = 2 * math.atan(0.5 * self.render_image_size[1] / focal_length)
        # self.fovy = 2 * math.atan(0.5 * self.render_image_size[0] / focal_length)
        self.render_image_size = render_image_size
        self.fovy = FOV
        self.fovx = FOV * self.render_image_size[1] / self.render_image_size[0]

    def render_from_view(self, blender_view: torch.Tensor):
        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = blender_view
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1
        # get the world-to-camera transform and set R, T
        w2c = c2w.inverse()
        R = w2c[:3,:3].t()  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        view = Camera(R=R, T=T, FoVx=self.fovx, FoVy=self.fovy, image_width=self.render_image_size[1], image_height=self.render_image_size[0])

        rendering = render(view, self.gaussians, self.pipeline, self.background)["render"]
        return rendering


if __name__ == "__main__":
    pass