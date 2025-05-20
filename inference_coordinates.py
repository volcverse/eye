import sys
import os
import torch
import torch.utils.data
import argparse
from data.render import GaussianRender
from PIL import Image
from torchvision import transforms as T
from model.network import EyeRealNet
import math
from tqdm import tqdm
import cv2
import warnings
from config.scene_dict import *
from config.args import get_gaussian_parser
from data.dataset import eye2world_pytroch
from train_eyeReal import init_scene_args
warnings.filterwarnings('ignore')


def convert_RGB(img):
    width = img.width
    height = img.height

    image = Image.new('RGB', size=(width, height), color=(255, 255, 255))
    image.paste(img, (0, 0), mask=img)

    return image
def get_transform(args):
    transforms = [
        T.Resize((args.image_height, args.image_width)),
        T.ToTensor(),
    ]

    return T.Compose(transforms)


def get_eyes_from_mid(mid_coord, vertical, scale_physical2world, orientation):
    mid_coord_t = torch.tensor(mid_coord)
    R = mid_coord_t.norm()
    eye_distance = scale_physical2world * 6

    if vertical == "x":
        phi = torch.arccos(mid_coord[0] / R)
        d = R*torch.sin(phi)
        theta = torch.arccos(mid_coord[1] / d)
        delta = torch.arcsin(eye_distance/2/d)
        right_eye = [mid_coord[0], (d*math.cos(theta-delta)).item(), (d*math.sin(theta-delta)).item()]
        left_eye = [mid_coord[0], (d*math.cos(theta+delta)).item(), (d*math.sin(theta+delta)).item()]
    elif vertical == "y":
        phi = torch.arccos(mid_coord[1] / R)
        d = R*torch.sin(phi)
        if orientation == "xoy":
            theta = torch.arccos(mid_coord[0] / d)
            delta = torch.arcsin(eye_distance/2/d)
            right_eye = [(d*math.cos(theta-delta)).item(), mid_coord[1], (d*math.sin(theta-delta)).item()]
            left_eye = [(d*math.cos(theta+delta)).item(), mid_coord[1], (d*math.sin(theta+delta)).item()]
        elif orientation == "zox":
            theta = torch.arccos(mid_coord[2] / d)
            delta = torch.arcsin(eye_distance/2/d)
            right_eye = [(d*math.sin(theta-delta)).item(), mid_coord[1], (d*math.cos(theta-delta)).item()]
            left_eye = [(d*math.sin(theta+delta)).item(), mid_coord[1], (d*math.cos(theta+delta)).item()]
    elif vertical == "z":
        phi = torch.arccos(mid_coord[2] / R)
        d = R*torch.sin(phi)
        if orientation == "xoz":
            theta = torch.arccos(mid_coord[0] / d)
            delta = torch.arcsin(eye_distance/2/d)
            right_eye = [(d*math.cos(theta-delta)).item(), (d*math.sin(theta-delta)).item(), mid_coord[2]]
            left_eye = [(d*math.cos(theta+delta)).item(), (d*math.sin(theta+delta)).item(), mid_coord[2]]
        elif orientation == "yoz":
            theta = torch.arccos(mid_coord[2] / d)
            delta = torch.arcsin(eye_distance/2/d)
            right_eye = [(d*math.sin(theta-delta)).item(),(d*math.cos(theta-delta)).item(),  mid_coord[2]]
            left_eye = [(d*math.sin(theta+delta)).item(), (d*math.cos(theta+delta)).item(), mid_coord[2]]
    
    return [left_eye, right_eye]




def get_imgs(gaussian_path, FOV, view_l, view_r):
    render = GaussianRender(
        parser=get_gaussian_parser(),
        sh_degree=3, 
        gaussians_path=gaussian_path,
        white_background=True, FOV=FOV)
    eye1_img = render.render_from_view(view_l.cuda())
    eye2_img = render.render_from_view(view_r.cuda())
    return (eye1_img, eye2_img)
    

def load_inference(mid_coord, eyeRealNet_weights, save_path, gaussian_path, args):
    
    
    mid_coord = [mid_coord[0]*args.scale_physical2world, mid_coord[1]*args.scale_physical2world,mid_coord[2]*args.scale_physical2world]
    
    os.makedirs(save_path, exist_ok=True)


    FOV = args.FOV
    if FOV > math.pi: 
        FOV = FOV / 180 * math.pi
    model = EyeRealNet(args=args, FOV=FOV)
    model.load_state_dict(torch.load(eyeRealNet_weights, map_location='cpu')['model'])
    model = model.cuda()
    model.eval()
    transform = get_transform(args=args)
    # image_paths = ['view1.png', 'view2.png']
    delta = torch.tensor([args.delta_x, args.delta_y, args.delta_z])
    left_eye, right_eye = get_eyes_from_mid(mid_coord=mid_coord, vertical=args.vertical, scale_physical2world=args.scale_physical2world, orientation=args.orientation)
    left_eye_t = torch.tensor(left_eye)
    right_eye_t = torch.tensor(right_eye)
    view_l = eye2world_pytroch(args.vertical, left_eye_t, delta)
    view_r = eye2world_pytroch(args.vertical, right_eye_t, delta)
    img_l, img_r = get_imgs(gaussian_path=gaussian_path, FOV=FOV, view_l=view_l, view_r=view_r)

    images = torch.stack([img_l, img_r], dim=0)[None]
    views = torch.stack([view_l, view_r], dim=0)[None]
    images, views = images.cuda(non_blocking=True), views.cuda(non_blocking=True)
    patterns = model(images, views)
    patterns_layer = patterns[0].detach().clone()
    for j, pred in enumerate(patterns_layer):
        pred = pred.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path + 'layer-'+str(j+1)+'.png', pred)
    outs = model.get_loss(patterns, gt=images, views=views, return_preds=True) 
    preds = outs['preds'][0].detach().clone()
    for j, pred in enumerate(preds):
        pred = pred.mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).type(torch.uint8).cpu().numpy()
        pred = cv2.cvtColor(pred, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path + 'view-'+str(j+1)+'.png', pred)

if __name__ == "__main__":
    
    from config.args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    args.scene = 'lego_bulldozer'
    init_scene_args(args=args)
    
    save_path = r"./outputs/inference/demo/coords/{}/".format(args.scene)
    eyeRealNet_weights = r"weight/model_ckpts/lego_bulldozer.pth"
    gaussian_path = r'weight/gaussian_ply/lego_bulldozer.ply'
    mid_coord = [100, 0, 0] 
    os.makedirs(save_path, exist_ok=True)
    load_inference(mid_coord=mid_coord, eyeRealNet_weights=eyeRealNet_weights, 
                   gaussian_path=gaussian_path, save_path=save_path, args=args)