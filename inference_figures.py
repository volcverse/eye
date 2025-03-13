import os
import torch
import torch.utils.data
from PIL import Image
from torchvision import transforms as T
from model.network import EyeRealNet
import math
import cv2
import warnings
from config.scene_dict import *
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


    
def get_ori_coord(s):
    pairs = s.split('_')

    x, y, z = float(pairs[1][1:]), float(pairs[2][1:]), float(pairs[3][1:])
    return (x,y,z)
   
def get_img_matrix(s, vertical, delta):
    ori_x, ori_y, ori_z = get_ori_coord(s)
    eye_world = torch.tensor([ori_x, ori_y, ori_z])
    return eye2world_pytroch(vertical=vertical, eye_world=eye_world, delta=delta)

def load_inference(eyeRealNet_weights, save_path, data_path, left_eye_path, right_eye_path, args):
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
    img_l = transform(Image.open(data_path + left_eye_path))
    img_r = transform(Image.open(data_path + right_eye_path))
    view_l = get_img_matrix(left_eye_path, args.vertical, delta)
    view_r = get_img_matrix(right_eye_path, args.vertical, delta)

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
    args.scene = 'matrixcity'
    # args.scene = 'lego_bulldozer'
    init_scene_args(args=args)

    save_path = r"./outputs/inference/demo/figures/{}/".format(args.scene)

    eyeRealNet_weights = r"./weight\model_ckpts\matrixcity.pth"
    data_path = r'./dataset\demo\matrixcity/'
    left_eye_path = "pair0_x1.21_y-3.401_z21.214_0.jpg"
    right_eye_path = "pair0_x-0.169_y-3.401_z21.248_1.jpg"

    # eyeRealNet_weights = r"weight\model_ckpts\lego_bulldozer.pth"
    # data_path = r'dataset\demo\lego_bulldozer/'
    # left_eye_path = "pair0_x6.487_y0.237_z2.004_0.jpg"
    # right_eye_path = "pair0_x6.486_y-0.263_z2.004_1.jpg"


    os.makedirs(save_path, exist_ok=True)
    load_inference(eyeRealNet_weights=eyeRealNet_weights, 
                   data_path=data_path, left_eye_path=left_eye_path, right_eye_path=right_eye_path, save_path=save_path, args=args)