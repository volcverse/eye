import random
import math
import torch
import os
from data.render import GaussianRender
from dataset import eye2world_pytroch
from train_eyeReal import init_scene_args
from config.args import get_gaussian_parser

# 定义基本参数
FOV = 40  # 视场角（Field of View）
R_min, R_max = 10, 150  # 相机到物体中心的距离范围
num = 10000  # 生成的数据对数量
phi_min, phi_max = 60, 120  # 仰角范围（度）
theta_min, theta_max = 40, 140  # 方位角范围（度）

def randomize(vertical, orientation, scale_physical2world):
    """
    随机生成两个相机位置（双目视觉）
    参数:
        vertical: 垂直轴方向 ('x', 'y', 或 'z')
        orientation: 相机朝向平面 ('xoy', 'xoz', 'yox', 'yoz', 'zox')
        scale_physical2world: 物理世界到虚拟世界的缩放比例
    返回:
        两个相机位置的坐标元组 (x1,y1,z1), (x2,y2,z2)
    """
    # 设置双目相机之间的距离
    eye_distance = 6*scale_physical2world
    # 随机生成相机到物体中心的距离
    R = scale_physical2world*random.uniform(R_min, R_max)
    # 随机生成仰角和方位角
    phi = math.radians(random.uniform(phi_min, phi_max))
    theta = math.radians(random.uniform(theta_min, theta_max))

    # 根据垂直轴方向设置固定坐标
    if vertical == "z":
        z1 = z2 = round(R*math.cos(phi), 3)
    elif vertical == "x":
        x1 = x2 = round(R*math.cos(phi), 3)
    elif vertical == "y":
        y1 = y2 = round(R*math.cos(phi), 3)
    else:
        raise ValueError("wrong input vertical")
    
    # 计算双目相机的相对位置
    d = R*math.sin(phi)
    sign = -1 if d <= 0 else 1
    delta = abs(math.atan(0.5*eye_distance/d))
    r = math.sqrt(d**2 + (0.5*eye_distance)**2) * sign
    
    # 根据不同的朝向平面计算相机位置
    if orientation == "xoy":
        z1 = round(r*math.sin(theta-delta), 3)
        z2 = round(r*math.sin(theta+delta), 3)
        x1 = round(r*math.cos(theta-delta), 3) 
        x2 = round(r*math.cos(theta+delta), 3)
    elif orientation == "xoz":
        y1 = round(r*math.sin(theta-delta), 3)
        y2 = round(r*math.sin(theta+delta), 3)
        x1 = round(r*math.cos(theta-delta), 3) 
        x2 = round(r*math.cos(theta+delta), 3)
    elif orientation == "yox":
        z1 = round(r*math.sin(theta-delta), 3)
        z2 = round(r*math.sin(theta+delta), 3)
        y1 = round(r*math.cos(theta-delta), 3) 
        y2 = round(r*math.cos(theta+delta), 3)
    elif orientation == "yoz":
        x1 = round(r*math.sin(theta-delta), 3)
        x2 = round(r*math.sin(theta+delta), 3)
        y1 = round(r*math.cos(theta-delta), 3) 
        y2 = round(r*math.cos(theta+delta), 3)
    elif orientation == "zox":
        y1 = round(r*math.sin(theta-delta), 3)
        y2 = round(r*math.sin(theta+delta), 3)
        z1 = round(r*math.cos(theta-delta), 3) 
        z2 = round(r*math.cos(theta+delta), 3)

    return (x1, y1, z1), (x2, y2, z2)

if __name__ == "__main__":
    # 设置高斯渲染器的参数
    gaussian_path = 'weight/gaussian_ply/lego_bulldozer.ply'
    render = GaussianRender(
        parser=get_gaussian_parser(),
        sh_degree=3, 
        gaussians_path=gaussian_path,
        white_background=True, 
        FOV=FOV / 180 * math.pi)
    
    # 初始化场景参数
    from config.args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    args.scene = 'lego_bulldozer'
    init_scene_args(args=args)

    # 创建数据保存目录
    file_path = 'dataset/scene_data/{}/'.format(args.scene)
    data_folder = '{}{}_scale_{}_R_{}_{}_FOV_{}_theta_{}_{}_phi_{}_{}'.format(
        args.scene, num, round(args.scale_physical2world,3), 
        R_min, R_max, FOV, theta_min, theta_max, phi_min, phi_max)
    data_path = os.path.join(file_path, data_folder)
    os.makedirs(data_path, exist_ok=True)

    # 生成训练数据
    from tqdm import tqdm
    import torchvision
    for i in tqdm(range(num)):
        # 随机生成双目相机位置
        eye1, eye2 = randomize(args.vertical, args.orientation, args.scale_physical2world)
        eye1_t = torch.tensor(eye1)
        eye2_t = torch.tensor(eye2)
        
        # 渲染第一个视角的图像
        eye1_view = eye2world_pytroch(args.vertical, eye1_t).cuda()
        eye1_img = render.render_from_view(eye1_view)
        torchvision.utils.save_image(
            eye1_img, 
            data_path + "/" + 'pair{}_x{}_y{}_z{}_{}'.format(
                i, round(eye1[0], 3), round(eye1[1], 3), round(eye1[2], 3), 0) + ".jpg")
        
        # 渲染第二个视角的图像
        eye2_view = eye2world_pytroch(args.vertical, eye2_t).cuda()
        eye2_img = render.render_from_view(eye2_view)
        torchvision.utils.save_image(
            eye2_img, 
            data_path + "/" + 'pair{}_x{}_y{}_z{}_{}'.format(
                i, round(eye2[0], 3), round(eye2[1], 3), round(eye2[2], 3), 1) + ".jpg")
    


    
