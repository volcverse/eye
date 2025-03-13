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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import pdb

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    
    pose_rotation=[0,0,0]
    pose_translation=[0,0,0]

    if "desk_bookshelf" in cam_info.image_path:
        pose_rotation=[-114, 0, 90]
        pose_translation=[4.8, -2.5, 0]
    if "hotdog" in cam_info.image_path:
        pose_rotation=[0, -92, 14] 
        pose_translation=[2.9, 0.04, 0.16]
    if "orchids" in cam_info.image_path:
        pose_rotation=[-90, 0, 90]
        pose_translation=[19.16, -1.78, 0]
    if "traffic" in cam_info.image_path:
        pose_rotation=[216, -1, -28]
        pose_translation=[-0.36, 0, 2.54]
    if "street" in cam_info.image_path:
        pose_rotation=[-116, 0, 114]
        pose_translation=[2.56, 0, 1.46]
    if "truck" in cam_info.image_path:
        pose_rotation=[80, 182, -92]
        pose_translation=[0.5, -0.24, 0.1]
    if "desk_bookshelf" in cam_info.image_path:
        pose_rotation=[234, 0, 122]
        pose_translation=[2.6, -0.24, 3.1]
    if "room_floor" in cam_info.image_path:
        pose_rotation=[-20, -90, 86]
        pose_translation=[0, 0.48, -2.92]
    if "wukang_mansion" in cam_info.image_path:
        pose_rotation=[194, 23, -31]
        pose_translation=[0.34, -0.02, 2.02]
    if "shoe_rack" in cam_info.image_path:
        pose_rotation=[234,  0, 122]
        pose_translation=[2.6, -0.24, 3.1]
    if "china_art_museum" in cam_info.image_path:
        pose_rotation=[234, 4, -61]
        pose_translation=[0.54, 0.33, 0.72]
    

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                  pose_rotation=pose_rotation,pose_translation=pose_translation)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
