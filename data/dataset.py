from PIL import Image
import torch
import os
from torch.utils.data import Dataset


def eye2world_pytroch(vertical, eye_world: torch.Tensor, delta=torch.tensor([0., 0., 0.])):
    eye_world = eye_world.float()
    vecz = eye_world
    vecz = vecz / torch.linalg.norm(vecz)
    if vertical == "z":
        vec_w = torch.tensor([1e-5, 1e-6, 1.]).to(eye_world.device)
    elif vertical == "x":
        vec_w = torch.tensor([1., 1e-6, 1e-5]).to(eye_world.device)
    elif vertical == "y":
        vec_w = torch.tensor([1e-6, 1., 1e-5]).to(eye_world.device)
    else:
        raise ValueError("wrong input vertical")

    vecx = torch.cross(vec_w, vecz)
    vecx = vecx / torch.linalg.norm(vecx)
    vecy = torch.cross(vecz, vecx)
    vecy = vecy / torch.linalg.norm(vecy)
    rot = torch.stack([vecx, vecy, vecz]).T
    rt = torch.eye(4).to(eye_world.device)
    rt[:3, :3] = rot
    rt[:3, 3] = eye_world + delta
    
    return rt

class SceneDataset(Dataset):

    def __init__(self, images_path: list, transform=None, data_prefix=None,  delta=None, vertical=None, train_NTF=False):
        self.images_path = images_path
        self.data_prefix = data_prefix 
        self.transform = transform
        self.train_NTF = train_NTF
        if self.train_NTF:
            self.images_view = self.views_generation()
            self.N = -1
        else:
            self.delta = torch.tensor(delta)
            self.vertical = vertical
            self.N = len(self.images_path) // 2

    def get_ori_coord(self, s):
        pairs = s.split('_')
        x, y, z = float(pairs[1][1:]), float(pairs[2][1:]), float(pairs[3][1:])
        return (x,y,z)
   
    def get_img_matrix(self, s):
        ori_x, ori_y, ori_z = self.get_ori_coord(s)
        eye_world = torch.tensor([ori_x, ori_y, ori_z])
        return eye2world_pytroch(self.vertical, eye_world, self.delta)

    def __len__(self):
        return self.N

    def __getitem__(self, ind):
        
        left_id = ind * 2
        right_id = ind * 2 + 1
        indice = [left_id, right_id]

        imgs = list()
        views = list()
        for idx in indice:
            if self.data_prefix is not None:
                img = Image.open(os.path.join(self.data_prefix, self.images_path[idx]))
            else:
                img = Image.open(self.images_path[idx])

            if img.mode == 'RGBA':
                img = self.convert_RGB(img)
            elif img.mode != 'RGB':
                raise ValueError("image: {} isn't RGB mode.".format(self.images_path[idx]))
            
            if self.train_NTF:
                view = self.images_view[idx%2]
            else:
                view = self.get_img_matrix(self.images_path[idx])

            if self.transform is not None:
                img = self.transform(img)

            imgs.append(img)
            views.append(view)

        return torch.stack(imgs), torch.stack(views)
    
    def views_generation(self):
        return torch.tensor([[0, -0.5], [0, 0.5]])
    
    @staticmethod
    def convert_RGB(img):
        width = img.width
        height = img.height

        image = Image.new('RGB', size=(width, height), color=(255, 255, 255))
        image.paste(img, (0, 0), mask=img)

        return image

    @staticmethod
    def collate_fn(batch):
        images, views = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        views = torch.stack(views, dim=0)
        return images, views
