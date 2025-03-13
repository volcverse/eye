from PIL import Image
import torch
import os
from torch.utils.data import Dataset


class NTFDataset(Dataset):

    def __init__(self, images_path: list, transform=None, data_prefix=None,  delta=None, vertical=None, train_NTF=False):
        self.images_path = images_path
        self.data_prefix = data_prefix 
        self.transform = transform
        self.train_NTF = train_NTF

        self.images_view = self.views_generation()
        self.N = -1



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
            

            view = self.images_view[idx%2]


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
