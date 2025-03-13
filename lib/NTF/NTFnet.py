import math
import torch
from torch import nn
from torch.nn import functional as F

from torchvision.transforms.functional import perspective
from model.loss import *
from model.metric import *


class NTFNet(nn.Module):
    def __init__(self, args, view_ratio=None, device=None):
        super(NTFNet, self).__init__()

        self.train_NTF = args.train_NTF
        self.ssim_calc = SSIM()
        self.pattern_size = torch.Size((3, args.image_height, args.image_width)) # N 3 H W
        self.device = device
        self.rear = torch.rand(self.pattern_size).to(device=device)
        self.middle = torch.rand(self.pattern_size).to(device=device)
        self.front = torch.rand(self.pattern_size).to(device=device)
        self.view_ratio = view_ratio
        self.view = self.views_generation().to(device=device)
        self.bias = torch.full(self.pattern_size, 1e-5, dtype=torch.float32).to(device=device)
        
    
    def views_generation(self):
        return torch.tensor([
                    [0, -0.5],
                    [0, 0.5]
                ])
    def mask_generation(self, view):
        H, W = self.middle.shape[-2:]
        h, w = view
        h_abs = int(round(abs(h)))
        
        w_abs = int(round(abs(w)))
        mask = torch.ones(H, W).to(device=self.device)
        if h >= 0:
            mask[:h_abs, :] = 0
        else:
            mask[H-h_abs:, :] = 0
        if w >= 0:
            mask[:, :w_abs] = 0
        else:
            mask[:, W-w_abs:] = 0

        return mask
    
    def refineLayer(self, views, images, refine_layer=None):
        
        numerator = torch.zeros(self.pattern_size, dtype=torch.float32).to(device=self.device)
        denominator = torch.zeros(self.pattern_size, dtype=torch.float32).to(device=self.device)
        for i, view_ in enumerate(views):
            view = torch.stack([view_[0] * self.view_ratio[0], 
                                view_[1] * self.view_ratio[1]]).to(device=self.device)
            
            view_shifts = (view[0].round().int().item(), view[1].round().int().item()) 
            anti_view_shifts = (-view[0].round().int().item(), -view[1].round().int().item()) 

            add_result = (torch.roll(self.front, shifts=view_shifts, dims=(-2, -1)) * self.mask_generation(view_shifts)) + \
                            self.middle + \
                        (torch.roll(self.rear, shifts=anti_view_shifts, dims=(-2, -1)) * self.mask_generation(anti_view_shifts))
            if refine_layer == 'rear':
                numerator += torch.roll(images[i], shifts=view_shifts, dims=(-2, -1)) * self.mask_generation(view_shifts) 
                denominator += torch.roll(add_result, shifts=view_shifts, dims=(-2, -1))  * self.mask_generation(view_shifts)
            elif refine_layer == 'middle':
                numerator += images[i]
                denominator += add_result 
            elif refine_layer == 'front':
                numerator += torch.roll(images[i], shifts=anti_view_shifts, dims=(-2, -1)) * self.mask_generation(anti_view_shifts)  
                denominator += torch.roll(add_result, shifts=anti_view_shifts, dims=(-2, -1)) * self.mask_generation(anti_view_shifts) 
            else:
                raise ValueError("refine_layer not formal")

        if refine_layer == 'rear':
            self.rear = self.rear * (numerator + self.bias) / (denominator + self.bias) 
            self.rear = torch.clamp(self.rear, 0, 1)
        elif refine_layer == 'middle':
            self.middle = self.middle * (numerator + self.bias) / (denominator + self.bias) 
            self.middle = torch.clamp(self.middle, 0, 1)
        elif refine_layer == 'front':
            self.front = self.front * (numerator + self.bias) / (denominator + self.bias) 
            self.front = torch.clamp(self.front, 0, 1)
        else:
            raise ValueError("refine_layer not formal")
    
    def update(self, views, images):
        self.refineLayer(views, images, "front")
        self.refineLayer(views, images, "middle")
        self.refineLayer(views, images, "rear")

    def getResults(self, views, images):
        add_results = list()
        res_masks = list()
        for i, view_ in enumerate(views):
            view = torch.stack([view_[0] * self.view_ratio[0], 
                                view_[1] * self.view_ratio[1]]).to(device=self.device)
            view_shifts = (view[0].round().int().item(), view[1].round().int().item()) 
            anti_view_shifts = (-view[0].round().int().item(), -view[1].round().int().item()) 
            add_result = (torch.roll(self.front, shifts=view_shifts, dims=(-2, -1)) * self.mask_generation(view_shifts)) + \
                            self.middle + \
                        (torch.roll(self.rear, shifts=anti_view_shifts, dims=(-2, -1)) * self.mask_generation(anti_view_shifts))

            add_results.append(add_result)
            res_masks.append((self.mask_generation(view_shifts) * self.mask_generation(anti_view_shifts))[None])
        return torch.stack(add_results), torch.stack(res_masks)

   

    
    @staticmethod
    def view_tranform(imgs, view, FOV, coord_src, coord_src_img, reverse=False):
        N, _, H, W = imgs.shape
        fx = W/2 / math.tan(FOV/2)
        coord_src_homo = torch.cat([coord_src, torch.ones(N,4,1)], dim=-1).to(imgs.device)
        coord_dst = torch.matmul(torch.inverse(view)[:, None], coord_src_homo[..., None]).squeeze(-1)[..., :3] # N 4 3
        u = (-fx*coord_dst[..., [0]]/coord_dst[..., [2]] + W/2)
        v = (fx*coord_dst[..., [1]]/coord_dst[..., [2]] + H/2)
        coord_dst_img = torch.cat([u, v], dim=-1)

        if coord_dst_img.isinf().any() or coord_dst_img.isnan().any():
            coord_dst_img = coord_src_img
            masks = torch.zeros_like(imgs)
        else:
            masks = torch.ones_like(imgs)
        
        if not reverse:
            imgs_new = torch.stack([perspective(img, src.tolist(), dst.tolist()) 
                                for img, src, dst in zip(imgs, coord_src_img, coord_dst_img)])
            masks_new = torch.stack([perspective(mask, src.tolist(), dst.tolist()) 
                                for mask, src, dst in zip(masks, coord_src_img, coord_dst_img)])
        else:
            imgs_new = torch.stack([perspective(img, src.tolist(), dst.tolist()) 
                               for img, src, dst in zip(imgs, coord_dst_img, coord_src_img)])
            masks_new = torch.stack([perspective(mask, src.tolist(), dst.tolist()) 
                                for mask, src, dst in zip(masks, coord_dst_img, coord_src_img)])

        return imgs_new, masks_new
    
    def aggregation(self, patterns=None, views=None, FOV=None):
        
        # patterns: N_s 3 H W
        patterns = self.patterns.unsqueeze(0)
        # views: B N_in 4 4
        B, N_in = views.shape[0], views.shape[1]
        
        patterns = patterns.repeat(B, 1, 1, 1, 1)
        B, N_s, C_rgb, H, W = patterns.shape
        FOV = self.FOV


        # B N_s 3 H W -> B N_s N_in 3 H W -> B*N_s*N_in 3 H W
        patterns_new, masks_new = self.view_tranform(
            patterns.view(B, N_s, 1, C_rgb, H, W).repeat(1, 1, N_in, 1, 1, 1).flatten(0, 2), 
            views.view(B, 1, N_in, 4, 4).repeat(1, N_s, 1, 1, 1).flatten(0, 2), FOV,
            self.coord_screen_world.view(1, N_s, 1, 4, 3).repeat(B, 1, N_in, 1, 1).flatten(0, 2), 
            self.coord_pixel_init.view(1, N_s, 1, 4, 2).repeat(B, 1, N_in, 1, 1).flatten(0, 2),
        )
        patterns_new = patterns_new.view(B, N_s, N_in, C_rgb, H, W)
        masks_new = masks_new.view(B, N_s, N_in, C_rgb, H, W)
        
        results = patterns_new.sum(dim=1)
        results = torch.sin(results*math.pi/2)**2
        masks = masks_new.prod(dim=1)

        return results, masks
    
    def get_loss(self, patterns, gt, views, FOV=None, return_preds=False):
        

        # patterns: B N_s 3 H W
        self.patterns.data = self.patterns.data.clamp(0, 1)
        # views: B N_in 4 4
        results, masks = self.aggregation(views=views)



        loss = F.mse_loss(results*masks, gt*masks)
        psnr = get_PSNR(loss.item(), masks)

        outs = dict(loss_mse=loss, PSNR=psnr)
        if self.l1_mutex:
            outs['loss_mutex'] = 1 - self.mutex_loss((results*masks).flatten(0,1), (gt*masks).flatten(0,1))
        if self.l1_mutex or self.l1_loss:
            outs['loss_l1'] = torch.abs((results*masks - gt*masks)).mean()
        if return_preds:
            outs['preds'] = results.detach().clone().reshape(*gt.shape)
        return outs

    def get_prediction(self, patterns, predict_views, FOV=None):
        if FOV is None:
            FOV = self.FOV
        return self.aggregation(patterns, predict_views, FOV)[0]



