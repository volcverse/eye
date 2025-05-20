import math
import torch
from torch import nn
from torch.nn import functional as F

from torchvision.transforms.functional import perspective
from model.loss import *
from model.metric import *

class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            nn.Conv2d(in_channels, mid_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels, kernel_size)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, kernel_size, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=kernel_size, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, kernel_size)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

class LearnableDecompose(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(LearnableDecompose, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c, kernel_size)
        self.down1 = Down(base_c, base_c * 2, kernel_size)
        self.down2 = Down(base_c * 2, base_c * 4, kernel_size)
        self.down3 = Down(base_c * 4, base_c * 8, kernel_size)
        self.down4 = Down(base_c * 8, base_c * 16, kernel_size)
        factor = 2 if bilinear else 1
        self.down5 = Down(base_c * 16, base_c * 32 // factor, kernel_size=3)

        self.up1 = Up(base_c * 32, base_c * 16 // factor, 3, bilinear)
        self.up2 = Up(base_c * 16, base_c * 8 // factor, kernel_size, bilinear)
        self.up3 = Up(base_c * 8, base_c * 4 // factor, kernel_size, bilinear)
        self.up4 = Up(base_c * 4, base_c * 2 // factor, kernel_size, bilinear)
        self.up5 = Up(base_c * 2, base_c, kernel_size, bilinear)

        self.out_conv = OutConv(base_c, out_channels)

    def forward(self, x: torch.Tensor):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)
        x = self.up1(x6, x5)
        x = self.up2(x, x4)
        x = self.up3(x, x3)
        x = self.up4(x, x2)
        x = self.up5(x, x1)
        out = self.out_conv(x)

        return out


class EyeRealNet(nn.Module):
    def __init__(self, args, FOV=None, bilinear=True):
        super(EyeRealNet, self).__init__()

        self.ssim_calc = SSIM()
        self._init_geometry_(
            pattern_size=(args.image_height, args.image_width),
            FOV=FOV,
            thickness=args.thickness,
            scale_physical2world=args.scale_physical2world,
            physical_width = args.physical_width,
            ground = args.ground,
            ground_coefficient = args.ground_coefficient,
            orientation = args.orientation,
            delta=[args.delta_x, args.delta_y, args.delta_z]
        )
        self.aux_loss = args.aux_loss
        self.l1_loss = args.l1_loss
        self.l1_mutex = args.l1_mutex
        self.use_mutex = args.mutex
        
        self.embed_dim = args.embed_dim
        self.subnets = nn.ModuleList([LearnableDecompose(
                in_channels=2*3, out_channels=3, 
                base_c=args.embed_dim, bilinear=bilinear) 
            for _ in range(3)])
    
    
    def _init_geometry_(self, pattern_size, FOV, thickness, scale_physical2world, physical_width, ground, ground_coefficient, orientation, delta):
        '''
        thickness: the physical length of whole thickness between screens at both sides
        '''
        self.FOV = FOV
        self.N_screen = 3
        H, W = pattern_size
        scale_pixel2world = scale_physical2world * physical_width / W
        Z_world = thickness * scale_physical2world
        if ground_coefficient != None:
            z_min = Z_world * ground_coefficient
        else:
            z_min = ground
        z_max = (z_min + Z_world)
        W_w = W * scale_pixel2world
        H_w = H * scale_pixel2world
        
        import pdb; pdb.set_trace()
        
        if orientation == "xoy":
            self.coord_screen_world = torch.stack([
                torch.Tensor([[-W_w/2, H_w/2, z], [W_w/2, H_w/2, z], [-W_w/2, -H_w/2, z], [W_w/2, -H_w/2, z]
            ]) for z in torch.linspace(z_min, z_max, self.N_screen).tolist()])
        elif orientation == "xoz":
            self.coord_screen_world = torch.stack([
                torch.Tensor([[ -W_w/2,z, H_w/2], [ W_w/2, z,H_w/2], [-W_w/2,z,  -H_w/2], [W_w/2, z, -H_w/2]
            ]) for z in torch.linspace(z_min, z_max, self.N_screen).tolist()])
        elif orientation == "yox":
            self.coord_screen_world = torch.stack([
                torch.Tensor([[-H_w/2, -W_w/2, z], [-H_w/2, W_w/2, z], [H_w/2, -W_w/2, z], [H_w/2, W_w/2, z]
            ]) for z in torch.linspace(z_min, z_max, self.N_screen).tolist()])
        elif orientation == "yoz":
            self.coord_screen_world = torch.stack([
                torch.Tensor([[z, -W_w/2, H_w/2], [z, W_w/2, H_w/2], [z, -W_w/2, -H_w/2], [z, W_w/2, -H_w/2]
            ]) for z in torch.linspace(z_min, z_max, self.N_screen).tolist()])
        elif orientation == "zox":
            self.coord_screen_world = torch.stack([
                torch.Tensor([[H_w/2, z, -W_w/2], [H_w/2, z, W_w/2], [-H_w/2, z, -W_w/2], [-H_w/2, z, W_w/2]
            ]) for z in torch.linspace(z_min, z_max, self.N_screen).tolist()])
            
        self.coord_pixel_init = torch.Tensor([(0, 0), (W, 0), (0, H), (W, H)]).view(1,4,2).repeat(self.N_screen,1,1)
        for index in range(3):
            self.coord_screen_world[..., index] = self.coord_screen_world[..., index] + delta[index]


    def transfrom_screen(self, coord_screen_world_input, a):

        a_expanded = a.unsqueeze(0).repeat(3, 1, 1)

        coord_screen_world_input_transposed = coord_screen_world_input.permute(0, 2, 1)

        result = torch.matmul(a_expanded, coord_screen_world_input_transposed)

        result = result.permute(0, 2, 1)

        self.coord_screen_world = result
        
        return result

    def forward(self, imgs: torch.Tensor, views: torch.Tensor, FOV=None):
        if FOV is None: 
            FOV = self.FOV
        # imgs: B, N_in, 3, H, W
        # view: B, N_in, 4, 4
        # self.coord_screen_world: N_s 4 3
        B, N_in, C_rgb, H, W = imgs.shape
        N_s = self.N_screen
        # B N_in 3 H W -> B*N_s*N_in 3 H W
        x, _ = self.view_tranform(
            imgs.view(B, 1, N_in, C_rgb, H, W).repeat(1, N_s, 1, 1, 1, 1).flatten(0, 2),
            views.view(B, 1, N_in, 4, 4).repeat(1, N_s, 1, 1, 1).flatten(0, 2), FOV,
            self.coord_screen_world.view(1, N_s, 1, 4, 3).repeat(B, 1, N_in, 1, 1).flatten(0, 2),
            self.coord_pixel_init.view(1, N_s, 1, 4, 2).repeat(B, 1, N_in, 1, 1).flatten(0, 2),
            reverse=True,
        )
        # B*N_s*N 3 H W -> B*N_s N*3 H W -> B*N_s C H W


        x = x.view(B, N_s, N_in*C_rgb, H, W)
        patterns = torch.stack([self.subnets[i](x[:, i]) for i in range(self.N_screen)], dim=1)

        
        patterns = torch.sigmoid(patterns) - 0.2

        return patterns
    
    def calibrate(self, imgs, views, FOV=None):
        if FOV is None: 
            FOV = self.FOV
        # imgs: B, N_in, 3, H, W
        # view: B, N_in, 4, 4
        # self.coord_screen_world: N_s 4 3
        B, N_in, C_rgb, H, W = imgs.shape
        N_s = self.N_screen
        # B N_in 3 H W -> B*N_s*N_in 3 H W
        x, _ = self.view_tranform(
            imgs.view(B, 1, N_in, C_rgb, H, W).repeat(1, N_s, 1, 1, 1, 1).flatten(0, 2),
            views.view(B, 1, N_in, 4, 4).repeat(1, N_s, 1, 1, 1).flatten(0, 2), FOV,
            self.coord_screen_world.view(1, N_s, 1, 4, 3).repeat(B, 1, N_in, 1, 1).flatten(0, 2),
            self.coord_pixel_init.view(1, N_s, 1, 4, 2).repeat(B, 1, N_in, 1, 1).flatten(0, 2),
            reverse=True,
        )

        return x
    
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
        

        # patterns: B N_s 3 H W
        # views: B N_in 4 4
        B, N_s, C_rgb, H, W = patterns.shape
        N_in = views.shape[1]

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
        

        if FOV is None:
            FOV = self.FOV
        # patterns: B N_s 3 H W
        # views: B N_in 4 4
        results, masks = self.aggregation(patterns=patterns, views=views, FOV=FOV)


        loss = F.mse_loss(results*masks, gt*masks)
        psnr = get_PSNR(loss.item(), masks)

        outs = dict(loss_mse=loss, PSNR=psnr)
        if self.l1_mutex or self.use_mutex:
            outs['loss_mutex'] = 1 - get_mutex_loss((results*masks).flatten(0,1), (gt*masks).flatten(0,1))
        if self.l1_mutex or self.l1_loss:
            outs['loss_l1'] = torch.abs((results*masks - gt*masks)).mean()
        if return_preds:
            outs['preds'] = results.detach().clone().reshape(*gt.shape)
        return outs

    def get_prediction(self, patterns, predict_views, FOV=None):
        if FOV is None:
            FOV = self.FOV
        return self.aggregation(patterns, predict_views, FOV)[0]



