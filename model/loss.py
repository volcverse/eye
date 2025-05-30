import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

def get_mutex_loss(img1, img2, window_size=11, sigma=1.5, size_average=True):
    (_, channel, _, _) = img1.size()

    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    _1D_window = (gauss/gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C = 0.03**2

    ssim_map = (2*sigma12 + C)/(sigma1_sq + sigma2_sq + C)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def get_aux_loss(patterns, epoch_ratio, aux_weight=1):
    aux_weight *= 10.**(-epoch_ratio*4)
    return 10*aux_weight*F.mse_loss(patterns, torch.zeros_like(patterns))

