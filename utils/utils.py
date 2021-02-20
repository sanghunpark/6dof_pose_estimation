import torch
import numpy as np
import cv2 as cv

# OpenPose Eq(7)
# https://arxiv.org/abs/1812.08008
def get_confidence_map(x, p, sigma): # x: ground truth (Bx9x2) [0,1]
    # p = (kornia.create_meshgrid(H, W).permute(0,3,1,2).to(device) + 1) / 2 # (1xHxWx2) > (1x2xHxW), [-1,1] > [0, 1]  
    x = x.unsqueeze(-1).unsqueeze(-1) 
    p = p.unsqueeze(0)
    return torch.exp(-torch.linalg.norm(x-p, ord=2, dim=2) / (sigma**2))

# PVNet Eq(1)
# http://arxiv.org/abs/1812.11788
def get_vector_field(x, p, mask):
    B, C, H, W = mask.size()
    # p = (kornia.create_meshgrid(H, W).permute(0,3,1,2).to(device) + 1) / 2 # [-1,1] > [0, 1]
    x = x.unsqueeze(-1).unsqueeze(-1) # Bx(n_points)x2x1x1
    p = p.unsqueeze(1) # Bx1x2xHxW
    return (x-p)/ torch.linalg.norm(x-p, ord=2, dim=2, keepdim=True)

def get_keypoints(confidence_map): # Bx(n_point)xHxW->Bx(n_points)x2
    B, C, H, W = confidence_map.size()
    m = confidence_map.view(B*C, -1).argmax(dim=1).view(B,C,1) # find locations of max value in 1-dimension
    return torch.cat((m % H, m // H), dim=2) # Indices (x, y) of max values, Bx(n_points)x2

def draw_keypoints(rgb, gt_pnts, pr_pnts):
    B, _, H, W = rgb.size()
    imgs = rgb.cpu().detach().permute(0,2,3,1).numpy() # BxCxHxW > BxHxWxC
    imgs = np.array(imgs)
    B, n_pnts, _ = pr_pnts.size()
    for b in range(B):
        img = cv.cvtColor(imgs[b].copy(), cv.COLOR_BGR2RGB)
        for i in range(n_pnts):
            x, y = gt_pnts[b][i]
            img = cv.circle(img, (x*W, y*H), 8, (0,255,0), 1)
            x, y = pr_pnts[b][i]
            img = cv.circle(img, (x*W, y*H), 5, (0,0,255), -1)
            
        imgs[b] = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    # return torch.from_numpy(imgs).permute(0, 3, 1, 2)
    return imgs.transpose((0, 3, 1, 2))
            



