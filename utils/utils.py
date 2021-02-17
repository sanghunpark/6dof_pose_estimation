import torch
import kornia

_SIGMA = 1.6
# OpenPose Eq(7)
# https://arxiv.org/abs/1812.08008
def get_confidence_map(x, n_keypoints, H, W, device): # x: ground truth (9x2) [0,1]
    p = (kornia.create_meshgrid(H, W).permute(0,3,1,2).to(device) + 1) / 2 # (1xHxWx2) > (1x2xHxW), [-1,1] > [0, 1]  
    x = x.unsqueeze(-1).unsqueeze(-1) 
    p = p.unsqueeze(0)
    return torch.exp(-torch.linalg.norm(p-x, ord=2, dim=2) / (_SIGMA**2))