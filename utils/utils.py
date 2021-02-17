import torch

_SIGMA = 1.6
# OpenPose Eq(7)
# https://arxiv.org/abs/1812.08008
def get_confidence_map(x, p, n_keypoints, H, W): # x: ground truth (9x2) [0,1]
    # p = (kornia.create_meshgrid(H, W).permute(0,3,1,2).to(device) + 1) / 2 # (1xHxWx2) > (1x2xHxW), [-1,1] > [0, 1]  
    x = x.unsqueeze(-1).unsqueeze(-1) 
    p = p.unsqueeze(0)
    return torch.exp(-torch.linalg.norm(x-p, ord=2, dim=2) / (_SIGMA**2))

# PVNet Eq(1)
# http://arxiv.org/abs/1812.11788
def get_vector_field(x, p, mask):
    B, C, H, W = mask.size()
    # p = (kornia.create_meshgrid(H, W).permute(0,3,1,2).to(device) + 1) / 2 # [-1,1] > [0, 1]
    x = x.unsqueeze(-1).unsqueeze(-1) # Bx(n_keypoins)x2x1x1
    p = p.unsqueeze(1) # Bx1x2xHxW
    return (x-p)/ torch.linalg.norm(x-p, ord=2, dim=2, keepdim=True)