# sys
import sys

# PyTorch 
import torch

# etc
import numpy as np
import random
import cv2 as cv


# OpenPose Eq(7)
# https://arxiv.org/abs/1812.08008
def get_confidence_map(x, p, sigma):
    ''' x: ground truth (Bx9x2) [0,1], p: grid position '''
    x = x.unsqueeze(-1).unsqueeze(-1) 
    p = p.unsqueeze(0)
    return torch.exp(-torch.linalg.norm(x-p, ord=2, dim=2) / (sigma**2))

# PVNet Eq(1)
# http://arxiv.org/abs/1812.11788
def get_vector_field(x, p):
    # p = (kornia.create_meshgrid(H, W).permute(0,3,1,2).to(device) + 1) / 2 # [-1,1] > [0, 1]
    x = x.unsqueeze(-1).unsqueeze(-1) # Bx(n_points)x2x1x1
    p = p.unsqueeze(1) # Bx1x2xHxW
    return (x-p)/ torch.linalg.norm(x-p, ord=2, dim=2, keepdim=True)

def draw_keypoints(rgb, gt_pnts, pr_pnts):
    B, _, H, W = rgb.size()
    imgs = rgb.cpu().detach().permute(0,2,3,1).numpy() # BxCxHxW > BxHxWxC
    imgs = np.array(imgs)
    B, n_pnts, _ = pr_pnts.size()
    for b in range(B):
        img = cv.cvtColor(imgs[b].copy(), cv.COLOR_RGB2BGR)
        for i in range(n_pnts):
            x, y = gt_pnts[b][i]
            img = cv.circle(img, (x*W, y*H), 8, ((i+1)/9, (10-i)/9, (10-i)/9), 1)
            x, y = pr_pnts[b][i]
            img = cv.circle(img, (x*W, y*H), 3, ((i+1)/9, (10-i)/9, (10-i)/9), -1)
        imgs[b] = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # return torch.from_numpy(imgs).permute(0, 3, 1, 2)
    return imgs.transpose((0, 3, 1, 2))

def get_keypoints_cf(confidence_map): # Bx(n_point)xHxW->Bx(n_points)x2
    B, C, H, W = confidence_map.size()
    m = confidence_map.view(B*C, -1).argmax(dim=1).view(B,C,1) # find locations of max value in 1-dimension
    return torch.cat(((m % H)/W, (m // H)/H), dim=2) # Indices (x, y) of max values, Bx(n_points)x2, return position [0,1]

def get_keypoints_vf(vector_field, obj_seg, p, k):
    ''' vector_field: Bx(2*n_pnts)xHxW '''
    ''' segmentation: Bx(n_objects=1)xHxW '''
    # normalize segmentation
    B, n_cls, H, W = obj_seg.size() # Bx(n_cls=1)xHxW
    B, C, H, W = vector_field.size() # Bx(2*n_pnts)xHxW
    n_pnts = int(C/2)
    obj_seg = obj_seg.view(B, n_cls, -1)
    obj_seg -= torch.min(obj_seg, dim=2, keepdim=True)[0]
    obj_seg /= torch.max(obj_seg, dim=2, keepdim=True)[0]
    obj_seg = obj_seg.view(B, n_cls, H, W)
    mask = (obj_seg > 0.5)

    # mask vector fields by oject segmentation
    vector_field = vector_field.view(B,n_pnts,2,H,W)
    mask = mask.expand_as(vector_field)
    vf = torch.masked_select(vector_field, mask).view(B,n_pnts,2,-1).unsqueeze(-2) # Bx(n_pnts)x2x1x(n_sample)
    vf_pos = torch.masked_select(p, mask).view(B,n_pnts,2,-1).unsqueeze(-2) # Bx(n_pnts)x2x1x(n_sample)
    x_pos = p.unsqueeze(1).reshape(1,1,2,H*W).unsqueeze(-1) # (B:1)x(1)x2x(HxW)x(n_sample:1)

    n_sample = vf.size(-1)
    k = min(n_sample, k)
    k_idx = random.sample(range(n_sample), k)
    vf = vf[:,:,:,:,k_idx] # sample for hough voting
    vf_pos = vf_pos[:,:,:,:,k_idx]
    
    # compute cosine similarity (hough voting manner)
    diff = x_pos - vf_pos # Bxn_pntsx2x(HxW)x(k)
    diff_norm = torch.linalg.norm(diff, ord=2, dim=2, keepdim=True)
    dot = vf[:,:,0:1,:,:]*diff[:,:,0:1,:,:] + vf[:,:,1:2,:,:]*diff[:,:,1:2,:,:]
    score = torch.div(dot, diff_norm+ +sys.float_info.epsilon) # to avoid dividing by zero
    score = torch.sum(score, dim=4).squeeze(2)
    val, m = score.max(dim=2, keepdim=True)
    return torch.cat(((m % H)/W, (m // H)/H), dim=2) # predicted keypoints of 2D bouding box (x, y) [0, 1]





