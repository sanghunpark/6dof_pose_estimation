# PyTorch
import torch
import torch.nn as nn
from torch.nn import functional as F

# etc
import kornia

# My library
from utils.utils import get_confidence_map, get_vector_field
    
def compute_losses(out, data, device, config):
    _N_KEYPOINT = 9
    gt_mask = data['mask'].to(device)
    label = data['label'].to(device)
    # mask = data['mask'].to(self.device)
    sg = out['sg'] # Bx(n_class)xHxW
    B, n_class, H, W = sg.size()
    batch_idx = torch.LongTensor(range(B)).to(device)
    cls_idx = label[:,0].long() # B

    ## loss for segmentation mask    
    pr_mask = sg[batch_idx,cls_idx,:,:].unsqueeze(1)
    loss_sg = F.mse_loss(gt_mask, pr_mask)

    ## loss for boudding box corners
    pos = (kornia.create_meshgrid(H, W).permute(0,3,1,2).to(device) + 1) / 2 # (1xHxWx2) > (1x2xHxW), [-1,1] > [0, 1]  
    gt_pnts = label[:,1:2*_N_KEYPOINT+1].view(-1, _N_KEYPOINT, 2) # Bx(2*n_points+3) > Bx(n_points)x2
    
    gt_conf = get_confidence_map(gt_pnts, pos.clone(), config['sigma']) # Bx(n_points)xHxW
    pr_conf = out['bb']
    # pr_conf = bb[batch_idx,cls_idx,:,:,:]# Bx(n_keypoints)xHxW
    loss_bb = F.mse_loss(gt_conf, pr_conf)


    ## loss for vector field
    pr_vf = out['vf']
    gt_vf = get_vector_field(gt_pnts, pos.clone(), gt_mask).view(B, 2*_N_KEYPOINT, H, W)
    loss_vf = F.mse_loss(gt_vf, pr_vf)

    ## loss for class confidence
    pr_cls = out['cl']
    gt_cls = torch.zeros((B, n_class)).to(device)
    gt_cls.scatter_(1, cls_idx.unsqueeze(-1), 1) # one-hot encoding
    loss_cl = F.mse_loss(gt_cls, pr_cls)

    return [loss_bb, loss_vf, loss_sg, loss_cl]

# def bb_loss(gt, pr):    
#     return F.mse_loss(gt, pr)
