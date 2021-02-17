# PyTorch
import torch
import torch.nn as nn

# My library
from utils.utils import get_confidence_map
    
def compute_losses(out, data, device):
    _N_KEYPOINT = 9
    rgb = data['rgb'] # not used
    label = data['label'].to(device)
    # mask = data['mask'].to(self.device)
    B, _, H, W = rgb.size()

    cls_idx = label[:,0].long() # B
    gt_pnts = label[:,1:2*_N_KEYPOINT+1].view(-1, _N_KEYPOINT, 2) # Bx(2*n_points+3) > Bx(n_points)x2
    gt_conf = get_confidence_map(gt_pnts, _N_KEYPOINT, H, W, device=device) # Bx(n_points)xHxW

    pr_conf = out['bb'][cls_idx,:,:,:,:][0] # (n_class)xBx(n_keypoints)xHxW
    loss_bb = bb_loss(gt_conf, pr_conf)

    return [loss_bb, torch.zeros([1]).to(device)]

def bb_loss(gt, pr):
    return nn.L1Loss()(gt, pr).mean()