# PyTorch
from torch import nn
import torch

# My library
from model.nets import Encoder, Decoder, Unet

class PoseNet(nn.Module):
    def __init__(self, config, device):
        super(PoseNet, self).__init__()
        self.device = device
        ### 1-stage
        self.rgb_enc = Encoder(
            downs=config['rgb_2_vf_seg']['n_down'],
            input_dim=3,
            dim=config['rgb_2_vf_seg']['n_f'], 
            n_res_blks=config['rgb_2_vf_seg']['n_res_blks'],
            norm='in',
            activ='relu',
            pad_type='reflect')

        self.n_keypoint = 9 # bouding box (8) + controid (1)
        self.vf_dec = Decoder(
            ups=config['rgb_2_vf_seg']['n_down'],
            dim=self.rgb_enc.output_dim,
            output_dim=2*self.n_keypoint,            
            n_res_blks=config['rgb_2_vf_seg']['n_res_blks'],
            norm='in',
            activ='relu',
            pad_type='reflect',
            upsample=False)
        
        self.n_class = len(config['class'])
        self.sg_dec = Decoder(
            ups=config['rgb_2_vf_seg']['n_down'],
            dim=self.rgb_enc.output_dim,
            output_dim=self.n_class,
            n_res_blks=config['rgb_2_vf_seg']['n_res_blks'],
            norm='in',
            activ='relu',
            pad_type='reflect',
            upsample=False)
        
        ### 2-stage
        self.bb_net = Unet(
            n_updown=config['bb_net']['n_updown'],
            n_res_blks=config['bb_net']['n_res_blks'],
            input_dim=1 + 2*self.n_keypoint, # segment for each object (1) + vector fields (2*number of keypoints)
            dim=config['bb_net']['n_f'],
            output_dim=self.n_keypoint, # confidence map for each 2D bouding box corner (2*number of keypoints) like OpenPose
            norm='in',
            activ='relu',
            pad_type='reflect')

    def forward(self, x):
        ### 1-stage
        B, _, H, W = x.size()
        x = self.rgb_enc(x)
        vf = self.vf_dec(x) # Bx(2*n_keypoints)xHxW (vector field)
        sg = self.sg_dec(x) # Bx(n_class)xHxW (segmenation)
        
        ### 2-stage
        bb = torch.zeros(self.n_class, B, self.n_keypoint, H, W).to(self.device) # CLSxBx(n_keypoints)xHxW
        for c in range(self.n_class):
            bb[c] = self.bb_net(torch.cat((sg[:,c:c+1,:,:], vf), dim=1)) # Bx(n_keypoints)xHxW
        return {'vf': vf, 'sg': sg, 'bb':bb}





    
