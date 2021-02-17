# PyTorch
from torch import nn
import torch

# My library
from model.nets import Encoder, Decoder, Unet, MLP
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
        
        self.n_class = config['n_class']
        self.sg_dec = Decoder(
            ups=config['rgb_2_vf_seg']['n_down'],
            dim=self.rgb_enc.output_dim,
            output_dim=self.n_class,
            n_res_blks=config['rgb_2_vf_seg']['n_res_blks'],
            norm='in',
            activ='relu',
            pad_type='reflect',
            upsample=False)

        self.mlp = MLP(
            input_dim=self.rgb_enc.output_dim,
            dim=config['mlp']['n_f'],
            output_dim=self.n_class,
            n_blk=config['mlp']['n_blks'],
            norm='none',
            activ='relu',
            global_pool=True)
        
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
        cl = self.mlp(x)
        
        ### 2-stage
        batch_idx = torch.LongTensor(range(B)).to(self.device)
        cls_idx = torch.argmax(cl, dim=1).to(self.device)
        sg_1 = sg[batch_idx,cls_idx,:,:].unsqueeze(1)
        bb = self.bb_net(torch.cat((sg_1, vf), dim=1))

        # bb = torch.zeros(B, self.n_class, self.n_keypoint, H, W).to(self.device) # Bx(n_class)xx(n_keypoints)xHxW
        # for c in range(self.n_class):
        #     bb[batch_idx,c:c+1,:,:,:] = self.bb_net(torch.cat((sg[:,c:c+1,:,:], vf), dim=1)) # Bx(1)x(n_keypoints)xHxW
        return {'bb':bb, 'vf': vf, 'sg': sg, 'cl':cl}





    
