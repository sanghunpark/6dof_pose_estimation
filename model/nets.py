# PyTorch
import torch
from torch import nn
from torch.nn import functional as F

# My library
from model.blocks import UpdownUnetBlock, ResBlocks, Conv2dBlock, UpdownResBlock, ActFirstResBlocks, LinearBlock

class Encoder(nn.Module):
    def __init__(self, downs, input_dim, dim, n_res_blks, norm, activ, pad_type, global_pool=False, keepdim=False):
        super(Encoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3,
                                   norm='none',
                                   activation=activ,
                                   pad_type=pad_type)]
        for i in range(downs):
            self.model += [UpdownResBlock(dim, dim*2, norm=norm, activation=activ, updown='down')]
            dim *= 2
        # resblks
        self.model += [ActFirstResBlocks(n_res_blks, fin=dim, fout=dim,
                                norm=norm,
                                activation=activ,
                                pad_type=pad_type)]
        # global pooling
        self.global_pool = global_pool
        self.keepdim = keepdim

        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        x = self.model(x)
        if self.global_pool:
           x = torch.sum(x, dim=(2,3), keepdim=self.keepdim)
        return x


class Decoder(nn.Module):
    def __init__(self, ups, dim, output_dim, n_res_blks, norm, activ, pad_type, upsample=False):
        super(Decoder, self).__init__()
        self.model = []
        # resblks
        self.model += [ActFirstResBlocks(n_res_blks, fin=dim, fout=dim,
                                norm=norm,
                                activation=activ,
                                pad_type=pad_type)]
        for i in range(ups): # AdaIN used only upsample layer
            self.model += [UpdownResBlock(dim, dim//2, norm=norm, activation=activ, updown='up')]
            dim //= 2
        self.model += [Conv2dBlock(dim, output_dim, 7, 1, 3, 
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

        self.upsample = upsample

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, scale_factor=2)
        return self.model(x)


class Unet(nn.Module): # Unet
    def __init__(self, n_updown, n_res_blks, input_dim, dim, output_dim, norm='none', activ='lrelu', pad_type='zero'):
        super(Unet, self).__init__()
        use_bias = True if norm == 'in' else False
        self.models = []
        self.n_feat = n_updown # in_conv + downs = ups + out_conv
        
        # in-conv
        self.models += [UpdownUnetBlock(input_dim, dim, 3, 1, 1,
                                    norm=norm,
                                    activation=activ,
                                    pad_type=pad_type,
                                    updown='down',
                                    use_bias=use_bias)]
        # downs 
        for i in range(n_updown-1):
            self.models += [UpdownUnetBlock(dim, 2 * dim, 3, 1, 1,
                                    norm=norm,
                                    activation=activ,
                                    pad_type=pad_type,
                                    updown='down',
                                    use_bias=use_bias)]
            dim *= 2

        # inner-most
        self.models += [ResBlocks(  num_blocks=n_res_blks, 
                                    dim=dim,
                                    norm=norm,
                                    activation=activ,
                                    pad_type=pad_type)]

        
        # ups
        for i in range(n_updown-1):
            self.models += [UpdownUnetBlock(2*dim, dim // 2, 3, 1, 1, # (2*in_dim) for skip connection
                                    norm=norm,
                                    activation=activ,
                                    pad_type=pad_type,
                                    updown='up',
                                    use_bias=use_bias)]
            dim //= 2
        
        # out-conv
        self.models += [UpdownUnetBlock(2 * dim, output_dim, 3, 1, 1,
                                   norm=norm,
                                   activation=activ,
                                   pad_type=pad_type,
                                   updown='up',
                                   use_bias=use_bias)]

        self.models = nn.Sequential(*self.models)

    def forward(self, x):
        feats = []
        for m_idx in range(self.n_feat): # in_conv0 + downs
            x = self.models[m_idx](x)
            feats += [x]        
        x = self.models[self.n_feat](x) # inner_down
        feats.reverse()
        for f_idx in range(self.n_feat): # ups0~4 + out_conv
            m_idx = self.n_feat + 1 + f_idx # offset for index of model
            x_skip = feats[f_idx]
            x = torch.cat([x, x_skip], dim=1)
            x = self.models[m_idx](x)
        return x

class MLP(nn.Module):
    def __init__(self, input_dim, dim, output_dim, n_blk, norm, activ, global_pool=True):
        super(MLP, self).__init__()
        self.global_pool = global_pool
        self.model = []
        self.model += [LinearBlock(input_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, output_dim,
                                   norm='none', activation='none')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        if self.global_pool:
           x = torch.sum(x, dim=(2,3))
        return self.model(x.view(x.size(0), -1))