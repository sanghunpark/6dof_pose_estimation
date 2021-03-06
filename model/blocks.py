from torch import nn

# DataParallel wappper to save/load network models
class WappedDataParallel(nn.DataParallel):
        def __getattr__(self, name):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def state_dict(self, *args):
            return self.module.state_dict(*args)

        def load_state_dict(self, *args):
            return self.module.load_state_dict(*args)

class UpdownUnetBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, pad, norm='none', activation='lrelu', pad_type='zero', updown='up', use_bias=True):
        super(UpdownUnetBlock, self).__init__()
        self.model = []
        if updown == 'down':
            self.model += [Conv2dBlock(in_dim, out_dim, ks, st=st, padding=pad,
                                       norm=norm, activation=activation, pad_type=pad_type, use_bias=use_bias)]
            self.model += [nn.AvgPool2d(kernel_size=2)]            
        elif updown == 'up':
            self.model += [nn.Upsample(scale_factor=2, mode='bilinear')]
            self.model += [Conv2dBlock(in_dim, out_dim, ks, st=st, padding=pad,
                            norm=norm, activation=activation, pad_type=pad_type, use_bias=use_bias)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
            return self.model(x)

class UpdownResBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm, activation, updown='up'):
        super(UpdownResBlock, self).__init__()
        self.model = []
        if updown == 'down':
            self.model += [ActFirstResBlock(in_dim, out_dim,
                                        norm=norm,
                                        activation=activation)]
            self.model += [nn.ReflectionPad2d(1)]
            self.model += [nn.AvgPool2d(kernel_size=3, stride=2)]

        elif updown == 'up':
            self.model += [nn.Upsample(scale_factor=2)]
            self.model += [ActFirstResBlock(in_dim, out_dim,
                                        norm=norm,
                                        activation=activation)]
        self.model = nn.Sequential(*self.model)
    
    def forward(self, x):
        return self.model(x)

class ResBlocks(nn.Module):
    def __init__(self, num_blocks, dim, norm, activation, pad_type):
        super(ResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ResBlock(dim,
                                    norm=norm,
                                    activation=activation,
                                    pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ResBlock(nn.Module):
    def __init__(self, dim, norm='in', activation='relu', pad_type='zero'):
        super(ResBlock, self).__init__()
        model = []
        model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation=activation,
                              pad_type=pad_type)]
        model += [Conv2dBlock(dim, dim, 3, 1, 1,
                              norm=norm,
                              activation='none',
                              pad_type=pad_type)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        residual = x
        out = self.model(x)
        out += residual
        return out


class ActFirstResBlocks(nn.Module):
    def __init__(self, num_blocks, fin, fout, norm, activation, pad_type):
        super(ActFirstResBlocks, self).__init__()
        self.model = []
        for i in range(num_blocks):
            self.model += [ActFirstResBlock(fin, fout,
                                            norm=norm,
                                            activation=activation,
                                            pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class ActFirstResBlock(nn.Module):
    def __init__(self, fin, fout, fhid=None,
                 activation='lrelu', norm='none', pad_type='reflect'):
        super().__init__()
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        self.fhid = min(fin, fout) if fhid is None else fhid
        self.conv_0 = Conv2dBlock(self.fin, self.fhid, 3, 1,
                                  padding=1, pad_type=pad_type, norm=norm,
                                  activation=activation, activation_first=True)
        self.conv_1 = Conv2dBlock(self.fhid, self.fout, 3, 1,
                                  padding=1, pad_type=pad_type, norm=norm,
                                  activation=activation, activation_first=True)
        if self.learned_shortcut:
            self.conv_s = Conv2dBlock(self.fin, self.fout, 1, 1,
                                      activation='none', use_bias=False)

    def forward(self, x):
        x_s = self.conv_s(x) if self.learned_shortcut else x
        dx = self.conv_0(x)
        dx = self.conv_1(dx)
        out = x_s + dx
        return out


class Conv2dBlock(nn.Module):
    def __init__(self, in_dim, out_dim, ks, st, padding=0,
                norm='none', activation='relu', pad_type='zero',
                use_bias=True, activation_first=False,
                spectral_norm=True):
        super(Conv2dBlock, self).__init__()
        self.use_bias = use_bias
        self.activation_first = activation_first
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        # elif norm == 'adain':
        #     self.norm = AdaptiveInstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False) # 'inplace=Ture' causes 'inplace operator Runtime errors when use it with 'instance norm'
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        if spectral_norm:
            self.conv = nn.utils.spectral_norm(nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias))
        else:
            self.conv = nn.Conv2d(in_dim, out_dim, ks, st, bias=self.use_bias)

    def forward(self, x):
        if self.activation_first:
            if self.activation:
                x = self.activation(x)
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
        else:
            x = self.conv(self.pad(x))
            if self.norm:
                x = self.norm(x)
            if self.activation:
                x = self.activation(x)
        return x

class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        self.fc = nn.Linear(in_dim, out_dim, bias=use_bias)

        # initialize normalization
        norm_dim = out_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=False)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=False)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out