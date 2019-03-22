import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import functools
from torch.optim import lr_scheduler
from memcnn.models.revop import ReversibleBlock
from torch.nn import Parameter
import numpy as np
import re

###############################################################################
# Helper Functions
###############################################################################


class Layer(nn.Module):
    def __init__(self):
        super(Layer, self).__init__()

    def forward(self, x):
        raise NotImplementedError

    def inverse(self, y):
        raise NotImplementedError

class Squeeze(Layer):
    def __init__(self, factor=2):
        super(Squeeze, self).__init__()
        assert factor > 1 and isinstance(factor, int), 'no point of using this if factor <= 1'
        self.factor = factor

    def squeeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert h % self.factor == 0 and w % self.factor == 0, pdb.set_trace()
        
        # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
        x = x.view(bs, c, h // self.factor, self.factor, w // self.factor, self.factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(bs, c * self.factor * self.factor, h // self.factor, w // self.factor)

        return x
 
    def unsqueeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert c >= 4 and c % 4 == 0

        # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
        x = x.view(bs, c // self.factor ** 2, self.factor, self.factor, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(bs, c // self.factor ** 2, h * self.factor, w * self.factor)
        return x
    
    def forward(self, x):
        if len(x.size()) != 4: 
            raise NotImplementedError # Maybe ValueError would be more appropriate

        return self.squeeze_bchw(x)
        
    def inverse(self, x):
        if len(x.size()) != 4: 
            raise NotImplementedError

        return self.unsqueeze_bchw(x)


class Unsqueeze(Layer):
    def __init__(self, factor=2):
        super(Unsqueeze, self).__init__()
        assert factor > 1 and isinstance(factor, int), 'no point of using this if factor <= 1'
        self.factor = factor

    def squeeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert h % self.factor == 0 and w % self.factor == 0, pdb.set_trace()
        
        # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
        x = x.view(bs, c, h // self.factor, self.factor, w // self.factor, self.factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
        x = x.view(bs, c * self.factor * self.factor, h // self.factor, w // self.factor)

        return x
 
    def unsqueeze_bchw(self, x):
        bs, c, h, w = x.size()
        assert c >= 4 and c % 4 == 0

        # taken from https://github.com/chaiyujin/glow-pytorch/blob/master/glow/modules.py
        x = x.view(bs, c // self.factor ** 2, self.factor, self.factor, h, w)
        x = x.permute(0, 1, 4, 2, 5, 3).contiguous()
        x = x.view(bs, c // self.factor ** 2, h * self.factor, w * self.factor)
        return x

    def forward(self, x):
        if len(x.size()) != 4: 
            raise NotImplementedError # Maybe ValueError would be more appropriate

        return self.unsqueeze_bchw(x)
        
    def inverse(self, x):
        if len(x.size()) != 4: 
            raise NotImplementedError

        return self.squeeze_bchw(x)




def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        print(gpu_ids)
        device = torch.device('cuda:{}'.format(gpu_ids[0])) if gpu_ids else torch.device('cpu')
        net.to(device)
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], output_tanh=True, n_downsampling=2):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG.startswith('resnet_') and which_model_netG.endswith('blocks'):
        n_blocks = int(re.findall(r'\d+', which_model_netG)[0])
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=False, n_blocks=n_blocks, output_tanh=output_tanh, n_downsampling=n_downsampling)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain, gpu_ids)

def define_G_enc(input_nc, output_nc, ngf, which_model_netG, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], n_downsampling=2):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG.startswith('resnet_') and which_model_netG.endswith('blocks'):
        n_blocks = int(re.findall(r'\d+', which_model_netG)[0])
        netG = ResnetGenerator_enc(input_nc, output_nc, ngf, norm_layer=norm_layer, n_blocks=n_blocks, n_downsampling=n_downsampling)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain, gpu_ids)

def define_G_core(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_naive=False, 
                  init_type='normal', init_gain=0.02, gpu_ids=[], invertible=False, n_downsampling=2, coupling='additive'):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG.startswith('resnet_') and which_model_netG.endswith('blocks'):
        n_blocks = int(re.findall(r'\d+', which_model_netG)[0])
        netG = ResnetGenerator_core(input_nc, output_nc, ngf, norm_layer=norm_layer, use_naive=use_naive, n_blocks=n_blocks, invertible=invertible, n_downsampling=n_downsampling, coupling=coupling)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain, gpu_ids)

def define_G_dec(input_nc, output_nc, ngf, which_model_netG, norm='batch', init_type='normal', init_gain=0.02, gpu_ids=[], output_tanh=True, n_downsampling=2):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG.startswith('resnet_') and which_model_netG.endswith('blocks'):
        n_blocks = int(re.findall(r'\d+', which_model_netG)[0])
        netG = ResnetGenerator_dec(input_nc, output_nc, ngf, norm_layer=norm_layer, n_blocks=n_blocks, output_tanh=output_tanh, n_downsampling=n_downsampling)
    elif which_model_netG.startswith('noise_') and which_model_netG.endswith('blocks'):
        netG = Noise_dec(input_nc, output_nc, ngf)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'paraml':
        netD = ParamLDiscriminator(input_nc)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    return init_net(netD, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', output_tanh=True, n_downsampling = 2):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if output_tanh:
            model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)



class inv1x1(Layer, nn.Conv2d):
    def __init__(self, num_channels):
        self.num_channels = num_channels
        nn.Conv2d.__init__(self, num_channels, num_channels, 1, bias=False)

    def reset_parameters(self):
        w_init = np.linalg.qr(np.random.randn(self.num_channels, self.num_channels))[0]
        w_init = torch.from_numpy(w_init.astype('float32'))
        w_init = w_init.unsqueeze(-1).unsqueeze(-1)
        self.weight.data.copy_(w_init)

    def forward(self, x):
        output = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, \
            self.dilation, self.groups)
 
        return output

    def inverse(self, x):
        weight_inv = torch.inverse(self.weight.squeeze()).unsqueeze(-1).unsqueeze(-1)
        output = F.conv2d(x, weight_inv, self.bias, self.stride, self.padding, \
                    self.dilation, self.groups)
        return output



class ResnetGenerator_enc(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, n_blocks=6, padding_type='reflect', n_downsampling = 2, coupling='additive'):
        assert(n_blocks >= 0)
        super(ResnetGenerator_enc, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        

        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]
                                      
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)



class ResnetGenerator_core(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_naive=False, n_blocks=6,
                 padding_type='reflect', invertible=False, n_downsampling=2, coupling='additive'):
        assert(n_blocks >= 0)
        super(ResnetGenerator_core, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []

          
        mult = 2**n_downsampling
                
        for i in range(n_blocks):
            if invertible:
                model += [ReversibleResnetBlock(ngf * mult, padding_type, norm_layer, use_naive, use_bias, coupling)]
            else:
                model += [ResnetBlock(mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

                
        self.model = nn.Sequential(*model)

    def forward(self, input, inverse=False):
        out = input
        if inverse:
            for block in reversed(self.model):
                out = block.inverse(out)
        else:
            for block in self.model:
                out = block(out)

        return out


class ResnetGenerator_dec(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, n_blocks=6, padding_type='reflect', output_tanh=True, n_downsampling=2, coupling='additive'):
        assert(n_blocks >= 0)
        super(ResnetGenerator_dec, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = []

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]

        if output_tanh:
            model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ReversibleConvBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, coupling, kernel_size=3):
        super(ReversibleConvBlock, self).__init__()
        F = self.build_conv_block(dim // 2, padding_type, norm_layer, use_dropout, use_bias, kernel_size)
        G = self.build_conv_block(dim // 2, padding_type, norm_layer, use_dropout, use_bias, kernel_size)
        self.rev_block = ReversibleBlock(F, G, coupling)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, kernel_size):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(kernel_size//2)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(kernel_size//2)]
        elif padding_type == 'zero':
            p = kernel_size//2
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        return self.rev_block(x)
        
    def inverse(self, x):
        return self.rev_block.inverse(x)
        
class ReversibleResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_naive, use_bias, coupling):
        super(ReversibleResnetBlock, self).__init__()
        F = self.build_conv_block(dim // 2, padding_type, norm_layer, use_bias)
        G = self.build_conv_block(dim // 2, padding_type, norm_layer, use_bias)
        
        if use_naive:
            self.rev_block = ReversibleBlock(F, G, coupling,
                                             keep_input=True, implementation_fwd=2, implementation_bwd=2)
        else:
            self.rev_block = ReversibleBlock(F, G, coupling)

    def build_conv_block(self, dim, padding_type, norm_layer, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        return self.rev_block(x)
        
    def inverse(self, x):
        return self.rev_block.inverse(x)

class ZeroInit(nn.Conv2d):
    def __init__(self, channels_in, channels_out, filter_size, stride=1, padding=0, logscale=3.):
        super().__init__(channels_in, channels_out, filter_size, stride=stride, padding=padding)

    def reset_parameters(self):
        self.weight.data.zero_()
        self.bias.data.zero_()


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)



class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        out = self.net(input)
        return out

        

