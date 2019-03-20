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
        norm_layer = functools.partial(nn.BatchNorm3d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm3d, affine=False, track_running_stats=True)
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
        elif classname.find('BatchNorm3d') != -1:
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


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_naive=False, init_type='normal', init_gain=0.02, gpu_ids=[], n_downsampling=2):
    netG = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netG.startswith('srcnn_'):
        depth = int(re.findall(r'\d+', which_model_netG)[0])
        netG = SrcnnGenerator3d(input_nc, output_nc, depth, ngf)
    elif which_model_netG.startswith('edsrF_'):
        depth = int(re.findall(r'\d+', which_model_netG)[0])
        netG = EdsrFGenerator3d(input_nc, output_nc, depth, ngf, use_naive)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    return init_net(netG, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    netD = None
    norm_layer = get_norm_layer(norm_type=norm)

    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=2, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif which_model_netD == 'pixel':
        netD = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
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


class ZeroInit(nn.Conv3d):
    def reset_parameters(self):
        self.weight.data.zero_()
        self.bias.data.zero_()

class ThickBlocknaive3d(nn.Module):
    def __init__(self, dim, use_bias):
        super(ThickBlocknaive3d, self).__init__()
        self.F = self.build_conv_block(dim, True)

    def build_conv_block(self, dim, use_bias):
        conv_block = []
        conv_block += [nn.InstanceNorm3d(dim)]
        conv_block += [nn.ReplicationPad3d(1)]
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=0, bias=use_bias)]
        conv_block += [nn.InstanceNorm3d(dim)]
        conv_block += [nn.ReLU(True)]
        conv_block += [nn.ReplicationPad3d(1)]
        conv_block += [ZeroInit(dim, dim, kernel_size=3, padding=0, bias=use_bias)]


        return nn.Sequential(*conv_block)

    def forward(self, input):
        return self.F(input) + input


class ThickBlock3d(nn.Module):
    def __init__(self, dim, use_bias, use_naive=False):
        super(ThickBlock3d, self).__init__()
        F = self.build_conv_block(dim // 2, True)
        G = self.build_conv_block(dim // 2, True)
        if use_naive:
            self.rev_block = ReversibleBlock(F, G, 'additive',
                                             keep_input=True, implementation_fwd=2, implementation_bwd=2)
        else:
            self.rev_block = ReversibleBlock(F, G, 'additive')


    def build_conv_block(self, dim, use_bias):
        conv_block = []
        conv_block += [nn.InstanceNorm3d(dim)]
        conv_block += [nn.ReplicationPad3d(1)]
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=0, bias=use_bias)]
        conv_block += [nn.InstanceNorm3d(dim)]
        conv_block += [nn.ReLU(True)]
        conv_block += [nn.ReplicationPad3d(1)]
        conv_block += [ZeroInit(dim, dim, kernel_size=3, padding=0, bias=use_bias)]


        return nn.Sequential(*conv_block)

    def forward(self, x):
        return self.rev_block(x)

    def inverse(self, x):
        return self.rev_block.inverse(x)



class RevBlock3d(nn.Module):
    def __init__(self, dim, use_bias, norm_layer, use_naive):
        super(RevBlock3d, self).__init__()
        self.F = self.build_conv_block(dim // 2, True, norm_layer)
        self.G = self.build_conv_block(dim // 2, True, norm_layer)
        if use_naive:
            self.rev_block = ReversibleBlock(F, G, 'additive',
                                             keep_input=True, implementation_fwd=2, implementation_bwd=2)
        else:
            self.rev_block = ReversibleBlock(F, G, 'additive')

    def build_conv_block(self, dim, use_bias, norm_layer):
        conv_block = []
        conv_block += [nn.ReplicationPad3d(1)]
        conv_block += [nn.Conv3d(dim, dim, kernel_size=3, padding=0, bias=use_bias)]
        conv_block += [norm_layer(dim)]
        conv_block += [nn.ReLU(True)]
        conv_block += [nn.ReplicationPad3d(1)]
        conv_block += [ZeroInit(dim, dim, kernel_size=3, padding=0, bias=use_bias)]


        return nn.Sequential(*conv_block)

    def forward(self, x):
        return self.rev_block(x)

    def inverse(self, x):
        return self.rev_block.inverse(x)



class PixelUnshuffle3D(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelUnshuffle3D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_unshuffle_3d(input, self.upscale_factor)

    def __repr__(self):
        return self.__class__.__name__ + '(upscale_factor=' + str(self.upscale_factor) + ')'


def pixel_unshuffle_3d(input, upscale_factor):
    batch_size, channels, in_height, in_width, in_depth = input.size()

    out_channels = channels * upscale_factor ** 3
    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor
    out_depth = in_depth // upscale_factor

    fm_view = input.view(batch_size, channels, out_height, upscale_factor, out_width, upscale_factor, out_depth, upscale_factor)
    fm_perm = fm_view.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
    return fm_perm.view(batch_size, out_channels, out_height, out_width, out_depth)





class PixelShuffle3D(nn.Module):
    def __init__(self, upscale_factor):
        super(PixelShuffle3D, self).__init__()
        self.upscale_factor = upscale_factor

    def forward(self, input):
        return pixel_shuffle_3d(input, self.upscale_factor)

    def __repr__(self):
        return self.__class__.__name__ + '(upscale_factor=' + str(self.upscale_factor) + ')'


def pixel_shuffle_3d(input, upscale_factor):
    batch_size, channels, in_height, in_width, in_depth = input.size()
    channels //= upscale_factor ** 3

    out_height = in_height * upscale_factor
    out_width = in_width * upscale_factor
    out_depth = in_depth * upscale_factor

    input_view = input.contiguous().view(
        batch_size, channels, upscale_factor, upscale_factor, upscale_factor,
        in_height, in_width, in_depth)

    shuffle_out = input_view.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    return shuffle_out.view(batch_size, channels, out_height, out_width, out_depth)


class EdsrFGenerator3d(nn.Module):
    def __init__(self, input_nc, output_nc, depth, ngf=64, use_naive=False):
        super(EdsrFGenerator3d, self).__init__()

        use_bias = True
        downconv_ab = [nn.ReplicationPad3d(2),
                       nn.Conv3d(input_nc, ngf, kernel_size=5,
                                 stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm3d(ngf),
                       nn.ReLU(True),
                       nn.Conv3d(ngf, ngf * 2, kernel_size=3,
                                 stride=2, padding=1, bias=use_bias),
                       nn.InstanceNorm3d(ngf * 2),
                       nn.ReLU(True)]
        downconv_ba = [nn.ReplicationPad3d(2),
                       nn.Conv3d(input_nc, ngf, kernel_size=5,
                                 stride=1, padding=0, bias=use_bias),
                       nn.InstanceNorm3d(ngf),
                       nn.ReLU(True),
                       nn.Conv3d(ngf, ngf * 2, kernel_size=3,
                                 stride=2, padding=1, bias=use_bias),
                       nn.InstanceNorm3d(ngf * 2),
                       nn.ReLU(True)]

        core = []
        for _ in range(depth):
            core += [ThickBlock3d(ngf * 2, use_bias, use_naive)]

        upconv_ab = [nn.ConvTranspose3d(ngf * 2, ngf,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                       bias=use_bias),
                     nn.InstanceNorm3d(ngf),
                     nn.ReLU(True),
                     nn.ReplicationPad3d(2),
                     nn.Conv3d(ngf, output_nc, kernel_size=5, padding=0),
                     nn.Tanh()]
        upconv_ba = [nn.ConvTranspose3d(ngf * 2, ngf,
                                        kernel_size=3, stride=2,
                                        padding=1, output_padding=1,
                                       bias=use_bias),
                     nn.InstanceNorm3d(ngf),
                     nn.ReLU(True),
                     nn.ReplicationPad3d(2),
                     nn.Conv3d(ngf, output_nc, kernel_size=5, padding=0),
                     nn.Tanh()]

        self.downconv_ab = nn.Sequential(*downconv_ab)
        self.downconv_ba = nn.Sequential(*downconv_ba)
        self.core = nn.ModuleList(core)
        self.upconv_ab = nn.Sequential(*upconv_ab)
        self.upconv_ba = nn.Sequential(*upconv_ba)

    def forward(self, input, inverse=False):
        out = input

        if inverse:
            out = self.downconv_ba(out)
            for block in reversed(self.core):
                out = block.inverse(out)
            return self.upconv_ba(out)
        else:
            out = self.downconv_ab(out)
            for block in self.core:
                out = block(out)
            return self.upconv_ab(out)



class SrcnnGenerator3d(nn.Module):
    def __init__(self, input_nc, output_nc, depth, ngf=64, norm_layer=nn.BatchNorm3d, use_naive=False):
        super(SrcnnGenerator3d, self).__init__()

        use_bias = True
        downconv_ab = [nn.ReplicationPad3d(1),
                    nn.Conv3d(input_nc, ngf, kernel_size=3,
                              stride=1, padding=0, bias=use_bias)]
        downconv_ba = [nn.ReplicationPad3d(1),
                    nn.Conv3d(input_nc, ngf, kernel_size=3,
                              stride=1, padding=0, bias=use_bias)]

        core = []
        for _ in range(depth):
            core += [RevBlock3d(ngf, use_bias, norm_layer, use_naive)]

        upconv_ab = [nn.Conv3d(ngf, output_nc, kernel_size=1,
                            stride=1, padding=0, bias=use_bias)]
        upconv_ba = [nn.Conv3d(ngf, output_nc, kernel_size=1,
                            stride=1, padding=0, bias=use_bias)]

        self.downconv_ab = nn.Sequential(*downconv_ab)
        self.downconv_ba = nn.Sequential(*downconv_ba)
        self.core = nn.ModuleList(core)
        self.upconv_ab = nn.Sequential(*upconv_ab)
        self.upconv_ba = nn.Sequential(*upconv_ba)

    def forward(self, input, inverse=False):
        orig_shape = input.shape[2:]
        out = input

        if inverse:
            out = self.downconv_ab(out)
            for block in reversed(self.core):
                out = block.inverse(out)
            out = self.upconv_ab(out)
        else:
            out = self.downconv_ba(out)
            for block in self.core:
                out = block(out)
            out = self.upconv_ba(out)
        return out


class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm3d, use_dropout=False):
        super(UnetGenerator, self).__init__()

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 4, input_nc=None, submodule=None, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm3d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv3d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose3d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose3d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]
            
            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([x, self.model(x)], 1)




# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=2, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv3d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv3d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv3d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm3d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm3d
        else:
            use_bias = norm_layer == nn.InstanceNorm3d

        self.net = [
            nn.Conv3d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv3d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)

        
