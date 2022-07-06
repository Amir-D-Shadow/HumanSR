'''
@paper: GAN Prior Embedded Network for Blind Face Restoration in the Wild (CVPR2021)
@author: yangxy (yangtao9009@gmail.com)
'''
import math
import random
import functools
import operator
import itertools

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

from math import ceil, floor

class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2, device='cpu'):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)
        self.device = device

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad, device=self.device)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2, device='cpu'):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)
        self.device = device

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad, device=self.device)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1, device='cpu'):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad
        self.device = device

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad, device=self.device)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None, device='cpu'
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation
        self.device = device

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul, device=self.device)

        else:
            out = F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        device='cpu'
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor, device=device)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), device=device)

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self, isconcat=True):
        super().__init__()

        self.isconcat = isconcat
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, channel, height, width = image.shape
            noise = image.new_empty(batch, channel, height, width).normal_()

        if self.isconcat:
            return torch.cat((image, self.weight * noise), dim=1)
        else:
            return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
        isconcat=True,
        device='cpu'
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
            device=device
        )

        self.noise = NoiseInjection(isconcat)
        #self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        #self.activate = ScaledLeakyReLU(0.2)
        feat_multiplier = 2 if isconcat else 1
        self.activate = FusedLeakyReLU(out_channel*feat_multiplier, device=device)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1], device='cpu'):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel, device=device)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False, device=device)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out

class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        isconcat=True,
        narrow=1,
        device='cpu'
    ):
        super().__init__()

        self.size = size
        self.n_mlp = n_mlp
        self.style_dim = style_dim
        self.feat_multiplier = 2 if isconcat else 1

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu', device=device
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow),
            2048: int(8 * channel_multiplier * narrow)
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel, isconcat=isconcat, device=device
        )
        self.to_rgb1 = ToRGB(self.channels[4]*self.feat_multiplier, style_dim, upsample=False, device=device)

        self.log_size = int(math.log(size, 2))

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()

        in_channel = self.channels[4]

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel*self.feat_multiplier,
                    out_channel,
                    3,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    isconcat=isconcat,
                    device=device
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel*self.feat_multiplier, out_channel, 3, style_dim, blur_kernel=blur_kernel, isconcat=isconcat, device=device
                )
            )

            self.to_rgbs.append(ToRGB(out_channel*self.feat_multiplier, style_dim, device=device))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
    ):
        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            '''
            noise = [None] * (2 * (self.log_size - 2) + 1)
            '''
            noise = []
            batch = styles[0].shape[0]
            for i in range(self.n_mlp + 1):
                size = 2 ** (i+2)
                noise.append(torch.randn(batch, self.channels[size], size, size, device=styles[0].device))
            
        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

        else:
            if inject_index is None:
                #inject_index = random.randint(1, self.n_latent - 1)
                inject_index = int( self.n_latent // 2 )

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent

        else:
            return image, None

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
        device='cpu'
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1), device=device))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel, device=device))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1], device='cpu'):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3, device=device)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True, device=device)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out

class KernelPredictor(nn.Module):

    def __init__(self, in_channels, out_channels, n_groups, style_channels, kernel_size, device = 'cpu'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.w_channels = style_channels
        self.n_groups = n_groups
        self.kernel_size = kernel_size

        self.spatial = nn.Sequential(
            ConvLayer(in_channel = style_channels,
            out_channel = in_channels * in_channels // n_groups,
            kernel_size = 1,
            activate = False,
            downsample = False,
            device = device        
            ),
            nn.AdaptiveAvgPool2d((kernel_size, kernel_size))
        )


        self.pointwise = nn.Sequential(
            ConvLayer(in_channel = style_channels,
            out_channel = in_channels * out_channels // self.n_groups,
            kernel_size = 1,
            activate = False,
            downsample = False,
            device = device        
            ),
            nn.AdaptiveAvgPool2d((1, 1))
        )


    def forward(self, w):

        w_spatial = self.spatial(w)
        w_spatial = w_spatial.view(w.shape[0],
                                   self.in_channels,
                                   self.in_channels // self.n_groups,
                                   self.kernel_size, self.kernel_size)


        w_pointwise = self.pointwise(w)
        w_pointwise = w_pointwise.view(w.shape[0],
                                       self.out_channels,
                                       self.in_channels // self.n_groups,
                                       1, 1)

        return w_spatial, w_pointwise


class AdaConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, n_groups, kernel_size, device = 'cpu'):
        super().__init__()
        self.n_groups =  n_groups 
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.spatial_weight = nn.Parameter(
            torch.randn(1, self.in_channels, self.in_channels // self.n_groups, 1, 1)
        )

        self.spatial_scale = 1 / math.sqrt(in_channels * kernel_size * kernel_size / self.n_groups)

        self.pointwise_weight = nn.Parameter(
            torch.randn(1, self.out_channels, self.in_channels, 1, 1)
        )

        self.pointwise_scale = 1 / math.sqrt(in_channels * 1 * 1)

    def forward(self, x, w_spatial, w_pointwise):
        assert len(x) == len(w_spatial) == len(w_pointwise)

        batch = x.shape[0]
        w_spatial = self.spatial_scale * self.spatial_weight * w_spatial
        demod = torch.rsqrt(w_spatial.pow(2).sum([2, 3, 4]) + 1e-8)
        w_spatial = w_spatial * demod.view(batch, self.in_channels, 1, 1, 1)

        w_pointwise = self.pointwise_scale * self.pointwise_weight * w_pointwise
        demod = torch.rsqrt(w_pointwise.pow(2).sum([2, 3, 4]) + 1e-8)
        w_pointwise = w_pointwise * demod.view(batch, self.out_channels, 1, 1, 1)

        # F.conv2d does not work with batched filters (as far as I can tell)...
        # Hack for inputs with > 1 sample
        ys = []
        for i in range(len(x)):
            y = self._forward_single(x[i:i + 1], w_spatial[i], w_pointwise[i])
            ys.append(y)
        ys = torch.cat(ys, dim=0)

        return ys

    def _forward_single(self, x, w_spatial, w_pointwise):
        # Only square kernels
        assert w_spatial.size(-1) == w_spatial.size(-2)
        padding = (w_spatial.size(-1) - 1) / 2
        pad = (ceil(padding), floor(padding), ceil(padding), floor(padding))

        x = F.pad(x, pad=pad, mode='reflect')
        x = F.conv2d(x, w_spatial, groups=self.n_groups)
        x = F.conv2d(x, w_pointwise)
        return x

class Adablocks(nn.Module):

    def __init__(self, in_channels, mid_channels, out_channels, n_groups, style_channels, kernel_size, device = 'cpu'):

        super(Adablocks,self).__init__()

        self.kernel_generator = KernelPredictor(in_channels = in_channels, 
        out_channels = mid_channels, 
        n_groups = n_groups, 
        style_channels = style_channels, 
        kernel_size = kernel_size, 
        device = device
        )

        self.conv1 = AdaConv2d(in_channels = in_channels,
        out_channels = mid_channels, 
        n_groups = n_groups, 
        kernel_size = kernel_size,
        device = device
        )

        self.conv2 = ConvLayer(in_channel = mid_channels,
        out_channel = out_channels,
        kernel_size = 1,
        activate = False,
        downsample = False,
        device = device        
        )

        self.act_fn = FusedLeakyReLU(mid_channels, device=device)#nn.LeakyReLU(negative_slope=0.2)

    def forward(self,x,feat):

        w_s, w_p = self.kernel_generator(feat)
        y = self.conv1(x, w_s, w_p)
        y = self.act_fn(y)
        y = self.conv2(y)

        return y
    

class FullGenerator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        isconcat=True,
        narrow=1,
        device='cpu'
    ):
        super().__init__()
        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow),
            2048: int(8 * channel_multiplier * narrow)
        }

        self.log_size = int(math.log(size, 2))
        self.generator = Generator(size, style_dim, n_mlp, channel_multiplier=channel_multiplier, blur_kernel=blur_kernel, lr_mlp=lr_mlp, isconcat=isconcat, narrow=narrow, device=device)
        
        self.lq_convs = nn.ModuleList()
        conv = [ConvLayer(3, channels[size], 1, device=device)]
        self.lq_convs.append( nn.Sequential(*conv) )
        in_channel = channels[size]
        for i in range(self.log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            conv = [ConvLayer(in_channel, out_channel, 3, downsample=True, device=device)] 
            self.lq_convs.append(nn.Sequential(*conv))
            in_channel = out_channel

        self.mapE = []
        self.mapE.append(ConvLayer(in_channel, style_dim, 1, downsample=False, activate=False, device=device))
        for i in range(n_mlp):
            self.mapE.append(ConvLayer(style_dim, style_dim, 1, downsample=False, activate=True, device=device))
        self.mapE = nn.Sequential(*self.mapE)

        #HQ Ref 
        in_channel = channels[size] 
        groups_div = 1
        self.hq_convs = nn.ModuleList()

        self.hq_convs.append( Adablocks(in_channels=in_channel, 
        mid_channels = in_channel, 
        out_channels = in_channel,
        n_groups = in_channel//groups_div, 
        style_channels = style_dim, 
        kernel_size = 3, 
        device = device)
        )

        for i in range(self.log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            self.hq_convs.append( Adablocks(in_channels=out_channel, 
            mid_channels = out_channel*2, 
            out_channels = out_channel,
            n_groups = out_channel//groups_div, 
            style_channels = style_dim, 
            kernel_size = 3, 
            device = device)
            )

            in_channel = out_channel

        self.final_linear = nn.Sequential(EqualLinear(channels[4] * 4 * 4, style_dim, activation='fused_lrelu', device=device))

    def forward(self,
        xl,xr,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
    ):

        y = xl
        for convs1 in self.lq_convs:
            ecd = convs1
            y = ecd(y)

        LQ_feat = y.clone()
        LQ_feat = LQ_feat.view(LQ_feat.shape[0],-1)
        LQ_feat = self.final_linear(LQ_feat)

        #HQ
        exStyleW = self.mapE(y)
        hq_batch = xr.shape[0]
        hq_noise = []
        y_for = self.lq_convs[0](xr)
        y_skip = self.hq_convs[0](y_for,exStyleW.clone().expand(hq_batch,-1,-1,-1))
        hq_noise.append(y_skip)
        for conv1, conv2 in zip(self.lq_convs[1:], self.hq_convs[1:]):
            y_for = conv1(y_for)
            y_skip = conv2(y_for,exStyleW.clone().expand(hq_batch,-1,-1,-1))
            hq_noise.append(y_skip)

        HQ_feat = y_for.view(y_for.shape[0], -1)
        HQ_feat = self.final_linear(HQ_feat)

        cos_sim = torch.matmul(HQ_feat, LQ_feat.permute(1,0).contiguous()) / ( torch.sqrt(torch.square(LQ_feat).sum(dim=-1,keepdim=True)) * torch.sqrt(torch.square(HQ_feat).sum(dim=-1,keepdim=True)))
        #cos_sim = (cos_sim + 1) / 2
        cos_sim = F.softmax(cos_sim,dim=0)
        HQ_feat = torch.sum(cos_sim * HQ_feat ,dim=0 ,keepdim=True)
        hq_noise = list(itertools.chain.from_iterable(itertools.repeat(torch.sum(x*cos_sim.view(hq_batch,1,1,1), dim=0, keepdim=True), 2) for x in hq_noise))[::-1]
        outs = self.generator([HQ_feat,LQ_feat], return_latents, inject_index, truncation, truncation_latent, input_is_latent, noise=hq_noise[1:])
        return outs

class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1], narrow=1, device='cpu'):
        super().__init__()

        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow),
            2048: int(8 * channel_multiplier * narrow)
        }

        convs = [ConvLayer(3, channels[size], 1, device=device)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel, device=device))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3, device=device)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu', device=device),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)
        return out

class FullGenerator_SR(nn.Module):
    def __init__(
        self,
        in_size,
        out_size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        isconcat=True,
        narrow=1,
        device='cpu'
    ):
        super().__init__()
        channels = {
            4: int(512 * narrow),
            8: int(512 * narrow),
            16: int(512 * narrow),
            32: int(512 * narrow),
            64: int(256 * channel_multiplier * narrow),
            128: int(128 * channel_multiplier * narrow),
            256: int(64 * channel_multiplier * narrow),
            512: int(32 * channel_multiplier * narrow),
            1024: int(16 * channel_multiplier * narrow),
            2048: int(8 * channel_multiplier * narrow),
        }

        self.log_insize = int(math.log(in_size, 2))
        self.log_outsize = int(math.log(out_size, 2))
        self.generator = Generator(out_size, style_dim, n_mlp, channel_multiplier=channel_multiplier, blur_kernel=blur_kernel, lr_mlp=lr_mlp, isconcat=isconcat, narrow=narrow, device=device)

        conv = [ConvLayer(3, channels[in_size], 1, device=device)]
        self.ecd0 = nn.Sequential(*conv)
        in_channel = channels[in_size]

        self.names = ['ecd%d'%i for i in range(self.log_insize-1)]
        for i in range(self.log_insize, 2, -1):
            out_channel = channels[2 ** (i - 1)]
            #conv = [ResBlock(in_channel, out_channel, blur_kernel)]
            conv = [ConvLayer(in_channel, out_channel, 3, downsample=True, device=device)]
            setattr(self, self.names[self.log_insize-i+1], nn.Sequential(*conv))
            in_channel = out_channel
        self.final_linear = nn.Sequential(EqualLinear(channels[4] * 4 * 4, style_dim, activation='fused_lrelu', device=device))

    def forward(self,
        inputs,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
    ):
        noise = []
        for i in range(self.log_outsize-self.log_insize):
            noise.append(None)
        for i in range(self.log_insize-1):
            ecd = getattr(self, self.names[i])
            inputs = ecd(inputs)
            noise.append(inputs)
            #print(inputs.shape)
        inputs = inputs.view(inputs.shape[0], -1)
        outs = self.final_linear(inputs)
        #print(outs.shape)
        noise = list(itertools.chain.from_iterable(itertools.repeat(x, 2) for x in noise))[::-1]
        image, latent = self.generator([outs], return_latents, inject_index, truncation, truncation_latent, input_is_latent, noise=noise[1:])
        return image, latent

if __name__ == "__main__":

    x = torch.randn(2,256,32,32)
    eStyle = torch.randn(1,512,4,4)
    net = Adablocks(in_channels=512, mid_channels = 512, out_channels=256,n_groups = 256, style_channels = 512, kernel_size = 3, device = 'cpu')
    x = torch.cat([x,torch.randn(2,256,32,32)],dim=1)
    y = net(x,eStyle.clone().expand(2,-1,-1,-1))
    print(y.shape)