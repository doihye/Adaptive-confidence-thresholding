from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from numbers import Number
from itertools import repeat

import numpy as np
import torch
import torch.nn as nn

import torchvision.models as models
import pprint
from torch.autograd.function import Function, once_differentiable
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from torch._thnn import type2backend

F = nn.functional
DEBUG = False
NUM_INPUT_CHANNELS = 3

h = 192
w = 480
BATCH_SIZE = 8

vgg16_dims = [
    (64, 64, 'M'),  # Stage - 1
    (128, 128, 'M'),  # Stage - 2
    (256, 256, 256, 'M'),  # Stage - 3
    (512, 512, 512, 'M'),  # Stage - 4
    (512, 512, 512, 'M')  # Stage - 5
]

decoder_dims = [
    ('U', 512, 512, 512),  # Stage - 5
    ('U', 512, 512, 512),  # Stage - 4
    ('U', 256, 256, 256),  # Stage - 3
    ('U', 128, 128),  # Stage - 2
    ('U', 64, 64)  # Stage - 1
]

def convert_to_single_channel(x):
    bs, ch, h, w = x.shape
    if ch != 1:
        x = x.reshape(bs * ch, 1, h, w)
    return x, ch


def recover_from_single_channel(x, ch):
    if ch != 1:
        bs_ch, _ch, h, w = x.shape
        assert _ch == 1
        assert bs_ch % ch == 0
        x = x.reshape(bs_ch // ch, ch, h, w)
    return x


def repeat_for_channel(x, ch):
    if ch != 1:
        bs, _ch, h, w = x.shape
        x = x.repeat(1, ch, 1, 1).reshape(bs * ch, _ch, h, w)
    return x


def th_rmse(pred, gt):
    return (pred - gt).pow(2).mean(dim=3).mean(dim=2).sum(dim=1).sqrt().mean()


def th_epe(pred, gt, small_flow=-1.0, unknown_flow_thresh=1e7):
    pred_u, pred_v = pred[:, 0].contiguous().view(-1), pred[:, 1].contiguous().view(-1)
    gt_u, gt_v = gt[:, 0].contiguous().view(-1), gt[:, 1].contiguous().view(-1)
    if gt_u.abs().max() > unknown_flow_thresh or gt_v.abs().max() > unknown_flow_thresh:
        idx_unknown = ((gt_u.abs() > unknown_flow_thresh) + (gt_v.abs() > unknown_flow_thresh)).nonzero()[:, 0]
        pred_u[idx_unknown] = 0
        pred_v[idx_unknown] = 0
        gt_u[idx_unknown] = 0
        gt_v[idx_unknown] = 0
    epe = ((pred_u - gt_u).pow(2) + (pred_v - gt_v).pow(2)).sqrt()
    if small_flow >= 0.0 and (gt_u.abs().min() <= small_flow or gt_v.abs().min() <= small_flow):
        idx_valid = ((gt_u.abs() > small_flow) + (gt_v.abs() > small_flow)).nonzero()[:, 0]
        epe = epe[idx_valid]
    return epe.mean()


class SegNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(SegNet, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.num_channels = input_channels

        self.vgg16 = models.vgg16(pretrained=True)

        # Encoder layers

        self.encoder_conv_00 = nn.Sequential(*[
            nn.Conv2d(in_channels=self.input_channels,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64)
        ])
        self.encoder_conv_01 = nn.Sequential(*[
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64)
        ])
        self.encoder_conv_10 = nn.Sequential(*[
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(128)
        ])
        self.encoder_conv_11 = nn.Sequential(*[
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(128)
        ])
        self.encoder_conv_20 = nn.Sequential(*[
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256)
        ])
        self.encoder_conv_21 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256)
        ])
        self.encoder_conv_22 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256)
        ])
        self.encoder_conv_30 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        self.encoder_conv_31 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        self.encoder_conv_32 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        self.encoder_conv_40 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        self.encoder_conv_41 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        self.encoder_conv_42 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])

        self.init_vgg_weigts()

        # Decoder layers

        self.decoder_convtr_42 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        self.decoder_convtr_41 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        self.decoder_convtr_40 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        self.decoder_convtr_32 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        self.decoder_convtr_31 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        self.decoder_convtr_30 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256)
        ])
        self.decoder_convtr_22 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256)
        ])
        self.decoder_convtr_21 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256)
        ])
        self.decoder_convtr_20 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(128)
        ])
        self.decoder_convtr_11 = nn.Sequential(*[
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(128)
        ])
        self.decoder_convtr_10 = nn.Sequential(*[
            nn.Conv2d(in_channels=128,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64)
        ])
        self.decoder_convtr_01 = nn.Sequential(*[
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64)
        ])
        self.decoder_convtr_00 = nn.Sequential(*[
            nn.Conv2d(in_channels=64,
                      out_channels=self.output_channels,
                      kernel_size=3,
                      padding=1)
        ])

        self.pacconv = PacConv2d(1, 1, 5, padding=2)
        self.conv1by1 = nn.Conv2d(in_channels=64+128+256+512+512, out_channels=1, kernel_size=1)

    def forward(self, input_img):

        activations = list()
        indiceslist = list()
        dimlist = list()

        # Encoder

        # Encoder Stage - 1
        dim_0 = input_img.size()
        x_00 = F.relu(self.encoder_conv_00(input_img))
        x_01 = F.relu(self.encoder_conv_01(x_00))
        x_0, indices_0 = F.max_pool2d(x_01, kernel_size=2, stride=2, return_indices=True)

        activations.append(x_0)
        indiceslist.append(indices_0)


        # Encoder Stage - 2
        dim_1 = x_0.size()
        x_10 = F.relu(self.encoder_conv_10(x_0))
        x_11 = F.relu(self.encoder_conv_11(x_10))
        x_1, indices_1 = F.max_pool2d(x_11, kernel_size=2, stride=2, return_indices=True)

        activations.append(x_1)
        indiceslist.append(indices_1)

        # Encoder Stage - 3
        dim_2 = x_1.size()
        x_20 = F.relu(self.encoder_conv_20(x_1))
        x_21 = F.relu(self.encoder_conv_21(x_20))
        x_22 = F.relu(self.encoder_conv_22(x_21))
        x_2, indices_2 = F.max_pool2d(x_22, kernel_size=2, stride=2, return_indices=True)

        activations.append(x_2)
        indiceslist.append(indices_2)

        # Encoder Stage - 4
        dim_3 = x_2.size()
        x_30 = F.relu(self.encoder_conv_30(x_2))
        x_31 = F.relu(self.encoder_conv_31(x_30))
        x_32 = F.relu(self.encoder_conv_32(x_31))
        x_3, indices_3 = F.max_pool2d(x_32, kernel_size=2, stride=2, return_indices=True)

        activations.append(x_3)
        indiceslist.append(indices_3)

        # Encoder Stage - 5
        dim_4 = x_3.size()
        x_40 = F.relu(self.encoder_conv_40(x_3))
        x_41 = F.relu(self.encoder_conv_41(x_40))
        x_42 = F.relu(self.encoder_conv_42(x_41))
        x_4, indices_4 = F.max_pool2d(x_42, kernel_size=2, stride=2, return_indices=True)

        activations.append(x_4)
        indiceslist.append(indices_4)

        a0 = F.interpolate(activations[0], size=(h, w), mode='bilinear', align_corners=True)
        a1 = F.interpolate(activations[1], size=(h, w), mode='bilinear', align_corners=True)
        a2 = F.interpolate(activations[2], size=(h, w), mode='bilinear', align_corners=True)
        a3 = F.interpolate(activations[3], size=(h, w), mode='bilinear', align_corners=True)
        a4 = F.interpolate(activations[4], size=(h, w), mode='bilinear', align_corners=True)

        aconcat = torch.cat([a0, a1, a2, a3, a4], dim=1)

        aconcatresult = self.conv1by1(aconcat)

        # Decoder

        dim_d = x_4.size()

        # Decoder Stage - 5
        x_4d = F.max_unpool2d(x_4, indices_4, kernel_size=2, stride=2, output_size=dim_4)
        x_42d = F.relu(self.decoder_convtr_42(x_4d))
        x_41d = F.relu(self.decoder_convtr_41(x_42d))
        x_40d = F.relu(self.decoder_convtr_40(x_41d))
        dim_4d = x_40d.size()

        # Decoder Stage - 4
        x_3d = F.max_unpool2d(x_40d, indices_3, kernel_size=2, stride=2, output_size=dim_3)
        x_32d = F.relu(self.decoder_convtr_32(x_3d))
        x_31d = F.relu(self.decoder_convtr_31(x_32d))
        x_30d = F.relu(self.decoder_convtr_30(x_31d))
        dim_3d = x_30d.size()

        # Decoder Stage - 3
        x_2d = F.max_unpool2d(x_30d, indices_2, kernel_size=2, stride=2, output_size=dim_2)
        x_22d = F.relu(self.decoder_convtr_22(x_2d))
        x_21d = F.relu(self.decoder_convtr_21(x_22d))
        x_20d = F.relu(self.decoder_convtr_20(x_21d))
        dim_2d = x_20d.size()

        # Decoder Stage - 2
        x_1d = F.max_unpool2d(x_20d, indices_1, kernel_size=2, stride=2, output_size=dim_1)
        x_11d = F.relu(self.decoder_convtr_11(x_1d))
        x_10d = F.relu(self.decoder_convtr_10(x_11d))
        dim_1d = x_10d.size()

        # Decoder Stage - 1
        x_0d = F.max_unpool2d(x_10d, indices_0, kernel_size=2, stride=2, output_size=dim_0)
        x_01d = F.relu(self.decoder_convtr_01(x_0d))
        x_00d = self.decoder_convtr_00(x_01d)
        dim_0d = x_00d.size()
        depthresult = x_00d

        if DEBUG:
            print("dim_0: {}".format(dim_0))
            print("dim_1: {}".format(dim_1))
            print("dim_2: {}".format(dim_2))
            print("dim_3: {}".format(dim_3))
            print("dim_4: {}".format(dim_4))

            print("dim_d: {}".format(dim_d))
            print("dim_4d: {}".format(dim_4d))
            print("dim_3d: {}".format(dim_3d))
            print("dim_2d: {}".format(dim_2d))
            print("dim_1d: {}".format(dim_1d))
            print("dim_0d: {}".format(dim_0d))

        pacresult = self.pacconv(depthresult, aconcatresult)

        return depthresult, activations, indiceslist, pacresult

    def init_vgg_weigts(self):#optional
        assert self.encoder_conv_00[0].weight.size() == self.vgg16.features[0].weight.size()
        self.encoder_conv_00[0].weight.data = self.vgg16.features[0].weight.data
        assert self.encoder_conv_00[0].bias.size() == self.vgg16.features[0].bias.size()
        self.encoder_conv_00[0].bias.data = self.vgg16.features[0].bias.data

        assert self.encoder_conv_01[0].weight.size() == self.vgg16.features[2].weight.size()
        self.encoder_conv_01[0].weight.data = self.vgg16.features[2].weight.data
        assert self.encoder_conv_01[0].bias.size() == self.vgg16.features[2].bias.size()
        self.encoder_conv_01[0].bias.data = self.vgg16.features[2].bias.data

        assert self.encoder_conv_10[0].weight.size() == self.vgg16.features[5].weight.size()
        self.encoder_conv_10[0].weight.data = self.vgg16.features[5].weight.data
        assert self.encoder_conv_10[0].bias.size() == self.vgg16.features[5].bias.size()
        self.encoder_conv_10[0].bias.data = self.vgg16.features[5].bias.data

        assert self.encoder_conv_11[0].weight.size() == self.vgg16.features[7].weight.size()
        self.encoder_conv_11[0].weight.data = self.vgg16.features[7].weight.data
        assert self.encoder_conv_11[0].bias.size() == self.vgg16.features[7].bias.size()
        self.encoder_conv_11[0].bias.data = self.vgg16.features[7].bias.data

        assert self.encoder_conv_20[0].weight.size() == self.vgg16.features[10].weight.size()
        self.encoder_conv_20[0].weight.data = self.vgg16.features[10].weight.data
        assert self.encoder_conv_20[0].bias.size() == self.vgg16.features[10].bias.size()
        self.encoder_conv_20[0].bias.data = self.vgg16.features[10].bias.data

        assert self.encoder_conv_21[0].weight.size() == self.vgg16.features[12].weight.size()
        self.encoder_conv_21[0].weight.data = self.vgg16.features[12].weight.data
        assert self.encoder_conv_21[0].bias.size() == self.vgg16.features[12].bias.size()
        self.encoder_conv_21[0].bias.data = self.vgg16.features[12].bias.data

        assert self.encoder_conv_22[0].weight.size() == self.vgg16.features[14].weight.size()
        self.encoder_conv_22[0].weight.data = self.vgg16.features[14].weight.data
        assert self.encoder_conv_22[0].bias.size() == self.vgg16.features[14].bias.size()
        self.encoder_conv_22[0].bias.data = self.vgg16.features[14].bias.data

        assert self.encoder_conv_30[0].weight.size() == self.vgg16.features[17].weight.size()
        self.encoder_conv_30[0].weight.data = self.vgg16.features[17].weight.data
        assert self.encoder_conv_30[0].bias.size() == self.vgg16.features[17].bias.size()
        self.encoder_conv_30[0].bias.data = self.vgg16.features[17].bias.data

        assert self.encoder_conv_31[0].weight.size() == self.vgg16.features[19].weight.size()
        self.encoder_conv_31[0].weight.data = self.vgg16.features[19].weight.data
        assert self.encoder_conv_31[0].bias.size() == self.vgg16.features[19].bias.size()
        self.encoder_conv_31[0].bias.data = self.vgg16.features[19].bias.data

        assert self.encoder_conv_32[0].weight.size() == self.vgg16.features[21].weight.size()
        self.encoder_conv_32[0].weight.data = self.vgg16.features[21].weight.data
        assert self.encoder_conv_32[0].bias.size() == self.vgg16.features[21].bias.size()
        self.encoder_conv_32[0].bias.data = self.vgg16.features[21].bias.data

        assert self.encoder_conv_40[0].weight.size() == self.vgg16.features[24].weight.size()
        self.encoder_conv_40[0].weight.data = self.vgg16.features[24].weight.data
        assert self.encoder_conv_40[0].bias.size() == self.vgg16.features[24].bias.size()
        self.encoder_conv_40[0].bias.data = self.vgg16.features[24].bias.data

        assert self.encoder_conv_41[0].weight.size() == self.vgg16.features[26].weight.size()
        self.encoder_conv_41[0].weight.data = self.vgg16.features[26].weight.data
        assert self.encoder_conv_41[0].bias.size() == self.vgg16.features[26].bias.size()
        self.encoder_conv_41[0].bias.data = self.vgg16.features[26].bias.data

        assert self.encoder_conv_42[0].weight.size() == self.vgg16.features[28].weight.size()
        self.encoder_conv_42[0].weight.data = self.vgg16.features[28].weight.data
        assert self.encoder_conv_42[0].bias.size() == self.vgg16.features[28].bias.size()
        self.encoder_conv_42[0].bias.data = self.vgg16.features[28].bias.data

def nd2col(input_nd, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, transposed=False,
           use_pyinn_if_possible=False):

    n_dims = len(input_nd.shape[2:])
    kernel_size = (kernel_size,) * n_dims if isinstance(kernel_size, Number) else kernel_size
    stride = (stride,) * n_dims if isinstance(stride, Number) else stride
    padding = (padding,) * n_dims if isinstance(padding, Number) else padding
    output_padding = (output_padding,) * n_dims if isinstance(output_padding, Number) else output_padding
    dilation = (dilation,) * n_dims if isinstance(dilation, Number) else dilation

    if transposed:
        assert n_dims == 2, 'Only 2D is supported for fractional strides.'
        w_one = input_nd.new_ones(1, 1, 1, 1)
        pad = [(k - 1) * d - p for (k, d, p) in zip(kernel_size, dilation, padding)]
        input_nd = F.conv_transpose2d(input_nd, w_one, stride=stride)
        input_nd = F.pad(input_nd, (pad[1], pad[1] + output_padding[1], pad[0], pad[0] + output_padding[0]))
        stride = _pair(1)
        padding = _pair(0)

    (bs, nch), in_sz = input_nd.shape[:2], input_nd.shape[2:]
    out_sz = tuple([((i + 2 * p - d * (k - 1) - 1) // s + 1)
                    for (i, k, d, p, s) in zip(in_sz, kernel_size, dilation, padding, stride)])

    if n_dims == 2 and dilation == 1 and has_pyinn and torch.cuda.is_available() and use_pyinn_if_possible:
        output = P.im2col(input_nd, kernel_size, stride, padding)
    else:
        output = F.unfold(input_nd, kernel_size, dilation, padding, stride)
        out_shape = (bs, nch) + tuple(kernel_size) + out_sz
        output = output.view(*out_shape).contiguous()
    return output

def packernel2d(input, mask=None, kernel_size=0, stride=1, padding=0, output_padding=0, dilation=1,
                kernel_type='gaussian', smooth_kernel_type='none', smooth_kernel=None, inv_alpha=None, inv_lambda=None,
                channel_wise=False, normalize_kernel=False, transposed=False, native_impl=False):
    kernel_size = _pair(kernel_size)
    dilation = _pair(dilation)
    padding = _pair(padding)
    output_padding = _pair(output_padding)
    stride = _pair(stride)
    output_mask = False if mask is None else True
    norm = None

    if mask is not None and mask.dtype != input.dtype:
        mask = torch.tensor(mask, dtype=input.dtype, device=input.device)

    if transposed:
        in_sz = tuple(int((o - op - 1 - (k - 1) * d + 2 * p) // s) + 1 for (o, k, s, p, op, d) in
                      zip(input.shape[-2:], kernel_size, stride, padding, output_padding, dilation))
    else:
        in_sz = input.shape[-2:]

    if mask is not None or normalize_kernel:
        mask_pattern = input.new_ones(1, 1, *in_sz)
        mask_pattern = nd2col(mask_pattern, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
                              dilation=dilation, transposed=transposed)
        if mask is not None:
            mask = nd2col(mask, kernel_size, stride=stride, padding=padding, output_padding=output_padding,
                          dilation=dilation, transposed=transposed)
            if not normalize_kernel:
                norm = mask.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True) \
                       / mask_pattern.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        else:
            mask = mask_pattern

    if transposed:
        stride = _pair(1)
        padding = tuple((k - 1) * d // 2 for (k, d) in zip(kernel_size, dilation))

    if native_impl:
        bs, k_ch, in_h, in_w = input.shape

        x = nd2col(input, kernel_size, stride=stride, padding=padding, dilation=dilation)
        x = x.view(bs, k_ch, -1, *x.shape[-2:]).contiguous()

        if smooth_kernel_type == 'none':
            self_idx = kernel_size[0] * kernel_size[1] // 2
            feat_0 = x[:, :, self_idx:self_idx + 1, :, :]
        else:
            smooth_kernel_size = smooth_kernel.shape[2:]
            smooth_padding = (int(padding[0] - (kernel_size[0] - smooth_kernel_size[0]) / 2),
                              int(padding[1] - (kernel_size[1] - smooth_kernel_size[1]) / 2))
            crop = tuple(-1 * np.minimum(0, smooth_padding))
            input_for_kernel_crop = input.view(-1, 1, in_h, in_w)[:, :,
                                    crop[0]:_neg_idx(crop[0]), crop[1]:_neg_idx(crop[1])]
            smoothed = F.conv2d(input_for_kernel_crop, smooth_kernel,
                                stride=stride, padding=tuple(np.maximum(0, smooth_padding)))
            feat_0 = smoothed.view(bs, k_ch, 1, *x.shape[-2:])
        x = x - feat_0
        if kernel_type.find('_asym') >= 0:
            x = F.relu(x, inplace=True)

        x = x * x
        if not channel_wise:
            x = torch.sum(x, dim=1, keepdim=True)
        if kernel_type == 'gaussian':
            x = torch.exp_(x.mul_(-0.5))

        elif kernel_type.startswith('inv_'):
            epsilon = 1e-4
            x = inv_alpha.view(1, -1, 1, 1, 1) \
                + torch.pow(x + epsilon, 0.5 * inv_lambda.view(1, -1, 1, 1, 1))
        else:
            raise ValueError()
        output = x.view(*(x.shape[:2] + tuple(kernel_size) + x.shape[-2:])).contiguous()
    else:
        assert (smooth_kernel_type == 'none' and
                kernel_type == 'gaussian')
        output = GaussKernel2dFn.apply(input, kernel_size, stride, padding, dilation, channel_wise)

    if mask is not None:
        output = output * mask

    if normalize_kernel:
        norm = output.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)

    if norm is not None:
        empty_mask = (norm == 0)
        output = output / (norm + torch.tensor(empty_mask, dtype=input.dtype, device=input.device))
        output_mask = (1 - empty_mask) if output_mask else None
    else:
        output_mask = None

    return output, output_mask

class GaussKernel2dFn(Function):
    @staticmethod
    def forward(ctx, input, kernel_size, stride, padding, dilation, channel_wise):
        ctx.kernel_size = _pair(kernel_size)
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)
        bs, ch, in_h, in_w = input.shape
        out_h = (in_h + 2 * ctx.padding[0] - ctx.dilation[0] * (ctx.kernel_size[0] - 1) - 1) // ctx.stride[0] + 1
        out_w = (in_w + 2 * ctx.padding[1] - ctx.dilation[1] * (ctx.kernel_size[1] - 1) - 1) // ctx.stride[1] + 1
        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
        cols = cols.view(bs, ch, ctx.kernel_size[0], ctx.kernel_size[1], out_h, out_w)
        center_y, center_x = ctx.kernel_size[0] // 2, ctx.kernel_size[1] // 2
        feat_0 = cols.contiguous()[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :]
        diff_sq = (cols - feat_0).pow(2)
        if not channel_wise:
            diff_sq = diff_sq.sum(dim=1, keepdim=True)
        output = torch.exp(-0.5 * diff_sq)
        ctx._backend = type2backend[input.type()]
        ctx.save_for_backward(input, output)

        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        bs, ch, in_h, in_w = input.shape
        out_h, out_w = output.shape[-2:]
        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
        cols = cols.view(bs, ch, ctx.kernel_size[0], ctx.kernel_size[1], out_h, out_w)
        center_y, center_x = ctx.kernel_size[0] // 2, ctx.kernel_size[1] // 2
        feat_0 = cols.contiguous()[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :]
        diff = cols - feat_0
        grad = -0.5 * grad_output * output
        grad_diff = grad.expand_as(cols) * (2 * diff)
        grad_diff[:, :, center_y:center_y + 1, center_x:center_x + 1, :, :] -= \
            grad_diff.sum(dim=2, keepdim=True).sum(dim=3, keepdim=True)
        grad_input = grad_output.new()
        ctx._backend.Im2Col_updateGradInput(ctx._backend.library_state,
                                            grad_diff.view(bs, ch * ctx.kernel_size[0] * ctx.kernel_size[1], -1),
                                            grad_input,
                                            in_h, in_w,
                                            ctx.kernel_size[0], ctx.kernel_size[1],
                                            ctx.dilation[0], ctx.dilation[1],
                                            ctx.padding[0], ctx.padding[1],
                                            ctx.stride[0], ctx.stride[1])

        return grad_input, None, None, None, None, None

class _PacConvNd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, bias,
                 pool_only, kernel_type, smooth_kernel_type,
                 channel_wise, normalize_kernel, shared_filters, filler):
        super(_PacConvNd, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.pool_only = pool_only
        self.kernel_type = kernel_type
        self.smooth_kernel_type = smooth_kernel_type
        self.channel_wise = channel_wise
        self.normalize_kernel = normalize_kernel
        self.shared_filters = shared_filters
        self.filler = filler
        if any([k % 2 != 1 for k in kernel_size]):
            raise ValueError('kernel_size only accept odd numbers')
        if smooth_kernel_type.find('_') >= 0 and int(smooth_kernel_type[smooth_kernel_type.rfind('_') + 1:]) % 2 != 1:
            raise ValueError('smooth_kernel_type only accept kernels of odd widths')
        if shared_filters:
            assert in_channels == out_channels, 'when specifying shared_filters, number of channels should not change'
        if any([p > d * (k - 1) / 2 for (p, d, k) in zip(padding, dilation, kernel_size)]):
            pass
        if not pool_only:
            if self.filler in {'pool', 'crf_pool'}:
                assert shared_filters
                self.register_buffer('weight', torch.ones(1, 1, *kernel_size))
                if self.filler == 'crf_pool':
                    self.weight[(0, 0) + tuple(k // 2 for k in kernel_size)] = 0
            elif shared_filters:
                self.weight = Parameter(torch.Tensor(1, 1, *kernel_size))
            elif transposed:
                self.weight = Parameter(torch.Tensor(in_channels, out_channels, *kernel_size))
            else:
                self.weight = Parameter(torch.Tensor(out_channels, in_channels, *kernel_size))
            if bias:
                self.bias = Parameter(torch.Tensor(out_channels))
            else:
                self.register_parameter('bias', None)
        if kernel_type.startswith('inv_'):
            self.inv_alpha_init = float(kernel_type.split('_')[1])
            self.inv_lambda_init = float(kernel_type.split('_')[2])
            if self.channel_wise and kernel_type.find('_fixed') < 0:
                if out_channels <= 0:
                    raise ValueError('out_channels needed for channel_wise {}'.format(kernel_type))
                inv_alpha = self.inv_alpha_init * torch.ones(out_channels)
                inv_lambda = self.inv_lambda_init * torch.ones(out_channels)
            else:
                inv_alpha = torch.tensor(float(self.inv_alpha_init))
                inv_lambda = torch.tensor(float(self.inv_lambda_init))
            if kernel_type.find('_fixed') < 0:
                self.register_parameter('inv_alpha', Parameter(inv_alpha))
                self.register_parameter('inv_lambda', Parameter(inv_lambda))
            else:
                self.register_buffer('inv_alpha', inv_alpha)
                self.register_buffer('inv_lambda', inv_lambda)
        elif kernel_type != 'gaussian':
            raise ValueError('kernel_type set to invalid value ({})'.format(kernel_type))
        if smooth_kernel_type.startswith('full_'):
            smooth_kernel_size = int(smooth_kernel_type.split('_')[-1])
            self.smooth_kernel = Parameter(torch.Tensor(1, 1, *repeat(smooth_kernel_size, len(kernel_size))))
        elif smooth_kernel_type == 'gaussian':
            smooth_1d = torch.tensor([.25, .5, .25])
            smooth_kernel = smooth_1d
            for d in range(1, len(kernel_size)):
                smooth_kernel = smooth_kernel * smooth_1d.view(-1, *repeat(1, d))
            self.register_buffer('smooth_kernel', smooth_kernel.unsqueeze(0).unsqueeze(0))
        elif smooth_kernel_type.startswith('average_'):
            smooth_kernel_size = int(smooth_kernel_type.split('_')[-1])
            smooth_1d = torch.tensor((1.0 / smooth_kernel_size,) * smooth_kernel_size)
            smooth_kernel = smooth_1d
            for d in range(1, len(kernel_size)):
                smooth_kernel = smooth_kernel * smooth_1d.view(-1, *repeat(1, d))
            self.register_buffer('smooth_kernel', smooth_kernel.unsqueeze(0).unsqueeze(0))
        elif smooth_kernel_type != 'none':
            raise ValueError('smooth_kernel_type set to invalid value ({})'.format(smooth_kernel_type))

        self.reset_parameters()

    def reset_parameters(self):
        if not (self.pool_only or self.filler in {'pool', 'crf_pool'}):
            if self.filler == 'uniform':
                n = self.in_channels
                for k in self.kernel_size:
                    n *= k
                stdv = 1. / math.sqrt(n)
                if self.shared_filters:
                    stdv *= self.in_channels
                self.weight.data.uniform_(-stdv, stdv)
                if self.bias is not None:
                    self.bias.data.uniform_(-stdv, stdv)
            elif self.filler == 'linear':
                effective_kernel_size = tuple(2 * s - 1 for s in self.stride)
                pad = tuple(int((k - ek) // 2) for k, ek in zip(self.kernel_size, effective_kernel_size))
                assert self.transposed and self.in_channels == self.out_channels
                assert all(k >= ek for k, ek in zip(self.kernel_size, effective_kernel_size))
                w = 1.0
                for i, (p, s, k) in enumerate(zip(pad, self.stride, self.kernel_size)):
                    d = len(pad) - i - 1
                    w = w * (np.array((0.0,) * p + tuple(range(1, s)) + tuple(range(s, 0, -1)) + (0,) * p) / s).reshape(
                        (-1,) + (1,) * d)
                    if self.normalize_kernel:
                        w = w * np.array(tuple(((k - j - 1) // s) + (j // s) + 1.0 for j in range(k))).reshape(
                            (-1,) + (1,) * d)
                self.weight.data.fill_(0.0)
                for c in range(1 if self.shared_filters else self.in_channels):
                    self.weight.data[c, c, :] = torch.tensor(w)
                if self.bias is not None:
                    self.bias.data.fill_(0.0)
            elif self.filler in {'crf', 'crf_perturbed'}:
                assert len(self.kernel_size) == 2 and self.kernel_size[0] == self.kernel_size[1] \
                       and self.in_channels == self.out_channels
                perturb_range = 0.001
                n_classes = self.in_channels
                gauss = np_gaussian_2d(self.kernel_size[0]) * self.kernel_size[0] * self.kernel_size[0]
                gauss[self.kernel_size[0] // 2, self.kernel_size[1] // 2] = 0
                if self.shared_filters:
                    self.weight.data[0, 0, :] = torch.tensor(gauss)
                else:
                    compat = 1.0 - np.eye(n_classes, dtype=np.float32)
                    self.weight.data[:] = torch.tensor(compat.reshape(n_classes, n_classes, 1, 1) * gauss)
                if self.filler == 'crf_perturbed':
                    self.weight.data.add_((torch.rand_like(self.weight.data) - 0.5) * perturb_range)
                if self.bias is not None:
                    self.bias.data.fill_(0.0)
            else:
                raise ValueError('Initialization method ({}) not supported.'.format(self.filler))
        if hasattr(self, 'inv_alpha') and isinstance(self.inv_alpha, Parameter):
            self.inv_alpha.data.fill_(self.inv_alpha_init)
            self.inv_lambda.data.fill_(self.inv_lambda_init)
        if hasattr(self, 'smooth_kernel') and isinstance(self.smooth_kernel, Parameter):
            self.smooth_kernel.data.fill_(1.0 / np.multiply.reduce(self.smooth_kernel.shape))

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', kernel_type={kernel_type}')
        if self.stride != (1,) * len(self.stride):
            s += ', stride={stride}'
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.bias is None:
            s += ', bias=False'
        if self.smooth_kernel_type != 'none':
            s += ', smooth_kernel_type={smooth_kernel_type}'
        if self.channel_wise:
            s += ', channel_wise=True'
        if self.normalize_kernel:
            s += ', normalize_kernel=True'
        if self.shared_filters:
            s += ', shared_filters=True'
        return s.format(**self.__dict__)

class PacConv2d(_PacConvNd):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, bias=True,
                 kernel_type='gaussian', smooth_kernel_type='none', normalize_kernel=False, shared_filters=False,
                 filler='uniform', native_impl=False):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(PacConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride,
            padding, dilation, False, _pair(0), bias,
            False, kernel_type, smooth_kernel_type, False, normalize_kernel, shared_filters, filler)

        self.native_impl = native_impl

    def compute_kernel(self, input_for_kernel, input_mask=None):
        return packernel2d(input_for_kernel, input_mask,
                           kernel_size=self.kernel_size, stride=self.stride, padding=self.padding,
                           dilation=self.dilation, kernel_type=self.kernel_type,
                           smooth_kernel_type=self.smooth_kernel_type,
                           smooth_kernel=self.smooth_kernel if hasattr(self, 'smooth_kernel') else None,
                           inv_alpha=self.inv_alpha if hasattr(self, 'inv_alpha') else None,
                           inv_lambda=self.inv_lambda if hasattr(self, 'inv_lambda') else None,
                           channel_wise=False, normalize_kernel=self.normalize_kernel, transposed=False,
                           native_impl=self.native_impl)

    def forward(self, input_2d, input_for_kernel, kernel=None, mask=None):
        output_mask = None
        if kernel is None:
            kernel, output_mask = self.compute_kernel(input_for_kernel, mask)

        output = pacconv2d(input_2d, kernel, self.weight, self.bias, self.stride, self.padding, self.dilation,
                           self.shared_filters, self.native_impl)

        return output if output_mask is None else (output, output_mask)

def pacconv2d(input, kernel, weight, bias=None, stride=1, padding=0, dilation=1, shared_filters=False,
              native_impl=False):
    kernel_size = tuple(weight.shape[-2:])
    stride = _pair(stride)
    padding = _pair(padding)
    dilation = _pair(dilation)

    if native_impl:

        im_cols = nd2col(input, kernel_size, stride=stride, padding=padding, dilation=dilation)

        if shared_filters:
            output = torch.einsum('ijklmn,zykl->ijmn', (im_cols * kernel, weight))
        else:
            output = torch.einsum('ijklmn,ojkl->iomn', (im_cols * kernel, weight))

        if bias is not None:
            output += bias.view(1, -1, 1, 1)
    else:
        output = PacConv2dFn.apply(input, kernel, weight, bias, stride, padding, dilation, shared_filters)

    return output

class PacConv2dFn(Function):
    @staticmethod
    def forward(ctx, input, kernel, weight, bias=None, stride=1, padding=0, dilation=1, shared_filters=False):
        (bs, ch), in_sz = input.shape[:2], input.shape[2:]
        if kernel.size(1) > 1:
            raise ValueError('Non-singleton channel is not allowed for kernel.')
        ctx.input_size = in_sz
        ctx.in_ch = ch
        ctx.kernel_size = tuple(weight.shape[-2:])
        ctx.dilation = _pair(dilation)
        ctx.padding = _pair(padding)
        ctx.stride = _pair(stride)
        ctx.shared_filters = shared_filters
        ctx.save_for_backward(input if (ctx.needs_input_grad[1] or ctx.needs_input_grad[2]) else None,
                              kernel if (ctx.needs_input_grad[0] or ctx.needs_input_grad[2]) else None,
                              weight if (ctx.needs_input_grad[0] or ctx.needs_input_grad[1]) else None)
        ctx._backend = type2backend[input.type()]

        cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)

        in_mul_k = cols.view(bs, ch, *kernel.shape[2:]) * kernel

        if shared_filters:
            output = torch.einsum('ijklmn,zykl->ijmn', (in_mul_k, weight))
        else:
            output = torch.einsum('ijklmn,ojkl->iomn', (in_mul_k, weight))

        if bias is not None:
            output += bias.view(1, -1, 1, 1)

        return output.clone()

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        grad_input = grad_kernel = grad_weight = grad_bias = None
        (bs, out_ch), out_sz = grad_output.shape[:2], grad_output.shape[2:]
        in_ch = ctx.in_ch

        input, kernel, weight = ctx.saved_tensors
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1]:
            if ctx.shared_filters:
                grad_in_mul_k = grad_output.view(bs, out_ch, 1, 1, out_sz[0], out_sz[1]) \
                                * weight.view(ctx.kernel_size[0], ctx.kernel_size[1], 1, 1)
            else:
                grad_in_mul_k = torch.einsum('iomn,ojkl->ijklmn', (grad_output, weight))
        if ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            in_cols = F.unfold(input, ctx.kernel_size, ctx.dilation, ctx.padding, ctx.stride)
            in_cols = in_cols.view(bs, in_ch, ctx.kernel_size[0], ctx.kernel_size[1], out_sz[0], out_sz[1])
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.new()
            grad_im2col_output = grad_in_mul_k * kernel
            grad_im2col_output = grad_im2col_output.view(bs, -1, out_sz[0] * out_sz[1])
            ctx._backend.Im2Col_updateGradInput(ctx._backend.library_state,
                                                grad_im2col_output,
                                                grad_input,
                                                ctx.input_size[0], ctx.input_size[1],
                                                ctx.kernel_size[0], ctx.kernel_size[1],
                                                ctx.dilation[0], ctx.dilation[1],
                                                ctx.padding[0], ctx.padding[1],
                                                ctx.stride[0], ctx.stride[1])
        if ctx.needs_input_grad[1]:
            grad_kernel = in_cols * grad_in_mul_k
            grad_kernel = grad_kernel.sum(dim=1, keepdim=True)
        if ctx.needs_input_grad[2]:
            in_mul_k = in_cols * kernel
            if ctx.shared_filters:
                grad_weight = torch.einsum('ijmn,ijklmn->kl', (grad_output, in_mul_k))
                grad_weight = grad_weight.view(1, 1, ctx.kernel_size[0], ctx.kernel_size[1]).contiguous()
            else:
                grad_weight = torch.einsum('iomn,ijklmn->ojkl', (grad_output, in_mul_k))
        if ctx.needs_input_grad[3]:
            grad_bias = torch.einsum('iomn->o', (grad_output,))

        return grad_input, grad_kernel, grad_weight, grad_bias, None, None, None, None

class ParamNet(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(ParamNet, self).__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels

        self.num_channels = input_channels

        self.vgg16 = models.vgg16(pretrained=True)

        self.conv1_1 = nn.Sequential(
            *[nn.Conv2d(self.input_channels, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64)])
        self.conv1_2 = nn.Sequential(*[nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64)])

        self.conv2_1 = nn.Sequential(*[nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128)])
        self.conv2_2 = nn.Sequential(*[nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128)])

        self.conv3_1 = nn.Sequential(*[nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256)])
        self.conv3_2 = nn.Sequential(*[nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256)])
        self.conv3_3 = nn.Sequential(*[nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256)])

        self.conv4_1 = nn.Sequential(*[nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512)])
        self.conv4_2 = nn.Sequential(*[nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512)])
        self.conv4_3 = nn.Sequential(*[nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512)])

        self.conv5_1 = nn.Sequential(*[nn.Conv2d(1024, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512)])
        self.conv5_2 = nn.Sequential(*[nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512)])
        self.conv5_3 = nn.Sequential(*[nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512)])

        self.pool = nn.MaxPool2d(2, 2)

        self.fc8 = nn.Linear(512, 1)
        self.activation = nn.Sigmoid()

    def forward(self, confidence_map, activations):
        x = F.relu(self.conv1_1(confidence_map))
        x = F.relu(self.conv1_2(x))
        x = self.pool(x)
        x = torch.cat([x, activations[0]], dim=1)

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = self.pool(x)
        x = torch.cat([x, activations[1]], dim=1)

        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.relu(self.conv3_3(x))
        x = self.pool(x)
        x = torch.cat([x, activations[2]], dim=1)

        x = F.relu(self.conv4_1(x))
        x = F.relu(self.conv4_2(x))
        x = F.relu(self.conv4_3(x))
        x = self.pool(x)
        x = torch.cat([x, activations[3]], dim=1)

        x = F.relu(self.conv5_1(x))
        x = F.relu(self.conv5_2(x))
        x = F.relu(self.conv5_3(x))

        x = F.adaptive_avg_pool2d(x, (1, 1))

        x = x.view(confidence_map.shape[0], 512)
        x = self.fc8(x)
        x = self.activation(x)

        return x

class ParamNet2(nn.Module):
    def __init__(self):
        super(ParamNet2, self).__init__()
        self.cost_conv1 = nn.Conv2d(1, 64, stride=2, kernel_size=9, padding=4)
        self.cost_bn1 = nn.BatchNorm2d(64)
        self.cost_conv2 = nn.Conv2d(64, 64, stride=2, kernel_size=7, padding=3)
        self.cost_bn2 = nn.BatchNorm2d(64)
        self.cost_conv3 = nn.Conv2d(64, 64, stride=2, kernel_size=5, padding=2)
        self.cost_bn3 = nn.BatchNorm2d(64)
        self.cost_conv4 = nn.Conv2d(64, 64, stride=2, kernel_size=3, padding=1)
        self.cost_bn4 = nn.BatchNorm2d(64)

        self.fc8 = nn.Linear(64, 1)
        self.activation = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def L2normalize(self, x):
        norm = x ** 2
        norm = norm.sum(dim=1, keepdim=True) + 1e-6
        norm = norm ** (0.5)
        return (x / norm)

    def forward(self, confidence_map):
        x = F.relu(self.cost_bn1(self.cost_conv1(confidence_map)))
        x = F.relu(self.cost_bn2(self.cost_conv2(x)))
        x = F.relu(self.cost_bn3(self.cost_conv3(x)))
        x = F.relu(self.cost_bn4(self.cost_conv4(x)))
        x = F.adaptive_avg_pool2d(x,(1, 1))

        x = x.view(confidence_map.shape[0], 64)
        x = self.fc8(x)

        x = self.activation(x)

        return x

class CCNN(nn.Module):
    def __init__(self):
        super(CCNN, self).__init__()
        self.cost_conv1 = nn.Conv2d(1, 64, kernel_size=9, padding=4)
        self.cost_bn1 = nn.BatchNorm2d(64)
        self.cost_conv2 = nn.Conv2d(64, 64, kernel_size=7, padding=3)
        self.cost_bn2 = nn.BatchNorm2d(64)
        self.cost_conv3 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        self.cost_bn3 = nn.BatchNorm2d(64)
        self.cost_conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.cost_bn4 = nn.BatchNorm2d(64)
        self.cost_pred = nn.Conv2d(64, 1, kernel_size=1, padding=0)

        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def L2normalize(self, x):
        norm = x ** 2
        norm = norm.sum(dim=1, keepdim=True) + 1e-6
        norm = norm ** (0.5)
        return (x / norm)

    def forward(self, disp):
        x = F.relu(self.cost_bn1(self.cost_conv1(disp)))
        x = F.relu(self.cost_bn2(self.cost_conv2(x)))
        x = F.relu(self.cost_bn3(self.cost_conv3(x)))
        x = F.relu(self.cost_bn4(self.cost_conv4(x)))
        out = self.sigmoid(self.cost_pred(x))

        return out

class UncertainNet(nn.Module):
    def __init__(self):
        super(UncertainNet, self).__init__()

        self.uncertain_convtr_42 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        self.uncertain_convtr_41 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        self.uncertain_convtr_40 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        self.uncertain_convtr_32 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        self.uncertain_convtr_31 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=512,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(512)
        ])
        self.uncertain_convtr_30 = nn.Sequential(*[
            nn.Conv2d(in_channels=512,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256)
        ])
        self.uncertain_convtr_22 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256)
        ])
        self.uncertain_convtr_21 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(256)
        ])
        self.uncertain_convtr_20 = nn.Sequential(*[
            nn.Conv2d(in_channels=256,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(128)
        ])
        self.uncertain_convtr_11 = nn.Sequential(*[
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(128)
        ])
        self.uncertain_convtr_10 = nn.Sequential(*[
            nn.Conv2d(in_channels=128,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64)
        ])
        self.uncertain_convtr_01 = nn.Sequential(*[
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(64)
        ])
        self.uncertain_convtr_00 = nn.Sequential(*[
            nn.Conv2d(in_channels=64,
                      out_channels=1,
                      kernel_size=3,
                      padding=1)
        ])

    def forward(self, input_img, indiceslist):

        y_4d = F.max_unpool2d(input_img, indiceslist[4], kernel_size=2, stride=2, output_size=torch.Size([BATCH_SIZE, 512, 12, 30]))
        y_42d = F.relu(self.uncertain_convtr_42(y_4d))
        y_40d = F.relu(self.uncertain_convtr_40(y_42d))

        y_3d = F.max_unpool2d(y_40d, indiceslist[3], kernel_size=2, stride=2, output_size=torch.Size([BATCH_SIZE, 256, 24, 60]))
        y_32d = F.relu(self.uncertain_convtr_32(y_3d))
        y_30d = F.relu(self.uncertain_convtr_30(y_32d))

        y_2d = F.max_unpool2d(y_30d, indiceslist[2], kernel_size=2, stride=2, output_size=torch.Size([BATCH_SIZE, 128, 48, 120]))
        y_22d = F.relu(self.uncertain_convtr_22(y_2d))
        y_20d = F.relu(self.uncertain_convtr_20(y_22d))

        y_1d = F.max_unpool2d(y_20d, indiceslist[1], kernel_size=2, stride=2, output_size=torch.Size([BATCH_SIZE, 64, 96, 240]))
        y_11d = F.relu(self.uncertain_convtr_11(y_1d))
        y_10d = F.relu(self.uncertain_convtr_10(y_11d))

        y_0d = F.max_unpool2d(y_10d, indiceslist[0], kernel_size=2, stride=2, output_size=torch.Size([BATCH_SIZE, 3, 192, 480]))
        y_01d = F.relu(self.uncertain_convtr_01(y_0d))
        y_00d = F.relu(self.uncertain_convtr_00(y_01d))

        uncertainresult = y_00d + 1e-6

        return uncertainresult

