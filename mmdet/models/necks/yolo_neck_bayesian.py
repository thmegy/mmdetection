# Copyright (c) OpenMMLab. All rights reserved.
# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, build_norm_layer, build_activation_layer
from mmcv.runner import BaseModule

from ..builder import NECKS


class DetectionBlock(BaseModule):
    """Detection block in YOLO neck.

    Let out_channels = n, the DetectionBlock contains:
    Six ConvLayers, 1 Conv2D Layer and 1 YoloLayer.
    The first 6 ConvLayers are formed the following way:
        1x1xn, 3x3x2n, 1x1xn, 3x3x2n, 1x1xn, 3x3x2n.
    The Conv2D layer is 1x1x255.
    Some block will have branch after the fifth ConvLayer.
    The input channel is arbitrary (in_channels)

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg=None):
        super(DetectionBlock, self).__init__(init_cfg)
        double_out_channels = out_channels * 2

        # add dropout between convolution and normalisation layers
        self.conv1 = build_conv_layer(conv_cfg, in_channels, out_channels, 1)
        self.conv2 = build_conv_layer(conv_cfg, out_channels, double_out_channels, 3, padding=1)
        self.conv3 = build_conv_layer(conv_cfg, double_out_channels, out_channels, 1)
        self.conv4 = build_conv_layer(conv_cfg, out_channels, double_out_channels, 3, padding=1)
        self.conv5 = build_conv_layer(conv_cfg, double_out_channels, out_channels, 1)

        # dropout layer
        self.dropout = torch.nn.Dropout(p=0.1)

        # normalisation layers
        self.norm = build_norm_layer(norm_cfg, out_channels)[1] # build_norm_layer returns (name, layer)
        self.norm_double = build_norm_layer(norm_cfg, double_out_channels)[1]

        # activation layers
        self.act = build_activation_layer(act_cfg)

    def forward(self, x):
        tmp = self.conv1(x)
        tmp = self.dropout(tmp)
        tmp = self.norm(tmp)
        tmp = self.act(tmp)
        
        tmp = self.conv2(tmp)
        tmp = self.dropout(tmp)
        tmp = self.norm_double(tmp)
        tmp = self.act(tmp)
        
        tmp = self.conv3(tmp)
        tmp = self.dropout(tmp)
        tmp = self.norm(tmp)
        tmp = self.act(tmp)
        
        tmp = self.conv4(tmp)
        tmp = self.dropout(tmp)
        tmp = self.norm_double(tmp)
        tmp = self.act(tmp)
        
        tmp = self.conv5(tmp)
        tmp = self.dropout(tmp)
        tmp = self.norm(tmp)
        out = self.act(tmp)
        
        return out


@NECKS.register_module()
class YOLOV3BayesianNeck(BaseModule):
    """The neck of YOLOV3, made bayesian with addition od dropout layers after convolutional layers.

    It can be treated as a simplified version of FPN. It
    will take the result from Darknet backbone and do some upsampling and
    concatenation. It will finally output the detection result.

    Note:
        The input feats should be from top to bottom.
            i.e., from high-lvl to low-lvl
        But YOLOV3Neck will process them in reversed order.
            i.e., from bottom (high-lvl) to top (low-lvl)

    Args:
        num_scales (int): The number of scales / stages.
        in_channels (List[int]): The number of input channels per scale.
        out_channels (List[int]): The number of output channels  per scale.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Default: None.
        norm_cfg (dict, optional): Dictionary to construct and config norm
            layer. Default: dict(type='BN', requires_grad=True)
        act_cfg (dict, optional): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """

    def __init__(self,
                 num_scales,
                 in_channels,
                 out_channels,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 init_cfg=None):
        super(YOLOV3BayesianNeck, self).__init__(init_cfg)
        assert (num_scales == len(in_channels) == len(out_channels))
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.out_channels = out_channels

        # shortcut
        cfg = dict(conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)

        # To support arbitrary scales, the code looks awful, but it works.
        # Better solution is welcomed.
        self.detect1 = DetectionBlock(in_channels[0], out_channels[0], **cfg)
        for i in range(1, self.num_scales):
            in_c, out_c = self.in_channels[i], self.out_channels[i]
            inter_c = out_channels[i - 1]
            self.add_module(f'conv{i}', ConvModule(inter_c, out_c, 1, **cfg))
            # in_c + out_c : High-lvl feats will be cat with low-lvl feats
            self.add_module(f'detect{i+1}',
                            DetectionBlock(in_c + out_c, out_c, **cfg))

    def forward(self, feats):
        assert len(feats) == self.num_scales

        # processed from bottom (high-lvl) to top (low-lvl)
        outs = []
        out = self.detect1(feats[-1])
        outs.append(out)

        for i, x in enumerate(reversed(feats[:-1])):
            conv = getattr(self, f'conv{i+1}')
            tmp = conv(out)

            # Cat with low-lvl feats
            tmp = F.interpolate(tmp, scale_factor=2)
            tmp = torch.cat((tmp, x), 1)

            detect = getattr(self, f'detect{i+2}')
            out = detect(tmp)
            outs.append(out)

        return tuple(outs)
