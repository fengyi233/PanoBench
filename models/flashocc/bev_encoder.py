import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer
from mmdet.models.backbones.resnet import BasicBlock, Bottleneck
from mmdet3d.registry import MODELS


@MODELS.register_module()
class FlashCustomFPN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 out_ids=(0,),
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest')):
        super().__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = int(num_outs)
        self.start_level = int(start_level)
        self.backbone_end_level = self.num_ins if end_level == -1 else int(end_level)
        self.out_ids = tuple(out_ids)
        self.upsample_cfg = upsample_cfg.copy()

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            self.lateral_convs.append(ConvModule(
                in_channels[i], out_channels, 1, conv_cfg=conv_cfg,
                norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False))
            if i in self.out_ids:
                self.fpn_convs.append(ConvModule(
                    out_channels, out_channels, 3, padding=1, conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg, act_cfg=act_cfg, inplace=False))

    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        for i in range(len(laterals) - 1, 0, -1):
            if 'scale_factor' in self.upsample_cfg:
                laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], **self.upsample_cfg)
            else:
                laterals[i - 1] = laterals[i - 1] + F.interpolate(
                    laterals[i], size=laterals[i - 1].shape[2:], **self.upsample_cfg)
        return [self.fpn_convs[i](laterals[out_id]) for i, out_id in enumerate(self.out_ids[:self.num_outs])]


@MODELS.register_module()
class FlashCustomResNet(nn.Module):
    def __init__(self,
                 numC_input,
                 num_layer=(2, 2, 2),
                 num_channels=None,
                 stride=(2, 2, 2),
                 backbone_output_ids=None,
                 norm_cfg=dict(type='BN'),
                 with_cp=False,
                 block_type='Basic'):
        super().__init__()
        assert len(num_layer) == len(stride)
        num_channels = [numC_input * 2 ** (i + 1) for i in range(len(num_layer))] \
            if num_channels is None else num_channels
        self.backbone_output_ids = range(len(num_layer)) if backbone_output_ids is None else backbone_output_ids
        self.with_cp = with_cp

        layers = []
        curr_numC = numC_input
        for i, n_blocks in enumerate(num_layer):
            if block_type == 'BottleNeck':
                block = Bottleneck
                first = block(
                    inplanes=curr_numC,
                    planes=num_channels[i] // 4,
                    stride=stride[i],
                    downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1),
                    norm_cfg=norm_cfg)
                rest = [block(
                    inplanes=num_channels[i],
                    planes=num_channels[i] // 4,
                    stride=1,
                    downsample=None,
                    norm_cfg=norm_cfg) for _ in range(n_blocks - 1)]
            else:
                block = BasicBlock
                first = block(
                    inplanes=curr_numC,
                    planes=num_channels[i],
                    stride=stride[i],
                    downsample=nn.Conv2d(curr_numC, num_channels[i], 3, stride[i], 1),
                    norm_cfg=norm_cfg)
                rest = [block(
                    inplanes=num_channels[i],
                    planes=num_channels[i],
                    stride=1,
                    downsample=None,
                    norm_cfg=norm_cfg) for _ in range(n_blocks - 1)]
            curr_numC = num_channels[i]
            layers.append(nn.Sequential(first, *rest))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        feats = []
        for lid, layer in enumerate(self.layers):
            x = layer(x)
            if lid in self.backbone_output_ids:
                feats.append(x)
        return feats


@MODELS.register_module()
class FlashFPNLSS(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 scale_factor=4,
                 input_feature_index=(0, 2),
                 norm_cfg=dict(type='BN'),
                 extra_upsample=2,
                 lateral=None):
        super().__init__()
        self.input_feature_index = input_feature_index
        self.extra_upsample = extra_upsample is not None
        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True)

        channels_factor = 2 if self.extra_upsample else 1
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * channels_factor, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels * channels_factor, out_channels * channels_factor, 3, padding=1, bias=False),
            build_norm_layer(norm_cfg, out_channels * channels_factor)[1],
            nn.ReLU(inplace=True),
        )
        if self.extra_upsample:
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=extra_upsample, mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels * channels_factor, out_channels, 3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 1),
            )
        self.lateral = lateral is not None
        if self.lateral:
            self.lateral_conv = nn.Sequential(
                nn.Conv2d(lateral, lateral, 1, bias=False),
                build_norm_layer(norm_cfg, lateral)[1],
                nn.ReLU(inplace=True),
            )

    def forward(self, feats):
        x2 = feats[self.input_feature_index[0]]
        x1 = feats[self.input_feature_index[1]]
        if self.lateral:
            x2 = self.lateral_conv(x2)
        x1 = self.up(x1)
        x = self.conv(torch.cat([x2, x1], dim=1))
        if self.extra_upsample:
            x = self.up2(x)
        return x
