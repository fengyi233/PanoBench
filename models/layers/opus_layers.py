import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch.utils.checkpoint import checkpoint as cp

from utils.data_converter import decode_points
from .multiview_sampling import sampling_4d


class OPUSSampling(BaseModule):
    """Adaptive spatio-temporal sampling shared by OPUS variants."""

    with_prototype_offsets = False

    def __init__(
            self,
            embed_dims=256,
            num_frames=4,
            num_views=6,
            num_groups=4,
            num_points=8,
            num_levels=4,
            pc_range=None,
            init_cfg=None):
        super().__init__(init_cfg)

        self.num_frames = num_frames
        self.num_points = num_points
        self.num_views = int(num_views)
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.pc_range = [] if pc_range is None else pc_range

        if self.with_prototype_offsets:
            self.sampling_prototype = nn.Embedding(num_groups * num_points, 3)
        else:
            self.sampling_prototype = None
        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels)

    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(
            self.num_groups * self.num_points, 3)
        if self.sampling_prototype is not None:
            nn.init.normal_(self.sampling_prototype.weight, mean=0, std=1)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)

    def inner_forward(self, query_points, query_feat, mlvl_feats, occ2img, image_shape):
        B, Q = query_points.shape[:2]
        image_h, image_w = image_shape

        query_points = decode_points(query_points, self.pc_range)
        if query_points.shape[2] == 1:
            query_center = query_points
            query_scale = (
                torch.ones_like(query_center)
                if self.with_prototype_offsets
                else torch.zeros_like(query_center)
            )
        else:
            query_center = query_points.mean(dim=2, keepdim=True)
            query_scale = query_points.std(dim=2, keepdim=True)

        sampling_offset = self.sampling_offset(query_feat).view(B, Q, -1, 3)
        if self.sampling_prototype is not None:
            prototype = self.sampling_prototype.weight[None, None, ...].expand(
                B, Q, -1, -1)
            sampling_points = query_center + prototype * query_scale + sampling_offset
        else:
            sampling_points = query_center + sampling_offset * query_scale

        sampling_points = sampling_points.view(
            B, Q, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.reshape(
            B, Q, 1, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.expand(
            B, Q, self.num_frames, self.num_groups, self.num_points, 3)

        scale_weights = self.scale_weights(query_feat).view(
            B, Q, self.num_groups, 1, self.num_points, self.num_levels)
        scale_weights = torch.softmax(scale_weights, dim=-1)
        scale_weights = scale_weights.expand(
            B, Q, self.num_groups, self.num_frames,
            self.num_points, self.num_levels)

        return sampling_4d(
            sampling_points,
            mlvl_feats,
            scale_weights,
            occ2img,
            image_h,
            image_w,
            num_views=self.num_views,
        )

    def forward(self, query_points, query_feat, mlvl_feats, occ2img, image_shape):
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, query_points, query_feat, mlvl_feats,
                      occ2img, image_shape, use_reentrant=False)
        return self.inner_forward(query_points, query_feat, mlvl_feats,
                                  occ2img, image_shape)


