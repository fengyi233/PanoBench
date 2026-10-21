import torch
import torch.nn as nn
from mmengine.model import BaseModule

from models.layers import sampling_4d
from .sparsebev_sampling import make_sample_points_from_bbox


class SparseBEVSampling(BaseModule):
    def __init__(
            self,
            embed_dims=256,
            num_frames=4,
            num_views=6,
            num_groups=4,
            num_points=8,
            num_levels=4,
            pc_range=[],
            init_cfg=None):
        super().__init__(init_cfg)

        self.num_frames = num_frames
        self.num_views = int(num_views)
        self.num_points = num_points
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.pc_range = pc_range

        self.sampling_offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels)

    def init_weights(self):
        bias = self.sampling_offset.bias.data.view(self.num_groups * self.num_points, 3)
        nn.init.zeros_(self.sampling_offset.weight)
        nn.init.uniform_(bias[:, 0:3], -0.5, 0.5)

    def forward(self, query_bbox, query_feat, mlvl_feats, img_metas):
        '''
        query_bbox: [B, Q, 10]
        query_feat: [B, Q, C]
        '''
        B, Q = query_bbox.shape[:2]
        image_h, image_w, _ = img_metas[0]['img_shape']

        # sampling offset of all frames
        sampling_offset = self.sampling_offset(query_feat)
        sampling_offset = sampling_offset.view(B, Q, self.num_groups * self.num_points, 3)
        sampling_points = make_sample_points_from_bbox(query_bbox, sampling_offset, self.pc_range)  # [B, Q, GP, 3]
        sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.expand(B, Q, self.num_frames, self.num_groups, self.num_points, 3)

        # warp sample points based on velocity
        if query_bbox.shape[-1] > 8:
            time_diff = img_metas[0]['time_diff']  # [B, F]
            time_diff = time_diff[:, None, :, None]  # [B, 1, F, 1]
            vel = query_bbox[..., 8:].detach()  # [B, Q, 2]
            vel = vel[:, :, None, :]  # [B, Q, 1, 2]
            dist = vel * time_diff  # [B, Q, F, 2]
            dist = dist[:, :, :, None, None, :]  # [B, Q, F, 1, 1, 2]
            sampling_points = torch.cat([
                sampling_points[..., 0:2] - dist,
                sampling_points[..., 2:3]
            ], dim=-1)

        # scale weights
        scale_weights = self.scale_weights(query_feat).view(B, Q, self.num_groups, 1, self.num_points, self.num_levels)
        scale_weights = torch.softmax(scale_weights, dim=-1)
        scale_weights = scale_weights.expand(B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels)

        # sampling
        sampled_feats = sampling_4d(
            sampling_points,
            mlvl_feats,
            scale_weights,
            img_metas[0]['occ2img'],
            image_h, image_w,
            num_views=self.num_views,
        )  # [B, Q, G, FP, C]

        return sampled_feats
