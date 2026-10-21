import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import MultiheadAttention
from mmengine.model import BaseModule

from utils.data_converter import decode_points, decode_bbox


class SelfAttention(BaseModule):
    """Scale-adaptive self-attention."""

    def __init__(
            self,
            embed_dims=256,
            num_heads=8,
            dropout=0.1,
            pc_range=None,
            init_cfg=None):
        super().__init__(init_cfg)
        self.pc_range = [] if pc_range is None else pc_range

        self.attention = MultiheadAttention(embed_dims, num_heads, dropout, batch_first=True)
        self.gen_tau = nn.Linear(embed_dims, num_heads)

    @torch.no_grad()
    def init_weights(self):
        nn.init.zeros_(self.gen_tau.weight)
        nn.init.uniform_(self.gen_tau.bias, 0.0, 2.0)

    def forward(self, queries, query_feat, query_mode='bbox'):
        if query_mode == 'bbox':
            dist = self.calc_bbox_dists(queries)
        elif query_mode == 'points':
            dist = self.calc_points_dists(queries)
        else:
            raise ValueError("")
        tau = self.gen_tau(query_feat).permute(0, 2, 1)
        attn_mask = dist[:, None, :, :] * tau[..., None]
        attn_mask = attn_mask.flatten(0, 1)
        return self.attention(query_feat, attn_mask=attn_mask)

    @torch.no_grad()
    def calc_points_dists(self, points):
        points = decode_points(points, self.pc_range)
        points = points.mean(dim=2)
        dist = torch.norm(points.unsqueeze(-2) - points.unsqueeze(-3), dim=-1)
        return -dist

    @torch.no_grad()
    def calc_bbox_dists(self, bboxes):
        centers = decode_bbox(bboxes, self.pc_range)[..., :2]  # [B, Q, 2]

        dist = []
        for b in range(centers.shape[0]):
            dist_b = torch.norm(centers[b].reshape(-1, 1, 2) - centers[b].reshape(1, -1, 2), dim=-1)
            dist.append(dist_b[None, ...])

        dist = torch.cat(dist, dim=0)  # [B, Q, Q]
        dist = -dist

        return dist
