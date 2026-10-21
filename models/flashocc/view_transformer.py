import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS

from models.csrc.wrapper import bev_pool_v2
from utils.labels import get_dataset_meta


class SimpleDepthContextNet(nn.Module):
    def __init__(self, in_channels, mid_channels, depth_channels, context_channels):
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.depth = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, depth_channels, 1),
        )
        self.context = nn.Conv2d(mid_channels, context_channels, 1)

    def forward(self, x):
        x = self.reduce(x)
        return torch.cat([self.depth(x), self.context(x)], dim=1)


@MODELS.register_module()
class Occ2ImgViewTransformer(nn.Module):
    def __init__(self,
                 dataset_type,
                 occ_variant,
                 input_size=(256, 704),
                 depth=(1.0, 45.0, 0.5),
                 downsample=16,
                 in_channels=256,
                 out_channels=80,
                 mid_channels=None,
                 pool_z=1):
        super().__init__()
        meta = get_dataset_meta(dataset_type, occ_variant)
        self.volume_size = tuple(int(v) for v in meta.volume_size)
        self.pc_range = tuple(float(v) for v in meta.point_cloud_range)
        self.pool_z = int(pool_z)
        self.pool_shape = (self.volume_size[0], self.volume_size[1], self.pool_z)
        self.pool_voxel_size = (
            float(meta.voxel_size),
            float(meta.voxel_size),
            (self.pc_range[5] - self.pc_range[2]) / self.pool_z,
        )
        self.depth_cfg = tuple(float(v) for v in depth)
        self.downsample = int(downsample)
        self.out_channels = int(out_channels)
        self.frustum = self.create_frustum(self.depth_cfg, input_size, downsample)
        self.D = self.frustum.shape[0]
        self.depth_net = SimpleDepthContextNet(
            in_channels=in_channels,
            mid_channels=mid_channels or in_channels,
            depth_channels=self.D,
            context_channels=out_channels,
        )

    @staticmethod
    def create_frustum(depth_cfg, input_size, downsample):
        h_in, w_in = input_size
        h_feat, w_feat = h_in // downsample, w_in // downsample
        d = torch.arange(*depth_cfg, dtype=torch.float32).view(-1, 1, 1).expand(-1, h_feat, w_feat)
        x = torch.linspace(0, w_in - 1, w_feat, dtype=torch.float32).view(1, 1, w_feat).expand_as(d)
        y = torch.linspace(0, h_in - 1, h_feat, dtype=torch.float32).view(1, h_feat, 1).expand_as(d)
        return torch.stack((x, y, d), dim=-1)

    def _occ_coordinates(self, occ2img, dtype):
        b, n = occ2img.shape[:2]
        frustum = self.frustum.to(device=occ2img.device, dtype=dtype)
        d, h, w = frustum.shape[:3]
        img_homo = torch.cat([
            frustum[..., :2] * frustum[..., 2:3],
            frustum[..., 2:3],
            torch.ones_like(frustum[..., 2:3]),
        ], dim=-1)
        img_homo = img_homo.reshape(1, 1, d * h * w, 4, 1).expand(b, n, -1, -1, -1)
        occ2img_inv = torch.linalg.inv(occ2img.to(dtype=dtype))
        occ = torch.matmul(occ2img_inv[:, :, None], img_homo).reshape(b, n, d, h, w, 4)
        xyz = occ[..., :3] / occ[..., 3:4].clamp(min=1e-5)
        return xyz

    def _pooling_prepare(self, coor):
        b, n, d, h, w, _ = coor.shape
        num_points = b * n * d * h * w
        device = coor.device
        ranks_depth = torch.arange(num_points, dtype=torch.int32, device=device)
        ranks_feat = torch.arange(b * n * h * w, dtype=torch.int32, device=device).reshape(b, n, 1, h, w)
        ranks_feat = ranks_feat.expand(b, n, d, h, w).flatten()

        pc_min = coor.new_tensor(self.pc_range[:3])
        voxel = coor.new_tensor(self.pool_voxel_size)
        grid_size = coor.new_tensor(self.pool_shape)
        idx = torch.floor((coor - pc_min) / voxel).long().view(num_points, 3)
        batch_idx = torch.arange(b, dtype=torch.long, device=device).view(b, 1).expand(b, num_points // b).reshape(num_points, 1)
        idx = torch.cat([idx, batch_idx], dim=1)

        kept = ((idx[:, 0] >= 0) & (idx[:, 0] < grid_size[0]) &
                (idx[:, 1] >= 0) & (idx[:, 1] < grid_size[1]) &
                (idx[:, 2] >= 0) & (idx[:, 2] < grid_size[2]))
        if not kept.any():
            return None

        idx = idx[kept]
        ranks_depth = ranks_depth[kept]
        ranks_feat = ranks_feat[kept]
        x_size, y_size, z_size = self.pool_shape
        ranks_bev = (idx[:, 3] * (z_size * y_size * x_size) +
                     idx[:, 2] * (y_size * x_size) +
                     idx[:, 1] * x_size + idx[:, 0])
        order = ranks_bev.argsort()
        ranks_bev = ranks_bev[order].int().contiguous()
        ranks_depth = ranks_depth[order].int().contiguous()
        ranks_feat = ranks_feat[order].int().contiguous()
        kept = torch.ones(ranks_bev.shape[0], device=device, dtype=torch.bool)
        kept[1:] = ranks_bev[1:] != ranks_bev[:-1]
        interval_starts = torch.where(kept)[0].int().contiguous()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]
        return ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths

    def forward(self, x, occ2img):
        b, n, c, h, w = x.shape
        x = x.reshape(b * n, c, h, w)
        depth_context = self.depth_net(x)
        depth_logits = depth_context[:, :self.D]
        context = depth_context[:, self.D:self.D + self.out_channels]
        depth = F.softmax(depth_logits, dim=1)

        coor = self._occ_coordinates(occ2img, dtype=depth.dtype)
        prepared = self._pooling_prepare(coor)
        if prepared is None:
            return context.new_zeros((b, self.out_channels * self.pool_z, self.pool_shape[1], self.pool_shape[0])), depth
        ranks_bev, ranks_depth, ranks_feat, interval_starts, interval_lengths = prepared
        feat = context.reshape(b, n, self.out_channels, h, w).permute(0, 1, 3, 4, 2).contiguous()
        bev_shape = (b, self.pool_shape[2], self.pool_shape[1], self.pool_shape[0], self.out_channels)
        bev = bev_pool_v2(depth.reshape(b, n, self.D, h, w), feat, ranks_depth,
                          ranks_feat, ranks_bev, bev_shape, interval_starts, interval_lengths)
        bev = torch.cat(bev.unbind(dim=2), dim=1)
        return bev, depth
