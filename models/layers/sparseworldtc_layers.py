import torch
from torch.utils.checkpoint import checkpoint as cp

from utils.data_converter import decode_points
from .multiview_sampling import sampling_4d
from .opus_layers import OPUSSampling


class SparseWorldTCSampling(OPUSSampling):
    """Trajectory-conditioned variant of :class:`OPUSSampling`.

    Query points live in a *future* ego frame. ``fut2cur`` brings them back to
    the key-frame occupancy frame that ``occ2img`` is referenced to, so the
    history image pyramid can be sampled without ever loading future images.
    """

    def inner_forward(self, query_points, query_feat, mlvl_feats, occ2img,
                      image_shape, fut2cur, num_horizons):
        B, Q = query_points.shape[:2]
        image_h, image_w = image_shape

        query_points = decode_points(query_points, self.pc_range)
        if query_points.shape[2] == 1:
            query_center = query_points
            query_scale = torch.zeros_like(query_center)
        else:
            query_center = query_points.mean(dim=2, keepdim=True)
            query_scale = query_points.std(dim=2, keepdim=True)

        sampling_offset = self.sampling_offset(query_feat).view(B, Q, -1, 3)
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
            fut2cur=fut2cur,
            num_horizons=num_horizons,
        )

    def forward(self, query_points, query_feat, mlvl_feats, occ2img,
                image_shape, fut2cur, num_horizons):
        args = (query_points, query_feat, mlvl_feats, occ2img, image_shape,
                fut2cur, num_horizons)
        if self.training and query_feat.requires_grad:
            return cp(self.inner_forward, *args, use_reentrant=False)
        return self.inner_forward(*args)
