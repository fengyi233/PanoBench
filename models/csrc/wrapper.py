import torch
from torch.autograd.function import once_differentiable


class MSMVSamplingC2345(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat_c2, feat_c3, feat_c4, feat_c5, sampling_locations, scale_weights):
        from ._msmv_sampling_cuda import _ms_deform_attn_cuda_c2345_forward

        ctx.save_for_backward(feat_c2, feat_c3, feat_c4, feat_c5, sampling_locations, scale_weights)
        return _ms_deform_attn_cuda_c2345_forward(
            feat_c2, feat_c3, feat_c4, feat_c5,
            sampling_locations, scale_weights)

    @staticmethod
    def backward(ctx, grad_output):
        from ._msmv_sampling_cuda import _ms_deform_attn_cuda_c2345_backward

        feat_c2, feat_c3, feat_c4, feat_c5, sampling_locations, scale_weights = ctx.saved_tensors
        grad_value_c2, grad_value_c3, grad_value_c4, grad_value_c5, grad_sampling_loc, grad_attn_weight = _ms_deform_attn_cuda_c2345_backward(
            grad_output.contiguous(), feat_c2, feat_c3, feat_c4, feat_c5,
            sampling_locations, scale_weights
        )
        return grad_value_c2, grad_value_c3, grad_value_c4, grad_value_c5, grad_sampling_loc, grad_attn_weight


class MSMVSamplingC23456(torch.autograd.Function):
    @staticmethod
    def forward(ctx, feat_c2, feat_c3, feat_c4, feat_c5, feat_c6, sampling_locations, scale_weights):
        from ._msmv_sampling_cuda import _ms_deform_attn_cuda_c23456_forward

        ctx.save_for_backward(feat_c2, feat_c3, feat_c4, feat_c5, feat_c6, sampling_locations, scale_weights)
        return _ms_deform_attn_cuda_c23456_forward(
            feat_c2, feat_c3, feat_c4, feat_c5, feat_c6,
            sampling_locations, scale_weights)

    @staticmethod
    def backward(ctx, grad_output):
        from ._msmv_sampling_cuda import _ms_deform_attn_cuda_c23456_backward

        feat_c2, feat_c3, feat_c4, feat_c5, feat_c6, sampling_locations, scale_weights = ctx.saved_tensors
        grad_value_c2, grad_value_c3, grad_value_c4, grad_value_c5, grad_value_c6, grad_sampling_loc, grad_attn_weight = _ms_deform_attn_cuda_c23456_backward(
            grad_output.contiguous(), feat_c2, feat_c3, feat_c4, feat_c5, feat_c6,
            sampling_locations, scale_weights
        )
        return grad_value_c2, grad_value_c3, grad_value_c4, grad_value_c5, grad_value_c6, grad_sampling_loc, grad_attn_weight


def msmv_sampling(mlvl_feats, sampling_locations, scale_weights):
    if len(mlvl_feats) == 4:
        return MSMVSamplingC2345.apply(*mlvl_feats, sampling_locations, scale_weights)
    if len(mlvl_feats) == 5:
        return MSMVSamplingC23456.apply(*mlvl_feats, sampling_locations, scale_weights)
    raise ValueError(f'Unsupported number of feature levels: {len(mlvl_feats)}')


class BEVPoolV2(torch.autograd.Function):
    @staticmethod
    def forward(ctx, depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
        ranks_bev = ranks_bev.int()
        depth = depth.contiguous().float()
        feat = feat.contiguous().float()
        ranks_depth = ranks_depth.contiguous().int()
        ranks_feat = ranks_feat.contiguous().int()
        interval_lengths = interval_lengths.contiguous().int()
        interval_starts = interval_starts.contiguous().int()

        from ._bev_pool_v2_cuda import bev_pool_v2_forward

        out = feat.new_zeros(bev_feat_shape)
        bev_pool_v2_forward(depth, feat, out, ranks_depth, ranks_feat,
                            ranks_bev, interval_lengths, interval_starts)
        ctx.save_for_backward(ranks_bev, depth, feat, ranks_feat, ranks_depth)
        return out

    @staticmethod
    def backward(ctx, out_grad):
        ranks_bev, depth, feat, ranks_feat, ranks_depth = ctx.saved_tensors
        order = ranks_feat.argsort()
        ranks_feat = ranks_feat[order]
        ranks_depth = ranks_depth[order]
        ranks_bev = ranks_bev[order]
        kept = torch.ones(ranks_bev.shape[0], device=ranks_bev.device, dtype=torch.bool)
        kept[1:] = ranks_feat[1:] != ranks_feat[:-1]
        interval_starts = torch.where(kept)[0].int()
        interval_lengths = torch.zeros_like(interval_starts)
        interval_lengths[:-1] = interval_starts[1:] - interval_starts[:-1]
        interval_lengths[-1] = ranks_bev.shape[0] - interval_starts[-1]

        depth_grad = depth.new_zeros(depth.shape)
        feat_grad = feat.new_zeros(feat.shape)
        from ._bev_pool_v2_cuda import bev_pool_v2_backward

        bev_pool_v2_backward(out_grad.contiguous(), depth_grad, feat_grad,
                             depth.contiguous(), feat.contiguous(),
                             ranks_depth.contiguous(), ranks_feat.contiguous(),
                             ranks_bev.contiguous(), interval_lengths.contiguous(),
                             interval_starts.contiguous())
        return depth_grad, feat_grad, None, None, None, None, None, None


def bev_pool_v2(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                bev_feat_shape, interval_starts, interval_lengths):
    x = BEVPoolV2.apply(depth, feat, ranks_depth, ranks_feat, ranks_bev,
                        bev_feat_shape, interval_starts, interval_lengths)
    return x.permute(0, 4, 1, 2, 3).contiguous()


class LocalAggregate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, pts, points_int, means3D, means3D_int, opacities, semantics,
                radii, cov3D, H, W, D):
        from ._local_aggregate_cuda import local_aggregate_forward

        num_rendered, logits, geom_buffer, binning_buffer, img_buffer = local_aggregate_forward(
            pts, points_int, means3D, means3D_int, opacities, semantics, radii, cov3D, H, W, D)
        ctx.num_rendered = num_rendered
        ctx.grid_shape = (H, W, D)
        ctx.save_for_backward(geom_buffer, binning_buffer, img_buffer, means3D, pts,
                              points_int, cov3D, opacities, semantics)
        return logits

    @staticmethod
    def backward(ctx, out_grad):
        from ._local_aggregate_cuda import local_aggregate_backward

        (geom_buffer, binning_buffer, img_buffer, means3D, pts, points_int, cov3D,
         opacities, semantics) = ctx.saved_tensors
        h, w, d = ctx.grid_shape
        means3D_grad, opacity_grad, semantics_grad, cov3D_grad = local_aggregate_backward(
            geom_buffer, binning_buffer, img_buffer, h, w, d, ctx.num_rendered,
            means3D, pts, points_int, cov3D, opacities, semantics, out_grad)
        return (None, None, means3D_grad, None, opacity_grad, semantics_grad,
                None, cov3D_grad, None, None, None)


def local_aggregate(pts, points_int, means3D, means3D_int, opacities, semantics,
                    radii, cov3D, grid_shape):
    return LocalAggregate.apply(pts, points_int, means3D, means3D_int, opacities,
                                semantics, radii, cov3D, *grid_shape)


class DeformableAggregation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights):
        from ._deformable_aggregation_cuda import deformable_aggregation_forward

        mc_ms_feat = mc_ms_feat.contiguous().float()
        spatial_shape = spatial_shape.contiguous().int()
        scale_start_index = scale_start_index.contiguous().int()
        sampling_location = sampling_location.contiguous().float()
        weights = weights.contiguous().float()
        output = deformable_aggregation_forward(
            mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights)
        ctx.save_for_backward(mc_ms_feat, spatial_shape, scale_start_index,
                              sampling_location, weights)
        return output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        from ._deformable_aggregation_cuda import deformable_aggregation_backward

        mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights = ctx.saved_tensors
        grad_mc_ms_feat = torch.zeros_like(mc_ms_feat)
        grad_sampling_location = torch.zeros_like(sampling_location)
        grad_weights = torch.zeros_like(weights)
        deformable_aggregation_backward(
            mc_ms_feat, spatial_shape, scale_start_index, sampling_location, weights,
            grad_output.contiguous(), grad_mc_ms_feat, grad_sampling_location, grad_weights)
        return grad_mc_ms_feat, None, None, grad_sampling_location, grad_weights


def flatten_multi_scale_feats(feature_maps):
    """Pack per-level ``[B, N, C, H, W]`` maps into the layout expected by
    :func:`deformable_aggregation`."""
    bs, num_cams = feature_maps[0].shape[:2]
    spatial_shape = []
    scale_start_index = [0]
    col_feats = []
    for feat in feature_maps:
        spatial_shape.append(feat.shape[-2:])
        scale_start_index.append(feat.shape[-1] * feat.shape[-2] + scale_start_index[-1])
        col_feats.append(feat.reshape(bs, num_cams, feat.shape[2], -1))
    scale_start_index.pop()
    col_feats = torch.cat(col_feats, dim=-1).permute(0, 1, 3, 2)
    return (
        col_feats,
        torch.tensor(spatial_shape, dtype=torch.int64, device=col_feats.device),
        torch.tensor(scale_start_index, dtype=torch.int64, device=col_feats.device),
    )


def deformable_aggregation(feature_maps, sampling_location, weights):
    return DeformableAggregation.apply(*feature_maps, sampling_location, weights)
