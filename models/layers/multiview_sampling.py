import torch

from models.csrc.wrapper import msmv_sampling


def sampling_4d(
        sample_points,
        mlvl_feats,
        scale_weights,
        occ2img,
        image_h,
        image_w,
        num_views=None,
        fut2cur=None,
        num_horizons=1,
        eps=1e-5):
    """Project 3D points and sample multi-level multi-view image features.

    Args:
        sample_points: [B, Q, T, G, P, 3] points in occupancy coordinates.
        mlvl_feats: feature pyramid reshaped to [B*T*G, N, H, W, C].
        scale_weights: [B, Q, G, T, P, L] level weights.
        occ2img: [B, T*N, 4, 4] projection matrices.
        fut2cur: [B, 4, 4] mapping each sample's occupancy frame into the frame
            ``occ2img`` is referenced to. Used by forecasting models whose query
            points live in a future ego frame; ``None`` for single-frame models.
        num_horizons: Number of forecast horizons folded into ``B``. The image
            pyramid is only replicated ``B / num_horizons`` times, so sampling
            is issued in ``num_horizons`` horizon-major chunks.
    """
    B, Q, T, G, P, _ = sample_points.shape
    N = int(num_views) if num_views is not None else occ2img.shape[1] // T
    if occ2img.shape[1] < T * N:
        raise ValueError(
            f'occ2img has {occ2img.shape[1]} views, but T={T}, N={N}.')
    if B % num_horizons != 0:
        raise ValueError(
            f'batch {B} is not divisible by num_horizons {num_horizons}.')

    sample_points = sample_points.reshape(B, Q, T, G * P, 3)

    occ2img = occ2img[:, :(T * N), None, None, :, :]
    occ2img = occ2img.expand(B, T * N, Q, G * P, 4, 4)
    occ2img = occ2img.reshape(B, T, N, Q, G * P, 4, 4)

    ones = torch.ones_like(sample_points[..., :1])
    sample_points = torch.cat([sample_points, ones], dim=-1)
    sample_points = sample_points[:, :, None, ..., None]
    sample_points = sample_points.expand(B, Q, N, T, G * P, 4, 1)
    sample_points = sample_points.transpose(1, 3)

    if fut2cur is not None:
        # [B, T, N, Q, GP, 4, 1] left-multiplied by the per-sample [B, 4, 4].
        sample_points = torch.einsum(
            'bij,btnqpjk->btnqpik', fut2cur, sample_points)

    sample_points_cam = torch.matmul(occ2img, sample_points).squeeze(-1)
    homo = sample_points_cam[..., 2:3]
    homo_nonzero = torch.maximum(homo, torch.zeros_like(homo) + eps)
    sample_points_cam = sample_points_cam[..., 0:2] / homo_nonzero

    sample_points_cam[..., 0] /= image_w
    sample_points_cam[..., 1] /= image_h

    valid_mask = (
        (homo > eps)
        & (sample_points_cam[..., 1:2] > 0.0)
        & (sample_points_cam[..., 1:2] < 1.0)
        & (sample_points_cam[..., 0:1] > 0.0)
        & (sample_points_cam[..., 0:1] < 1.0)
    ).squeeze(-1).float()

    valid_mask = valid_mask.permute(0, 1, 3, 4, 2)
    sample_points_cam = sample_points_cam.permute(0, 1, 3, 4, 2, 5)

    i_batch = torch.arange(B, dtype=torch.long, device=sample_points.device)
    i_query = torch.arange(Q, dtype=torch.long, device=sample_points.device)
    i_time = torch.arange(T, dtype=torch.long, device=sample_points.device)
    i_point = torch.arange(G * P, dtype=torch.long, device=sample_points.device)
    i_batch = i_batch.view(B, 1, 1, 1, 1).expand(B, T, Q, G * P, 1)
    i_time = i_time.view(1, T, 1, 1, 1).expand(B, T, Q, G * P, 1)
    i_query = i_query.view(1, 1, Q, 1, 1).expand(B, T, Q, G * P, 1)
    i_point = i_point.view(1, 1, 1, G * P, 1).expand(B, T, Q, G * P, 1)

    i_view = torch.argmax(valid_mask, dim=-1)[..., None]
    sample_points_cam = sample_points_cam[
        i_batch, i_time, i_query, i_point, i_view, :]
    valid_mask = valid_mask[i_batch, i_time, i_query, i_point, i_view]

    if N > 1:
        view_coord = i_view[..., None].float() / (N - 1)
    else:
        view_coord = torch.zeros_like(
            i_view[..., None], dtype=sample_points_cam.dtype)
    sample_points_cam = torch.cat([sample_points_cam, view_coord], dim=-1)

    sample_points_cam = sample_points_cam.reshape(B, T, Q, G, P, 1, 3)
    sample_points_cam = sample_points_cam.permute(0, 1, 3, 2, 4, 5, 6)
    sample_points_cam = sample_points_cam.reshape(B * T * G, Q, P, 3).contiguous()

    scale_weights = scale_weights.reshape(B, Q, G, T, P, -1)
    scale_weights = scale_weights.permute(0, 2, 3, 1, 4, 5)
    scale_weights = scale_weights.reshape(B * G * T, Q, P, -1).contiguous()

    if num_horizons == 1:
        final = msmv_sampling(mlvl_feats, sample_points_cam, scale_weights)
    else:
        # ``mlvl_feats`` holds one pyramid per real sample, while the points
        # carry an extra horizon-major axis. Reuse the pyramid per horizon.
        chunk = (B // num_horizons) * T * G
        final = torch.cat([
            msmv_sampling(mlvl_feats, points, weights)
            for points, weights in zip(
                torch.split(sample_points_cam, chunk, dim=0),
                torch.split(scale_weights, chunk, dim=0))
        ], dim=0)

    C = final.shape[2]
    final = final.reshape(B, T, G, Q, C, P)
    final = final.permute(0, 3, 2, 1, 5, 4)
    return final.flatten(3, 4)
