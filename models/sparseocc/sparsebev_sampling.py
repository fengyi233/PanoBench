import torch

from utils.data_converter import decode_bbox


def rotation_3d_in_axis(points, angles):
    assert points.shape[-1] == 3
    assert angles.shape[-1] == 1
    angles = angles[..., 0]

    n_points = points.shape[-2]
    input_dims = angles.shape

    if len(input_dims) > 1:
        points = points.reshape(-1, n_points, 3)
        angles = angles.reshape(-1)

    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)

    rot_mat_T = torch.stack([
        rot_cos, rot_sin, zeros,
        -rot_sin, rot_cos, zeros,
        zeros, zeros, ones,
    ]).transpose(0, 1).reshape(-1, 3, 3)

    points = torch.bmm(points, rot_mat_T)

    if len(input_dims) > 1:
        points = points.reshape(*input_dims, n_points, 3)

    return points


def make_sample_points_from_bbox(query_bbox, offset, pc_range):
    '''
    query_bbox: [B, Q, 10]
    offset: [B, Q, num_points, 4], normalized by stride
    '''
    query_bbox = decode_bbox(query_bbox, pc_range)  # [B, Q, 9]

    xyz = query_bbox[..., 0:3]  # [B, Q, 3]
    wlh = query_bbox[..., 3:6]  # [B, Q, 3]

    # NOTE: different from SparseBEV
    xyz += wlh / 2  # conver to center

    delta_xyz = offset[..., 0:3]  # [B, Q, P, 3]
    delta_xyz = wlh[:, :, None, :] * delta_xyz  # [B, Q, P, 3]

    if query_bbox.shape[-1] > 6:
        ang = query_bbox[..., 6:7]  # [B, Q, 1]
        delta_xyz = rotation_3d_in_axis(delta_xyz, ang)  # [B, Q, P, 3]

    sample_xyz = xyz[:, :, None, :] + delta_xyz  # [B, Q, P, 3]

    return sample_xyz  # [B, Q, P, 3]


def make_sample_points_from_mask(valid_map, pc_range, occ_size, num_points, occ_loc=None, offset=None):
    '''
    valid_map: [B, Q, W, H, D] or [B, Q, N]
    occ_loc: [B, N, 3] if valid map is sparse
    Return: [B, Q, GP, 3] in pc_range
    '''
    B, Q = valid_map.shape[:2]
    occ_size = torch.tensor(occ_size).to(valid_map.device)

    sampling_pts = []
    for b in range(B):
        indices = torch.where(valid_map[b])
        if indices[0].shape[0] == 0:
            pts = torch.rand((Q, num_points, 3)).to(valid_map.device)
        else:
            if len(valid_map.shape) == 5:
                bin_count = valid_map[b].sum(dim=(1, 2, 3))
            else:
                bin_count = valid_map[b].sum(dim=1)
            sampling_rand = torch.rand((Q, num_points)).to(bin_count.device)
            sampling_index = (sampling_rand * bin_count[:, None]).floor().long()
            low_bound = torch.cumsum(bin_count, dim=0) - bin_count
            sampling_index = sampling_index + low_bound[:, None]
            sampling_index[sampling_index >= indices[0].shape[0]] = indices[0].shape[
                                                                        0] - 1  # this can happen when zeros appear in the tail
            sampling_index = sampling_index.to(valid_map.device)

            if occ_loc is None:  # dense occ points
                pts = torch.stack((indices[1][sampling_index], indices[2][sampling_index], indices[3][sampling_index]))
                pts = pts.permute(1, 2, 0)
            else:
                occ_idx = indices[1][sampling_index]
                pts = occ_loc[b][occ_idx]

            # pad queries with no valid occ
            pts = pts.float()
            rand_sampling_points = torch.rand(((bin_count == 0).sum(), num_points, 3)).to(pts.device) * occ_size
            pts[bin_count == 0] = rand_sampling_points
        sampling_pts.append(pts)

    sampling_pts = torch.stack(sampling_pts)
    if offset is not None:
        sampling_pts = sampling_pts + offset

    sampling_pts = sampling_pts / occ_size
    sampling_pts[..., 0] = sampling_pts[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    sampling_pts[..., 1] = sampling_pts[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    sampling_pts[..., 2] = sampling_pts[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]

    return sampling_pts
