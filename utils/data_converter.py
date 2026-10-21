from typing import Tuple, Union

import numpy as np
import torch


def decode_global_id(global_id: Union[np.uint16, np.ndarray]):
    """
    Decode global ID into semantic ID and instance ID. Supports scalar or ndarray.

    The global ID is: global_id = 1000 * semantic_id + instance_id.
    semantic_id in [0, 64], instance_id in [0, 999] for uint16.

    Args:
        global_id: uint16 or ndarray of uint16

    Returns:
        tuple: (semantic_id, instance_id) with same shape
    """
    global_id = np.asarray(global_id, dtype=np.uint32)  # avoid overflow in division

    semantic_id = (global_id // 1000).astype(np.uint16)
    instance_id = (global_id % 1000).astype(np.uint16)
    return semantic_id, instance_id


def encode_global_id(semantic_id: Union[np.uint16, np.ndarray],
                     instance_id: Union[np.uint16, np.ndarray]):
    """
    Encode semantic ID and instance ID into a global ID. Supports scalar or ndarray.

    global_id = 1000 * semantic_id + instance_id.
    For uint16 (max 65535): semantic_id in [0, 64], instance_id in [0, 999].

    Args:
        semantic_id: uint16 or ndarray of uint16 (0-64)
        instance_id:  uint16 or ndarray of uint16 (0-999)

    Returns:
        global_id: uint16 or ndarray of uint16
    """
    semantic_id = np.asarray(semantic_id, dtype=np.uint32)
    instance_id = np.asarray(instance_id, dtype=np.uint32)
    if np.any(semantic_id > 64):
        raise ValueError("Semantic ID must be in range [0, 64]")
    if np.any(instance_id > 999):
        raise ValueError("Instance ID must be in range [0, 999]")

    global_id = 1000 * semantic_id + instance_id
    return global_id.astype(np.uint16)


def occ_dense2sparse(
        dense_data: Union[np.ndarray, torch.Tensor],
        free_id: int,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert dense voxel grid to sparse representation.

    Args:
        dense_data: 3D array [X, Y, Z] with semantic/global values.
        free_id: Value treated as empty/free (excluded from output).

    Returns:
        Sparse array/tensor [N, 4] with columns (x, y, z, value).
    """
    if isinstance(dense_data, torch.Tensor):
        assert dense_data.ndim == 3, "dense_data must be 3D"
        occupied_mask = dense_data != free_id
        indices = torch.nonzero(occupied_mask, as_tuple=False)
        values = dense_data[occupied_mask].reshape(-1, 1)
        dtype = torch.promote_types(indices.dtype, values.dtype)
        return torch.cat((indices.to(dtype), values.to(dtype)), dim=1)

    dense_data = np.asarray(dense_data)
    assert dense_data.ndim == 3, "dense_data must be 3D"

    occupied_mask = dense_data != free_id
    indices = np.argwhere(occupied_mask)  # [N, 3]
    values = dense_data[occupied_mask]
    return np.hstack((indices, values.reshape(-1, 1)))


def occ_sparse2dense(
        sparse_data: Union[np.ndarray, torch.Tensor],
        volume_size: Tuple[int, int, int],
        free_id: int,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Convert sparse occupancy to dense voxel grid.

    Args:
        sparse_data: [N, 4] array/tensor with columns (x, y, z, value).
        volume_size: (X, Y, Z) shape of the output grid.
        free_id: Fill value for non-occupied/free voxels.

    Returns:
        Dense 3D array/tensor [X, Y, Z] with same dtype as sparse_data values.
    """
    if isinstance(sparse_data, torch.Tensor):
        sparse = sparse_data
        assert sparse.ndim == 2 and sparse.shape[1] == 4, \
            f"sparse_data must be [N, 4], got {tuple(sparse.shape)}"

        coords = sparse[:, :3].long()
        values = sparse[:, 3]
        dense_grid = values.new_full(tuple(volume_size), free_id)
        if sparse.shape[0] > 0:
            dense_grid[coords[:, 0], coords[:, 1], coords[:, 2]] = values
        return dense_grid

    sparse = np.asarray(sparse_data)
    assert sparse.ndim == 2 and sparse.shape[1] == 4, \
        f"sparse_data must be [N, 4], got {sparse.shape}"

    coords = sparse[:, :3].astype(np.intp)
    values = sparse[:, 3]
    dense_grid = np.full(volume_size, free_id, dtype=values.dtype)
    if sparse.shape[0] > 0:
        dense_grid[coords[:, 0], coords[:, 1], coords[:, 2]] = values
    return dense_grid


def encode_points(points, pc_range=None):
    points = points.clone()
    points[..., 0] = (points[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
    points[..., 1] = (points[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
    points[..., 2] = (points[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])
    return points


def decode_points(points, pc_range=None):
    points = points.clone()
    points[..., 0] = points[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
    points[..., 1] = points[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
    points[..., 2] = points[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]
    return points


def encode_bbox(bboxes, pc_range=None):
    xyz = bboxes[..., 0:3].clone()
    wlh = bboxes[..., 3:6].log()
    rot = bboxes[..., 6:7]

    if pc_range is not None:
        xyz[..., 0] = (xyz[..., 0] - pc_range[0]) / (pc_range[3] - pc_range[0])
        xyz[..., 1] = (xyz[..., 1] - pc_range[1]) / (pc_range[4] - pc_range[1])
        xyz[..., 2] = (xyz[..., 2] - pc_range[2]) / (pc_range[5] - pc_range[2])

    if bboxes.shape[-1] > 7:
        vel = bboxes[..., 7:9].clone()
        return torch.cat([xyz, wlh, rot.sin(), rot.cos(), vel], dim=-1)
    else:
        return torch.cat([xyz, wlh, rot.sin(), rot.cos()], dim=-1)


def decode_bbox(bboxes, pc_range=None):
    xyz = bboxes[..., 0:3].clone()
    wlh = bboxes[..., 3:6].exp()
    rot = torch.atan2(bboxes[..., 6:7], bboxes[..., 7:8])

    if pc_range is not None:
        xyz[..., 0] = xyz[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0]
        xyz[..., 1] = xyz[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1]
        xyz[..., 2] = xyz[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2]

    if bboxes.shape[-1] > 8:
        vel = bboxes[..., 8:10].clone()
        return torch.cat([xyz, wlh, rot, vel], dim=-1)
    else:
        return torch.cat([xyz, wlh, rot], dim=-1)
