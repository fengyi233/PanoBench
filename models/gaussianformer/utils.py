"""Shared primitives for the GaussianFormer modules."""

from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

LOGIT_MAX = 0.9999
SIGMOID_MAX = 9.21


class GaussianPrediction(NamedTuple):
    means: Tensor
    scales: Tensor
    rotations: Tensor
    opacities: Tensor
    semantics: Tensor


def safe_sigmoid(tensor):
    return torch.sigmoid(torch.clamp(tensor, -SIGMOID_MAX, SIGMOID_MAX))


def safe_inverse_sigmoid(tensor):
    tensor = torch.clamp(tensor, 1 - LOGIT_MAX, LOGIT_MAX)
    return torch.log(tensor / (1 - tensor))


def cartesian(anchor, pc_range, use_sigmoid=True):
    """Map normalised anchor coordinates to metric coordinates in ``pc_range``."""
    if use_sigmoid:
        xyz = safe_sigmoid(anchor[..., :3])
    else:
        xyz = anchor[..., :3].clamp(min=1e-6, max=1 - 1e-6)
    return torch.stack([
        xyz[..., 0] * (pc_range[3] - pc_range[0]) + pc_range[0],
        xyz[..., 1] * (pc_range[4] - pc_range[1]) + pc_range[1],
        xyz[..., 2] * (pc_range[5] - pc_range[2]) + pc_range[2],
    ], dim=-1)


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers


def get_rotation_matrix(tensor):
    """Convert (w, x, y, z) quaternions into 3x3 rotation matrices."""
    assert tensor.shape[-1] == 4

    tensor = F.normalize(tensor, dim=-1)
    mat1 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat1[..., 0, 0] = tensor[..., 0]
    mat1[..., 0, 1] = -tensor[..., 1]
    mat1[..., 0, 2] = -tensor[..., 2]
    mat1[..., 0, 3] = -tensor[..., 3]

    mat1[..., 1, 0] = tensor[..., 1]
    mat1[..., 1, 1] = tensor[..., 0]
    mat1[..., 1, 2] = -tensor[..., 3]
    mat1[..., 1, 3] = tensor[..., 2]

    mat1[..., 2, 0] = tensor[..., 2]
    mat1[..., 2, 1] = tensor[..., 3]
    mat1[..., 2, 2] = tensor[..., 0]
    mat1[..., 2, 3] = -tensor[..., 1]

    mat1[..., 3, 0] = tensor[..., 3]
    mat1[..., 3, 1] = -tensor[..., 2]
    mat1[..., 3, 2] = tensor[..., 1]
    mat1[..., 3, 3] = tensor[..., 0]

    mat2 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat2[..., 0, 0] = tensor[..., 0]
    mat2[..., 0, 1] = -tensor[..., 1]
    mat2[..., 0, 2] = -tensor[..., 2]
    mat2[..., 0, 3] = -tensor[..., 3]

    mat2[..., 1, 0] = tensor[..., 1]
    mat2[..., 1, 1] = tensor[..., 0]
    mat2[..., 1, 2] = tensor[..., 3]
    mat2[..., 1, 3] = -tensor[..., 2]

    mat2[..., 2, 0] = tensor[..., 2]
    mat2[..., 2, 1] = -tensor[..., 3]
    mat2[..., 2, 2] = tensor[..., 0]
    mat2[..., 2, 3] = tensor[..., 1]

    mat2[..., 3, 0] = tensor[..., 3]
    mat2[..., 3, 1] = tensor[..., 2]
    mat2[..., 3, 2] = -tensor[..., 1]
    mat2[..., 3, 3] = tensor[..., 0]

    mat2 = torch.conj(mat2).transpose(-1, -2)
    return torch.matmul(mat1, mat2)[..., 1:, 1:]
