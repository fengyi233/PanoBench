"""Building blocks of a GaussianFormer decoder layer."""

from typing import List

import numpy as np
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale, build_activation_layer, build_norm_layer
from mmcv.cnn.bricks.drop import build_dropout
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule, constant_init, xavier_init

from models.csrc.wrapper import deformable_aggregation, flatten_multi_scale_feats
from models.gaussianformer.utils import (GaussianPrediction, cartesian, get_rotation_matrix,
                                         linear_relu_ln, safe_sigmoid)


@MODELS.register_module()
class SparseGaussian3DEncoder(BaseModule):
    """Embed the raw anchor parameters into the instance feature space."""

    def __init__(self, embed_dims=256, include_opa=True, semantic_dim=None, init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.include_opa = include_opa
        self.semantic_dim = int(semantic_dim)
        self.semantic_start = 10 + int(include_opa)

        def embedding_layer(input_dims):
            return nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, input_dims))

        self.xyz_fc = embedding_layer(3)
        self.scale_fc = embedding_layer(3)
        self.rot_fc = embedding_layer(4)
        if include_opa:
            self.opacity_fc = embedding_layer(1)
        self.semantics_fc = embedding_layer(self.semantic_dim)
        self.output_fc = embedding_layer(self.embed_dims)

    def forward(self, anchor: torch.Tensor):
        output = self.xyz_fc(anchor[..., :3])
        output = output + self.scale_fc(anchor[..., 3:6])
        output = output + self.rot_fc(anchor[..., 6:10])
        if self.include_opa:
            output = output + self.opacity_fc(anchor[..., 10:11])
        semantic_end = self.semantic_start + self.semantic_dim
        output = output + self.semantics_fc(anchor[..., self.semantic_start:semantic_end])
        return self.output_fc(output)


@MODELS.register_module()
class GaussianAsymmetricFFN(BaseModule):
    """FFN whose input width may differ from ``embed_dims`` (residual-cat friendly)."""

    def __init__(self,
                 in_channels=None,
                 pre_norm=None,
                 embed_dims=256,
                 feedforward_channels=1024,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 ffn_drop=0.0,
                 dropout_layer=None,
                 add_identity=True,
                 init_cfg=None):
        super().__init__(init_cfg)
        assert num_fcs >= 2, f'num_fcs should be no less than 2. got {num_fcs}.'
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.pre_norm = None

        if in_channels is None:
            in_channels = embed_dims
        if pre_norm is not None:
            self.pre_norm = build_norm_layer(pre_norm, in_channels)[1]

        activate = build_activation_layer(act_cfg)
        layers = []
        for _ in range(num_fcs - 1):
            layers.append(nn.Sequential(
                nn.Linear(in_channels, feedforward_channels),
                activate,
                nn.Dropout(ffn_drop),
            ))
            in_channels = feedforward_channels
        layers.append(nn.Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = nn.Sequential(*layers)

        self.dropout_layer = build_dropout(dropout_layer) if dropout_layer else nn.Identity()
        self.add_identity = add_identity
        if self.add_identity:
            self.identity_fc = (
                nn.Identity() if self.in_channels == embed_dims
                else nn.Linear(self.in_channels, embed_dims))

    def forward(self, x, identity=None):
        if self.pre_norm is not None:
            x = self.pre_norm(x)
        out = self.layers(x)
        if not self.add_identity:
            return self.dropout_layer(out)
        if identity is None:
            identity = x
        return self.identity_fc(identity) + self.dropout_layer(out)


@MODELS.register_module()
class SparseGaussian3DKeyPointsGenerator(BaseModule):
    """Sample keypoints inside each Gaussian ellipsoid for image feature lookup."""

    def __init__(self,
                 embed_dims=256,
                 num_learnable_pts=0,
                 learnable_fixed_scale=1,
                 fix_scale=None,
                 pc_range=None,
                 scale_range=None,
                 xyz_activation='sigmoid',
                 scale_activation='sigmoid',
                 init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_learnable_pts = num_learnable_pts
        self.learnable_fixed_scale = learnable_fixed_scale
        self.fix_scale = np.array(((0.0, 0.0, 0.0),) if fix_scale is None else fix_scale)
        self.num_pts = len(self.fix_scale) + num_learnable_pts
        if num_learnable_pts > 0:
            self.learnable_fc = nn.Linear(self.embed_dims, num_learnable_pts * 3)

        self.pc_range = pc_range
        self.scale_range = scale_range
        self.xyz_act = xyz_activation
        self.scale_act = scale_activation

    def init_weight(self):
        if self.num_learnable_pts > 0:
            xavier_init(self.learnable_fc, distribution='uniform', bias=0.0)

    def forward(self, anchor, instance_feature=None):
        bs, num_anchor = anchor.shape[:2]
        scale = anchor.new_tensor(self.fix_scale)[None, None].tile([bs, num_anchor, 1, 1])
        if self.num_learnable_pts > 0 and instance_feature is not None:
            learnable_scale = safe_sigmoid(
                self.learnable_fc(instance_feature)
                .reshape(bs, num_anchor, self.num_learnable_pts, 3)) - 0.5
            scale = torch.cat([scale, learnable_scale * self.learnable_fixed_scale], dim=-2)

        gs_scales = anchor[..., None, 3:6]
        if self.scale_act == 'sigmoid':
            gs_scales = safe_sigmoid(gs_scales)
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales

        rotation_mat = get_rotation_matrix(anchor[..., 6:10]).transpose(-1, -2)
        key_points = torch.matmul(
            rotation_mat[:, :, None], (scale * gs_scales)[..., None]).squeeze(-1)

        xyz = cartesian(anchor, self.pc_range, use_sigmoid=self.xyz_act == 'sigmoid')
        return key_points + xyz.unsqueeze(2)


@MODELS.register_module()
class DeformableFeatureAggregation(BaseModule):
    """Aggregate multi-view multi-scale image features at the Gaussian keypoints."""

    def __init__(self,
                 embed_dims: int = 256,
                 num_groups: int = 8,
                 num_levels: int = 4,
                 num_cams: int = 6,
                 proj_drop: float = 0.0,
                 attn_drop: float = 0.0,
                 kps_generator: dict = None,
                 use_camera_embed=False,
                 residual_mode='add',
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_groups != 0:
            raise ValueError('embed_dims must be divisible by num_groups, '
                             f'but got {embed_dims} and {num_groups}')
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_groups = num_groups
        self.num_cams = num_cams
        self.attn_drop = attn_drop
        self.residual_mode = residual_mode
        self.proj_drop = nn.Dropout(proj_drop)

        kps_generator = dict(kps_generator, embed_dims=embed_dims)
        self.kps_generator = MODELS.build(kps_generator)
        self.num_pts = self.kps_generator.num_pts
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        if use_camera_embed:
            self.camera_encoder = nn.Sequential(*linear_relu_ln(embed_dims, 1, 2, 12))
            self.weights_fc = nn.Linear(embed_dims, num_groups * num_levels * self.num_pts)
        else:
            self.camera_encoder = None
            self.weights_fc = nn.Linear(
                embed_dims, num_groups * num_cams * num_levels * self.num_pts)

    def init_weight(self):
        constant_init(self.weights_fc, val=0.0, bias=0.0)
        xavier_init(self.output_proj, distribution='uniform', bias=0.0)

    def forward(self,
                instance_feature: torch.Tensor,
                anchor: torch.Tensor,
                anchor_embed: torch.Tensor,
                feature_maps: List[torch.Tensor],
                metas: dict):
        bs, num_anchor = instance_feature.shape[:2]
        key_points = self.kps_generator(anchor, instance_feature)

        weights, weight_mask = self._get_weights(instance_feature, anchor_embed, metas)
        shape = (bs, num_anchor, self.num_pts, self.num_cams, self.num_levels, self.num_groups)
        weights = weights.permute(0, 1, 4, 2, 3, 5).contiguous().reshape(shape)
        weight_mask = weight_mask.permute(0, 1, 4, 2, 3, 5).contiguous().reshape(shape)

        points_2d, mask = self.project_points(
            key_points, metas['projection_mat'], metas.get('image_wh'))
        points_2d = points_2d.permute(0, 2, 3, 1, 4).reshape(
            bs, num_anchor * self.num_pts, self.num_cams, 2)

        # A keypoint that misses every view would softmax over an all -inf row,
        # so those anchors keep uniform logits and are zeroed out afterwards.
        mask = mask.permute(0, 2, 3, 1)[..., None, None] & weight_mask
        all_miss = mask.sum(dim=[2, 3, 4], keepdim=True) == 0
        all_miss = all_miss.expand(-1, -1, self.num_pts, self.num_cams, self.num_levels, -1)
        weights[~mask] = -torch.inf
        weights[all_miss] = 0.0
        weights = weights.flatten(2, 4).softmax(dim=-2).reshape(
            bs, num_anchor * self.num_pts, self.num_cams, self.num_levels, self.num_groups)
        weights = weights * (1 - all_miss.flatten(1, 2).float())

        features = deformable_aggregation(
            flatten_multi_scale_feats(feature_maps), points_2d, weights)
        features = features.reshape(bs, num_anchor, self.num_pts, self.embed_dims).sum(dim=2)

        output = self.proj_drop(self.output_proj(features))
        if self.residual_mode == 'add':
            return output + instance_feature
        if self.residual_mode == 'cat':
            return torch.cat([output, instance_feature], dim=-1)
        raise NotImplementedError(f'Unsupported residual mode: {self.residual_mode}')

    def _get_weights(self, instance_feature, anchor_embed, metas=None):
        bs, num_anchor = instance_feature.shape[:2]
        feature = instance_feature + anchor_embed
        if self.camera_encoder is not None:
            camera_embed = self.camera_encoder(
                metas['projection_mat'][:, :, :3].reshape(bs, self.num_cams, -1))
            feature = feature[:, :, None] + camera_embed[:, None]
        weights = self.weights_fc(feature).reshape(
            bs, num_anchor, self.num_cams, self.num_levels, self.num_pts, self.num_groups)
        if self.training and self.attn_drop > 0:
            mask = torch.rand_like(weights) > self.attn_drop
        else:
            mask = torch.ones_like(weights) > 0
        return weights, mask

    @staticmethod
    def project_points(key_points, projection_mat, image_wh=None):
        pts_extend = torch.cat([key_points, torch.ones_like(key_points[..., :1])], dim=-1)
        points_2d = torch.matmul(
            projection_mat[:, :, None, None], pts_extend[:, None, ..., None]).squeeze(-1)
        depth = points_2d[..., 2]
        points_2d = points_2d[..., :2] / torch.clamp(points_2d[..., 2:3], min=1e-5)
        if image_wh is not None:
            points_2d = points_2d / image_wh[:, :, None, None]
        mask = ((depth > 1e-5)
                & (points_2d[..., 0] > 0) & (points_2d[..., 0] < 1)
                & (points_2d[..., 1] > 0) & (points_2d[..., 1] < 1))
        return points_2d, mask


@MODELS.register_module()
class SparseGaussian3DRefinementModule(BaseModule):
    """Predict the per-layer residual update of every Gaussian anchor."""

    def __init__(self,
                 embed_dims=256,
                 pc_range=None,
                 scale_range=None,
                 restrict_xyz=False,
                 unit_xyz=None,
                 refine_manual=None,
                 semantic_dim=None,
                 include_opa=True,
                 semantics_activation='softmax',
                 xyz_activation='sigmoid',
                 scale_activation='sigmoid',
                 init_cfg=None):
        super().__init__(init_cfg)
        self.embed_dims = embed_dims
        self.semantic_dim = int(semantic_dim)
        self.semantic_start = 10 + int(include_opa)
        self.output_dim = self.semantic_start + self.semantic_dim
        self.include_opa = include_opa
        self.semantics_activation = semantics_activation
        self.xyz_act = xyz_activation
        self.scale_act = scale_activation

        self.pc_range = pc_range
        self.scale_range = scale_range
        self.restrict_xyz = restrict_xyz
        if restrict_xyz:
            assert unit_xyz is not None
            unit_prob = [unit_xyz[i] / (pc_range[i + 3] - pc_range[i]) for i in range(3)]
            if xyz_activation == 'sigmoid':
                unit_prob = [4 * p for p in unit_prob]
            self.unit_sigmoid = unit_prob

        assert isinstance(refine_manual, list)
        assert all(refine_manual[i] == i for i in range(len(refine_manual)))
        self.refine_state = refine_manual

        self.layers = nn.Sequential(
            *linear_relu_ln(embed_dims, 2, 2),
            nn.Linear(self.embed_dims, self.output_dim),
            Scale([1.0] * self.output_dim))

    def forward(self,
                instance_feature: torch.Tensor,
                anchor: torch.Tensor,
                anchor_embed: torch.Tensor):
        output = self.layers(instance_feature + anchor_embed)

        if self.restrict_xyz:
            delta_xyz_prob = 2 * safe_sigmoid(output[..., :3]) - 1
            delta_xyz = torch.stack([
                delta_xyz_prob[..., 0] * self.unit_sigmoid[0],
                delta_xyz_prob[..., 1] * self.unit_sigmoid[1],
                delta_xyz_prob[..., 2] * self.unit_sigmoid[2],
            ], dim=-1)
            output = torch.cat([delta_xyz, output[..., 3:]], dim=-1)

        if len(self.refine_state) > 0:
            refined = output[..., self.refine_state] + anchor[..., self.refine_state]
            output = torch.cat([refined, output[..., len(self.refine_state):]], dim=-1)

        if self.xyz_act == 'sigmoid':
            xyz = output[..., :3]
        else:
            xyz = output[..., :3].clamp(min=1e-6, max=1 - 1e-6)
        if self.scale_act == 'sigmoid':
            scale = output[..., 3:6]
        else:
            scale = output[..., 3:6].clamp(min=1e-6, max=1 - 1e-6)
        rot = F.normalize(output[..., 6:10], dim=-1)
        output = torch.cat([xyz, scale, rot, output[..., 10:]], dim=-1)

        means = cartesian(xyz, self.pc_range, use_sigmoid=self.xyz_act == 'sigmoid')
        if self.scale_act == 'sigmoid':
            scale = safe_sigmoid(scale)
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * scale

        semantics = output[..., self.semantic_start:(self.semantic_start + self.semantic_dim)]
        if self.semantics_activation == 'softmax':
            semantics = semantics.softmax(dim=-1)
        elif self.semantics_activation == 'softplus':
            semantics = F.softplus(semantics)

        gaussian = GaussianPrediction(
            means=means,
            scales=gs_scales,
            rotations=rot,
            opacities=safe_sigmoid(output[..., 10:(10 + int(self.include_opa))]),
            semantics=semantics)
        return output, gaussian


@MODELS.register_module()
class GaussianSparseConv3D(BaseModule):
    """Exchange information between Gaussians that fall into neighbouring voxels."""

    def __init__(self,
                 in_channels,
                 embed_channels,
                 pc_range,
                 grid_size,
                 kernel_size=5,
                 xyz_activation='sigmoid',
                 use_out_proj=False,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.layer = spconv.SubMConv3d(
            in_channels,
            embed_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False)
        self.output_proj = nn.Linear(embed_channels, embed_channels) if use_out_proj else nn.Identity()
        self.use_sigmoid = xyz_activation == 'sigmoid'
        self.register_buffer('pc_range', torch.tensor(pc_range, dtype=torch.float))
        self.register_buffer('grid_size', torch.tensor(grid_size, dtype=torch.float))

    def forward(self, instance_feature, anchor):
        bs, g = instance_feature.shape[:2]

        anchor_xyz = cartesian(anchor, self.pc_range, use_sigmoid=self.use_sigmoid).flatten(0, 1)
        indices = ((anchor_xyz - self.pc_range[None, :3]) / self.grid_size[None, :]).to(torch.int32)
        batch_ids = torch.arange(bs, device=indices.device, dtype=torch.int32)
        batch_ids = batch_ids.reshape(bs, 1, 1).expand(-1, g, -1).flatten(0, 1)
        spatial_shape = ((self.pc_range[3:] - self.pc_range[:3]) / self.grid_size).to(torch.int32)

        sparse_input = spconv.SparseConvTensor(
            instance_feature.flatten(0, 1),
            indices=torch.cat([batch_ids, indices], dim=-1),
            spatial_shape=spatial_shape,
            batch_size=bs)
        output = self.layer(sparse_input).features.unflatten(0, (bs, g))
        return self.output_proj(output)
