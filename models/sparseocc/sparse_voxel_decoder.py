import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks.transformer import FFN
from mmengine.model import BaseModule

from models.layers import AdaptiveMixing, SelfAttention
from utils.data_converter import encode_bbox
from .sparsebev_transformer import SparseBEVSampling


@torch.no_grad()
def generate_grid(n_vox, interval, device='cuda'):
    grid_range = [torch.arange(0, n_vox[axis], interval, device=device)
                  for axis in range(3)]
    grid = torch.stack(torch.meshgrid(
        grid_range[0], grid_range[1], grid_range[2], indexing='ij'))
    grid = grid.view(3, -1).permute(1, 0)
    return grid[None]


def batch_indexing(
        batched_data: torch.Tensor,
        batched_indices: torch.Tensor):
    """
    Channel-last batch indexing.
    """
    assert batched_data.shape[0] == batched_indices.shape[0]
    batch_size = batched_data.shape[0]
    view_shape = [batch_size] + [1] * (len(batched_indices.shape) - 1)
    expand_shape = [batch_size] + list(batched_indices.shape)[1:]
    indices_of_batch = torch.arange(
        batch_size, dtype=torch.long, device=batched_data.device)
    indices_of_batch = indices_of_batch.view(view_shape).expand(expand_shape)
    if len(batched_data.shape) == 2:
        return batched_data[indices_of_batch, batched_indices.to(torch.long)]
    return batched_data[indices_of_batch, batched_indices.to(torch.long), :]


def index2point(coords, pc_range, voxel_size):
    """Convert voxel indices to 3D point coordinates.

    Args:
        coords: [B, N, 3] - Voxel indices (integer coordinates)
        pc_range: [x_min, y_min, z_min, x_max, y_max, z_max] - Point cloud range
        voxel_size: float - Size of each voxel

    Returns:
        [B, N, 3] - 3D point coordinates in world space
    """
    # Scale indices by voxel size and add minimum point cloud range
    coords = coords.float()
    voxel_size = torch.as_tensor(voxel_size, device=coords.device, dtype=coords.dtype)
    coords = coords * voxel_size
    coords = coords + torch.tensor(pc_range[:3], device=coords.device, dtype=coords.dtype)
    return coords


def point2bbox(coords, box_size):
    """Convert point coordinates to bounding boxes.

    Args:
        coords: [B, N, 3] - 3D point coordinates (center positions)
        box_size: float - Size of bounding box in meters (width, length, height)

    Returns:
        [B, N, 6] - Bounding boxes [x, y, z, w, l, h]
    """
    # Create boxes with uniform size centered at points
    box_size = torch.as_tensor(box_size, device=coords.device, dtype=coords.dtype)
    wlh = torch.ones_like(coords.float()) * box_size
    bboxes = torch.cat([coords, wlh], dim=-1)  # [B, N, 6]
    return bboxes


def upsample(pre_feat, pre_coords, interval):
    """Octree-style 8x upsampling: split each voxel into 8 child voxels.

    Each parent voxel is divided into 8 children by splitting along x, y, z axes.
    Feature channels are evenly distributed among the 8 children.

    Args:
        pre_feat: [B, N, C] - Features from previous level
        pre_coords: [B, N, 3] - Voxel coordinates from previous level
        interval: int - Voxel interval (spacing between voxels)

    Returns:
        up_feat: [B, N*8, C//8] - Upsampled features (8x more voxels, 1/8 channels)
        up_coords: [B, N*8, 3] - Upsampled coordinates
    """
    # Offset patterns for 8 children: [x], [y], [z], [x,y], [x,z], [y,z], [x,y,z]
    pos_list = [0, 1, 2, [0, 1], [0, 2], [1, 2], [0, 1, 2]]
    bs, num_query, num_channels = pre_feat.shape

    # Split features into 8 parts: [B, N, C] -> [B, N, 8, C//8]
    up_feat = pre_feat.reshape(bs, num_query, 8, num_channels // 8)  # [B, N, 8, C/8]
    # Replicate coordinates for 8 children: [B, N, 3] -> [B, N, 8, 3]
    up_coords = pre_coords.unsqueeze(2).repeat(1, 1, 8, 1).contiguous()  # [B, N, 8, 3]
    # Apply offsets to create 8 child positions
    for i in range(len(pos_list)):
        up_coords[:, :, i + 1, pos_list[i]] += interval

    # Flatten: [B, N, 8, ...] -> [B, N*8, ...]
    up_feat = up_feat.reshape(bs, -1, num_channels // 8)
    up_coords = up_coords.reshape(bs, -1, 3)

    return up_feat, up_coords


class SparseVoxelDecoder(BaseModule):
    """Sparse voxel decoder with coarse-to-fine hierarchical decoding.

    Decodes sparse voxel features from multi-level image features through multiple stages.
    Each stage upsamples voxels 8x (octree-style) and sparsifies using top-k selection.
    """

    def __init__(self,
                 embed_dims=None,
                 num_layers=None,
                 num_frames=None,
                 num_views=6,
                 num_points=None,
                 num_groups=None,
                 num_levels=None,
                 num_classes=None,
                 free_id=0,
                 voxel_size=None,
                 max_self_attn_queries=4096,
                 semantic=False,
                 topk_training=None,
                 topk_testing=None,
                 pc_range=None,
                 occ_size=None):
        super().__init__()

        self.embed_dims = embed_dims
        self.num_frames = num_frames
        self.num_views = int(num_views)
        self.num_layers = num_layers
        self.pc_range = pc_range
        self.free_id = int(free_id)
        self.semantic = semantic
        self.voxel_dim = list(occ_size)  # Voxel grid dimensions [W, H, D]
        if voxel_size is None:
            self.voxel_size = [
                (pc_range[3] - pc_range[0]) / self.voxel_dim[0],
                (pc_range[4] - pc_range[1]) / self.voxel_dim[1],
                (pc_range[5] - pc_range[2]) / self.voxel_dim[2],
            ]
        else:
            self.voxel_size = [float(voxel_size)] * 3 if isinstance(voxel_size, (int, float)) else list(voxel_size)
        self.topk_training = topk_training  # Topk values for training (per layer)
        self.topk_testing = topk_testing  # Topk values for testing (per layer)

        self.decoder_layers = nn.ModuleList()
        self.lift_feat_heads = nn.ModuleList()  # Feature expansion before upsampling

        if semantic:
            self.seg_pred_heads = nn.ModuleList()

        # Build decoder layers: each layer processes at different resolution
        for i in range(num_layers):
            self.decoder_layers.append(SparseVoxelDecoderLayer(
                embed_dims=embed_dims,
                num_frames=num_frames,
                num_views=self.num_views,
                num_points=num_points // (2 ** i),
                num_groups=num_groups,
                num_levels=num_levels,
                pc_range=pc_range,
                self_attn=i in [0, 1],
                max_self_attn_queries=max_self_attn_queries,
            ))
            # Expand features 8x before upsampling: [B, N, C] -> [B, N, 8C]
            self.lift_feat_heads.append(nn.Sequential(
                nn.Linear(embed_dims, embed_dims * 8),
                nn.ReLU(inplace=True)
            ))

            if semantic:
                # Semantic segmentation prediction head
                self.seg_pred_heads.append(nn.Linear(embed_dims, num_classes))

    @torch.no_grad()
    def init_weights(self):
        for i in range(len(self.decoder_layers)):
            self.decoder_layers[i].init_weights()

    def forward(self, mlvl_feats, img_metas):
        occ_preds = []  # List of predictions per layer: (occ_loc, occ_pred, seg_pred, mask_feat, scale)

        topk = self.topk_training if self.training else self.topk_testing

        B = len(img_metas)
        device = mlvl_feats[0].device
        # Initialize query coordinates: start with the coarsest grid
        interval = 2 ** self.num_layers  # Initial interval (e.g., 8)
        query_coord = generate_grid(self.voxel_dim, interval, device=device).expand(B, -1, -1)  # [B, N=25*25*2=1250, 3]
        query_feat = torch.zeros([B, query_coord.shape[1], self.embed_dims], device=device)  # [B, N, C=256]

        # Coarse-to-fine decoding through multiple layers
        for i, layer in enumerate(self.decoder_layers):

            # Interval decreases: 8 -> 4 -> 2 -> 1 (finer resolution)
            interval = 2 ** (self.num_layers - i)  # 8 4 2 1

            # Convert voxel indices to bounding boxes for transformer
            query_bbox = index2point(query_coord, self.pc_range, self.voxel_size)  # [B, N, 3]
            query_bbox = point2bbox(
                query_bbox,
                box_size=[v * interval for v in self.voxel_size],
            )  # [B, N, 6]
            query_bbox = encode_bbox(query_bbox, pc_range=self.pc_range)  # [B, N, 6]

            # SparseVoxelDecoderLayer: sample features from images and update query features
            query_feat = layer(query_feat, query_bbox, mlvl_feats, img_metas)  # [B, N, C=256]

            # Upsample: expand features and split each voxel into 8 children
            query_feat = self.lift_feat_heads[i](query_feat)  # [B, N, 8C]
            query_feat_2x, query_coord_2x = upsample(query_feat, query_coord, interval // 2)  # [B, 8N, C//8]

            # Semantic prediction at upsampled resolution
            if self.semantic:
                seg_pred_2x = self.seg_pred_heads[i](query_feat_2x)  # [B, 8N, num_classes]
            else:
                seg_pred_2x = None

            # Sparsify: keep only topk most likely non-empty voxels
            non_free_prob = 1 - F.softmax(seg_pred_2x, dim=-1)[..., self.free_id]  # [B, 8N]
            k = min(int(topk[i]), non_free_prob.shape[1])
            indices = torch.topk(non_free_prob, k=k, dim=1)[1]  # [B, K]

            # Select topk voxels
            query_coord_2x = batch_indexing(query_coord_2x, indices)  # [B, K, 3]
            query_feat_2x = batch_indexing(query_feat_2x, indices)  # [B, K, C]
            seg_pred_2x = batch_indexing(seg_pred_2x, indices)  # [B, K, num_classes]

            # Store predictions for this layer
            occ_preds.append((
                torch.div(query_coord_2x, interval // 2, rounding_mode='trunc').long(),  # Normalized coordinates
                None,  # occ_pred (not used)
                seg_pred_2x,  # Semantic predictions
                query_feat_2x,  # Voxel features
                interval // 2)  # Scale
            )

            # Update for next layer (detach to stop gradients)
            query_coord = query_coord_2x.detach()
            query_feat = query_feat_2x.detach()

        return occ_preds


class SparseVoxelDecoderLayer(BaseModule):
    """Single decoder layer for sparse voxel decoding.

    Architecture: Position Encoding -> Self-Attention (optional) -> Sampling -> Adaptive Mixing -> FFN
    """

    def __init__(self,
                 embed_dims=None,
                 num_frames=None,
                 num_views=6,
                 num_points=None,
                 num_groups=None,
                 num_levels=None,
                 pc_range=None,
                 self_attn=True,
                 max_self_attn_queries=4096):
        super().__init__()
        self.max_self_attn_queries = int(max_self_attn_queries)

        # Position encoder: encodes 3D bbox centers to positional embeddings
        self.position_encoder = nn.Sequential(
            nn.Linear(3, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dims, embed_dims),
            nn.LayerNorm(embed_dims),
            nn.ReLU(inplace=True),
        )

        # Self-attention: only used in first two layers for voxel interaction
        if self_attn:
            self.self_attn = SelfAttention(embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range)
            self.norm1 = nn.LayerNorm(embed_dims)
        else:
            self.self_attn = None

        # Multi-view feature sampling from image features
        self.sampling = SparseBEVSampling(
            embed_dims=embed_dims,
            num_frames=num_frames,
            num_views=num_views,
            num_groups=num_groups,
            num_points=num_points,
            num_levels=num_levels,
            pc_range=pc_range
        )
        # Adaptive mixing: combines sampled features with query features
        self.mixing = AdaptiveMixing(
            in_dim=embed_dims,
            in_points=num_points * num_frames,
            n_groups=num_groups,
            out_points=num_points * num_frames * num_groups
        )
        # Feed-forward network
        self.ffn = FFN(embed_dims, feedforward_channels=embed_dims * 2, ffn_drop=0.1)

        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

    @torch.no_grad()
    def init_weights(self):
        if self.self_attn is not None:
            self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()
        self.ffn.init_weights()

    def forward(self, query_feat, query_bbox, mlvl_feats, img_metas):
        """Forward pass through decoder layer.

        Args:
            query_feat: [B, N, C] - Query features
            query_bbox: [B, N, 6] - Bounding boxes (normalized)
            mlvl_feats: Multi-level image features
            img_metas: Image metadata

        Returns:
            [B, N, C] - Updated query features
        """
        # Add positional encoding from bbox centers
        query_pos = self.position_encoder(query_bbox[..., :3])  # [B, N, C]
        query_feat = query_feat + query_pos

        # Self-attention: voxel-to-voxel interaction
        if self.self_attn is not None and query_feat.shape[1] <= self.max_self_attn_queries:
            query_feat = self.norm1(self.self_attn(query_bbox, query_feat, query_mode='bbox'))
        # Sample features from multi-view images
        sampled_feat = self.sampling(query_bbox, query_feat, mlvl_feats, img_metas)
        # Adaptive mixing: combine sampled features with query features
        query_feat = self.norm2(self.mixing(sampled_feat, query_feat))
        # Feed-forward network
        query_feat = self.norm3(self.ffn(query_feat))

        return query_feat
