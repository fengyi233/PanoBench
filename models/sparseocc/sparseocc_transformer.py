import torch
import torch.nn as nn
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS
from mmcv.cnn.bricks.transformer import FFN
from torch.utils.checkpoint import checkpoint as cp
from models.layers import AdaptiveMixing, sampling_4d

from .sparsebev_sampling import make_sample_points_from_mask
from .sparse_voxel_decoder import SparseVoxelDecoder


@MODELS.register_module()
class SparseOccTransformer(BaseModule):
    """Main transformer for sparse occupancy prediction.

    Consists of two stages:
    1. Voxel decoder: generates sparse voxel features from multi-level image features
    2. Mask decoder: performs instance and semantic segmentation using query-based approach
    """

    def __init__(self,
                 embed_dims=None,
                 num_layers=None,
                 num_queries=None,
                 num_frames=None,
                 num_views=6,
                 num_points=None,
                 num_groups=None,
                 num_levels=None,
                 num_classes=None,
                 free_id=0,
                 voxel_size=None,
                 pc_range=None,
                 occ_size=None,
                 topk_training=None,
                 topk_testing=None,
                 max_self_attn_queries=4096,
                 with_mask_decoder=True):
        super().__init__()
        self.num_frames = num_frames
        self.num_views = int(num_views)
        self.num_groups = int(num_groups)
        self.with_mask_decoder = bool(with_mask_decoder)

        self.voxel_decoder = SparseVoxelDecoder(
            embed_dims=embed_dims,
            num_layers=3,
            num_frames=num_frames,
            num_views=self.num_views,
            num_points=num_points,
            num_groups=num_groups,
            num_levels=num_levels,
            num_classes=num_classes,
            free_id=free_id,
            voxel_size=voxel_size,
            pc_range=pc_range,
            semantic=True,
            topk_training=topk_training,
            topk_testing=topk_testing,
            max_self_attn_queries=max_self_attn_queries,
            occ_size=occ_size
        )
        if self.with_mask_decoder:
            self.decoder = MaskFormerOccDecoder(
                embed_dims=embed_dims,
                num_layers=num_layers,
                num_frames=num_frames,
                num_views=self.num_views,
                num_queries=num_queries,
                num_points=num_points,
                num_groups=num_groups,
                num_levels=num_levels,
                num_classes=num_classes,
                pc_range=pc_range,
                occ_size=occ_size,
            )
        else:
            self.decoder = None

    @torch.no_grad()
    def init_weights(self):
        self.voxel_decoder.init_weights()
        if self.decoder is not None:
            self.decoder.init_weights()

    def forward(self, mlvl_feats, img_metas):
        # Reshape multi-level features: [B, TN, GC, H, W] -> [BTG, N, H, W, C]
        # TN: total num views (T frames * N cameras), GC: grouped channels (G groups * C channels)
        for lvl, feat in enumerate(mlvl_feats):
            B, TN, GC, H, W = feat.shape  # [B, TN, GC, H, W]
            N = self.num_views
            if TN % N != 0:
                raise ValueError(f"feature views {TN} is not divisible by num_views={N}")
            T, G, C = TN // N, self.num_groups, GC // self.num_groups
            feat = feat.reshape(B, T, N, G, C, H, W)
            feat = feat.permute(0, 1, 3, 2, 5, 6, 4)  # [B, T, G, N, H, W, C]
            feat = feat.reshape(B * T * G, N, H, W, C)  # [BTG, N, H, W, C]
            mlvl_feats[lvl] = feat.contiguous()

        img_metas[0]['occ2img'] = torch.stack(
            [m['occ2img'] for m in img_metas], dim=0)

        occ_preds = self.voxel_decoder(mlvl_feats, img_metas=img_metas)  # SparseVoxelDecoder
        if self.decoder is not None:
            mask_preds, class_preds = self.decoder(occ_preds, mlvl_feats, img_metas)  # MaskFormerOccDecoder
        else:
            mask_preds, class_preds = [], []

        return occ_preds, mask_preds, class_preds


class MaskFormerOccDecoder(BaseModule):
    """MaskFormer decoder for occupancy prediction.

    Uses learnable query embeddings to generate instance masks and class predictions.
    Iteratively refines predictions through multiple decoder layers.
    """

    def __init__(self,
                 embed_dims=None,
                 num_layers=None,
                 num_frames=None,
                 num_views=6,
                 num_queries=None,
                 num_points=None,
                 num_groups=None,
                 num_levels=None,
                 num_classes=None,
                 pc_range=None,
                 occ_size=None):
        super().__init__()

        self.num_layers = num_layers
        self.num_queries = num_queries
        self.num_frames = num_frames

        # Single decoder layer shared across all iterations
        self.decoder_layer = MaskFormerOccDecoderLayer(
            embed_dims=embed_dims,
            mask_dim=embed_dims,
            num_frames=num_frames,
            num_views=num_views,
            num_points=num_points,
            num_groups=num_groups,
            num_levels=num_levels,
            num_classes=num_classes,
            pc_range=pc_range,
            occ_size=occ_size,
        )

        # Learnable query embeddings for instance segmentation
        self.query_feat = nn.Embedding(num_queries, embed_dims)  # Query features
        self.query_pos = nn.Embedding(num_queries, embed_dims)  # Query positional embeddings

    @torch.no_grad()
    def init_weights(self):
        self.decoder_layer.init_weights()

    def forward(self, occ_preds, mlvl_feats, img_metas):
        # Extract voxel features from the last decoder stage
        occ_loc, _, _, mask_feat, _ = occ_preds[-1]  # occ_loc, None, sem_pred, mask/voxel_feat, scale
        bs = mask_feat.shape[0]
        # Initialize query embeddings for batch
        query_feat = self.query_feat.weight[None].repeat(bs, 1, 1)  # [B, num_queries, embed_dims]
        query_pos = self.query_pos.weight[None].repeat(bs, 1, 1)  # [B, num_queries, embed_dims]

        # Initial segmentation prediction before refinement
        valid_map, mask_pred, class_pred = self.decoder_layer.pred_segmentation(query_feat, mask_feat)

        class_preds = [class_pred]
        mask_preds = [mask_pred]

        # Iteratively refine predictions through decoder layers
        for i in range(self.num_layers):
            query_feat, valid_map, mask_pred, class_pred = self.decoder_layer(  # MaskFormerOccDecoderLayer
                query_feat, valid_map, mask_pred, occ_preds, mlvl_feats, query_pos, img_metas
            )
            mask_preds.append(mask_pred)  # [B, Q, N], Q masks for N voxels
            class_preds.append(class_pred)  # [B, Q, num_classes-1], Q class scores

        return mask_preds, class_preds


class MaskFormerOccDecoderLayer(BaseModule):
    """Single decoder layer for MaskFormer occupancy prediction.

    Architecture: Self-Attention -> Sampling -> Adaptive Mixing -> FFN -> Segmentation
    """

    def __init__(self,
                 embed_dims=None,
                 mask_dim=None,
                 num_frames=None,
                 num_views=6,
                 num_queries=None,
                 num_points=None,
                 num_groups=None,
                 num_levels=None,
                 num_classes=None,
                 pc_range=None,
                 occ_size=None):
        super().__init__()

        self.pc_range = pc_range
        self.occ_size = occ_size

        # Self-attention for query interaction
        self.self_attn = MaskFormerSelfAttention(embed_dims, num_heads=8)
        # Multi-view feature sampling from image features
        self.sampling = MaskFormerSampling(embed_dims, num_frames, num_views, num_groups, num_points, num_levels,
                                           pc_range=pc_range, occ_size=occ_size)
        # Adaptive feature mixing with learned parameters
        self.mixing = AdaptiveMixing(in_dim=embed_dims, in_points=num_points * num_frames, n_groups=num_groups,
                                     out_points=128)
        # Feed-forward network
        self.ffn = FFN(embed_dims, feedforward_channels=512, ffn_drop=0.1)
        # Projection heads for segmentation
        self.mask_proj = nn.Linear(embed_dims, mask_dim)  # Mask prediction
        self.classifier = nn.Linear(embed_dims, num_classes - 1)  # Excludes the no-object query class.

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

    @torch.no_grad()
    def init_weights(self):
        self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()
        self.ffn.init_weights()

    def forward(self, query_feat, valid_map, mask_pred, occ_preds, mlvl_feats, query_pos, img_metas):
        """
        Args:
            query_feat: [bs, num_query, embed_dim] - Query features
            valid_map: [bs, num_query, num_voxel] - Valid voxel mask for each query
            mask_pred: [bs, num_query, num_voxel] - Previous mask predictions
            occ_preds: list of (occ_loc, occ_pred, seg_pred, mask_feat, scale) from voxel decoder
            mlvl_feats: Multi-level image features
            query_pos: [bs, num_query, embed_dim] - Query positional embeddings
            img_metas: Image metadata including transformation matrices
        """
        occ_loc, _, _, mask_feat, _ = occ_preds[-1]
        # Self-attention: query-to-query interaction
        query_feat = self.norm1(self.self_attn(query_feat, query_pos=query_pos))

        # Multi-view feature sampling from images
        sampled_feat = self.sampling(query_feat, valid_map, occ_loc, mlvl_feats, img_metas)
        # Adaptive mixing: combine sampled features with query features
        query_feat = self.norm2(self.mixing(sampled_feat, query_feat))

        # Feed-forward network
        query_feat = self.norm3(self.ffn(query_feat))

        # Update segmentation predictions
        valid_map, mask_pred, class_pred = self.pred_segmentation(query_feat, mask_feat)
        return query_feat, valid_map, mask_pred, class_pred

    def pred_segmentation(self, query_feat, mask_feat):
        """Predict segmentation masks and classes (with official checkpointing)."""
        if self.training and query_feat.requires_grad:
            return cp(self.inner_pred_segmentation, query_feat, mask_feat, use_reentrant=False)
        return self.inner_pred_segmentation(query_feat, mask_feat)

    def inner_pred_segmentation(self, query_feat, mask_feat):
        class_pred = self.classifier(query_feat)  # [B, Q, num_classes-1]
        feat_proj = self.mask_proj(query_feat)  # [B, Q, C]
        mask_pred = torch.einsum("bqc,bnc->bqn", feat_proj, mask_feat)  # [B, Q, N]
        valid_map = (mask_pred > 0.0)
        return valid_map, mask_pred, class_pred


class MaskFormerSelfAttention(BaseModule):
    """Self-attention module for query-to-query interaction.

    Enables queries to attend to each other for better feature refinement.
    """

    def __init__(self, embed_dims, num_heads, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embed_dims, num_heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU(inplace=True)

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos=None):
        """Add positional embedding to tensor."""
        return tensor if pos is None else tensor + pos

    def inner_forward(self, query, mask=None, key_padding_mask=None, query_pos=None):
        q = k = self.with_pos_embed(query, query_pos)
        tgt = self.self_attn(q, k, value=query, attn_mask=mask, key_padding_mask=key_padding_mask)[0]
        return query + self.dropout(tgt)

    def forward(self, query, mask=None, key_padding_mask=None, query_pos=None):
        if self.training and query.requires_grad:
            return cp(self.inner_forward, query, mask, key_padding_mask, query_pos, use_reentrant=False)
        return self.inner_forward(query, mask, key_padding_mask, query_pos)


class MaskFormerSampling(BaseModule):
    """Multi-view feature sampling module.

    Samples image features at 3D voxel locations projected to multiple camera views.
    Uses learnable offsets and scale weights for adaptive sampling.
    """

    def __init__(
            self,
            embed_dims=256,
            num_frames=4,
            num_views=6,
            num_groups=4,
            num_points=8,
            num_levels=4,
            pc_range=[],
            occ_size=[],
            init_cfg=None):
        super().__init__(init_cfg)

        self.num_frames = num_frames
        self.num_views = int(num_views)
        self.num_points = num_points
        self.num_groups = num_groups
        self.num_levels = num_levels
        self.pc_range = pc_range
        self.occ_size = occ_size

        # Learnable sampling offset: [B, Q, G*P, 3]
        self.offset = nn.Linear(embed_dims, num_groups * num_points * 3)
        # Learnable scale weights for multi-level feature fusion: [B, Q, G, P, L]
        self.scale_weights = nn.Linear(embed_dims, num_groups * num_points * num_levels)

    def init_weights(self, ):
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)

    def inner_forward(self, query_feat, valid_map, occ_loc, mlvl_feats, img_metas):
        """Sample multi-view image features at 3D voxel locations."""
        B, Q = query_feat.shape[:2]
        image_h, image_w, _ = img_metas[0]['img_shape']

        offset = self.offset(query_feat).view(B, Q, self.num_groups * self.num_points, 3)
        sampling_points = make_sample_points_from_mask(
            valid_map, self.pc_range, self.occ_size,
            self.num_groups * self.num_points, occ_loc, offset)
        sampling_points = sampling_points.reshape(B, Q, 1, self.num_groups, self.num_points, 3)
        sampling_points = sampling_points.expand(
            B, Q, self.num_frames, self.num_groups, self.num_points, 3)

        scale_weights = self.scale_weights(query_feat).view(
            B, Q, self.num_groups, 1, self.num_points, self.num_levels)
        scale_weights = torch.softmax(scale_weights, dim=-1)
        scale_weights = scale_weights.expand(
            B, Q, self.num_groups, self.num_frames, self.num_points, self.num_levels)

        return sampling_4d(
            sampling_points,
            mlvl_feats,
            scale_weights,
            img_metas[0]['occ2img'],
            image_h, image_w,
            num_views=self.num_views,
        )

    def forward(self, query_feat, valid_map, occ_loc, mlvl_feats, img_metas):
        if self.training and query_feat.requires_grad:
            return cp(
                self.inner_forward, query_feat, valid_map, occ_loc, mlvl_feats, img_metas,
                use_reentrant=False)
        return self.inner_forward(query_feat, valid_map, occ_loc, mlvl_feats, img_metas)
