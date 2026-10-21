import math

import torch
import torch.nn as nn
from mmcv.cnn.bricks.transformer import FFN, MultiheadAttention
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule, ModuleList, bias_init_with_prob

from models.layers import MLN, AdaptiveMixing, SelfAttention, SparseWorldTCSampling
from utils.data_converter import decode_points, encode_points
from utils.labels import get_dataset_meta


@MODELS.register_module()
class SparseWorldTCTransformer(BaseModule):
    """Trajectory-conditioned sparse decoder for 4D occupancy forecasting.

    One shared set of queries is replicated over the forecast horizons. Each
    replica is tagged with a sinusoidal time embedding and modulated by the
    flattened future-to-current ego transform, so a single decoder predicts all
    horizons from the same history image features.
    """

    def __init__(self,
                 embed_dims,
                 dataset_type,
                 occ_variant,
                 num_frames=8,
                 num_views=6,
                 num_points=4,
                 num_layers=6,
                 num_levels=4,
                 num_groups=4,
                 num_refines=[1, 2, 4, 8, 16, 32],
                 scales=[1.0],
                 use_trajectory=True,
                 init_cfg=None):
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)

        meta = get_dataset_meta(dataset_type, occ_variant)

        self.embed_dims = embed_dims
        self.dataset_type = dataset_type
        self.occ_variant = occ_variant
        self.pc_range = meta.point_cloud_range
        self.num_refines = num_refines
        self.use_trajectory = bool(use_trajectory)

        self.decoder = SparseWorldTCTransformerDecoder(
            embed_dims, num_frames, num_views, num_points, num_layers,
            num_levels, meta.num_classes, num_refines, num_groups, scales,
            pc_range=self.pc_range, use_trajectory=self.use_trajectory)

    @torch.no_grad()
    def init_weights(self):
        self.decoder.init_weights()

    def forward(self, query_points, query_feat, mlvl_feats, occ2img,
                image_shape, fut2cur, fut_list):
        cls_scores, refine_pts = self.decoder(
            query_points, query_feat, mlvl_feats, occ2img, image_shape,
            fut2cur, fut_list)

        cls_scores = [torch.nan_to_num(score) for score in cls_scores]
        refine_pts = [torch.nan_to_num(pts) for pts in refine_pts]

        return cls_scores, refine_pts


class SparseWorldTCTransformerDecoder(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_frames=8,
                 num_views=6,
                 num_points=4,
                 num_layers=6,
                 num_levels=4,
                 num_classes=10,
                 num_refines=16,
                 num_groups=4,
                 scales=[1.0],
                 pc_range=[],
                 use_trajectory=True,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.num_layers = num_layers
        self.pc_range = pc_range
        self.num_frames = num_frames
        self.num_views = num_views
        self.num_groups = num_groups
        self.embed_dims = embed_dims
        self.use_trajectory = bool(use_trajectory)

        if len(scales) == 1:
            scales = scales * num_layers
        if not isinstance(num_refines, list):
            num_refines = [num_refines]
        if len(num_refines) == 1:
            num_refines = num_refines * num_layers
        last_refines = [1] + num_refines

        self.decoder_layers = ModuleList()
        for i in range(num_layers):
            self.decoder_layers.append(
                SparseWorldTCTransformerDecoderLayer(
                    embed_dims, num_frames, num_views, num_points, num_levels,
                    num_classes, num_groups, num_refines[i], last_refines[i],
                    layer_idx=i, scale=scales[i], pc_range=pc_range)
            )

        # Trajectory-conditioned SparseWorld-TC only: modulate queries with
        # the flattened 4x4 future->current transform. Sampling still uses
        # ``fut2cur`` either way, because query points live in a future frame.
        self.pe_mln = MLN(16, f_dim=embed_dims) if self.use_trajectory else None

    @torch.no_grad()
    def init_weights(self):
        self.decoder_layers.init_weights()

    def time_embedding(self, fut_list, device):
        """Sinusoidal embedding of the horizon offsets. Returns ``[FT, C]``."""
        frames = torch.as_tensor(fut_list, dtype=torch.float32, device=device)
        frames = frames.reshape(-1)

        half_dim = self.embed_dims // 2
        scale = math.log(10000) / (half_dim - 1)
        freqs = torch.exp(
            torch.arange(half_dim, dtype=torch.float32, device=device) * -scale)
        embeddings = frames[:, None] * freqs[None, :]
        embeddings = torch.cat(
            [torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        if self.embed_dims % 2 == 1:
            pad = torch.zeros(frames.shape[0], 1, device=device)
            embeddings = torch.cat([embeddings, pad], dim=-1)
        return embeddings

    def forward(self, query_points, query_feat, mlvl_feats, occ2img,
                image_shape, fut2cur, fut_list):
        """
        Symbol meaning:
            B: batch size
            FT: num of forecast horizons
            Q: num of queries
            T: num of history frames
            G: num of groups (group sampling mechanism of AdaMixer)
            P: num of sampling points per frame per group
            N: num of views
            L: num of feature pyramid levels

        Args:
            query_points: [B, Q, 1, 3] normalized query anchors.
            query_feat: [B, Q, C].
            occ2img: [B, T*N, 4, 4] key-frame occupancy -> image.
            fut2cur: [B, FT, 4, 4] horizon occupancy -> key-frame occupancy.
            fut_list: Horizon offsets in key-frame steps, length FT.
        """
        cls_scores, refine_pts = [], []
        B, Q, C = query_feat.shape
        FT = fut2cur.shape[1]

        # group image features in advance for sampling, see `sampling_4d`
        for lvl, feat in enumerate(mlvl_feats):
            B_feat, TN, GC, H, W = feat.shape
            N, T, G = self.num_views, self.num_frames, self.num_groups
            C_g = GC // G
            assert T * N == TN
            feat = feat.reshape(B_feat, T, N, G, C_g, H, W)
            feat = feat.permute(0, 1, 3, 2, 5, 6, 4)  # [B, T, G, N, H, W, C]
            mlvl_feats[lvl] = feat.reshape(B_feat * T * G, N, H, W, C_g).contiguous()

        # Horizon-major expansion: index b of the expanded batch is h * B + b0,
        # matching the chunking that `sampling_4d` applies to the pyramid.
        time_embed = self.time_embedding(fut_list, query_feat.device)
        time_embed = time_embed.repeat_interleave(B, dim=0)[:, None, :]

        query_points = query_points.repeat(FT, 1, 1, 1)
        query_feat = query_feat.repeat(FT, 1, 1) + time_embed
        occ2img = occ2img.repeat(FT, 1, 1, 1)

        fut2cur = fut2cur.transpose(0, 1).reshape(B * FT, 4, 4)
        if self.pe_mln is not None:
            pos_mat = fut2cur.reshape(B * FT, 1, 16).expand(-1, Q, -1)
            query_feat = self.pe_mln(query_feat, pos_mat)

        for decoder_layer in self.decoder_layers:
            query_points = query_points.detach()
            query_feat, cls_score, query_points = decoder_layer(
                query_points, query_feat, mlvl_feats, occ2img, image_shape,
                fut2cur, FT)

            cls_scores.append(cls_score)
            refine_pts.append(query_points)

        return cls_scores, refine_pts


class SparseWorldTCTransformerDecoderLayer(BaseModule):
    def __init__(self,
                 embed_dims,
                 num_frames=8,
                 num_views=6,
                 num_points=4,
                 num_levels=4,
                 num_classes=10,
                 num_groups=4,
                 num_refines=16,
                 last_refines=16,
                 num_cls_fcs=2,
                 num_reg_fcs=2,
                 layer_idx=0,
                 scale=1.0,
                 pc_range=[],
                 init_cfg=None):
        super().__init__(init_cfg)

        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.pc_range = pc_range
        self.num_points = num_points
        self.num_refines = num_refines
        self.last_refines = last_refines
        self.layer_idx = layer_idx
        self.scale = scale

        self.position_encoder = nn.Sequential(
            nn.Linear(3 * self.last_refines, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims),
            nn.ReLU(inplace=True),
        )

        self.self_attn = SelfAttention(
            embed_dims, num_heads=8, dropout=0.1, pc_range=pc_range)
        self.sampling = SparseWorldTCSampling(
            embed_dims, num_frames=num_frames, num_views=num_views,
            num_groups=num_groups, num_points=num_points,
            num_levels=num_levels, pc_range=pc_range)
        self.mixing = AdaptiveMixing(
            in_dim=embed_dims, in_points=num_points * num_frames,
            n_groups=num_groups, out_points=32)
        self.ffn = FFN(embed_dims, feedforward_channels=512, ffn_drop=0.1)

        self.norm1 = nn.LayerNorm(embed_dims)
        self.norm2 = nn.LayerNorm(embed_dims)
        self.norm3 = nn.LayerNorm(embed_dims)

        # Lets each query exchange information across forecast horizons.
        self.tempo_attn = MultiheadAttention(
            embed_dims, num_heads=8, dropout=0.1, batch_first=True)
        self.norm_tempo = nn.LayerNorm(embed_dims)

        cls_branch = []
        for _ in range(num_cls_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(
            self.embed_dims, self.num_classes * self.num_refines))
        self.cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU(inplace=True))
        reg_branch.append(nn.Linear(self.embed_dims, 3 * self.num_refines))
        self.reg_branch = nn.Sequential(*reg_branch)

    @torch.no_grad()
    def init_weights(self):
        self.self_attn.init_weights()
        self.sampling.init_weights()
        self.mixing.init_weights()

        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls_branch[-1].bias, bias_init)

    def refine_points(self, points_proposal, points_delta):
        B, Q = points_delta.shape[:2]
        points_delta = points_delta.reshape(B, Q, self.num_refines, 3)

        points_proposal = decode_points(points_proposal, self.pc_range)
        points_proposal = points_proposal.mean(dim=2, keepdim=True)
        new_points = points_proposal + points_delta
        return encode_points(new_points, self.pc_range)

    def forward(self, query_points, query_feat, mlvl_feats, occ2img,
                image_shape, fut2cur, num_horizons):
        query_pos = self.position_encoder(query_points.flatten(2, 3))
        query_feat = query_feat + query_pos

        sampled_feat = self.sampling(
            query_points, query_feat, mlvl_feats, occ2img, image_shape,
            fut2cur, num_horizons)
        query_feat = self.norm1(self.mixing(sampled_feat, query_feat))

        # Temporal attention runs over the horizon axis, so fold the
        # horizon-major batch back into [B, FT * Q, C] before attending.
        BFT, Q, C = query_feat.shape
        B = BFT // num_horizons
        query_feat = query_feat.reshape(num_horizons, B, Q, C)
        query_feat = query_feat.permute(1, 0, 2, 3).reshape(B, num_horizons * Q, C)
        query_feat = self.norm_tempo(self.ffn(self.tempo_attn(query_feat)))
        query_feat = query_feat.reshape(B, num_horizons, Q, C)
        query_feat = query_feat.permute(1, 0, 2, 3).reshape(BFT, Q, C)

        query_feat = self.norm2(
            self.self_attn(query_points, query_feat, query_mode='points'))
        query_feat = self.norm3(self.ffn(query_feat))

        B, Q = query_points.shape[:2]
        cls_score = self.cls_branch(query_feat)
        cls_score = cls_score.reshape(B, Q, self.num_refines, self.num_classes)
        reg_offset = self.scale * self.reg_branch(query_feat)
        refine_pt = self.refine_points(query_points, reg_offset)

        return query_feat, cls_score, refine_pt
