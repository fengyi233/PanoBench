import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import Voxelization, knn
from mmdet.models.utils import multi_apply
from mmdet.registry import MODELS as MMDET_MODELS
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule

from utils.data_converter import decode_points
from utils.labels import get_dataset_meta


@MODELS.register_module()
class SparseWorldTCHead(BaseModule):
    """Sparse point head predicting occupancy for every forecast horizon.

    Queries, losses and decoding are identical to the single-frame OPUS head;
    the difference is that the batch axis carries an extra horizon-major factor
    ``FT``. Ground truth is stacked the same way by the data pipeline, so the
    Chamfer-style supervision applies per (sample, horizon) pair unchanged.
    """

    def __init__(self,
                 in_channels,
                 num_query,
                 dataset_type,
                 occ_variant,
                 future_frames=(0, 2, 4, 6),
                 transformer=None,
                 train_cfg=dict(),
                 test_cfg=None,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=2.0),
                 loss_pts=dict(type='L1Loss'),
                 init_cfg=None
                 ):
        super().__init__(init_cfg)
        meta = get_dataset_meta(dataset_type, occ_variant)
        num_classes = meta.num_classes
        pc_range = meta.point_cloud_range
        voxel_size = meta.voxel_size_3d
        free_id = meta.free_id
        if isinstance(transformer, dict):
            transformer = transformer.copy()
            transformer['dataset_type'] = dataset_type
            transformer['occ_variant'] = occ_variant

        self.dataset_type = dataset_type
        self.occ_variant = occ_variant
        self.future_frames = list(future_frames)
        self.num_horizons = len(self.future_frames)
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg if test_cfg is not None else dict()
        self.free_id = free_id
        self.loss_cls = MMDET_MODELS.build(loss_cls)
        self.loss_pts = MMDET_MODELS.build(loss_pts)
        self.transformer = MODELS.build(transformer)
        self.num_refines = self.transformer.num_refines
        self.embed_dims = self.transformer.embed_dims
        self.voxel_generator = Voxelization(
            voxel_size=voxel_size,
            point_cloud_range=pc_range,
            max_num_points=10,
            max_voxels=self.num_query * self.num_refines[-1],
            deterministic=False
        )

        # prepare scene
        pc_range = torch.tensor(pc_range)
        scene_size = pc_range[3:] - pc_range[:3]
        voxel_size = torch.tensor(voxel_size)
        voxel_num = (scene_size / voxel_size).long()
        self.register_buffer('pc_range', pc_range)
        self.register_buffer('scene_size', scene_size)
        self.register_buffer('voxel_size', voxel_size)
        self.register_buffer('voxel_num', voxel_num)

        self._init_layers()

    def _init_layers(self):
        self.init_points = nn.Embedding(self.num_query, 3)
        nn.init.uniform_(self.init_points.weight, 0, 1)

    def init_weights(self):
        self.transformer.init_weights()

    def forward(self, mlvl_feats, occ2img, image_shape, fut2cur):
        """
        Args:
            fut2cur: [B, FT, 4, 4] horizon occupancy -> key-frame occupancy.

        Returns:
            dict with ``all_cls_scores`` / ``all_refine_pts`` per decoder layer,
            each shaped ``[B * FT, Q, R, ...]`` in horizon-major order.
        """
        B, Q = mlvl_feats[0].shape[0], self.num_query
        init_points = self.init_points.weight[None, :, None, :].repeat(B, 1, 1, 1)
        query_feat = init_points.new_zeros(B, Q, self.embed_dims)

        cls_scores, refine_pts = self.transformer(
            init_points,
            query_feat,
            mlvl_feats,
            occ2img=occ2img,
            image_shape=image_shape,
            fut2cur=fut2cur,
            fut_list=self.future_frames,
        )

        return dict(init_points=init_points.repeat(self.num_horizons, 1, 1, 1),
                    all_cls_scores=cls_scores,
                    all_refine_pts=refine_pts)

    def get_dis_weight(self, pts):
        max_dist = torch.sqrt(
            self.scene_size[0] ** 2 + self.scene_size[1] ** 2)
        centers = (self.pc_range[:3] + self.pc_range[3:]) / 2
        dist = (pts - centers[None, ...])[..., :2]
        dist = torch.norm(dist, dim=-1)
        return dist / max_dist + 1

    @torch.no_grad()
    def _get_target_single(self, refine_pts, gt_points, gt_masks, gt_labels):
        # knn to apply Chamfer distance
        gt_paired_idx = knn(1, refine_pts[None, ...], gt_points[None, ...])
        gt_paired_idx = gt_paired_idx.permute(0, 2, 1).squeeze().long()
        pred_paired_idx = knn(1, gt_points[None, ...], refine_pts[None, ...])
        pred_paired_idx = pred_paired_idx.permute(0, 2, 1).squeeze().long()
        gt_paired_pts = refine_pts[gt_paired_idx]
        pred_paired_pts = gt_points[pred_paired_idx]

        # cls assignment
        refine_pts_labels = gt_labels[pred_paired_idx]
        cls_weights = self.train_cfg.get('cls_weights', [1] * self.num_classes)
        cls_weights = refine_pts.new_tensor(cls_weights)
        label_weights = cls_weights * \
                        self.get_dis_weight(pred_paired_pts)[..., None]

        # gt side assignment
        empty_dist_thr = self.train_cfg.get('empty_dist_thr', 0.2)
        empty_weights = self.train_cfg.get('empty_weights', 5)

        gt_pts_weights = refine_pts.new_ones(gt_paired_pts.shape[0])
        dist = torch.norm(gt_points - gt_paired_pts, dim=-1)
        mask = (dist > empty_dist_thr) & gt_masks
        gt_pts_weights[mask] = empty_weights

        rare_classes = self.train_cfg.get('rare_classes', None)
        if rare_classes:
            rare_weights = self.train_cfg.get('rare_weights', 10)
            for cls_idx in rare_classes:
                mask = (gt_labels == cls_idx) & gt_masks
                gt_pts_weights[mask] = gt_pts_weights[mask].clamp(min=rare_weights)

        return (refine_pts_labels, gt_paired_idx, pred_paired_idx, label_weights,
                gt_pts_weights)

    def loss_single(self,
                    cls_scores,
                    refine_pts,
                    gt_points_list,
                    gt_masks_list,
                    gt_labels_list):
        num_imgs = cls_scores.size(0)  # B * FT
        cls_scores = cls_scores.reshape(num_imgs, -1, self.num_classes)
        refine_pts = refine_pts.reshape(num_imgs, -1, 3)
        refine_pts = decode_points(refine_pts, self.pc_range)
        cls_scores_list = [cls_scores[i] for i in range(num_imgs)]
        refine_pts_list = [refine_pts[i] for i in range(num_imgs)]

        (labels_list, gt_paired_idx_list, pred_paired_idx_list, cls_weights,
         gt_pts_weights) = multi_apply(
            self._get_target_single, refine_pts_list, gt_points_list,
            gt_masks_list, gt_labels_list)

        gt_paired_pts, pred_paired_pts = [], []
        for i in range(num_imgs):
            gt_paired_pts.append(refine_pts_list[i][gt_paired_idx_list[i]])
            pred_paired_pts.append(gt_points_list[i][pred_paired_idx_list[i]])

        # concatenate all results from different samples
        cls_scores = torch.cat(cls_scores_list)
        labels = torch.cat(labels_list)
        cls_weights = torch.cat(cls_weights)
        gt_pts = torch.cat(gt_points_list)
        gt_paired_pts = torch.cat(gt_paired_pts)
        gt_pts_weights = torch.cat(gt_pts_weights)
        pred_pts = torch.cat(refine_pts_list)
        pred_paired_pts = torch.cat(pred_paired_pts)

        loss_cls = self.loss_cls(cls_scores,
                                 labels,
                                 weight=cls_weights,
                                 avg_factor=cls_scores.shape[0])
        loss_pts = pred_pts.new_tensor(0)
        loss_pts += self.loss_pts(gt_pts,
                                  gt_paired_pts,
                                  weight=gt_pts_weights[..., None],
                                  avg_factor=gt_pts.shape[0])
        loss_pts += self.loss_pts(pred_pts,
                                  pred_paired_pts,
                                  avg_factor=pred_pts.shape[0])

        return loss_cls, loss_pts

    def loss(self, voxel_semantics, valid_mask, preds_dicts):
        """
        Args:
            voxel_semantics: [B, FT, X, Y, Z], free voxels use ``self.free_id``.
            valid_mask: [B, FT, X, Y, Z].
        """
        init_points = preds_dicts['init_points']
        all_cls_scores = preds_dicts['all_cls_scores']
        all_refine_pts = preds_dicts['all_refine_pts']

        voxel_semantics = self._to_horizon_major(voxel_semantics)
        valid_mask = self._to_horizon_major(valid_mask)

        num_dec_layers = len(all_cls_scores)
        gt_points_list, gt_masks_list, gt_labels_list = \
            self.get_sparse_voxels(voxel_semantics, valid_mask)
        all_gt_points_list = [gt_points_list for _ in range(num_dec_layers)]
        all_gt_masks_list = [gt_masks_list for _ in range(num_dec_layers)]
        all_gt_labels_list = [gt_labels_list for _ in range(num_dec_layers)]

        losses_cls, losses_pts = multi_apply(
            self.loss_single, all_cls_scores, all_refine_pts,
            all_gt_points_list, all_gt_masks_list, all_gt_labels_list)

        loss_dict = dict()
        if init_points is not None:
            pseudo_scores = init_points.new_zeros(
                *init_points.shape[:-1], self.num_classes)
            _, init_loss_pts = self.loss_single(
                pseudo_scores, init_points, gt_points_list,
                gt_masks_list, gt_labels_list)
            loss_dict['init_loss_pts'] = init_loss_pts

        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_pts'] = losses_pts[-1]

        for i, (loss_cls_i, loss_pts_i) in enumerate(
                zip(losses_cls[:-1], losses_pts[:-1])):
            loss_dict[f'd{i}.loss_cls'] = loss_cls_i
            loss_dict[f'd{i}.loss_pts'] = loss_pts_i
        return loss_dict

    @staticmethod
    def _to_horizon_major(tensor):
        """[B, FT, X, Y, Z] -> [FT * B, X, Y, Z] matching the decoder layout."""
        if tensor.dim() != 5:
            raise ValueError(
                f'expected [B, FT, X, Y, Z] ground truth, got {tuple(tensor.shape)}')
        return tensor.transpose(0, 1).reshape(-1, *tensor.shape[2:])

    def get_occ(self, pred_dicts):
        """Decode the last decoder layer into per-sample, per-horizon voxels.

        Returns:
            List over samples; each entry is a list over horizons of
            ``dict(sem_pred, occ_loc)``.
        """
        cls_scores = pred_dicts['all_cls_scores'][-1].sigmoid()
        refine_pts = pred_dicts['all_refine_pts'][-1]

        num_rows = refine_pts.shape[0]
        batch_size = num_rows // self.num_horizons
        results = [[None] * self.num_horizons for _ in range(batch_size)]
        for row in range(num_rows):
            horizon, sample_idx = divmod(row, batch_size)
            results[sample_idx][horizon] = self._decode_single(
                refine_pts[row], cls_scores[row])
        return results

    def _decode_single(self, refine_pts, cls_scores):
        ctr_dist_thr = self.test_cfg.get('ctr_dist_thr', 3.)
        score_thr = self.test_cfg.get('score_thr', 0.)

        refine_pts = decode_points(refine_pts, self.pc_range)

        # filter weak points by distance and score
        centers = refine_pts.mean(dim=1, keepdim=True)
        ctr_dists = torch.norm(refine_pts - centers, dim=-1)
        mask = (ctr_dists < ctr_dist_thr) & (cls_scores > score_thr).any(dim=-1)
        refine_pts = refine_pts[mask]
        cls_scores = cls_scores[mask]

        pts = torch.cat([refine_pts, cls_scores], dim=-1)
        pts_infos, voxels, num_pts = self.voxel_generator(pts)
        voxels = torch.flip(voxels, [1]).long()
        scores = pts_infos[..., 3:]
        scores = scores.sum(dim=1) / num_pts[..., None]

        if self.test_cfg.get('padding', True):
            occ = scores.new_zeros((self.voxel_num[0], self.voxel_num[1],
                                    self.voxel_num[2], self.num_classes))
            occ[voxels[:, 0], voxels[:, 1], voxels[:, 2]] = scores
            occ = occ.permute(3, 0, 1, 2).unsqueeze(0)
            dilated_occ = F.max_pool3d(occ, 3, stride=1, padding=1)
            eroded_occ = -F.max_pool3d(-dilated_occ, 3, stride=1, padding=1)
            original_mask = (occ > score_thr).any(dim=1, keepdim=True)
            original_mask = original_mask.expand_as(eroded_occ)
            eroded_occ[original_mask] = occ[original_mask]
            eroded_occ = eroded_occ.squeeze(0).permute(1, 2, 3, 0)
            voxels = torch.nonzero((eroded_occ > score_thr).any(dim=-1))
            scores = eroded_occ[voxels[:, 0], voxels[:, 1], voxels[:, 2], :]

        labels = scores.argmax(dim=-1)
        return dict(sem_pred=labels.detach().cpu().numpy(),
                    occ_loc=voxels.detach().cpu().numpy())

    def get_sparse_voxels(self, voxel_semantics, valid_mask):
        B, W, H, Z = voxel_semantics.shape
        device = voxel_semantics.device
        voxel_semantics = voxel_semantics.long()

        x = torch.arange(0, W, dtype=torch.float32, device=device)
        x = (x + 0.5) / W * self.scene_size[0] + self.pc_range[0]
        y = torch.arange(0, H, dtype=torch.float32, device=device)
        y = (y + 0.5) / H * self.scene_size[1] + self.pc_range[1]
        z = torch.arange(0, Z, dtype=torch.float32, device=device)
        z = (z + 0.5) / Z * self.scene_size[2] + self.pc_range[2]

        xx = x[:, None, None].expand(W, H, Z)
        yy = y[None, :, None].expand(W, H, Z)
        zz = z[None, None, :].expand(W, H, Z)
        coors = torch.stack([xx, yy, zz], dim=-1)  # actual space

        # Cap GT points for Chamfer/knn. Fine voxels can exceed ~4e5 occupied
        # cells; mmcv knn then hits illegal memory access.
        max_gt_points = self.train_cfg.get('max_gt_points', None)

        gt_points, gt_masks, gt_labels = [], [], []
        for i in range(B):
            mask = voxel_semantics[i] != self.free_id
            pts = coors[mask]
            masks_i = valid_mask[i][mask]
            labels_i = voxel_semantics[i][mask]
            if max_gt_points is not None and pts.shape[0] > max_gt_points:
                idx = torch.randperm(pts.shape[0], device=device)[:max_gt_points]
                pts = pts[idx]
                masks_i = masks_i[idx]
                labels_i = labels_i[idx]
            gt_points.append(pts)
            gt_masks.append(masks_i)
            gt_labels.append(labels_i)

        return gt_points, gt_masks, gt_labels
