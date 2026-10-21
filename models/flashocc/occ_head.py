import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule

from models.sparseocc.loss_utils import GeoScalLoss, SemScalLoss
from utils.data_converter import occ_dense2sparse
from utils.labels import get_dataset_meta


@MODELS.register_module()
class FlashOCCHead2D(BaseModule):
    def __init__(self,
                 dataset_type,
                 occ_variant,
                 in_dim=256,
                 out_dim=256,
                 use_predicter=True,
                 use_mask=True,
                 cls_weights=None,
                 loss_weight=1.0,
                 loss_geo_weight=1.0,
                 loss_sem_weight=1.0,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        meta = get_dataset_meta(dataset_type, occ_variant)
        self.volume_size = tuple(int(v) for v in meta.volume_size)
        self.num_classes = int(meta.num_classes)
        self.free_id = int(meta.free_id)
        self.Dz = self.volume_size[2]
        self.use_predicter = bool(use_predicter)
        self.use_mask = bool(use_mask)
        self.loss_weight = float(loss_weight)
        self.cls_weights = cls_weights

        self.final_conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        if self.use_predicter:
            self.predicter = nn.Sequential(
                nn.Linear(out_dim, out_dim * 2),
                nn.Softplus(),
                nn.Linear(out_dim * 2, self.num_classes * self.Dz),
            )
        else:
            self.classifier = nn.Conv2d(out_dim, self.num_classes * self.Dz, 1)
        self.geo_loss = GeoScalLoss(self.num_classes, free_id=self.free_id, loss_weight=loss_geo_weight)
        self.sem_loss = SemScalLoss(self.num_classes, class_weights=cls_weights, loss_weight=loss_sem_weight)

    def forward(self, img_feats):
        feat = self.final_conv(img_feats)
        if self.use_predicter:
            pred = feat.permute(0, 3, 2, 1).contiguous()
            b, x, y = pred.shape[:3]
            pred = self.predicter(pred).view(b, x, y, self.Dz, self.num_classes)
        else:
            pred = self.classifier(feat).permute(0, 3, 2, 1).contiguous()
            b, x, y = pred.shape[:3]
            pred = pred.view(b, x, y, self.Dz, self.num_classes)
        return pred

    def loss(self, occ_pred, voxel_semantics, valid_mask=None):
        voxel_semantics = voxel_semantics.long()
        pred = occ_pred.permute(0, 4, 1, 2, 3).contiguous()
        target_raw = voxel_semantics
        target = target_raw
        if valid_mask is not None and self.use_mask:
            target = target.masked_fill(~valid_mask.bool(), 255)
        class_weight = None
        if self.cls_weights is not None:
            class_weight = pred.new_tensor(self.cls_weights)
        loss_occ = F.cross_entropy(pred, target, weight=class_weight, ignore_index=255) * self.loss_weight

        flat_pred = pred.flatten(2)
        flat_target = target_raw.flatten(1)
        losses = dict(loss_occ=loss_occ)
        with torch.cuda.amp.autocast(enabled=False):
            flat_pred_fp32 = flat_pred.float()
            losses['loss_voxel_geo_scal'] = self.geo_loss(flat_pred_fp32, flat_target)
            losses['loss_voxel_sem_scal'] = self.sem_loss(flat_pred_fp32, flat_target)
        return losses

    def get_occ(self, occ_pred):
        occ = occ_pred.argmax(dim=-1)
        results = []
        for dense in occ:
            sparse = occ_dense2sparse(dense, free_id=self.free_id)
            results.append(dict(
                occ_loc=sparse[:, :3].detach().cpu().numpy(),
                sem_pred=sparse[:, 3].detach().cpu().numpy(),
            ))
        return results
