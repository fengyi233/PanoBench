import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule

from utils.labels import get_dataset_meta


@MODELS.register_module()
class FlashOCCPanopticHead(BaseModule):
    def __init__(self,
                 dataset_type,
                 occ_variant,
                 in_dim=256,
                 hidden_dim=128,
                 center_loss_weight=1.0,
                 offset_loss_weight=1.0,
                 center_threshold=0.1,
                 max_centers_per_class=100,
                 assign_chunk_size=65536,
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        meta = get_dataset_meta(dataset_type, occ_variant)
        self.volume_size = tuple(int(v) for v in meta.volume_size)
        self.free_id = int(meta.free_id)
        self.thing_ids = tuple(sorted(meta.thing_ids))
        self.stuff_ids = tuple(
            cls_id
            for cls_id in range(meta.num_classes)
            if cls_id not in meta.thing_ids and cls_id != self.free_id
        )
        self.num_things = len(self.thing_ids)
        self.thing_to_index = {int(cls_id): idx for idx, cls_id in enumerate(self.thing_ids)}
        self.center_loss_weight = float(center_loss_weight)
        self.offset_loss_weight = float(offset_loss_weight)
        self.center_threshold = float(center_threshold)
        self.max_centers_per_class = int(max_centers_per_class)
        self.assign_chunk_size = int(assign_chunk_size)

        self.shared = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.center_head = nn.Conv2d(hidden_dim, self.num_things, 1)
        self.offset_head = nn.Conv2d(hidden_dim, self.volume_size[2] * 3, 1)

    def forward(self, bev_feat):
        feat = self.shared(bev_feat)
        center_logits = self.center_head(feat)
        offset = self.offset_head(feat)
        b, _, y, x = offset.shape
        offset = offset.permute(0, 3, 2, 1).contiguous().view(b, x, y, self.volume_size[2], 3)
        return dict(center_logits=center_logits, offset=offset)

    def _make_targets(self, voxel_instances, instance_class_ids, device):
        b, x_size, y_size, z_size = voxel_instances.shape
        center_target = torch.zeros((b, self.num_things, y_size, x_size), device=device)
        offset_target = torch.zeros((b, x_size, y_size, z_size, 3), device=device)
        thing_mask = torch.zeros((b, x_size, y_size, z_size), dtype=torch.bool, device=device)

        for bidx in range(b):
            class_ids = instance_class_ids[bidx].to(device=device).long()
            for inst_id in range(int(class_ids.numel())):
                cls_id = int(class_ids[inst_id].item())
                thing_idx = self.thing_to_index.get(cls_id)
                if thing_idx is None:
                    continue
                coords = torch.nonzero(voxel_instances[bidx] == inst_id, as_tuple=False)
                if coords.numel() == 0:
                    continue
                center = coords.float().mean(dim=0)
                cx = int(center[0].round().clamp(0, x_size - 1).item())
                cy = int(center[1].round().clamp(0, y_size - 1).item())
                center_target[bidx, thing_idx, cy, cx] = 1.0
                offset_target[bidx, coords[:, 0], coords[:, 1], coords[:, 2]] = center.view(1, 3) - coords.float()
                thing_mask[bidx, coords[:, 0], coords[:, 1], coords[:, 2]] = True
        return center_target, offset_target, thing_mask

    def loss(self, preds, voxel_instances=None, instance_class_ids=None):
        if voxel_instances is None or instance_class_ids is None:
            zero = preds['center_logits'].sum() * 0.0 + preds['offset'].sum() * 0.0
            return dict(loss_pano_center=zero, loss_pano_offset=zero)

        voxel_instances = voxel_instances.long().to(preds['center_logits'].device)
        center_target, offset_target, thing_mask = self._make_targets(
            voxel_instances, instance_class_ids, preds['center_logits'].device)
        center_loss = F.binary_cross_entropy_with_logits(
            preds['center_logits'].float(), center_target.float()) * self.center_loss_weight
        if thing_mask.any():
            offset_loss = F.l1_loss(
                preds['offset'].float()[thing_mask], offset_target.float()[thing_mask]) * self.offset_loss_weight
        else:
            offset_loss = preds['offset'].sum() * 0.0
        return dict(loss_pano_center=center_loss, loss_pano_offset=offset_loss)

    def _centers_from_logits(self, center_logits):
        centers = []
        center_prob = center_logits.sigmoid()
        for thing_idx, cls_id in enumerate(self.thing_ids):
            prob = center_prob[thing_idx]
            keep = prob > self.center_threshold
            coords_yx = torch.nonzero(keep, as_tuple=False)
            if coords_yx.numel() == 0:
                topk = min(1, prob.numel())
                scores, flat = prob.flatten().topk(topk)
                coords_yx = torch.stack((flat // prob.shape[1], flat % prob.shape[1]), dim=1)
            else:
                scores = prob[coords_yx[:, 0], coords_yx[:, 1]]
                if scores.numel() > self.max_centers_per_class:
                    scores, order = scores.topk(self.max_centers_per_class)
                    coords_yx = coords_yx[order]
            for cyx in coords_yx:
                y, x = cyx.long()
                centers.append((int(cls_id), torch.stack((x, y)).float().to(center_logits.device)))
        return centers

    @torch.no_grad()
    def predict(self, preds, dense_sem):
        b = dense_sem.shape[0]
        pano_sem = dense_sem.int()
        pano_inst = torch.zeros_like(pano_sem, dtype=torch.int32)
        thing_ids = set(int(v) for v in self.thing_ids)
        stuff_ids = set(int(v) for v in self.stuff_ids)

        for bidx in range(b):
            for cls_id in stuff_ids:
                pano_inst[bidx][dense_sem[bidx] == cls_id] = 1

            centers = self._centers_from_logits(preds['center_logits'][bidx])
            instance_id = 2
            centers_by_class = {}
            center_ids_by_class = {}
            for cls_id, xy in centers:
                centers_by_class.setdefault(cls_id, []).append(xy)
                center_ids_by_class.setdefault(cls_id, []).append(instance_id)
                instance_id += 1

            for cls_id in thing_ids:
                mask = dense_sem[bidx] == cls_id
                if not mask.any():
                    continue
                cls_centers = centers_by_class.get(cls_id)
                if not cls_centers:
                    pano_inst[bidx][mask] = instance_id
                    instance_id += 1
                    continue
                center_xy = torch.stack(cls_centers, dim=0)
                center_ids = torch.tensor(center_ids_by_class[cls_id], device=dense_sem.device, dtype=torch.int32)
                coords = torch.nonzero(mask, as_tuple=False)
                projected = coords[:, :2].float() + preds['offset'][bidx][mask][:, :2]
                assigned = []
                for start in range(0, projected.shape[0], self.assign_chunk_size):
                    chunk = projected[start:start + self.assign_chunk_size]
                    dist = ((chunk[:, None, :] - center_xy[None]) ** 2).sum(dim=-1)
                    assigned.append(center_ids[dist.argmin(dim=1)])
                pano_inst[bidx, coords[:, 0], coords[:, 1], coords[:, 2]] = torch.cat(assigned, dim=0)
        return pano_sem, pano_inst
