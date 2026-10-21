import torch
from mmdet3d.registry import MODELS
from mmengine.model import BaseModule

from utils.labels import get_dataset_meta
from .loss_utils import CE_ssc_loss, lovasz_softmax, get_voxel_decoder_loss_input
from .matcher import HungarianMatcher


@MODELS.register_module()
class SparseOccHead(BaseModule):
    def __init__(self,
                 transformer=None,
                 class_names=None,
                 embed_dims=None,
                 occ_size=None,
                 pc_range=None,
                 loss_cfgs=None,
                 panoptic=False,
                 dataset_type=None,
                 occ_variant=None,
                 free_id=None,
                 class_weights=None,
                 score_threshold=0.3,
                 free_score_threshold=0.01,
                 use_mask=False,
                 **kwargs):
        super().__init__()
        meta = get_dataset_meta(dataset_type, occ_variant)
        class_names = meta.class_names if class_names is None else class_names
        occ_size = meta.volume_size if occ_size is None else occ_size
        pc_range = meta.point_cloud_range if pc_range is None else pc_range
        free_id = meta.free_id if free_id is None else free_id
        voxel_size = meta.voxel_size

        self.num_classes = len(class_names)
        self.class_names = class_names
        self.pc_range = tuple(pc_range)
        self.occ_size = tuple(occ_size)
        self.embed_dims = embed_dims
        self.free_id = self.num_classes - 1 if free_id is None else int(free_id)
        self.score_threshold = float(score_threshold)
        self.free_score_threshold = float(free_score_threshold)
        self.use_mask = bool(use_mask)
        self.overlap_threshold = 0.8
        self.panoptic = panoptic
        self.instance_ignore_id = 255
        non_free_ids = [idx for idx in range(self.num_classes) if idx != self.free_id]
        semantic_to_query = torch.full((self.num_classes,), -1, dtype=torch.long)
        semantic_to_query[torch.tensor(non_free_ids, dtype=torch.long)] = torch.arange(
            len(non_free_ids), dtype=torch.long)
        self.register_buffer(
            'semantic_to_query',
            semantic_to_query,
            persistent=False,
        )
        self.register_buffer(
            'query_to_semantic',
            torch.tensor(non_free_ids, dtype=torch.long),
            persistent=False,
        )
        self.mask_num_classes = len(non_free_ids) + 1
        self.no_object_id = len(non_free_ids)
        thing_ids = meta.thing_ids
        self.register_buffer(
            'thing_class_mask',
            torch.tensor(
                [idx in thing_ids for idx in range(self.num_classes)],
                dtype=torch.bool,
            ),
            persistent=False,
        )

        transformer = transformer.copy()
        transformer.setdefault('num_classes', self.num_classes)
        transformer.setdefault('pc_range', self.pc_range)
        transformer.setdefault('occ_size', self.occ_size)
        transformer.setdefault('free_id', self.free_id)
        transformer.setdefault('voxel_size', voxel_size)
        self.with_mask_decoder = bool(transformer.get('with_mask_decoder', True))
        if class_weights is None:
            class_weights_tensor = torch.ones(self.num_classes, dtype=torch.float32)
        else:
            class_weights_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
            if class_weights_tensor.numel() != self.num_classes:
                raise ValueError(
                    f"class_weights length {class_weights_tensor.numel()} does not "
                    f"match num_classes={self.num_classes}."
                )
        self.register_buffer('class_weights', class_weights_tensor, persistent=False)
        self.transformer = MODELS.build(transformer)
        self.criterions = {
            k: MODELS.build(self._prepare_loss_cfg(k, loss_cfg))
            for k, loss_cfg in (loss_cfgs or {}).items()
        }
        self.matcher = HungarianMatcher(cost_class=2.0, cost_mask=5.0, cost_dice=5.0)

    def _prepare_loss_cfg(self, loss_name, loss_cfg):
        loss_cfg = loss_cfg.copy()
        if loss_name in {'loss_geo_scal', 'loss_sem_scal'}:
            loss_cfg.setdefault('num_classes', self.num_classes)
        if loss_name == 'loss_mask2former':
            loss_cfg.setdefault('num_classes', self.mask_num_classes)
        if loss_name == 'loss_geo_scal':
            loss_cfg.setdefault('free_id', self.free_id)
        return loss_cfg

    def init_weights(self):
        self.transformer.init_weights()

    def forward(self, mlvl_feats, img_metas):
        occ_preds, mask_preds, class_preds = self.transformer(mlvl_feats, img_metas=img_metas)  # SparseOccTransformer

        return {
            'occ_preds': occ_preds,
            'mask_preds': mask_preds,
            'class_preds': class_preds
        }

    def loss(self, voxel_semantics, valid_mask, voxel_instances, instance_class_ids, preds_dicts):
        loss_dict = {}
        B = voxel_semantics.shape[0]
        if not self.use_mask:
            valid_mask = None

        for i, (occ_loc_i, _, seg_pred_i, _, scale) in enumerate(preds_dicts['occ_preds']):
            loss_dict_i = {}
            for b in range(B):
                loss_dict_i_b = {}
                seg_pred_i_sparse, voxel_semantics_sparse, sparse_mask = get_voxel_decoder_loss_input(
                    voxel_semantics[b:b + 1],
                    valid_mask[b:b + 1] if valid_mask is not None else None,
                    occ_loc_i[b:b + 1],
                    seg_pred_i[b:b + 1] if seg_pred_i is not None else None,
                    scale,
                    self.num_classes,
                    occ_size=self.occ_size
                )

                valid_sem_mask = (voxel_semantics_sparse < 255)
                if not valid_sem_mask.any():
                    zero = seg_pred_i_sparse.sum() * 0.0
                    loss_dict_i_b['loss_sem_lovasz'] = zero
                    if 'loss_geo_scal' in self.criterions.keys():
                        loss_dict_i_b['loss_geo_scal'] = zero
                    if 'loss_sem_scal' in self.criterions.keys():
                        loss_dict_i_b['loss_sem_scal'] = zero
                    loss_dict_i_b['loss_sem_ce'] = zero
                    for loss_key in loss_dict_i_b.keys():
                        loss_dict_i[loss_key] = loss_dict_i.get(loss_key, 0) + loss_dict_i_b[loss_key] / B
                    continue

                loss_dict_i_b['loss_sem_lovasz'] = lovasz_softmax(
                    torch.softmax(seg_pred_i_sparse, dim=1),
                    voxel_semantics_sparse,
                    ignore=255,
                )

                seg_pred_i_sparse = seg_pred_i_sparse[valid_sem_mask].transpose(0, 1).unsqueeze(
                    0)  # [K, CLS] -> [B, CLS, K]
                voxel_semantics_sparse = voxel_semantics_sparse[valid_sem_mask].unsqueeze(0)  # [K] -> [B, K]

                if 'loss_geo_scal' in self.criterions.keys():
                    loss_dict_i_b['loss_geo_scal'] = self.criterions['loss_geo_scal'](seg_pred_i_sparse,
                                                                                      voxel_semantics_sparse)
                if 'loss_sem_scal' in self.criterions.keys():
                    loss_dict_i_b['loss_sem_scal'] = self.criterions['loss_sem_scal'](seg_pred_i_sparse,
                                                                                      voxel_semantics_sparse)

                loss_dict_i_b['loss_sem_ce'] = CE_ssc_loss(seg_pred_i_sparse, voxel_semantics_sparse,
                                                           self.class_weights.type_as(seg_pred_i_sparse))

                for loss_key in loss_dict_i_b.keys():
                    loss_dict_i[loss_key] = loss_dict_i.get(loss_key, 0) + loss_dict_i_b[loss_key] / B

            for k, v in loss_dict_i.items():
                loss_dict['%s_%d' % (k, i)] = v

        if (
                not preds_dicts['mask_preds']
                or voxel_instances is None
                or instance_class_ids is None
        ):
            return loss_dict

        occ_loc = preds_dicts['occ_preds'][-1][0]  # [B, N=64000, 3]
        batch_idx = torch.arange(B, device=occ_loc.device)[:, None].expand(B, occ_loc.shape[1])
        flat_occ_loc = occ_loc.reshape(-1, 3)
        flat_batch_idx = batch_idx.reshape(-1)
        voxel_instances = voxel_instances[
            flat_batch_idx,
            flat_occ_loc[:, 0],
            flat_occ_loc[:, 1],
            flat_occ_loc[:, 2],
        ].reshape(B, -1)
        if valid_mask is not None:
            valid_sparse = valid_mask[
                flat_batch_idx,
                flat_occ_loc[:, 0],
                flat_occ_loc[:, 1],
                flat_occ_loc[:, 2],
            ].reshape(B, -1).bool()
            voxel_instances = voxel_instances.masked_fill(
                ~valid_sparse, self.instance_ignore_id)

        # drop instances if it has no positive voxels
        for b in range(B):
            instance_count = instance_class_ids[b].shape[0]
            instance_voxel_counts = torch.bincount(
                voxel_instances[b].long(),
                minlength=max(instance_count + 1, self.instance_ignore_id + 1),
            )
            id_map = torch.cumsum(instance_voxel_counts > 0, dim=0) - 1
            id_map[self.instance_ignore_id] = self.instance_ignore_id
            voxel_instances[b] = id_map[voxel_instances[b].long()]
            kept = instance_voxel_counts[:instance_count] > 0
            instance_class_ids[b] = instance_class_ids[b][kept]
            if instance_class_ids[b].numel() > 0:
                remapped = self.semantic_to_query[instance_class_ids[b].long()]
                if (remapped < 0).any():
                    bad = instance_class_ids[b][remapped < 0].detach().cpu().tolist()
                    raise ValueError(f'Cannot train SparseOcc panoptic on free/unknown classes: {bad}')
                instance_class_ids[b] = remapped

        for i, pred in enumerate(preds_dicts['mask_preds']):
            indices = self.matcher(pred, preds_dicts['class_preds'][i], voxel_instances, instance_class_ids)
            loss_mask, loss_dice, loss_class = self.criterions['loss_mask2former'](
                pred, preds_dicts['class_preds'][i], voxel_instances, instance_class_ids, indices)
            loss_dict['loss_mask_{:d}'.format(i)] = loss_mask
            loss_dict['loss_dice_mask_{:d}'.format(i)] = loss_dice
            loss_dict['loss_class_{:d}'.format(i)] = loss_class

        return loss_dict

    def merge_occ_pred(self, outs):
        if not outs['mask_preds']:
            occ_indices, _, seg_pred, _, _ = outs['occ_preds'][-1]
            outs['sem_pred'] = seg_pred.argmax(dim=-1)
            outs['occ_loc'] = occ_indices
            return outs

        mask_cls = outs['class_preds'][-1].sigmoid()
        mask_pred = outs['mask_preds'][-1].sigmoid()
        occ_indices = outs['occ_preds'][-1][0]

        sem_pred = self.merge_semseg(mask_cls, mask_pred)
        outs['sem_pred'] = sem_pred
        outs['occ_loc'] = occ_indices

        if self.panoptic:
            pano_inst, pano_sem = self.merge_panoseg(mask_cls, mask_pred)
            outs['pano_inst'] = pano_inst
            outs['pano_sem'] = pano_sem

        return outs

    # https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/mask_former_model.py#L242
    def merge_semseg(self, mask_cls, mask_pred):
        valid_mask = mask_cls.max(dim=-1).values > self.score_threshold
        mask_cls[~valid_mask] = 0.0

        semseg = torch.einsum("bqc,bqn->bcn", mask_cls, mask_pred)
        cls_score, query_id = torch.max(semseg, dim=1)
        sem_id = self.query_to_semantic[query_id]
        sem_id = sem_id.masked_fill(cls_score < self.free_score_threshold, self.free_id)
        return sem_id

    def merge_panoseg(self, mask_cls, mask_pred):
        pano_inst, pano_sem = [], []
        for b in range(mask_cls.shape[0]):
            pano_inst_b, pano_sem_b = self.merge_panoseg_single(
                mask_cls[b:b + 1],
                mask_pred[b:b + 1]
            )
            pano_inst.append(pano_inst_b)
            pano_sem.append(pano_sem_b)

        pano_inst = torch.cat(pano_inst, dim=0)
        pano_sem = torch.cat(pano_sem, dim=0)

        return pano_inst, pano_sem

    # https://github.com/facebookresearch/Mask2Former/blob/main/mask2former/maskformer_model.py#L286
    def merge_panoseg_single(self, mask_cls, mask_pred):
        assert mask_cls.shape[0] == 1, "bs != 1"
        scores, labels = mask_cls.max(-1)

        keep = labels.ne(self.no_object_id) & (scores > self.score_threshold)
        cur_scores = scores[keep]
        cur_classes = self.query_to_semantic[labels[keep]]
        cur_masks = mask_pred[keep]

        cur_prob_masks = cur_scores.view(-1, 1) * cur_masks

        N = cur_masks.shape[-1]
        instance_seg = torch.zeros((N), dtype=torch.int32, device=cur_masks.device)
        semantic_seg = torch.ones((N), dtype=torch.int32, device=cur_masks.device) * self.free_id

        current_segment_id = 0
        stuff_memory_list = {self.free_id: 0}

        # skip all process if no mask is detected
        if cur_masks.shape[0] != 0:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)  # [N]
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                is_thing = bool(self.thing_class_mask[pred_class].item())

                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not is_thing:
                        if int(pred_class) in stuff_memory_list.keys():
                            instance_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    instance_seg[mask] = current_segment_id
                    semantic_seg[mask] = pred_class

        instance_seg = instance_seg.unsqueeze(0)
        semantic_seg = semantic_seg.unsqueeze(0)

        return instance_seg, semantic_seg  # [B, N]
