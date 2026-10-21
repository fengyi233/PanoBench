import torch
import torch.nn.functional as F
from mmdet.registry import MODELS as MMDET_MODELS
from mmdet3d.registry import MODELS

from loaders.occ_data import OccData
from models.panobench_occ_model import PanoBenchOccModel


@MODELS.register_module()
class FlashOCC(PanoBenchOccModel):
    def __init__(self,
                 dataset_type,
                 occ_variant,
                 img_backbone=None,
                 img_neck=None,
                 img_view_transformer=None,
                 img_bev_encoder_backbone=None,
                 img_bev_encoder_neck=None,
                 occ_head=None,
                 panoptic_head=None,
                 num_cams=6,
                 init_cfg=None):
        super().__init__(num_cams=num_cams, init_cfg=init_cfg)
        self.dataset_type = dataset_type
        self.occ_variant = occ_variant
        self.img_backbone = MMDET_MODELS.build(img_backbone)
        self.img_neck = MODELS.build(img_neck) if img_neck is not None else None

        for cfg in (img_view_transformer, occ_head, panoptic_head):
            if cfg is None:
                continue
            cfg.setdefault('dataset_type', dataset_type)
            cfg.setdefault('occ_variant', occ_variant)
        self.img_view_transformer = MODELS.build(img_view_transformer)
        self.img_bev_encoder_backbone = MODELS.build(img_bev_encoder_backbone)
        self.img_bev_encoder_neck = MODELS.build(img_bev_encoder_neck)
        self.occ_head = MODELS.build(occ_head)
        self.panoptic_head = MODELS.build(panoptic_head) if panoptic_head is not None else None

    def image_encoder(self, img):
        b, n, c, h, w = img.shape
        img = img.reshape(b * n, c, h, w).float()
        img = self.apply_data_aug(img)
        img_feats = self.extract_img_feat(img)
        feat = img_feats[0]
        return feat.reshape(b, n, *feat.shape[1:])

    def bev_encoder(self, x):
        x = self.img_bev_encoder_backbone(x)
        x = self.img_bev_encoder_neck(x)
        if isinstance(x, (list, tuple)):
            x = x[0]
        return x

    def extract_bev_feat(self, inputs):
        img = inputs['img']
        occ2img = inputs['occ2img']
        assert img.shape[1] == self.num_cams, (
            f'FlashOCC uses single-frame inputs; got {img.shape[1]} views for num_cams={self.num_cams}. '
            'Set data.num_frames=1.')
        img_feat = self.image_encoder(img)
        bev_feat, depth = self.img_view_transformer(img_feat, occ2img)
        bev_feat = self.bev_encoder(bev_feat)
        return bev_feat, depth

    def _predict_logits(self, inputs):
        bev_feat, depth = self.extract_bev_feat(inputs)
        occ_pred = self.occ_head(bev_feat)
        pano_pred = self.panoptic_head(bev_feat) if self.panoptic_head is not None else None
        return occ_pred, pano_pred, depth

    def _resize_occ_pred(self, occ_pred, target_shape):
        if occ_pred.shape[1:4] == target_shape:
            return occ_pred
        pred = occ_pred.permute(0, 4, 3, 2, 1)
        pred = F.interpolate(pred, size=target_shape[2::-1], mode='trilinear', align_corners=False)
        return pred.permute(0, 4, 3, 2, 1).contiguous()

    def _resize_panoptic_pred(self, pano_pred, target_shape):
        if pano_pred is None:
            return None
        x_size, y_size, z_size = target_shape
        center_logits = pano_pred['center_logits']
        if center_logits.shape[-2:] != (y_size, x_size):
            center_logits = F.interpolate(center_logits, size=(y_size, x_size), mode='bilinear', align_corners=False)
        offset = pano_pred['offset']
        if offset.shape[1:4] != target_shape:
            offset = offset.permute(0, 4, 3, 2, 1)
            offset = F.interpolate(offset, size=(z_size, y_size, x_size), mode='trilinear', align_corners=False)
            offset = offset.permute(0, 4, 3, 2, 1).contiguous()
        return dict(center_logits=center_logits, offset=offset)

    def loss(self, inputs, data_samples, **kwargs):
        occ_pred, pano_pred, _ = self._predict_logits(inputs)
        voxel_semantics = self._stack_gt_occ_field(data_samples, 'voxel_semantics', occ_pred.device)
        valid_mask = self._stack_gt_occ_field(data_samples, 'valid_mask', occ_pred.device)
        target_shape = tuple(voxel_semantics.shape[1:4])
        occ_pred = self._resize_occ_pred(occ_pred, target_shape)
        losses = self.occ_head.loss(occ_pred, voxel_semantics, valid_mask)
        if self.panoptic_head is not None:
            pano_pred = self._resize_panoptic_pred(pano_pred, target_shape)
            voxel_instances = None
            instance_class_ids = None
            if hasattr(data_samples[0].gt_occ, 'voxel_instances'):
                voxel_instances = self._stack_gt_occ_field(data_samples, 'voxel_instances', occ_pred.device)
                instance_class_ids = self._get_instance_class_ids(data_samples, occ_pred.device)
            losses.update(self.panoptic_head.loss(pano_pred, voxel_instances, instance_class_ids))
        return losses

    def predict(self, inputs, data_samples, **kwargs):
        occ_pred, pano_pred, _ = self._predict_logits(inputs)
        results = self.occ_head.get_occ(occ_pred)
        dense_sem = occ_pred.argmax(dim=-1)
        pano_sem = pano_inst = None
        if self.panoptic_head is not None:
            pano_pred = self._resize_panoptic_pred(pano_pred, tuple(dense_sem.shape[1:4]))
            pano_sem, pano_inst = self.panoptic_head.predict(pano_pred, dense_sem)
        for idx, (data_sample, result) in enumerate(zip(data_samples, results)):
            pred_occ = OccData(
                semantics=result['sem_pred'],
                occ_loc=result['occ_loc'],
            )
            if pano_sem is not None:
                loc = torch.as_tensor(result['occ_loc'], device=occ_pred.device, dtype=torch.long)
                pred_occ.pano_sem = pano_sem[idx, loc[:, 0], loc[:, 1], loc[:, 2]].detach().cpu().numpy()
                pred_occ.pano_inst = pano_inst[idx, loc[:, 0], loc[:, 1], loc[:, 2]].detach().cpu().numpy()
            data_sample.pred_occ = pred_occ
        return data_samples

    def _forward(self, inputs, data_samples=None, **kwargs):
        return self._predict_logits(inputs)[0]
