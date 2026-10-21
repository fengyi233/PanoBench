import torch
from mmdet3d.registry import MODELS

from loaders.occ_data import OccData
from models.panobench_occ_model import PanoBenchOccModel


@MODELS.register_module()
class OPUSV1(PanoBenchOccModel):

    def loss(self, inputs, data_samples, **kwargs):
        img = inputs["img"]
        occ2img = inputs["occ2img"]
        image_shape = img.shape[-2:]
        voxel_semantics = self._stack_gt_occ_field(data_samples, "voxel_semantics", img.device)  # [B, X, Y, Z]
        valid_mask = self._stack_gt_occ_field(data_samples, "valid_mask", img.device)  # [B, X, Y, Z]

        with self._amp_context(img.device, enabled=False):
            img_feats = self.extract_feat(img)
            outs = self.occ_head(img_feats, occ2img, image_shape)
            if valid_mask is None and voxel_semantics is not None:
                valid_mask = torch.ones_like(voxel_semantics)
            return self.occ_head.loss(voxel_semantics, valid_mask, outs)

    def predict(self, inputs, data_samples, **kwargs):
        img = inputs["img"]
        occ2img = inputs["occ2img"]
        image_shape = img.shape[-2:]
        filenames = self._get_filenames(data_samples)

        img_feats = self.extract_cached_feat(img, filenames)
        outs = self.occ_head(img_feats, occ2img, image_shape)
        results = self.occ_head.get_occ(outs, rescale=False)
        for data_sample, result in zip(data_samples, results):
            data_sample.pred_occ = OccData(
                semantics=result["sem_pred"],
                occ_loc=result["occ_loc"],
            )
        return data_samples

    def _forward(self, inputs, data_samples=None, **kwargs):
        img = inputs["img"]
        occ2img = inputs["occ2img"]
        img_feats = self.extract_feat(img)
        return self.occ_head(img_feats, occ2img, img.shape[-2:])
