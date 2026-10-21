"""SparseOcc model adapted to PanoBench's MMEngine-style interface."""

import numpy as np
from mmdet3d.registry import MODELS

from loaders.occ_data import OccData
from models.panobench_occ_model import PanoBenchOccModel


@MODELS.register_module()
class SparseOcc(PanoBenchOccModel):

    def loss(self, inputs, data_samples, **kwargs):
        img = inputs["img"]
        occ2img = inputs["occ2img"]
        img_metas = self._make_img_metas(img, data_samples, occ2img)
        voxel_semantics = self._stack_gt_occ_field(data_samples, "voxel_semantics", img.device)
        valid_mask = self._stack_gt_occ_field(data_samples, "valid_mask", img.device)
        voxel_instances = None
        instance_class_ids = None
        if hasattr(data_samples[0].gt_occ, "voxel_instances"):
            voxel_instances = self._stack_gt_occ_field(
                data_samples, "voxel_instances", img.device)
            instance_class_ids = self._get_instance_class_ids(data_samples, img.device)

        with self._amp_context(img.device, enabled=False):
            img_feats = self.extract_feat(img)
            outs = self.occ_head(img_feats, img_metas)
            return self.occ_head.loss(
                voxel_semantics,
                valid_mask,
                voxel_instances,
                instance_class_ids,
                outs,
            )

    def predict(self, inputs, data_samples, **kwargs):
        img = inputs["img"]
        occ2img = inputs["occ2img"]
        img_metas = self._make_img_metas(img, data_samples, occ2img)
        filenames = [meta["filename"] for meta in img_metas]

        img_feats = self.extract_cached_feat(img, filenames)
        outs = self.occ_head(img_feats, img_metas)
        results = self.occ_head.merge_occ_pred(outs)
        sem_pred = results["sem_pred"].detach().cpu().numpy().astype(np.uint8)
        occ_loc = results["occ_loc"].detach().cpu().numpy().astype(np.uint16)
        for i, data_sample in enumerate(data_samples):
            pred_occ = dict(
                semantics=sem_pred[i:i + 1],
                occ_loc=occ_loc[i:i + 1],
            )
            if "pano_inst" in results:
                pred_occ["pano_inst"] = (
                    results["pano_inst"][i:i + 1].detach().cpu().numpy().astype(np.int16))
                pred_occ["pano_sem"] = (
                    results["pano_sem"][i:i + 1].detach().cpu().numpy().astype(np.uint8))
            data_sample.pred_occ = OccData(**pred_occ)
        return data_samples

    def _forward(self, inputs, data_samples=None, **kwargs):
        img = inputs["img"]
        occ2img = inputs["occ2img"]
        img_metas = self._make_img_metas(img, data_samples, occ2img)
        img_feats = self.extract_feat(img)
        return self.occ_head(img_feats, img_metas)
