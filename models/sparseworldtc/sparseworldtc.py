import torch
from mmdet3d.registry import MODELS

from loaders.occ_data import OccData
from models.panobench_occ_model import PanoBenchOccModel


@MODELS.register_module()
class SparseWorldTC(PanoBenchOccModel):
    """Trajectory-conditioned 4D occupancy forecasting.

    Only history images are consumed. Future ego poses enter through
    ``fut2cur``, which conditions the queries and reprojects each horizon's
    sampling points back into the history views.
    """

    def _stack_future_gt(self, data_samples, field, device):
        """Stack per-sample ``[FT, X, Y, Z]`` GT into ``[B, FT, X, Y, Z]``."""
        return torch.stack(
            [getattr(ds.gt_occ, field).to(device) for ds in data_samples], dim=0)

    def loss(self, inputs, data_samples, **kwargs):
        img = inputs["img"]
        occ2img = inputs["occ2img"]
        fut2cur = inputs["fut2cur"]
        image_shape = img.shape[-2:]
        voxel_semantics = self._stack_future_gt(data_samples, "voxel_semantics", img.device)
        valid_mask = self._stack_future_gt(data_samples, "valid_mask", img.device)

        with self._amp_context(img.device, enabled=False):
            img_feats = self.extract_feat(img)
            outs = self.occ_head(img_feats, occ2img, image_shape, fut2cur)
            return self.occ_head.loss(voxel_semantics, valid_mask, outs)

    def predict(self, inputs, data_samples, **kwargs):
        img = inputs["img"]
        occ2img = inputs["occ2img"]
        fut2cur = inputs["fut2cur"]
        image_shape = img.shape[-2:]
        filenames = self._get_filenames(data_samples)

        img_feats = self.extract_cached_feat(img, filenames)
        outs = self.occ_head(img_feats, occ2img, image_shape, fut2cur)
        results = self.occ_head.get_occ(outs)
        for data_sample, per_horizon in zip(data_samples, results):
            data_sample.pred_occ = OccData(
                semantics=[r["sem_pred"] for r in per_horizon],
                occ_loc=[r["occ_loc"] for r in per_horizon],
            )
        return data_samples

    def _forward(self, inputs, data_samples=None, **kwargs):
        img = inputs["img"]
        img_feats = self.extract_feat(img)
        return self.occ_head(
            img_feats, inputs["occ2img"], img.shape[-2:], inputs["fut2cur"])
