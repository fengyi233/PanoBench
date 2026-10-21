"""GaussianFormer adapted to PanoBench's MMEngine-style interface."""

from mmdet3d.registry import MODELS

from loaders.occ_data import OccData
from models.panobench_occ_model import PanoBenchOccModel


@MODELS.register_module()
class GaussianFormerV1(PanoBenchOccModel):

    def __init__(self, lifter, encoder, **kwargs):
        super().__init__(**kwargs)
        self.lifter = MODELS.build(lifter)
        self.encoder = MODELS.build(encoder)

    def _make_metas(self, img, occ2img):
        """``occ2img`` already maps metric occupancy coordinates to pixels."""
        img_h, img_w = img.shape[-2:]
        image_wh = occ2img.new_tensor([img_w, img_h]).view(1, 1, 2)
        return dict(
            projection_mat=occ2img,
            image_wh=image_wh.expand(*occ2img.shape[:2], 2),
        )

    def _predict_occ(self, img, img_feats, occ2img):
        assert img.shape[1] == self.num_cams, (
            f'GaussianFormer uses single-frame inputs; got {img.shape[1]} views for '
            f'num_cams={self.num_cams}. Set data.num_frames=1.')
        metas = self._make_metas(img, occ2img)
        lifted = self.lifter(img_feats)
        gaussians = self.encoder(
            lifted['representation'], lifted['rep_features'], img_feats, metas)
        return self.occ_head(gaussians)

    def loss(self, inputs, data_samples, **kwargs):
        img = inputs['img']
        occ2img = inputs['occ2img']
        voxel_semantics = self._stack_gt_occ_field(data_samples, 'voxel_semantics', img.device)
        valid_mask = self._stack_gt_occ_field(data_samples, 'valid_mask', img.device)

        with self._amp_context(img.device, enabled=False):
            img_feats = self.extract_feat(img)
            outs = self._predict_occ(img, img_feats, occ2img)
            return self.occ_head.loss(voxel_semantics, valid_mask, outs)

    def predict(self, inputs, data_samples, **kwargs):
        img = inputs['img']
        occ2img = inputs['occ2img']
        filenames = self._get_filenames(data_samples)

        img_feats = self.extract_cached_feat(img, filenames)
        outs = self._predict_occ(img, img_feats, occ2img)
        results = self.occ_head.get_occ(outs)
        for data_sample, result in zip(data_samples, results):
            data_sample.pred_occ = OccData(
                semantics=result['sem_pred'],
                occ_loc=result['occ_loc'],
            )
        return data_samples

    def _forward(self, inputs, data_samples=None, **kwargs):
        img = inputs['img']
        img_feats = self.extract_feat(img)
        return self._predict_occ(img, img_feats, inputs['occ2img'])
