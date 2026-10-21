"""Pack a PanoBench occupancy sample into MMEngine model inputs.

After this transform a sample dict is::

    {
        'inputs': {
            'img': Tensor[N, C, H, W],
            'occ2img': Tensor[N, 4, 4],
            'fut2cur': Tensor[FT, 4, 4],        # forecasting datasets only
        },                                      # N = num_frames * num_views
        'data_samples': PanoBenchOccDataSample,
    }

``default_collate`` stacks tensors in ``inputs`` along the batch dimension and
keeps ``data_samples`` as a list of per-sample data elements.
"""
from typing import Optional, Sequence

import numpy as np
import torch
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS

from loaders.occ_data import OccData, PanoBenchOccDataSample


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.float() if x.dtype == torch.float64 else x
    if isinstance(x, np.ndarray):
        t = torch.from_numpy(np.ascontiguousarray(x))
        return t.float() if t.dtype == torch.float64 else t
    return x


@TRANSFORMS.register_module()
class PackInputs(BaseTransform):
    """Pack ``results`` produced by Load* / Random* transforms.

    Args:
        meta_keys: Keys collected from ``results`` into ``data_sample.metainfo``.
            Only keys that exist in ``results`` are added.
    """
    DEFAULT_META_KEYS = (
        'filename',
        'frame_token',
        'occ_path',
        # Forecasting datasets only; skipped when absent.
        'future_occ_paths',
        'fut_list',
    )
    GT_OCC_FIELDS = (
        'voxel_semantics',
        'valid_mask',
        'voxel_instances',
        'instance_class_ids',
    )

    def __init__(
            self,
            meta_keys: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.meta_keys = tuple(meta_keys) if meta_keys is not None else self.DEFAULT_META_KEYS

    def _pack_img(self, imgs):
        """list[HWC ndarray] → Tensor[N, C, H, W]; ndarray pass-through with HWC→CHW."""
        n = len(imgs)
        h, w, c = imgs[0].shape
        out = np.empty((n, c, h, w), dtype=imgs[0].dtype)
        for i, img in enumerate(imgs):
            out[i] = img.transpose(2, 0, 1)
        return _to_tensor(out)

    def _pack_matrices(self, matrices):
        stacked = np.stack([np.asarray(x, dtype=np.float32) for x in matrices], axis=0)
        return _to_tensor(stacked)

    def transform(self, results: dict) -> dict:
        inputs = {
            'img': self._pack_img(results['img']),
            'occ2img': self._pack_matrices(results['occ2img'])
        }
        if 'fut2cur' in results:
            inputs['fut2cur'] = self._pack_matrices(results['fut2cur'])

        data_sample = PanoBenchOccDataSample()
        gt_occ = {}
        for key in self.GT_OCC_FIELDS:
            if key not in results:
                continue
            value = results[key]
            if isinstance(value, np.ndarray):
                value = _to_tensor(value)
            gt_occ[key] = value
        if gt_occ:
            data_sample.gt_occ = OccData(**gt_occ)

        # Minimal metadata consumed by feature cache and evaluator.
        metainfo = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(metainfo)

        return {'inputs': inputs, 'data_samples': data_sample}

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'meta_keys={self.meta_keys})')
