"""Common MMEngine-style base class for PanoBench occupancy models."""

import queue

import torch
from mmdet.registry import MODELS as MMDET_MODELS
from mmdet3d.registry import MODELS
from mmengine.dist import get_dist_info
from mmengine.model import BaseModel
from torch.cuda.amp import autocast

from utils.model_ops import GpuPhotoMetricDistortion, IMG_NORM_CFG


class PanoBenchOccModel(BaseModel):
    """Base class for camera-only PanoBench occupancy models.

    The public model interface follows mmdet3d's convention:
    ``forward(inputs, data_samples=None, mode='tensor', **kwargs)`` dispatches
    to ``loss``, ``predict`` or ``_forward``.
    """

    def __init__(
            self,
            img_backbone=None,
            img_neck=None,
            occ_head=None,
            num_cams=6,
            init_cfg=None,
    ):
        super().__init__(init_cfg=init_cfg)

        if img_backbone is not None:
            self.img_backbone = MMDET_MODELS.build(img_backbone)
        if img_neck is not None:
            self.img_neck = MMDET_MODELS.build(img_neck)
        if occ_head is not None:
            self.occ_head = MODELS.build(occ_head)

        self.num_cams = int(num_cams)
        self.cache_size = 16
        self.color_aug = GpuPhotoMetricDistortion()

        self.register_buffer("_img_mean", torch.tensor(IMG_NORM_CFG["mean"]).view(1, 3, 1, 1))
        self.register_buffer("_img_std", torch.tensor(IMG_NORM_CFG["std"]).view(1, 3, 1, 1))

        self.memory = {}
        self.queue = queue.Queue()

    def forward(self, inputs, data_samples=None, mode="tensor", **kwargs):
        """
        inputs (Dict):
            - img (Tensor, uint8): [B, N, C, H, W], N = num_frames * num_views
            - occ2img (Tensor, float32): [B, N, 4, 4]

        data_samples (List[PanoBenchOccDataSample]):
            - gt_occ (OccData):
                - voxel_semantics (Tensor, uint8): [X, Y, Z] dense grid
                - valid_mask (Tensor, bool): [X, Y, Z] dense grid
                - voxel_instances (Tensor, optional): [X, Y, Z] compact
                  instance/stuff target ids for SparseOcc.
                - instance_class_ids (Tensor, optional): [num_targets],
                  semantic class id for each compact target id.
            - metainfo (Dict):
                - filename (List[str]): Image paths for the current and history frames.
                - occ_path (str): Canonical occupancy ``.npz`` path for the sample.
                - frame_token (str): Canonical frame token.
        """
        if mode == "loss":
            return self.loss(inputs, data_samples, **kwargs)
        if mode == "predict":
            return self.predict(inputs, data_samples, **kwargs)
        if mode == "tensor":
            return self._forward(inputs, data_samples, **kwargs)
        raise RuntimeError(f'Invalid mode "{mode}". Only supports loss, predict and tensor mode')

    def loss(self, inputs, data_samples, **kwargs):
        raise NotImplementedError

    def predict(self, inputs, data_samples, **kwargs):
        raise NotImplementedError

    def _forward(self, inputs, data_samples=None, **kwargs):
        raise NotImplementedError

    def _amp_context(self, device, enabled: bool):
        return autocast(enabled=enabled and device is not None and device.type == "cuda")

    def _get_filenames(self, data_samples):
        return [ds.metainfo["filename"] for ds in data_samples]

    def _make_img_metas(self, img, data_samples, occ2img):
        img_h, img_w = img.shape[-2:]
        metas = []
        for i, ds in enumerate(data_samples):
            meta = dict(ds.metainfo)
            meta["occ2img"] = occ2img[i]
            meta["img_shape"] = (int(img_h), int(img_w), 3)
            metas.append(meta)
        return metas

    def _stack_gt_occ_field(self, data_samples, field, device):
        return torch.stack([getattr(ds.gt_occ, field).to(device) for ds in data_samples], dim=0)

    def _get_instance_class_ids(self, data_samples, device):
        if not hasattr(data_samples[0].gt_occ, "instance_class_ids"):
            return None
        return [
            ds.gt_occ.instance_class_ids.to(device).long()
            for ds in data_samples
        ]

    def extract_img_feat(self, img):
        device = img.device
        with self._amp_context(device, enabled=True):
            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
            if getattr(self, "img_neck", None) is not None:
                img_feats = self.img_neck(img_feats)
        return [feat.float() for feat in img_feats]

    def extract_feat(self, img):
        assert img.dim() == 5, f"expected [B, N, C, H, W], got {tuple(img.shape)}"
        b, n, c, h, w = img.size()
        img = img.reshape(b * n, c, h, w).float()

        img = self.apply_data_aug(img)
        img_feats = self.extract_img_feat(img)
        return [feat.reshape(b, n, *feat.shape[1:]) for feat in img_feats]

    def apply_data_aug(self, img):
        if self.training:
            img = self.color_aug(img)

        img = img[:, [2, 1, 0]]
        img = (img - self._img_mean) / self._img_std
        return img

    def extract_cached_feat(self, img, filenames):
        world_size = get_dist_info()[1]
        return self._extract_with_cache(img, filenames, cache=world_size == 1)

    def _extract_with_cache(self, img, filenames, cache: bool):
        assert img.shape[0] == 1 and len(filenames) == 1, "val/test expects batch_size=1"
        n_cams = self.num_cams
        filenames = filenames[0]
        n_frames = len(filenames) // n_cams

        b, total_views, c, h, w = img.shape
        n_frames_loaded = total_views // n_cams
        assert (
                total_views == n_frames_loaded * n_cams
        ), f"img views {total_views} is not a multiple of n_cams {n_cams}"
        img = img.reshape(b, n_frames_loaded, n_cams, c, h, w)

        feats_per_frame = []
        for i in range(n_frames):
            idxs = list(range(i * n_cams, (i + 1) * n_cams))
            key = filenames[idxs[0]]
            if cache and key in self.memory:
                feats_i = self.memory[key]
            else:
                assert i < n_frames_loaded, (
                    f"frame {i} missing from cache and not in current batch "
                    f"(only {n_frames_loaded} frames loaded); {key!r}"
                )
                feats_i = self.extract_feat(img[:, i])
                if cache:
                    self._cache_put(key, feats_i)
            feats_per_frame.append(feats_i)

        img_feats = []
        for lvl in range(len(feats_per_frame[0])):
            feat = torch.cat([frame_feats[lvl] for frame_feats in feats_per_frame], dim=0)
            img_feats.append(feat.flatten(0, 1)[None, ...].float())
        return img_feats

    def _cache_put(self, key, feats):
        self.memory[key] = feats
        self.queue.put(key)
        while self.queue.qsize() > self.cache_size:
            self.memory.pop(self.queue.get())
