"""Current-frame PanoBench occupancy metrics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mmdet3d.registry import METRICS
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from utils.labels import get_dataset_meta
from utils.metas import carlaocc_camera_names
from utils.panoptic_occupancy import PanopticOccupancy
from .panoptic import (
    merge_panoptic_stats,
    normalize_panoptic_grid,
    panoptic_stats,
    summarize_panoptic,
)
from .ray import (
    build_camera_rays,
    init_dvr,
    ray_iou_stats,
    ray_pq_stats,
    render_camera_ray_hits,
    summarize_rayiou,
)
from .semantic import (
    format_class_iou,
    semantic_confusion_matrix,
    summarize_iou,
)


@dataclass(frozen=True)
class OccupancyTarget:
    semantic: np.ndarray
    instance: np.ndarray
    valid_mask: np.ndarray


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _require_attr(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        if name not in obj:
            raise KeyError(f"Expected key '{name}' in prediction dict.")
        return obj[name]
    if not hasattr(obj, name):
        raise KeyError(f"Expected field '{name}' on {type(obj).__name__}.")
    return getattr(obj, name)


def _require_pred_occ(data_sample: Any) -> Any:
    if isinstance(data_sample, dict):
        if "pred_occ" not in data_sample:
            raise KeyError("PanoBenchOccMetric expects data_sample['pred_occ'].")
        return data_sample["pred_occ"]
    if not hasattr(data_sample, "pred_occ"):
        raise KeyError("PanoBenchOccMetric expects data_sample.pred_occ.")
    return data_sample.pred_occ


def _require_metainfo(data_sample: Any) -> dict:
    if isinstance(data_sample, dict):
        return data_sample
    if not hasattr(data_sample, "metainfo"):
        raise KeyError("PanoBenchOccMetric expects data_sample.metainfo.")
    return dict(data_sample.metainfo)


def _sparse_field_to_dense(
    occ_loc: Any,
    values: Any,
    volume_size: tuple[int, int, int],
    fill_value: int,
    dtype,
) -> np.ndarray:
    loc = _as_numpy(occ_loc).reshape(-1, 3).astype(np.int64, copy=False)
    vals = _as_numpy(values).reshape(-1).astype(dtype, copy=False)
    if loc.shape[0] != vals.shape[0]:
        raise ValueError(
            f"Prediction coordinate/value length mismatch: {loc.shape[0]} "
            f"vs {vals.shape[0]}."
        )
    if loc.shape[0] > 0:
        in_bounds = ((loc >= 0) & (loc < np.asarray(volume_size))).all(axis=1)
        if not np.all(in_bounds):
            bad = loc[~in_bounds][:5].tolist()
            raise ValueError(f"Prediction occ_loc contains out-of-volume coordinates: {bad}")

    dense = np.full(volume_size, fill_value, dtype=dtype)
    if loc.shape[0] > 0:
        dense[loc[:, 0], loc[:, 1], loc[:, 2]] = vals
    return dense


def _prediction_semantic_grid(
    pred_occ: Any,
    volume_size: tuple[int, int, int],
    free_id: int,
    num_classes: int,
) -> np.ndarray:
    semantic = _sparse_field_to_dense(
        _require_attr(pred_occ, "occ_loc"),
        _require_attr(pred_occ, "semantics"),
        volume_size,
        free_id,
        np.int64,
    )
    bad = (semantic < 0) | (semantic >= num_classes)
    if np.any(bad):
        bad_values = np.unique(semantic[bad])[:10].tolist()
        raise ValueError(f"Prediction semantic ids outside [0, {num_classes}): {bad_values}")
    return semantic


def _prediction_panoptic_grids(
    pred_occ: Any,
    volume_size: tuple[int, int, int],
    free_id: int,
    num_classes: int,
    thing_ids: frozenset[int],
) -> tuple[np.ndarray, np.ndarray]:
    sem = _sparse_field_to_dense(
        _require_attr(pred_occ, "occ_loc"),
        _require_attr(pred_occ, "pano_sem"),
        volume_size,
        free_id,
        np.int64,
    )
    inst = _sparse_field_to_dense(
        _require_attr(pred_occ, "occ_loc"),
        _require_attr(pred_occ, "pano_inst"),
        volume_size,
        0,
        np.int64,
    )
    bad = (sem < 0) | (sem >= num_classes)
    if np.any(bad):
        bad_values = np.unique(sem[bad])[:10].tolist()
        raise ValueError(f"Prediction panoptic semantic ids outside [0, {num_classes}): {bad_values}")
    return normalize_panoptic_grid(sem, inst, free_id, thing_ids)


def _masked_semantic(semantic: np.ndarray, valid_mask: np.ndarray, free_id: int) -> np.ndarray:
    out = np.asarray(semantic, dtype=np.int64).copy()
    out[~valid_mask] = int(free_id)
    return out


def _masked_panoptic(
    semantic: np.ndarray,
    instance: np.ndarray,
    valid_mask: np.ndarray,
    free_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    sem = np.asarray(semantic, dtype=np.int64).copy()
    inst = np.asarray(instance, dtype=np.int64).copy()
    sem[~valid_mask] = int(free_id)
    inst[~valid_mask] = 0
    return sem, inst


def _metric_enabled(metrics: set[str], *names: str) -> bool:
    return any(name in metrics for name in names)


@METRICS.register_module()
class PanoBenchOccMetric(BaseMetric):
    """Strict current-frame occupancy metric for canonical PanoBench samples."""

    default_prefix = "occ"

    VALID_METRICS = ("iou", "miou", "rayiou", "pq", "raypq")
    RAY_STRIDE = 4
    RAY_THRESHOLDS = (1.0, 2.0, 4.0)
    CURRENT_NUM_VIEWS = {
        "panobench_nuscenes": 6,
        "panobench_carlaocc": 6,
        "panobench_kitti360": 1,
    }

    def __init__(
        self,
        dataset_type: str,
        occ_variant,
        metrics: Sequence[str] = ("iou", "miou"),
        use_mask: bool = True,
        collect_device: str = "cpu",
        prefix: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix, **kwargs)
        metric_set = tuple(str(metric).lower() for metric in metrics)
        unknown = sorted(set(metric_set) - set(self.VALID_METRICS))
        if unknown:
            raise KeyError(
                f"Unsupported occupancy metric(s): {unknown}. "
                f"Supported metrics are {self.VALID_METRICS}."
            )
        if not metric_set:
            raise ValueError("PanoBenchOccMetric requires at least one metric.")
        if not isinstance(use_mask, bool):
            raise TypeError("use_mask must be a bool.")

        self.dataset_type = str(dataset_type)
        self.occ_variant = occ_variant
        self.metrics = set(metric_set)
        self.use_mask = use_mask
        self.occ_meta = get_dataset_meta(self.dataset_type, self.occ_variant)
        self.class_names = tuple(self.occ_meta.class_names)
        self.num_classes = len(self.class_names)
        self.free_id = int(self.occ_meta.free_id)
        self.eval_class_ids = tuple(idx for idx in range(self.num_classes) if idx != self.free_id)

        self.thing_ids = self.occ_meta.thing_ids

        self.need_semantic = _metric_enabled(self.metrics, "iou", "miou", "rayiou")
        self.need_panoptic = _metric_enabled(self.metrics, "pq", "raypq")
        self.need_ray = _metric_enabled(self.metrics, "rayiou", "raypq")
        if self.need_ray and self.dataset_type not in self.CURRENT_NUM_VIEWS:
            raise KeyError(f"No current-view count for dataset_type={self.dataset_type!r}.")

        self._dvr = None

    def _load_target(self, occ_path: Path) -> OccupancyTarget:
        sample = PanopticOccupancy.load(occ_path)
        if sample.volume_size != tuple(self.occ_meta.volume_size):
            raise ValueError(
                f"{occ_path} volume_size={sample.volume_size} does not match "
                f"dataset metadata volume_size={tuple(self.occ_meta.volume_size)}."
            )
        if not np.isclose(sample.voxel_size, float(self.occ_meta.voxel_size)):
            raise ValueError(
                f"{occ_path} voxel_size={sample.voxel_size} does not match "
                f"dataset metadata voxel_size={float(self.occ_meta.voxel_size)}."
            )
        if not np.allclose(
            np.asarray(sample.voxel_origin, dtype=np.float32),
            np.asarray(self.occ_meta.voxel_origin, dtype=np.float32),
        ):
            raise ValueError(
                f"{occ_path} voxel_origin={sample.voxel_origin} does not match "
                f"dataset metadata voxel_origin={tuple(self.occ_meta.voxel_origin)}."
            )

        semantic = sample.dense_semantic(self.free_id, dtype=np.int64)
        instance = sample.dense_instance(dtype=np.int64)
        semantic, instance = normalize_panoptic_grid(
            semantic,
            instance,
            self.free_id,
            self.thing_ids,
            strict_thing_instance=self.need_panoptic,
        )
        valid_mask = (
            sample.valid_mask.astype(np.bool_, copy=False)
            if self.use_mask
            else np.ones(sample.volume_size, dtype=np.bool_)
        )
        return OccupancyTarget(semantic=semantic, instance=instance, valid_mask=valid_mask)

    def _current_occ2img(self, data_batch: dict, sample_idx: int) -> np.ndarray:
        occ2img = data_batch["inputs"]["occ2img"]
        arr = _as_numpy(occ2img)
        if arr.ndim == 4:
            arr = arr[sample_idx]
        if arr.ndim != 3 or arr.shape[-2:] != (4, 4):
            raise ValueError(f"data_batch inputs.occ2img must be [B,N,4,4], got {arr.shape}.")

        num_views = self.CURRENT_NUM_VIEWS[self.dataset_type]
        if self.dataset_type == "panobench_carlaocc":
            num_views = len(carlaocc_camera_names(self.occ_variant))
        if arr.shape[0] < num_views:
            raise ValueError(
                f"data_batch inputs.occ2img has {arr.shape[0]} views, but "
                f"{self.dataset_type} current-frame eval requires {num_views}."
            )
        return np.asarray(arr[:num_views], dtype=np.float32)

    @staticmethod
    def _image_shape(data_batch: dict) -> tuple[int, int]:
        img = data_batch["inputs"]["img"]
        return int(img.shape[-2]), int(img.shape[-1])

    def _ensure_dvr(self):
        if self._dvr is None:
            self._dvr = init_dvr(Path(__file__).parent / "dvr")
        return self._dvr

    def process(self, data_batch: dict, data_samples: Sequence[Any]) -> None:
        for sample_idx, data_sample in enumerate(data_samples):
            meta = _require_metainfo(data_sample)
            if "occ_path" not in meta:
                raise KeyError("Current-frame evaluation requires metainfo['occ_path'].")

            target = self._load_target(Path(meta["occ_path"]))
            pred_occ = _require_pred_occ(data_sample)
            result = {}

            pred_sem = None
            if self.need_semantic:
                pred_sem = _prediction_semantic_grid(
                    pred_occ,
                    target.semantic.shape,
                    self.free_id,
                    self.num_classes,
                )
                if _metric_enabled(self.metrics, "iou", "miou"):
                    result["semantic_confusion"] = semantic_confusion_matrix(
                        pred_sem,
                        target.semantic,
                        target.valid_mask,
                        self.num_classes,
                    )

            pred_pano_sem = pred_pano_inst = None
            if self.need_panoptic:
                pred_pano_sem, pred_pano_inst = _prediction_panoptic_grids(
                    pred_occ,
                    target.semantic.shape,
                    self.free_id,
                    self.num_classes,
                    self.thing_ids,
                )
                if "pq" in self.metrics:
                    result["panoptic_stats"] = panoptic_stats(
                        pred_pano_sem,
                        pred_pano_inst,
                        target.semantic,
                        target.instance,
                        target.valid_mask,
                        self.eval_class_ids,
                    )

            if self.need_ray:
                occ2img = self._current_occ2img(data_batch, sample_idx)
                rays = build_camera_rays(occ2img, self._image_shape(data_batch), self.RAY_STRIDE)
                dvr = self._ensure_dvr()
                gt_sem_ray, gt_inst_ray = _masked_panoptic(
                    target.semantic,
                    target.instance,
                    target.valid_mask,
                    self.free_id,
                )
                gt_hits = render_camera_ray_hits(
                    gt_sem_ray,
                    gt_inst_ray,
                    rays,
                    dvr,
                    tuple(self.occ_meta.point_cloud_range),
                    float(self.occ_meta.voxel_size),
                    self.free_id,
                )

                if "rayiou" in self.metrics:
                    assert pred_sem is not None
                    pred_sem_ray = _masked_semantic(pred_sem, target.valid_mask, self.free_id)
                    pred_hits = render_camera_ray_hits(
                        pred_sem_ray,
                        np.zeros_like(pred_sem_ray, dtype=np.int64),
                        rays,
                        dvr,
                        tuple(self.occ_meta.point_cloud_range),
                        float(self.occ_meta.voxel_size),
                        self.free_id,
                    )
                    ray_confusion, ray_depth_failed = ray_iou_stats(
                        pred_hits,
                        gt_hits,
                        self.RAY_THRESHOLDS,
                        self.num_classes,
                        self.free_id,
                    )
                    result["ray_confusion"] = ray_confusion
                    result["ray_depth_failed"] = ray_depth_failed

                if "raypq" in self.metrics:
                    assert pred_pano_sem is not None and pred_pano_inst is not None
                    pred_pano_sem_ray, pred_pano_inst_ray = _masked_panoptic(
                        pred_pano_sem,
                        pred_pano_inst,
                        target.valid_mask,
                        self.free_id,
                    )
                    pred_pano_hits = render_camera_ray_hits(
                        pred_pano_sem_ray,
                        pred_pano_inst_ray,
                        rays,
                        dvr,
                        tuple(self.occ_meta.point_cloud_range),
                        float(self.occ_meta.voxel_size),
                        self.free_id,
                    )
                    result["raypq_stats"] = ray_pq_stats(
                        pred_pano_hits,
                        gt_hits,
                        self.RAY_THRESHOLDS,
                        self.eval_class_ids,
                        self.free_id,
                    )

            self.results.append(result)

    @staticmethod
    def _percent(value: float) -> float:
        return round(float(value) * 100.0, 2)

    def compute_metrics(self, results: list[dict]) -> dict:
        logger = MMLogger.get_current_instance()
        metrics = {}
        if not results:
            return metrics

        if _metric_enabled(self.metrics, "iou", "miou"):
            logger.info("Starting Evaluation (IoU/mIoU)...")
            confusion = sum(result["semantic_confusion"] for result in results)
            per_class_iou, iou, miou = summarize_iou(confusion, self.free_id)
            logger.info("\n" + format_class_iou(per_class_iou, self.class_names, self.eval_class_ids))
            if "iou" in self.metrics:
                metrics["IoU"] = self._percent(iou)
            if "miou" in self.metrics:
                metrics["mIoU"] = self._percent(miou)

        if "rayiou" in self.metrics:
            logger.info("Starting Evaluation (RayIoU)...")
            ray_confusion = sum(result["ray_confusion"] for result in results)
            ray_depth_failed = sum(result["ray_depth_failed"] for result in results)
            rayiou = summarize_rayiou(ray_confusion, ray_depth_failed, self.free_id)
            for threshold, value in zip(self.RAY_THRESHOLDS, rayiou):
                metrics[f"RayIoU@{threshold:g}m"] = self._percent(value)

        if "pq" in self.metrics:
            logger.info("Starting Evaluation (PQ)...")
            stats = merge_panoptic_stats(
                [result["panoptic_stats"] for result in results],
                self.eval_class_ids,
            )
            sq, rq, pq = summarize_panoptic(stats, self.eval_class_ids)
            metrics["SQ"] = self._percent(sq)
            metrics["RQ"] = self._percent(rq)
            metrics["PQ"] = self._percent(pq)

        if "raypq" in self.metrics:
            logger.info("Starting Evaluation (RayPQ)...")
            for threshold in self.RAY_THRESHOLDS:
                stats = merge_panoptic_stats(
                    [result["raypq_stats"][float(threshold)] for result in results],
                    self.eval_class_ids,
                )
                _, _, pq = summarize_panoptic(stats, self.eval_class_ids)
                metrics[f"RayPQ@{threshold:g}m"] = self._percent(pq)

        torch.cuda.empty_cache()
        return metrics
