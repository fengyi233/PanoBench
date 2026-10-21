"""Per-horizon occupancy forecasting metrics."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from mmdet3d.registry import METRICS
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger

from utils.labels import get_dataset_meta
from utils.panoptic_occupancy import PanopticOccupancy
from .semantic import format_class_iou, semantic_confusion_matrix, summarize_iou


def _as_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _get_field(obj: Any, name: str) -> Any:
    """Read a field from either a data element or its ``to_dict`` form.

    ``mmengine.Evaluator`` flattens data samples with ``to_dict()`` before
    handing them to metrics, so both shapes reach ``process``.
    """
    if isinstance(obj, dict):
        if name not in obj:
            raise KeyError(f"Expected key '{name}' in prediction dict.")
        return obj[name]
    if not hasattr(obj, name):
        raise KeyError(f"Expected field '{name}' on {type(obj).__name__}.")
    return getattr(obj, name)


def _get_metainfo(data_sample: Any) -> dict:
    if isinstance(data_sample, dict):
        return data_sample
    return dict(data_sample.metainfo)


@METRICS.register_module()
class PanoBenchOcc4DMetric(BaseMetric):
    """Semantic mIoU and geometric IoU reported per forecast horizon.

    Each (sample, horizon) pair contributes one confusion matrix, accumulated
    on the rank that produced it so ``compute_metrics`` only has to sum them.
    Both metrics come from the same matrix: ``mIoU`` averages per-class IoU over
    the non-free classes, ``IoU`` is the binary occupied-vs-free IoU.

    ``*_avg`` averages the forecast horizons only. Horizon 0 is the key frame,
    which is perception rather than forecasting, so it is reported but excluded.

    Args:
        future_frames: Horizon offsets in key frames; must match the dataset.
        step_seconds: Wall-clock seconds per key frame, used only to name the
            reported metrics (nuScenes key frames are 2 Hz, so 0.5).
        use_mask: Restrict scoring to each horizon's own ``valid_mask``.
    """

    default_prefix = "occ4d"

    def __init__(
        self,
        dataset_type: str,
        occ_variant,
        future_frames: Sequence[int] = (0, 2, 4, 6),
        step_seconds: float = 0.5,
        use_mask: bool = True,
        collect_device: str = "cpu",
        prefix: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix, **kwargs)
        self.future_frames = [int(f) for f in future_frames]
        if not self.future_frames or self.future_frames[0] != 0:
            raise ValueError(
                f"future_frames must start with 0, got {self.future_frames}.")
        self.forecast_horizons = [f for f in self.future_frames if f > 0]
        if not self.forecast_horizons:
            raise ValueError(
                "future_frames must contain at least one horizon beyond 0.")
        self.step_seconds = float(step_seconds)
        self.use_mask = bool(use_mask)

        self.dataset_type = str(dataset_type)
        self.occ_variant = occ_variant
        self.occ_meta = get_dataset_meta(self.dataset_type, self.occ_variant)
        self.class_names = tuple(self.occ_meta.class_names)
        self.num_classes = len(self.class_names)
        self.free_id = int(self.occ_meta.free_id)
        self.eval_class_ids = tuple(
            idx for idx in range(self.num_classes) if idx != self.free_id)

    def _load_target(self, occ_path: Path) -> tuple[np.ndarray, np.ndarray]:
        sample = PanopticOccupancy.load(occ_path)
        if sample.volume_size != tuple(self.occ_meta.volume_size):
            raise ValueError(
                f"{occ_path} volume_size={sample.volume_size} does not match "
                f"dataset metadata volume_size={tuple(self.occ_meta.volume_size)}."
            )
        semantic = sample.dense_semantic(self.free_id, dtype=np.int64)
        valid_mask = (
            sample.valid_mask.astype(np.bool_, copy=False)
            if self.use_mask
            else np.ones(sample.volume_size, dtype=np.bool_)
        )
        return semantic, valid_mask

    def _dense_prediction(self, occ_loc, semantics, volume_size) -> np.ndarray:
        loc = _as_numpy(occ_loc).reshape(-1, 3).astype(np.int64, copy=False)
        vals = _as_numpy(semantics).reshape(-1).astype(np.int64, copy=False)
        if loc.shape[0] != vals.shape[0]:
            raise ValueError(
                f"Prediction coordinate/value length mismatch: {loc.shape[0]} "
                f"vs {vals.shape[0]}."
            )
        if loc.shape[0] > 0:
            in_bounds = ((loc >= 0) & (loc < np.asarray(volume_size))).all(axis=1)
            if not np.all(in_bounds):
                bad = loc[~in_bounds][:5].tolist()
                raise ValueError(
                    f"Prediction occ_loc contains out-of-volume coordinates: {bad}")

        dense = np.full(volume_size, self.free_id, dtype=np.int64)
        if loc.shape[0] > 0:
            dense[loc[:, 0], loc[:, 1], loc[:, 2]] = vals
        return dense

    def process(self, data_batch: dict, data_samples: Sequence[Any]) -> None:
        num_horizons = len(self.future_frames)
        for data_sample in data_samples:
            meta = _get_metainfo(data_sample)
            if "future_occ_paths" not in meta:
                raise KeyError(
                    "Forecast evaluation requires metainfo['future_occ_paths'].")
            occ_paths = meta["future_occ_paths"]
            if len(occ_paths) != num_horizons:
                raise ValueError(
                    f"Sample has {len(occ_paths)} future occupancy paths, but "
                    f"future_frames has {num_horizons} entries.")

            pred_occ = _get_field(data_sample, "pred_occ")
            pred_locs = _get_field(pred_occ, "occ_loc")
            pred_semantics = _get_field(pred_occ, "semantics")
            confusions = []
            for horizon_idx in range(num_horizons):
                target, valid_mask = self._load_target(Path(occ_paths[horizon_idx]))
                pred = self._dense_prediction(
                    pred_locs[horizon_idx],
                    pred_semantics[horizon_idx],
                    target.shape,
                )
                confusions.append(
                    semantic_confusion_matrix(
                        pred, target, valid_mask, self.num_classes))

            self.results.append(np.stack(confusions, axis=0))

    @staticmethod
    def _percent(value: float) -> float:
        return round(float(value) * 100.0, 2)

    def _tag(self, horizon: int) -> str:
        return f"{horizon * self.step_seconds:g}s"

    def compute_metrics(self, results: list[np.ndarray]) -> dict:
        logger = MMLogger.get_current_instance()
        if not results:
            logger.warning("PanoBenchOcc4DMetric received zero samples; skipping.")
            return {}

        total = np.stack(results, axis=0).sum(axis=0)  # [FT, C, C]
        if total.shape[0] != len(self.future_frames):
            raise RuntimeError(
                f"Collected {total.shape[0]} horizons; "
                f"expected {len(self.future_frames)}.")

        metrics: dict[str, float] = {}
        per_horizon: dict[int, tuple[float, float]] = {}
        for i, horizon in enumerate(self.future_frames):
            per_class_iou, iou, miou = summarize_iou(total[i], self.free_id)
            tag = self._tag(horizon)
            logger.info(
                f"[forecast {tag}] IoU = {self._percent(iou):.2f}  "
                f"mIoU = {self._percent(miou):.2f}\n"
                + format_class_iou(per_class_iou, self.class_names, self.eval_class_ids)
            )
            per_horizon[horizon] = (self._percent(iou), self._percent(miou))
            metrics[f"IoU_{tag}"] = per_horizon[horizon][0]
            metrics[f"mIoU_{tag}"] = per_horizon[horizon][1]

        metrics["IoU_avg"] = round(
            float(np.mean([per_horizon[h][0] for h in self.forecast_horizons])), 2)
        metrics["mIoU_avg"] = round(
            float(np.mean([per_horizon[h][1] for h in self.forecast_horizons])), 2)

        logger.info("\n" + self._summary_table(per_horizon, metrics, len(results)))
        torch.cuda.empty_cache()
        return metrics

    def _summary_table(self, per_horizon, metrics, num_samples) -> str:
        tags = [self._tag(h) for h in self.future_frames]
        width = max(14, *(len(tag) for tag in tags))
        header = "Horizon".ljust(16) + "".join(t.rjust(width) for t in tags)
        header += "Avg".rjust(width)
        rows = []
        for name, index, avg_key in (
                ("Geometric IoU", 0, "IoU_avg"),
                ("Semantic mIoU", 1, "mIoU_avg"),
        ):
            values = [f"{per_horizon[h][index]:.2f}" for h in self.future_frames]
            values.append(f"{metrics[avg_key]:.2f}")
            rows.append(name.ljust(16) + "".join(v.rjust(width) for v in values))
        avg_over = ", ".join(self._tag(h) for h in self.forecast_horizons)
        return "\n".join([
            f"[forecast] {num_samples} samples | Avg over {avg_over}",
            header,
            *rows,
        ])
