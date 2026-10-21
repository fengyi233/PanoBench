"""Semantic occupancy IoU statistics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def semantic_confusion_matrix(
    pred: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray,
    num_classes: int,
) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.int64)
    gt = np.asarray(gt, dtype=np.int64)
    valid_mask = np.asarray(valid_mask, dtype=np.bool_)
    if pred.shape != gt.shape or gt.shape != valid_mask.shape:
        raise ValueError(
            f"pred/gt/mask shapes must match, got {pred.shape}, {gt.shape}, {valid_mask.shape}."
        )

    valid = valid_mask & (gt >= 0) & (gt < num_classes)
    pred_valid = pred[valid]
    if pred_valid.size > 0 and (np.any(pred_valid < 0) or np.any(pred_valid >= num_classes)):
        bad_values = np.unique(pred_valid[(pred_valid < 0) | (pred_valid >= num_classes)])[:10]
        raise ValueError(f"Prediction semantic ids outside [0, {num_classes}): {bad_values.tolist()}")

    encoded = num_classes * gt[valid] + pred_valid
    return np.bincount(encoded, minlength=num_classes * num_classes).reshape(
        num_classes, num_classes
    ).astype(np.int64)


def summarize_iou(
    hist: np.ndarray,
    free_id: int,
) -> tuple[np.ndarray, float, float]:
    hist = np.asarray(hist, dtype=np.float64)
    if hist.ndim != 2 or hist.shape[0] != hist.shape[1]:
        raise ValueError(f"confusion matrix must be square, got {hist.shape}")

    intersection = np.diag(hist)
    gt_count = hist.sum(axis=1)
    pred_count = hist.sum(axis=0)
    union = gt_count + pred_count - intersection
    per_class_iou = np.divide(
        intersection,
        union,
        out=np.full_like(intersection, np.nan, dtype=np.float64),
        where=union > 0,
    )
    per_class_iou = np.where(gt_count > 0, per_class_iou, np.nan)

    non_free = np.ones(hist.shape[0], dtype=np.bool_)
    non_free[free_id] = False
    spatial_intersection = hist[non_free][:, non_free].sum()
    spatial_gt = hist[non_free, :].sum()
    spatial_pred = hist[:, non_free].sum()
    spatial_union = spatial_gt + spatial_pred - spatial_intersection
    iou = float(spatial_intersection / spatial_union) if spatial_union > 0 else float("nan")
    non_free_iou = per_class_iou[non_free]
    miou = float(np.nanmean(non_free_iou)) if np.any(~np.isnan(non_free_iou)) else float("nan")
    return per_class_iou, iou, miou


def format_class_iou(per_class_iou: np.ndarray, class_names: Sequence[str], class_ids) -> str:
    lines = []
    for class_id in class_ids:
        value = per_class_iou[class_id]
        text = "nan" if np.isnan(value) else f"{value * 100:.2f}"
        lines.append(f"{class_names[class_id]} - IoU = {text}")
    return "\n".join(lines)
