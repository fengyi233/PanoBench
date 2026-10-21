"""Panoptic occupancy quality statistics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def normalize_panoptic_grid(
    semantic: np.ndarray,
    instance: np.ndarray,
    free_id: int,
    thing_ids: frozenset[int],
    *,
    strict_thing_instance: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize panoptic ids for PQ.

    Free voxels use instance id 0. Stuff classes use a single pseudo instance
    per class with id 1. Thing classes keep their predicted / annotated
    instance ids.
    """
    semantic = np.asarray(semantic, dtype=np.int64)
    instance = np.asarray(instance, dtype=np.int64).copy()
    if semantic.shape != instance.shape:
        raise ValueError(
            f"semantic and instance shapes must match, got {semantic.shape} "
            f"and {instance.shape}."
        )

    free = semantic == free_id
    occupied = ~free
    thing = np.isin(semantic, tuple(thing_ids))
    stuff = occupied & ~thing

    if strict_thing_instance:
        bad = thing & (instance <= 0)
        if np.any(bad):
            bad_classes = sorted(int(v) for v in np.unique(semantic[bad]))
            raise ValueError(f"GT thing classes have instance_id <= 0: {bad_classes}")

    instance[free] = 0
    instance[stuff] = 1
    return semantic, instance


def panoptic_class_stat(
    pred_sem: np.ndarray,
    pred_inst: np.ndarray,
    gt_sem: np.ndarray,
    gt_inst: np.ndarray,
    valid_mask: np.ndarray,
    class_id: int,
    intersection_mask: np.ndarray | None = None,
) -> dict:
    pred_sem = np.asarray(pred_sem, dtype=np.int64)
    pred_inst = np.asarray(pred_inst, dtype=np.int64)
    gt_sem = np.asarray(gt_sem, dtype=np.int64)
    gt_inst = np.asarray(gt_inst, dtype=np.int64)
    valid_mask = np.asarray(valid_mask, dtype=np.bool_)
    if pred_sem.shape != pred_inst.shape or gt_sem.shape != gt_inst.shape:
        raise ValueError("panoptic semantic/instance shapes must match.")
    if pred_sem.shape != gt_sem.shape or gt_sem.shape != valid_mask.shape:
        raise ValueError(
            f"pred/gt/mask shapes must match, got {pred_sem.shape}, "
            f"{gt_sem.shape}, {valid_mask.shape}."
        )

    class_mask = valid_mask & ((gt_sem == class_id) | (pred_sem == class_id))
    if not np.any(class_mask):
        return {"tp": 0, "fp": 0, "fn": 0, "iou": 0.0}

    gt_ids = np.where(gt_sem[class_mask] == class_id, gt_inst[class_mask], 0)
    pred_ids = np.where(pred_sem[class_mask] == class_id, pred_inst[class_mask], 0)
    gt_ids = gt_ids.astype(np.int64, copy=False)
    pred_ids = pred_ids.astype(np.int64, copy=False)

    gt_area = {
        int(seg_id): int((gt_ids == seg_id).sum())
        for seg_id in np.unique(gt_ids)
        if seg_id > 0
    }
    pred_area = {
        int(seg_id): int((pred_ids == seg_id).sum())
        for seg_id in np.unique(pred_ids)
        if seg_id > 0
    }

    pair_mask = (gt_ids > 0) & (pred_ids > 0)
    if intersection_mask is not None:
        local_intersection_mask = np.asarray(intersection_mask, dtype=np.bool_)[class_mask]
        if local_intersection_mask.shape != pair_mask.shape:
            raise ValueError("intersection_mask shape must match panoptic grids.")
        pair_mask &= local_intersection_mask

    intersections = {}
    if np.any(pair_mask):
        pairs = np.stack((gt_ids[pair_mask], pred_ids[pair_mask]), axis=1)
        unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
        for (gt_id, pred_id), count in zip(unique_pairs, counts):
            intersections[(int(gt_id), int(pred_id))] = int(count)

    candidates = []
    for (gt_id, pred_id), inter in intersections.items():
        union = gt_area[gt_id] + pred_area[pred_id] - inter
        iou = inter / union if union > 0 else 0.0
        if iou > 0.5:
            candidates.append((iou, gt_id, pred_id))
    candidates.sort(reverse=True)

    matched_gt = set()
    matched_pred = set()
    iou_sum = 0.0
    for iou, gt_id, pred_id in candidates:
        if gt_id in matched_gt or pred_id in matched_pred:
            continue
        matched_gt.add(gt_id)
        matched_pred.add(pred_id)
        iou_sum += iou

    return {
        "tp": len(matched_gt),
        "fp": len(pred_area) - len(matched_pred),
        "fn": len(gt_area) - len(matched_gt),
        "iou": iou_sum,
    }


def panoptic_stats(
    pred_sem: np.ndarray,
    pred_inst: np.ndarray,
    gt_sem: np.ndarray,
    gt_inst: np.ndarray,
    valid_mask: np.ndarray,
    class_ids: Sequence[int],
    intersection_mask: np.ndarray | None = None,
) -> dict[int, dict]:
    return {
        int(class_id): panoptic_class_stat(
            pred_sem,
            pred_inst,
            gt_sem,
            gt_inst,
            valid_mask,
            int(class_id),
            intersection_mask,
        )
        for class_id in class_ids
    }


def merge_panoptic_stats(results: Sequence[dict[int, dict]], class_ids: Sequence[int]) -> dict[int, dict]:
    merged = {
        int(class_id): {"tp": 0, "fp": 0, "fn": 0, "iou": 0.0}
        for class_id in class_ids
    }
    for stats in results:
        for class_id, stat in stats.items():
            if class_id not in merged:
                continue
            merged[class_id]["tp"] += int(stat["tp"])
            merged[class_id]["fp"] += int(stat["fp"])
            merged[class_id]["fn"] += int(stat["fn"])
            merged[class_id]["iou"] += float(stat["iou"])
    return merged


def summarize_panoptic(stats: dict[int, dict], class_ids: Sequence[int]) -> tuple[float, float, float]:
    pq_values, sq_values, rq_values = [], [], []
    for class_id in class_ids:
        stat = stats[int(class_id)]
        denom = stat["tp"] + 0.5 * stat["fp"] + 0.5 * stat["fn"]
        if denom <= 0:
            continue
        pq_values.append(stat["iou"] / denom)
        sq_values.append(stat["iou"] / stat["tp"] if stat["tp"] > 0 else 0.0)
        rq_values.append(stat["tp"] / denom)

    if not pq_values:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.mean(sq_values)),
        float(np.mean(rq_values)),
        float(np.mean(pq_values)),
    )
