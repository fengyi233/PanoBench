"""Camera-ray occupancy metrics."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.cpp_extension import load

from .panoptic import panoptic_stats
from .semantic import semantic_confusion_matrix


@dataclass(frozen=True)
class CameraRays:
    origins: torch.Tensor
    points: torch.Tensor
    tindex: torch.Tensor


@dataclass(frozen=True)
class RayHits:
    semantic: np.ndarray
    instance: np.ndarray
    depth: np.ndarray


def build_camera_rays(
    occ2img: np.ndarray,
    image_shape: tuple[int, int],
    stride: int,
) -> CameraRays:
    occ2img = np.asarray(occ2img, dtype=np.float64)
    if occ2img.ndim != 3 or occ2img.shape[-2:] != (4, 4):
        raise ValueError(f"occ2img must be [V, 4, 4], got {occ2img.shape}.")
    if stride <= 0:
        raise ValueError(f"ray stride must be positive, got {stride}.")

    height, width = (int(image_shape[0]), int(image_shape[1]))
    ys = np.arange(0, height, stride, dtype=np.float64) + 0.5
    xs = np.arange(0, width, stride, dtype=np.float64) + 0.5
    yy, xx = np.meshgrid(ys, xs, indexing="ij")
    pixels = np.stack(
        (
            xx.reshape(-1),
            yy.reshape(-1),
            np.ones(xx.size, dtype=np.float64),
            np.ones(xx.size, dtype=np.float64),
        ),
        axis=1,
    )

    origins = []
    points = []
    tindex = []
    for cam_idx, matrix in enumerate(occ2img):
        inv = np.linalg.inv(matrix)
        origin_h = inv @ np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        origin = origin_h[:3] / origin_h[3]
        points_h = pixels @ inv.T
        ray_points = points_h[:, :3] / points_h[:, 3:4]
        direction = ray_points - origin[None, :]
        direction_norm = np.linalg.norm(direction, axis=1, keepdims=True)
        if np.any(direction_norm <= 0):
            raise ValueError("occ2img produced zero-length camera rays.")
        direction = direction / direction_norm
        points.append(origin[None, :] + direction)
        origins.append(origin)
        tindex.append(np.full((pixels.shape[0],), cam_idx, dtype=np.float32))

    return CameraRays(
        origins=torch.from_numpy(np.stack(origins, axis=0).astype(np.float32)),
        points=torch.from_numpy(np.concatenate(points, axis=0).astype(np.float32)),
        tindex=torch.from_numpy(np.concatenate(tindex, axis=0).astype(np.float32)),
    )


def _cudart_ldflags() -> list[str]:
    # Conda nvcc keeps libcudart in env/lib; torch JIT looks in env/lib64.
    cuda_home = Path(os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or "/usr/local/cuda")
    candidates = [
        cuda_home / "lib64",
        cuda_home / "targets/x86_64-linux/lib",
        Path("/usr/local/cuda-11.8/targets/x86_64-linux/lib"),
        Path(sys.prefix) / "lib",
    ]
    flags = []
    for path in candidates:
        if path.is_dir() and any(path.glob("libcudart.so*")):
            flags.append(f"-L{path}")
    return flags


def init_dvr(dvr_dir: str | Path):
    return load(
        "dvr",
        sources=[f"{dvr_dir}/dvr.cpp", f"{dvr_dir}/dvr.cu"],
        verbose=True,
        extra_cuda_cflags=["-allow-unsupported-compiler"],
        extra_ldflags=_cudart_ldflags(),
    )


def render_camera_ray_hits(
    semantic: np.ndarray,
    instance: np.ndarray,
    rays: CameraRays,
    dvr,
    point_cloud_range: tuple[float, float, float, float, float, float],
    voxel_size: float,
    free_id: int,
) -> RayHits:
    if not torch.cuda.is_available():
        raise RuntimeError("Ray metrics require CUDA for the DVR renderer.")

    semantic_t = torch.as_tensor(np.asarray(semantic, dtype=np.int64), dtype=torch.long)
    instance_t = torch.as_tensor(np.asarray(instance, dtype=np.int64), dtype=torch.long)
    if semantic_t.shape != instance_t.shape:
        raise ValueError(
            f"semantic and instance shapes must match, got {semantic_t.shape} "
            f"and {instance_t.shape}."
        )

    offset = torch.tensor(point_cloud_range[:3], dtype=torch.float32)
    origin_render = ((rays.origins - offset[None, :]) / float(voxel_size))[None]
    points_render = ((rays.points - offset[None, :]) / float(voxel_size))[None]
    tindex = rays.tindex[None].contiguous()
    sigma = (semantic_t != int(free_id)).float().permute(2, 1, 0)[None, None].contiguous()
    grid = [1, semantic_t.shape[2], semantic_t.shape[1], semantic_t.shape[0]]

    pred_dist, _, coord_index = dvr.render_forward(
        sigma.cuda(),
        origin_render.cuda().contiguous(),
        points_render.cuda().contiguous(),
        tindex.cuda().contiguous(),
        grid,
        "test",
    )

    depth = (pred_dist[0].cpu() * float(voxel_size)).float()
    coords = coord_index[0].long().cpu()
    max_coord = torch.tensor(semantic_t.shape, dtype=torch.long) - 1
    coords = torch.maximum(torch.minimum(coords, max_coord[None, :]), torch.zeros_like(coords))

    ray_sem = semantic_t[coords[:, 0], coords[:, 1], coords[:, 2]].cpu()
    ray_inst = instance_t[coords[:, 0], coords[:, 1], coords[:, 2]].cpu()
    hit = (depth >= 0) & (ray_sem != int(free_id))
    ray_sem = torch.where(hit, ray_sem, torch.full_like(ray_sem, int(free_id)))
    ray_inst = torch.where(hit, ray_inst, torch.zeros_like(ray_inst))
    depth = torch.where(hit, depth, torch.full_like(depth, -1.0))

    return RayHits(
        semantic=ray_sem.numpy().astype(np.int64, copy=False),
        instance=ray_inst.numpy().astype(np.int64, copy=False),
        depth=depth.numpy().astype(np.float32, copy=False),
    )


def ray_iou_stats(
    pred: RayHits,
    gt: RayHits,
    thresholds: tuple[float, ...],
    num_classes: int,
    free_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    valid_ray = gt.semantic != int(free_id)
    confusion = semantic_confusion_matrix(pred.semantic, gt.semantic, valid_ray, num_classes)
    same_class = pred.semantic == gt.semantic
    depth_error = np.abs(pred.depth - gt.depth)

    failed = []
    for threshold in thresholds:
        failed_mask = valid_ray & same_class & (depth_error >= float(threshold))
        failed.append(
            np.bincount(gt.semantic[failed_mask], minlength=num_classes).astype(np.int64)
        )
    return confusion, np.stack(failed, axis=0)


def summarize_rayiou(
    confusion: np.ndarray,
    depth_failed: np.ndarray,
    free_id: int,
) -> np.ndarray:
    confusion = np.asarray(confusion, dtype=np.float64)
    depth_failed = np.asarray(depth_failed, dtype=np.float64)
    same_class = np.diag(confusion)
    gt_count = confusion.sum(axis=1)
    pred_count = confusion.sum(axis=0)
    tp = same_class[None, :] - depth_failed
    union = gt_count[None, :] + pred_count[None, :] - tp
    ray_iou = np.divide(
        tp,
        union,
        out=np.full_like(tp, np.nan, dtype=np.float64),
        where=union > 0,
    )
    ray_iou = np.where(gt_count[None, :] > 0, ray_iou, np.nan)

    non_free = np.ones(confusion.shape[0], dtype=np.bool_)
    non_free[int(free_id)] = False
    ray_iou = ray_iou[:, non_free]
    valid = ~np.isnan(ray_iou)
    return np.divide(
        np.nansum(ray_iou, axis=1),
        valid.sum(axis=1),
        out=np.full((ray_iou.shape[0],), np.nan, dtype=np.float64),
        where=valid.sum(axis=1) > 0,
    )


def ray_pq_stats(
    pred: RayHits,
    gt: RayHits,
    thresholds: tuple[float, ...],
    class_ids: tuple[int, ...],
    free_id: int,
) -> dict[float, dict[int, dict]]:
    valid_ray = gt.semantic != int(free_id)
    depth_error = np.abs(pred.depth - gt.depth)
    stats = {}
    for threshold in thresholds:
        depth_ok = (pred.depth >= 0) & (gt.depth >= 0) & (depth_error < float(threshold))
        stats[float(threshold)] = panoptic_stats(
            pred.semantic,
            pred.instance,
            gt.semantic,
            gt.instance,
            valid_ray,
            class_ids,
            intersection_mask=depth_ok,
        )
    return stats
