"""Visualize released PanoBench labels (PanopticOccupancy npz).

Reads :class:`datasets.canonical.CanonicalOccDataset`. PanoBench defaults to
``data/canonical_occ``; ``canonical.mask_key=valid_mask`` keeps only
voxels marked valid in the packed dense mask.

Usage:
    python vis_canonical_occ.py canonical=panobench_carlaocc
    python vis_canonical_occ.py canonical=panobench_carlaocc canonical.variant=vs_0.2_forward_view
    python vis_canonical_occ.py canonical=panobench_nuscenes
    python vis_canonical_occ.py canonical=panobench_kitti360

Output images::
    outputs/vis_occ_gt/canonical_occ/<type>/<variant>/valid_mask/<scene>_<idx>_<token>.png
"""

from pathlib import Path

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from datasets.canonical import CanonicalOccDataset
from utils.vis_utils import CameraSetting, vis_pano_occ


def _safe_path_part(value) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))


def output_path(vis_cfg, dataset: CanonicalOccDataset, record):
    if vis_cfg.interactive or vis_cfg.output_dir is None:
        return None
    out_dir = Path(to_absolute_path(vis_cfg.output_dir))
    store_name = Path(to_absolute_path(str(dataset.cfg.path))).name
    out_dir = out_dir / _safe_path_part(store_name) / _safe_path_part(dataset.dataset_type)
    if dataset.variant is not None:
        out_dir = out_dir / _safe_path_part(dataset.variant)
    if dataset.mask_key:
        out_dir = out_dir / _safe_path_part(dataset.mask_key)
    scene = _safe_path_part(record.scene_name)
    token = _safe_path_part(record.token)
    return out_dir / f"{scene}_{record.frame_idx:06d}_{token}.png"


@hydra.main(version_base=None, config_path="configs", config_name="vis_canonical_occ")
def main(cfg: DictConfig) -> None:
    dataset = CanonicalOccDataset(cfg.canonical)
    vis_cfg = cfg.visualization
    camera = CameraSetting.for_dataset(dataset.dataset_type)

    interval = max(int(vis_cfg.get("vis_interval", 10)), 1)
    records = [r for r in dataset.frames if r.frame_idx % interval == 0]

    source_note = f" from {Path(str(dataset.cfg.path)).name}"
    mask_note = f" (mask={dataset.mask_key})" if dataset.mask_key else ""
    interval_note = f", every {interval} frame(s)" if interval > 1 else ""
    variant_note = f"/{dataset.variant}" if dataset.variant else ""
    print(
        f"Rendering {len(records)}/{len(dataset.frames)} frame(s) from canonical "
        f"{dataset.dataset_type}{variant_note}{source_note}{mask_note}{interval_note}"
    )
    for record in records:
        save_path = output_path(vis_cfg, dataset, record)
        if save_path is not None and save_path.exists():
            continue
        occ = dataset.load(record, return_sparse=True)
        print(
            f"{record.scene_name} frame={record.frame_idx} token={record.token} N={occ.shape[0]}"
        )
        vis_pano_occ(
            occ,
            occ_type=dataset.dataset_type,
            occ_meta=dataset.meta,
            camera_setting=camera,
            show_axis=vis_cfg.show_axis,
            save_path=save_path,
        )


if __name__ == "__main__":
    main()
