"""Interactive / offline visualization of dense PanopticOccupancy valid_mask.

Usage::

    python vis_valid_mask.py
    python vis_valid_mask.py --path data/canonical_occ/panobench_carlaocc/vs_0.4_surround_view/Town02_Opt_Seq00/0500.npz
    python vis_valid_mask.py --no-free          # occupied only
    python vis_valid_mask.py --save out.png     # offline screenshot
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d

from utils.data_converter import decode_global_id
from utils.labels import get_dataset_meta
from utils.panoptic_occupancy import PanopticOccupancy
from utils.vis_utils import CameraSetting, panoptic_voxel_colors

FREE_VALID = np.array([0.70, 0.82, 0.95])
OCC_INVALID = np.array([0.40, 0.40, 0.40])


def _build_geometries(
    sample: PanopticOccupancy,
    meta,
    show_free: bool,
    show_invalid_occ: bool,
    max_free: int,
):
    mask = sample.valid_mask.astype(bool)
    coords = sample.coords
    sem, inst = decode_global_id(sample.global_id)
    origin = np.asarray(meta.voxel_origin, dtype=np.float32)

    occ_grid = np.zeros(mask.shape, dtype=bool)
    if coords.shape[0]:
        occ_grid[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    occ_valid = (
        mask[coords[:, 0], coords[:, 1], coords[:, 2]]
        if coords.shape[0]
        else np.zeros(0, dtype=bool)
    )

    pts, cols = [], []
    if show_free:
        free_valid = np.argwhere(mask & ~occ_grid)
        if free_valid.shape[0] > max_free:
            free_valid = free_valid[
                np.random.default_rng(0).choice(
                    free_valid.shape[0], max_free, replace=False
                )
            ]
        if free_valid.shape[0]:
            pts.append(
                (free_valid.astype(np.float32) + 0.5) * meta.voxel_size + origin
            )
            cols.append(np.repeat(FREE_VALID[None], free_valid.shape[0], axis=0))

    if coords.shape[0] and occ_valid.any():
        c = coords[occ_valid]
        pts.append((c.astype(np.float32) + 0.5) * meta.voxel_size + origin)
        cols.append(
            panoptic_voxel_colors(
                sem[occ_valid],
                inst[occ_valid],
                meta.color_lut,
                thing_ids=meta.thing_ids,
            )
        )

    if show_invalid_occ and coords.shape[0] and (~occ_valid).any():
        c = coords[~occ_valid]
        pts.append((c.astype(np.float32) + 0.5) * meta.voxel_size + origin)
        cols.append(np.repeat(OCC_INVALID[None], c.shape[0], axis=0))

    if not pts:
        raise RuntimeError("Nothing to visualize for this frame.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.concatenate(pts, axis=0))
    pcd.colors = o3d.utility.Vector3dVector(np.concatenate(cols, axis=0))
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(
        pcd, voxel_size=float(meta.voxel_size)
    )
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
    stats = dict(
        valid_mean=float(mask.mean()),
        free_valid=int((mask & ~occ_grid).sum()),
        occ_valid=int(occ_valid.sum()),
        occ_total=int(coords.shape[0]),
    )
    return voxel_grid, axis, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--path",
        type=Path,
        default=Path(
            # "data/canonical_occ/panobench_carlaocc/"
            # "vs_0.1_surround_view/Town01_Opt_Seq09/0500.npz"
            "data/canonical_occ/panobench_nuscenes/"
            "vs_0.2/scene-0001/1e19d0a5189b46f4b62aa47508f2983e.npz"
        ),
    )
    ap.add_argument("--dataset-type", default="panobench_nuscenes")
    ap.add_argument("--variant", default="vs_0.2")
    ap.add_argument("--no-free", action="store_true")
    ap.add_argument(
        "--show-invalid-occ",
        action="store_true",
        help="Also show occupied voxels with valid_mask=False (gray).",
    )
    ap.add_argument("--max-free", type=int, default=1200000)
    ap.add_argument("--save", type=Path, default=None)
    args = ap.parse_args()

    meta = get_dataset_meta(args.dataset_type, args.variant)
    sample = PanopticOccupancy.load(args.path)
    voxel_grid, axis, stats = _build_geometries(
        sample,
        meta,
        show_free=not args.no_free,
        show_invalid_occ=args.show_invalid_occ,
        max_free=args.max_free,
    )

    print(f"path: {args.path}")
    print("colors: cyan=free&valid  semantic=occupied&valid"
          + ("  gray=occupied&invalid" if args.show_invalid_occ else ""))
    print(
        f"stats: valid={stats['valid_mean']:.1%} "
        f"free_valid={stats['free_valid']} "
        f"occ_valid={stats['occ_valid']}/{stats['occ_total']}"
    )

    camera = CameraSetting.for_dataset(args.dataset_type)
    # Occupancy is Z-up; Open3D's default draw_geometries orbit often starts
    # under the ground plane ("road backside"). Always apply dataset camera.
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name=f"valid_mask  {args.path.parent.name}/{args.path.stem}",
        width=camera.image_size[0] if args.save else 1400,
        height=camera.image_size[1] if args.save else 1000,
        visible=args.save is None,
    )
    vis.add_geometry(voxel_grid)
    vis.add_geometry(axis)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    opt.mesh_show_back_face = True
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(
        camera.get_pinhole_camera_parameters(), allow_arbitrary=True
    )

    if args.save is None:
        print("interactive: elevated Z-up camera; rotate/pan/zoom; q/Esc to close")
        vis.run()
        vis.destroy_window()
        return

    args.save.parent.mkdir(parents=True, exist_ok=True)
    vis.capture_screen_image(str(args.save), do_render=True)
    vis.destroy_window()
    print(f"saved: {args.save}")


if __name__ == "__main__":
    main()
