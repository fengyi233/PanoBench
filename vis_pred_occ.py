"""Visualize occupancy predictions for any PanoBench eval model.

Default Pred path is ``outputs/vis_eval/<model>/<dataset>_<variant>/<seq>_<frame>.png``.
With ``--save-gt``, GT goes to ``outputs/vis_eval/gt/<dataset>_<variant>/<seq>_<frame>.png``.
Voxels are masked by
``valid_mask`` unless ``--no-mask``. Camera is a 45-degree downward view at
1000x1000.

Examples::

    python vis_pred_occ.py \\
      --config-name experiments/sparseocc/panobench_carlaocc \\
      --checkpoint outputs/experiments/sparseocc/panobench_carlaocc/vs_0.4_surround_view/20260825_165331/epoch_20.pth \\
      --occ-variant vs_0.4_surround_view \\
      --scene Town10HD_Opt_Seq00 --frame 0200

    python vis_pred_occ.py ... --scene Town10HD_Opt_Seq00 --start-frame 200 --num-frames 10 --save-gt

    python vis_pred_occ.py ... --index 0 --interactive --no-save
"""

from __future__ import annotations

import argparse
from pathlib import Path

import hydra  # noqa: F401
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from mmdet3d.registry import DATASETS, MODELS
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import load_checkpoint
from mmengine.utils import import_modules_from_strings
from omegaconf import OmegaConf

from utils.data_converter import decode_global_id
from utils.labels import get_dataset_meta
from utils.panoptic_occupancy import PanopticOccupancy
from utils.vis_utils import CameraSetting, vis_sem_occ

EVAL_CAMERA = CameraSetting.from_position(
    image_size=(1000, 1000),
    focal_length=250.0,
    cam_position=(-25.0, 0.0, 20.0),
    cam_pitch=45.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config-name", required=True, help="Hydra experiment config.")
    parser.add_argument(
        "--occ-variant",
        default=None,
        help=(
            "Occupancy variant (e.g. vs_0.4_surround_view). "
            "Omit, or pass default/none, when the dataset has no variant "
            "(KITTI-360). Inferred from vs_* or default in the checkpoint path."
        ),
    )
    parser.add_argument("--horizon", type=int, default=0, help="Forecast horizon index (0=current); use 1/2/3 for future predictions.")
    parser.add_argument("--index", type=int, default=None, help="Dataset index.")
    parser.add_argument("--scene", default=None, help="Exact scene_name.")
    parser.add_argument("--frame", default=None, help="Exact frame_token (e.g. 0200).")
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Keep frames with integer token >= this value (requires --scene).",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=None,
        help="Max frames after --start-frame / scene start.",
    )
    parser.add_argument("--split", default="val", choices=("val", "train"))
    parser.add_argument(
        "--out-dir",
        default="outputs/vis_eval",
        help="Root directory. Files go to <out-dir>/<model>/<dataset>_<variant>/<seq>_<frame>.png",
    )
    parser.add_argument(
        "--save-gt",
        action="store_true",
        help="Also save GT to <out-dir>/gt/<dataset>_<variant>/<seq>_<frame>.png",
    )
    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Do not filter Pred by valid_mask (GT is still masked).",
    )
    parser.add_argument("--interactive", action="store_true", help="Open Open3D on the first frame.")
    parser.add_argument("--no-save", action="store_true", help="Do not write PNG files.")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--show-axis", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


# train.py writes work_dir as ``default`` when ``data.occ_variant`` is null.
_NULL_OCC_VARIANTS = {"", "default", "none", "null"}


def normalize_occ_variant(value) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text.lower() in _NULL_OCC_VARIANTS:
        return None
    return text


def infer_occ_variant(checkpoint: str, explicit_variant: str | None) -> str | None:
    if explicit_variant is not None:
        return normalize_occ_variant(explicit_variant)
    path = Path(checkpoint).resolve()
    for part in (path.parent.name, path.parent.parent.name):
        if part.startswith("vs_"):
            return part
        if part.lower() in _NULL_OCC_VARIANTS:
            return None
    return None


def occ_variant_path_part(occ_variant) -> str:
    return "default" if occ_variant is None else safe_path_part(occ_variant)


def config_short_name(config_name: str) -> str:
    parts = [p for p in config_name.replace("\\", "/").split("/") if p]
    if parts and parts[0] == "experiments" and len(parts) >= 2:
        return parts[1]
    return parts[-1] if parts else "unknown"


def safe_path_part(value) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(value))


def frame_png_path(
    out_root: Path,
    model_name: str,
    dataset_type: str,
    occ_variant: str | None,
    scene_name: str,
    frame_token: str,
) -> Path:
    dataset_variant = f"{safe_path_part(dataset_type)}_{occ_variant_path_part(occ_variant)}"
    seq_frame = f"{safe_path_part(scene_name)}_{safe_path_part(frame_token)}"
    return Path(out_root) / safe_path_part(model_name) / dataset_variant / f"{seq_frame}.png"


def load_hydra_cfg(config_name: str, occ_variant: str | None) -> Config:
    overrides = []
    if occ_variant is not None:
        overrides.append(f"data.occ_variant={occ_variant}")
    with initialize_config_dir(config_dir=str(Path("configs").resolve()), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    for key in ("data", "run", "checkpoint", "pred_occ_save_dir"):
        cfg_dict.pop(key, None)
    return Config(cfg_dict)


def init_project_modules(cfg: Config) -> None:
    init_default_scope(cfg.get("default_scope", "mmdet3d"))
    if cfg.get("custom_imports") is not None:
        import_modules_from_strings(**cfg["custom_imports"])


def build_model(cfg: Config, checkpoint: str, device: torch.device):
    init_project_modules(cfg)
    model = MODELS.build(cfg.model)
    load_checkpoint(
        model,
        checkpoint,
        map_location="cpu",
        revise_keys=[(r"^module\.", ""), (r"^pts_bbox_head\.", "occ_head.")],
        strict=True,
    )
    model.to(device)
    model.eval()
    return model


def build_dataset(cfg: Config, split: str):
    init_project_modules(cfg)
    ds_cfg = cfg.val_dataloader.dataset.copy()
    ds_cfg.test_mode = split != "train"
    return DATASETS.build(ds_cfg)


def raw_infos(dataset) -> list:
    infos = getattr(dataset, "_raw_infos", None)
    if infos is not None:
        return infos
    return dataset.data_list


def _token_sort_key(token: str):
    try:
        return (0, int(token))
    except ValueError:
        return (1, str(token))


def _token_equal(token: str, target: str) -> bool:
    if token == target:
        return True
    try:
        return int(token) == int(target)
    except ValueError:
        return False


def resolve_indices(dataset, args: argparse.Namespace) -> list[int]:
    infos = raw_infos(dataset)
    if args.index is not None:
        if args.scene is not None or args.frame is not None:
            raise ValueError("--index cannot be combined with --scene/--frame.")
        if args.index < 0 or args.index >= len(dataset):
            raise IndexError(f"index {args.index} is outside dataset length {len(dataset)}")
        return [args.index]
    if args.scene is None:
        raise ValueError("Specify --index or --scene.")

    scene_hits = [i for i, info in enumerate(infos) if info["scene_name"] == args.scene]
    if not scene_hits:
        names = sorted({info["scene_name"] for info in infos})
        close = [name for name in names if args.scene in name]
        hint = f" Close matches: {close[:12]}" if close else f" Available (first 20): {names[:20]}"
        raise ValueError(f"No scene named {args.scene!r}.{hint}")

    if args.frame is not None:
        if args.start_frame is not None or args.num_frames is not None:
            raise ValueError("--frame cannot be combined with --start-frame/--num-frames.")
        matched = [i for i in scene_hits if _token_equal(str(infos[i]["frame_token"]), str(args.frame))]
        if not matched:
            tokens = [str(infos[i]["frame_token"]) for i in scene_hits]
            raise ValueError(
                f"No frame {args.frame!r} in {args.scene}. "
                f"tokens e.g. {tokens[:16]}"
            )
        return matched

    rows = [(i, str(infos[i]["frame_token"])) for i in scene_hits]
    rows.sort(key=lambda item: _token_sort_key(item[1]))
    if args.start_frame is not None:
        kept = []
        for i, token in rows:
            try:
                token_i = int(token)
            except ValueError as exc:
                raise ValueError("--start-frame requires integer frame tokens.") from exc
            if token_i >= args.start_frame:
                kept.append((i, token))
        rows = kept
    if args.horizon < 0:
        raise ValueError("--horizon must be nonnegative")
    if args.num_frames is not None:
        rows = rows[: args.num_frames]
    if not rows:
        raise ValueError(f"No frames left in {args.scene} after start/num-frames filters.")
    return [i for i, _ in rows]


def move_inputs_to_device(inputs: dict, device: torch.device) -> dict:
    moved = {}
    for key, value in inputs.items():
        moved[key] = value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
    return moved


def data_field(data, key: str):
    if isinstance(data, dict):
        return data[key]
    return getattr(data, key)


def pred_to_sparse_semantic(pred_occ, free_id: int, horizon: int = 0) -> np.ndarray:
    occ_loc = data_field(pred_occ, "occ_loc")
    semantics = data_field(pred_occ, "semantics")
    if isinstance(occ_loc, (list, tuple)):
        if not 0 <= horizon < len(occ_loc):
            raise ValueError(f"Invalid forecast horizon {horizon}")
        occ_loc, semantics = occ_loc[horizon], semantics[horizon]
    elif horizon != 0:
        raise ValueError("Non-forecast models only support --horizon 0")
    occ_loc = np.asarray(
        occ_loc.detach().cpu().numpy() if hasattr(occ_loc, "detach") else occ_loc
    ).reshape(-1, 3)
    semantics = np.asarray(
        semantics.detach().cpu().numpy() if hasattr(semantics, "detach") else semantics
    ).reshape(-1)
    keep = semantics != int(free_id)
    if not keep.any():
        return np.zeros((0, 4), dtype=np.int64)
    return np.concatenate([occ_loc[keep], semantics[keep, None]], axis=1)


def gt_to_sparse_semantic(occ_path: Path, free_id: int, apply_mask: bool) -> tuple[np.ndarray, np.ndarray]:
    sample = PanopticOccupancy.load(occ_path)
    occupancy = sample.masked_occupancy() if apply_mask else sample.occupancy
    semantics, _ = decode_global_id(occupancy[:, 3])
    keep = semantics != int(free_id)
    occupancy = occupancy[keep]
    semantics = semantics[keep]
    sparse = np.concatenate(
        [occupancy[:, :3].astype(np.int64, copy=False), semantics.astype(np.int64, copy=False)[:, None]],
        axis=1,
    )
    return sparse, np.asarray(sample.valid_mask, dtype=np.bool_)


def filter_sparse_by_valid_mask(sparse_occ: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if sparse_occ.size == 0:
        return sparse_occ
    coords = sparse_occ[:, :3].astype(np.int64)
    x_size, y_size, z_size = valid_mask.shape
    in_bounds = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < x_size)
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < y_size)
        & (coords[:, 2] >= 0)
        & (coords[:, 2] < z_size)
    )
    coords = coords[in_bounds]
    semantics = sparse_occ[in_bounds, 3]
    keep = valid_mask[coords[:, 0], coords[:, 1], coords[:, 2]]
    if not keep.any():
        return np.zeros((0, 4), dtype=np.int64)
    return np.concatenate([coords[keep], semantics[keep, None]], axis=1)


def render_occ(
    occ: np.ndarray,
    dataset_type: str,
    occ_variant: str | None,
    show_axis: bool,
    save_path: Path | None,
) -> None:
    if occ.shape[0] == 0:
        if save_path is not None:
            from PIL import Image

            save_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (1000, 1000), (255, 255, 255)).save(save_path)
        return
    vis_sem_occ(
        occ,
        occ_type=dataset_type,
        occ_meta=get_dataset_meta(dataset_type, occ_variant),
        camera_setting=EVAL_CAMERA,
        show_axis=show_axis,
        save_path=save_path,
    )


def main() -> None:
    args = parse_args()
    if args.interactive is False and args.no_save:
        raise ValueError("Nothing to do: pass --interactive and/or omit --no-save.")
    if args.num_frames is not None and args.num_frames <= 0:
        raise ValueError("--num-frames must be positive.")

    checkpoint = str(Path(args.checkpoint).resolve())
    occ_variant = infer_occ_variant(checkpoint, args.occ_variant)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    cfg = load_hydra_cfg(args.config_name, occ_variant)
    dataset = build_dataset(cfg, split=args.split)
    occ_variant = getattr(dataset, "occ_variant", occ_variant)
    model = build_model(cfg, checkpoint, device)
    indices = resolve_indices(dataset, args)
    infos = raw_infos(dataset)

    model_name = config_short_name(args.config_name)
    out_root = Path(args.out_dir)
    if hasattr(dataset, "future_frames"):
        if args.horizon >= len(dataset.future_frames):
            raise ValueError("--horizon exceeds available future frames")
        out_root = out_root / f"horizon_{args.horizon}"
    elif args.horizon != 0:
        raise ValueError("--horizon requires a forecasting dataset")
    dataset_type = getattr(dataset, "DATASET_TYPE", None) or infos[indices[0]]["dataset_type"]
    if not args.no_save:
        example = frame_png_path(
            out_root,
            model_name,
            dataset_type,
            occ_variant,
            infos[indices[0]]["scene_name"],
            infos[indices[0]]["frame_token"],
        )
        print(f"Saving {len(indices)} frame(s) under {example.parent}/<seq>_<frame>.png", flush=True)
    else:
        print(f"Interactive only, {len(indices)} frame(s)", flush=True)

    if args.interactive and len(indices) > 1:
        print("Interactive window will open for the first frame only.", flush=True)

    for step, index in enumerate(indices, 1):
        info = infos[index]
        scene_name = info["scene_name"]
        frame_token = str(info["frame_token"])
        data_info = dataset.get_data_info(index)
        dataset_type = data_info["dataset_type"]
        meta = get_dataset_meta(dataset_type, occ_variant)
        pred_path = None if args.no_save else frame_png_path(
            out_root, model_name, dataset_type, occ_variant, scene_name, frame_token)
        gt_path = None if args.no_save or not args.save_gt else frame_png_path(
            out_root, "gt", dataset_type, occ_variant, scene_name, frame_token)

        if args.skip_existing and pred_path is not None and pred_path.is_file():
            if gt_path is None or gt_path.is_file():
                print(f"[skip] {pred_path}", flush=True)
                continue

        sample = dataset[index]
        inputs = move_inputs_to_device(sample["inputs"], device)
        with torch.no_grad():
            outputs = model(inputs, data_samples=[sample["data_samples"]], mode="predict")
        pred = pred_to_sparse_semantic(outputs[0].pred_occ, free_id=meta.free_id, horizon=args.horizon)
        gt_path_source = data_info["future_occ_paths"][args.horizon] if "future_occ_paths" in data_info else data_info["occ_path"]
        gt, valid_mask = gt_to_sparse_semantic(Path(gt_path_source), meta.free_id, apply_mask=True)
        if not args.no_mask:
            pred = filter_sparse_by_valid_mask(pred, valid_mask)

        print(
            f"[{step}/{len(indices)}] {scene_name} frame={frame_token} "
            f"pred={len(pred)} gt={len(gt)}",
            flush=True,
        )
        render_occ(pred, dataset_type, occ_variant, args.show_axis, pred_path)
        if gt_path is not None:
            render_occ(gt, dataset_type, occ_variant, args.show_axis, gt_path)

        if args.interactive and step == 1:
            render_occ(pred, dataset_type, occ_variant, args.show_axis, save_path=None)


if __name__ == "__main__":
    main()
