"""Check released PanoBench label files, packed masks and frame coverage."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.labels import DATASET_META, DATASET_VARIANTS  # noqa: E402
from utils.mappings import load_mapping  # noqa: E402

FIELDS = {"occupancy", "valid_mask", "volume_size", "voxel_size", "voxel_origin"}
META_TABLE: dict = {}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "data/canonical_occ",
                        help="Parent directory of the three PanoBench label subsets.")
    parser.add_argument("--dataset", choices=tuple(DATASET_VARIANTS),
                        help="Check one subset; defaults to all three.")
    parser.add_argument("--variant", help="Check one variant of the selected dataset.")
    parser.add_argument("--metadata-root", type=Path, default=ROOT / "data",
                        help="Directory containing CarlaOccV1Preprocessed/splits/.")
    parser.add_argument("--kitti-manifest", type=Path,
                        default=ROOT / "datasets/splits/panobench_kitti360/labels.txt")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit", type=int,
                        help="Check at most this many evenly spaced files per variant; coverage is always complete.")
    parser.add_argument("--report", type=Path,
                        default=ROOT / "outputs/dataset_checks/report.json")
    args = parser.parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        parser.error("--workers and --limit must be positive")
    if args.variant and (not args.dataset or args.variant not in DATASET_VARIANTS[args.dataset]):
        parser.error("--variant must be one of the selected dataset's variants; omit it for KITTI360")
    return args


def variant_root(root: Path, gt_type: str, variant) -> Path:
    root = root / gt_type
    return root if variant is None else root / variant


def list_npz(root: Path) -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    if not root.is_dir():
        return found
    for scene_dir in sorted(root.iterdir()):
        if scene_dir.is_dir():
            found[scene_dir.name] = {p.stem for p in scene_dir.glob("*.npz")}
    return found


def expected_frames(gt_type: str, metadata_root: Path, kitti_manifest: Path) -> set[tuple[str, str]]:
    if gt_type == "panobench_nuscenes":
        mapping = load_mapping(ROOT / "datasets/mappings/panobench_nuscenes.json")
        return {(scene, token) for scene, tokens in mapping["gt_files"].items() for token in tokens}
    pairs = []
    if gt_type == "panobench_kitti360":
        for line in kitti_manifest.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            frame = Path(line.strip())
            if len(frame.parts) != 2 or frame.is_absolute() or ".." in frame.parts or frame.suffix != ".npz":
                raise ValueError(f"Invalid KITTI360 frame entry: {line}")
            pairs.append((frame.parent.name, frame.stem))
    else:
        for split in ("train", "val", "test"):
            path = metadata_root / "CarlaOccV1Preprocessed/splits" / f"{split}.txt"
            for line in path.read_text().splitlines():
                if not line.strip() or line.lstrip().startswith("#"):
                    continue
                town, sequence, frame = line.split()
                town_name = "Town10HD_Opt" if town == "10" else f"Town{town}_Opt"
                pairs.append((f"{town_name}_Seq{sequence}", f"{int(frame):04d}"))
    if not pairs or len(set(pairs)) != len(pairs):
        raise ValueError(f"{gt_type} frame lists are empty or contain duplicates")
    return set(pairs)


def meta_payload(gt_type: str, variant) -> dict:
    meta = DATASET_META[(gt_type, variant)]
    return {
        "volume_size": meta.volume_size,
        "voxel_size": float(meta.voxel_size),
        "voxel_origin": tuple(float(v) for v in meta.voxel_origin),
        "num_classes": int(meta.num_classes),
        "free_id": int(meta.free_id),
        "thing_ids": np.asarray(sorted(meta.thing_ids), dtype=np.int64),
    }


def _init(table: dict) -> None:
    global META_TABLE
    META_TABLE = table


def check_batch(jobs: list[tuple[str, str]]) -> dict:
    counts: Counter = Counter()
    examples = []
    empty = 0
    checked = 0
    for key, path in jobs:
        checked += 1
        meta = META_TABLE[key]
        try:
            with np.load(path, allow_pickle=False) as data:
                if set(data.files) != FIELDS:
                    _add(counts, examples, path, f"fields={tuple(data.files)}")
                    continue
                occ = np.asarray(data["occupancy"])
                mask = np.asarray(data["valid_mask"])
                volume = np.asarray(data["volume_size"])
                voxel_size = np.asarray(data["voxel_size"])
                origin = np.asarray(data["voxel_origin"]).reshape(-1)
        except Exception as exc:  # noqa: BLE001
            _add(counts, examples, path, f"load={type(exc).__name__}")
            continue

        if occ.dtype != np.uint16 or occ.ndim != 2 or occ.shape[1] != 4:
            _add(counts, examples, path, f"occ={occ.dtype}{occ.shape}")
            continue
        if not occ.flags.f_contiguous:
            _add(counts, examples, path, "occ_not_fortran")
        if volume.dtype != np.int32 or volume.shape != (3,) or (volume <= 0).any():
            _add(counts, examples, path, f"volume_dtype={volume.dtype}")
            continue
        if voxel_size.dtype != np.float32 or voxel_size.shape != ():
            _add(counts, examples, path, f"voxel_size_dtype={voxel_size.dtype}{voxel_size.shape}")
            continue
        if origin.dtype != np.float32 or origin.shape != (3,):
            _add(counts, examples, path, f"origin_dtype={origin.dtype}{origin.shape}")
            continue
        if mask.dtype != np.uint8 or mask.ndim != 1:
            _add(counts, examples, path, f"mask={mask.dtype}{mask.shape}")
            continue

        vol = tuple(int(v) for v in volume)
        if vol != tuple(meta["volume_size"]):
            _add(counts, examples, path, f"volume_size={vol}")
        if not np.isclose(float(voxel_size), meta["voxel_size"]):
            _add(counts, examples, path, f"voxel_size={float(voxel_size)}")
        if not np.allclose(origin, np.asarray(meta["voxel_origin"], dtype=np.float32)):
            _add(counts, examples, path, f"origin={tuple(float(v) for v in origin)}")

        num_voxels = int(np.prod(vol)) if len(vol) == 3 else 0
        expected_mask = (num_voxels + 7) // 8
        if mask.shape != (expected_mask,):
            _add(counts, examples, path, f"mask_len={int(mask.shape[0])}")
        else:
            rem = num_voxels % 8
            if rem and (int(mask[-1]) & ((1 << (8 - rem)) - 1)):
                _add(counts, examples, path, "mask_tail_bits")
            if not mask.any():
                _add(counts, examples, path, "mask_all_zero")

        n = int(occ.shape[0])
        if n == 0:
            empty += 1
            continue

        coords = occ[:, :3].astype(np.int32, copy=False)
        gid = occ[:, 3].astype(np.int32, copy=False)
        if (
            (coords < 0).any()
            or (coords[:, 0] >= vol[0]).any()
            or (coords[:, 1] >= vol[1]).any()
            or (coords[:, 2] >= vol[2]).any()
        ):
            _add(counts, examples, path, "coords_out_of_range")
        key64 = (
            coords[:, 0].astype(np.uint32) << np.uint32(20)
            | coords[:, 1].astype(np.uint32) << np.uint32(10)
            | coords[:, 2].astype(np.uint32)
        )
        key64.sort()
        if n > 1 and np.any(np.diff(key64) == 0):
            _add(counts, examples, path, "duplicate_voxels")

        sem = gid // 1000
        inst = gid % 1000
        bad_sem = (sem < 0) | (sem >= meta["num_classes"]) | (sem == meta["free_id"])
        if bad_sem.any():
            _add(counts, examples, path, f"bad_semantic={np.unique(sem[bad_sem])[:6].tolist()}")
        is_thing = np.isin(sem, meta["thing_ids"])
        if np.any((~is_thing) & (inst != 0) & (~bad_sem)):
            _add(counts, examples, path, "stuff_instance_nonzero")
        if np.any(is_thing & (inst == 0)):
            _add(counts, examples, path, "thing_instance_zero")
    return {"checked": checked, "empty": empty, "counts": dict(counts), "examples": examples}


def _add(counts: Counter, examples: list, path: str, issue: str) -> None:
    tag = issue.split("=")[0]
    counts[tag] += 1
    if len(examples) < 8 and tag != "empty_occupancy":
        examples.append({"file": path, "issue": issue})


def coverage(root: Path, found: dict[str, set[str]], expected_pairs: set[tuple[str, str]]) -> dict:
    entry = {
        "root_exists": root.is_dir(),
        "scenes": len(found),
        "files": sum(len(v) for v in found.values()),
        "scene_size_min": min((len(v) for v in found.values()), default=0),
        "scene_size_max": max((len(v) for v in found.values()), default=0),
    }
    have_pairs = {(scene, token) for scene, stems in found.items() for token in stems}
    missing = expected_pairs - have_pairs
    extra = have_pairs - expected_pairs
    entry["expected_frames"] = len(expected_pairs)
    entry["missing_frames"] = len(missing)
    entry["extra_frames"] = len(extra)
    entry["missing_examples"] = [f"{s}/{t}" for s, t in sorted(missing)[:5]]
    entry["extra_examples"] = [f"{s}/{t}" for s, t in sorted(extra)[:5]]
    return entry


def main() -> int:
    args = parse_args()
    table = {}
    structural = {}
    jobs_by_key = {}
    datasets = (args.dataset,) if args.dataset else tuple(DATASET_VARIANTS)
    for gt_type in datasets:
        variants = (args.variant,) if args.variant else DATASET_VARIANTS[gt_type]
        try:
            expected = expected_frames(gt_type, args.metadata_root, args.kitti_manifest)
            metadata_error = None
        except (OSError, ValueError, KeyError, TypeError) as exc:
            expected = set()
            metadata_error = str(exc)
        for variant in variants:
            key = f"{gt_type}/{variant or '-'}"
            table[key] = meta_payload(gt_type, variant)
            root = variant_root(args.root, gt_type, variant)
            found = list_npz(root)
            structural[key] = coverage(root, found, expected)
            if metadata_error is not None:
                structural[key]["metadata_error"] = metadata_error
            jobs = [
                (key, str(root / scene / f"{stem}.npz"))
                for scene, stems in found.items()
                for stem in sorted(stems)
            ]
            if args.limit is not None and len(jobs) > args.limit:
                indices = np.linspace(0, len(jobs) - 1, args.limit, dtype=int)
                jobs = [jobs[index] for index in indices]
            jobs_by_key[key] = jobs
            print("COVER", key, json.dumps(structural[key], ensure_ascii=False), flush=True)

    # JSON cannot store ndarrays. Convert before shipping to workers via copy.
    worker_table = {}
    for key, meta in table.items():
        worker_table[key] = dict(meta)
        worker_table[key]["thing_ids"] = np.asarray(meta["thing_ids"], dtype=np.int64)

    content = {}
    for key, jobs in jobs_by_key.items():
        batches = [jobs[i:i + 24] for i in range(0, len(jobs), 24)]
        counts: Counter = Counter()
        examples = []
        checked = 0
        empty = 0
        if batches:
            with ProcessPoolExecutor(max_workers=args.workers, initializer=_init, initargs=(worker_table,)) as pool:
                for result in pool.map(check_batch, batches, chunksize=1):
                    checked += result["checked"]
                    empty += result["empty"]
                    counts.update(result["counts"])
                    for item in result["examples"]:
                        if len(examples) < 12:
                            examples.append(item)
        content[key] = {
            "checked": checked,
            "empty_occupancy": empty,
            "issue_counts": dict(counts),
            "examples": examples,
        }
        print("CONTENT", key, json.dumps({k: content[key][k] for k in ("checked", "empty_occupancy", "issue_counts")}, ensure_ascii=False), flush=True)

    passed = all(
        item["root_exists"] and item["expected_frames"] > 0
        and item["missing_frames"] == 0 and item["extra_frames"] == 0
        and "metadata_error" not in item
        for item in structural.values()
    ) and all(not item["issue_counts"] for item in content.values())
    out = {
        "passed": passed,
        "content_check": "sampled" if args.limit is not None else "full",
        "limit_per_variant": args.limit,
        "root": str(args.root),
        "structural": structural,
        "content": content,
    }
    report_path = args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", report_path)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
