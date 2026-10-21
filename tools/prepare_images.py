"""Resize source camera images to the resolution expected by PanoBench."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["kitti360", "carlaocc"])
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path(
        "datasets/splits/panobench_kitti360/labels.txt"))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional number of images for a preparation check.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def image_pairs(args):
    if args.dataset == "kitti360":
        for line in args.manifest.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            label = Path(line.strip())
            if (len(label.parts) != 2 or label.is_absolute()
                    or ".." in label.parts or label.suffix != ".npz"):
                raise ValueError(f"Invalid frame entry: {line}")
            relative = Path("data_2d_raw") / label.parent / "image_00"
            name = label.with_suffix(".png").name
            yield (args.source / relative / "data_rect" / name,
                   args.destination / relative / "data_192x640" / name)
    else:
        for scene in sorted(args.source.glob("Town*_Seq*")):
            for camera in sorted((scene / "rgb").glob("image_0[0-5]")):
                for source in sorted(camera.glob("*.png")):
                    yield source, args.destination / source.relative_to(args.source)


def prepare(args, pair):
    source, destination = pair
    if destination.exists() and not args.overwrite:
        existing = cv2.imread(str(destination), cv2.IMREAD_COLOR)
        if existing is None or existing.shape[:2] != (192, 640):
            raise ValueError(f"Existing image is not a readable 640x192 PNG: {destination}")
        return "existing"
    if source.resolve() == destination.resolve():
        raise ValueError(f"Source and destination must differ: {source}")
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing or unreadable source image: {source}")
    if args.dataset == "kitti360":
        # Match the float-image resize/normalization used for the baseline inputs.
        image = cv2.resize(image.astype(np.float32) / 255.0, (640, 192),
                           interpolation=cv2.INTER_LINEAR)
        image = np.rint(((image * 2.0 - 1.0) * 0.5 + 0.5) * 255.0)
        image = image.clip(0, 255).astype(np.uint8)
    else:
        image = cv2.resize(image, (640, 192), interpolation=cv2.INTER_AREA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".partial.png")
    if not cv2.imwrite(str(temporary), image):
        raise OSError(f"Could not write image: {temporary}")
    temporary.replace(destination)
    return "written"


def main():
    args = parse_args()
    if args.workers < 1 or (args.limit is not None and args.limit < 1):
        raise ValueError("workers and limit must be positive")
    if not args.source.is_dir():
        raise FileNotFoundError(args.source)
    cv2.setNumThreads(1)
    pairs = image_pairs(args)
    if args.limit is not None:
        pairs = islice(pairs, args.limit)
    counts = {"written": 0, "existing": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        while batch := list(islice(pairs, 1024)):
            for result in pool.map(lambda pair: prepare(args, pair), batch):
                counts[result] += 1
    if not sum(counts.values()):
        raise FileNotFoundError(f"No matching source images under {args.source}")
    print(counts)


if __name__ == "__main__":
    main()
