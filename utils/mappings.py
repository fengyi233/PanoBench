import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class FrameRecord:
    scene_name: str
    token: str
    path: Path
    frame_idx: int


def load_mapping(path: str | Path) -> dict:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Mapping file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def scene_items(mapping: dict, scene_name: str = "all"):
    if scene_name == "all":
        return mapping["gt_files"].items()
    return [(scene_name, mapping["gt_files"][scene_name])]
