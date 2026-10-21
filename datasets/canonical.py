"""Unified reader over the canonical panoptic occupancy store.

Layout::

    <root>/<type>/<scene>/<frame>.npz                         # single GT
    <root>/<type>/<variant>/<scene>/<frame>.npz               # multi GT

Each npz uses :class:`utils.panoptic_occupancy.PanopticOccupancy` fields:
``occupancy``, packed dense ``valid_mask``, ``volume_size``, ``voxel_size`` and
``voxel_origin``. Dataset metadata in :data:`utils.labels.DATASET_META` is still
used for label definitions and default paths.
"""

from pathlib import Path
from typing import List, Optional

import numpy as np

from utils.data_converter import encode_global_id, occ_sparse2dense
from utils.labels import DATASET_META
from utils.mappings import FrameRecord, load_mapping, scene_items
from utils.panoptic_occupancy import PanopticOccupancy


class CanonicalOccDataset:
    """Reads sparse occupancy + dense valid masks from the canonical store."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.dataset_type = str(cfg.type)
        self.variant = cfg.get("variant")
        self.meta = DATASET_META[(self.dataset_type, self.variant)]

        root = Path(str(cfg.path)) / self.dataset_type
        if self.variant is not None:
            root = root / self.variant
        self.root = root

        self.scene_name = str(cfg.get("scene_name", "all"))
        self.frame_token = str(cfg.get("frame_token", "all"))
        self.mask_key: Optional[str] = cfg.get("mask_key", None)
        mappings = cfg.get("mappings", None)
        self.mapping = load_mapping(mappings) if mappings else None

        self.frames = self._discover_frames()
        if not self.frames:
            note = f", variant={self.variant}" if self.variant else ""
            raise FileNotFoundError(
                f"No canonical frames under {self.root} "
                f"(scene_name={self.scene_name}, frame_token={self.frame_token}{note})"
            )

    def _discover_frames(self) -> List[FrameRecord]:
        if self.mapping is not None:
            return self._discover_frames_from_mapping()
        return self._discover_frames_by_sort()

    def _discover_frames_from_mapping(self) -> List[FrameRecord]:
        frames: List[FrameRecord] = []
        for scene, tokens in scene_items(self.mapping, self.scene_name):
            for idx, token in enumerate(tokens):
                if self.frame_token != "all" and token != self.frame_token:
                    continue
                path = self.occ_path(scene, token)
                if path.is_file():
                    frames.append(FrameRecord(scene, token, path, idx))
        return frames

    def _discover_frames_by_sort(self) -> List[FrameRecord]:
        if not self.root.exists():
            return []
        scenes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        if self.scene_name != "all":
            scenes = [s for s in scenes if s == self.scene_name]

        frames: List[FrameRecord] = []
        for scene in scenes:
            paths = sorted((self.root / scene).glob("*.npz"))
            for idx, path in enumerate(paths):
                token = path.stem
                if self.frame_token != "all" and token != self.frame_token:
                    continue
                frames.append(FrameRecord(scene, token, path, idx))
        return frames

    def occ_path(self, scene_name: str, token: str) -> Path:
        return self.root / scene_name / f"{token}.npz"

    @staticmethod
    def _resolve_mask_key(mask_key: Optional[str]) -> Optional[str]:
        if mask_key in (None, False):
            return None
        if mask_key != "valid_mask":
            raise KeyError(
                "Canonical occupancy only supports mask_key='valid_mask' in the "
                "new packed dense mask format."
            )
        return "valid_mask"

    def read(self, record: FrameRecord) -> PanopticOccupancy:
        if not record.path.is_file():
            raise FileNotFoundError(f"Canonical occupancy not found: {record.path}")
        sample = PanopticOccupancy.load(record.path)
        expected_volume = tuple(self.meta.volume_size)
        if sample.volume_size != expected_volume:
            raise ValueError(
                f"{record.path} volume_size={sample.volume_size} does not match "
                f"dataset metadata volume_size={expected_volume}."
            )
        if not np.isclose(sample.voxel_size, float(self.meta.voxel_size)):
            raise ValueError(
                f"{record.path} voxel_size={sample.voxel_size} does not match "
                f"dataset metadata voxel_size={float(self.meta.voxel_size)}."
            )
        if not np.allclose(
            np.asarray(sample.voxel_origin, dtype=np.float32),
            np.asarray(self.meta.voxel_origin, dtype=np.float32),
        ):
            raise ValueError(
                f"{record.path} voxel_origin={sample.voxel_origin} does not match "
                f"dataset metadata voxel_origin={tuple(self.meta.voxel_origin)}."
            )
        return sample

    def load(
        self,
        record: FrameRecord,
        return_sparse: bool = True,
        mask_key: Optional[str] = "__default__",
    ) -> np.ndarray:
        sample = self.read(record)
        requested_key = self.mask_key if mask_key == "__default__" else mask_key
        key = self._resolve_mask_key(requested_key)
        occupancy = sample.masked_occupancy() if key == "valid_mask" else sample.occupancy

        if return_sparse:
            return occupancy
        return occ_sparse2dense(
            occupancy,
            volume_size=sample.volume_size,
            free_id=int(encode_global_id(self.meta.free_id, 0)),
        )
