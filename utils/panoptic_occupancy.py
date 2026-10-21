"""Panoptic occupancy npz container.

Canonical on-disk format:

    occupancy    : [N, 4] uint16, columns (x, y, z, 1000 * semantic + instance)
    valid_mask   : [ceil(X * Y * Z / 8)] uint8, np.packbits dense bool mask
    volume_size  : [3] int32, dense grid shape (X, Y, Z)
    voxel_size   : scalar float32, isotropic voxel size in meters
    voxel_origin : [3] float32, origin of voxel (0, 0, 0)

The dataclass stores ``valid_mask`` as a dense bool ndarray because
training and evaluation need the dense evaluation domain.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from utils.data_converter import decode_global_id

PANOPTIC_OCCUPANCY_FIELDS = (
    "occupancy",
    "valid_mask",
    "volume_size",
    "voxel_size",
    "voxel_origin",
)



def unpack_dense_mask(packed: np.ndarray, volume_size) -> np.ndarray:
    """Unpack a canonical packed dense bool mask."""
    volume_size_arr = np.asarray(volume_size, dtype=np.int64).reshape(-1)
    shape = tuple(int(v) for v in volume_size_arr)
    packed = np.asarray(packed)
    num_voxels = int(np.prod(shape))
    return np.unpackbits(packed, count=num_voxels).view(np.bool_).reshape(shape)


@dataclass
class PanopticOccupancy:
    """Sparse panoptic occupancy plus a dense valid/evaluation domain."""

    occupancy: np.ndarray  # [N,4]
    valid_mask: np.ndarray  # [X, Y, Z]
    volume_size: tuple[int, int, int]
    voxel_size: float
    voxel_origin: tuple[float, float, float]

    def __post_init__(self) -> None:
        occupancy = np.asfortranarray(self.occupancy, dtype=np.uint16)
        self.occupancy = occupancy

        valid_mask = np.asarray(self.valid_mask, dtype=np.bool_)
        self.valid_mask = np.ascontiguousarray(valid_mask)

        volume_size = np.asarray(self.volume_size, dtype=np.int64).reshape(-1)
        self.volume_size = tuple(int(v) for v in volume_size)

        voxel_origin = np.asarray(self.voxel_origin, dtype=np.float32).reshape(-1)
        self.voxel_origin = tuple(float(v) for v in voxel_origin)

        self.voxel_size = float(self.voxel_size)

    @classmethod
    def load(cls, path: str | Path) -> "PanopticOccupancy":
        with np.load(path, allow_pickle=False) as data:
            assert set(data.files) == set(PANOPTIC_OCCUPANCY_FIELDS), (
                f"Panoptic occupancy npz fields must be {PANOPTIC_OCCUPANCY_FIELDS}, "
                f"got {tuple(data.files)}"
            )
            volume_size = np.asarray(data["volume_size"], dtype=np.int64).reshape(-1)
            volume_size = tuple(int(v) for v in volume_size)
            return cls(
                occupancy=data["occupancy"],
                valid_mask=unpack_dense_mask(data["valid_mask"], volume_size),
                volume_size=data["volume_size"],
                voxel_size=data["voxel_size"],
                voxel_origin=data["voxel_origin"],
            )

    @property
    def coords(self) -> np.ndarray:
        return self.occupancy[:, :3].astype(np.int64, copy=False)

    @property
    def global_id(self) -> np.ndarray:
        return self.occupancy[:, 3]

    @property
    def semantic(self) -> np.ndarray:
        semantic, _ = decode_global_id(self.global_id)
        return semantic

    @property
    def instance(self) -> np.ndarray:
        _, instance = decode_global_id(self.global_id)
        return instance

    def dense_semantic(self, free_id: int, dtype=np.uint8) -> np.ndarray:
        dense = np.full(self.volume_size, free_id, dtype=dtype)
        if self.occupancy.shape[0] > 0:
            coords = self.coords
            dense[coords[:, 0], coords[:, 1], coords[:, 2]] = self.semantic.astype(dtype)
        return dense

    def dense_instance(self, dtype=np.int32) -> np.ndarray:
        dense = np.zeros(self.volume_size, dtype=dtype)
        if self.occupancy.shape[0] > 0:
            coords = self.coords
            dense[coords[:, 0], coords[:, 1], coords[:, 2]] = self.instance.astype(dtype)
        return dense

    def valid_at_occupancy(self) -> np.ndarray:
        if self.occupancy.shape[0] == 0:
            return np.zeros((0,), dtype=np.bool_)
        coords = self.coords
        return self.valid_mask[coords[:, 0], coords[:, 1], coords[:, 2]]

    def masked_occupancy(self) -> np.ndarray:
        return self.occupancy[self.valid_at_occupancy()]
