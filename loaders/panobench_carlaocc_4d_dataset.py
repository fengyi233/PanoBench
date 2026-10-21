from typing import Optional

import numpy as np
from mmdet3d.registry import DATASETS

from .panobench_4d_mixin import PanoBench4DMixin
from .panobench_carlaocc_dataset import PanoBenchCarlaOccDataset


@DATASETS.register_module()
class PanoBenchCarlaOcc4DDataset(PanoBench4DMixin, PanoBenchCarlaOccDataset):
    """PanoBench-CarlaOcc occupancy forecasting.

    Occupancy is stored in each frame's LiDAR frame. ``lidar2global`` is that
    frame's world pose, so ``fut2cur = inv(lidar2global_key) @ lidar2global_fut``.
    CarlaOcc is recorded at 10 Hz; ``[0, 10, 20, 30]`` is 0/1/2/3 seconds.
    """

    METAINFO = PanoBenchCarlaOccDataset.METAINFO
    DATASET_TYPE = PanoBenchCarlaOccDataset.DATASET_TYPE

    def __init__(self, *args, future_frames=(0, 10, 20, 30), **kwargs):
        super().__init__(*args, future_frames=list(future_frames), **kwargs)

    def _horizon_pose(self, info: dict) -> Optional[np.ndarray]:
        return np.asarray(info["lidar2global"], dtype=np.float64)
