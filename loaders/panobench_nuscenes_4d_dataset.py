from typing import Optional

import numpy as np
from mmdet3d.registry import DATASETS

from .panobench_4d_mixin import PanoBench4DMixin
from .panobench_nuscenes_dataset import PanoBenchNuScenesDataset


@DATASETS.register_module()
class PanoBenchNuScenes4DDataset(PanoBench4DMixin, PanoBenchNuScenesDataset):
    """PanoBench-nuScenes for trajectory-conditioned occupancy forecasting.

    Extends the single-frame dataset with, per sample, one occupancy target and
    one ``fut2cur`` transform for every entry of ``future_frames``. Horizon
    offsets are counted in key frames (nuScenes key frames are 2 Hz, so
    ``[0, 2, 4, 6]`` covers 0/1/2/3 seconds).

    Canonical occupancy is stored in each frame's own ego frame, so horizon
    ``k``'s ground truth and its query points live in the ego frame of the
    frame ``k`` steps ahead. ``fut2cur = inv(ego2global_key) @ ego2global_fut``
    brings those points back to the key frame, where ``occ2img`` is defined.

    Samples whose scene ends before ``future_frames[-1]`` are dropped, so every
    horizon is scored against real ground truth rather than a replayed frame.

    Returns from ``get_data_info(idx)``, in addition to the single-frame keys:
        fut2cur (List[np.ndarray[4, 4]]): One transform per horizon;
            ``fut2cur[0]`` is the identity.
        future_occ_paths (List[str]): Canonical ``.npz`` path per horizon.
        fut_list (List[int]): The horizon offsets themselves.
    """

    METAINFO = PanoBenchNuScenesDataset.METAINFO
    DATASET_TYPE = PanoBenchNuScenesDataset.DATASET_TYPE

    def __init__(self, *args, future_frames=(0, 2, 4, 6), **kwargs):
        super().__init__(*args, future_frames=list(future_frames), **kwargs)

    def _horizon_pose(self, info: dict) -> Optional[np.ndarray]:
        return np.asarray(info["ego2global"], dtype=np.float64)
