import bisect
from pathlib import Path
from typing import List, Optional

import numpy as np
from mmdet3d.registry import DATASETS
from scipy.spatial.transform import Rotation, Slerp

from .panobench_4d_mixin import PanoBench4DMixin
from .panobench_kitti360_dataset import PanoBenchKITTI360Dataset


def _interpolate_pose(pose_a: np.ndarray, pose_b: np.ndarray, ratio: float) -> np.ndarray:
    """Lerp translation and SLERP rotation between two 4x4 poses."""
    pose_a = np.asarray(pose_a, dtype=np.float64)
    pose_b = np.asarray(pose_b, dtype=np.float64)
    ratio = float(ratio)
    key_rots = Rotation.from_matrix(np.stack([pose_a[:3, :3], pose_b[:3, :3]], axis=0))
    rotation = Slerp([0.0, 1.0], key_rots)([ratio]).as_matrix()[0]
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = (1.0 - ratio) * pose_a[:3, 3] + ratio * pose_b[:3, 3]
    return pose


@DATASETS.register_module()
class PanoBenchKITTI3604DDataset(PanoBench4DMixin, PanoBenchKITTI360Dataset):
    """PanoBench-KITTI360 occupancy forecasting.

    Occupancy lives in the OCC frame (x-forward, y-left, z-up). Official
    ``cam0_to_world`` is sparse, so missing frames are SLERP-interpolated from
    neighboring official poses. The occupancy-to-world pose is
    ``cam0_to_world @ occ_to_optical``, and
    ``fut2cur = inv(T_world_occ_key) @ T_world_occ_fut``.

    KITTI-360 is 10 Hz; ``[0, 10, 20, 30]`` is 0/1/2/3 seconds. Horizons are
    resolved by raw frame id, not by the filtered sample index, because empty
    ``valid_mask`` holes would otherwise stretch the forecast window.
    Every horizon must belong to the loaded split's frame list. Additional
    label files on disk do not expand the released benchmark scope.
    """

    METAINFO = PanoBenchKITTI360Dataset.METAINFO
    DATASET_TYPE = PanoBenchKITTI360Dataset.DATASET_TYPE

    def __init__(self, *args, future_frames=(0, 10, 20, 30), **kwargs):
        self._occ_world_poses = {}
        self._pose_ids = {}
        self._horizon_tokens: List[List[str]] = []
        super().__init__(*args, future_frames=list(future_frames), **kwargs)

    def _horizon_pose(self, info: dict) -> Optional[np.ndarray]:
        return self._occ_world_pose(info["scene_name"], info["frame_token"])

    def filter_data(self) -> List[dict]:
        data_list = super(PanoBench4DMixin, self).filter_data()
        available_frames = {
            (info["scene_name"], info["frame_token"]) for info in data_list
        }
        keep, horizon_tokens = [], []
        for idx, info in enumerate(data_list):
            scene_name = info["scene_name"]
            base_id = int(info["frame_token"])
            tokens = []
            ok = True
            for offset in self.future_frames:
                token = f"{base_id + offset:010d}"
                if (scene_name, token) not in available_frames:
                    ok = False
                    break
                if not Path(self.get_occ_path(scene_name, token)).is_file():
                    ok = False
                    break
                if self._occ_world_pose(scene_name, token) is None:
                    ok = False
                    break
                tokens.append(token)
            if ok:
                keep.append(idx)
                horizon_tokens.append(tokens)

        self._horizon_tokens = horizon_tokens
        dropped = len(data_list) - len(keep)
        if dropped:
            print(
                f"[{type(self).__name__}] dropped {dropped}/{len(data_list)} samples "
                f"without frame-id horizons {list(self.future_frames)}",
                flush=True,
            )
        return [data_list[idx] for idx in keep]

    def get_data_info(self, idx: int) -> dict:
        info = super(PanoBench4DMixin, self).get_data_info(idx)
        scene_name = info["scene_name"]
        tokens = self._horizon_tokens[idx]
        key_pose = self._occ_world_pose(scene_name, tokens[0])
        global2key = np.linalg.inv(key_pose)

        fut2cur, future_occ_paths = [], []
        for token in tokens:
            pose = self._occ_world_pose(scene_name, token)
            fut2cur.append((global2key @ pose).astype(np.float32))
            future_occ_paths.append(str(self.get_occ_path(scene_name, token)))

        info["fut2cur"] = fut2cur
        info["future_occ_paths"] = future_occ_paths
        info["fut_list"] = list(self.future_frames)
        return info

    def _cam0_pose(self, scene_name: str, frame_token: str) -> Optional[np.ndarray]:
        poses = self._load_scene_poses(scene_name)
        if frame_token in poses:
            return np.asarray(poses[frame_token], dtype=np.float64)

        frame_id = int(frame_token)
        ids = self._pose_ids.setdefault(
            scene_name, sorted(int(token) for token in poses))
        insert = bisect.bisect_right(ids, frame_id)
        if insert == 0 or insert == len(ids):
            return None
        left, right = ids[insert - 1], ids[insert]
        ratio = (frame_id - left) / (right - left)
        return _interpolate_pose(
            poses[f"{left:010d}"], poses[f"{right:010d}"], ratio)

    def _occ_world_pose(self, scene_name: str, frame_token: str) -> Optional[np.ndarray]:
        cache = self._occ_world_poses.setdefault(scene_name, {})
        if frame_token in cache:
            return cache[frame_token]
        cam_pose = self._cam0_pose(scene_name, frame_token)
        if cam_pose is None:
            cache[frame_token] = None
            return None
        # T_world_occ = T_world_cam @ T_cam_occ
        pose = cam_pose @ self.occ_to_optical
        cache[frame_token] = pose
        return pose
