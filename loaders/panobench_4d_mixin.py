from typing import List, Optional

import numpy as np


class PanoBench4DMixin:
    """Attach forecast horizons to a single-frame PanoBench dataset.

    Drops tail samples that lack a full ``future_frames`` window, then writes
    ``fut2cur`` / ``future_occ_paths`` / ``fut_list`` in ``get_data_info``.
    Subclasses only implement :meth:`_horizon_pose`, which must return the
    occupancy-frame-to-world transform of that sample (or ``None`` to drop it).
    """

    def _horizon_pose(self, info: dict) -> Optional[np.ndarray]:
        raise NotImplementedError

    def filter_data(self) -> List[dict]:
        data_list = super().filter_data()
        max_future = self.future_frames[-1]

        scene_end = [0] * len(data_list)
        end = len(data_list) - 1
        for idx in range(len(data_list) - 1, -1, -1):
            if idx < len(data_list) - 1 and \
                    data_list[idx]["scene_name"] != data_list[idx + 1]["scene_name"]:
                end = idx
            scene_end[idx] = end

        self._future_infos = []
        for info in data_list:
            pose = self._horizon_pose(info)
            self._future_infos.append({
                "scene_name": info["scene_name"],
                "frame_token": info["frame_token"],
                "pose": None if pose is None else np.asarray(pose, dtype=np.float64),
            })

        keep = []
        for idx in range(len(data_list)):
            if idx + max_future > scene_end[idx]:
                continue
            if any(
                    self._future_infos[idx + offset]["pose"] is None
                    for offset in self.future_frames):
                continue
            keep.append(idx)

        self._unfiltered_index = keep
        dropped = len(data_list) - len(keep)
        if dropped:
            print(
                f"[{type(self).__name__}] dropped {dropped}/{len(data_list)} samples "
                f"without {max_future} future key frames",
                flush=True,
            )
        return [data_list[idx] for idx in keep]

    def get_data_info(self, idx: int) -> dict:
        info = super().get_data_info(idx)

        base = self._unfiltered_index[idx]
        key_pose = self._future_infos[base]["pose"]
        global2key = np.linalg.inv(key_pose)

        fut2cur, future_occ_paths = [], []
        for offset in self.future_frames:
            future = self._future_infos[base + offset]
            fut2cur.append((global2key @ future["pose"]).astype(np.float32))
            future_occ_paths.append(
                str(self.get_occ_path(future["scene_name"], future["frame_token"])))

        info["fut2cur"] = fut2cur
        info["future_occ_paths"] = future_occ_paths
        info["fut_list"] = list(self.future_frames)
        return info
