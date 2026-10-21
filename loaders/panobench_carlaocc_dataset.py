import bisect
from pathlib import Path
from typing import List

import numpy as np
import yaml
from mmdet3d.registry import DATASETS

from utils.labels import CLASS_NAMES
from utils.metas import carlaocc_camera_names
from .panobench_base_dataset import PanoBenchBaseDataset


@DATASETS.register_module()
class PanoBenchCarlaOccDataset(PanoBenchBaseDataset):
    """PanoBench-CarlaOcc dataset built from raw images/poses and canonical GT.

    Canonical PanoBench-CarlaOcc occupancy is stored in the keyframe LiDAR
    coordinate system. Projection matrices are returned directly as ``occ2img``.
    """

    METAINFO = {
        "dataset_type": "panobench_carlaocc",
        "classes": CLASS_NAMES["panobench_carlaocc"],
    }
    DATASET_TYPE = "panobench_carlaocc"

    def __init__(
            self,
            data_root: str,
            occ_gt_root: str,
            occ_variant: str,
            pipeline: List,
            ann_file: str = "",
            test_mode: bool = False,
            num_frames: int = 8,
            train_interval: int = 1,
            test_interval: int = 1,
            **kwargs,
    ):
        self.split = "val" if test_mode else "train"
        self._poses_by_scene = {}
        self._frame_token_ids_by_scene = {}
        self.cams = carlaocc_camera_names(occ_variant)
        self.img_h, self.img_w = (192, 640)
        self._static_lidar2img = self._build_lidar2img(data_root)
        super().__init__(
            data_root=data_root,
            occ_gt_root=occ_gt_root,
            occ_variant=occ_variant,
            pipeline=pipeline,
            ann_file=ann_file,
            test_mode=test_mode,
            num_frames=num_frames,
            train_interval=train_interval,
            test_interval=test_interval,
            **kwargs,
        )
        self._raw_infos = self.build_raw_infos()

    def _build_lidar2img(self, data_root: str) -> dict:
        calib_path = Path(data_root) / "calib" / "calib.yaml"
        with calib_path.open("r", encoding="utf-8") as f:
            calib = yaml.safe_load(f)

        ori_h, ori_w = calib["cam_settings"]["height"], calib["cam_settings"]["width"]
        k_mat = np.asarray(calib["cam_settings"]["intrinsics"], dtype=np.float64)
        k_mat[0] *= self.img_w / ori_w
        k_mat[1] *= self.img_h / ori_h

        viewpad = np.eye(4, dtype=np.float64)
        viewpad[:3, :3] = k_mat
        r_opt_pad = np.eye(4, dtype=np.float64)
        r_opt_pad[:3, :3] = np.array([[0, -1, 0],
                                      [0, 0, -1],
                                      [1, 0, 0]], dtype=np.float64)

        lidar2img = {}
        for cam in self.cams:
            cam2lidar = np.array(calib["sensors"][cam]["transform"], dtype=np.float64)
            lidar2cam = np.linalg.inv(cam2lidar)
            lidar2img[cam] = (viewpad @ r_opt_pad @ lidar2cam).astype(np.float32)
        return lidar2img

    def _load_scene_poses(self, scene_name: str) -> dict:
        if scene_name in self._poses_by_scene:
            return self._poses_by_scene[scene_name]

        pose_path = Path(self.data_root) / scene_name / "poses" / "lidar.txt"
        poses = {f"{int(arr[0]):04d}": arr[1:].reshape(4, 4) for arr in np.loadtxt(pose_path, dtype=np.float32)}
        self._poses_by_scene[scene_name] = poses
        self._frame_token_ids_by_scene[scene_name] = sorted(
            int(frame_token) for frame_token in poses.keys())
        return self._poses_by_scene[scene_name]

    def load_data_list(self) -> list:
        split_path = Path(self.data_root) / "splits" / f"{self.split}.txt"
        data_list = []
        with split_path.open("r", encoding="utf-8") as f:
            for line in f:
                parts = line.split()
                if len(parts) != 3:
                    continue
                town_id, seq_id, frame_token = parts
                town_name = "Town10HD_Opt" if town_id == "10" else f"Town{town_id}_Opt"
                scene_name = f"{town_name}_Seq{seq_id}"
                poses = self._load_scene_poses(scene_name)
                if frame_token not in poses:
                    continue
                data_list.append({
                    "frame_token": frame_token,
                    "scene_name": scene_name,
                    "lidar2global": poses[frame_token],
                })
        return data_list

    def _select_prev_frame_tokens(self, scene_name: str, frame_token: str) -> list:
        frame_token_ids = self._frame_token_ids_by_scene[scene_name]
        frame_token_id = int(frame_token)
        cur_pos = bisect.bisect_left(frame_token_ids, frame_token_id)
        return [
            None if pos is None else f"{frame_token_ids[pos]:04d}"
            for pos in self.history_indices(cur_pos)
        ]

    def _frame_token_views(self, scene_name: str, frame_token: str,
                           key_lidar2global: np.ndarray) -> tuple:
        token_lidar2global = self._poses_by_scene[scene_name][frame_token]
        key_from_token = np.linalg.inv(token_lidar2global) @ key_lidar2global
        filenames = []
        lidar2imgs = []
        for cam in self.cams:
            cam_dir = f"image_{cam.split('_')[1]}"
            filenames.append(str(Path(scene_name) / "rgb" / cam_dir / f"{frame_token}.png"))
            lidar2imgs.append(
                (self._static_lidar2img[cam] @ key_from_token).astype(np.float32))
        return filenames, lidar2imgs

    def get_data_info(self, idx: int) -> dict:
        info = self._raw_infos[idx]
        scene_name = info["scene_name"]
        frame_token = info["frame_token"]
        key_lidar2global = info["lidar2global"]

        cur_filenames, cur_lidar2imgs = self._frame_token_views(
            scene_name, frame_token, key_lidar2global)
        filenames = list(cur_filenames)
        lidar2imgs = list(cur_lidar2imgs)

        for prev_frame_token in self._select_prev_frame_tokens(scene_name, frame_token):
            if prev_frame_token is None:
                filenames.extend(cur_filenames)
                lidar2imgs.extend(mat.copy() for mat in cur_lidar2imgs)
                continue
            prev_filenames, prev_lidar2imgs = self._frame_token_views(
                scene_name, prev_frame_token, key_lidar2global)
            filenames.extend(prev_filenames)
            lidar2imgs.extend(prev_lidar2imgs)

        return self.pack_sample_info(
            scene_name=scene_name,
            frame_token=frame_token,
            filenames=filenames,
            occ2img=lidar2imgs,
        )
