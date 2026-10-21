import bisect
from pathlib import Path
from typing import Dict, List

import numpy as np
from mmdet3d.registry import DATASETS

from utils.labels import CLASS_NAMES
from utils.metas import KITTI360_META
from .panobench_base_dataset import PanoBenchBaseDataset

DEFAULT_VAL_SCENES = ("2013_05_28_drive_0009_sync",)


def _parse_perspective_txt(path: Path) -> Dict[str, np.ndarray]:
    """Parse KITTI-360 ``calibration/perspective.txt`` (same keys as ViPOcc)."""
    parsed: Dict[str, np.ndarray] = {}
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            try:
                parsed[key.strip()] = np.asarray(
                    [float(x) for x in value.split()], dtype=np.float64)
            except ValueError:
                continue
    return parsed


def _load_image00_intrinsic(data_root: Path, height: int, width: int) -> np.ndarray:
    """Load ``P_rect_00`` and scale to ``(height, width)`` like ViPOcc preprocessing."""
    perspective_path = Path(data_root) / "calibration" / "perspective.txt"
    if perspective_path.is_file():
        data = _parse_perspective_txt(perspective_path)
        if "P_rect_00" not in data:
            raise KeyError(f"{perspective_path} missing P_rect_00")
        k = data["P_rect_00"].reshape(3, 4)[:3, :3].astype(np.float64, copy=True)
        if "S_rect_00" in data:
            calib_w, calib_h = [int(v) for v in data["S_rect_00"][:2]]
        else:
            calib_w, calib_h = KITTI360_META.img_wh
        k[0, :] *= float(width) / float(calib_w)
        k[1, :] *= float(height) / float(calib_h)
        return k
    return KITTI360_META.intrinsic_for_image(height=height, width=width)


@DATASETS.register_module()
class PanoBenchKITTI360Dataset(PanoBenchBaseDataset):
    """PanoBench-KITTI360 canonical occupancy with KITTI-360 image_00 input.

    Calibration follows ViPOcc / official KITTI-360:
    - Images: ``image_00/data_192x640``
    - Intrinsics: ``P_rect_00`` scaled from ``S_rect_00`` (1408x376) to 192x640
    - Extrinsics: occupancy (x-forward, y-left, z-up) -> rectified optical via
      ``KITTI360_META.occ_to_optical``; temporal frames use rectified
      ``cam0_to_world`` (prefer ``cam0_to_world_complete.txt`` when present).
    """

    METAINFO = {
        "dataset_type": "panobench_kitti360",
        "classes": CLASS_NAMES["panobench_kitti360"],
    }
    DATASET_TYPE = "panobench_kitti360"
    IMAGE_FOLDER = "data_192x640"

    def __init__(
            self,
            data_root: str,
            occ_gt_root: str,
            occ_variant,
            pipeline: List,
            ann_file: str = "",
            frame_manifest: str = "",
            test_mode: bool = False,
            num_frames: int = 8,
            train_interval: int = 1,
            test_interval: int = 1,
            **kwargs,
    ):
        self.use_pose = True
        self._poses_by_scene: Dict[str, Dict[str, np.ndarray]] = {}
        self._frame_tokens_by_scene: Dict[str, List[str]] = {}
        self.frame_manifest = Path(frame_manifest) if frame_manifest else None

        img_h, img_w = KITTI360_META.preprocessed_img_hw
        self.img_h = int(img_h)
        self.img_w = int(img_w)
        self.intrinsic = _load_image00_intrinsic(
            Path(data_root), height=self.img_h, width=self.img_w)

        self.occ_to_optical = np.eye(4, dtype=np.float64)
        self.occ_to_optical[:3, :3] = KITTI360_META.occ_to_optical
        self.viewpad = np.eye(4, dtype=np.float64)
        self.viewpad[:3, :3] = self.intrinsic

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

    def _image_dir(self, scene_name: str) -> Path:
        return (
            Path(self.data_root) / "data_2d_raw" / scene_name / "image_00" /
            self.IMAGE_FOLDER
        )

    def _discover_scene_frame_tokens(self) -> Dict[str, List[str]]:
        root = Path(self.occ_gt_root)
        if self.frame_manifest is not None:
            scene_frame_tokens = {}
            seen = set()
            for line in self.frame_manifest.read_text().splitlines():
                entry = line.strip()
                if not entry or entry.startswith("#"):
                    continue
                relative = Path(entry)
                if (len(relative.parts) != 2 or relative.suffix != ".npz"
                        or ".." in relative.parts or relative.is_absolute()):
                    raise ValueError(f"Invalid KITTI360 manifest entry: {entry}")
                if entry in seen:
                    raise ValueError(f"Duplicate KITTI360 manifest entry: {entry}")
                seen.add(entry)
                scene, token = relative.parent.name, relative.stem
                image = self._image_dir(scene) / f"{token}.png"
                if not (root / relative).is_file() or not image.is_file():
                    raise FileNotFoundError(
                        f"KITTI360 manifest frame {entry} requires both "
                        f"{root / relative} and {image}.")
                scene_frame_tokens.setdefault(scene, []).append(token)
            if not seen:
                raise ValueError(f"Empty KITTI360 frame manifest: {self.frame_manifest}")
            return {scene: sorted(tokens)
                    for scene, tokens in sorted(scene_frame_tokens.items())}
        scene_frame_tokens = {}
        if not root.exists():
            return scene_frame_tokens
        for scene_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            image_dir = self._image_dir(scene_dir.name)
            frame_tokens = []
            for occ_path in sorted(scene_dir.glob("*.npz")):
                frame_token = occ_path.stem
                if (image_dir / f"{frame_token}.png").is_file():
                    frame_tokens.append(frame_token)
            if frame_tokens:
                scene_frame_tokens[scene_dir.name] = frame_tokens
        return scene_frame_tokens

    def _load_scene_poses(self, scene_name: str) -> Dict[str, np.ndarray]:
        if scene_name in self._poses_by_scene:
            return self._poses_by_scene[scene_name]

        pose_dir = Path(self.data_root) / "data_poses" / scene_name
        # Prefer the dense rectified poses distributed with the labels.
        candidates = (
            pose_dir / "cam0_to_world_complete.txt",
            pose_dir / "cam0_to_world.txt",
        )
        pose_path = next((p for p in candidates if p.is_file()), None)
        if pose_path is None:
            raise FileNotFoundError(
                f"KITTI-360 cam0_to_world not found under {pose_dir} "
                "(expected cam0_to_world_complete.txt or cam0_to_world.txt)."
            )
        poses = {
            f"{int(arr[0]):010d}": arr[1:].reshape(4, 4)
            for arr in np.loadtxt(pose_path, dtype=np.float64)
        }
        self._poses_by_scene[scene_name] = poses
        return poses

    def load_data_list(self) -> list:
        scene_frame_tokens = self._discover_scene_frame_tokens()
        if self.test_mode:
            scenes = list(DEFAULT_VAL_SCENES)
        else:
            scenes = [
                s for s in scene_frame_tokens.keys()
                if s not in DEFAULT_VAL_SCENES
            ]

        data_list = []
        skipped_empty = 0
        for scene_name in scenes:
            frame_tokens = sorted(scene_frame_tokens.get(scene_name, []))
            kept_tokens = []
            for frame_token in frame_tokens:
                occ_path = self.occ_gt_root / scene_name / f"{frame_token}.npz"
                try:
                    with np.load(occ_path) as occ:
                        if "valid_mask" in occ.files and int(occ["valid_mask"].sum()) == 0:
                            if self.frame_manifest is not None:
                                raise ValueError(
                                    f"Empty valid_mask in listed frame: {occ_path}")
                            skipped_empty += 1
                            continue
                except OSError:
                    if self.frame_manifest is not None:
                        raise
                    continue
                kept_tokens.append(frame_token)
                data_list.append({
                    "scene_name": scene_name,
                    "frame_token": frame_token,
                })
            self._frame_tokens_by_scene[scene_name] = kept_tokens
        if skipped_empty:
            print(
                f"[PanoBenchKITTI360] skipped {skipped_empty} frames with empty valid_mask",
                flush=True,
            )
        if not data_list:
            discovered = sorted(scene_frame_tokens)
            raise FileNotFoundError(
                f"No KITTI-360 samples under {self.occ_gt_root} "
                f"(test_mode={self.test_mode}, requested={scenes}, "
                f"discovered={discovered}). "
                "KITTI-360 occupancy lives at "
                "<occ_root>/panobench_kitti360/<scene>/, not under a variant "
                "subdir; use occ_variant=null (checkpoint folder 'default')."
            )
        return data_list

    def _select_prev_frame_tokens(self, scene_name: str, frame_token: str) -> list:
        frame_tokens = self._frame_tokens_by_scene[scene_name]
        cur_pos = bisect.bisect_left(frame_tokens, frame_token)
        return [
            None if pos is None else frame_tokens[pos]
            for pos in self.history_indices(cur_pos)
        ]

    def _occ2img(self, scene_name: str, key_frame_token: str,
                 frame_token: str) -> np.ndarray:
        poses = self._load_scene_poses(scene_name)
        key_pose = poses.get(key_frame_token)
        token_pose = poses.get(frame_token)
        if key_pose is not None and token_pose is not None:
            key_optical_to_token_optical = np.linalg.inv(token_pose) @ key_pose
        else:
            raise KeyError(
                f"Missing KITTI360 pose in {scene_name}: "
                f"key={key_frame_token}, frame={frame_token}. "
                "Use the supplied dense cam0_to_world_complete.txt.")
        return (
                self.viewpad @ key_optical_to_token_optical @ self.occ_to_optical
        ).astype(np.float32)

    def get_data_info(self, idx: int) -> dict:
        info = self._raw_infos[idx]
        scene_name = info["scene_name"]
        frame_token = info["frame_token"]

        filenames = [
            f"data_2d_raw/{scene_name}/image_00/{self.IMAGE_FOLDER}/{frame_token}.png"
        ]
        occ2imgs = [self._occ2img(scene_name, frame_token, frame_token)]
        for prev_frame_token in self._select_prev_frame_tokens(scene_name, frame_token):
            if prev_frame_token is None:
                filenames.append(filenames[0])
                occ2imgs.append(occ2imgs[0].copy())
                continue
            filenames.append(
                f"data_2d_raw/{scene_name}/image_00/{self.IMAGE_FOLDER}/"
                f"{prev_frame_token}.png"
            )
            occ2imgs.append(self._occ2img(scene_name, frame_token, prev_frame_token))

        return self.pack_sample_info(
            scene_name=scene_name,
            frame_token=frame_token,
            filenames=filenames,
            occ2img=occ2imgs,
        )
