from typing import List, Optional

import numpy as np
from mmdet3d.registry import DATASETS

from utils.labels import CLASS_NAMES
from utils.metas import NUS_CAM_TYPES
from .panobench_base_dataset import PanoBenchBaseDataset


@DATASETS.register_module()
class PanoBenchNuScenesDataset(PanoBenchBaseDataset):
    """PanoBench-nuScenes dataset backed by canonical sparse occupancy GT.

    The dataset owns the definition of a *sample view*: it picks one keyframe
    plus ``sweeps_num = num_frames - 1`` previous keyframes (fixed interval,
    with edge-padding near scene starts), and pre-composes ``occ2img`` for each
    of the resulting ``num_frames * 6`` camera entries. The pipeline is then a
    pure I/O stage that only reads images by ``filename``.

    Args:
        data_root: Root directory under which ``filename`` paths are resolved.
        occ_gt_root: Root directory of canonical GT files laid out as
            ``<scene_name>/<frame_token>.npz``.
        pipeline: Sequence of transforms (typically ``LoadMultiViewImages`` and
            optional ``LoadCanonicalOcc`` + augmentations + format / collect).
        test_mode: When True, use ``test_interval`` (deterministic, no random).
        num_frames: Total frames per sample (current + previous sweeps).
        train_interval: Keyframe stride between selected history frames.
        test_interval: Keyframe stride between selected history frames during eval.

    Returns from ``get_data_info(idx)``:
        data_root (str): Same as ``self.data_root``; pipeline joins it with
            ``filename`` entries to form absolute image paths.
        frame_token (str): Canonical frame token; NuScenes sample token here.
        scene_name (str): NuScenes scene name (e.g. ``scene-0001``).
        occ2img (List[np.ndarray[4, 4]]): One projection matrix per camera
            entry, length ``(1 + sweeps_num) * 6``; ``cam_info['global2img'] @
            ego2global``, aligned with the keyframe occupancy frame.
        filename (List[str]): Relative image paths, length matches ``occ2img``.
        occ_path (str): Absolute path to canonical ``.npz`` (mIoU/RayIoU ground
            truth), used by ``LoadCanonicalOcc`` and the evaluator.
    """

    METAINFO = {
        "dataset_type": "panobench_nuscenes",
        "classes": CLASS_NAMES["panobench_nuscenes"]
    }
    DATASET_TYPE = "panobench_nuscenes"

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
            **kwargs
    ):
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
            **kwargs
        )

        self._raw_infos, self._scene_start_by_idx = self._build_scene_cache()

    def _build_scene_cache(self):
        """Materialize per-sample infos and per-sample scene-start indices.

        ``mmengine.BaseDataset`` clears ``self.data_list`` after serialization
        (``serialize_data=True`` by default), so we cache the deserialized info
        dicts once at init. ``scene_start_by_idx[i]`` is the smallest index
        ``j <= i`` that still belongs to the same scene as ``i``; used to stop
        prev-sweep walking at scene boundaries.
        """
        raw_infos = []
        scene_start_by_idx = []
        scene_first_idx = 0
        prev_scene_name = None
        for idx in range(len(self)):
            info = super().get_data_info(idx)
            raw_infos.append(info)
            if info["scene_name"] != prev_scene_name:
                scene_first_idx = idx
                prev_scene_name = info["scene_name"]
            scene_start_by_idx.append(scene_first_idx)
        return raw_infos, scene_start_by_idx

    def parse_data_info(self, info: dict) -> dict:
        """Attach canonical ``occ_path`` for transforms and evaluator."""
        if "frame_token" not in info:
            info["frame_token"] = info["token"]
        if not info.get("occ_path"):
            info["occ_path"] = self.get_occ_path(info["scene_name"], info["frame_token"])
        return info

    def _select_prev_infos(self, index: int) -> List[Optional[dict]]:
        """Return exactly ``sweeps_num`` previous keyframes (or ``None`` for padding).

        ``None`` indicates the slot should be filled by replaying the current
        keyframe (which is what ``get_data_info`` does). This happens at the
        very first keyframe of a scene where no prev sweeps exist.
        """
        scene_start = self._scene_start_by_idx[index]
        selected: List[Optional[dict]] = []
        for hist_idx in self.history_indices(index, min_pos=scene_start):
            if hist_idx is not None:
                selected.append(self._raw_infos[hist_idx])
            elif selected:
                selected.append(selected[-1])
            else:
                selected.append(None)
        return selected

    def get_data_info(self, idx: int) -> dict:
        """Build one sample: keyframe + selected prev sweeps in a single flat view.

        All ``occ2img`` matrices are referenced to the keyframe occupancy frame, so each
        prev camera entry already accounts for the motion between the previous
        keyframe timestamp and the current keyframe via ``global2img``.
        """
        info = self._raw_infos[idx]
        ego2global = info["ego2global"]

        filenames: List[str] = []
        occ2imgs: List[np.ndarray] = []
        cur_filenames: List[str] = []
        cur_occ2imgs: List[np.ndarray] = []
        for cam in NUS_CAM_TYPES:
            cam_info = info["cams"][cam]
            cur_filenames.append(cam_info["data_path"])
            cur_occ2imgs.append(cam_info["global2img"] @ ego2global)
        filenames.extend(cur_filenames)
        occ2imgs.extend(cur_occ2imgs)

        for prev_info in self._select_prev_infos(idx):
            if prev_info is None:
                filenames.extend(cur_filenames)
                occ2imgs.extend(m.copy() for m in cur_occ2imgs)
                continue
            for cam in NUS_CAM_TYPES:
                cam_info = prev_info["cams"][cam]
                filenames.append(cam_info["data_path"])
                occ2imgs.append(cam_info["global2img"] @ ego2global)

        return self.pack_sample_info(
            scene_name=info["scene_name"],
            frame_token=info["frame_token"],
            filenames=filenames,
            occ2img=occ2imgs,
        )
