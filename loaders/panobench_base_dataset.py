from pathlib import Path
from typing import List, Optional

from mmdet3d.datasets import Det3DDataset


class PanoBenchBaseDataset(Det3DDataset):
    DATASET_TYPE: str

    def __init__(
            self,
            data_root: str,
            occ_gt_root: str,
            occ_variant,
            pipeline: List,
            ann_file: str = "",
            test_mode: bool = False,
            num_frames: int = 8,
            train_interval: int = 1,
            test_interval: int = 1,
            future_frames: Optional[List[int]] = None,
            **kwargs,
    ):
        self.occ_variant = occ_variant
        if occ_variant:
            self.occ_gt_root = Path(occ_gt_root) / self.DATASET_TYPE / occ_variant
        else:
            self.occ_gt_root = Path(occ_gt_root) / self.DATASET_TYPE
        self.num_frames = int(num_frames)
        self.sweeps_num = max(0, self.num_frames - 1)
        self.train_interval = int(train_interval)
        self.test_interval = int(test_interval)
        # ``[0]`` means "key frame only", i.e. plain single-frame occupancy.
        self.future_frames = [0] if future_frames is None else [int(f) for f in future_frames]
        if self.future_frames[0] != 0:
            raise ValueError(
                f"future_frames must start with 0 (the key frame), got {self.future_frames}.")
        if any(b <= a for a, b in zip(self.future_frames, self.future_frames[1:])):
            raise ValueError(
                f"future_frames must be strictly increasing, got {self.future_frames}.")
        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            modality=dict(use_lidar=False, use_camera=True),
            test_mode=test_mode,
            filter_empty_gt=False,
            load_eval_anns=False,
            **kwargs,
        )

    def get_occ_path(self, scene_name: str, frame_token: str) -> Path:
        return self.occ_gt_root / scene_name / f"{frame_token}.npz"

    def prepare_data(self, index: int):
        return self.pipeline(self.get_data_info(index))

    def build_raw_infos(self) -> list:
        get_base_data_info = super().get_data_info
        return [get_base_data_info(idx) for idx in range(len(self))]

    def pack_sample_info(
            self,
            scene_name: str,
            frame_token: str,
            filenames: list,
            occ2img: list,
            fut2cur: Optional[list] = None,
            future_occ_paths: Optional[list] = None,
    ) -> dict:
        """Assemble the dict consumed by the transform pipeline.

        ``fut2cur`` / ``future_occ_paths`` are only set by forecasting datasets.
        Entry ``i`` of both corresponds to ``self.future_frames[i]``, and
        ``fut2cur[0]`` is the identity because horizon 0 is the key frame.
        """
        info = dict(
            data_root=self.data_root,
            dataset_type=self.DATASET_TYPE,
            occ_variant=self.occ_variant,
            scene_name=scene_name,
            frame_token=frame_token,
            filename=filenames,
            occ_path=self.get_occ_path(scene_name, frame_token),
            occ2img=occ2img,
        )
        if fut2cur is not None:
            info["fut2cur"] = fut2cur
            info["future_occ_paths"] = future_occ_paths
            info["fut_list"] = list(self.future_frames)
        return info

    def history_indices(self, cur_pos: int, min_pos: int = 0) -> List[Optional[int]]:
        if self.sweeps_num == 0:
            return []
        interval = self.test_interval if self.test_mode else self.train_interval
        selected: List[Optional[int]] = []
        for step in range(1, self.sweeps_num + 1):
            pos = cur_pos - step * interval
            if pos >= min_pos:
                selected.append(pos)
            elif selected:
                selected.append(selected[-1])
            else:
                selected.append(None)
        return selected
