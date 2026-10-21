from dataclasses import dataclass

import numpy as np


NUS_CAM_TYPES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_FRONT_LEFT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_BACK_RIGHT",
]


@dataclass(frozen=True)
class CarlaOccMeta:
    cam_types: tuple[str, ...]


@dataclass(frozen=True)
class Kitti360Meta:
    cam_type: str
    img_wh: tuple[int, int]
    preprocessed_img_hw: tuple[int, int]
    intrinsic: np.ndarray
    occ_to_optical: np.ndarray

    def intrinsic_for_image(self, height: int | None = None, width: int | None = None) -> np.ndarray:
        """Scale ``P_rect_00`` intrinsics to a target image size."""
        h = int(self.preprocessed_img_hw[0] if height is None else height)
        w = int(self.preprocessed_img_hw[1] if width is None else width)
        calib_w, calib_h = self.img_wh
        k = np.asarray(self.intrinsic, dtype=np.float64).copy()
        k[0, :] *= float(w) / float(calib_w)
        k[1, :] *= float(h) / float(calib_h)
        return k


CARLA_OCC_META = CarlaOccMeta(
    cam_types=("cam_00", "cam_01", "cam_02", "cam_03", "cam_04", "cam_05"),
)


def carlaocc_camera_names(variant: str) -> tuple[str, ...]:
    """Select the cameras used by a CarlaOcc occupancy variant."""
    if str(variant).endswith("_forward_view"):
        return ("cam_00",)
    if str(variant).endswith("_surround_view"):
        return CARLA_OCC_META.cam_types
    raise ValueError(f"Unrecognized CarlaOcc view variant: {variant!r}")


KITTI360_META = Kitti360Meta(
    cam_type="image_00",
    img_wh=(1408, 376),
    preprocessed_img_hw=(192, 640),
    intrinsic=np.array([
        [552.554261, 0.0, 682.049453],
        [0.0, 552.554261, 238.769549],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64),
    occ_to_optical=np.array([
        [0.0, -1.0, 0.0],
        [0.0, 0.0, -1.0],
        [1.0, 0.0, 0.0],
    ], dtype=np.float64),
)
