"""
Dataset metas for occupancy datasets
"""
from collections import namedtuple
from dataclasses import dataclass

import numpy as np

# ========================================================================================
# PanoBench-nuScenes: 17 classes, NuScenes
# ========================================================================================
PanoBenchNuScenesLabel = namedtuple("PanoBenchNuScenesLabel", ["name", "id", "is_instance", "color"])
panobench_nuscenes_labels = [
    PanoBenchNuScenesLabel("free"                ,  0, False, (0, 0, 0)),
    PanoBenchNuScenesLabel("barrier"             ,  1, True , (112, 128, 144)),
    PanoBenchNuScenesLabel("bicycle"             ,  2, True , (220, 20, 60)),
    PanoBenchNuScenesLabel("bus"                 ,  3, True , (255, 127, 80)),
    PanoBenchNuScenesLabel("car"                 ,  4, True , (255, 158, 0)),
    PanoBenchNuScenesLabel("construction_vehicle",  5, True , (233, 150, 70)),
    PanoBenchNuScenesLabel("motorcycle"          ,  6, True , (255, 61, 99)),
    PanoBenchNuScenesLabel("pedestrian"          ,  7, True , (0, 0, 230)),
    PanoBenchNuScenesLabel("traffic_cone"        ,  8, True , (47, 79, 79)),
    PanoBenchNuScenesLabel("trailer"             ,  9, True , (255, 140, 0)),
    PanoBenchNuScenesLabel("truck"               , 10, True , (255, 99, 71)),
    PanoBenchNuScenesLabel("driveable_surface"   , 11, False, (0, 207, 191)),
    PanoBenchNuScenesLabel("other_flat"          , 12, False, (175, 0, 75)),
    PanoBenchNuScenesLabel("sidewalk"            , 13, False, (75, 0, 75)),
    PanoBenchNuScenesLabel("terrain"             , 14, False, (112, 180, 60)),
    PanoBenchNuScenesLabel("manmade"             , 15, False, (222, 184, 135)),
    PanoBenchNuScenesLabel("vegetation"          , 16, False, (0, 175, 0)),
]

# ========================================================================================
# PanoBench-KITTI360: 19 classes, KITTI-360
# ========================================================================================
PanoBenchKITTI360Label = namedtuple("PanoBenchKITTI360Label", ["name", "id", "is_instance", "color"])
panobench_kitti360_labels = [
    PanoBenchKITTI360Label("free"           ,  0, False, (0, 0, 0)),
    PanoBenchKITTI360Label("car"            ,  1, True , (100, 150, 245)),
    PanoBenchKITTI360Label("bicycle"        ,  2, True , (100, 230, 245)),
    PanoBenchKITTI360Label("motorcycle"     ,  3, True , (30, 60, 150)),
    PanoBenchKITTI360Label("truck"          ,  4, True , (80, 30, 180)),
    PanoBenchKITTI360Label("other-vehicle"  ,  5, True , (0, 0, 255)),
    PanoBenchKITTI360Label("person"         ,  6, True , (255, 30, 30)),
    PanoBenchKITTI360Label("road"           ,  7, False, (255, 0, 255)),
    PanoBenchKITTI360Label("parking"        ,  8, False, (255, 150, 255)),
    PanoBenchKITTI360Label("sidewalk"       ,  9, False, (75, 0, 75)),
    PanoBenchKITTI360Label("other-ground"   , 10, False, (175, 0, 75)),
    PanoBenchKITTI360Label("building"       , 11, False, (255, 200, 0)),
    PanoBenchKITTI360Label("fence"          , 12, False, (255, 120, 50)),
    PanoBenchKITTI360Label("vegetation"     , 13, False, (0, 175, 0)),
    PanoBenchKITTI360Label("terrain"        , 14, False, (150, 240, 80)),
    PanoBenchKITTI360Label("pole"           , 15, False, (255, 240, 150)),
    PanoBenchKITTI360Label("traffic-sign"   , 16, False, (255, 0, 0)),
    PanoBenchKITTI360Label("other-structure", 17, False, (255, 150, 0)),
    PanoBenchKITTI360Label("other-object"   , 18, False, (50, 255, 255)),
]

# ========================================================================================
# PanoBench-CarlaOcc: 30 classes, CarlaOcc
# ========================================================================================
PanoBenchCarlaOccLabel = namedtuple("PanoBenchCarlaOccLabel", ["name", "id", "is_instance", "color"])
panobench_carlaocc_labels = [
    PanoBenchCarlaOccLabel("None"        ,  0, False, (0, 0, 0)),
    PanoBenchCarlaOccLabel("Road"        ,  1, False, (128, 64, 128)),
    PanoBenchCarlaOccLabel("Sidewalk"    ,  2, False, (244, 35, 232)),
    PanoBenchCarlaOccLabel("Building"    ,  3, False, (70, 70, 70)),
    PanoBenchCarlaOccLabel("Wall"        ,  4, False, (102, 102, 156)),
    PanoBenchCarlaOccLabel("Fence"       ,  5, False, (190, 153, 153)),
    PanoBenchCarlaOccLabel("Pole"        ,  6, True , (153, 153, 153)),
    PanoBenchCarlaOccLabel("TrafficLight",  7, True , (250, 170, 30)),
    PanoBenchCarlaOccLabel("TrafficSign" ,  8, True , (220, 220, 0)),
    PanoBenchCarlaOccLabel("Vegetation"  ,  9, False, (107, 142, 35)),
    PanoBenchCarlaOccLabel("Terrain"     , 10, False, (152, 251, 152)),
    PanoBenchCarlaOccLabel("Sky"         , 11, False, (70, 130, 180)),
    PanoBenchCarlaOccLabel("Pedestrian"  , 12, True , (220, 20, 60)),
    PanoBenchCarlaOccLabel("Rider"       , 13, True , (255, 0, 0)),
    PanoBenchCarlaOccLabel("Car"         , 14, True , (0, 0, 142)),
    PanoBenchCarlaOccLabel("Truck"       , 15, True , (0, 0, 70)),
    PanoBenchCarlaOccLabel("Bus"         , 16, True , (0, 60, 100)),
    PanoBenchCarlaOccLabel("Train"       , 17, True , (0, 80, 100)),
    PanoBenchCarlaOccLabel("Motorcycle"  , 18, True , (0, 0, 230)),
    PanoBenchCarlaOccLabel("Bicycle"     , 19, True , (119, 11, 32)),
    PanoBenchCarlaOccLabel("Static"      , 20, False, (110, 190, 160)),
    PanoBenchCarlaOccLabel("Dynamic"     , 21, False, (170, 120, 50)),
    PanoBenchCarlaOccLabel("Other"       , 22, False, (55, 90, 80)),
    PanoBenchCarlaOccLabel("Water"       , 23, False, (45, 60, 150)),
    PanoBenchCarlaOccLabel("RoadLine"    , 24, False, (157, 234, 50)),
    PanoBenchCarlaOccLabel("Ground"      , 25, False, (81, 0, 81)),
    PanoBenchCarlaOccLabel("Bridge"      , 26, False, (150, 100, 100)),
    PanoBenchCarlaOccLabel("RailTrack"   , 27, False, (230, 150, 140)),
    PanoBenchCarlaOccLabel("GuardRail"   , 28, False, (180, 165, 180)),
    PanoBenchCarlaOccLabel("Rock"        , 29, False, (180, 130, 70)),
]

# ========================================================================================
# CONSTANTS
# ========================================================================================
CLASS_NAMES = {
    'panobench_nuscenes': [lbl.name for lbl in panobench_nuscenes_labels],
    'panobench_kitti360': [lbl.name for lbl in panobench_kitti360_labels],
    'panobench_carlaocc': [lbl.name for lbl in panobench_carlaocc_labels],
}

THING_IDS = {
    'panobench_nuscenes': frozenset(lbl.id for lbl in panobench_nuscenes_labels if lbl.is_instance),
    'panobench_kitti360': frozenset(lbl.id for lbl in panobench_kitti360_labels if lbl.is_instance),
    'panobench_carlaocc': frozenset(lbl.id for lbl in panobench_carlaocc_labels if lbl.is_instance),
}

VOXEL_SIZES = {
    'panobench_nuscenes': {'vs_0.2': 0.2, 
                           'vs_0.4': 0.4},
    'panobench_kitti360': {None: 0.2},
    'panobench_carlaocc': {'vs_0.1_forward_view': 0.1,
                           'vs_0.1_surround_view': 0.1,
                           'vs_0.2_forward_view': 0.2,
                           'vs_0.2_surround_view': 0.2,
                           'vs_0.4_forward_view': 0.4,
                           'vs_0.4_surround_view': 0.4},
}

# Voxel origin in LiDAR/Ego Coordinate System
VOXEL_ORIGINS = {
    'panobench_nuscenes': {'vs_0.2': (-40.0, -40.0, -2.0),
                           'vs_0.4': (-40.0, -40.0, -2.0)},
    'panobench_kitti360': {None: (0.0, -25.6, -2.0)},
    'panobench_carlaocc': {'vs_0.1_forward_view': (0.0, -25.6, -2.4),
                           'vs_0.1_surround_view': (-25.6, -25.6, -2.4),
                           'vs_0.2_forward_view': (0.0, -25.6, -2.4),
                           'vs_0.2_surround_view': (-25.6, -25.6, -2.4),
                           'vs_0.4_forward_view': (0.0, -25.6, -2.4),
                           'vs_0.4_surround_view': (-25.6, -25.6, -2.4)},
}

# Grid shape (X, Y, Z)
VOLUME_SIZES = {
    'panobench_nuscenes': {'vs_0.2': (400, 400, 32), 
                           'vs_0.4': (200, 200, 16)},
    'panobench_kitti360': {None: (256, 256, 32)},
    'panobench_carlaocc': {'vs_0.1_forward_view': (512, 512, 64),
                           'vs_0.1_surround_view': (512, 512, 64),
                           'vs_0.2_forward_view': (256, 256, 32),
                           'vs_0.2_surround_view': (256, 256, 32),
                           'vs_0.4_forward_view': (128, 128, 16),
                           'vs_0.4_surround_view': (128, 128, 16)},
}

FREE_ID = {
    'panobench_nuscenes': 0,
    'panobench_kitti360': 0,
    'panobench_carlaocc': 0,
}

COLOR_LUTS = {
    'panobench_nuscenes': np.array([np.array(lbl.color) / 255.0 for lbl in panobench_nuscenes_labels], dtype=np.float32),
    'panobench_kitti360': np.array([np.array(lbl.color) / 255.0 for lbl in panobench_kitti360_labels], dtype=np.float32),
    'panobench_carlaocc': np.array([np.array(lbl.color) / 255.0 for lbl in panobench_carlaocc_labels], dtype=np.float32),
}


# ========================================================================================
# DATASET METAS
# ========================================================================================

DATASET_VARIANTS = {
    'panobench_nuscenes': ('vs_0.2', 'vs_0.4'),
    'panobench_kitti360': (None,),
    'panobench_carlaocc': ('vs_0.1_forward_view',
                           'vs_0.1_surround_view',
                           'vs_0.2_forward_view',
                           'vs_0.2_surround_view',
                           'vs_0.4_forward_view',
                           'vs_0.4_surround_view',),
}

@dataclass(frozen=True)
class DatasetMeta:
    gt_type: str
    variant: str | None
    class_names: list[str]
    color_lut: np.ndarray
    voxel_size: float
    voxel_origin: tuple[float, float, float]
    volume_size: tuple[int, int, int]
    free_id: int
    thing_ids: frozenset[int]

    @property
    def num_classes(self) -> int:
        return len(self.class_names)

    @property
    def occupied_class_ids(self) -> tuple[int, ...]:
        return tuple(idx for idx in range(self.num_classes) if idx != self.free_id)

    @property
    def voxel_size_3d(self) -> tuple[float, float, float]:
        return (self.voxel_size, self.voxel_size, self.voxel_size)

    @property
    def point_cloud_range(self) -> tuple[float, float, float, float, float, float]:
        return (
            self.voxel_origin[0],
            self.voxel_origin[1],
            self.voxel_origin[2],
            self.voxel_origin[0] + self.volume_size[0] * self.voxel_size,
            self.voxel_origin[1] + self.volume_size[1] * self.voxel_size,
            self.voxel_origin[2] + self.volume_size[2] * self.voxel_size,
        )

DATASET_META = {
    (gt_type, variant): DatasetMeta(
        gt_type=gt_type,
        variant=variant,
        class_names=CLASS_NAMES[gt_type],
        color_lut=COLOR_LUTS[gt_type],
        voxel_size=VOXEL_SIZES[gt_type][variant],
        voxel_origin=VOXEL_ORIGINS[gt_type][variant],
        volume_size=VOLUME_SIZES[gt_type][variant],
        free_id=FREE_ID[gt_type],
        thing_ids=THING_IDS[gt_type],
    )
    for gt_type, variants in DATASET_VARIANTS.items()
    for variant in variants
}

def get_dataset_meta(gt_type: str, variant) -> DatasetMeta:
    key = (gt_type, variant)
    if key not in DATASET_META:
        raise KeyError(f"Unknown dataset type: gt_type={gt_type!r}, variant={variant!r}")
    return DATASET_META[key]
