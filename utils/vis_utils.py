from pathlib import Path
from typing import Optional, Union, Tuple

import numpy as np
import open3d as o3d

from utils.labels import DATASET_META, DatasetMeta


def _occ_dense2sparse(*args, **kwargs):
    from utils.data_converter import occ_dense2sparse

    return occ_dense2sparse(*args, **kwargs)


def _decode_global_id(*args, **kwargs):
    from utils.data_converter import decode_global_id

    return decode_global_id(*args, **kwargs)


class CameraSettingConfig:
    def __init__(
            self,
            image_size: Tuple[int, int] = (1000, 1000),
            focal_length: float = 500.0,
            cam_position: Tuple[float, float, float] = (0., 0., 0.),
            cam_pitch: float = 0.,
    ):
        """
        Camera setting configuration class.

        Args:
            image_size: Image size as (width, height)
            focal_length: Focal length
            cam_position: Camera position (x, y, z) in occupancy coordinates
            cam_pitch: Camera pitch angle in degrees
        """
        self.image_size = image_size
        # Occupancy coordinate: x-forward, y-left, z-up
        # Open3D camera: x-right, y-down, z-forward
        self.extrinsic = np.array([[0, -1, 0, 0],
                                   [0, 0, -1, 0],
                                   [1, 0, 0, 0],
                                   [0, 0, 0, 1]], dtype=np.float64)
        # occupancy --> cam
        rel_pos = -1 * np.array([
            -cam_position[1],
            -cam_position[2],
            cam_position[0]
        ])

        self.extrinsic[:3, 3] = rel_pos

        cam_pitch = np.radians(cam_pitch)
        pitch_mat = np.array([
            [1, 0, 0, 0],
            [0, np.cos(cam_pitch), -np.sin(cam_pitch), 0],
            [0, np.sin(cam_pitch), np.cos(cam_pitch), 0],
            [0, 0, 0, 1]
        ], dtype=np.float64)
        self.extrinsic = pitch_mat @ self.extrinsic

        self.intrinsic = np.array([
            [focal_length, 0.0, image_size[0] / 2],
            [0.0, focal_length, image_size[1] / 2],
            [0.0, 0.0, 1.0]
        ])


class CameraSetting:
    """
    Camera settings class for Open3D visualization with predefined camera configurations.
    """

    def __init__(self,
                 extrinsic: np.ndarray,
                 intrinsic: np.ndarray,
                 image_size: Tuple[int, int]):
        """
        Initialize camera setting with extrinsic and intrinsic parameters.
        
        Args:
            extrinsic: 4x4 extrinsic transformation matrix
            intrinsic: 3x3 intrinsic camera matrix
            image_size: Image dimensions as (width, height)
        """
        assert extrinsic.shape == (4, 4), f"Extrinsic matrix must be 4x4, got {extrinsic.shape}"
        assert intrinsic.shape == (3, 3), f"Intrinsic matrix must be 3x3, got {intrinsic.shape}"
        assert len(image_size) == 2, f"Image size must be (width, height), got {image_size}"

        self.extrinsic = extrinsic
        self.intrinsic = intrinsic
        self.image_size = image_size

    @classmethod
    def from_config(cls, config: CameraSettingConfig) -> 'CameraSetting':
        return cls(config.extrinsic, config.intrinsic, config.image_size)

    @classmethod
    def from_position(
            cls,
            image_size: Tuple[int, int],
            focal_length: float,
            cam_position: Tuple[float, float, float],
            cam_pitch: float,
    ) -> 'CameraSetting':
        return cls.from_config(
            CameraSettingConfig(
                image_size=image_size,
                focal_length=focal_length,
                cam_position=cam_position,
                cam_pitch=cam_pitch,
            )
        )

    @classmethod
    def for_dataset(cls, dataset_type: str) -> 'CameraSetting':
        builders = {
            "panobench_carlaocc": cls.vis_panobench_carlaocc_forward_down_view,
            "panobench_nuscenes": cls.vis_panobench_nuscenes_cam0,
            "panobench_kitti360": cls.vis_kitti360,
        }
        if dataset_type not in builders:
            raise ValueError(f"No camera setting registered for dataset '{dataset_type}'")
        return builders[dataset_type]()

    @classmethod
    def vis_kitti360(cls) -> 'CameraSetting':
        return cls.from_position(
            image_size=(500, 500),
            focal_length=250.0,
            cam_position=(-5.0, 0.0, 20.0),
            cam_pitch=45.0,
        )
        
    @classmethod
    def vis_panobench_carlaocc_forward_down_view(cls) -> 'CameraSetting':
        return cls.from_position(
            image_size=(500, 500),
            focal_length=250.0,
            cam_position=(0., 0., 50.),
            cam_pitch=90,
        )

    @classmethod
    def vis_panobench_nuscenes_cam0(cls) -> 'CameraSetting':
        """PanoBench-nuScenes CAM_FRONT at raw 1600x900 resolution.

        Hard-coded from the original nuScenes CAM_FRONT calibration used by the
        val split's first sensor rig.  The occupancy frame is the keyframe ego
        frame and the extrinsic is ``T_ego_to_cam_front``.
        """
        intrinsic = np.array([
            [1266.417203046554, 0.0, 816.2670197447984],
            [0.0, 1266.417203046554, 491.50706579294757],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        extrinsic = np.array([
            [0.0056847787, -0.9999835174, 0.0008050713, 0.0050603194],
            [-0.0056366677, -0.0008371153, -0.9999837635, 1.5205332496],
            [0.9999679551, 0.0056801485, -0.0056413336, -1.6923034453],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return cls(
            extrinsic=extrinsic,
            intrinsic=intrinsic,
            image_size=(1600, 900),
        )

    def get_pinhole_camera_parameters(self) -> o3d.camera.PinholeCameraParameters:
        """
        Convert camera setting to Open3D PinholeCameraParameters.
        
        Returns:
            Open3D PinholeCameraParameters object
        """
        camera_params = o3d.camera.PinholeCameraParameters()
        camera_params.extrinsic = self.extrinsic
        camera_params.intrinsic.intrinsic_matrix = self.intrinsic
        camera_params.intrinsic.width = self.image_size[0]
        camera_params.intrinsic.height = self.image_size[1]
        return camera_params


# Unit cube mesh centered at the origin.
_CUBE_V = np.array([
    [0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1],
    [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0],
    [0, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 0],
    [0, 1, 0], [1, 1, 0], [1, 1, 1], [0, 1, 1],
    [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
    [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1],
], dtype=np.float32) - 0.5

_CUBE_N = np.repeat(np.array([
    [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1],
], dtype=np.float32), 4, axis=0)

_CUBE_F = (
    np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)[None]
    + 4 * np.arange(6, dtype=np.int64)[:, None, None]
).reshape(-1, 3)


def cubes_mesh(
        centers: np.ndarray,
        colors: np.ndarray,
        size: float,
        shrink: float = 1.0,
) -> o3d.geometry.TriangleMesh:
    """Build a lit cube mesh for each voxel center (comparison-render style)."""
    n = len(centers)
    if n == 0:
        return o3d.geometry.TriangleMesh()
    vertices = (_CUBE_V[None] * (size * shrink) + centers[:, None]).reshape(-1, 3)
    faces = (
        _CUBE_F[None] + 24 * np.arange(n, dtype=np.int64)[:, None, None]
    ).reshape(-1, 3)
    normals = np.broadcast_to(_CUBE_N[None], (n, 24, 3)).reshape(-1, 3)
    vertex_colors = np.repeat(np.asarray(colors, dtype=np.float64), 24, axis=0)
    mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(vertices.astype(np.float64)),
        o3d.utility.Vector3iVector(faces),
    )
    mesh.vertex_normals = o3d.utility.Vector3dVector(normals.astype(np.float64))
    mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    return mesh


def _default_lit_material():
    """PBR material for the offscreen renderer."""
    import open3d.visualization.rendering as rendering

    mat = rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = (1.0, 1.0, 1.0, 1.0)
    mat.base_roughness = 0.65
    mat.base_metallic = 0.0
    mat.base_reflectance = 0.4
    return mat


def _render_filament_interactive(
        geometry: o3d.geometry.TriangleMesh,
        camera_setting: CameraSetting,
        show_axis: bool,
) -> None:
    """Interactive Filament GUI (same backend as the comparison renders)."""
    import open3d.visualization.rendering as rendering

    mat = _default_lit_material()
    geoms = [{"name": "occupancy", "geometry": geometry, "material": mat}]
    if show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        geoms.append({"name": "axis", "geometry": axis, "material": mat})

    sun = np.asarray([0.4, -0.6, -1.0], dtype=np.float64)
    sun /= np.linalg.norm(sun)

    def on_init(o3dvis) -> None:
        o3dvis.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, sun
        )

    o3d.visualization.draw(
        geoms,
        title="occupancy",
        width=int(camera_setting.image_size[0]),
        height=int(camera_setting.image_size[1]),
        intrinsic_matrix=np.asarray(camera_setting.intrinsic, dtype=np.float64),
        extrinsic_matrix=np.asarray(camera_setting.extrinsic, dtype=np.float64),
        bg_color=(1.0, 1.0, 1.0, 1.0),
        show_skybox=False,
        show_ui=False,
        on_init=on_init,
    )


def _render_visualizer(
        geometry: Union[o3d.geometry.VoxelGrid, o3d.geometry.TriangleMesh],
        camera_setting: CameraSetting,
        show_axis: bool,
        save_path: Optional[Union[str, Path]],
) -> None:
    """Legacy Open3D Visualizer (interactive + offline screenshot)."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        width=camera_setting.image_size[0],
        height=camera_setting.image_size[1],
        visible=(save_path is None),
    )
    vis.add_geometry(geometry)

    if show_axis:
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0)
        vis.add_geometry(axis)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([1.0, 1.0, 1.0])
    opt.light_on = True
    if isinstance(geometry, o3d.geometry.TriangleMesh):
        opt.mesh_show_back_face = True
        opt.mesh_color_option = o3d.visualization.MeshColorOption.Color

    ctr = vis.get_view_control()
    camera_params = camera_setting.get_pinhole_camera_parameters()
    ctr.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        vis.capture_screen_image(str(save_path), do_render=True)
    else:
        vis.run()
    vis.destroy_window()


def render_geometry(
        geometry: Union[o3d.geometry.VoxelGrid, o3d.geometry.TriangleMesh],
        camera_setting: CameraSetting,
        show_axis: bool,
        save_path: Optional[Union[str, Path]],
) -> None:
    # Interactive TriangleMesh: prefer Filament GUI (comparison-render look); else Visualizer.
    if save_path is None and isinstance(geometry, o3d.geometry.TriangleMesh):
        try:
            _render_filament_interactive(geometry, camera_setting, show_axis)
            return
        except Exception as exc:
            print(f"[vis] Filament GUI failed ({exc}); falling back to Visualizer")
    _render_visualizer(geometry, camera_setting, show_axis, save_path)


def render_voxel_grid(
        voxel_grid: o3d.geometry.VoxelGrid,
        camera_setting: CameraSetting,
        show_axis: bool,
        save_path: Optional[Union[str, Path]],
) -> None:
    render_geometry(voxel_grid, camera_setting, show_axis, save_path)


def _validate_semantic_ids(sem_ids: np.ndarray, color_lut: np.ndarray, occ_type: str) -> np.ndarray:
    sem_ids = np.asarray(sem_ids, dtype=np.int64)
    invalid = (sem_ids < 0) | (sem_ids >= len(color_lut))
    if np.any(invalid):
        bad_ids = np.unique(sem_ids[invalid])[:10]
        raise ValueError(
            f"{occ_type} contains semantic ids outside color LUT range "
            f"[0, {len(color_lut) - 1}]: {bad_ids.tolist()}"
        )
    return sem_ids.astype(np.intp, copy=False)


def _voxel_centers(occ: np.ndarray, voxel_size: float, voxel_origin) -> np.ndarray:
    # Sparse occupancy stores voxel indices; Open3D point-cloud voxelization expects centers.
    voxel_centers = (occ[:, :3].astype(np.float32) + 0.5) * voxel_size + np.asarray(voxel_origin, dtype=np.float32)
    return voxel_centers


def vis_sem_occ(occ: np.ndarray,
                occ_type: str = 'panobench_kitti360',
                camera_setting: Optional[CameraSetting] = None,
                show_axis: bool = False,
                save_path: Optional[Union[str, Path]] = None,
                occ_meta: Optional[DatasetMeta] = None) -> None:
    """
    Visualize 3D semantic occupancy grid using Open3D.

    Args:
        occ: Semantic occupancy grid in Nx4 (x, y, z, sem_or_global_id) or XxYxZ format
        occ_type: Occupancy type defined in utils/labels.py
        camera_setting: Optional CameraSetting object for predefined camera configurations
        show_axis: Whether to display the coordinate frame
        save_path: Optional path to save visualization image
        occ_meta: Optional pre-resolved grid metadata (needed for multi-variant datasets).
    """
    assert occ.ndim == 3 or (occ.ndim == 2 and occ.shape[1] == 4), "Occupancy grid must be in shape of XxYxZ or Nx4"
    if camera_setting is None:
        camera_setting = CameraSetting.for_dataset(occ_type)
    occ_metas = occ_meta if occ_meta is not None else DATASET_META[(occ_type, None)]
    voxel_size = occ_metas.voxel_size
    voxel_origin = occ_metas.voxel_origin
    color_lut = occ_metas.color_lut
    free_id = occ_metas.free_id

    if occ.ndim == 3:
        occ = _occ_dense2sparse(occ, free_id=free_id)

    points = _voxel_centers(occ, voxel_size, voxel_origin)
    labels = np.asarray(occ[:, 3], dtype=np.int64)
    # Accept panoptic packing ``sem * 1000 + inst`` as well as bare semantic ids.
    if labels.size and int(labels.max()) >= 1000:
        labels = labels // 1000
    sem_ids = _validate_semantic_ids(labels, color_lut, occ_type)
    colors = lighten_lut(color_lut)[sem_ids]

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # Convert point cloud to voxel grid
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

    render_voxel_grid(voxel_grid, camera_setting, show_axis, save_path)


def vis_pano_occ(occ: np.ndarray,
                 occ_type: str = 'panobench_kitti360',
                 camera_setting: Optional[CameraSetting] = None,
                 show_axis: bool = False,
                 save_path: Optional[Union[str, Path]] = None,
                 occ_meta: Optional[DatasetMeta] = None,
                 meshlike: bool = False) -> None:
    """
    Visualize 3D panoptic occupancy grid using Open3D.

    Coloring: lightened semantic LUT; thing voxels with ``instance_id > 0``
    keep the semantic hue and add a ``sin(id)`` RGB offset for instance contrast.

    Args:
        occ: Panoptic occupancy grid in Nx4 (x, y, z, global_id) or XxYxZ format.
        occ_type: Occupancy type defined in utils/labels.py
        camera_setting: Optional CameraSetting object for predefined camera configurations
        show_axis: Whether to display the coordinate frame
        save_path: Optional path to save visualization image
        occ_meta: Optional pre-resolved grid metadata (required for multi-variant canonical GT).
        meshlike: If True, render lit voxel cubes as a TriangleMesh.
            If False (default), use Open3D VoxelGrid.
    """
    assert occ.ndim == 3 or (occ.ndim == 2 and occ.shape[1] == 4), "Occupancy grid must be in shape of XxYxZ or Nx4"
    if camera_setting is None:
        camera_setting = CameraSetting.for_dataset(occ_type)
    occ_metas = occ_meta if occ_meta is not None else DATASET_META[(occ_type, None)]
    voxel_size = occ_metas.voxel_size
    voxel_origin = occ_metas.voxel_origin
    color_lut = occ_metas.color_lut
    free_id = occ_metas.free_id

    if occ.ndim == 3:
        occ = _occ_dense2sparse(occ, free_id=free_id)

    points = _voxel_centers(occ, voxel_size, voxel_origin)
    global_ids = occ[:, 3].astype(np.uint16)
    sem_ids, inst_ids = _decode_global_id(global_ids)
    sem_ids = _validate_semantic_ids(sem_ids, color_lut, occ_type)

    colors = panoptic_voxel_colors(
        sem_ids, inst_ids, color_lut, thing_ids=occ_metas.thing_ids
    )

    if meshlike:
        geometry = cubes_mesh(points, colors, voxel_size)
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        geometry = o3d.geometry.VoxelGrid.create_from_point_cloud(
            pcd, voxel_size=voxel_size
        )

    render_geometry(geometry, camera_setting, show_axis, save_path)


def lighten_lut(lut: np.ndarray, factor: float = 0.0) -> np.ndarray:
    """Push LUT colors toward white (comparison ``lighten_lut``)."""
    lut = np.asarray(lut, dtype=np.float64)
    return lut + (1.0 - lut) * factor


def apply_instance_color(base_colors, instance_ids) -> np.ndarray:
    """Tint semantic colors by a per-instance ``sin(id)`` RGB offset.

    ``instance_id == 0`` keeps the base color unchanged.
    """
    colors = np.asarray(base_colors, dtype=np.float64).copy()
    instance_ids = np.asarray(instance_ids, dtype=np.float64)
    offsets = np.column_stack(
        [
            np.sin(instance_ids * 0.1) * 0.4,
            np.sin(instance_ids * 0.3) * 0.4,
            np.sin(instance_ids * 0.5) * 0.4,
        ]
    )
    return np.clip(colors + offsets, 0.0, 1.0)


def panoptic_voxel_colors(
        sem_ids: np.ndarray,
        inst_ids: np.ndarray,
        color_lut: np.ndarray,
        thing_ids: frozenset | None = None,
) -> np.ndarray:
    """Semantic LUT (+ optional lighten); things with ``inst > 0`` get ``sin(id)`` tint."""
    sem_ids = np.asarray(sem_ids)
    inst_ids = np.asarray(inst_ids)
    colors = lighten_lut(color_lut)[sem_ids]
    if not thing_ids:
        return apply_instance_color(colors, inst_ids)
    mask = np.isin(sem_ids, list(thing_ids)) & (inst_ids > 0)
    if not np.any(mask):
        return colors
    colors = np.asarray(colors, dtype=np.float64).copy()
    colors[mask] = apply_instance_color(colors[mask], inst_ids[mask])
    return colors
