import os.path as osp

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmdet3d.registry import TRANSFORMS
from mmengine.dist import get_dist_info

from utils.labels import get_dataset_meta
from utils.metas import NUS_CAM_TYPES
from utils.panoptic_occupancy import PanopticOccupancy

try:
    mmcv.use_backend("turbojpeg")
    _DEFAULT_BACKEND = "turbojpeg"
except (ImportError, RuntimeError):
    mmcv.use_backend("cv2")
    _DEFAULT_BACKEND = "cv2"


def _imread_backend(path):
    # turbojpeg only decodes JPEG; fall back to cv2 for PNG/other (e.g. CarlaOcc).
    if _DEFAULT_BACKEND == "turbojpeg" and not path.lower().endswith((".jpg", ".jpeg")):
        return "cv2"
    return _DEFAULT_BACKEND


@TRANSFORMS.register_module()
class LoadMultiViewImages(BaseTransform):

    def __init__(
            self,
            test_mode=False,
            online=True,
            num_cams=None,
    ):
        super().__init__()
        self.num_cams = len(NUS_CAM_TYPES) if num_cams is None else int(num_cams)
        # Single-GPU eval can load only the keyframe and reuse cached history
        # features, but that assumes each scene is processed from its first
        # frame (history = self-replay). Datasets whose val split starts mid
        # sequence (CarlaOcc) must set ``online=False`` to load every frame.
        self.online_test = online and test_mode and get_dist_info()[1] == 1

    def transform(self, results):
        data_root = results["data_root"]
        filenames = results["filename"]
        results["filename"] = [osp.join(data_root, f) for f in filenames]
        if self.online_test:
            # Single-GPU eval: load keyframe only; history frames reuse backbone cache.
            load_paths = results["filename"][:self.num_cams]
        else:
            load_paths = results["filename"]
        results["img"] = [mmcv.imread(p, "color", backend=_imread_backend(p)) for p in load_paths]  # HWC, uint8, BGR
        return results


INSTANCE_IGNORE_ID = 255


def _assign_sparseocc_instances(
        semantics: np.ndarray,
        instances: np.ndarray | None,
        dataset_type: str,
        thing_ids: frozenset[int],
        free_id: int,
        missing_instance_policy: str = "error",
) -> tuple[np.ndarray, np.ndarray]:
    """Build SparseOcc instance targets from dense semantic/instance grids.

    ``voxel_instances`` uses ``255`` as the ignore/free sentinel. Each thing
    instance gets its own target id. Each stuff semantic class gets one target id,
    matching the original SparseOcc loader convention.
    """
    if missing_instance_policy not in {"error", "class_pseudo_instance"}:
        raise ValueError(
            "missing_instance_policy must be 'error' or "
            f"'class_pseudo_instance', got {missing_instance_policy!r}.")

    voxel_instances = np.full(semantics.shape, INSTANCE_IGNORE_ID, dtype=np.int64)
    class_ids = []
    next_id = 0

    for cls_id in np.unique(semantics):
        cls_id = int(cls_id)
        if cls_id == free_id:
            continue
        cls_mask = semantics == cls_id
        if not np.any(cls_mask):
            continue

        if cls_id in thing_ids:
            raw_inst_ids = np.unique(instances[cls_mask])
            if np.any(raw_inst_ids == 0) and missing_instance_policy == "error":
                raise ValueError(
                    f"{dataset_type} semantic class {cls_id} has instance_id=0 "
                    "in canonical occupancy."
                )
            for raw_inst_id in raw_inst_ids:
                if next_id >= INSTANCE_IGNORE_ID:
                    raise ValueError(
                        "SparseOcc instance target uses 255 as ignore id, but "
                        f"this sample has at least {next_id + 1} targets."
                    )
                voxel_instances[cls_mask & (instances == raw_inst_id)] = next_id
                class_ids.append(cls_id)
                next_id += 1
        else:
            if next_id >= INSTANCE_IGNORE_ID:
                raise ValueError(
                    "SparseOcc instance target uses 255 as ignore id, but "
                    f"this sample has at least {next_id + 1} targets."
                )
            voxel_instances[cls_mask] = next_id
            class_ids.append(cls_id)
            next_id += 1

    return voxel_instances, np.asarray(class_ids, dtype=np.int64)


def _load_canonical_sample(occ_path, meta):
    sample = PanopticOccupancy.load(occ_path)
    if sample.volume_size != tuple(meta.volume_size):
        raise ValueError(
            f"{occ_path} volume_size={sample.volume_size} does not "
            f"match dataset metadata volume_size={tuple(meta.volume_size)}."
        )
    return sample


def _resolve_valid_mask(sample, mask_key, owner):
    if mask_key is None:
        return np.ones(sample.volume_size, dtype=np.bool_)
    if mask_key == "valid_mask":
        return sample.valid_mask
    raise KeyError(
        f"{owner} only supports mask_key='valid_mask' in the "
        "new packed dense mask format."
    )


@TRANSFORMS.register_module()
class LoadCanonicalOcc(BaseTransform):

    def __init__(self,
                 mask_key=None,
                 load_instances: bool = False,
                 missing_instance_policy: str = "error"):
        super().__init__()
        self.mask_key = mask_key
        self.load_instances = bool(load_instances)
        if missing_instance_policy not in {"error", "class_pseudo_instance"}:
            raise ValueError(
                "missing_instance_policy must be 'error' or "
                f"'class_pseudo_instance', got {missing_instance_policy!r}.")
        self.missing_instance_policy = missing_instance_policy

    def transform(self, results):
        dataset_type = results["dataset_type"]
        occ_variant = results["occ_variant"]
        meta = get_dataset_meta(dataset_type, occ_variant)
        sample = _load_canonical_sample(results["occ_path"], meta)

        free_id = meta.free_id
        semantics = sample.dense_semantic(free_id, dtype=np.uint8)
        if self.load_instances:
            instances = sample.dense_instance(dtype=np.int64)

        valid_mask = _resolve_valid_mask(sample, self.mask_key, "LoadCanonicalOcc")

        results["voxel_semantics"] = semantics
        results["valid_mask"] = valid_mask
        if self.load_instances:
            voxel_instances, instance_class_ids = _assign_sparseocc_instances(
                semantics,
                instances,
                dataset_type,
                meta.thing_ids,
                free_id,
                missing_instance_policy=self.missing_instance_policy,
            )
            results["voxel_instances"] = voxel_instances
            results["instance_class_ids"] = instance_class_ids
        return results


@TRANSFORMS.register_module()
class LoadCanonicalOcc4D(BaseTransform):
    """Load one canonical occupancy target per forecast horizon.

    Reads ``future_occ_paths`` (written by a forecasting dataset) and stacks
    the horizons along a leading axis, so ``voxel_semantics`` and
    ``valid_mask`` become ``[FT, X, Y, Z]``. Each horizon's grid stays in its
    own ego frame; ``fut2cur`` is what relates them.
    """

    def __init__(self, mask_key=None):
        super().__init__()
        self.mask_key = mask_key

    def transform(self, results):
        meta = get_dataset_meta(results["dataset_type"], results["occ_variant"])

        semantics, valid_masks = [], []
        for occ_path in results["future_occ_paths"]:
            sample = _load_canonical_sample(occ_path, meta)
            semantics.append(sample.dense_semantic(meta.free_id, dtype=np.uint8))
            valid_masks.append(
                _resolve_valid_mask(sample, self.mask_key, "LoadCanonicalOcc4D"))

        results["voxel_semantics"] = np.stack(semantics, axis=0)
        results["valid_mask"] = np.stack(valid_masks, axis=0)
        return results
