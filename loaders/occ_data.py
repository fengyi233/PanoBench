"""Data containers for PanoBench occupancy dataloading."""

from mmengine.structures import BaseDataElement


class OccData(BaseDataElement):
    """Container for occupancy fields.

    ``OccData`` intentionally has no fixed schema. The active fields depend on
    whether it is attached as ``gt_occ`` or ``pred_occ`` on
    :class:`PanoBenchOccDataSample`.

    Ground-truth fields written by ``LoadCanonicalOcc`` / ``PackInputs``:
        voxel_semantics (Tensor):
            Dense semantic occupancy grid with shape ``[X, Y, Z]``. Values are
            canonical PanoBench class ids, with the dataset ``free_id`` used for
            free voxels.
        valid_mask (Tensor):
            Dense boolean grid with shape ``[X, Y, Z]``. ``True`` marks voxels
            that participate in supervised loss / masked evaluation.
        voxel_instances (Tensor, optional):
            Dense compact instance/stuff target grid with shape ``[X, Y, Z]``.
            This is loaded only when ``LoadCanonicalOcc(load_instances=True)``
            is used, currently for SparseOcc instance supervision.
        instance_class_ids (Tensor, optional):
            One-dimensional tensor where index ``i`` stores the semantic class
            id of compact target id ``i`` in ``voxel_instances``.

    Prediction fields written by model ``predict`` methods:
        semantics (Tensor | ndarray):
            Sparse semantic predictions aligned with ``occ_loc``.
        occ_loc (Tensor | ndarray):
            Sparse voxel coordinates. The evaluator flattens the prediction and
            converts ``occ_loc + semantics`` back to a dense grid.
        pano_sem / pano_inst (Tensor | ndarray, optional):
            Sparse panoptic semantic and instance predictions aligned with
            ``occ_loc``. These are required only when evaluating PQ / RayPQ.
    """


class PanoBenchOccDataSample(BaseDataElement):
    """Data sample contract for one PanoBench occupancy frame.

    ``inputs`` are deliberately not stored on this object. ``PackInputs``
    returns them in the batch-level ``inputs`` dict:
        img (Tensor):
            Multi-frame multi-view image tensor with shape ``[N, C, H, W]``
            before collation and ``[B, N, C, H, W]`` after
            ``default_collate``.
        occ2img (Tensor):
            Projection matrices with shape ``[N, 4, 4]`` before collation and
            ``[B, N, 4, 4]`` after ``default_collate``.

    Required ``metainfo`` keys:
        filename (List[str]):
            Image paths for the key frame and history frames. These are used by
            feature caching and as an evaluator fallback when an ``ann_file`` is
            present.
        frame_token (str):
            Canonical frame token. The evaluator can use it to map predictions
            back to ``ann_file`` records.
        occ_path (str | Path):
            Canonical occupancy ``.npz`` path for the sample. The evaluator
            uses it to load ground truth directly.

    Data fields:
        gt_occ (OccData):
            Dense ground truth. Expected fields are ``voxel_semantics`` and
            ``valid_mask``; SparseOcc training may also require
            ``voxel_instances`` and ``instance_class_ids``.
        pred_occ (OccData):
            Sparse model prediction attached by ``predict``. Expected evaluator
            fields are ``occ_loc`` and ``semantics``. Panoptic evaluation also
            expects ``pano_sem`` and ``pano_inst``.
    """
