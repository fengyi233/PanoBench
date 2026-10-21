# Training, evaluation and visualization

Follow [INSTALL.md](INSTALL.md) and [DATA.md](DATA.md) first. Run commands from the repository root.

## Backbone initialization

OPUS-V1, SparseOcc, GaussianFormer-V1 and SparseWorld-TC use the nuImages ResNet-50 backbone provided by MMDetection3D:

```bash
mkdir -p pretrain
wget -O pretrain/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth \
  https://download.openmmlab.com/mmdetection3d/v0.1.0_models/nuimages_semseg/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim/cascade_mask_rcnn_r50_fpn_coco-20e_20e_nuim_20201009_124951-40963960.pth
```

FlashOCC uses the ImageNet-1K V1 ResNet-50 checkpoint:

```bash
wget -O pretrain/resnet50-0676ba61.pth \
  https://download.pytorch.org/models/resnet50-0676ba61.pth
```

The configurations load both files directly from `pretrain/`. Existing copies can be reused without downloading them again. These are backbone initialization weights; benchmark checkpoints are listed in the [README pretrained weights section](../README.md#pretrained-weights).

## Training

Configuration names use `experiments/<model>/<dataset>` without the `.yaml` suffix.

- Models: `opusv1`, `sparseocc`, `flashocc`, `gaussianformerv1`, `sparseworldtc`.
- Datasets: `panobench_carlaocc`, `panobench_nuscenes`, `panobench_kitti360`.
- CarlaOcc variants: `vs_0.1_forward_view`, `vs_0.1_surround_view`, `vs_0.2_forward_view`, `vs_0.2_surround_view`, `vs_0.4_forward_view`, `vs_0.4_surround_view`.
- nuScenes variants: `vs_0.2`, `vs_0.4`.
- KITTI360 uses `occ_variant=null` and no variant subdirectory.

```bash
bash scripts/train.sh experiments/opusv1/panobench_carlaocc vs_0.4_surround_view
GPUS=8 bash scripts/train.sh experiments/sparseocc/panobench_nuscenes vs_0.4
bash scripts/train.sh experiments/flashocc/panobench_kitti360
bash scripts/train.sh experiments/sparseworldtc/panobench_carlaocc vs_0.4_surround_view
```

Hydra overrides can be appended to any command:

```bash
bash scripts/train.sh experiments/opusv1/panobench_carlaocc vs_0.4_forward_view \
  data.train_bs=1 train_dataloader.num_workers=2
```

`data.train_bs` is the batch size **per GPU**. Changing GPU count or batch size changes the effective training batch size; a small-memory run is not automatically a reproduction of the paper's training schedule.

The forward-view CarlaOcc variant selects only `image_00` for both current and historical observations. Surround-view variants use six cameras. OPUS-V1 excludes free space from classifier outputs and maps occupied dataset IDs to contiguous classifier indices.

Training writes checkpoints and logs to `outputs/<config>/<variant>/<timestamp>/`, with `default` used for KITTI360. The shell entrypoints in `scripts/` locate the repository root automatically; relative paths in their arguments are interpreted from that root.

## Evaluation

Published benchmark checkpoints are available on [Hugging Face](https://huggingface.co/fengyi233/PanoBench). Download one checkpoint into `checkpoints/`:

```bash
hf download fengyi233/PanoBench \
  opusv1_nuscenes_vs_0.4.pth \
  --local-dir checkpoints
```

Install the optional download CLI with `pip install huggingface_hub` if needed. The [model card](https://huggingface.co/fengyi233/PanoBench#measured-validation-results) reports measured validation results. Paper-reported results are available in the [paper](https://arxiv.org/abs/2610.23537).

Use the configuration and variant listed for the selected checkpoint in the [README pretrained weights section](../README.md#pretrained-weights). Replace `CHECKPOINT` below with its local filename.

Semantic occupancy, including the three RayIoU thresholds:

```bash
bash scripts/eval.sh experiments/opusv1/panobench_carlaocc \
  CHECKPOINT vs_0.4_surround_view \
  'val_evaluator.metrics.0.metrics=[iou,miou,rayiou]'
```

Panoptic occupancy with SparseOcc or FlashOCC:

```bash
bash scripts/eval.sh experiments/sparseocc/panobench_nuscenes \
  CHECKPOINT vs_0.4 \
  'val_evaluator.metrics.0.metrics=[iou,miou,rayiou,pq,raypq]'
```

KITTI360 has no variant argument:

```bash
bash scripts/eval.sh experiments/flashocc/panobench_kitti360 CHECKPOINT
```

SparseWorld-TC uses its forecasting evaluator:

```bash
bash scripts/eval.sh experiments/sparseworldtc/panobench_carlaocc \
  CHECKPOINT vs_0.4_surround_view
```

Its default horizons correspond to the current frame and 1 s, 2 s and 3 s into the future. Samples without all required future labels or poses are excluded by the forecasting dataset. KITTI360 horizons use raw frame IDs, so gaps in the released frame list do not change their elapsed time.

KITTI360 also requires every forecast target to occur in its released split list. The resulting forecasting split contains 49,299 training and 12,782 validation samples. Historical KITTI360 forecasting metrics refer to the original evaluation scope; no new full validation on this restricted split is reported.

Evaluation is restricted to the released `valid_mask`; unmasked space must not be treated as evaluated free space. Add `pred_occ_save_dir=outputs/pred_occ` to save prediction archives. Full-resolution evaluation may need a lower `val_dataloader.batch_size` than training.

Evaluation logs are saved beside the checkpoint in `<checkpoint_stem>_eval/` by default, or in the directory specified by `work_dir=...`. Hydra does not create separate `train.log` or `eval.log` files in the repository root.

## Visualization

View released PanoBench labels with the canonical reader:

```bash
bash scripts/vis_canonical_occ.sh canonical=panobench_carlaocc \
  canonical.variant=vs_0.4_forward_view \
  canonical.scene_name=Town10HD_Opt_Seq00 canonical.frame_token=0200
```

Use `canonical=panobench_nuscenes` or `canonical=panobench_kitti360` for the other subsets. The packed `valid_mask` is applied by default. Ground-truth renders are saved under `outputs/vis_occ_gt/`. The `vis_valid_mask.py` viewer can additionally display valid free space.

Render predictions alongside ground truth:

```bash
python vis_pred_occ.py \
  --config-name experiments/opusv1/panobench_carlaocc \
  --checkpoint CHECKPOINT --occ-variant vs_0.4_surround_view \
  --index 0 --save-gt
```

Select a particular scene and frame:

```bash
python vis_pred_occ.py \
  --config-name experiments/sparseworldtc/panobench_carlaocc \
  --checkpoint CHECKPOINT --occ-variant vs_0.4_surround_view \
  --scene Town10HD_Opt_Seq00 --frame 0200 --horizon 1 --save-gt
```

`--horizon 0` shows the current frame; `1`, `2` and `3` show future predictions with ground truth at the corresponding horizon. The default output directory is `outputs/vis_eval/`. Use `--interactive` to open the first frame in Open3D.
