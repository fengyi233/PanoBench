# Dataset preparation

PanoBench provides occupancy labels, preprocessed CarlaOcc and KITTI360 images, calibration, poses, frame splits and nuScenes sweep information in the [`panobench/` directory on Hugging Face](https://huggingface.co/datasets/fengyi233/PanoBench/tree/main/panobench). Its directory names match the local `data/` directory. KITTI360 benchmark frame lists, label definitions and mappings are included in the code repository. Obtain nuScenes images from the official dataset.

## Download and extract

Install the downloader with `pip install huggingface_hub`. Run all extraction commands from the PanoBench code repository root: every archive contains its complete `data/...` destination path. The downloader preserves remote paths under `--local-dir downloads`; downloaded files therefore appear in `downloads/panobench/`.

### Occupancy labels

Labels are stored in [`panobench/canonical_occ/`](https://huggingface.co/datasets/fengyi233/PanoBench/tree/main/panobench/canonical_occ). Each voxel/view variant is one complete `.tar` archive. KITTI360 has no variant directory and uses a single `panobench_kitti360.tar`.

For example, download CarlaOcc 0.4 m surround-view labels:

```bash
hf download fengyi233/PanoBench \
  panobench/canonical_occ/panobench_carlaocc/vs_0.4_surround_view.tar \
  --repo-type dataset --local-dir downloads

tar -xf downloads/panobench/canonical_occ/panobench_carlaocc/vs_0.4_surround_view.tar
```

This extracts labels directly into `data/canonical_occ/panobench_carlaocc/vs_0.4_surround_view/`. KITTI360 benchmark split files are already included in `datasets/splits/panobench_kitti360/` in the cloned code repository.

Use these archive paths for other subsets:

- CarlaOcc: `panobench/canonical_occ/panobench_carlaocc/vs_0.1_forward_view.tar`, or replace the resolution/view with another of the six supported combinations.
- nuScenes: `panobench/canonical_occ/panobench_nuscenes/vs_0.2.tar` or `panobench/canonical_occ/panobench_nuscenes/vs_0.4.tar`.
- KITTI360: `panobench/canonical_occ/panobench_kitti360.tar`.

Use the same archive filename in the download and extraction commands. See the [Hugging Face download guide](https://huggingface.co/docs/huggingface_hub/guides/download) for alternative download methods.

### CarlaOcc images, calibration, splits and poses

The [`panobench/CarlaOccV1Preprocessed/`](https://huggingface.co/datasets/fengyi233/PanoBench/tree/main/panobench/CarlaOccV1Preprocessed) directory contains **104 sequence archives**, named `<scene>.tar.gz`. Each contains six cameras of **640 × 192** RGB images and that sequence's `poses/cam_00.txt` and `poses/lidar.txt`. The separate `calib.tar` and `splits.tar` provide the official CarlaOcc calibration and train/validation/test frame lists. All 106 files together occupy approximately **122.84 GB**.

```bash
hf download fengyi233/PanoBench \
  --repo-type dataset --local-dir downloads \
  --include "panobench/CarlaOccV1Preprocessed/*.tar*"

for archive in downloads/panobench/CarlaOccV1Preprocessed/*.tar.gz; do
  tar -xzf "$archive"
done
tar -xf downloads/panobench/CarlaOccV1Preprocessed/calib.tar
tar -xf downloads/panobench/CarlaOccV1Preprocessed/splits.tar
```

To download an individual sequence, specify its path, for example `panobench/CarlaOccV1Preprocessed/Town01_Opt_Seq00.tar.gz`. The image archives are shared by all six label variants.

### nuScenes sweep information

Download the nuScenes sweep information files for [training](https://huggingface.co/datasets/fengyi233/PanoBench/resolve/main/panobench/nuscenes/pkl_files_panobench/nuscenes_infos_train_sweep.pkl) and [validation](https://huggingface.co/datasets/fengyi233/PanoBench/resolve/main/panobench/nuscenes/pkl_files_panobench/nuscenes_infos_val_sweep.pkl), and place them in `data/nuscenes/pkl_files_panobench/`:

```bash
hf download fengyi233/PanoBench \
  panobench/nuscenes/pkl_files_panobench/nuscenes_infos_train_sweep.pkl \
  panobench/nuscenes/pkl_files_panobench/nuscenes_infos_val_sweep.pkl \
  --repo-type dataset --local-dir downloads

mkdir -p data/nuscenes/pkl_files_panobench
cp downloads/panobench/nuscenes/pkl_files_panobench/nuscenes_infos_{train,val}_sweep.pkl data/nuscenes/pkl_files_panobench/
```

### KITTI360 images, calibration and poses

The [`panobench/KITTI-360/`](https://huggingface.co/datasets/fengyi233/PanoBench/tree/main/panobench/KITTI-360) directory contains **nine image archives** under `data_2d_raw/<scene>.tar.gz`, plus `data_poses.tar` and `calibration.tar`. The image archives contain only **640 × 192** `image_00` PNGs, including temporal context frames. `data_poses.tar` provides the nine `data/KITTI-360/data_poses/<scene>/cam0_to_world_complete.txt` files; `calibration.tar` provides the official calibration files. Images and supporting archives together occupy approximately **16.37 GB**.

```bash
hf download fengyi233/PanoBench \
  --repo-type dataset --local-dir downloads \
  --include "panobench/KITTI-360/data_2d_raw/*.tar.gz" \
  --include "panobench/KITTI-360/*.tar"

for archive in downloads/panobench/KITTI-360/data_2d_raw/*.tar.gz; do
  tar -xzf "$archive"
done
tar -xf downloads/panobench/KITTI-360/data_poses.tar
tar -xf downloads/panobench/KITTI-360/calibration.tar
```

To download an individual sequence, specify its path, for example `panobench/KITTI-360/data_2d_raw/2013_05_28_drive_0000_sync.tar.gz`. Use the benchmark frame lists, class definitions and label mappings included in this code repository.

## Directory layout

The default configurations expect the following paths under the repository root:

```text
data/
├── canonical_occ/
│   ├── panobench_carlaocc/<variant>/<scene>/<frame>.npz
│   ├── panobench_nuscenes/<variant>/<scene>/<frame>.npz
│   └── panobench_kitti360/<scene>/<frame>.npz
├── CarlaOccV1Preprocessed/
│   ├── calib/calib.yaml
│   ├── splits/{train,val,test}.txt
│   └── <scene>/
│       ├── rgb/image_00/<frame>.png
│       ├── ... image_05/
│       └── poses/lidar.txt
├── nuscenes/
│   ├── samples/
│   ├── sweeps/
│   ├── v1.0-trainval/
│   └── pkl_files_panobench/
│       ├── nuscenes_infos_train_sweep.pkl
│       └── nuscenes_infos_val_sweep.pkl
└── KITTI-360/
    ├── calibration/perspective.txt
    ├── data_poses/<scene>/cam0_to_world_complete.txt
    └── data_2d_raw/<scene>/image_00/data_192x640/<frame>.png
```

To store data elsewhere, override `data.data_root` and `data.occ_root`. The latter is the parent of the three label subset directories. Keep scene names and frame identifiers unchanged.

## PanoBench-CarlaOcc

The [sequence archives above](#carlaocc-images-calibration-splits-and-poses) are ready for training and evaluation. PanoBench expects RGB PNGs at **640 × 192** pixels; the loader scales the original camera intrinsics to this input resolution.

Alternatively, obtain original images and poses from [CarlaOcc](https://huggingface.co/datasets/fengyi233/CarlaOcc) and prepare the images yourself. Its [repository](https://github.com/fengyi233/CarlaOcc) describes the source layout:

```bash
python tools/prepare_images.py carlaocc \
  --source data/CarlaOccV1 \
  --destination data/CarlaOccV1Preprocessed
```

The image preparation command only resizes images. Also download and extract the calibration and split files, and copy each original sequence's poses into the preprocessed directory:

```bash
hf download fengyi233/PanoBench \
  panobench/CarlaOccV1Preprocessed/calib.tar \
  panobench/CarlaOccV1Preprocessed/splits.tar \
  --repo-type dataset --local-dir downloads

tar -xf downloads/panobench/CarlaOccV1Preprocessed/calib.tar
tar -xf downloads/panobench/CarlaOccV1Preprocessed/splits.tar

for scene in data/CarlaOccV1/Town*_Seq*; do
  scene_name=$(basename "$scene")
  mkdir -p "data/CarlaOccV1Preprocessed/$scene_name/poses"
  cp -a "$scene/poses/." "data/CarlaOccV1Preprocessed/$scene_name/poses/"
done
```

```text
vs_0.1_forward_view     vs_0.1_surround_view
vs_0.2_forward_view     vs_0.2_surround_view
vs_0.4_forward_view     vs_0.4_surround_view
```

Each variant contains 104,000 label frames. The official CarlaOcc splits contain 80,000 training, 12,000 validation and 12,000 test frames. Forward-view uses only `image_00` for images and visibility; surround-view uses `image_00` through `image_05`, including temporal observations.

## PanoBench-nuScenes

Obtain **v1.0-trainval** images, sweeps and metadata from [nuScenes](https://www.nuscenes.org/nuscenes) and keep the official directory layout. Both `vs_0.2` and `vs_0.4` contain 28,130 training and 6,019 validation keyframes. Place the supplied sweep information files in `data/nuscenes/pkl_files_panobench/`. Image resizing and augmentation are handled by the model's data pipeline.

## PanoBench-KITTI360

The [image archives above](#kitti360-images-calibration-and-poses) are ready for training and evaluation. Model inputs are **640 × 192** PNGs in `image_00/data_192x640/`, with dense `cam0_to_world_complete.txt` poses for temporal projection and forecasting.

Alternatively, obtain original rectified `image_00/data_rect/` images from [KITTI-360](https://www.cvlibs.net/datasets/kitti-360/) and resize them yourself:

```bash
python tools/prepare_images.py kitti360 \
  --source data/KITTI-360 --destination data/KITTI-360
```

This resizes `data_rect/` images into `data_192x640/` for the released frame list. Existing prepared images are preserved unless `--overwrite` is passed.

The release retains **64,418 label frames**, recorded in [`labels.txt`](../datasets/splits/panobench_kitti360/labels.txt). There is no variant directory. Validation uses drive `0009`; the other eight drives are training sequences. Forecasting requires labels and poses at every requested future frame ID.

The fixed benchmark input lists contain **51,254 training** and **13,164 validation** frames (64,418 total): [`train.txt`](../datasets/splits/panobench_kitti360/train.txt) and [`val.txt`](../datasets/splits/panobench_kitti360/val.txt). They preserve the input frame range used for the baseline experiments; the label archive contains exactly the union of these two lists. The configurations load these lists explicitly and report missing inputs instead of silently changing the evaluation set.

For 4D forecasting, every target at offsets `[0, 10, 20, 30]` must also belong to the released split. This yields **49,299 training** and **12,782 validation** samples with all four horizons. Additional labels outside the frame lists are ignored, even when present locally. Historical KITTI360 forecasting measurements used a larger set of future labels and are not measurements on this released evaluation subset.

## Label format

Each `.npz` contains:

- `occupancy`: an `N × 4` sparse array with integer grid indices `(x, y, z)` and the panoptic code `1000 * semantic_id + instance_id`.
- `valid_mask`: packed bits describing the evaluation domain, including occupied and free voxels.
- `volume_size`: the grid shape in `(x, y, z)` order.
- `voxel_size`: voxel edge length in meters.
- `voxel_origin`: grid origin in meters.

```python
import numpy as np

with np.load("FRAME.npz") as frame:
    occupancy = frame["occupancy"]
    shape = tuple(int(v) for v in frame["volume_size"])
    valid = np.unpackbits(frame["valid_mask"], count=int(np.prod(shape)))
    valid = valid.reshape(shape).astype(bool)
    semantic_ids = occupancy[:, 3] // 1000
    instance_ids = occupancy[:, 3] % 1000
```

Class names, thing/stuff definitions and label mappings are defined in [`utils/labels.py`](../utils/labels.py) and [`datasets/mappings/`](../datasets/mappings/). Semantic IDs are specific to each subset. The grid metadata in each file defines its voxel geometry. Use the released `valid_mask` for training and evaluation.

