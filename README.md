<div align="center">

<img src="assets/panobench-logo.svg" alt="PanoBench" width="360">

<h2>A Panoptic Occupancy Prediction Benchmark<br>for Autonomous Driving</h2>

<p>
  <a href="https://mias.group/PanoBench/"><img src="https://img.shields.io/badge/Project-Website-2878B5?style=for-the-badge" alt="Project website"></a>
  <a href="https://arxiv.org/abs/2610.23537"><img src="https://img.shields.io/badge/arXiv-2610.23537-B31B1B?style=for-the-badge&amp;logo=arxiv&amp;logoColor=white" alt="arXiv paper"></a>
  <a href="https://huggingface.co/datasets/fengyi233/PanoBench"><img src="https://img.shields.io/badge/🤗_Hugging_Face-Dataset-FFD21E?style=for-the-badge" alt="Dataset on Hugging Face"></a>
  <a href="https://www.bilibili.com/video/BV1aUJR6oEHT/"><img src="https://img.shields.io/badge/Bilibili-Demo-00A1D6?style=for-the-badge&amp;logo=bilibili&amp;logoColor=white" alt="Bilibili demo"></a>
</p>

<p><strong>200K+ frames &nbsp; · &nbsp; ~1K sequences &nbsp; · &nbsp; 3 subsets &nbsp; · &nbsp; 9 variants</strong><br>
Voxel resolutions: <strong>0.1 / 0.2 / 0.4 m</strong></p>


</div>

<p align="center">
  <img src="assets/panobench-intro.gif" alt="PanoBench in motion: three subsets and complete scene geometry" width="840">
</p>

## Overview

<p align="center">
  <img src="assets/teaser.webp" alt="PanoBench benchmark overview" width="840">
</p>

**PanoBench** is a camera-centric benchmark for panoptic occupancy prediction. It brings together synthetic and real-world driving scenes in a shared representation of 3D geometry, semantic categories, and object identities.

### Three complementary subsets

- **PanoBench-CarlaOcc** — synthetic driving scenes from CARLA, with complete mesh-derived geometry.
- **PanoBench-nuScenes** — real-world surround-view scenes with complete foreground annotations.
- **PanoBench-KITTI360** — real-world forward-view driving sequences with persistent instance identities.

### Three supported tasks

**Semantic occupancy prediction** · **Panoptic occupancy prediction** · **4D occupancy forecasting / Occupancy world modeling**

Complete foreground geometry, temporally consistent instance identities, and a shared valid-mask protocol support evaluation across the three subsets. See the [project page](https://mias.group/PanoBench/) for dataset videos, annotation comparisons, and benchmark results.

## Getting started

```bash
git clone https://github.com/fengyi233/PanoBench.git
cd PanoBench
```

1. **Install** the Python dependencies and compile the CUDA extensions using the [installation guide](docs/INSTALL.md).
2. **Prepare data** using the [dataset guide](docs/DATA.md).
3. **Prepare pretrained backbone weights** following the [training guide](docs/USAGE.md#backbone-initialization).


## Baselines

The repository currently includes **OPUS-V1**, **SparseOcc**, **FlashOCC**, **GaussianFormer-V1** and **SparseWorld-TC**, with configurations for all three subsets:

```text
configs/experiments/<model>/<dataset>.yaml

model:    opusv1 | sparseocc | flashocc | gaussianformerv1 | sparseworldtc
dataset:  panobench_carlaocc | panobench_nuscenes | panobench_kitti360
```

## Pretrained weights

Released checkpoints are available on [Hugging Face](https://huggingface.co/fengyi233/PanoBench). See the [model card](https://huggingface.co/fengyi233/PanoBench#measured-validation-results) for measured validation results.

| Model | Subset | Variant | Config | Weights |
| --- | --- | --- | --- | --- |
| OPUS-V1 | CarlaOcc | `vs_0.4_surround_view` | [Config](configs/experiments/opusv1/panobench_carlaocc.yaml) | [Weights](https://huggingface.co/fengyi233/PanoBench/resolve/main/opusv1_carlaocc_vs_0.4_surround_view.pth) |
| OPUS-V1 | nuScenes | `vs_0.4` | [Config](configs/experiments/opusv1/panobench_nuscenes.yaml) | [Weights](https://huggingface.co/fengyi233/PanoBench/resolve/main/opusv1_nuscenes_vs_0.4.pth) |
| SparseOcc | CarlaOcc | `vs_0.4_surround_view` | [Config](configs/experiments/sparseocc/panobench_carlaocc.yaml) | [Weights](https://huggingface.co/fengyi233/PanoBench/resolve/main/sparseocc_carlaocc_vs_0.4_surround_view.pth) |
| SparseOcc | KITTI360 | — | [Config](configs/experiments/sparseocc/panobench_kitti360.yaml) | [Weights](https://huggingface.co/fengyi233/PanoBench/resolve/main/sparseocc_kitti360.pth) |
| SparseOcc | nuScenes | `vs_0.4` | [Config](configs/experiments/sparseocc/panobench_nuscenes.yaml) | [Weights](https://huggingface.co/fengyi233/PanoBench/resolve/main/sparseocc_nuscenes_vs_0.4.pth) |
| FlashOCC | CarlaOcc | `vs_0.2_surround_view` | [Config](configs/experiments/flashocc/panobench_carlaocc.yaml) | [Weights](https://huggingface.co/fengyi233/PanoBench/resolve/main/flashocc_carlaocc_vs_0.2_surround_view.pth) |
| FlashOCC | KITTI360 | — | [Config](configs/experiments/flashocc/panobench_kitti360.yaml) | [Weights](https://huggingface.co/fengyi233/PanoBench/resolve/main/flashocc_kitti360.pth) |
| FlashOCC | nuScenes | `vs_0.4` | [Config](configs/experiments/flashocc/panobench_nuscenes.yaml) | [Weights](https://huggingface.co/fengyi233/PanoBench/resolve/main/flashocc_nuscenes_vs_0.4.pth) |
| GaussianFormer-V1 | CarlaOcc | `vs_0.4_surround_view` | [Config](configs/experiments/gaussianformerv1/panobench_carlaocc.yaml) | [Weights](https://huggingface.co/fengyi233/PanoBench/resolve/main/gaussianformerv1_carlaocc_vs_0.4_surround_view.pth) |
| GaussianFormer-V1 | KITTI360 | — | [Config](configs/experiments/gaussianformerv1/panobench_kitti360.yaml) | [Weights](https://huggingface.co/fengyi233/PanoBench/resolve/main/gaussianformerv1_kitti360.pth) |
| GaussianFormer-V1 | nuScenes | `vs_0.4` | [Config](configs/experiments/gaussianformerv1/panobench_nuscenes.yaml) | [Weights](https://huggingface.co/fengyi233/PanoBench/resolve/main/gaussianformerv1_nuscenes_vs_0.4.pth) |
| SparseWorld-TC | CarlaOcc | `vs_0.4_surround_view` | [Config](configs/experiments/sparseworldtc/panobench_carlaocc.yaml) | [Weights](https://huggingface.co/fengyi233/PanoBench/resolve/main/sparseworldtc_carlaocc_vs_0.4_surround_view.pth) |
| SparseWorld-TC | KITTI360 | — | [Config](configs/experiments/sparseworldtc/panobench_kitti360.yaml) | [Weights](https://huggingface.co/fengyi233/PanoBench/resolve/main/sparseworldtc_kitti360.pth) |
| SparseWorld-TC | nuScenes | `vs_0.4` | [Config](configs/experiments/sparseworldtc/panobench_nuscenes.yaml) | [Weights](https://huggingface.co/fengyi233/PanoBench/resolve/main/sparseworldtc_nuscenes_vs_0.4.pth) |

All listed checkpoints are from epoch 20. KITTI360 uses 0.2 m voxels and requires no variant argument.

Using the Hugging Face CLI (`pip install huggingface_hub`), run the following from the repository root to download all published checkpoints into `checkpoints/`:

```bash
hf download fengyi233/PanoBench --include "*.pth" --local-dir checkpoints
```

To download and evaluate a single checkpoint, for example OPUS-V1 on nuScenes:

```bash
hf download fengyi233/PanoBench opusv1_nuscenes_vs_0.4.pth --local-dir checkpoints
bash scripts/eval.sh experiments/opusv1/panobench_nuscenes checkpoints/opusv1_nuscenes_vs_0.4.pth vs_0.4
```

Checkpoint files are saved directly as `checkpoints/<model>_<subset>[_<variant>].pth`. Match the evaluation configuration and variant to the downloaded checkpoint.

## Training

For example, train OPUS-V1 on PanoBench-CarlaOcc at 0.4 m:

```bash
bash scripts/train.sh experiments/opusv1/panobench_carlaocc vs_0.4_surround_view
```

Select `vs_0.4_forward_view` to use only the front camera, `image_00`. Other CarlaOcc resolutions are 0.1 m and 0.2 m. nuScenes supports `vs_0.2` and `vs_0.4`; KITTI360 requires no variant argument.

```bash
# Multi-GPU training on nuScenes
GPUS=8 bash scripts/train.sh experiments/opusv1/panobench_nuscenes vs_0.4

# KITTI360
bash scripts/train.sh experiments/opusv1/panobench_kitti360
```

Logs and checkpoints are written to `outputs/<config>/<variant>/<timestamp>/`. See the [training and evaluation guide](docs/USAGE.md) for configuration overrides and forecasting.

## Evaluation

Replace `CHECKPOINT` with a checkpoint matching the model, subset and voxel variant:

```bash
bash scripts/eval.sh experiments/opusv1/panobench_carlaocc \
  CHECKPOINT vs_0.4_surround_view \
  'val_evaluator.metrics.0.metrics=[iou,miou,rayiou]'
```

The evaluation uses the released `valid_mask`. The [evaluation guide](docs/USAGE.md#evaluation) covers semantic, panoptic and forecasting metrics.

## Visualization

Render a prediction alongside its ground truth:

```bash
python vis_pred_occ.py \
  --config-name experiments/opusv1/panobench_carlaocc \
  --checkpoint CHECKPOINT --occ-variant vs_0.4_surround_view \
  --index 0 --save-gt
```

Images are saved under `outputs/vis_eval/`. Add `--interactive` for an Open3D viewer. For SparseWorld-TC, `--horizon 1`, `2` or `3` selects the 1 s, 2 s or 3 s forecast.


## Related Projects

Our work builds on [SparseOcc](https://github.com/MCG-NJU/SparseOcc), [GaussianFormer](https://github.com/huang-yh/GaussianFormer), [OPUS](https://github.com/jbwang1997/OPUS), [FlashOCC](https://github.com/Yzichen/FlashOCC), and [SparseWorld-TC](https://github.com/MrPicklesGG/SparseWorld-TC), etc. We thank the authors for their excellent work and open-source contributions!

PanoBench also builds on our previous work, [CarlaOcc](https://github.com/fengyi233/CarlaOcc). We welcome contributions and feedback. Feel free to open an issue or submit a pull request to help improve PanoBench.

## Citation

If you find this project helpful, please consider starring the project or citing the following papers:

```bibtex
@article{feng2026panobench,
  title={PanoBench: A Panoptic Occupancy Prediction Benchmark for Autonomous Driving},
  author={Feng, Yi and Guo, Zizhan and E, Junwu and Ma, Yu and Zhang, Mengtan and Wu, Jin and Ye, Wei and Wang, Hanli and Chen, Qijun and Wang, Danwei and Fan, Rui},
  journal={arXiv preprint arXiv:2610.23537},
  year={2026}
}

@inproceedings{feng2026carlaocc,
  title={An Instance-Centric Panoptic Occupancy Prediction Benchmark for Autonomous Driving},
  author={Feng, Yi and E, Junwu and Guo, Zizhan and Ma, Yu and Wang, Hanli and Fan, Rui},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```
