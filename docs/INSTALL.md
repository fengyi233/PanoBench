# Install

## 1. Conda + PyTorch

```bash
conda create -n panobench python=3.10 -y
conda activate panobench

pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
```

## 2. OpenMMLab

```bash
pip install -U openmim
mim install mmcv==2.0.1
mim install mmdet==3.0.0
mim install mmdet3d==1.1.1
```

## 3. Project dependencies

```bash
pip install hydra-core open3d pyyaml scipy pyquaternion
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install spconv-cu118
pip install opencv-python==4.9.0.80
pip install PyTurboJPEG==1.7.2
pip install numpy==1.26.4
```

We pin NumPy to 1.26.4 to prevent incompatibility. PyTurboJPEG is pinned to the 1.x line because 2.x changes the API used by the MMCV JPEG backend.

## 4. Build local CUDA extensions

All project CUDA ops live in `models/csrc` (one build covers SparseOcc / FlashOCC / GaussianFormer):

| Extension                      | Used by                               |
| ------------------------------ | ------------------------------------- |
| `_msmv_sampling_cuda`          | SparseOcc, OPUS                       |
| `_bev_pool_v2_cuda`            | FlashOCC                              |
| `_local_aggregate_cuda`        | GaussianFormer (local aggregate head) |
| `_deformable_aggregation_cuda` | GaussianFormer (deformable encoder)   |

Before building, install a CUDA **11.8** toolkit with `nvcc`:

```bash
# Option A: conda toolkit (recommended if no system CUDA)
conda install -y -c nvidia/label/cuda-11.8.0 cuda-toolkit
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"

# Option B: system toolkit instead
# export CUDA_HOME=/usr/local/cuda-11.8
# export PATH="$CUDA_HOME/bin:$PATH"
# export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
```

Then:

```bash
cd models/csrc
python setup.py build_ext --inplace
cd ../..
```
