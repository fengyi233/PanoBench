#!/usr/bin/env bash
# Evaluate one checkpoint; paths are relative to the repository root.
# Activate the panobench environment before running this script.
#
# Usage:
#   bash scripts/eval.sh CONFIG CHECKPOINT [OCC_VARIANT] [HYDRA_OVERRIDE ...]
#
# CarlaOcc and nuScenes configs require an occupancy variant.
# KITTI-360 configs already set occ_variant to null, so omit it.
#
# Examples:
#   bash scripts/eval.sh experiments/opusv1/panobench_carlaocc \
#     outputs/experiments/opusv1/panobench_carlaocc/vs_0.2_surround_view/<run>/epoch_20.pth \
#     vs_0.2_surround_view
#
#   bash scripts/eval.sh experiments/opusv1/panobench_kitti360 \
#     outputs/experiments/opusv1/panobench_kitti360/default/<run>/epoch_20.pth
#
# Multi-GPU:
#   torchrun --nproc_per_node=8 eval.py \
#     --config-name experiments/opusv1/panobench_carlaocc \
#     data.occ_variant=vs_0.2_surround_view \
#     checkpoint=outputs/.../epoch_20.pth

set -euo pipefail

PANOBENCH_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PANOBENCH_REPO_ROOT"

if [[ $# -lt 2 ]]; then
  echo "Usage: bash scripts/eval.sh CONFIG CHECKPOINT [OCC_VARIANT] [HYDRA_OVERRIDE ...]" >&2
  exit 1
fi

CONFIG_NAME="$1"
CHECKPOINT="$2"
shift 2

ARGS=(
  --config-name "${CONFIG_NAME}"
  "checkpoint=${CHECKPOINT}"
)

if [[ $# -gt 0 && "$1" != *=* ]]; then
  ARGS+=("data.occ_variant=$1")
  shift
fi

exec python eval.py "${ARGS[@]}" "$@"
