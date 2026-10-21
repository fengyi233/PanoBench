#!/usr/bin/env bash
# Usage: GPUS=8 bash scripts/train.sh CONFIG [OCC_VARIANT] [HYDRA_OVERRIDE ...]
set -euo pipefail

PANOBENCH_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PANOBENCH_REPO_ROOT"
if [[ $# -lt 1 ]]; then
  echo "Usage: GPUS=8 bash scripts/train.sh CONFIG [OCC_VARIANT] [HYDRA_OVERRIDE ...]" >&2
  exit 1
fi
ARGS=(--config-name "$1")
shift
if [[ $# -gt 0 && "$1" != *=* ]]; then
  ARGS+=("data.occ_variant=$1")
  shift
fi
GPUS="${GPUS:-1}"
if [[ ! "$GPUS" =~ ^[1-9][0-9]*$ ]]; then
  echo "GPUS must be a positive integer" >&2
  exit 1
fi
if [[ "$GPUS" == 1 ]]; then
  exec python train.py "${ARGS[@]}" "$@"
else
  exec torchrun --nproc_per_node="$GPUS" train.py "${ARGS[@]}" "$@"
fi
