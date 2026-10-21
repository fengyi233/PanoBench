#!/usr/bin/env bash
# Arguments are forwarded to the Python CLI; paths are relative to the repository root.
set -euo pipefail

PANOBENCH_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PANOBENCH_REPO_ROOT"
if [[ $# -eq 0 ]]; then
  python vis_pred_occ.py --help
  exit 0
fi
exec python vis_pred_occ.py "$@"
