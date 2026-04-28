#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <command...>" >&2
  echo "Example: $0 python script/test_render.py" >&2
  exit 1
fi

REMOTE_HOST="${REMOTE_HOST:-root@120.209.70.195}"
REMOTE_PORT="${REMOTE_PORT:-30331}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/gpufree-data}"
CONDA_INIT="${CONDA_INIT:-source /opt/conda/etc/profile.d/conda.sh}"
REMOTE_CMD="$*"

exec ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}" "\
${CONDA_INIT} && \
cd ${REMOTE_ROOT}/RoboTwin && \
conda run -n RoboTwin ${REMOTE_CMD}"
