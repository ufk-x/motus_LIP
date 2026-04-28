#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-root@120.209.70.195}"
REMOTE_PORT="${REMOTE_PORT:-30331}"

exec ssh -p "${REMOTE_PORT}" "${REMOTE_HOST}"
