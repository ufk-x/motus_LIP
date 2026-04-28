#!/usr/bin/env bash
set -euo pipefail

LOCAL_ROOT="${LOCAL_ROOT:-$HOME/gpufree}"
REMOTE_HOST="${REMOTE_HOST:-root@120.209.70.195}"
REMOTE_PORT="${REMOTE_PORT:-30331}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/gpufree-data}"

rsync -avz \
  -e "ssh -p ${REMOTE_PORT}" \
  --exclude "conda/" \
  --exclude ".Trash-0/" \
  --exclude "lost+found/" \
  --exclude "RoboTwin/assets/" \
  --exclude "RoboTwin/datasets/" \
  --exclude "RoboTwin/outputs/" \
  --exclude "RoboTwin/logs/" \
  --exclude "RoboTwin/wandb/" \
  --exclude "RoboTwin/**/__pycache__/" \
  --exclude "RoboTwin/**/*.pyc" \
  --exclude "Motus/pretrained_models/" \
  --exclude "Motus/*.whl" \
  --exclude "RoboTwin/policy/Motus/*.whl" \
  "${LOCAL_ROOT}/" \
  "${REMOTE_HOST}:${REMOTE_ROOT}/"
