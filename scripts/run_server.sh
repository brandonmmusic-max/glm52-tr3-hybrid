#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-verdictai/glm52-tr3-hybrid:mtp3-dcp4-nvfp4-20260713}"
NAME="${NAME:-glm52-tr3-9300}"
MODELS_DIR="${MODELS_DIR:-/home/brandonmusic/models}"
CACHE_DIR="${CACHE_DIR:-/home/brandonmusic/.cache/glm52-tr3}"

docker rm -f "$NAME" >/dev/null 2>&1 || true
exec docker run -d \
  --name "$NAME" \
  --gpus all \
  --runtime nvidia \
  --ipc host \
  --shm-size 32g \
  --network host \
  --init \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$MODELS_DIR":/models-archive:ro \
  -v "$CACHE_DIR":/cache \
  "$IMAGE"
