#!/usr/bin/env bash

# Run a GPU job in a detached tmux session, serialized by the GPU lock (queued jobs wait), logged to <log>.
# Usage: scripts/gpu_job.sh <session> <log> <cmd> [args...]
set -euo pipefail
NAME=$1
LOG=$2
shift 2
mkdir -p "$(dirname "$LOG")"
CMD=$(printf "%q " "$@")
LOGQ=$(printf "%q" "$LOG")
tmux new-session -d -s "$NAME" -c "$PWD" \
"PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True flock /tmp/depthsplat_gpu.lock $CMD > $LOGQ 2>&1; echo exit=\$? >> $LOGQ"
echo "started tmux session $NAME, log $LOG"
