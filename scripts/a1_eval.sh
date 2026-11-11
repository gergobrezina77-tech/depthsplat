#!/usr/bin/env bash

# Budget eval of an Approach 1 checkpoint (frozen model by default).
# Usage: scripts/a1_eval.sh [ckpt] [data_root] [index_json] [output_dir]
CKPT=${1:-${CKPT:-/mnt/hdd/depthsplat/pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth}}
ROOT=${2:-${ROOT:-/mnt/hdd/depthsplat/datasets/re10k_overfit5}}
INDEX=${3:-${INDEX:-$ROOT/evaluation_index.json}}
OUT=${4:-${OUT:-outputs/approach1/eval}}

CUDA_VISIBLE_DEVICES=${GPU:-0} poetry run python -m src.approach1.evaluate \
+experiment=re10k +approach1=re10k_small mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=$INDEX \
dataset.view_sampler.num_context_views=2 \
dataset.roots=[$ROOT] \
checkpointing.pretrained_model=$CKPT \
output_dir=$OUT \
"${@:5}"
