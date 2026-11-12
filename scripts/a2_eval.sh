#!/usr/bin/env bash

# Budget eval of an Approach 2 head checkpoint. HEAD_CKPT is required for the 'score' method;
# without it pass approach2.eval_methods=[random,opacity] (evaluate.py raises otherwise).
# Usage: scripts/a2_eval.sh [head_ckpt] [data_root] [index_json] [output_dir]
HEAD_CKPT=${1:-${HEAD_CKPT:-}}
ROOT=${2:-${ROOT:-/mnt/hdd/depthsplat/datasets/re10k_overfit5}}
INDEX=${3:-${INDEX:-$ROOT/evaluation_index.json}}
OUT=${4:-${OUT:-outputs/approach2/eval}}

CUDA_VISIBLE_DEVICES=${GPU:-0} poetry run python -m src.approach2.evaluate \
+experiment=re10k +approach2=re10k_small mode=test \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=$INDEX \
dataset.view_sampler.num_context_views=2 \
dataset.roots=[$ROOT] \
checkpointing.pretrained_model=${PRETRAINED:-/mnt/hdd/depthsplat/pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth} \
approach2.head_ckpt=${HEAD_CKPT:-null} \
output_dir=$OUT \
"${@:5}"
