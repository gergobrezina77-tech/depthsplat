#!/usr/bin/env bash

# Approach 2 full training on RE10K (score head only, DepthSplat frozen).
# 30000 steps at batch_size 4 (~1.06 it/s measured on overfit5) is ~8-9 h.
# Usage: scripts/a2_train.sh [output_dir] [extra Hydra overrides...]
OUT=${1:-${OUT:-outputs/approach2/re10k_full}}

CUDA_VISIBLE_DEVICES=${GPU:-0} poetry run python -m src.approach2.train \
+experiment=re10k +approach2=re10k_small \
dataset.roots=[/mnt/hdd/depthsplat/datasets/re10k] \
data_loader.train.batch_size=4 \
trainer.max_steps=30000 \
trainer.val_check_interval=2000 \
checkpointing.every_n_train_steps=5000 \
output_dir=$OUT \
"${@:2}"
