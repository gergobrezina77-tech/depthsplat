#!/usr/bin/env bash

# Approach 2 overfit run on the 5-scene RE10K subset (score head only, DepthSplat frozen).
# Usage: scripts/a2_overfit.sh [output_dir] [extra Hydra overrides...]
# batch_size 4 is a placeholder until the memory probe (Task 13).
OUT=${1:-${OUT:-outputs/approach2/overfit5}}

CUDA_VISIBLE_DEVICES=${GPU:-0} poetry run python -m src.approach2.train \
+experiment=re10k +approach2=re10k_small \
dataset.roots=[/mnt/hdd/depthsplat/datasets/re10k_overfit5] \
data_loader.train.batch_size=4 \
trainer.max_steps=3000 \
trainer.val_check_interval=250 \
checkpointing.every_n_train_steps=500 \
output_dir=$OUT \
"${@:2}"
