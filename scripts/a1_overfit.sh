#!/usr/bin/env bash

# Approach 1 overfit feasibility run (Gate G1) on the 5-scene RE10K subset.
# Usage: scripts/a1_overfit.sh [output_dir] [extra Hydra overrides...]
# batch_size 4 peaks at ~9.4 GB on the 11 GB RTX 2080 Ti (probe: bs 1/2/4 = 2.7/6.0/9.4 GB).
OUT=${1:-${OUT:-outputs/approach1/overfit5}}

CUDA_VISIBLE_DEVICES=${GPU:-0} poetry run python -m src.approach1.train \
+experiment=re10k +approach1=re10k_small \
dataset.roots=[/mnt/hdd/depthsplat/datasets/re10k_overfit5] \
data_loader.train.batch_size=4 \
trainer.max_steps=3000 \
trainer.val_check_interval=250 \
checkpointing.every_n_train_steps=500 \
output_dir=$OUT \
"${@:2}"
