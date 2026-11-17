#!/usr/bin/env bash

# Approach 1 overfit run on 3 scenes with fixed views and varying budget k ~ U[0.01, 1].
# Usage: scripts/a1_overfit3.sh [output_dir] [extra Hydra overrides...]
# One epoch = the 3 indexed scenes, so batch_size=3 gives a full-batch step every step.
# Non-persistent, unshuffled val restarts each check at the first indexed scene (41b80c5fb43019e5).
OUT=${1:-${OUT:-outputs/approach1/overfit3}}
# Precision: PRECISION=16-mixed (fp16 autocast) or 32-true (default).
PREC=${PRECISION:-32-true}

CUDA_VISIBLE_DEVICES=${GPU:-0} poetry run python -m src.approach1.train \
+experiment=re10k +approach1=re10k_small \
dataset.roots=[/mnt/hdd/depthsplat/datasets/re10k_overfit5] \
dataset.augment=false \
dataset.shuffle_val=false \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_a1_overfit3.json \
dataset.view_sampler.num_context_views=2 \
data_loader.train.batch_size=3 \
data_loader.val.persistent_workers=false \
trainer.max_steps=3000 \
trainer.val_check_interval=100 \
checkpointing.every_n_train_steps=500 \
approach1.precision=$PREC \
output_dir=$OUT \
"${@:2}"
