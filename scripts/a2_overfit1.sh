#!/usr/bin/env bash

# Approach 2 overfit run on 1 scene with fixed views and varying budget k ~ U[0.01, 1].
# Usage: scripts/a2_overfit1.sh [output_dir] [extra Hydra overrides...]
# Same scene and views as scripts/a1_overfit1_k05.sh; validation runs on that scene too.
# train_times_per_scene=1000 keeps the persistent val iterator alive for the whole run.
OUT=${1:-${OUT:-outputs/approach2/overfit1}}
INDEX=assets/evaluation_index_a1_overfit1.json

CUDA_VISIBLE_DEVICES=${GPU:-0} poetry run python -m src.approach2.train \
+experiment=re10k +approach2=re10k_small \
dataset.roots=[/mnt/hdd/depthsplat/datasets/re10k_overfit5] \
dataset.train_times_per_scene=1000 \
dataset.augment=false \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=$INDEX \
dataset.view_sampler.num_context_views=2 \
approach2.val_root=/mnt/hdd/depthsplat/datasets/re10k_overfit5 \
approach2.val_index_path=$INDEX \
data_loader.train.batch_size=1 \
trainer.max_steps=3000 \
trainer.val_check_interval=100 \
checkpointing.every_n_train_steps=500 \
output_dir=$OUT \
"${@:2}"
