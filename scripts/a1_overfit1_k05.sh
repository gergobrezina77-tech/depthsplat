#!/usr/bin/env bash

# Approach 1 overfit run on 1 scene with fixed views and fixed budget k=0.5.
# Usage: scripts/a1_overfit1_k05.sh [output_dir] [extra Hydra overrides...]
# The 1-entry index keeps only scene 41b80c5fb43019e5 (the sampler skips scenes not in it).
# Upstream overfit_to_scene is unusable here: it asserts disjoint train/test keys.
# train_times_per_scene=1000 keeps the persistent val iterator alive for the whole run.
OUT=${1:-${OUT:-outputs/approach1/overfit1_k05}}

CUDA_VISIBLE_DEVICES=${GPU:-0} poetry run python -m src.approach1.train \
+experiment=re10k +approach1=re10k_small \
dataset.roots=[/mnt/hdd/depthsplat/datasets/re10k_overfit5] \
dataset.train_times_per_scene=1000 \
dataset.augment=false \
dataset/view_sampler=evaluation \
dataset.view_sampler.index_path=assets/evaluation_index_a1_overfit1.json \
dataset.view_sampler.num_context_views=2 \
approach1.k_min=0.5 \
approach1.k_max=0.5 \
data_loader.train.batch_size=1 \
trainer.max_steps=3000 \
trainer.val_check_interval=100 \
checkpointing.every_n_train_steps=500 \
output_dir=$OUT \
"${@:2}"
