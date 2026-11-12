#!/usr/bin/env bash
# Builds mask-diff-gaussian-rasterization (MaskGaussian CUDA extension) into the Poetry venv.
# It needs torch at build time, so it can't be a regular Poetry dependency.
# The pinned upstream commit is patched at install time (patches/mask_rasterizer_last_mask.patch)
# to fix the backward pass (previous Gaussian's alpha was paired with the current mask).
# Installs side by side with diff_gaussian_rasterization (different package/module name).
set -euo pipefail
cd "$(dirname "$0")/.."

SHA=540dc11ef44c64f4c55fb894b3a4879de0cebfe7
if [[ "${FORCE:-0}" != 1 ]] && poetry run python -c \
  "import mask_diff_gaussian_rasterization as m; assert m.__a2_patched__" 2>/dev/null; then
  echo "patched mask_diff_gaussian_rasterization already installed (FORCE=1 to rebuild)"
  exit 0
fi
echo "Compiling CUDA mask rasterizer, this takes a few minutes with little output..."

# nvcc 12.4 supports host compilers up to gcc 13
export CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=g++-13
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"  # RTX 2080 Ti
# Upstream rasterizer_impl.h uses uint32_t without #include <cstdint>
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} -include cstdint"

PATCH="$PWD/patches/mask_rasterizer_last_mask.patch"
SRC="$(mktemp -d)"
trap 'rm -rf "$SRC"' EXIT
git clone --recursive https://github.com/kaikai23/mask-diff-gaussian-rasterization "$SRC/r"
git -C "$SRC/r" checkout "$SHA"
git -C "$SRC/r" submodule update --init --recursive
git -C "$SRC/r" apply "$PATCH"

poetry run pip install --no-build-isolation --force-reinstall --no-deps "$SRC/r"
