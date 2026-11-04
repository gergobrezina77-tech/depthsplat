#!/usr/bin/env bash
# Builds diff-gaussian-rasterization (CUDA extension) into the Poetry venv.
# It needs torch at build time, so it can't be a regular Poetry dependency.
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "${FORCE:-0}" != 1 ]] && poetry run python -c "import diff_gaussian_rasterization" 2>/dev/null; then
  echo "diff_gaussian_rasterization already installed (FORCE=1 to rebuild)"
  exit 0
fi
echo "Compiling CUDA rasterizer, this takes a few minutes with little output..."

# nvcc 12.4 supports host compilers up to gcc 13
export CC=gcc-13 CXX=g++-13 CUDAHOSTCXX=g++-13
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-7.5}"  # RTX 2080 Ti
# Upstream rasterizer_impl.h uses uint32_t without #include <cstdint>
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-} -include cstdint"

poetry run pip install --no-build-isolation \
  "git+https://github.com/dcharatan/diff-gaussian-rasterization-modified@1250c420ebb945f0dce9945086e22faab9157c92"
