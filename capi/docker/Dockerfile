# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
#
# Build environment for the FlagSparse C API. The image carries the toolchain
# only -- the source is mounted, so a rebuild does not mean a re-image.
#
#   docker build -t flagsparse-dev -f docker/Dockerfile .
#   docker run --gpus all -v $PWD:/work -w /work flagsparse-dev \
#       bash -lc 'cmake -S . -B build -G Ninja -DPython_ROOT=/usr -DBACKEND=CUDA && \
#                 cmake --build build -j && ctest --test-dir build'

ARG CUDA_IMAGE=nvidia/cuda:12.8.0-devel-ubuntu24.04
FROM ${CUDA_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git ninja-build cmake python3 python3-pip python3-dev \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# libtriton_jit needs libtorch's CMake package and pybind11; both come from the
# pip wheels, so no separate libtorch download is required.
# The triton pin is libtriton_jit's: >=3.1.0,<3.7.0.
RUN python3 -m pip install --no-cache-dir --break-system-packages \
        "torch>=2.5" "triton>=3.1.0,<3.7.0" "pybind11" "numpy" "pyyaml"

WORKDIR /work
