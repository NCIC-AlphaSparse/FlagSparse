// Copyright 2026 FlagOS Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


// Gather / scatter between a sparse vector and a dense one.
//
//   flagsparseGather :  X_values[i] = Y[X_indices[i]]
//   flagsparseScatter:  Y[X_indices[i]] = X_values[i]
//
// Both dtypes reach the kernels as the COMPONENT dtype: a complex buffer is
// handed over as interleaved real/imag pairs, which is the only representation
// Triton has and the one the Python package already uses.

#include <algorithm>
#include <string>
#include <vector>

#include "adaptor/adaptor.hpp"
#include "core/internal.hpp"
#include "core/jit.hpp"

using namespace flagsparse;

namespace {

constexpr int kBlockSize = 256;
constexpr int kNumWarps = 8;
constexpr int kNumStages = 2;

// Cap for the gather kernel's grid-stride loop: enough programs to fill the
// device once. This number is NOT cosmetic -- the Python implementation used a
// literal 2 here, which pinned every gather to two thread blocks (1.2% of an
// RTX 5090's SMs) and made it 28x slower than it needed to be at nnz=65536.
// Deriving it from the device keeps it right across backends, whose SM counts
// and warp sizes differ.
int64_t gather_max_programs(const Context* c) {
    const int sm = adaptor::multiprocessor_count(c->device_index);
    const int threads_per_sm = adaptor::max_threads_per_multiprocessor(c->device_index);
    const int warp = adaptor::warp_size(c->device_index);
    if (sm <= 0) return 1024;  // no usable properties: big enough not to serialise
    const int block_threads = std::max(1, kNumWarps * (warp > 0 ? warp : 32));
    const int blocks_per_sm =
        threads_per_sm > 0 ? std::max(1, threads_per_sm / block_threads) : 1;
    return std::max<int64_t>(1, static_cast<int64_t>(sm) * blocks_per_sm);
}

struct Plan {
    const char* kernel_name;
    std::string signature;
    std::vector<jit::Arg> args;
    int64_t grid;
};

// Shared validation. `sp` is the sparse vector in both directions; `dn` the dense.
flagsparseStatus_t validate(flagsparseHandle_t handle, const SpVecDescr* sp,
                            const DnVecDescr* dn) {
    if (handle == nullptr) return FLAGSPARSE_STATUS_NOT_INITIALIZED;
    if (sp == nullptr || dn == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    if (sp->value_type != dn->value_type) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    if (sp->idx_base != FLAGSPARSE_INDEX_BASE_ZERO) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    // cuSPARSE requires the sparse vector's logical size to match the dense one;
    // without this an index could legitimately address past the dense buffer.
    if (sp->size != dn->size) return FLAGSPARSE_STATUS_INVALID_VALUE;
    if (sp->nnz < 0 || sp->nnz > sp->size) return FLAGSPARSE_STATUS_INVALID_VALUE;
    const flagsparseDataType_t comp = component_dtype(sp->value_type);
    if (triton_dtype(comp)[0] == '\0') return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    if (triton_index_dtype(sp->idx_type)[0] == '\0') return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    return FLAGSPARSE_STATUS_SUCCESS;
}

// Both kernels take (dst, src, indices, nnz, BLOCK_SIZE) in that shape, so the
// signature differs only in which buffer is which.
std::string build_signature(flagsparseDataType_t value_type,
                            flagsparseIndexType_t idx_type) {
    const char* vt = triton_dtype(component_dtype(value_type));
    const char* it = triton_index_dtype(idx_type);
    std::string sig;
    sig.reserve(64);
    sig += "*"; sig += vt; sig += ":16,";
    sig += "*"; sig += vt; sig += ":16,";
    sig += "*"; sig += it; sig += ":16,";
    sig += "i32,";
    sig += std::to_string(kBlockSize);
    return sig;
}

flagsparseStatus_t run(flagsparseHandle_t handle, const char* kernel_name,
                       void* dst, const void* src, const void* indices,
                       int64_t nnz, flagsparseDataType_t value_type,
                       flagsparseIndexType_t idx_type, int64_t grid) {
    std::vector<jit::Arg> args;
    args.reserve(4);
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(dst)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(const_cast<void*>(src))));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(const_cast<void*>(indices))));
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(nnz)));

    std::string err;
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("gather_scatter.py"), kernel_name,
        build_signature(value_type, idx_type), ctx(handle)->stream,
        grid, 1, 1, kNumWarps, kNumStages, args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}

}  // namespace

extern "C" {

flagsparseStatus_t flagsparseGather(flagsparseHandle_t handle,
                                    flagsparseConstDnVecDescr_t vecY,
                                    flagsparseSpVecDescr_t vecX) {
    return guard(handle, [&]() -> flagsparseStatus_t {
        const DnVecDescr* Y = dnvec(vecY);
        SpVecDescr* X = spvec(vecX);
        if (flagsparseStatus_t s = validate(handle, X, Y)) return s;
        if (X->nnz == 0) return FLAGSPARSE_STATUS_SUCCESS;

        // Grid-stride loop: cap at what fills the device, never more tiles than
        // there are.
        const int64_t tiles = (X->nnz + kBlockSize - 1) / kBlockSize;
        const int64_t grid = std::min(tiles, gather_max_programs(ctx(handle)));

        const char* kernel = is_complex(X->value_type) ? "_gather_complex_kernel"
                                                       : "_gather_real_kernel";
        return run(handle, kernel, X->values, Y->values, X->indices, X->nnz,
                   X->value_type, X->idx_type, grid);
    });
}

flagsparseStatus_t flagsparseScatter(flagsparseHandle_t handle,
                                     flagsparseConstSpVecDescr_t vecX,
                                     flagsparseDnVecDescr_t vecY) {
    return guard(handle, [&]() -> flagsparseStatus_t {
        const SpVecDescr* X = spvec(vecX);
        DnVecDescr* Y = dnvec(vecY);
        if (flagsparseStatus_t s = validate(handle, X, Y)) return s;
        if (X->nnz == 0) return FLAGSPARSE_STATUS_SUCCESS;

        // No grid-stride loop here, so the grid must cover every element.
        const int64_t grid = (X->nnz + kBlockSize - 1) / kBlockSize;

        const char* kernel = is_complex(X->value_type) ? "_scatter_complex_kernel"
                                                       : "_scatter_real_kernel";
        return run(handle, kernel, Y->values, X->values, X->indices, X->nnz,
                   X->value_type, X->idx_type, grid);
    });
}

}  // extern "C"
