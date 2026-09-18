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


// SDDMM: C = alpha * op(A) * op(B) . spy(C) + beta * C, with C in CSR.
//
// The dense product is only evaluated where C already has a nonzero, so the
// work is nnz-parallel rather than (m x n)-parallel: one program takes BLOCK_P
// nonzeros and reduces over k.
//
// The kernel needs one row id per nonzero, which the CSR indptr does not store.
// That array is the caller's scratch (flagsparseSDDMM_bufferSize reports
// nnz int32) and _preprocess fills it with a binary-search kernel.

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#include "adaptor/adaptor.hpp"
#include "core/internal.hpp"
#include "core/jit.hpp"

using namespace flagsparse;

namespace {

constexpr int kRowIdsBlock = 1024;
constexpr int kRowIdsNumWarps = 4;
constexpr int kNumStages = 2;

struct DenseStrides { int64_t row = 0; int64_t col = 0; };

DenseStrides strides_of(const DnMatDescr* M) {
    return (M->order == FLAGSPARSE_ORDER_ROW) ? DenseStrides{M->ld, 1}
                                              : DenseStrides{1, M->ld};
}

bool transposes(flagsparseOperation_t op) {
    return op != FLAGSPARSE_OPERATION_NON_TRANSPOSE;
}

flagsparseStatus_t read_scalar(flagsparseHandle_t handle, const void* p,
                               flagsparseDataType_t ctype, double* out) {
    if (p == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    if (ctx(handle)->pointer_mode != FLAGSPARSE_POINTER_MODE_HOST) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    switch (ctype) {
        case FLAGSPARSE_R_32F: *out = *static_cast<const float*>(p);  return FLAGSPARSE_STATUS_SUCCESS;
        case FLAGSPARSE_R_64F: *out = *static_cast<const double*>(p); return FLAGSPARSE_STATUS_SUCCESS;
        default: return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
}

// Tuned on sm_120 over 14 matrices x k in {32, 64, 128, 256}, ported from
// _resolve_sddmm_launch_config: BLOCK_K=32 wins at every k because the kernel is
// bound by the gathered y traffic, not by the reduction. BLOCK_P depends on the
// pattern and on the dtype -- fp64 doubles the register footprint of the
// [BLOCK_P, BLOCK_K] tiles and a wide block then regresses, so it always takes
// the narrow config.
void resolve_launch(int64_t k, int64_t nnz, int64_t rows, bool is_fp64,
                    int* block_p, int* block_k, int* num_warps) {
    if (k >= 32) {
        *block_k = 32;
    } else {
        int v = 1;
        while (v < k) v <<= 1;
        *block_k = std::max(1, v);
    }
    const double mean_row_len =
        (rows > 0) ? static_cast<double>(nnz) / static_cast<double>(rows) : 0.0;
    if (!is_fp64 && mean_row_len >= 16.0) { *block_p = 512; *num_warps = 4; }
    else                                  { *block_p = 64;  *num_warps = 8; }
}

// row_ids[p] = the row owning nonzero p, by binary search over indptr.
flagsparseStatus_t build_row_ids(flagsparseHandle_t handle, const SpMatDescr* C,
                                 void* buffer) {
    if (C->nnz == 0) return FLAGSPARSE_STATUS_SUCCESS;
    if (buffer == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    const char* ot = triton_index_dtype(C->offsets_type);
    if (ot[0] == '\0') return FLAGSPARSE_STATUS_NOT_SUPPORTED;

    // The search must be able to reach any row, so it needs ceil(log2(n_rows))
    // halvings; surplus steps are idempotent once the interval is one wide.
    const int64_t n_rows = std::max<int64_t>(2, C->rows);
    int steps = 1;
    while ((static_cast<int64_t>(1) << steps) < n_rows) ++steps;

    std::string sig;
    sig += "*"; sig += ot; sig += ":16,";
    sig += "*i32:16,";
    sig += "i32,i32,";
    sig += std::to_string(kRowIdsBlock) + ",";
    sig += std::to_string(steps);

    std::vector<jit::Arg> args;
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(C->offsets)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(buffer)));
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(C->rows)));
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(C->nnz)));

    const int64_t grid = (C->nnz + kRowIdsBlock - 1) / kRowIdsBlock;
    std::string err;
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("sddmm_csr.py"), "_row_ids_kernel", sig,
        ctx(handle)->stream, grid, 1, 1, kRowIdsNumWarps, kNumStages, args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}

flagsparseStatus_t validate(flagsparseHandle_t handle, flagsparseOperation_t opA,
                            flagsparseOperation_t opB,
                            flagsparseConstDnMatDescr_t matA,
                            flagsparseConstDnMatDescr_t matB,
                            flagsparseConstSpMatDescr_t matC,
                            flagsparseDataType_t computeType,
                            flagsparseSDDMMAlg_t alg) {
    if (handle == nullptr) return FLAGSPARSE_STATUS_NOT_INITIALIZED;
    if (matA == nullptr || matB == nullptr || matC == nullptr) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    if (alg != FLAGSPARSE_SDDMM_ALG_DEFAULT) return FLAGSPARSE_STATUS_NOT_SUPPORTED;

    const DnMatDescr* A = dnmat(matA);
    const DnMatDescr* B = dnmat(matB);
    const SpMatDescr* C = spmat(matC);
    if (A->value_type != computeType || B->value_type != computeType ||
        C->value_type != computeType) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (C->format != FLAGSPARSE_FORMAT_CSR) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    if (C->idx_base != FLAGSPARSE_INDEX_BASE_ZERO) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    // The operator package has no complex SDDMM kernel; reporting that beats
    // inventing one here.
    if (computeType != FLAGSPARSE_R_32F && computeType != FLAGSPARSE_R_64F) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (triton_index_dtype(C->indices_type)[0] == '\0' ||
        triton_index_dtype(C->offsets_type)[0] == '\0') {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (C->nnz > static_cast<int64_t>(INT32_MAX)) return FLAGSPARSE_STATUS_NOT_SUPPORTED;

    // op(A) is m x k, op(B) is k x n, C is m x n.
    const int64_t m = transposes(opA) ? A->cols : A->rows;
    const int64_t k = transposes(opA) ? A->rows : A->cols;
    const int64_t kb = transposes(opB) ? B->cols : B->rows;
    const int64_t n = transposes(opB) ? B->rows : B->cols;
    if (k != kb || C->rows != m || C->cols != n) return FLAGSPARSE_STATUS_INVALID_VALUE;
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t run(flagsparseHandle_t handle, flagsparseOperation_t opA,
                       flagsparseOperation_t opB, const void* alpha,
                       flagsparseConstDnMatDescr_t matA,
                       flagsparseConstDnMatDescr_t matB, const void* beta,
                       flagsparseSpMatDescr_t matC, flagsparseDataType_t computeType,
                       void* externalBuffer) {
    const DnMatDescr* A = dnmat(matA);
    const DnMatDescr* B = dnmat(matB);
    auto* C = spmat(matC);

    double alpha_v = 1.0, beta_v = 0.0;
    if (flagsparseStatus_t s = read_scalar(handle, alpha, computeType, &alpha_v)) return s;
    if (flagsparseStatus_t s = read_scalar(handle, beta, computeType, &beta_v)) return s;
    if (C->nnz == 0) return FLAGSPARSE_STATUS_SUCCESS;

    if (externalBuffer == nullptr) {
        ctx(handle)->last_error =
            "SDDMM needs the externalBuffer reported by flagsparseSDDMM_bufferSize; "
            "it holds one row id per nonzero, which a CSR indptr does not store.";
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    if (C->sddmm_row_ids_buffer != externalBuffer) {
        if (flagsparseStatus_t s = build_row_ids(handle, C, externalBuffer)) return s;
        C->sddmm_row_ids_buffer = externalBuffer;
    }

    const int64_t k = transposes(opA) ? A->rows : A->cols;
    const bool is_fp64 = (computeType == FLAGSPARSE_R_64F);
    int block_p = 64, block_k = 32, num_warps = 8;
    resolve_launch(k, C->nnz, C->rows, is_fp64, &block_p, &block_k, &num_warps);

    // x is indexed [row][k] and y [column][k], so op(A) and op(B) are both just
    // a choice of which stride is which -- nothing is transposed in memory.
    const DenseStrides as = strides_of(A);
    const DenseStrides bs = strides_of(B);
    const int64_t stride_xm = transposes(opA) ? as.col : as.row;
    const int64_t stride_xk = transposes(opA) ? as.row : as.col;
    const int64_t stride_ym = transposes(opB) ? bs.row : bs.col;
    const int64_t stride_yk = transposes(opB) ? bs.col : bs.row;

    const char* vt = triton_dtype(computeType);
    const char* it = triton_index_dtype(C->indices_type);
    const bool has_in = (beta_v != 0.0);

    std::string sig;
    sig.reserve(192);
    sig += "*"; sig += it; sig += ":16,";   // indices
    sig += "*i32:16,";                      // row_ids
    sig += "*"; sig += vt; sig += ":16,";   // x
    sig += "*"; sig += vt; sig += ":16,";   // y
    sig += "*"; sig += vt; sig += ":16,";   // in  (C values)
    sig += "*"; sig += vt; sig += ":16,";   // out (C values, in place)
    sig += "i32,i32,";                      // nnz, k_dim
    sig += "i64,i64,i64,i64,";              // strides
    sig += vt; sig += ",";                  // alpha
    sig += vt; sig += ",";                  // beta
    sig += has_in ? "True," : "False,";     // HAS_IN
    sig += std::to_string(block_p) + ",";
    sig += std::to_string(block_k) + ",";
    sig += is_fp64 ? "True" : "False";      // ACC_IS_FP64

    std::vector<jit::Arg> args;
    args.reserve(14);
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(C->indices)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(externalBuffer)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(B->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(C->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(C->values)));
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(C->nnz)));
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(k)));
    args.push_back(jit::Arg::i64v(stride_xm));
    args.push_back(jit::Arg::i64v(stride_xk));
    args.push_back(jit::Arg::i64v(stride_ym));
    args.push_back(jit::Arg::i64v(stride_yk));
    if (is_fp64) { args.push_back(jit::Arg::d(alpha_v)); args.push_back(jit::Arg::d(beta_v)); }
    else { args.push_back(jit::Arg::f(static_cast<float>(alpha_v)));
           args.push_back(jit::Arg::f(static_cast<float>(beta_v))); }

    const int64_t grid = (C->nnz + block_p - 1) / block_p;
    std::string err;
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("sddmm_csr.py"), "sddmm_csr_real", sig,
        ctx(handle)->stream, grid, 1, 1, num_warps, kNumStages, args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}

}  // namespace

extern "C" {

flagsparseStatus_t flagsparseSDDMM_bufferSize(
    flagsparseHandle_t handle, flagsparseOperation_t opA, flagsparseOperation_t opB,
    const void* alpha, flagsparseConstDnMatDescr_t matA, flagsparseConstDnMatDescr_t matB,
    const void* beta, flagsparseSpMatDescr_t matC, flagsparseDataType_t computeType,
    flagsparseSDDMMAlg_t alg, size_t* bufferSize) {
    (void)alpha; (void)beta;
    if (bufferSize == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *bufferSize = 0;
    return guard(handle, [&]() -> flagsparseStatus_t {
        if (flagsparseStatus_t s = validate(handle, opA, opB, matA, matB, matC,
                                            computeType, alg)) {
            return s;
        }
        // One row id per nonzero. The CSR indptr does not store it and the
        // kernel is nnz-parallel, so it cannot be derived on the fly.
        *bufferSize = static_cast<size_t>(spmat(matC)->nnz) * sizeof(std::int32_t);
        return FLAGSPARSE_STATUS_SUCCESS;
    });
}

flagsparseStatus_t flagsparseSDDMM_preprocess(
    flagsparseHandle_t handle, flagsparseOperation_t opA, flagsparseOperation_t opB,
    const void* alpha, flagsparseConstDnMatDescr_t matA, flagsparseConstDnMatDescr_t matB,
    const void* beta, flagsparseSpMatDescr_t matC, flagsparseDataType_t computeType,
    flagsparseSDDMMAlg_t alg, void* externalBuffer) {
    (void)alpha; (void)beta;
    return guard(handle, [&]() -> flagsparseStatus_t {
        if (flagsparseStatus_t s = validate(handle, opA, opB, matA, matB, matC,
                                            computeType, alg)) {
            return s;
        }
        auto* C = spmat(matC);
        if (C->nnz == 0) return FLAGSPARSE_STATUS_SUCCESS;
        if (externalBuffer == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
        // Expanding the pattern belongs outside the timed solve: it depends on
        // the sparsity only, so it survives every change of A, B, alpha, beta.
        if (flagsparseStatus_t s = build_row_ids(handle, C, externalBuffer)) return s;
        C->sddmm_row_ids_buffer = externalBuffer;
        return FLAGSPARSE_STATUS_SUCCESS;
    });
}

flagsparseStatus_t flagsparseSDDMM(
    flagsparseHandle_t handle, flagsparseOperation_t opA, flagsparseOperation_t opB,
    const void* alpha, flagsparseConstDnMatDescr_t matA, flagsparseConstDnMatDescr_t matB,
    const void* beta, flagsparseSpMatDescr_t matC, flagsparseDataType_t computeType,
    flagsparseSDDMMAlg_t alg, void* externalBuffer) {
    return guard(handle, [&]() -> flagsparseStatus_t {
        if (flagsparseStatus_t s = validate(handle, opA, opB, matA, matB, matC,
                                            computeType, alg)) {
            return s;
        }
        return run(handle, opA, opB, alpha, matA, matB, beta, matC, computeType,
                   externalBuffer);
    });
}

}  // extern "C"
