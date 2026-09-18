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


// SpMM: C = alpha * op(A) * op(B) + beta * C, with A in CSR or COO.
//
// Shaped like src/ops/spmv.cpp -- validate, route, size the launch, hand raw
// pointers to libtriton_jit. Two things are specific to this operator:
//
//  * Both dense operands are addressed through explicit strides, so
//    FLAGSPARSE_ORDER_ROW and _COL are the same kernel with different strides,
//    and opB = TRANSPOSE is the two B strides swapped. Nothing is materialised.
//  * The kernel's accumulator dtype is a Triton dtype OBJECT, which the
//    raw-args signature parser cannot spell. The call therefore goes through
//    the thin jit wrapper in flagsparse_codegen/spmm_csr.py, which takes a bool
//    and picks the dtype; Triton inlines it away. See that file for the why.

#include <algorithm>
#include <climits>
#include <cstdint>
#include <string>
#include <vector>

#include "adaptor/adaptor.hpp"
#include "core/internal.hpp"
#include "core/jit.hpp"

using namespace flagsparse;

namespace {

// ------------------------------------------------------------- launch ---

struct LaunchConfig {
    int block_n = 32;
    int block_nnz = 32;
    int num_warps = 4;
    int num_stages = 2;
};

// AlphaSparse's csrspmm_rb_sr warp/factor choice, as the Python package uses it
// for row-major dense operands (_select_alpha_spmm_alg1_warp_and_factor).
// BLOCK_N = warp_size * factor and BLOCK_NNZ = warp_size, so both stay powers
// of two, which tl.arange requires.
void select_warp_and_factor(int64_t n_dense_cols, int* warp_size, int* factor) {
    if (n_dense_cols > 64)      { *warp_size = 32; *factor = 4; }
    else if (n_dense_cols > 32) { *warp_size = 32; *factor = 2; }
    else if (n_dense_cols > 16) { *warp_size = 32; *factor = 1; }
    else if (n_dense_cols > 8)  { *warp_size = 16; *factor = 1; }
    else if (n_dense_cols > 4)  { *warp_size = 8;  *factor = 1; }
    else                        { *warp_size = 4;  *factor = 1; }
}

// The device caps how many warps a block may have. max_threads_per_block is not
// in the adaptor (it is not needed anywhere else), so the portable 1024 is used
// as the block cap and the real per-multiprocessor figure does the rest -- the
// same two bounds _clip_spmm_base_num_warps applies.
int clip_num_warps(int desired, int device_index) {
    const int warp = std::max(1, adaptor::warp_size(device_index));
    const int by_block = std::max(1, 1024 / warp);
    const int by_mp = std::max(1, adaptor::max_threads_per_multiprocessor(device_index) / warp);
    const int cap = std::min({16, by_block, by_mp});
    int chosen = 1;
    for (const int candidate : {1, 2, 4, 8, 16}) {
        if (candidate <= cap && candidate <= std::max(1, desired)) chosen = candidate;
    }
    return chosen;
}

// Mirrors _resolve_spmm_base_triton_launch's CUDA branch. Kept here rather than
// read out of the Python package because the C API must not import and run the
// operator library's orchestration -- replacing that orchestration is the whole
// point of this layer.
LaunchConfig resolve_launch(flagsparseDataType_t compute_type, int64_t n_dense_cols,
                            int64_t max_row_nnz, int device_index) {
    int warp_size = 32, factor = 1;
    select_warp_and_factor(n_dense_cols, &warp_size, &factor);

    LaunchConfig cfg;
    cfg.block_n = warp_size * factor;
    cfg.block_nnz = warp_size;

    int desired_warps;
    if (n_dense_cols <= 16)      desired_warps = 1;
    else if (n_dense_cols <= 32) desired_warps = 2;
    else if (n_dense_cols <= 64) desired_warps = 4;
    else                         desired_warps = (max_row_nnz >= 512) ? 8 : 4;
    if (cfg.block_n >= 128) desired_warps = std::max(desired_warps, 4);
    if (max_row_nnz >= 1024 && n_dense_cols > 32) desired_warps = std::max(desired_warps, 8);

    const bool wide = (compute_type == FLAGSPARSE_R_64F || compute_type == FLAGSPARSE_C_64F);
    desired_warps = std::min(desired_warps, wide ? 8 : 16);
    cfg.num_warps = clip_num_warps(desired_warps, device_index);

    if (n_dense_cols > 64 || cfg.block_n >= 128)        cfg.num_stages = 1;
    else if (wide && max_row_nnz >= 512)                cfg.num_stages = 1;
    else                                                cfg.num_stages = 2;
    return cfg;
}

// COO's rowrun route is tuned differently, and one constant matters more than
// the rest: BLOCK_NNZ. The kernel unrolls `tl.static_range(0, BLOCK_NNZ)`, so the
// body is emitted BLOCK_NNZ times regardless of how long the row actually is. The
// operator package's 30-matrix sweep landed on a flat 4; its old default of 256
// ran ~253 dead loads per useful one on a short-rowed matrix and cost roughly 7x.
// num_warps comes from the device's own warp size because the kernel vectorises
// over BLOCK_N dense columns and nothing else -- leaving Triton's default of 4
// warps idles three quarters of the threads on a 32-wide warp.
LaunchConfig resolve_coo_launch(int64_t n_dense_cols, int device_index) {
    int warp_size = 32, factor = 1;
    select_warp_and_factor(n_dense_cols, &warp_size, &factor);

    LaunchConfig cfg;
    cfg.block_n = warp_size * factor;
    cfg.block_nnz = 4;
    const int device_warp = std::max(1, adaptor::warp_size(device_index));
    cfg.num_warps = std::max(1, cfg.block_n / device_warp);
    // The Python path passes no num_stages, so Triton's own default applies;
    // spelling it out keeps the two front ends compiling the same kernel.
    cfg.num_stages = 3;
    return cfg;
}

// ------------------------------------------------------------ operands ---

// Row/column strides of a dense matrix, in elements of its value type.
struct DenseStrides { int64_t row = 0; int64_t col = 0; };

DenseStrides strides_of(const DnMatDescr* M) {
    return (M->order == FLAGSPARSE_ORDER_ROW) ? DenseStrides{M->ld, 1}
                                              : DenseStrides{1, M->ld};
}

bool transposes(flagsparseOperation_t op) {
    return op != FLAGSPARSE_OPERATION_NON_TRANSPOSE;
}

// alpha/beta arrive as void* under the handle's pointer mode. DEVICE mode would
// need a copy back or a kernel argument fetch; it is refused rather than
// silently dereferencing a device address on the host. Complex scalars are read
// as an interleaved pair, the way cuComplex is laid out.
flagsparseStatus_t read_scalar(flagsparseHandle_t handle, const void* p,
                               flagsparseDataType_t ctype, double* re, double* im) {
    if (p == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    if (ctx(handle)->pointer_mode != FLAGSPARSE_POINTER_MODE_HOST) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    *im = 0.0;
    switch (ctype) {
        case FLAGSPARSE_R_32F: *re = static_cast<const float*>(p)[0];  return FLAGSPARSE_STATUS_SUCCESS;
        case FLAGSPARSE_R_64F: *re = static_cast<const double*>(p)[0]; return FLAGSPARSE_STATUS_SUCCESS;
        case FLAGSPARSE_C_32F:
            *re = static_cast<const float*>(p)[0];
            *im = static_cast<const float*>(p)[1];
            return FLAGSPARSE_STATUS_SUCCESS;
        case FLAGSPARSE_C_64F:
            *re = static_cast<const double*>(p)[0];
            *im = static_cast<const double*>(p)[1];
            return FLAGSPARSE_STATUS_SUCCESS;
        default: return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
}

// One route per format, and an algorithm id is a performance hint over the same
// result, so every id belonging to the matrix's own format is accepted. An id
// naming a DIFFERENT format is refused rather than quietly ignored -- that is a
// caller mistake worth surfacing. Both routes are deterministic, which is never
// less than what a caller asking for a non-deterministic variant expects.
bool alg_supported(flagsparseFormat_t format, flagsparseSpMMAlg_t alg) {
    if (alg == FLAGSPARSE_SPMM_ALG_DEFAULT) return true;
    if (format == FLAGSPARSE_FORMAT_CSR) {
        return alg == FLAGSPARSE_SPMM_CSR_ALG1 || alg == FLAGSPARSE_SPMM_CSR_ALG2 ||
               alg == FLAGSPARSE_SPMM_CSR_ALG3;
    }
    if (format == FLAGSPARSE_FORMAT_COO) {
        return alg == FLAGSPARSE_SPMM_COO_ALG1 || alg == FLAGSPARSE_SPMM_COO_ALG2;
    }
    return false;
}

flagsparseStatus_t validate(flagsparseHandle_t handle, flagsparseOperation_t opA,
                            flagsparseOperation_t opB,
                            flagsparseConstSpMatDescr_t matA,
                            flagsparseConstDnMatDescr_t matB,
                            flagsparseConstDnMatDescr_t matC,
                            flagsparseDataType_t computeType,
                            flagsparseSpMMAlg_t alg) {
    if (handle == nullptr) return FLAGSPARSE_STATUS_NOT_INITIALIZED;
    if (matA == nullptr || matB == nullptr || matC == nullptr) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    const SpMatDescr* A = spmat(matA);
    const DnMatDescr* B = dnmat(matB);
    const DnMatDescr* C = dnmat(matC);

    if (A->value_type != computeType || B->value_type != computeType ||
        C->value_type != computeType) {
        // Mixed precision is a cuSPARSE feature this build does not have;
        // report it rather than computing in the wrong type.
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (A->format != FLAGSPARSE_FORMAT_CSR && A->format != FLAGSPARSE_FORMAT_COO) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (A->idx_base != FLAGSPARSE_INDEX_BASE_ZERO) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    if (!alg_supported(A->format, alg)) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    // COO segment bounds and the value index are int32 inside the kernel.
    if (A->format == FLAGSPARSE_FORMAT_COO && A->nnz > static_cast<int64_t>(INT32_MAX)) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }

    // op(A) = A^T would need the transposed CSR built first; that is a prepare
    // step this operator does not have yet, so it is refused, not approximated.
    if (transposes(opA)) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    // op(B) = B^T is free (strides swap), but conj(B) would have to negate the
    // imaginary part inside the kernel.
    if (opB == FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (opB != FLAGSPARSE_OPERATION_NON_TRANSPOSE &&
        opB != FLAGSPARSE_OPERATION_TRANSPOSE) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }

    const flagsparseDataType_t component = component_dtype(computeType);
    if (triton_dtype(component)[0] == '\0') return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    // fp16/bf16 would need a compute-precision copy of both dense operands, the
    // way the Python path casts to fp32 first. Not wired up.
    if (component != FLAGSPARSE_R_32F && component != FLAGSPARSE_R_64F) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (triton_index_dtype(A->indices_type)[0] == '\0' ||
        triton_index_dtype(A->offsets_type)[0] == '\0') {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }

    const int64_t m = A->rows;
    const int64_t k = A->cols;
    const int64_t n = C->cols;
    const int64_t b_rows = transposes(opB) ? B->cols : B->rows;
    const int64_t b_cols = transposes(opB) ? B->rows : B->cols;
    if (C->rows != m || b_rows != k || b_cols != n) return FLAGSPARSE_STATUS_INVALID_VALUE;
    return FLAGSPARSE_STATUS_SUCCESS;
}

// --------------------------------------------------------------- launch ---

// Everything the two formats agree on, resolved once.
struct Operands {
    bool complex_op = false;
    bool acc_is_fp64 = false;
    bool has_beta = false;
    double alpha_re = 1.0, alpha_im = 0.0, beta_re = 0.0, beta_im = 0.0;
    int64_t m = 0, n = 0;
    int64_t stride_bk = 0, stride_bn = 0, stride_cm = 0, stride_cn = 0;
    const char* vt = "";
};

flagsparseStatus_t resolve_operands(flagsparseHandle_t handle, flagsparseOperation_t opB,
                                    const void* alpha, const SpMatDescr* A,
                                    const DnMatDescr* B, const void* beta,
                                    const DnMatDescr* C,
                                    flagsparseDataType_t computeType, Operands* out) {
    if (flagsparseStatus_t s =
            read_scalar(handle, alpha, computeType, &out->alpha_re, &out->alpha_im)) {
        return s;
    }
    if (flagsparseStatus_t s =
            read_scalar(handle, beta, computeType, &out->beta_re, &out->beta_im)) {
        return s;
    }
    out->m = A->rows;
    out->n = C->cols;
    out->complex_op = is_complex(computeType);
    const flagsparseDataType_t component = component_dtype(computeType);
    out->acc_is_fp64 = (component == FLAGSPARSE_R_64F);
    out->has_beta = (out->beta_re != 0.0 || out->beta_im != 0.0);
    out->vt = triton_dtype(component);

    // Complex buffers are interleaved real/imag pairs of the component dtype, so
    // every stride doubles when counted in components and the step from a real
    // part to its imaginary part is 1.
    const int64_t unit = out->complex_op ? 2 : 1;
    const DenseStrides bs = strides_of(B);
    const DenseStrides cs = strides_of(C);
    out->stride_bk = (transposes(opB) ? bs.col : bs.row) * unit;
    out->stride_bn = (transposes(opB) ? bs.row : bs.col) * unit;
    out->stride_cm = cs.row * unit;
    out->stride_cn = cs.col * unit;
    return FLAGSPARSE_STATUS_SUCCESS;
}

// Pointer alignment ":16" is Triton's promise that the buffer is 16-byte aligned;
// every allocation we accept comes from a device allocator that guarantees at
// least that. Strides are i64 so a large leading dimension cannot overflow the
// index arithmetic inside the kernel.
void append_dense_signature(const Operands& ops, std::string* sig) {
    *sig += ops.vt; *sig += ",";                          // alpha (re)
    if (ops.complex_op) { *sig += ops.vt; *sig += ","; }  // alpha_im
    *sig += ops.vt; *sig += ",";                          // beta (re)
    if (ops.complex_op) { *sig += ops.vt; *sig += ","; }  // beta_im
    *sig += "i32,";                                       // n_rows / n_segs
    *sig += "i32,";                                       // n_dense_cols
    *sig += "i64,i64,";                                   // stride_bk, stride_bn
    if (ops.complex_op) *sig += "i64,";                    // stride_br
    *sig += "i64,i64,";                                   // stride_cm, stride_cn
    if (ops.complex_op) *sig += "i64,";                    // stride_cr
}

void append_dense_args(const Operands& ops, int64_t leading_extent,
                       std::vector<jit::Arg>* args) {
    const auto push_scalar = [&](double v) {
        if (ops.acc_is_fp64) args->push_back(jit::Arg::d(v));
        else                 args->push_back(jit::Arg::f(static_cast<float>(v)));
    };
    push_scalar(ops.alpha_re);
    if (ops.complex_op) push_scalar(ops.alpha_im);
    push_scalar(ops.beta_re);
    if (ops.complex_op) push_scalar(ops.beta_im);

    args->push_back(jit::Arg::i(static_cast<std::int32_t>(leading_extent)));
    args->push_back(jit::Arg::i(static_cast<std::int32_t>(ops.n)));
    args->push_back(jit::Arg::i64v(ops.stride_bk));
    args->push_back(jit::Arg::i64v(ops.stride_bn));
    if (ops.complex_op) args->push_back(jit::Arg::i64v(1));   // stride_br
    args->push_back(jit::Arg::i64v(ops.stride_cm));
    args->push_back(jit::Arg::i64v(ops.stride_cn));
    if (ops.complex_op) args->push_back(jit::Arg::i64v(1));   // stride_cr
}

flagsparseStatus_t run_csr(flagsparseHandle_t handle, SpMatDescr* A, const DnMatDescr* B,
                           DnMatDescr* C, flagsparseDataType_t computeType,
                           const Operands& ops) {
    (void)C;
    int64_t max_row_nnz = 0;
    if (flagsparseStatus_t s = ensure_max_row_nnz(A, &max_row_nnz)) return s;
    const LaunchConfig cfg =
        resolve_launch(computeType, ops.n, max_row_nnz, ctx(handle)->device_index);

    const char* it = triton_index_dtype(A->indices_type);
    const char* ot = triton_index_dtype(A->offsets_type);

    std::string sig;
    sig.reserve(192);
    sig += "*"; sig += ops.vt; sig += ":16,";   // data
    sig += "*"; sig += it; sig += ":16,";       // indices
    sig += "*"; sig += ot; sig += ":16,";       // indptr
    sig += "*"; sig += ops.vt; sig += ":16,";   // B
    sig += "*"; sig += ops.vt; sig += ":16,";   // C
    append_dense_signature(ops, &sig);
    sig += std::to_string(cfg.block_n) + ",";
    sig += std::to_string(cfg.block_nnz) + ",";
    sig += ops.acc_is_fp64 ? "True," : "False,";     // ACC_IS_FP64
    // ACCURACY selects the blocked dot-product accumulation. The Python path
    // leaves it off by default and so does this one: turning it on here would
    // make the C API's result differ from the operator library's for the same
    // input, which is a worse surprise than the extra rounding.
    if (!ops.complex_op) sig += "False,";            // ACCURACY
    sig += ops.has_beta ? "True" : "False";          // HAS_BETA

    std::vector<jit::Arg> args;
    args.reserve(18);
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->indices)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->offsets)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(B->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(C->values)));
    append_dense_args(ops, ops.m, &args);

    const int64_t grid_n = (ops.n + cfg.block_n - 1) / cfg.block_n;
    std::string err;
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("spmm_csr.py"),
        ops.complex_op ? "spmm_csr_complex" : "spmm_csr_real", sig,
        ctx(handle)->stream, ops.m, grid_n, 1, cfg.num_warps, cfg.num_stages, args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}

flagsparseStatus_t run_coo(flagsparseHandle_t handle, SpMatDescr* A, const DnMatDescr* B,
                           DnMatDescr* C, const Operands& ops, void* externalBuffer) {
    // The row-offsets array is scratch the CALLER owns, sized by
    // flagsparseSpMM_bufferSize. Allocating it behind the caller's back would be
    // the one place in this library that touches the device allocator, and it
    // would hide a per-matrix cost inside a timed solve.
    if (externalBuffer == nullptr) {
        ctx(handle)->last_error =
            "SpMM on a COO matrix needs the externalBuffer reported by "
            "flagsparseSpMM_bufferSize; it holds the row-offsets array.";
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    if (A->coo_offsets_buffer != externalBuffer) {
        if (flagsparseStatus_t s = build_coo_row_offsets(A, externalBuffer)) return s;
    }

    const LaunchConfig cfg = resolve_coo_launch(ops.n, ctx(handle)->device_index);
    const char* it = triton_index_dtype(A->indices_type);

    std::string sig;
    sig.reserve(208);
    sig += "*"; sig += ops.vt; sig += ":16,";   // data
    sig += "*"; sig += it; sig += ":16,";       // row indices
    sig += "*"; sig += it; sig += ":16,";       // col indices
    sig += "*"; sig += ops.vt; sig += ":16,";   // B
    sig += "*"; sig += ops.vt; sig += ":16,";   // C
    sig += "*i32:16,";                          // seg_starts (always i32)
    append_dense_signature(ops, &sig);
    sig += std::to_string(cfg.block_n) + ",";
    sig += std::to_string(cfg.block_nnz) + ",";
    sig += ops.acc_is_fp64 ? "True," : "False,";   // ACC_IS_FP64
    // SEG_IS_ROW: one segment per ROW, not per run of equal row ids. That is what
    // makes an empty row still run and pick up beta * C.
    sig += "True,";                                // SEG_IS_ROW
    sig += ops.has_beta ? "True" : "False";        // HAS_BETA

    std::vector<jit::Arg> args;
    args.reserve(20);
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->row_ind)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->indices)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(B->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(C->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(externalBuffer)));
    // n_segs == n_rows under SEG_IS_ROW.
    append_dense_args(ops, ops.m, &args);

    const int64_t grid_n = (ops.n + cfg.block_n - 1) / cfg.block_n;
    std::string err;
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("spmm_coo.py"),
        ops.complex_op ? "spmm_coo_complex" : "spmm_coo_real", sig,
        ctx(handle)->stream, ops.m, grid_n, 1, cfg.num_warps, cfg.num_stages, args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}

flagsparseStatus_t run(flagsparseHandle_t handle, flagsparseOperation_t opB,
                       const void* alpha, flagsparseConstSpMatDescr_t matA,
                       flagsparseConstDnMatDescr_t matB, const void* beta,
                       flagsparseDnMatDescr_t matC, flagsparseDataType_t computeType,
                       void* externalBuffer) {
    auto* A = const_cast<SpMatDescr*>(spmat(matA));
    const DnMatDescr* B = dnmat(matB);
    DnMatDescr* C = dnmat(matC);

    Operands ops;
    if (flagsparseStatus_t s =
            resolve_operands(handle, opB, alpha, A, B, beta, C, computeType, &ops)) {
        return s;
    }
    if (ops.m == 0 || ops.n == 0) return FLAGSPARSE_STATUS_SUCCESS;

    return (A->format == FLAGSPARSE_FORMAT_COO)
               ? run_coo(handle, A, B, C, ops, externalBuffer)
               : run_csr(handle, A, B, C, computeType, ops);
}

}  // namespace

extern "C" {

flagsparseStatus_t flagsparseSpMM_bufferSize(
    flagsparseHandle_t handle, flagsparseOperation_t opA, flagsparseOperation_t opB,
    const void* alpha, flagsparseConstSpMatDescr_t matA, flagsparseConstDnMatDescr_t matB,
    const void* beta, flagsparseDnMatDescr_t matC, flagsparseDataType_t computeType,
    flagsparseSpMMAlg_t alg, size_t* bufferSize) {
    (void)alpha; (void)beta;
    if (bufferSize == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *bufferSize = 0;
    return guard(handle, [&]() -> flagsparseStatus_t {
        if (flagsparseStatus_t s =
                validate(handle, opA, opB, matA, matB, matC, computeType, alg)) {
            return s;
        }
        // CSR needs no scratch: its indptr already is the row-offsets array. COO
        // has to be given one, so the caller allocates rows + 1 int32 and passes
        // it to preprocess and to every solve.
        const SpMatDescr* A = spmat(matA);
        if (A->format == FLAGSPARSE_FORMAT_COO) {
            *bufferSize = static_cast<size_t>(A->rows + 1) * sizeof(std::int32_t);
        }
        return FLAGSPARSE_STATUS_SUCCESS;
    });
}

flagsparseStatus_t flagsparseSpMM_preprocess(
    flagsparseHandle_t handle, flagsparseOperation_t opA, flagsparseOperation_t opB,
    const void* alpha, flagsparseConstSpMatDescr_t matA, flagsparseConstDnMatDescr_t matB,
    const void* beta, flagsparseDnMatDescr_t matC, flagsparseDataType_t computeType,
    flagsparseSpMMAlg_t alg, void* externalBuffer) {
    (void)alpha; (void)beta;
    return guard(handle, [&]() -> flagsparseStatus_t {
        if (flagsparseStatus_t s =
                validate(handle, opA, opB, matA, matB, matC, computeType, alg)) {
            return s;
        }
        // Pay the index readback here so the timed solve does not -- this is what
        // a preprocess step is for. For CSR it only tunes the launch; for COO it
        // builds the row-offsets array the kernel cannot run without, which is
        // also where a COO that is not sorted by row gets rejected.
        auto* A = const_cast<SpMatDescr*>(spmat(matA));
        if (A->format == FLAGSPARSE_FORMAT_COO) {
            if (A->rows == 0) return FLAGSPARSE_STATUS_SUCCESS;
            if (externalBuffer == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
            return build_coo_row_offsets(A, externalBuffer);
        }
        int64_t unused = 0;
        return ensure_max_row_nnz(A, &unused);
    });
}

flagsparseStatus_t flagsparseSpMM(
    flagsparseHandle_t handle, flagsparseOperation_t opA, flagsparseOperation_t opB,
    const void* alpha, flagsparseConstSpMatDescr_t matA, flagsparseConstDnMatDescr_t matB,
    const void* beta, flagsparseDnMatDescr_t matC, flagsparseDataType_t computeType,
    flagsparseSpMMAlg_t alg, void* externalBuffer) {
    return guard(handle, [&]() -> flagsparseStatus_t {
        if (flagsparseStatus_t s =
                validate(handle, opA, opB, matA, matB, matC, computeType, alg)) {
            return s;
        }
        return run(handle, opB, alpha, matA, matB, beta, matC, computeType,
                   externalBuffer);
    });
}

}  // extern "C"
