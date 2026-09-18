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


// SpSM: solve op(A) * Y = alpha * op(B) for a triangular A, with many
// right-hand sides.
//
// Same descriptor lifecycle as SpSV and for the same reason: the solve takes no
// externalBuffer, so the scratch given to _analysis lives on the descriptor.
//
// The route is polling: one program per (row, RHS tile), spinning on a
// per-(tile, row) done flag. Unlike SpSV it does not care about column order --
// it filters a row by col < row / col > row and scans all of it -- so ascending
// CSR works for both fill modes.
//
// The solver works IN PLACE on a packed row-major work array. cuSPARSE hands
// over a separate B and C with arbitrary order and leading dimension, so the
// scratch holds that work array and the solve copies in and out around it.

#include <algorithm>
#include <climits>
#include <cstdint>
#include <string>
#include <vector>

#include "adaptor/adaptor.hpp"
#include "core/internal.hpp"
#include "core/jit.hpp"
#include "core/prologue.hpp"

using namespace flagsparse;

namespace {

constexpr int64_t kMaxRhsTile = 1024;
constexpr int kDiagNumWarps = 1;
constexpr int kNumStages = 1;

struct SpSMDescr {
    void* buffer = nullptr;
    const void* analysed_matrix = nullptr;
    bool analysed = false;
};

SpSMDescr* spsm(flagsparseSpSMDescr_t d) { return reinterpret_cast<SpSMDescr*>(d); }

struct DenseStrides { int64_t row = 0; int64_t col = 0; };

DenseStrides strides_of(const DnMatDescr* M) {
    return (M->order == FLAGSPARSE_ORDER_ROW) ? DenseStrides{M->ld, 1}
                                              : DenseStrides{1, M->ld};
}

bool transposes(flagsparseOperation_t op) {
    return op != FLAGSPARSE_OPERATION_NON_TRANSPOSE;
}

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

int64_t rhs_tile_for(int64_t n_rhs) {
    int64_t v = 1;
    while (v < std::max<int64_t>(1, n_rhs)) v <<= 1;
    return std::min(kMaxRhsTile, v);
}

int num_warps_for(int64_t block_rhs) {
    if (block_rhs <= 32)  return 1;
    if (block_rhs <= 64)  return 2;
    if (block_rhs <= 256) return 4;
    return 8;
}

// Scratch layout, each section aligned so the ":16" pointer promise holds:
//   [ work : n_rows * n_rhs elements ][ done : tiles * n_rows int32 ][ diag : n_rows ]
struct Scratch {
    size_t work_bytes = 0, done_bytes = 0, diag_bytes = 0;
    size_t work_off = 0, done_off = 0, diag_off = 0, total = 0;
};

size_t align_up(size_t v) { return (v + 255u) & ~static_cast<size_t>(255u); }

Scratch plan_scratch(int64_t n_rows, int64_t n_rhs, flagsparseDataType_t computeType) {
    Scratch s;
    const size_t elem = dtype_size(computeType);
    const int64_t tiles = (n_rhs + rhs_tile_for(n_rhs) - 1) / rhs_tile_for(n_rhs);
    s.work_bytes = static_cast<size_t>(n_rows) * static_cast<size_t>(n_rhs) * elem;
    s.done_bytes = static_cast<size_t>(std::max<int64_t>(1, tiles)) *
                   static_cast<size_t>(n_rows) * sizeof(std::int32_t);
    s.diag_bytes = static_cast<size_t>(n_rows) * elem;
    s.work_off = 0;
    s.done_off = align_up(s.work_bytes);
    s.diag_off = align_up(s.done_off + s.done_bytes);
    s.total = align_up(s.diag_off + s.diag_bytes);
    return s;
}

flagsparseStatus_t validate(flagsparseHandle_t handle, flagsparseOperation_t opA,
                            flagsparseOperation_t opB,
                            flagsparseConstSpMatDescr_t matA,
                            flagsparseConstDnMatDescr_t matB,
                            flagsparseConstDnMatDescr_t matC,
                            flagsparseDataType_t computeType,
                            flagsparseSpSMAlg_t alg,
                            flagsparseSpSMDescr_t spsmDescr) {
    if (handle == nullptr) return FLAGSPARSE_STATUS_NOT_INITIALIZED;
    if (matA == nullptr || matB == nullptr || matC == nullptr || spsmDescr == nullptr) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    if (alg != FLAGSPARSE_SPSM_ALG_DEFAULT) return FLAGSPARSE_STATUS_NOT_SUPPORTED;

    const SpMatDescr* A = spmat(matA);
    const DnMatDescr* B = dnmat(matB);
    const DnMatDescr* C = dnmat(matC);
    if (A->value_type != computeType || B->value_type != computeType ||
        C->value_type != computeType) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (A->format != FLAGSPARSE_FORMAT_CSR && A->format != FLAGSPARSE_FORMAT_COO) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (A->idx_base != FLAGSPARSE_INDEX_BASE_ZERO) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    if (A->rows != A->cols) return FLAGSPARSE_STATUS_INVALID_VALUE;
    if (A->format == FLAGSPARSE_FORMAT_COO && A->nnz > static_cast<int64_t>(INT32_MAX)) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    // op(A) = A^T is a different traversal; op(B) = B^T is free, the copy into
    // the work array reads it with swapped strides.
    if (opA != FLAGSPARSE_OPERATION_NON_TRANSPOSE) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    if (opB == FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }

    const flagsparseDataType_t component = component_dtype(computeType);
    if (component != FLAGSPARSE_R_32F && component != FLAGSPARSE_R_64F) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (triton_index_dtype(A->indices_type)[0] == '\0' ||
        triton_index_dtype(A->offsets_type)[0] == '\0') {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (!A->fill_mode_set) {
        ctx(handle)->last_error =
            "SpSM needs FLAGSPARSE_SPMAT_FILL_MODE on the matrix descriptor "
            "(flagsparseSpMatSetAttribute) -- there is no safe default.";
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }

    const int64_t b_rows = transposes(opB) ? B->cols : B->rows;
    const int64_t b_cols = transposes(opB) ? B->rows : B->cols;
    if (b_rows != A->rows || C->rows != A->rows || C->cols != b_cols) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    return FLAGSPARSE_STATUS_SUCCESS;
}

// A row-sorted COO plus a row-offsets array is a CSR matrix; the offsets live
// where _analysis put them, at the end of the diag section.
SpMatDescr csr_view_of_coo(const SpMatDescr* A, void* offsets) {
    SpMatDescr view = *A;
    view.format = FLAGSPARSE_FORMAT_CSR;
    view.offsets = offsets;
    view.offsets_type = FLAGSPARSE_INDEX_32I;
    return view;
}

flagsparseStatus_t extract_diag(flagsparseHandle_t handle, const SpMatDescr* A,
                                void* diag, flagsparseDataType_t computeType) {
    const bool complex_op = is_complex(computeType);
    const flagsparseDataType_t component = component_dtype(computeType);
    const bool acc_fp64 = (component == FLAGSPARSE_R_64F);
    const bool unit_diag = (A->diag_type == FLAGSPARSE_DIAG_TYPE_UNIT);
    const char* vt = triton_dtype(component);
    const char* it = triton_index_dtype(A->indices_type);
    const char* ot = triton_index_dtype(A->offsets_type);

    std::string sig;
    sig += "*"; sig += vt; sig += ":16,";
    sig += "*"; sig += it; sig += ":16,";
    sig += "*"; sig += ot; sig += ":16,";
    sig += "*"; sig += vt; sig += ":16,";
    sig += "i32,";
    sig += unit_diag ? "True," : "False,";
    sig += acc_fp64 ? "True" : "False";

    std::vector<jit::Arg> args;
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->indices)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->offsets)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(diag)));
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(A->rows)));

    std::string err;
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("spsm.py"),
        complex_op ? "_spsm_extract_diag_kernel_complex" : "_spsm_extract_diag_kernel_real",
        sig, ctx(handle)->stream, A->rows, 1, 1, kDiagNumWarps, kNumStages, args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}

flagsparseStatus_t solve(flagsparseHandle_t handle, flagsparseOperation_t opB,
                         const void* alpha, flagsparseConstSpMatDescr_t matA,
                         flagsparseConstDnMatDescr_t matB, flagsparseDnMatDescr_t matC,
                         flagsparseDataType_t computeType, SpSMDescr* d) {
    const SpMatDescr* A0 = spmat(matA);
    const DnMatDescr* B = dnmat(matB);
    DnMatDescr* C = dnmat(matC);

    if (!d->analysed || d->buffer == nullptr) {
        ctx(handle)->last_error =
            "flagsparseSpSM_solve requires flagsparseSpSM_analysis first: the "
            "solve takes no externalBuffer, so the descriptor holds it.";
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    if (d->analysed_matrix != static_cast<const void*>(A0)) {
        ctx(handle)->last_error =
            "this SpSM descriptor was analysed for a different matrix; call "
            "flagsparseSpSM_analysis again.";
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }

    double alpha_re = 1.0, alpha_im = 0.0;
    if (flagsparseStatus_t s = read_scalar(handle, alpha, computeType, &alpha_re,
                                           &alpha_im)) {
        return s;
    }

    const int64_t n = A0->rows;
    const int64_t n_rhs = C->cols;
    if (n == 0 || n_rhs == 0) return FLAGSPARSE_STATUS_SUCCESS;

    const Scratch plan = plan_scratch(n, n_rhs, computeType);
    auto* base = static_cast<unsigned char*>(d->buffer);
    void* work = base + plan.work_off;
    void* done = base + plan.done_off;
    void* diag = base + plan.diag_off;
    const SpMatDescr view =
        (A0->format == FLAGSPARSE_FORMAT_COO)
            ? csr_view_of_coo(A0, base + plan.total)
            : *A0;
    const SpMatDescr* A = &view;

    const bool complex_op = is_complex(computeType);
    const flagsparseDataType_t component = component_dtype(computeType);
    const bool acc_fp64 = (component == FLAGSPARSE_R_64F);
    const int64_t unit = complex_op ? 2 : 1;

    // B -> work, applying op(B) as a stride swap. work is packed row-major, so
    // its row stride is n_rhs elements and its column stride is 1.
    const DenseStrides bs = strides_of(B);
    if (flagsparseStatus_t s = copy_dense(
            handle, B->values, work, component, complex_op, n, n_rhs,
            (transposes(opB) ? bs.col : bs.row) * unit,
            (transposes(opB) ? bs.row : bs.col) * unit,
            n_rhs * unit, unit)) {
        return s;
    }
    // The done flags must start at zero; the kernel only ever sets them.
    if (flagsparseStatus_t s = adaptor::memset_device(
            reinterpret_cast<adaptor::DevicePtr>(done), 0, plan.done_bytes)) {
        return s;
    }

    const int64_t block_rhs = rhs_tile_for(n_rhs);
    const int64_t tiles = (n_rhs + block_rhs - 1) / block_rhs;
    const bool lower = (A->fill_mode == FLAGSPARSE_FILL_MODE_LOWER);
    const bool unit_diag = (A->diag_type == FLAGSPARSE_DIAG_TYPE_UNIT);
    const char* vt = triton_dtype(component);
    const char* it = triton_index_dtype(A->indices_type);
    const char* ot = triton_index_dtype(A->offsets_type);

    std::string sig;
    sig.reserve(208);
    sig += "*"; sig += vt; sig += ":16,";   // data
    sig += "*"; sig += it; sig += ":16,";   // indices
    sig += "*"; sig += ot; sig += ":16,";   // indptr
    sig += "*"; sig += vt; sig += ":16,";   // diag
    sig += "*"; sig += vt; sig += ":16,";   // work
    sig += "*i32:16,";                      // done
    sig += "i32,i32,";                      // n_rows, n_rhs
    sig += "i64,";                          // stride_work0, in ELEMENTS
    sig += vt; sig += ",";                  // alpha (re)
    if (complex_op) { sig += vt; sig += ","; }
    sig += std::to_string(block_rhs) + ",";
    sig += acc_fp64 ? "True," : "False,";
    sig += lower ? "True," : "False,";
    sig += unit_diag ? "True" : "False";

    std::vector<jit::Arg> args;
    args.reserve(12);
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->indices)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->offsets)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(diag)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(work)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(done)));
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(n)));
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(n_rhs)));
    args.push_back(jit::Arg::i64v(n_rhs));
    if (acc_fp64) {
        args.push_back(jit::Arg::d(alpha_re));
        if (complex_op) args.push_back(jit::Arg::d(alpha_im));
    } else {
        args.push_back(jit::Arg::f(static_cast<float>(alpha_re)));
        if (complex_op) args.push_back(jit::Arg::f(static_cast<float>(alpha_im)));
    }

    std::string err;
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("spsm.py"),
        complex_op ? "_spsm_csr_polling_kernel_complex" : "_spsm_csr_polling_kernel_real",
        sig, ctx(handle)->stream, n, tiles, 1, num_warps_for(block_rhs), kNumStages,
        args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) {
        ctx(handle)->last_error = err;
        return st;
    }

    // work -> C, back out to whatever order and leading dimension C has.
    const DenseStrides cs = strides_of(C);
    return copy_dense(handle, work, C->values, component, complex_op, n, n_rhs,
                      n_rhs * unit, unit, cs.row * unit, cs.col * unit);
}

}  // namespace

extern "C" {

flagsparseStatus_t flagsparseSpSM_createDescr(flagsparseSpSMDescr_t* descr) {
    if (descr == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *descr = nullptr;
    auto* d = new (std::nothrow) SpSMDescr();
    if (d == nullptr) return FLAGSPARSE_STATUS_ALLOC_FAILED;
    *descr = reinterpret_cast<flagsparseSpSMDescr_t>(d);
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseSpSM_destroyDescr(flagsparseSpSMDescr_t descr) {
    if (descr == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    delete spsm(descr);
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseSpSM_bufferSize(
    flagsparseHandle_t handle, flagsparseOperation_t opA, flagsparseOperation_t opB,
    const void* alpha, flagsparseConstSpMatDescr_t matA, flagsparseConstDnMatDescr_t matB,
    flagsparseDnMatDescr_t matC, flagsparseDataType_t computeType,
    flagsparseSpSMAlg_t alg, flagsparseSpSMDescr_t spsmDescr, size_t* bufferSize) {
    (void)alpha;
    if (bufferSize == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *bufferSize = 0;
    return guard(handle, [&]() -> flagsparseStatus_t {
        if (flagsparseStatus_t s = validate(handle, opA, opB, matA, matB, matC,
                                            computeType, alg, spsmDescr)) {
            return s;
        }
        const SpMatDescr* A = spmat(matA);
        const Scratch plan = plan_scratch(A->rows, dnmat(matC)->cols, computeType);
        *bufferSize = plan.total;
        // COO needs its row-offsets array on top of the solver's own scratch.
        if (A->format == FLAGSPARSE_FORMAT_COO) {
            *bufferSize += static_cast<size_t>(A->rows + 1) * sizeof(std::int32_t);
        }
        return FLAGSPARSE_STATUS_SUCCESS;
    });
}

flagsparseStatus_t flagsparseSpSM_analysis(
    flagsparseHandle_t handle, flagsparseOperation_t opA, flagsparseOperation_t opB,
    const void* alpha, flagsparseConstSpMatDescr_t matA, flagsparseConstDnMatDescr_t matB,
    flagsparseDnMatDescr_t matC, flagsparseDataType_t computeType,
    flagsparseSpSMAlg_t alg, flagsparseSpSMDescr_t spsmDescr, void* externalBuffer) {
    (void)alpha;
    return guard(handle, [&]() -> flagsparseStatus_t {
        if (flagsparseStatus_t s = validate(handle, opA, opB, matA, matB, matC,
                                            computeType, alg, spsmDescr)) {
            return s;
        }
        auto* A = const_cast<SpMatDescr*>(spmat(matA));
        if (A->rows > 0 && externalBuffer == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
        auto* d = spsm(spsmDescr);
        if (A->rows == 0) { d->buffer = externalBuffer; d->analysed_matrix = A;
                            d->analysed = true; return FLAGSPARSE_STATUS_SUCCESS; }

        const Scratch plan = plan_scratch(A->rows, dnmat(matC)->cols, computeType);
        auto* base = static_cast<unsigned char*>(externalBuffer);
        SpMatDescr view = *A;
        if (A->format == FLAGSPARSE_FORMAT_COO) {
            auto* offsets = base + plan.total;
            if (flagsparseStatus_t s = build_coo_row_offsets(A, offsets)) return s;
            view = csr_view_of_coo(A, offsets);
        }
        // The diagonal depends only on the matrix, so extracting it here means a
        // repeated solve with new right-hand sides does not pay for it again.
        if (flagsparseStatus_t s = extract_diag(handle, &view, base + plan.diag_off,
                                                computeType)) {
            return s;
        }
        d->buffer = externalBuffer;
        d->analysed_matrix = static_cast<const void*>(A);
        d->analysed = true;
        return FLAGSPARSE_STATUS_SUCCESS;
    });
}

flagsparseStatus_t flagsparseSpSM_solve(
    flagsparseHandle_t handle, flagsparseOperation_t opA, flagsparseOperation_t opB,
    const void* alpha, flagsparseConstSpMatDescr_t matA, flagsparseConstDnMatDescr_t matB,
    flagsparseDnMatDescr_t matC, flagsparseDataType_t computeType,
    flagsparseSpSMAlg_t alg, flagsparseSpSMDescr_t spsmDescr) {
    return guard(handle, [&]() -> flagsparseStatus_t {
        if (flagsparseStatus_t s = validate(handle, opA, opB, matA, matB, matC,
                                            computeType, alg, spsmDescr)) {
            return s;
        }
        return solve(handle, opB, alpha, matA, matB, matC, computeType,
                     spsm(spsmDescr));
    });
}

}  // extern "C"
