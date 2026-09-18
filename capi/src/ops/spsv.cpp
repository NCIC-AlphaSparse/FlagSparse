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


// SpSV: solve op(A) * y = alpha * x for a triangular A.
//
// This is the first operator with a DESCRIPTOR lifecycle, and the reason for it
// is cuSPARSE's own shape: flagsparseSpSV_solve takes no externalBuffer, so the
// scratch handed to _analysis has to be remembered somewhere. That somewhere is
// the SpSV descriptor.
//
// The route is chain-wave: workers take rows from an atomic counter and spin on
// a per-row ready flag until every dependency has been published. One launch
// solves the whole system -- no level-by-level dispatch, and no level analysis
// to compute. Over-subscribing workers cannot deadlock, because a program only
// ever waits on rows with a LOWER logical index and those can only have been
// claimed by a program that already ran its atomic_add, i.e. a resident one.

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

// Where the worker-count sweep saturates on the reference card; past this the
// solve stops getting faster and only costs launch width.
constexpr int64_t kMaxWorkers = 2048;

bool is_sell_alg(flagsparseSpSVAlg_t alg) {
    return alg == FLAGSPARSE_SPSV_SELL_ALG1 || alg == FLAGSPARSE_SPSV_SELL_ALG2;
}
constexpr int kNumWarps = 1;
constexpr int kNumStages = 1;

// Internal shape of flagsparseSpSVDescr_t.
struct SpSVDescr {
    void* buffer = nullptr;                // scratch handed to _analysis
    const void* analysed_matrix = nullptr; // which matrix it was analysed for
    flagsparseOperation_t analysed_op = FLAGSPARSE_OPERATION_NON_TRANSPOSE;
    // SpSV_solve takes the alg again, but the SELL routes differ in how the
    // scratch is USED, so the one analysis ran with is what the solve must use.
    flagsparseSpSVAlg_t analysed_alg = FLAGSPARSE_SPSV_ALG_DEFAULT;
    bool analysed = false;
};

SpSVDescr* spsv(flagsparseSpSVDescr_t d) { return reinterpret_cast<SpSVDescr*>(d); }

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

// [ ready flags : n_rows int32 | row counter : 1 int32 ] and, for COO, the
// row-offsets array that makes it addressable as CSR: another n_rows + 1 int32.
// [ ready flags : n_rows int32 | row counter : 1 int32 ] for every format, plus
// for COO the row-offsets array that makes it addressable as CSR. Sliced-ELL
// needs no extra: its slice offsets already describe the structure.
size_t scratch_bytes(const SpMatDescr* A) {
    const size_t base = static_cast<size_t>(A->rows + 1) * sizeof(std::int32_t);
    return (A->format == FLAGSPARSE_FORMAT_COO) ? base * 2 : base;
}

// A row-sorted COO plus that offsets array IS a CSR matrix -- same column
// indices, same values -- so the solve runs the CSR kernel over this view
// rather than a second kernel or a converted copy.
SpMatDescr csr_view_of_coo(const SpMatDescr* A, void* offsets) {
    SpMatDescr view = *A;
    view.format = FLAGSPARSE_FORMAT_CSR;
    view.offsets = offsets;
    view.offsets_type = FLAGSPARSE_INDEX_32I;
    return view;
}

flagsparseStatus_t validate(flagsparseHandle_t handle, flagsparseOperation_t opA,
                            flagsparseConstSpMatDescr_t matA,
                            flagsparseConstDnVecDescr_t vecX,
                            flagsparseConstDnVecDescr_t vecY,
                            flagsparseDataType_t computeType,
                            flagsparseSpSVAlg_t alg,
                            flagsparseSpSVDescr_t spsvDescr) {
    if (handle == nullptr) return FLAGSPARSE_STATUS_NOT_INITIALIZED;
    if (matA == nullptr || vecX == nullptr || vecY == nullptr || spsvDescr == nullptr) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    // The SELL algorithm ids name a FORMAT, not a tuning knob: asking for one
    // on a CSR/COO matrix, or omitting one on a SELL matrix, is a caller mistake
    // rather than a hint to ignore.
    const bool sell = (spmat(matA)->format == FLAGSPARSE_FORMAT_SLICED_ELL);
    if (sell) {
        if (alg != FLAGSPARSE_SPSV_ALG_DEFAULT && !is_sell_alg(alg)) {
            return FLAGSPARSE_STATUS_NOT_SUPPORTED;
        }
    } else if (alg != FLAGSPARSE_SPSV_ALG_DEFAULT) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }

    const SpMatDescr* A = spmat(matA);
    const DnVecDescr* X = dnvec(vecX);
    const DnVecDescr* Y = dnvec(vecY);
    if (A->value_type != computeType || X->value_type != computeType ||
        Y->value_type != computeType) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (A->format != FLAGSPARSE_FORMAT_CSR && A->format != FLAGSPARSE_FORMAT_COO &&
        A->format != FLAGSPARSE_FORMAT_SLICED_ELL) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (A->format == FLAGSPARSE_FORMAT_COO &&
        A->nnz > static_cast<int64_t>(INT32_MAX)) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (A->idx_base != FLAGSPARSE_INDEX_BASE_ZERO) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    if (A->rows != A->cols) return FLAGSPARSE_STATUS_INVALID_VALUE;   // must be square
    if (X->size != A->rows || Y->size != A->rows) return FLAGSPARSE_STATUS_INVALID_VALUE;

    // op(A) = A^T needs the transposed traversal kernel, which is a separate
    // route in the operator package; refuse rather than solve the wrong system.
    if (opA != FLAGSPARSE_OPERATION_NON_TRANSPOSE) return FLAGSPARSE_STATUS_NOT_SUPPORTED;

    const flagsparseDataType_t component = component_dtype(computeType);
    if (component != FLAGSPARSE_R_32F && component != FLAGSPARSE_R_64F) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (triton_index_dtype(A->indices_type)[0] == '\0' ||
        triton_index_dtype(A->offsets_type)[0] == '\0') {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    // Which triangle and whether the diagonal is implicit are attributes, not
    // guesses: cuSPARSE requires them to be set before analysis, and picking a
    // default here would silently solve a different system.
    if (!A->fill_mode_set) {
        ctx(handle)->last_error =
            "SpSV needs FLAGSPARSE_SPMAT_FILL_MODE on the matrix descriptor "
            "(flagsparseSpMatSetAttribute) -- there is no safe default.";
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    if (A->format == FLAGSPARSE_FORMAT_SLICED_ELL) {
        if (A->slice_size <= 0) return FLAGSPARSE_STATUS_INVALID_VALUE;
        // All four SELL kernels test dependencies with a bare `col < row` and
        // carry no fill-mode constexpr, so an upper triangle would be solved as
        // though its entries were below the diagonal. Refusing beats returning
        // a plausible wrong answer.
        if (A->fill_mode != FLAGSPARSE_FILL_MODE_LOWER) {
            ctx(handle)->last_error =
                "SpSV on a sliced-ELL matrix supports FILL_MODE_LOWER only: the "
                "kernels have no fill-mode switch.";
            return FLAGSPARSE_STATUS_NOT_SUPPORTED;
        }
        // The column indices are i32 in every SELL kernel (padding is -1).
        if (A->indices_type != FLAGSPARSE_INDEX_32I ||
            A->offsets_type != FLAGSPARSE_INDEX_32I) {
            return FLAGSPARSE_STATUS_NOT_SUPPORTED;
        }
    }
    return FLAGSPARSE_STATUS_SUCCESS;
}

// Enough workers to fill the device once; rows are handed out dynamically, so
// more workers is strictly more parallelism until the machine is full.
int64_t resolve_worker_count(int64_t n_rows, int device_index) {
    const int warp = std::max(1, adaptor::warp_size(device_index));
    const int64_t per_mp = std::max<int64_t>(
        1, adaptor::max_threads_per_multiprocessor(device_index) / warp);
    const int64_t device_max =
        std::max<int64_t>(1, adaptor::multiprocessor_count(device_index) * per_mp);
    return std::max<int64_t>(1, std::min({n_rows, device_max, kMaxWorkers}));
}

// Sliced-ELL. Same scratch and the same no-deadlock argument as the CSR route
// -- workers take rows (ALG1) or slices (ALG2) from one ascending counter and
// only ever wait on lower indices -- so the only differences are the traversal
// and two extra constexprs.
flagsparseStatus_t solve_sell(flagsparseHandle_t handle, const SpMatDescr* A,
                              const DnVecDescr* X, DnVecDescr* Y,
                              flagsparseDataType_t computeType,
                              flagsparseSpSVAlg_t alg, std::int32_t* flags,
                              double alpha_re, double alpha_im) {
    const int64_t n = A->rows;
    const int64_t slice_size = A->slice_size;
    const int64_t n_slices = (n + slice_size - 1) / slice_size;
    // ALG2 gives one lane per row of a slice, and tl.arange needs a power of two.
    int64_t block_rows = 1;
    while (block_rows < slice_size) block_rows <<= 1;

    const bool complex_op = is_complex(computeType);
    const flagsparseDataType_t component = component_dtype(computeType);
    const bool acc_fp64 = (component == FLAGSPARSE_R_64F);
    const bool unit_diag = (A->diag_type == FLAGSPARSE_DIAG_TYPE_UNIT);
    // DEFAULT picks ALG1: it is the simpler traversal and the one that does not
    // depend on the slice being wide enough to keep its lanes busy.
    const bool use_alg2 = (alg == FLAGSPARSE_SPSV_SELL_ALG2);

    const char* vt = triton_dtype(component);
    std::string sig;
    sig.reserve(192);
    sig += "*"; sig += vt; sig += ":16,";   // values
    sig += "*i32:16,";                      // col_indices (padding is -1)
    sig += "*i32:16,";                      // slice_offsets
    sig += "*"; sig += vt; sig += ":16,";   // b
    sig += "*"; sig += vt; sig += ":16,";   // x
    sig += "*i32:16,";                      // ready
    sig += "*i32:16,";                      // row_counter
    sig += vt; sig += ",";                  // alpha (re)
    if (complex_op) { sig += vt; sig += ","; }
    sig += "i32,";                          // n_rows
    if (use_alg2) sig += "i32,";            // n_slices
    sig += std::to_string(slice_size) + ",";
    if (use_alg2) sig += std::to_string(block_rows) + ",";
    sig += unit_diag ? "True," : "False,";
    sig += acc_fp64 ? "True" : "False";

    std::vector<jit::Arg> args;
    args.reserve(12);
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->indices)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->offsets)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(X->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(Y->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(flags)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(flags + n)));
    if (acc_fp64) {
        args.push_back(jit::Arg::d(alpha_re));
        if (complex_op) args.push_back(jit::Arg::d(alpha_im));
    } else {
        args.push_back(jit::Arg::f(static_cast<float>(alpha_re)));
        if (complex_op) args.push_back(jit::Arg::f(static_cast<float>(alpha_im)));
    }
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(n)));
    if (use_alg2) args.push_back(jit::Arg::i(static_cast<std::int32_t>(n_slices)));

    const char* kernel;
    if (use_alg2) kernel = complex_op ? "_spsv_sell_slice_kernel_alg2_complex"
                                      : "_spsv_sell_slice_kernel_alg2";
    else          kernel = complex_op ? "_spsv_sell_cw_kernel_alg1_complex"
                                      : "_spsv_sell_cw_kernel_alg1";

    // ALG2's unit of work is a slice, so it needs proportionally fewer workers.
    const int64_t units = use_alg2 ? n_slices : n;
    const int64_t workers = resolve_worker_count(units, ctx(handle)->device_index);
    std::string err;
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("spsv.py"), kernel, sig, ctx(handle)->stream,
        workers, 1, 1, kNumWarps, kNumStages, args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}

flagsparseStatus_t solve(flagsparseHandle_t handle, const void* alpha,
                         flagsparseConstSpMatDescr_t matA,
                         flagsparseConstDnVecDescr_t vecX, flagsparseDnVecDescr_t vecY,
                         flagsparseDataType_t computeType, SpSVDescr* d) {
    const SpMatDescr* A0 = spmat(matA);
    const DnVecDescr* X = dnvec(vecX);
    DnVecDescr* Y = dnvec(vecY);

    if (!d->analysed || d->buffer == nullptr) {
        ctx(handle)->last_error =
            "flagsparseSpSV_solve requires flagsparseSpSV_analysis first: the "
            "solve takes no externalBuffer, so the descriptor holds it.";
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    if (d->analysed_matrix != static_cast<const void*>(A0)) {
        ctx(handle)->last_error =
            "this SpSV descriptor was analysed for a different matrix; call "
            "flagsparseSpSV_analysis again.";
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }

    double alpha_re = 1.0, alpha_im = 0.0;
    if (flagsparseStatus_t s = read_scalar(handle, alpha, computeType, &alpha_re,
                                           &alpha_im)) {
        return s;
    }
    const int64_t n = A0->rows;
    if (n == 0) return FLAGSPARSE_STATUS_SUCCESS;

    auto* flags32 = reinterpret_cast<std::int32_t*>(d->buffer);
    if (A0->format == FLAGSPARSE_FORMAT_SLICED_ELL) {
        // The ready flags and the row counter must start at zero; the kernels
        // only ever set flags, they never clear them.
        if (flagsparseStatus_t s = adaptor::memset_device(
                reinterpret_cast<adaptor::DevicePtr>(d->buffer), 0,
                static_cast<size_t>(n + 1) * sizeof(std::int32_t))) {
            return s;
        }
        return solve_sell(handle, A0, X, Y, computeType, d->analysed_alg, flags32,
                          alpha_re, alpha_im);
    }

    // COO runs through the CSR kernel over the offsets built by _analysis, which
    // live in the second half of the scratch.
    const SpMatDescr view =
        (A0->format == FLAGSPARSE_FORMAT_COO)
            ? csr_view_of_coo(A0, flags32 + (n + 1))
            : *A0;
    const SpMatDescr* A = &view;

    // The ready flags and the row counter must start at zero; the kernel only
    // ever sets flags, it never clears them.
    // Only the flags and the counter are cleared -- the offsets region, if any,
    // was filled by _analysis and must survive.
    if (flagsparseStatus_t s = adaptor::memset_device(
            reinterpret_cast<adaptor::DevicePtr>(d->buffer), 0,
            static_cast<size_t>(n + 1) * sizeof(std::int32_t))) {
        return s;
    }

    const bool complex_op = is_complex(computeType);
    const flagsparseDataType_t component = component_dtype(computeType);
    const bool acc_fp64 = (component == FLAGSPARSE_R_64F);
    const bool lower = (A->fill_mode == FLAGSPARSE_FILL_MODE_LOWER);
    const bool unit_diag = (A->diag_type == FLAGSPARSE_DIAG_TYPE_UNIT);
    // Matches _spsv_diag_eps_for_dtype: a diagonal smaller than this is treated
    // as 1 rather than dividing by ~0.
    const char* diag_eps = acc_fp64 ? "1e-12" : "1e-6";

    const char* vt = triton_dtype(component);
    const char* it = triton_index_dtype(A->indices_type);
    const char* ot = triton_index_dtype(A->offsets_type);

    std::string sig;
    sig.reserve(192);
    sig += "*"; sig += vt; sig += ":16,";   // data
    sig += "*"; sig += it; sig += ":16,";   // indices
    sig += "*"; sig += ot; sig += ":16,";   // indptr
    sig += "*"; sig += vt; sig += ":16,";   // b (x)
    sig += "*"; sig += vt; sig += ":16,";   // x (y)
    sig += "*i32:16,";                      // ready
    sig += "*i32:16,";                      // row_counter
    sig += vt; sig += ",";                  // alpha (re)
    if (complex_op) { sig += vt; sig += ","; }
    sig += "i32,";                          // n_rows
    sig += lower ? "True," : "False,";      // LOWER
    sig += lower ? "False," : "True,";      // REVERSE_ORDER: upper solves backwards
    sig += unit_diag ? "True," : "False,";  // UNIT_DIAG
    sig += acc_fp64 ? "True," : "False,";   // USE_FP64_ACC
    sig += diag_eps; sig += ",";            // DIAG_EPS
    sig += "False,";                        // SERIAL_EXECUTION
    // Scan from the far end of the row for an upper triangle. The scan stops at
    // the diagonal, so it needs the diagonal last; with cuSPARSE's universal
    // ascending column order that is the END of an upper row and the START of a
    // lower one. Doing it this way means no reordered copy of the matrix.
    sig += lower ? "False" : "True";        // SCAN_BACKWARD

    std::int32_t* flags = flags32;
    std::vector<jit::Arg> args;
    args.reserve(11);
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->indices)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->offsets)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(X->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(Y->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(flags)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(flags + n)));
    if (acc_fp64) {
        args.push_back(jit::Arg::d(alpha_re));
        if (complex_op) args.push_back(jit::Arg::d(alpha_im));
    } else {
        args.push_back(jit::Arg::f(static_cast<float>(alpha_re)));
        if (complex_op) args.push_back(jit::Arg::f(static_cast<float>(alpha_im)));
    }
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(n)));

    const int64_t workers = resolve_worker_count(n, ctx(handle)->device_index);
    std::string err;
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("spsv.py"),
        complex_op ? "_spsv_csr_cw_kernel_complex" : "_spsv_csr_cw_kernel", sig,
        ctx(handle)->stream, workers, 1, 1, kNumWarps, kNumStages, args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}

}  // namespace

extern "C" {

flagsparseStatus_t flagsparseSpSV_createDescr(flagsparseSpSVDescr_t* descr) {
    if (descr == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *descr = nullptr;
    auto* d = new (std::nothrow) SpSVDescr();
    if (d == nullptr) return FLAGSPARSE_STATUS_ALLOC_FAILED;
    *descr = reinterpret_cast<flagsparseSpSVDescr_t>(d);
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseSpSV_destroyDescr(flagsparseSpSVDescr_t descr) {
    if (descr == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    delete spsv(descr);
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseSpSV_bufferSize(
    flagsparseHandle_t handle, flagsparseOperation_t opA, const void* alpha,
    flagsparseConstSpMatDescr_t matA, flagsparseConstDnVecDescr_t vecX,
    flagsparseDnVecDescr_t vecY, flagsparseDataType_t computeType,
    flagsparseSpSVAlg_t alg, flagsparseSpSVDescr_t spsvDescr, size_t* bufferSize) {
    (void)alpha;
    if (bufferSize == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *bufferSize = 0;
    return guard(handle, [&]() -> flagsparseStatus_t {
        if (flagsparseStatus_t s = validate(handle, opA, matA, vecX, vecY, computeType,
                                            alg, spsvDescr)) {
            return s;
        }
        // One ready flag per row plus the shared row counter. No level table:
        // the chain-wave route discovers the order at run time.
        *bufferSize = scratch_bytes(spmat(matA));
        return FLAGSPARSE_STATUS_SUCCESS;
    });
}

flagsparseStatus_t flagsparseSpSV_analysis(
    flagsparseHandle_t handle, flagsparseOperation_t opA, const void* alpha,
    flagsparseConstSpMatDescr_t matA, flagsparseConstDnVecDescr_t vecX,
    flagsparseDnVecDescr_t vecY, flagsparseDataType_t computeType,
    flagsparseSpSVAlg_t alg, flagsparseSpSVDescr_t spsvDescr, void* externalBuffer) {
    (void)alpha;
    return guard(handle, [&]() -> flagsparseStatus_t {
        if (flagsparseStatus_t s = validate(handle, opA, matA, vecX, vecY, computeType,
                                            alg, spsvDescr)) {
            return s;
        }
        auto* A = const_cast<SpMatDescr*>(spmat(matA));
        if (A->rows > 0 && externalBuffer == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;

        // Sliced-ELL carries its own structure: slice offsets plus padding, and
        // the format already requires exactly one diagonal per row with padding
        // trailing. There is no CSR view to build and no column order to check.
        if (A->format == FLAGSPARSE_FORMAT_SLICED_ELL) {
            auto* sell = spsv(spsvDescr);
            sell->buffer = externalBuffer;
            sell->analysed_matrix = static_cast<const void*>(A);
            sell->analysed_op = opA;
            sell->analysed_alg = alg;
            sell->analysed = true;
            return FLAGSPARSE_STATUS_SUCCESS;
        }

        // A COO matrix gets its row-offsets array built here, into the second
        // half of the scratch; that is also where an unsorted COO is rejected.
        SpMatDescr checked = *A;
        if (A->format == FLAGSPARSE_FORMAT_COO && A->rows > 0) {
            auto* offsets = reinterpret_cast<std::int32_t*>(externalBuffer) + (A->rows + 1);
            if (flagsparseStatus_t s = build_coo_row_offsets(A, offsets)) return s;
            checked = csr_view_of_coo(A, offsets);
        }
        // The one structural requirement the kernel cannot check for itself: it
        // scans a row up to the diagonal, so an unsorted row stops early.
        if (flagsparseStatus_t s = check_csr_columns_sorted(&checked)) {
            ctx(handle)->last_error =
                "SpSV needs column indices sorted ascending within each row.";
            return s;
        }
        auto* d = spsv(spsvDescr);
        d->buffer = externalBuffer;
        d->analysed_matrix = static_cast<const void*>(A);
        d->analysed_op = opA;
        d->analysed_alg = alg;
        d->analysed = true;
        return FLAGSPARSE_STATUS_SUCCESS;
    });
}

flagsparseStatus_t flagsparseSpSV_solve(
    flagsparseHandle_t handle, flagsparseOperation_t opA, const void* alpha,
    flagsparseConstSpMatDescr_t matA, flagsparseConstDnVecDescr_t vecX,
    flagsparseDnVecDescr_t vecY, flagsparseDataType_t computeType,
    flagsparseSpSVAlg_t alg, flagsparseSpSVDescr_t spsvDescr) {
    return guard(handle, [&]() -> flagsparseStatus_t {
        if (flagsparseStatus_t s = validate(handle, opA, matA, vecX, vecY, computeType,
                                            alg, spsvDescr)) {
            return s;
        }
        return solve(handle, alpha, matA, vecX, vecY, computeType, spsv(spsvDescr));
    });
}

}  // extern "C"
