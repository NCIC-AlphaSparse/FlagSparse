// SpMV: y = alpha * op(A) * x + beta * y, with A in CSR, COO or CSC.
//
// The first operator taken end to end through the C++ dispatch layer; every
// other one follows this file's shape -- validate, route, size the launch, hand
// raw pointers to libtriton_jit.
//
// The three formats split into two kinds of route, and the split is what drives
// the code below:
//
//  * One program owns an output element (CSR, COO segments, CSC transposed).
//    Deterministic, and both alpha and beta fold into the store.
//  * Programs scatter into the output with atomics (CSC non-transposed). No
//    program owns an element, so only alpha fits in the kernel and `beta * y`
//    is applied first by core/prologue.hpp. Not bit-reproducible in fp32,
//    which is a property of atomic accumulation, not a defect.

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

constexpr int kCsrBlockNnz = 128;
constexpr int kCsrNumWarps = 4;
constexpr int kCsrNumStages = 2;
// COO's segment kernel walks a row with a `while pos < end` loop over
// BLOCK_INNER-wide tiles, so this is a tile width, not an unroll factor -- the
// BLOCK_NNZ=4 lesson from SpMM COO does not apply here.
constexpr int kCooBlockInner = 32;
constexpr int kCooNumWarps = 1;
// CSC's own default, and MAX_SEGMENTS is derived from it.
constexpr int kCscBlockNnz = 256;
constexpr int kCscNumWarps = 4;
// BSR: one program per (block row, inner row), vectorised over BLOCK_NNZ blocks.
constexpr int kBsrBlockNnz = 32;
constexpr int kBsrNumWarps = 4;
constexpr int kDefaultNumStages = 2;

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

struct Operands {
    bool complex_op = false;
    bool acc_is_fp64 = false;
    bool has_beta = false;
    double alpha_re = 1.0, alpha_im = 0.0, beta_re = 0.0, beta_im = 0.0;
    int64_t y_len = 0;
    const char* vt = "";
    const char* it = "";
    const char* ot = "";
};

void push_scalar(const Operands& ops, double v, std::vector<jit::Arg>* args) {
    if (ops.acc_is_fp64) args->push_back(jit::Arg::d(v));
    else                 args->push_back(jit::Arg::f(static_cast<float>(v)));
}

// Which algorithm ids belong to which format. An id naming a different format is
// refused rather than quietly ignored -- that is a caller mistake worth
// surfacing. This build has one route per (format, direction), and an alg is a
// performance hint over the same result, so every id of the matrix's own format
// is accepted.
bool alg_supported(flagsparseFormat_t format, flagsparseSpMVAlg_t alg) {
    if (alg == FLAGSPARSE_SPMV_ALG_DEFAULT) return true;
    switch (format) {
        case FLAGSPARSE_FORMAT_CSR:
        case FLAGSPARSE_FORMAT_CSC:
        case FLAGSPARSE_FORMAT_BSR:
            return alg == FLAGSPARSE_SPMV_CSR_ALG1 || alg == FLAGSPARSE_SPMV_CSR_ALG2;
        case FLAGSPARSE_FORMAT_COO:
            return alg == FLAGSPARSE_SPMV_COO_ALG1 || alg == FLAGSPARSE_SPMV_COO_ALG2;
        default:
            return false;
    }
}

// op(A) is m x n; y has m entries and x has n. A BSR descriptor counts BLOCKS
// in rows/cols (cuSPARSE does the same), so its scalar extents are the block
// counts times the block dimensions.
void operand_extents(const SpMatDescr* A, flagsparseOperation_t opA,
                     int64_t* need_x, int64_t* need_y) {
    int64_t m = A->rows, n = A->cols;
    if (A->format == FLAGSPARSE_FORMAT_BSR) {
        m *= A->row_block_dim;
        n *= A->col_block_dim;
    }
    *need_x = transposes(opA) ? m : n;
    *need_y = transposes(opA) ? n : m;
}

flagsparseStatus_t validate(flagsparseHandle_t handle, flagsparseOperation_t opA,
                            flagsparseConstSpMatDescr_t matA,
                            flagsparseConstDnVecDescr_t vecX,
                            flagsparseConstDnVecDescr_t vecY,
                            flagsparseDataType_t computeType,
                            flagsparseSpMVAlg_t alg) {
    if (handle == nullptr) return FLAGSPARSE_STATUS_NOT_INITIALIZED;
    if (matA == nullptr || vecX == nullptr || vecY == nullptr) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    const SpMatDescr* A = spmat(matA);
    const DnVecDescr* X = dnvec(vecX);
    const DnVecDescr* Y = dnvec(vecY);

    if (A->value_type != computeType || X->value_type != computeType ||
        Y->value_type != computeType) {
        // Mixed-precision SpMV is a cuSPARSE feature this build does not have;
        // report it rather than computing in the wrong type.
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (A->idx_base != FLAGSPARSE_INDEX_BASE_ZERO) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    if (!alg_supported(A->format, alg)) return FLAGSPARSE_STATUS_NOT_SUPPORTED;

    int64_t need_x = 0, need_y = 0;
    operand_extents(A, opA, &need_x, &need_y);
    if (X->size != need_x || Y->size != need_y) return FLAGSPARSE_STATUS_INVALID_VALUE;

    const flagsparseDataType_t component = component_dtype(computeType);
    if (component != FLAGSPARSE_R_32F && component != FLAGSPARSE_R_64F) {
        // fp16/bf16 would need a compute-precision copy of the operands, the way
        // the Python path casts first. Not wired up.
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (triton_index_dtype(A->indices_type)[0] == '\0' ||
        triton_index_dtype(A->offsets_type)[0] == '\0') {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }

    switch (A->format) {
        case FLAGSPARSE_FORMAT_CSR:
            // op(A) = A^T on CSR needs the transposed matrix built first; that is
            // a prepare step this operator does not have. A caller who has CSC
            // arrays gets the transposed direction for free -- see run_csc.
            if (transposes(opA)) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
            break;
        case FLAGSPARSE_FORMAT_COO:
            if (transposes(opA)) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
            if (A->nnz > static_cast<int64_t>(INT32_MAX)) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
            break;
        case FLAGSPARSE_FORMAT_CSC:
            // Both directions work here, and conjugate transpose costs nothing
            // extra: the kernel carries a CONJ constexpr. Conjugating a real
            // matrix is just the plain transpose.
            break;
        case FLAGSPARSE_FORMAT_BSR:
            // The kernel uses one BLOCK_DIM for both block extents and reads a
            // block row-major, so anything else would be read wrongly rather
            // than refused.
            if (A->row_block_dim != A->col_block_dim) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
            if (A->row_block_dim <= 0) return FLAGSPARSE_STATUS_INVALID_VALUE;
            if (A->order != FLAGSPARSE_ORDER_ROW) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
            break;
        default:
            return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t resolve_operands(flagsparseHandle_t handle, const void* alpha,
                                    const void* beta, const DnVecDescr* Y,
                                    flagsparseDataType_t computeType, Operands* out,
                                    const SpMatDescr* A) {
    if (flagsparseStatus_t s =
            read_scalar(handle, alpha, computeType, &out->alpha_re, &out->alpha_im)) {
        return s;
    }
    if (flagsparseStatus_t s =
            read_scalar(handle, beta, computeType, &out->beta_re, &out->beta_im)) {
        return s;
    }
    out->complex_op = is_complex(computeType);
    const flagsparseDataType_t component = component_dtype(computeType);
    out->acc_is_fp64 = (component == FLAGSPARSE_R_64F);
    out->has_beta = (out->beta_re != 0.0 || out->beta_im != 0.0);
    out->y_len = Y->size;
    out->vt = triton_dtype(component);
    out->it = triton_index_dtype(A->indices_type);
    out->ot = triton_index_dtype(A->offsets_type);
    return FLAGSPARSE_STATUS_SUCCESS;
}

// --------------------------------------------------------------- routes ---

// The row-parallel CSR kernel, parameterised by where the offsets come from.
// A row-sorted COO plus the row-offsets array the C API builds for it IS a CSR
// matrix -- same values, same column indices -- so this serves both.
flagsparseStatus_t launch_csr_rowpar(flagsparseHandle_t handle, SpMatDescr* A,
                                     void* offsets, const char* offsets_type,
                                     int64_t n_rows, int64_t max_row_nnz,
                                     const DnVecDescr* X, DnVecDescr* Y,
                                     const Operands& ops) {
    // At least one segment: a matrix with only empty rows still has to run so
    // that the beta term is applied to y.
    const int64_t segments =
        std::max<int64_t>(1, (max_row_nnz + kCsrBlockNnz - 1) / kCsrBlockNnz);

    // The complex kernel is a separate function with its own name, not a dtype
    // constexpr on the real one, so the only difference here is one extra scalar
    // per side: Triton has no complex type, and alpha/beta arrive split into
    // components the way the operands themselves are.
    std::string sig;
    sig.reserve(160);
    sig += "*"; sig += ops.vt; sig += ":16,";   // data
    sig += "*"; sig += ops.it; sig += ":16,";   // indices
    sig += "*"; sig += offsets_type; sig += ":16,";   // indptr
    sig += "*"; sig += ops.vt; sig += ":16,";   // x
    sig += "*"; sig += ops.vt; sig += ":16,";   // y
    sig += ops.vt; sig += ",";                  // alpha (re)
    if (ops.complex_op) { sig += ops.vt; sig += ","; }
    sig += ops.vt; sig += ",";                  // beta (re)
    if (ops.complex_op) { sig += ops.vt; sig += ","; }
    sig += "i32,";                              // n_rows
    sig += std::to_string(kCsrBlockNnz) + ",";
    sig += std::to_string(segments) + ",";
    sig += ops.has_beta ? "True" : "False";
    // XPU_COMPAT (real kernel only): the Python side sets it on Kunlunxin XPU,
    // where the C API does not run, so it is always False here. Leaving it out
    // of the signature fails every real CSR / COO_ALG2 launch on every backend.
    if (!ops.complex_op) sig += ",False";

    std::vector<jit::Arg> args;
    args.reserve(8);
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->indices)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(offsets)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(X->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(Y->values)));
    push_scalar(ops, ops.alpha_re, &args);
    if (ops.complex_op) push_scalar(ops, ops.alpha_im, &args);
    push_scalar(ops, ops.beta_re, &args);
    if (ops.complex_op) push_scalar(ops, ops.beta_im, &args);
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(n_rows)));

    std::string err;
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("spmv_csr.py"),
        ops.complex_op ? "_spmv_csr_complex_kernel" : "_spmv_csr_real_kernel", sig,
        ctx(handle)->stream, n_rows, 1, 1, kCsrNumWarps, kCsrNumStages, args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}

flagsparseStatus_t run_csr(flagsparseHandle_t handle, SpMatDescr* A,
                           const DnVecDescr* X, DnVecDescr* Y, const Operands& ops) {
    int64_t max_row_nnz = 0;
    if (flagsparseStatus_t s = ensure_max_row_nnz(A, &max_row_nnz)) return s;
    return launch_csr_rowpar(handle, A, A->offsets, ops.ot, A->rows, max_row_nnz, X, Y,
                             ops);
}

// COO uses the deterministic segment route with SEG_IS_ROW=True: seg_starts is
// a full row-offsets array, so the segment index IS the row and a row with no
// nonzeros still runs and picks up beta * y. Same contract as SpMM COO, same
// scratch buffer, same sorted-COO requirement.
flagsparseStatus_t run_coo(flagsparseHandle_t handle, SpMatDescr* A,
                           const DnVecDescr* X, DnVecDescr* Y, const Operands& ops,
                           void* externalBuffer, bool to_csr) {
    if (externalBuffer == nullptr) {
        ctx(handle)->last_error =
            "SpMV on a COO matrix needs the externalBuffer reported by "
            "flagsparseSpMV_bufferSize; it holds the row-offsets array.";
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    if (A->coo_offsets_buffer != externalBuffer) {
        if (flagsparseStatus_t s = build_coo_row_offsets(A, externalBuffer)) return s;
    }

    // COO_ALG2 = run the CSR row-parallel kernel over the offsets just built.
    // Nothing is converted or copied: a row-sorted COO's column indices and
    // values already ARE the CSR arrays, so the offsets are the only thing that
    // was missing. Complex goes through it too now that the CSR launcher picks
    // the complex kernel by name.
    if (to_csr) {
        return launch_csr_rowpar(handle, A, externalBuffer, "i32", A->rows,
                                 A->max_row_nnz < 0 ? 0 : A->max_row_nnz, X, Y, ops);
    }

    std::string sig;
    sig.reserve(160);
    sig += "*"; sig += ops.vt; sig += ":16,";   // data
    sig += "*"; sig += ops.it; sig += ":16,";   // col
    sig += "*"; sig += ops.it; sig += ":16,";   // row
    sig += "*"; sig += ops.vt; sig += ":16,";   // x
    sig += "*"; sig += ops.vt; sig += ":16,";   // y
    sig += "*i32:16,";                          // seg_starts
    sig += ops.vt; sig += ",";                  // alpha (re)
    if (ops.complex_op) { sig += ops.vt; sig += ","; }
    sig += ops.vt; sig += ",";                  // beta (re)
    if (ops.complex_op) { sig += ops.vt; sig += ","; }
    sig += "i32,";                              // n_segs
    sig += std::to_string(kCooBlockInner) + ",";
    if (ops.complex_op) sig += ops.acc_is_fp64 ? "True," : "False,";   // ACC_IS_FP64
    sig += "True,";                             // SEG_IS_ROW
    sig += ops.has_beta ? "True" : "False";

    std::vector<jit::Arg> args;
    args.reserve(12);
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->indices)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->row_ind)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(X->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(Y->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(externalBuffer)));
    push_scalar(ops, ops.alpha_re, &args);
    if (ops.complex_op) push_scalar(ops, ops.alpha_im, &args);
    push_scalar(ops, ops.beta_re, &args);
    if (ops.complex_op) push_scalar(ops, ops.beta_im, &args);
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(A->rows)));

    // The real kernels bake the accumulator into their name; only the complex
    // one takes it as a tl.dtype constexpr and so goes through the wrapper.
    const char* kernel = ops.complex_op ? "spmv_coo_seg_complex"
                                        : (ops.acc_is_fp64 ? "_spmv_coo_seg_f64"
                                                           : "_spmv_coo_seg_f32");
    std::string err;
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("spmv_coo.py"), kernel, sig, ctx(handle)->stream,
        A->rows, 1, 1, kCooNumWarps, kDefaultNumStages, args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}

flagsparseStatus_t run_csc(flagsparseHandle_t handle, flagsparseOperation_t opA,
                           SpMatDescr* A, const DnVecDescr* X, DnVecDescr* Y,
                           const Operands& ops) {
    int64_t max_col_nnz = 0;
    if (flagsparseStatus_t s = ensure_max_row_nnz(A, &max_col_nnz)) return s;
    const int64_t segments =
        std::max<int64_t>(1, (max_col_nnz + kCscBlockNnz - 1) / kCscBlockNnz);
    const bool trans = transposes(opA);

    // The non-transposed direction scatters into y with atomics, so beta cannot
    // ride along in the kernel; apply it first.
    if (!trans) {
        const int64_t unit = ops.complex_op ? 2 : 1;
        if (flagsparseStatus_t s = scale_dense(
                handle, Y->values, component_dtype(A->value_type), ops.complex_op,
                1, ops.y_len, 0, unit, ops.beta_re, ops.beta_im)) {
            return s;
        }
    }

    std::string sig;
    sig.reserve(176);
    sig += "*"; sig += ops.vt; sig += ":16,";   // data
    sig += "*"; sig += ops.it; sig += ":16,";   // indices (row ids)
    sig += "*"; sig += ops.ot; sig += ":16,";   // indptr (col offsets)
    sig += "*"; sig += ops.vt; sig += ":16,";   // x
    sig += "*"; sig += ops.vt; sig += ":16,";   // y
    sig += ops.vt; sig += ",";                  // alpha (re)
    if (ops.complex_op) { sig += ops.vt; sig += ","; }
    if (trans) {
        sig += ops.vt; sig += ",";              // beta (re)
        if (ops.complex_op) { sig += ops.vt; sig += ","; }
    }
    sig += "i32,";                              // n_cols
    sig += std::to_string(kCscBlockNnz) + ",";
    if (trans) {
        sig += std::to_string(segments) + ",";  // MAX_SEGMENTS
        if (ops.complex_op) {
            sig += (opA == FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE) ? "True," : "False,";
        }
        sig += ops.has_beta ? "True" : "False"; // HAS_BETA
    } else {
        sig.pop_back();                         // no constexpr follows BLOCK_NNZ
    }

    std::vector<jit::Arg> args;
    args.reserve(12);
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->indices)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->offsets)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(X->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(Y->values)));
    push_scalar(ops, ops.alpha_re, &args);
    if (ops.complex_op) push_scalar(ops, ops.alpha_im, &args);
    if (trans) {
        push_scalar(ops, ops.beta_re, &args);
        if (ops.complex_op) push_scalar(ops, ops.beta_im, &args);
    }
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(A->cols)));

    const char* kernel;
    if (trans) kernel = ops.complex_op ? "_spmv_csc_trans_complex_kernel"
                                       : "_spmv_csc_trans_real_kernel";
    else       kernel = ops.complex_op ? "_spmv_csc_non_complex_kernel"
                                       : "_spmv_csc_non_real_kernel";

    // Transposed: one program per column. Non-transposed: a (column, segment)
    // rectangle, because a column's nonzeros are spread over many programs.
    const int64_t grid_y = trans ? 1 : segments;
    std::string err;
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("spmv_csc.py"), kernel, sig, ctx(handle)->stream,
        A->cols, grid_y, 1, kCscNumWarps, kDefaultNumStages, args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}

// BSR scatters into y with atomics in both directions, so beta comes from the
// prologue and only alpha rides in the kernel. The segment index comes from the
// grid (SEG_FROM_GRID=True) rather than the SEG constexpr: as a constexpr every
// segment value compiles its own kernel, which turns one solve into as many JIT
// compilations as the longest block row has segments.
flagsparseStatus_t run_bsr(flagsparseHandle_t handle, flagsparseOperation_t opA,
                           SpMatDescr* A, const DnVecDescr* X, DnVecDescr* Y,
                           const Operands& ops) {
    int64_t max_block_row_nnz = 0;
    if (flagsparseStatus_t s = ensure_max_row_nnz(A, &max_block_row_nnz)) return s;
    const int64_t segments =
        std::max<int64_t>(1, (max_block_row_nnz + kBsrBlockNnz - 1) / kBsrBlockNnz);
    const bool trans = transposes(opA);
    const int64_t block_dim = A->row_block_dim;

    const int64_t unit = ops.complex_op ? 2 : 1;
    if (flagsparseStatus_t s = scale_dense(
            handle, Y->values, component_dtype(A->value_type), ops.complex_op,
            1, ops.y_len, 0, unit, ops.beta_re, ops.beta_im)) {
        return s;
    }
    if (A->nnz == 0) return FLAGSPARSE_STATUS_SUCCESS;   // beta * y is the answer

    std::string sig;
    sig.reserve(192);
    sig += "*"; sig += ops.vt; sig += ":16,";   // data
    sig += "*"; sig += ops.it; sig += ":16,";   // block col indices
    sig += "*"; sig += ops.ot; sig += ":16,";   // block row offsets
    sig += "*"; sig += ops.vt; sig += ":16,";   // x
    sig += "*"; sig += ops.vt; sig += ":16,";   // y
    sig += ops.vt; sig += ",";                  // alpha (re)
    if (ops.complex_op) { sig += ops.vt; sig += ","; }
    if (!trans) sig += "i32,i32,";              // n_rows, n_cols (scalar extents)
    sig += "i32,";                              // n_block_rows
    sig += std::to_string(block_dim) + ",";     // BLOCK_DIM
    sig += std::to_string(kBsrBlockNnz) + ",";  // BLOCK_NNZ
    sig += "0,";                                // SEG, unused under SEG_FROM_GRID
    if (trans && ops.complex_op) {
        sig += (opA == FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE) ? "True," : "False,";
    }
    sig += "True";                              // SEG_FROM_GRID

    std::vector<jit::Arg> args;
    args.reserve(12);
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->indices)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->offsets)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(X->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(Y->values)));
    push_scalar(ops, ops.alpha_re, &args);
    if (ops.complex_op) push_scalar(ops, ops.alpha_im, &args);
    if (!trans) {
        args.push_back(jit::Arg::i(static_cast<std::int32_t>(A->rows * block_dim)));
        args.push_back(jit::Arg::i(static_cast<std::int32_t>(A->cols * block_dim)));
    }
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(A->rows)));

    const char* kernel;
    if (trans) kernel = ops.complex_op ? "_spmv_bsr_trans_complex_kernel"
                                       : "_spmv_bsr_trans_real_kernel";
    else       kernel = ops.complex_op ? "_spmv_bsr_non_complex_kernel"
                                       : "_spmv_bsr_non_real_kernel";

    std::string err;
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("spmv_bsr.py"), kernel, sig, ctx(handle)->stream,
        A->rows, block_dim, segments, kBsrNumWarps, kDefaultNumStages, args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}

flagsparseStatus_t run(flagsparseHandle_t handle, flagsparseOperation_t opA,
                       const void* alpha, flagsparseConstSpMatDescr_t matA,
                       flagsparseConstDnVecDescr_t vecX, const void* beta,
                       flagsparseDnVecDescr_t vecY, flagsparseDataType_t computeType,
                       flagsparseSpMVAlg_t alg, void* externalBuffer) {
    auto* A = const_cast<SpMatDescr*>(spmat(matA));
    const DnVecDescr* X = dnvec(vecX);
    DnVecDescr* Y = dnvec(vecY);

    Operands ops;
    if (flagsparseStatus_t s = resolve_operands(handle, alpha, beta, Y, computeType,
                                                &ops, A)) {
        return s;
    }
    if (ops.y_len == 0) return FLAGSPARSE_STATUS_SUCCESS;

    switch (A->format) {
        case FLAGSPARSE_FORMAT_COO:
            return run_coo(handle, A, X, Y, ops, externalBuffer,
                           alg == FLAGSPARSE_SPMV_COO_ALG2);
        case FLAGSPARSE_FORMAT_CSC: return run_csc(handle, opA, A, X, Y, ops);
        case FLAGSPARSE_FORMAT_BSR: return run_bsr(handle, opA, A, X, Y, ops);
        default:                    return run_csr(handle, A, X, Y, ops);
    }
}

}  // namespace

extern "C" {

flagsparseStatus_t flagsparseSpMV_bufferSize(
    flagsparseHandle_t handle, flagsparseOperation_t opA, const void* alpha,
    flagsparseConstSpMatDescr_t matA, flagsparseConstDnVecDescr_t vecX,
    const void* beta, flagsparseDnVecDescr_t vecY,
    flagsparseDataType_t computeType, flagsparseSpMVAlg_t alg, size_t* bufferSize) {
    (void)alpha; (void)beta;
    if (bufferSize == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *bufferSize = 0;
    return guard(handle, [&]() -> flagsparseStatus_t {
        if (flagsparseStatus_t s = validate(handle, opA, matA, vecX, vecY, computeType, alg)) {
            return s;
        }
        // CSR and CSC need no scratch: their offsets array already describes the
        // segments. COO has to be given one -- rows + 1 int32 for the row-offsets
        // array its segment route runs on.
        const SpMatDescr* A = spmat(matA);
        if (A->format == FLAGSPARSE_FORMAT_COO) {
            *bufferSize = static_cast<size_t>(A->rows + 1) * sizeof(std::int32_t);
        }
        return FLAGSPARSE_STATUS_SUCCESS;
    });
}

flagsparseStatus_t flagsparseSpMV_preprocess(
    flagsparseHandle_t handle, flagsparseOperation_t opA, const void* alpha,
    flagsparseConstSpMatDescr_t matA, flagsparseConstDnVecDescr_t vecX,
    const void* beta, flagsparseDnVecDescr_t vecY,
    flagsparseDataType_t computeType, flagsparseSpMVAlg_t alg, void* externalBuffer) {
    (void)alpha; (void)beta;
    return guard(handle, [&]() -> flagsparseStatus_t {
        if (flagsparseStatus_t s = validate(handle, opA, matA, vecX, vecY, computeType, alg)) {
            return s;
        }
        // Pay the index readback here so the timed solve does not. CSR and CSC
        // get their segment count from it; COO gets the row-offsets array it
        // cannot run without, and an unsorted COO is rejected right here.
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

flagsparseStatus_t flagsparseSpMV(
    flagsparseHandle_t handle, flagsparseOperation_t opA, const void* alpha,
    flagsparseConstSpMatDescr_t matA, flagsparseConstDnVecDescr_t vecX,
    const void* beta, flagsparseDnVecDescr_t vecY,
    flagsparseDataType_t computeType, flagsparseSpMVAlg_t alg, void* externalBuffer) {
    return guard(handle, [&]() -> flagsparseStatus_t {
        if (flagsparseStatus_t s = validate(handle, opA, matA, vecX, vecY, computeType, alg)) {
            return s;
        }
        return run(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg,
                   externalBuffer);
    });
}

}  // extern "C"
