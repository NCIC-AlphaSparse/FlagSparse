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
#include <cstring>
#include <string>
#include <vector>

#include "adaptor/adaptor.hpp"
#include "core/internal.hpp"
#include "core/jit.hpp"
#include "core/prologue.hpp"
#include "core/sparse_gather.hpp"

using namespace flagsparse;

namespace {

constexpr int kCsrBlockNnz = 128;
constexpr int kCsrNumWarps = 4;
constexpr int kCsrNumStages = 2;
constexpr int kCsrShortRowLimit = 256;
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

bool musa_backend() {
    return std::string(adaptor::backend_name()) == "musa";
}

// Cached: SpMV is short enough that a string compare per call shows up.
bool cuda_backend() {
    static const bool cuda = std::string(adaptor::backend_name()) == "cuda";
    return cuda;
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
        case FLAGSPARSE_R_16F: *re = fp16_to_double(static_cast<const uint16_t*>(p)[0]); return FLAGSPARSE_STATUS_SUCCESS;
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
    bool acc_is_fp16 = false;
    bool has_beta = false;
    double alpha_re = 1.0, alpha_im = 0.0, beta_re = 0.0, beta_im = 0.0;
    int64_t y_len = 0;
    const char* vt = "";
    const char* it = "";
    const char* ot = "";
};

// alpha/beta's Triton-side type is ops.vt (matching the operand dtype, see the
// signature strings below), not a fixed fp32 -- so each width needs its own
// Arg. Missing the fp16 case here silently truncated the scalar (found via
// sddmm.cpp's identical bug: a 4-byte Arg::f where the kernel expected 2
// bytes reads only the low 16 bits of a float's bit pattern, e.g. alpha=1.0f
// decodes as fp16 bits 0x0000 = 0.0).
void push_scalar(const Operands& ops, double v, std::vector<jit::Arg>* args) {
    if (ops.acc_is_fp64) args->push_back(jit::Arg::d(v));
    else if (ops.acc_is_fp16) args->push_back(jit::Arg::h(double_to_fp16(v)));
    else args->push_back(jit::Arg::f(static_cast<float>(v)));
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
        case FLAGSPARSE_FORMAT_SLICED_ELL:
            return alg == FLAGSPARSE_SPMV_SELL_ALG1;
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
    if (component != FLAGSPARSE_R_32F && component != FLAGSPARSE_R_64F &&
        component != FLAGSPARSE_R_16F) {
        // bf16 would need a compute-precision copy of the operands. f16 does
        // not: _spmv_csr_real_kernel (and the COO/CSC/BSR real kernels) are
        // the exact same kernels flagsparse_spmv_csr's plain (non-mixed)
        // Python path already runs directly on fp16 storage -- no ACC_DTYPE
        // constexpr, no upcast copy, same as fp32/fp64.
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (triton_index_dtype(A->indices_type)[0] == '\0' ||
        triton_index_dtype(A->offsets_type)[0] == '\0') {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }

    switch (A->format) {
        case FLAGSPARSE_FORMAT_CSR:
            if (transposes(opA) && computeType != FLAGSPARSE_R_32F &&
                computeType != FLAGSPARSE_C_32F) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
            break;
        case FLAGSPARSE_FORMAT_COO:
            if (transposes(opA) && computeType != FLAGSPARSE_R_32F &&
                computeType != FLAGSPARSE_C_32F) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
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
        case FLAGSPARSE_FORMAT_SLICED_ELL:
            if (transposes(opA)) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
            if (A->slice_size <= 0 || A->slice_size > 1024) {
                return FLAGSPARSE_STATUS_NOT_SUPPORTED;
            }
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
    out->acc_is_fp16 = (component == FLAGSPARSE_R_16F);
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
    // The real and complex kernels both carry XPU_COMPAT. The C API does not
    // target XPU, so the compile-time branch is always disabled.
    sig += ",False";

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

// Several rows per program, LANES_PER_ROW nonzeros of each row per step: one
// program per row (launch_csr_rowpar) pays a 128-lane program for rows that
// average 3-25 nonzeros on the delivery matrices, and its time grows with the
// row count, not nnz -- about 0.46 ns a row on an RTX 5090, 0.02-0.4x of
// cuSPARSE (2026-11-02). The tile shapes below are measured, per backend.
struct RowTile { int rows_per_program, lanes_per_row, num_warps; };

RowTile row_tile_shape(const Operands& ops, double avg_row_nnz, int64_t max_row_nnz) {
    if (cuda_backend()) {
        // RTX 5090, the 10 delivery matrices, swept over R x V x warps. Uniformly
        // short rows (roadNet-TX: avg 2.8, max 12) want 4 lanes; a short average
        // with a long tail (ASIC_680ks: avg 3.4, max 210) still wants 8.
        const bool uniform_short = avg_row_nnz <= 4.0 && max_row_nnz <= 16;
        if (ops.acc_is_fp64) return avg_row_nnz <= 7.0 ? RowTile{32, 4, 2} : RowTile{8, 8, 2};
        return uniform_short ? RowTile{64, 4, 2} : RowTile{32, 8, 4};
    }
    // Tuned on Moore Threads MUSA.
    const bool long_tail = max_row_nnz > 64;
    const bool ultra_short = !long_tail && avg_row_nnz <= 2.0;
    const bool short_rows = avg_row_nnz <= 8.0;
    return {ultra_short ? 128 : (short_rows ? 64 : 32), ultra_short ? 4 : (short_rows ? 8 : 16), 4};
}

// Whether the real row tile takes this matrix: MUSA for every short-row matrix,
// CUDA only inside the average row length it was measured on.
bool use_real_row_tile(const Operands& ops, double avg_row_nnz, int64_t max_row_nnz) {
    if (ops.complex_op || max_row_nnz > kCsrShortRowLimit) return false;
    if (musa_backend()) return true;
    return cuda_backend() && avg_row_nnz <= 32.0;
}

flagsparseStatus_t launch_csr_row_tile(flagsparseHandle_t handle, SpMatDescr* A,
                                       void* offsets, const char* offsets_type,
                                       const DnVecDescr* X, DnVecDescr* Y,
                                       const Operands& ops, const RowTile& tile) {
    std::string sig;
    sig.reserve(160);
    sig += "*"; sig += ops.vt; sig += ":16,";
    sig += "*"; sig += ops.it; sig += ":16,";
    sig += "*"; sig += offsets_type; sig += ":16,";
    sig += "*"; sig += ops.vt; sig += ":16,";
    sig += "*"; sig += ops.vt; sig += ":16,";
    sig += ops.vt; sig += ",";
    sig += ops.vt; sig += ",i32,";
    sig += std::to_string(tile.rows_per_program) + ",";
    sig += std::to_string(tile.lanes_per_row) + ",";
    sig += ops.acc_is_fp64 ? "True," : "False,";
    sig += ops.has_beta ? "True," : "False,";
    const bool wide = A->rows > INT32_MAX - 128 || A->nnz > INT32_MAX - 128 ||
                      A->cols > INT32_MAX - 128;
    sig += wide ? "True" : "False";
    std::vector<jit::Arg> args;
    args.reserve(8);
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->indices)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(offsets)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(X->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(Y->values)));
    push_scalar(ops, ops.alpha_re, &args);
    push_scalar(ops, ops.beta_re, &args);
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(A->rows)));
    std::string err;
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("spmv_csr.py"), "spmv_csr_row_tile_real", sig,
        ctx(handle)->stream, (A->rows + tile.rows_per_program - 1) / tile.rows_per_program,
        1, 1, tile.num_warps, /*num_stages=*/2, args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}

// complex64 with alpha = 1, beta = 0 on CUDA: the Python package's own
// _spmv_csr_complex_row_tile_kernel, with its own row-length table
// (spmv_csr.py's complex route). That kernel has no alpha/beta, so any other
// scalars stay on the row-parallel kernel.
bool use_complex_row_tile(const Operands& ops, double avg_row_nnz, int64_t max_row_nnz) {
    return cuda_backend() && ops.complex_op && !ops.acc_is_fp64 && !ops.has_beta &&
           ops.alpha_re == 1.0 && ops.alpha_im == 0.0 && avg_row_nnz <= 32.0 &&
           max_row_nnz <= 512;
}

flagsparseStatus_t launch_csr_complex_row_tile(flagsparseHandle_t handle, SpMatDescr* A,
                                               void* offsets, const char* offsets_type,
                                               const DnVecDescr* X, DnVecDescr* Y,
                                               const Operands& ops, double avg_row_nnz) {
    int rows_per = 4, block_nnz = 16;
    if (avg_row_nnz <= 4.0) { rows_per = 32; block_nnz = 4; }
    else if (avg_row_nnz <= 8.0) { rows_per = 8; block_nnz = 8; }
    else if (avg_row_nnz <= 16.0) { rows_per = 8; block_nnz = 16; }
    std::string sig;
    sig.reserve(96);
    sig += "*fp32:16,";
    sig += "*"; sig += ops.it; sig += ":16,";
    sig += "*"; sig += offsets_type; sig += ":16,";
    sig += "*fp32:16,*fp32:16,i32,";
    sig += std::to_string(rows_per) + "," + std::to_string(block_nnz);
    std::vector<jit::Arg> args;
    args.reserve(6);
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->indices)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(offsets)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(X->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(Y->values)));
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(A->rows)));
    std::string err;
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("spmv_csr.py"), "_spmv_csr_complex_row_tile_kernel", sig,
        ctx(handle)->stream, (A->rows + rows_per - 1) / rows_per, 1, 1,
        /*num_warps=*/1, /*num_stages=*/2, args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}

// The row routes over any row-offsets array: A's own (CSR) or the one COO
// builds in its externalBuffer. False when neither row tile applies.
bool try_row_tile(flagsparseHandle_t handle, SpMatDescr* A, void* offsets,
                  const char* offsets_type, int64_t max_row_nnz, const DnVecDescr* X,
                  DnVecDescr* Y, const Operands& ops, flagsparseStatus_t* st) {
    const double avg = A->rows > 0 ? static_cast<double>(A->nnz) / A->rows : 0.0;
    if (use_real_row_tile(ops, avg, max_row_nnz)) {
        *st = launch_csr_row_tile(handle, A, offsets, offsets_type, X, Y, ops,
                                  row_tile_shape(ops, avg, max_row_nnz));
        return true;
    }
    if (use_complex_row_tile(ops, avg, max_row_nnz)) {
        *st = launch_csr_complex_row_tile(handle, A, offsets, offsets_type, X, Y, ops, avg);
        return true;
    }
    return false;
}

// ------------------------------------------------------- mixed-precision CSR
//
// int8 matrix/x -> int32 or float32 y; float16/bfloat16 matrix/x -> float32 y
// (cuSPARSE 12.5's SpMV type table, mirrored from
// src/flagsparse/sparse_operations/mixed_spmx.py's _resolve_types). A
// self-contained path, not routed through Operands/launch_csr_rowpar: those
// assume A/X/Y share one dtype, which is exactly the case this doesn't hold.
// op(A) = A^T is out of scope here the same way it already is for the
// non-mixed CSR path (see validate() above); alpha/beta are out of scope too
// -- _csr_spmv_mixed_kernel has no such terms, matching mixed_spmx.py's own
// spmv_csr_mixed, which takes no alpha/beta either.

bool mixed_narrow_wide_pair(flagsparseDataType_t data_type, flagsparseDataType_t y_type,
                    bool* acc_is_i32) {
    if (data_type == FLAGSPARSE_R_8I) {
        if (y_type == FLAGSPARSE_R_32I) { *acc_is_i32 = true; return true; }
        if (y_type == FLAGSPARSE_R_32F) { *acc_is_i32 = false; return true; }
        return false;
    }
    if (data_type == FLAGSPARSE_R_16F) {
        *acc_is_i32 = false;
        return y_type == FLAGSPARSE_R_32F;
    }
    return false;
}

// Ported from mixed_spmx.py's _csr_row_tile: the mean row length picks a tile
// that keeps ROWS * BLOCK around 512 lanes.
void csr_mixed_row_tile(int64_t nnz, int64_t n_rows, int* rows_per, int* block) {
    const double mean = n_rows > 0 ? static_cast<double>(nnz) / static_cast<double>(n_rows) : 0.0;
    if (mean <= 4.0)       { *rows_per = 128; *block = 4; }
    else if (mean <= 12.0) { *rows_per = 64;  *block = 8; }
    else if (mean <= 24.0) { *rows_per = 32;  *block = 16; }
    else                   { *rows_per = 16;  *block = 32; }
}

flagsparseStatus_t run_csr_mixed(flagsparseHandle_t handle, const SpMatDescr* A,
                                 const DnVecDescr* X, DnVecDescr* Y, bool acc_is_i32) {
    if (A->nnz > static_cast<int64_t>(INT32_MAX)) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    const char* it = triton_index_dtype(A->indices_type);
    const char* ot = triton_index_dtype(A->offsets_type);
    if (it[0] == '\0' || ot[0] == '\0') return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    const char* xt = triton_dtype(A->value_type);
    const char* yt = triton_dtype(Y->value_type);
    if (xt[0] == '\0' || yt[0] == '\0') return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    if (A->rows == 0) return FLAGSPARSE_STATUS_SUCCESS;

    int rows_per = 128, block = 4;
    csr_mixed_row_tile(A->nnz, A->rows, &rows_per, &block);

    std::string sig;
    sig.reserve(128);
    sig += "*"; sig += yt; sig += ":16,";                 // y
    sig += "*"; sig += xt; sig += ":16,";                 // data
    sig += "*"; sig += it; sig += ":16,";                 // cols
    sig += "*"; sig += ot; sig += ":16,";                 // indptr
    sig += "*"; sig += xt; sig += ":16,";                 // x
    sig += "i32,";                                        // n_rows
    sig += std::to_string(rows_per) + ",";
    sig += std::to_string(block);

    std::vector<jit::Arg> args;
    args.reserve(6);
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(Y->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->indices)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->offsets)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(X->values)));
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(A->rows)));

    const int64_t grid = (A->rows + rows_per - 1) / rows_per;
    std::string err;
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("mixed_spmx.py"),
        acc_is_i32 ? "csr_spmv_mixed_i32acc" : "csr_spmv_mixed_f32acc", sig,
        ctx(handle)->stream, grid, 1, 1, /*num_warps=*/4, /*num_stages=*/2, args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}

flagsparseStatus_t run_csr_real_by_complex(flagsparseHandle_t handle, const SpMatDescr* A,
                                           const DnVecDescr* X, DnVecDescr* Y) {
    const char* it = triton_index_dtype(A->indices_type);
    const char* ot = triton_index_dtype(A->offsets_type);
    if (it[0] == '\0' || ot[0] == '\0') return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    if (A->rows == 0) return FLAGSPARSE_STATUS_SUCCESS;

    int rows_per = 128, block = 4;
    csr_mixed_row_tile(A->nnz, A->rows, &rows_per, &block);

    // X/Y are complex64 descriptors, but the Triton kernel addresses their
    // interleaved real/imag components as fp32 values.
    std::string sig;
    sig.reserve(128);
    sig += "*fp32:16,*fp32:16,*";
    sig += it; sig += ":16,*";
    sig += ot; sig += ":16,*fp32:16,i32,";
    sig += std::to_string(rows_per) + "," + std::to_string(block);

    std::vector<jit::Arg> args;
    args.reserve(6);
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(Y->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->indices)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->offsets)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(X->values)));
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(A->rows)));

    const int64_t grid = (A->rows + rows_per - 1) / rows_per;
    std::string err;
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("mixed_spmx.py"), "_csr_spmv_real_by_complex_kernel", sig,
        ctx(handle)->stream, grid, 1, 1, /*num_warps=*/4, /*num_stages=*/2, args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}

// alpha=1, beta=0 exactly (component-wise): the mixed kernel has no scale
// terms, so anything else is a caller asking for something this path cannot
// give them, not a silent approximation.
bool is_identity_alpha_beta(flagsparseHandle_t handle, flagsparseDataType_t computeType,
                            const void* alpha, const void* beta) {
    if (ctx(handle)->pointer_mode != FLAGSPARSE_POINTER_MODE_HOST) return false;
    if (dtype_size(computeType) == 0) return false;
    std::vector<unsigned char> one(dtype_size(computeType), 0), zero(dtype_size(computeType), 0);
    if (computeType == FLAGSPARSE_R_32I) {
        std::int32_t v = 1; std::memcpy(one.data(), &v, sizeof(v));
    } else if (computeType == FLAGSPARSE_R_32F) {
        float v = 1.0f; std::memcpy(one.data(), &v, sizeof(v));
    } else if (computeType == FLAGSPARSE_C_32F) {
        float v = 1.0f; std::memcpy(one.data(), &v, sizeof(v));
    } else if (computeType == FLAGSPARSE_R_16F) {
        const std::uint16_t v = double_to_fp16(1.0); std::memcpy(one.data(), &v, sizeof(v));
    } else if (computeType == FLAGSPARSE_R_16BF) {
        const std::uint16_t v = 0x3F80;  // bfloat16 1.0
        std::memcpy(one.data(), &v, sizeof(v));
    } else {
        return false;
    }
    return alpha != nullptr && beta != nullptr &&
           std::memcmp(alpha, one.data(), one.size()) == 0 &&
           std::memcmp(beta, zero.data(), zero.size()) == 0;
}

// Whether (opA, matA, vecX, vecY) names one of the mixed-CSR combinations
// above -- checked before the general validate(), which requires A/X/Y to
// share computeType and would reject this on sight.
bool is_mixed_csr_request(flagsparseOperation_t opA, flagsparseConstSpMatDescr_t matA,
                          flagsparseConstDnVecDescr_t vecX, flagsparseConstDnVecDescr_t vecY,
                          bool* acc_is_i32) {
    if (matA == nullptr || vecX == nullptr || vecY == nullptr) return false;
    const SpMatDescr* A = spmat(matA);
    if (A->format != FLAGSPARSE_FORMAT_CSR || transposes(opA)) return false;
    const DnVecDescr* X = dnvec(vecX);
    const DnVecDescr* Y = dnvec(vecY);
    if (X->value_type != A->value_type) return false;
    // bfloat16 has no plain (non-mixed) CSR kernel, so bf16 -> bf16 rides the
    // float32-accumulating row tile too; its store rounds back to bf16. CSR only:
    // the COO mixed route accumulates with atomics straight into y.
    if (A->value_type == FLAGSPARSE_R_16BF) {
        *acc_is_i32 = false;
        return Y->value_type == FLAGSPARSE_R_32F || Y->value_type == FLAGSPARSE_R_16BF;
    }
    return mixed_narrow_wide_pair(A->value_type, Y->value_type, acc_is_i32);
}

bool is_real_by_complex_csr_request(flagsparseOperation_t opA,
                                    flagsparseConstSpMatDescr_t matA,
                                    flagsparseConstDnVecDescr_t vecX,
                                    flagsparseConstDnVecDescr_t vecY) {
    if (matA == nullptr || vecX == nullptr || vecY == nullptr) return false;
    const SpMatDescr* A = spmat(matA);
    const DnVecDescr* X = dnvec(vecX);
    const DnVecDescr* Y = dnvec(vecY);
    return A->format == FLAGSPARSE_FORMAT_CSR && !transposes(opA) &&
           A->value_type == FLAGSPARSE_R_32F && X->value_type == FLAGSPARSE_C_32F &&
           Y->value_type == FLAGSPARSE_C_32F;
}

flagsparseStatus_t validate_real_by_complex_csr(flagsparseHandle_t handle,
                                                flagsparseOperation_t opA,
                                                flagsparseConstSpMatDescr_t matA,
                                                flagsparseConstDnVecDescr_t vecX,
                                                flagsparseConstDnVecDescr_t vecY,
                                                flagsparseDataType_t computeType,
                                                flagsparseSpMVAlg_t alg) {
    if (handle == nullptr) return FLAGSPARSE_STATUS_NOT_INITIALIZED;
    if (!is_real_by_complex_csr_request(opA, matA, vecX, vecY) ||
        computeType != FLAGSPARSE_C_32F) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    const SpMatDescr* A = spmat(matA);
    const DnVecDescr* X = dnvec(vecX);
    const DnVecDescr* Y = dnvec(vecY);
    if (A->idx_base != FLAGSPARSE_INDEX_BASE_ZERO || !alg_supported(A->format, alg) ||
        triton_index_dtype(A->indices_type)[0] == '\0' ||
        triton_index_dtype(A->offsets_type)[0] == '\0') {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (X->size != A->cols || Y->size != A->rows) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    return FLAGSPARSE_STATUS_SUCCESS;
}

bool is_mixed_sell_request(flagsparseOperation_t opA,
                           flagsparseConstSpMatDescr_t matA,
                           flagsparseConstDnVecDescr_t vecX,
                           flagsparseConstDnVecDescr_t vecY) {
    if (matA == nullptr || vecX == nullptr || vecY == nullptr) return false;
    const SpMatDescr* A = spmat(matA);
    const DnVecDescr* X = dnvec(vecX);
    const DnVecDescr* Y = dnvec(vecY);
    return A->format == FLAGSPARSE_FORMAT_SLICED_ELL && !transposes(opA) &&
           A->value_type == FLAGSPARSE_R_8I && X->value_type == FLAGSPARSE_R_8I &&
           Y->value_type == FLAGSPARSE_R_32I;
}

flagsparseStatus_t validate_mixed_sell(flagsparseHandle_t handle,
                                       flagsparseOperation_t opA,
                                       flagsparseConstSpMatDescr_t matA,
                                       flagsparseConstDnVecDescr_t vecX,
                                       flagsparseConstDnVecDescr_t vecY,
                                       flagsparseDataType_t computeType,
                                       flagsparseSpMVAlg_t alg) {
    if (handle == nullptr) return FLAGSPARSE_STATUS_NOT_INITIALIZED;
    if (!is_mixed_sell_request(opA, matA, vecX, vecY) ||
        computeType != FLAGSPARSE_R_32I) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    const SpMatDescr* A = spmat(matA);
    const DnVecDescr* X = dnvec(vecX);
    const DnVecDescr* Y = dnvec(vecY);
    if (A->idx_base != FLAGSPARSE_INDEX_BASE_ZERO || !alg_supported(A->format, alg) ||
        A->slice_size <= 0 || A->slice_size > 1024 ||
        triton_index_dtype(A->indices_type)[0] == '\0' ||
        triton_index_dtype(A->offsets_type)[0] == '\0') {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (A->rows > 0 && (A->offsets == nullptr || A->indices == nullptr ||
                        A->values == nullptr)) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    if (X->size != A->cols || Y->size != A->rows) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    return FLAGSPARSE_STATUS_SUCCESS;
}

int next_power_of_two(int64_t value) {
    int out = 1;
    while (out < value && out < 1024) out *= 2;
    return out;
}

flagsparseStatus_t run_mixed_sell(flagsparseHandle_t handle, const SpMatDescr* A,
                                  const DnVecDescr* X, DnVecDescr* Y) {
    if (A->rows == 0) return FLAGSPARSE_STATUS_SUCCESS;
    const char* it = triton_index_dtype(A->indices_type);
    const char* ot = triton_index_dtype(A->offsets_type);
    const bool parallel = std::string(adaptor::backend_name()) == "musa";
    const int block_r = parallel ? 32 : next_power_of_two(A->slice_size);
    const int64_t n_slices = (A->rows + A->slice_size - 1) / A->slice_size;

    std::string sig;
    sig.reserve(128);
    sig += "*i32:16,*i8:16,*";
    sig += it; sig += ":16,*";
    sig += ot; sig += ":16,*i8:16,i32,";
    sig += std::to_string(A->slice_size) + "," + std::to_string(block_r);

    std::vector<jit::Arg> args;
    args.reserve(6);
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(Y->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->indices)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->offsets)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(X->values)));
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(A->rows)));

    std::string err;
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("spmv_sell.py"),
        parallel ? "spmv_sell_musa_i32acc" : "spmv_sell_i32acc", sig,
        ctx(handle)->stream, parallel ? (A->rows + 31) / 32 : n_slices, 1, 1, parallel ? 4 : 1,
        parallel ? 1 : 2, args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}

flagsparseStatus_t run_sell(flagsparseHandle_t handle, const SpMatDescr* A,
                            const DnVecDescr* X, DnVecDescr* Y,
                            const Operands& ops) {
    if (A->rows == 0) return FLAGSPARSE_STATUS_SUCCESS;
    const char* it = triton_index_dtype(A->indices_type);
    const char* ot = triton_index_dtype(A->offsets_type);
    const bool parallel = std::string(adaptor::backend_name()) == "musa";
    const int block_r = parallel ? 32 : next_power_of_two(A->slice_size);
    const int64_t n_slices = (A->rows + A->slice_size - 1) / A->slice_size;
    const bool complex = ops.complex_op;

    std::string sig;
    sig.reserve(144);
    sig += "*"; sig += ops.vt; sig += ":16,";  // Y (interleaved for complex)
    sig += "*"; sig += ops.vt; sig += ":16,";  // values
    sig += "*"; sig += it; sig += ":16,";
    sig += "*"; sig += ot; sig += ":16,";
    sig += "*"; sig += ops.vt; sig += ":16,i32,";
    sig += std::to_string(A->slice_size) + "," + std::to_string(block_r);
    if (complex) sig += parallel ? ",4" : ",1";  // SLOTS or GROUP

    std::vector<jit::Arg> args;
    args.reserve(6);
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(Y->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->indices)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->offsets)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(X->values)));
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(A->rows)));

    std::string err;
    if (complex) {
        args[0] = jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(Y->values));
        args[1] = jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->values));
        args[4] = jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(X->values));
    }
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("spmv_sell.py"),
        complex ? (parallel ? "spmv_sell_musa_c32" : "_spmv_sell_complex_kernel")
                : (parallel ? "spmv_sell_musa_f32acc" : "spmv_sell_f32acc"), sig,
        ctx(handle)->stream, parallel ? (A->rows + 31) / 32 : n_slices, 1, 1, parallel ? 4 : 1,
        parallel ? 1 : 2, args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}

// ------------------------------------------------------- mixed-precision COO
//
// Same dtype-pair table as CSR, but the kernel is nnz-parallel atomic_add
// (mixed_spmx.py's _coo_spmv_atomic_kernel/spmv_coo_mixed) rather than
// row-tiled: COO's row array is already what the kernel needs, no
// row-offsets/seg_starts preprocessing (run_coo's externalBuffer) required.
// Also why this is the one the delivery list asks for float16 COO mixed at all: the plain
// (non-mixed) COO path has no half-precision kernel, per mixed_spmx.py's own
// "COO's native path has no half-precision kernel" comment on spmv_needs_mixed.

bool is_mixed_coo_request(flagsparseOperation_t opA, flagsparseConstSpMatDescr_t matA,
                          flagsparseConstDnVecDescr_t vecX, flagsparseConstDnVecDescr_t vecY,
                          bool* acc_is_i32) {
    if (matA == nullptr || vecX == nullptr || vecY == nullptr) return false;
    const SpMatDescr* A = spmat(matA);
    if (A->format != FLAGSPARSE_FORMAT_COO || transposes(opA)) return false;
    const DnVecDescr* X = dnvec(vecX);
    const DnVecDescr* Y = dnvec(vecY);
    if (X->value_type != A->value_type) return false;
    return mixed_narrow_wide_pair(A->value_type, Y->value_type, acc_is_i32);
}

flagsparseStatus_t run_coo_mixed(flagsparseHandle_t handle, const SpMatDescr* A,
                                 const DnVecDescr* X, DnVecDescr* Y, bool acc_is_i32) {
    if (A->nnz > static_cast<int64_t>(INT32_MAX)) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    const char* it = triton_index_dtype(A->indices_type);
    if (it[0] == '\0') return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    const char* xt = triton_dtype(A->value_type);
    const char* yt = triton_dtype(Y->value_type);
    if (xt[0] == '\0' || yt[0] == '\0') return FLAGSPARSE_STATUS_NOT_SUPPORTED;

    const std::size_t y_bytes = static_cast<std::size_t>(Y->size) * dtype_size(Y->value_type);
    if (adaptor::memset_device(reinterpret_cast<adaptor::DevicePtr>(Y->values), 0, y_bytes) !=
        FLAGSPARSE_STATUS_SUCCESS) {
        return FLAGSPARSE_STATUS_EXECUTION_FAILED;
    }
    if (A->nnz == 0) return FLAGSPARSE_STATUS_SUCCESS;

    constexpr int kBlock = 256;
    std::string sig;
    sig.reserve(128);
    sig += "*"; sig += yt; sig += ":16,";   // acc (Y, zeroed above)
    sig += "*"; sig += xt; sig += ":16,";   // data
    sig += "*"; sig += it; sig += ":16,";   // rows
    sig += "*"; sig += it; sig += ":16,";   // cols
    sig += "*"; sig += xt; sig += ":16,";   // x
    sig += "i32,";                          // nnz
    sig += std::to_string(kBlock);

    std::vector<jit::Arg> args;
    args.reserve(6);
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(Y->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->row_ind)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->indices)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(X->values)));
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(A->nnz)));

    const int64_t grid = (A->nnz + kBlock - 1) / kBlock;
    std::string err;
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("mixed_spmx.py"),
        acc_is_i32 ? "coo_spmv_atomic_i32acc" : "coo_spmv_atomic_f32acc", sig,
        ctx(handle)->stream, grid, 1, 1, /*num_warps=*/4, /*num_stages=*/2, args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}

// ------------------------------------------------ scatter: y[col] += v * x[row]
//
// The products that walk the stored entries in the "wrong" direction for a
// row-owned kernel: CSR/COO with op(A) = A^T, and CSC with op(A) = A (a CSC
// matrix is the CSR of A^T). mixed_spmx.py's _csr_spmv_trans_atomic_kernel
// covers CSR and CSC, _coo_spmv_atomic_kernel with its two index arrays swapped
// covers COO, and the real-by-complex pair covers a float32 A times a complex64
// x. Only the narrow or real-by-complex types come here -- float32/complex64
// transposes keep run_transpose_atomic. The accumulator is zeroed and summed
// atomically; a float16/bfloat16 result accumulates in a float32 scratch (from
// bufferSize) and is converted once at the end. alpha = 1, beta = 0 only, like
// every other mixed path in this file.

struct ScatterPlan {
    bool real_by_complex = false;
    bool acc_is_i32 = false;
    bool via_scratch = false;   // result narrower than the float32 accumulator
};

bool musa_csr_real_complex_gather(const SpMatDescr* A, const ScatterPlan& plan) {
    return plan.real_by_complex && A->format == FLAGSPARSE_FORMAT_CSR &&
        std::strcmp(adaptor::backend_name(), "musa") == 0;
}

flagsparseStatus_t run_musa_csr_real_complex_gather(flagsparseHandle_t handle, SpMatDescr* A,
    const DnVecDescr* X, DnVecDescr* Y, void* buffer) {
    if (Y->size == 0) return FLAGSPARSE_STATUS_SUCCESS;
    if (!buffer) {
        ctx(handle)->last_error =
            "MUSA real-by-complex CSR transpose requires the topology externalBuffer "
            "reported by flagsparseSpMV_bufferSize";
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    if (A->sparse_gather_buffer != buffer) {
        if (auto s = prepare_sparse_gather(A, buffer)) return s;
    }
    const double average = static_cast<double>(A->nnz) / std::max<int64_t>(Y->size, 1);
    const int rows = average <= 4. ? 128 : 16;
    const int lanes = average <= 4. ? 4 : 8;
    const size_t ib = gather_index_bytes(A);
    auto* topology = static_cast<unsigned char*>(buffer);
    void* order = topology + gather_ptr_count(A) * ib;
    void* source_rows = topology + (gather_ptr_count(A) + gather_nnz_count(A)) * ib;
    const char* index_type = gather_i32(A) ? "i32" : "i64";
    // No alignment hints: user-owned values, vectors and scratch may be views.
    std::string sig = std::string("*fp32,*") + index_type + ",*" + index_type + ",*" +
        index_type + ",*fp32,*fp32,i64," + std::to_string(rows) + "," +
        std::to_string(lanes) + "," + (gather_i32(A) ? "False" : "True");
    std::vector<jit::Arg> args;
    for (void* p : {A->values, buffer, order, source_rows, X->values, Y->values})
        args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(p)));
    args.push_back(jit::Arg::i64v(Y->size));
    std::string err;
    auto s = jit::launch(jit::codegen_module("sparse_gather.py"),
        "musa_csr_spmv_trans_real_complex_f64", sig, ctx(handle)->stream,
        (Y->size + rows - 1) / rows, 1, 1, 4, 1, args, &err);
    if (s != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return s;
}

bool scatter_types(flagsparseDataType_t a, flagsparseDataType_t x, flagsparseDataType_t y,
                   ScatterPlan* plan) {
    if (a == FLAGSPARSE_R_32F && x == FLAGSPARSE_C_32F && y == FLAGSPARSE_C_32F) {
        plan->real_by_complex = true;
        return true;
    }
    if (x != a) return false;
    if (a == FLAGSPARSE_R_8I) {
        if (y != FLAGSPARSE_R_32I && y != FLAGSPARSE_R_32F) return false;
        plan->acc_is_i32 = (y == FLAGSPARSE_R_32I);
        return true;
    }
    if (a == FLAGSPARSE_R_16F || a == FLAGSPARSE_R_16BF) {
        if (y == a) { plan->via_scratch = true; return true; }
        return y == FLAGSPARSE_R_32F;
    }
    return false;
}

bool is_scatter_request(flagsparseOperation_t opA, flagsparseConstSpMatDescr_t matA,
                        flagsparseConstDnVecDescr_t vecX, flagsparseConstDnVecDescr_t vecY,
                        ScatterPlan* plan) {
    if (matA == nullptr || vecX == nullptr || vecY == nullptr) return false;
    const SpMatDescr* A = spmat(matA);
    const DnVecDescr* X = dnvec(vecX);
    const DnVecDescr* Y = dnvec(vecY);
    ScatterPlan p;
    if (!scatter_types(A->value_type, X->value_type, Y->value_type, &p)) return false;
    // conj(A)^T of a real matrix is A^T; a complex A never reaches here.
    const bool trans = transposes(opA);
    bool ok = false;
    switch (A->format) {
        case FLAGSPARSE_FORMAT_CSR: ok = trans; break;
        // COO's non-transposed narrow products already have run_coo_mixed (and
        // float16 -> float16 the plain kernel); real-by-complex has no other route.
        case FLAGSPARSE_FORMAT_COO: ok = trans || p.real_by_complex; break;
        // float16 -> float16 CSC has the plain kernel.
        case FLAGSPARSE_FORMAT_CSC:
            ok = !trans && !p.real_by_complex && !p.via_scratch; break;
        default: break;
    }
    if (ok) *plan = p;
    return ok;
}

int64_t scatter_output_len(const SpMatDescr* A, flagsparseOperation_t opA) {
    return transposes(opA) ? A->cols : A->rows;
}

flagsparseStatus_t validate_scatter(flagsparseHandle_t handle, flagsparseOperation_t opA,
                                    flagsparseConstSpMatDescr_t matA,
                                    flagsparseConstDnVecDescr_t vecX,
                                    flagsparseConstDnVecDescr_t vecY) {
    if (handle == nullptr) return FLAGSPARSE_STATUS_NOT_INITIALIZED;
    const SpMatDescr* A = spmat(matA);
    const DnVecDescr* X = dnvec(vecX);
    const DnVecDescr* Y = dnvec(vecY);
    if (A->idx_base != FLAGSPARSE_INDEX_BASE_ZERO || A->nnz > static_cast<int64_t>(INT32_MAX) ||
        triton_index_dtype(A->indices_type)[0] == '\0' ||
        (A->format != FLAGSPARSE_FORMAT_COO && triton_index_dtype(A->offsets_type)[0] == '\0')) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    const int64_t n_in = transposes(opA) ? A->rows : A->cols;
    if (X->size != n_in || Y->size != scatter_output_len(A, opA)) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t run_scatter(flagsparseHandle_t handle, flagsparseOperation_t opA,
                               const SpMatDescr* A, const DnVecDescr* X, DnVecDescr* Y,
                               const ScatterPlan& plan, void* scratch) {
    const int64_t y_len = Y->size;
    if (y_len == 0) return FLAGSPARSE_STATUS_SUCCESS;
    if (plan.via_scratch && scratch == nullptr) {
        ctx(handle)->last_error =
            "a float16/bfloat16 SpMV of this shape accumulates in the float32 "
            "externalBuffer reported by flagsparseSpMV_bufferSize";
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    const flagsparseDataType_t acc_type = plan.real_by_complex ? FLAGSPARSE_C_32F
        : (plan.acc_is_i32 ? FLAGSPARSE_R_32I : FLAGSPARSE_R_32F);
    void* acc = plan.via_scratch ? scratch : Y->values;
    if (adaptor::memset_device(reinterpret_cast<adaptor::DevicePtr>(acc), 0,
                               static_cast<std::size_t>(y_len) * dtype_size(acc_type)) !=
        FLAGSPARSE_STATUS_SUCCESS) {
        return FLAGSPARSE_STATUS_EXECUTION_FAILED;
    }
    const char* it = triton_index_dtype(A->indices_type);
    const char* xt = plan.real_by_complex ? "fp32" : triton_dtype(A->value_type);
    const char* at = plan.real_by_complex ? "fp32" : triton_dtype(acc_type);
    std::string err;
    flagsparseStatus_t st = FLAGSPARSE_STATUS_SUCCESS;
    if (A->nnz > 0 && A->format == FLAGSPARSE_FORMAT_COO) {
        // A^T swaps which index array names the output.
        const bool trans = transposes(opA);
        void* out_idx = trans ? A->indices : A->row_ind;
        void* in_idx = trans ? A->row_ind : A->indices;
        constexpr int kBlock = 256;
        std::string sig = std::string("*") + at + ":16,*" + (plan.real_by_complex ? "fp32" : xt) +
                          ":16,*" + it + ":16,*" + it + ":16,*" + xt + ":16,i32," +
                          std::to_string(kBlock);
        std::vector<jit::Arg> args;
        args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(acc)));
        args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->values)));
        args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(out_idx)));
        args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(in_idx)));
        args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(X->values)));
        args.push_back(jit::Arg::i(static_cast<std::int32_t>(A->nnz)));
        const char* kernel = plan.real_by_complex ? "_coo_spmv_real_by_complex_atomic_kernel"
            : (plan.acc_is_i32 ? "coo_spmv_atomic_i32acc" : "coo_spmv_atomic_f32acc");
        st = jit::launch(jit::codegen_module("mixed_spmx.py"), kernel, sig, ctx(handle)->stream,
                         (A->nnz + kBlock - 1) / kBlock, 1, 1, /*num_warps=*/4,
                         /*num_stages=*/2, args, &err);
    } else if (A->nnz > 0) {
        // CSR (op = A^T) and CSC (op = A) both walk "rows" of the stored pointer
        // array and scatter into the indices it lists.
        const char* ot = triton_index_dtype(A->offsets_type);
        const int64_t n_major = A->format == FLAGSPARSE_FORMAT_CSC ? A->cols : A->rows;
        int rows_per = 128, block = 4;
        csr_mixed_row_tile(A->nnz, n_major, &rows_per, &block);
        std::string sig = std::string("*") + at + ":16,*" + xt + ":16,*" + it + ":16,*" + ot +
                          ":16,*" + xt + ":16,i32," + std::to_string(rows_per) + "," +
                          std::to_string(block);
        std::vector<jit::Arg> args;
        args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(acc)));
        args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->values)));
        args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->indices)));
        args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->offsets)));
        args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(X->values)));
        args.push_back(jit::Arg::i(static_cast<std::int32_t>(n_major)));
        const char* kernel = plan.real_by_complex ? "_csr_spmv_trans_real_by_complex_kernel"
            : (plan.acc_is_i32 ? "csr_spmv_trans_i32acc" : "csr_spmv_trans_f32acc");
        st = jit::launch(jit::codegen_module("mixed_spmx.py"), kernel, sig, ctx(handle)->stream,
                         (n_major + rows_per - 1) / rows_per, 1, 1, /*num_warps=*/4,
                         /*num_stages=*/2, args, &err);
    }
    if (st != FLAGSPARSE_STATUS_SUCCESS) {
        ctx(handle)->last_error = err;
        return st;
    }
    if (plan.via_scratch) {
        return copy_dense_cast(handle, scratch, FLAGSPARSE_R_32F, Y->values, Y->value_type,
                               false, 1, y_len, 0, 1, 0, 1);
    }
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t run_csr(flagsparseHandle_t handle, SpMatDescr* A,
                           const DnVecDescr* X, DnVecDescr* Y, const Operands& ops) {
    int64_t max_row_nnz = 0;
    if (flagsparseStatus_t s = ensure_max_row_nnz(A, &max_row_nnz)) return s;
    // The row tiles are measured on MUSA and CUDA only. Other backends keep the
    // portable row-parallel route so their launch configuration does not shift.
    flagsparseStatus_t st = FLAGSPARSE_STATUS_SUCCESS;
    if (try_row_tile(handle, A, A->offsets, ops.ot, max_row_nnz, X, Y, ops, &st)) return st;
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
    // CUDA's row tiles run over the same offsets, for either COO algorithm: the
    // segment kernel below is one program per row too. (MUSA keeps its routes.)
    flagsparseStatus_t tile_st = FLAGSPARSE_STATUS_SUCCESS;
    if (cuda_backend() &&
        try_row_tile(handle, A, externalBuffer, "i32",
                     A->max_row_nnz < 0 ? 0 : A->max_row_nnz, X, Y, ops, &tile_st)) {
        return tile_st;
    }
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
    sig += ops.has_beta ? "True" : "False";    // HAS_BETA
    if (!ops.complex_op) sig += ",False";       // USE_MASKED_SELECT

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

// CSR/COO transpose cannot use the one-program-per-output row kernels: their
// storage is indexed by source rows.  Scatter each contribution to its column
// after the shared prologue has applied beta.  The delivery scope is fp32 transpose
// and complex32 conjugate-transpose, which are the two atomic types supported
// by these kernels.
flagsparseStatus_t run_transpose_atomic(flagsparseHandle_t handle,
                                        flagsparseOperation_t opA, SpMatDescr* A,
                                        const DnVecDescr* X, DnVecDescr* Y,
                                        const Operands& ops) {
    const int64_t unit = ops.complex_op ? 2 : 1;
    if (flagsparseStatus_t s = scale_dense(
            handle, Y->values, component_dtype(A->value_type), ops.complex_op,
            1, ops.y_len, 0, unit, ops.beta_re, ops.beta_im)) {
        return s;
    }
    if (A->nnz == 0) return FLAGSPARSE_STATUS_SUCCESS;

    constexpr int kBlock = 256;
    const bool csr = A->format == FLAGSPARSE_FORMAT_CSR;
    int64_t segments = 1;
    if (csr) {
        int64_t max_row_nnz = 0;
        if (flagsparseStatus_t s = ensure_max_row_nnz(A, &max_row_nnz)) return s;
        segments = std::max<int64_t>(1, (max_row_nnz + kBlock - 1) / kBlock);
    }
    const char* it = triton_index_dtype(A->indices_type);
    const char* ot = triton_index_dtype(A->offsets_type);
    if (it[0] == '\0' || (csr && ot[0] == '\0')) return FLAGSPARSE_STATUS_NOT_SUPPORTED;

    std::string sig;
    sig.reserve(176);
    sig += "*"; sig += ops.vt; sig += ":16,*";
    if (csr) { sig += it; sig += ":16,*"; sig += ot; sig += ":16,*"; }
    else { sig += it; sig += ":16,*"; sig += it; sig += ":16,*"; }
    sig += ops.vt; sig += ":16,*"; sig += ops.vt; sig += ":16,";
    sig += ops.vt; sig += ",";
    if (ops.complex_op) sig += ops.vt + std::string(",");
    sig += "i32," + std::to_string(kBlock);
    if (csr) sig += "," + std::to_string(segments);
    if (ops.complex_op) {
        sig += ",";
        sig += (opA == FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE) ? "True" : "False";
    }

    std::vector<jit::Arg> args;
    args.reserve(9);
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(A->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(
        csr ? A->indices : A->row_ind)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(
        csr ? A->offsets : A->indices)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(X->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(Y->values)));
    push_scalar(ops, ops.alpha_re, &args);
    if (ops.complex_op) push_scalar(ops, ops.alpha_im, &args);
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(csr ? A->rows : A->nnz)));

    const char* kernel = csr
        ? (ops.complex_op ? "spmv_csr_transpose_atomic_c32" : "spmv_csr_transpose_atomic_f32")
        : (ops.complex_op ? "spmv_coo_transpose_atomic_c32" : "spmv_coo_transpose_atomic_f32");
    const int64_t grid_x = csr ? A->rows : (A->nnz + kBlock - 1) / kBlock;
    std::string err;
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("spmv_transpose.py"), kernel, sig, ctx(handle)->stream,
        grid_x, csr ? segments : 1, 1, /*num_warps=*/4, kDefaultNumStages, args, &err);
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
    if (musa_gather_type(A)) {
        const bool indirect = (A->format == FLAGSPARSE_FORMAT_CSC && !transposes(opA)) ||
            (A->format == FLAGSPARSE_FORMAT_CSR && transposes(opA));
        const bool direct = A->format == FLAGSPARSE_FORMAT_CSR &&
            !transposes(opA) && ops.complex_op;
        const bool csc_transpose = A->format == FLAGSPARSE_FORMAT_CSC &&
            transposes(opA) && A->idx_base == FLAGSPARSE_INDEX_BASE_ZERO;
        const int unit = ops.complex_op ? 2 : 1;
        if ((indirect && externalBuffer) || direct || csc_transpose)
            return launch_sparse_gather(handle, A, X->values, Y->values,
                ops.y_len, 1, unit, unit, unit, unit,
                ops.alpha_re, ops.alpha_im, ops.beta_re, ops.beta_im,
                opA == FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE, externalBuffer, indirect);
    }

    switch (A->format) {
        case FLAGSPARSE_FORMAT_COO:
            if (transposes(opA)) return run_transpose_atomic(handle, opA, A, X, Y, ops);
            return run_coo(handle, A, X, Y, ops, externalBuffer,
                           alg == FLAGSPARSE_SPMV_COO_ALG2);
        case FLAGSPARSE_FORMAT_CSC: return run_csc(handle, opA, A, X, Y, ops);
        case FLAGSPARSE_FORMAT_BSR: return run_bsr(handle, opA, A, X, Y, ops);
        case FLAGSPARSE_FORMAT_SLICED_ELL: return run_sell(handle, A, X, Y, ops);
        default:
            return transposes(opA) ? run_transpose_atomic(handle, opA, A, X, Y, ops)
                                   : run_csr(handle, A, X, Y, ops);
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
        bool acc_is_i32 = false;
        ScatterPlan scatter;
        if (is_scatter_request(opA, matA, vecX, vecY, &scatter)) {
            if (flagsparseStatus_t s = validate_scatter(handle, opA, matA, vecX, vecY)) return s;
            if (musa_csr_real_complex_gather(spmat(matA), scatter)) {
                *bufferSize = gather_bytes(spmat(matA));
            } else if (scatter.via_scratch) {
                *bufferSize = static_cast<size_t>(dnvec(vecY)->size) * sizeof(float);
            }
            return FLAGSPARSE_STATUS_SUCCESS;
        }
        if (is_mixed_sell_request(opA, matA, vecX, vecY)) {
            return validate_mixed_sell(handle, opA, matA, vecX, vecY,
                                       computeType, alg);
        }
        if (is_real_by_complex_csr_request(opA, matA, vecX, vecY)) {
            return validate_real_by_complex_csr(handle, opA, matA, vecX, vecY,
                                                computeType, alg);
        }
        if (is_mixed_csr_request(opA, matA, vecX, vecY, &acc_is_i32) ||
            is_mixed_coo_request(opA, matA, vecX, vecY, &acc_is_i32)) {
            return FLAGSPARSE_STATUS_SUCCESS;  // no scratch: same as gather/scatter.
        }
        if (flagsparseStatus_t s = validate(handle, opA, matA, vecX, vecY, computeType, alg)) {
            return s;
        }
        // CSR and CSC need no scratch: their offsets array already describes the
        // segments. COO has to be given one -- rows + 1 int32 for the row-offsets
        // array its segment route runs on.
        const SpMatDescr* A = spmat(matA);
        if (musa_gather_type(A) &&
            ((A->format == FLAGSPARSE_FORMAT_CSC && !transposes(opA)) ||
             (A->format == FLAGSPARSE_FORMAT_CSR && transposes(opA)))) {
            *bufferSize = gather_bytes(A);
            return FLAGSPARSE_STATUS_SUCCESS;
        }
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
        bool acc_is_i32 = false;
        ScatterPlan scatter;
        if (is_scatter_request(opA, matA, vecX, vecY, &scatter)) {
            if (auto s = validate_scatter(handle, opA, matA, vecX, vecY)) return s;
            if (musa_csr_real_complex_gather(spmat(matA), scatter))
                return prepare_sparse_gather(const_cast<SpMatDescr*>(spmat(matA)), externalBuffer);
            return FLAGSPARSE_STATUS_SUCCESS;  // nothing to precompute
        }
        if (is_mixed_sell_request(opA, matA, vecX, vecY)) {
            return validate_mixed_sell(handle, opA, matA, vecX, vecY,
                                       computeType, alg);
        }
        if (is_real_by_complex_csr_request(opA, matA, vecX, vecY)) {
            return validate_real_by_complex_csr(handle, opA, matA, vecX, vecY,
                                                computeType, alg);
        }
        if (is_mixed_csr_request(opA, matA, vecX, vecY, &acc_is_i32) ||
            is_mixed_coo_request(opA, matA, vecX, vecY, &acc_is_i32)) {
            return FLAGSPARSE_STATUS_SUCCESS;  // row_tile/atomic needs no precomputed state.
        }
        if (flagsparseStatus_t s = validate(handle, opA, matA, vecX, vecY, computeType, alg)) {
            return s;
        }
        // Pay the index readback here so the timed solve does not. CSR and CSC
        // get their segment count from it; COO gets the row-offsets array it
        // cannot run without, and an unsorted COO is rejected right here.
        auto* A = const_cast<SpMatDescr*>(spmat(matA));
        if (musa_gather_type(A) && externalBuffer &&
            ((A->format == FLAGSPARSE_FORMAT_CSC && !transposes(opA)) ||
             (A->format == FLAGSPARSE_FORMAT_CSR && transposes(opA))))
            return prepare_sparse_gather(A, externalBuffer);
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
        bool acc_is_i32 = false;
        ScatterPlan scatter;
        if (is_scatter_request(opA, matA, vecX, vecY, &scatter)) {
            if (flagsparseStatus_t s = validate_scatter(handle, opA, matA, vecX, vecY)) return s;
            if (!is_identity_alpha_beta(handle, computeType, alpha, beta)) {
                return FLAGSPARSE_STATUS_NOT_SUPPORTED;
            }
            if (musa_csr_real_complex_gather(spmat(matA), scatter))
                return run_musa_csr_real_complex_gather(handle,
                    const_cast<SpMatDescr*>(spmat(matA)), dnvec(vecX), dnvec(vecY), externalBuffer);
            return run_scatter(handle, opA, spmat(matA), dnvec(vecX), dnvec(vecY), scatter,
                               externalBuffer);
        }
        if (is_mixed_sell_request(opA, matA, vecX, vecY)) {
            if (flagsparseStatus_t s = validate_mixed_sell(
                    handle, opA, matA, vecX, vecY, computeType, alg)) {
                return s;
            }
            if (!is_identity_alpha_beta(handle, computeType, alpha, beta)) {
                return FLAGSPARSE_STATUS_NOT_SUPPORTED;
            }
            return run_mixed_sell(handle, spmat(matA), dnvec(vecX), dnvec(vecY));
        }
        if (is_real_by_complex_csr_request(opA, matA, vecX, vecY)) {
            if (flagsparseStatus_t s = validate_real_by_complex_csr(
                    handle, opA, matA, vecX, vecY, computeType, alg)) {
                return s;
            }
            if (!is_identity_alpha_beta(handle, computeType, alpha, beta)) {
                return FLAGSPARSE_STATUS_NOT_SUPPORTED;
            }
            return run_csr_real_by_complex(handle, spmat(matA), dnvec(vecX), dnvec(vecY));
        }
        if (is_mixed_csr_request(opA, matA, vecX, vecY, &acc_is_i32)) {
            if (!is_identity_alpha_beta(handle, computeType, alpha, beta)) {
                return FLAGSPARSE_STATUS_NOT_SUPPORTED;
            }
            const SpMatDescr* A = spmat(matA);
            const DnVecDescr* X = dnvec(vecX);
            DnVecDescr* Y = dnvec(vecY);
            return run_csr_mixed(handle, A, X, Y, acc_is_i32);
        }
        if (is_mixed_coo_request(opA, matA, vecX, vecY, &acc_is_i32)) {
            if (!is_identity_alpha_beta(handle, computeType, alpha, beta)) {
                return FLAGSPARSE_STATUS_NOT_SUPPORTED;
            }
            const SpMatDescr* A = spmat(matA);
            const DnVecDescr* X = dnvec(vecX);
            DnVecDescr* Y = dnvec(vecY);
            return run_coo_mixed(handle, A, X, Y, acc_is_i32);
        }
        if (flagsparseStatus_t s = validate(handle, opA, matA, vecX, vecY, computeType, alg)) {
            return s;
        }
        return run(handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg,
                   externalBuffer);
    });
}

}  // extern "C"
