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

// muSPARSE baseline (MUSA SDK 4.3.5). Its generic descriptors have the same broad shape
// as cuSPARSE, but SpMM/SpSV/SpSM/SpGEMM use stage arguments rather than the
// cuSPARSE split functions.  This is intentionally a native implementation,
// not a prefix-table instantiation: treating the two APIs as call-for-call gave
// a buildable-looking but invalid baseline.

#include <musparse.h>

#include <algorithm>
#include <chrono>
#include <complex>
#include <exception>
#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "adaptor/adaptor.hpp"
#include "baseline/baseline.hpp"

namespace fstest::baseline {
namespace {

namespace ad = flagsparse::adaptor;

struct VendorHandle {
    musparseHandle_t h = nullptr;
    VendorHandle() { musparseCreate(&h); }
    // The benchmark process tears down the MUSA/Python runtime before C++ static
    // destruction.  Destroying this process-global handle afterwards faults in
    // SDK 4.3.5; leave its tiny host allocation to process exit instead.
    ~VendorHandle() = default;
};

musparseHandle_t vendor() {
    static VendorHandle g;
    return g.h;
}

std::string fail(const char* call, musparseStatus_t status) {
    return std::string("muSPARSE ") + call + " returned status " +
           std::to_string(static_cast<int>(status));
}

struct Scratch {
    void* p = nullptr;
    ~Scratch() { if (p) ad::device_free(reinterpret_cast<ad::DevicePtr>(p)); }
    void* release() { void* out = p; p = nullptr; return out; }
    bool grab(std::size_t bytes) {
        if (!bytes) return true;
        ad::DevicePtr d = 0;
        if (ad::device_malloc(&d, bytes) != FLAGSPARSE_STATUS_SUCCESS) return false;
        p = reinterpret_cast<void*>(d);
        return true;
    }
};

bool map_dtype(flagsparseDataType_t in, musaDataType_t* out) {
    switch (in) {
        case FLAGSPARSE_R_32F: *out = MUSA_R_32F; return true;
        case FLAGSPARSE_R_64F: *out = MUSA_R_64F; return true;
        case FLAGSPARSE_C_32F: *out = MUSA_C_32F; return true;
        case FLAGSPARSE_C_64F: *out = MUSA_C_64F; return true;
        default: return false;  // SDK 4.3.5 generic sparse does not expose fp16.
    }
}

bool map_op(flagsparseOperation_t in, musparseOperation_t* out) {
    switch (in) {
        case FLAGSPARSE_OPERATION_NON_TRANSPOSE:
            *out = MUSPARSE_OPERATION_NON_TRANSPOSE; return true;
        case FLAGSPARSE_OPERATION_TRANSPOSE:
            *out = MUSPARSE_OPERATION_TRANSPOSE; return true;
        case FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
            *out = MUSPARSE_OPERATION_CONJUGATE_TRANSPOSE; return true;
        default: return false;
    }
}

Status make_matrix(const DeviceCsr& A, musparseSpMatDescr_t* out) {
    musaDataType_t type;
    if (!map_dtype(A.dtype, &type)) return Status::no("muSPARSE: dtype unsupported");
    musparseStatus_t s;
    if (A.is_coo()) {
        s = musparseCreateCoo(out, A.rows, A.cols, A.nnz, A.coo_rows, A.indices,
                              A.values, MUSPARSE_INDEX_32I, MUSPARSE_INDEX_BASE_ZERO,
                              type);
    } else if (A.is_csc()) {
        // SDK 4.3.5 rejects CSC descriptors in the generic SpMV/SpMM paths.
        // A CSC(A) buffer is bit-for-bit CSR(A^T), so represent it that way and
        // flip the requested operation below. No conversion or allocation is
        // introduced into either side's timed region.
        s = musparseCreateCsr(out, A.cols, A.rows, A.nnz, A.indptr, A.indices,
                              A.values, MUSPARSE_INDEX_32I, MUSPARSE_INDEX_32I,
                              MUSPARSE_INDEX_BASE_ZERO, type);
    } else {
        s = musparseCreateCsr(out, A.rows, A.cols, A.nnz, A.indptr, A.indices,
                              A.values, MUSPARSE_INDEX_32I, MUSPARSE_INDEX_32I,
                              MUSPARSE_INDEX_BASE_ZERO, type);
    }
    return s == MUSPARSE_STATUS_SUCCESS ? Status::good() : Status::no(fail("CreateSpMat", s));
}

musparseOrder_t map_order(flagsparseOrder_t order) {
    return order == FLAGSPARSE_ORDER_ROW ? MUSPARSE_ORDER_ROW : MUSPARSE_ORDER_COLUMN;
}

// CSC(A) goes to muSPARSE as CSR(A^T) = M, so op(A) is re-expressed on M:
// A = M^T, A^T = M. A^H = conj(M) is a conjugate WITHOUT a transpose, which the
// generic API cannot spell, so a complex conjugate-transpose on CSC is declined.
// Passing CONJUGATE_TRANSPOSE through used to compute M^H x = conj(A) x, and on a
// rectangular A it wrote rows-many outputs into a cols-long y: past the buffer,
// which broke every later row of the SpMV sweep (2026-11-03 run, from GL7d14 on).
// On a real A conjugation is a no-op, so A^H = A^T = M.
Status csc_as_csr_operation(const DeviceCsr& A, musparseOperation_t* op) {
    if (!A.is_csc()) return Status::good();
    if (*op == MUSPARSE_OPERATION_NON_TRANSPOSE) {
        *op = MUSPARSE_OPERATION_TRANSPOSE;
    } else if (*op == MUSPARSE_OPERATION_TRANSPOSE) {
        *op = MUSPARSE_OPERATION_NON_TRANSPOSE;
    } else if (A.dtype == FLAGSPARSE_C_32F || A.dtype == FLAGSPARSE_C_64F) {
        return Status::no("muSPARSE: CSC is passed as CSR(A^T), and op(A)=A^H would be "
                          "conj(A^T) without a transpose, which the generic API cannot express");
    } else {
        *op = MUSPARSE_OPERATION_NON_TRANSPOSE;
    }
    return Status::good();
}

Status set_triangle(musparseSpMatDescr_t A, flagsparseFillMode_t fill,
                    flagsparseDiagType_t diag) {
    const musparseFillMode_t mf = fill == FLAGSPARSE_FILL_MODE_UPPER
        ? MUSPARSE_FILL_MODE_UPPER : MUSPARSE_FILL_MODE_LOWER;
    const musparseDiagType_t md = diag == FLAGSPARSE_DIAG_TYPE_UNIT
        ? MUSPARSE_DIAG_TYPE_UNIT : MUSPARSE_DIAG_TYPE_NON_UNIT;
    musparseStatus_t s = musparseSpMatSetAttribute(A, MUSPARSE_SPMAT_FILL_MODE,
                                                     &mf, sizeof(mf));
    if (s != MUSPARSE_STATUS_SUCCESS) return Status::no(fail("SpMatSetAttribute(fill)", s));
    s = musparseSpMatSetAttribute(A, MUSPARSE_SPMAT_DIAG_TYPE, &md, sizeof(md));
    return s == MUSPARSE_STATUS_SUCCESS ? Status::good()
                                        : Status::no(fail("SpMatSetAttribute(diag)", s));
}

template <typename F>
Status timed(F&& once, int warmup, int iters, Timing* out,
             const std::function<Status()>& prepare = {}) {
    if (!vendor()) return Status::no("muSPARSE Create failed");
    if (!out || warmup < 0 || iters < 1) return Status::no("muSPARSE: invalid timing arguments");
    auto reset = [&]() {
        if (!prepare) return Status::good();
        Status s = prepare();
        ad::synchronize();
        return s;
    };
    if (Status s = reset(); !s.ok) return s;
    if (Status s = once(); !s.ok) return s;  // JIT/setup outside measurements.
    for (int i = 0; i < warmup; ++i) {
        if (Status s = reset(); !s.ok) return s;
        if (Status s = once(); !s.ok) return s;
    }
    ad::synchronize();
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        if (Status s = reset(); !s.ok) return s;
        const auto start = std::chrono::steady_clock::now();
        if (Status s = once(); !s.ok) return s;
        ad::synchronize();
        samples.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count());
    }
    std::sort(samples.begin(), samples.end());
    out->median_ms = samples[samples.size() / 2];
    return Status::good();
}

Status native(musparseStatus_t s, const char* call) {
    return s == MUSPARSE_STATUS_SUCCESS ? Status::good() : Status::no(fail(call, s));
}

std::size_t value_bytes(flagsparseDataType_t dt) {
    switch (dt) {
        case FLAGSPARSE_R_32F: return sizeof(float);
        case FLAGSPARSE_R_64F: return sizeof(double);
        case FLAGSPARSE_C_32F: return sizeof(std::complex<float>);
        case FLAGSPARSE_C_64F: return sizeof(std::complex<double>);
        default: return 0;
    }
}

// Host copies are benchmark preparation, never a replacement compute kernel.
template <typename T>
bool download(const void* p, std::vector<T>& v) {
    return v.empty() || ad::memcpy_d2h(v.data(), reinterpret_cast<ad::DevicePtr>(p),
                                      v.size() * sizeof(T)) == FLAGSPARSE_STATUS_SUCCESS;
}
template <typename T>
bool upload(Scratch& dst, const std::vector<T>& v) {
    return dst.grab(v.size() * sizeof(T)) && (v.empty() ||
        ad::memcpy_h2d(reinterpret_cast<ad::DevicePtr>(dst.p), v.data(),
                       v.size() * sizeof(T)) == FLAGSPARSE_STATUS_SUCCESS);
}
template <typename T> T conjugated(T x) { return x; }
template <typename T> std::complex<T> conjugated(std::complex<T> x) { return std::conj(x); }

struct PreparedCsr {
    DeviceCsr matrix;
    Scratch ptr, idx, val;
};

template <typename T>
Status prepare_transposed(const DeviceCsr& A, bool conjugate, PreparedCsr* out) {
    std::vector<T> values(static_cast<std::size_t>(A.nnz));
    if (!download(A.values, values)) return Status::no("muSPARSE: preparation download failed");
    out->matrix = A;
    out->matrix.rows = A.cols;
    out->matrix.cols = A.rows;
    out->matrix.format = DeviceCsr::Format::Csr;
    out->matrix.coo_rows = nullptr;
    if (A.is_csc()) {
        // CSC(A) already has the structure of CSR(A^T); only values need conjugation.
        if (conjugate) for (auto& x : values) x = conjugated(x);
        if (!upload(out->val, values)) return Status::no("muSPARSE: preparation upload failed");
    } else {
        std::vector<int> rows(A.nnz), cols(A.nnz), ptr(A.cols + 1, 0), idx(A.nnz);
        if (!download(A.coo_rows, rows) || !download(A.indices, cols))
            return Status::no("muSPARSE: COO preparation download failed");
        for (int64_t i = 0; i < A.nnz; ++i) {
            if (rows[i] < 0 || rows[i] >= A.rows || cols[i] < 0 || cols[i] >= A.cols)
                return Status::no("muSPARSE: COO preparation index out of range");
            ++ptr[cols[i] + 1];
        }
        for (int64_t i = 0; i < A.cols; ++i) ptr[i + 1] += ptr[i];
        auto next = ptr;
        std::vector<T> reordered(A.nnz);
        for (int64_t i = 0; i < A.nnz; ++i) {
            const int j = next[cols[i]]++;
            idx[j] = rows[i];
            reordered[j] = conjugate ? conjugated(values[i]) : values[i];
        }
        if (!upload(out->ptr, ptr) || !upload(out->idx, idx) || !upload(out->val, reordered))
            return Status::no("muSPARSE: CSR preparation upload failed");
        out->matrix.indptr = out->ptr.p;
        out->matrix.indices = out->idx.p;
    }
    out->matrix.values = out->val.p;
    return Status::good();
}

Status prepare_transposed(const DeviceCsr& A, bool conjugate, PreparedCsr* out, Timing* t) {
    const auto start = std::chrono::steady_clock::now();
    Status s = Status::no("muSPARSE: preparation dtype unsupported");
    switch (A.dtype) {
        case FLAGSPARSE_R_32F: s = prepare_transposed<float>(A, conjugate, out); break;
        case FLAGSPARSE_R_64F: s = prepare_transposed<double>(A, conjugate, out); break;
        case FLAGSPARSE_C_32F: s = prepare_transposed<std::complex<float>>(A, conjugate, out); break;
        case FLAGSPARSE_C_64F: s = prepare_transposed<std::complex<double>>(A, conjugate, out); break;
        default: break;
    }
    ad::synchronize();
    t->prepare_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - start).count();
    return s;
}

}  // namespace

bool available() { return true; }
const char* name() {
    static const std::string label = [] {
        int v = 0;
        if (musparseGetVersion(vendor(), &v) != MUSPARSE_STATUS_SUCCESS)
            return std::string("muSPARSE (version unavailable)");
        return std::string("muSPARSE ") + std::to_string(v / 100000) + "." +
            std::to_string((v / 100) % 1000) + "." + std::to_string(v % 100);
    }();
    return label.c_str();
}

Status spmv_csr(const DeviceCsr& A, const void* x, void* y, const void* alpha,
                const void* beta, flagsparseOperation_t op, int warmup, int iters,
                Timing* out) {
    if (A.is_csc() && op == FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE &&
        (A.dtype == FLAGSPARSE_C_32F || A.dtype == FLAGSPARSE_C_64F)) {
        out->method = "CSC(A) -> CSR(A^H), conjugated values; muSPARSE SpMV non-transpose";
        PreparedCsr prepared;
        if (Status s = prepare_transposed(A, true, &prepared, out); !s.ok) return s;
        return spmv_csr(prepared.matrix, x, y, alpha, beta,
                        FLAGSPARSE_OPERATION_NON_TRANSPOSE, warmup, iters, out);
    }
    musaDataType_t type; musparseOperation_t mop;
    if (!map_dtype(A.dtype, &type) || !map_op(op, &mop))
        return Status::no("muSPARSE: unsupported SpMV type or operation");
    if (Status s = csc_as_csr_operation(A, &mop); !s.ok) return s;
    musparseSpMatDescr_t mat = nullptr; musparseDnVecDescr_t vx = nullptr, vy = nullptr;
    if (Status s = make_matrix(A, &mat); !s.ok) return s;
    const int64_t xlen = op == FLAGSPARSE_OPERATION_NON_TRANSPOSE ? A.cols : A.rows;
    const int64_t ylen = op == FLAGSPARSE_OPERATION_NON_TRANSPOSE ? A.rows : A.cols;
    musparseStatus_t st = musparseCreateDnVec(&vx, xlen, const_cast<void*>(x), type);
    if (st == MUSPARSE_STATUS_SUCCESS) st = musparseCreateDnVec(&vy, ylen, y, type);
    std::size_t bytes = 0;
    if (st == MUSPARSE_STATUS_SUCCESS) st = musparseSpMV_bufferSize(
        vendor(), mop, alpha, mat, vx, beta, vy, type, MUSPARSE_SPMV_ALG_DEFAULT, &bytes);
    Scratch scratch;
    Status result = native(st, "SpMV setup");
    if (result.ok && !scratch.grab(bytes)) result = Status::no("muSPARSE: SpMV scratch allocation failed");
    if (result.ok) result = timed([&] { return native(musparseSpMV(
        vendor(), mop, alpha, mat, vx, beta, vy, type, MUSPARSE_SPMV_ALG_DEFAULT,
        scratch.p), "SpMV"); }, warmup, iters, out);
    if (vx) musparseDestroyDnVec(vx); if (vy) musparseDestroyDnVec(vy);
    musparseDestroySpMat(mat);
    return result;
}

Status spmv_mixed(const DeviceCsr& A, const void* x, flagsparseDataType_t x_dtype,
                  void* y, flagsparseDataType_t y_dtype,
                  flagsparseDataType_t compute_dtype, const void* alpha,
                  const void* beta, flagsparseOperation_t op, int warmup, int iters,
                  Timing* out) {
    if (x_dtype != A.dtype || y_dtype != A.dtype || compute_dtype != A.dtype)
        return Status::no("muSPARSE: mixed-precision SpMV baseline is not wired");
    return spmv_csr(A, x, y, alpha, beta, op, warmup, iters, out);
}

Status spmm_csr(const DeviceCsr& A, const void* B, int64_t n, int64_t ldb, void* C,
                int64_t ldc, const void* alpha, const void* beta,
                flagsparseOperation_t opA, flagsparseOperation_t opB, int warmup,
                int iters, Timing* out, flagsparseOrder_t orderB,
                flagsparseOrder_t orderC) {
    if (A.is_coo() && opA != FLAGSPARSE_OPERATION_NON_TRANSPOSE) {
        out->method = "COO(A) -> CSR(op(A)); muSPARSE SpMM non-transpose";
        PreparedCsr prepared;
        if (Status s = prepare_transposed(A, opA == FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
                                          &prepared, out); !s.ok) return s;
        return spmm_csr(prepared.matrix, B, n, ldb, C, ldc, alpha, beta,
                        FLAGSPARSE_OPERATION_NON_TRANSPOSE, opB, warmup, iters, out,
                        orderB, orderC);
    }
    musaDataType_t type; musparseOperation_t ma, mb;
    if (!map_dtype(A.dtype, &type) || !map_op(opA, &ma) || !map_op(opB, &mb))
        return Status::no("muSPARSE: unsupported SpMM type or operation");
    if (Status s = csc_as_csr_operation(A, &ma); !s.ok) return s;
    musparseSpMatDescr_t mat = nullptr; musparseDnMatDescr_t b = nullptr, c = nullptr;
    if (Status s = make_matrix(A, &mat); !s.ok) return s;
    const int64_t k = opA == FLAGSPARSE_OPERATION_NON_TRANSPOSE ? A.cols : A.rows;
    const int64_t m = opA == FLAGSPARSE_OPERATION_NON_TRANSPOSE ? A.rows : A.cols;
    musparseStatus_t st = musparseCreateDnMat(&b, k, n, ldb, const_cast<void*>(B), type,
                                               map_order(orderB));
    if (st == MUSPARSE_STATUS_SUCCESS) st = musparseCreateDnMat(&c, m, n, ldc, C, type,
                                                                  map_order(orderC));
    std::size_t bytes = 0;
    if (st == MUSPARSE_STATUS_SUCCESS) st = musparseSpMM(
        vendor(), ma, mb, alpha, mat, b, beta, c, type, MUSPARSE_SPMM_ALG_DEFAULT,
        MUSPARSE_SPMM_STAGE_BUFFER_SIZE, &bytes, nullptr);
    Scratch scratch;
    Status result = native(st, "SpMM buffer size");
    if (result.ok && !scratch.grab(bytes)) result = Status::no("muSPARSE: SpMM scratch allocation failed");
    if (result.ok) result = native(musparseSpMM(
        vendor(), ma, mb, alpha, mat, b, beta, c, type, MUSPARSE_SPMM_ALG_DEFAULT,
        MUSPARSE_SPMM_STAGE_PREPROCESS, nullptr, scratch.p), "SpMM preprocess");
    if (result.ok) result = timed([&] { return native(musparseSpMM(
        vendor(), ma, mb, alpha, mat, b, beta, c, type, MUSPARSE_SPMM_ALG_DEFAULT,
        MUSPARSE_SPMM_STAGE_COMPUTE, nullptr, scratch.p), "SpMM compute"); }, warmup, iters, out);
    if (b) musparseDestroyDnMat(b); if (c) musparseDestroyDnMat(c);
    musparseDestroySpMat(mat);
    return result;
}

Status sddmm_csr(const DeviceCsr& A, const void* B, int64_t k, int64_t ldb,
                 const void* D, int64_t ldd, const void* alpha, const void* beta,
                 flagsparseOperation_t opA, flagsparseOperation_t opB,
                 flagsparseOrder_t orderB, flagsparseOrder_t orderD,
                 int warmup, int iters, Timing* out) {
    musaDataType_t type; musparseOperation_t ma, mb;
    if (!map_dtype(A.dtype, &type) || !map_op(opA, &ma) || !map_op(opB, &mb))
        return Status::no("muSPARSE: unsupported SDDMM dtype or operation");
    musparseSpMatDescr_t mat = nullptr; musparseDnMatDescr_t b = nullptr, d = nullptr;
    if (Status s = make_matrix(A, &mat); !s.ok) return s;
    const int64_t b_rows = opA == FLAGSPARSE_OPERATION_TRANSPOSE ? k : A.rows;
    const int64_t b_cols = opA == FLAGSPARSE_OPERATION_TRANSPOSE ? A.rows : k;
    const int64_t d_rows = opB == FLAGSPARSE_OPERATION_TRANSPOSE ? A.cols : k;
    const int64_t d_cols = opB == FLAGSPARSE_OPERATION_TRANSPOSE ? k : A.cols;
    musparseStatus_t st = musparseCreateDnMat(&b, b_rows, b_cols, ldb, const_cast<void*>(B),
                                               type, map_order(orderB));
    if (st == MUSPARSE_STATUS_SUCCESS) st = musparseCreateDnMat(
        &d, d_rows, d_cols, ldd, const_cast<void*>(D), type, map_order(orderD));
    std::size_t bytes = 0;
    if (st == MUSPARSE_STATUS_SUCCESS) st = musparseSDDMM_bufferSize(
        vendor(), ma, mb,
        alpha, b, d, beta, mat, type, MUSPARSE_SDDMM_ALG_DEFAULT, &bytes);
    Scratch scratch;
    Status result = native(st, "SDDMM buffer size");
    if (result.ok && !scratch.grab(bytes)) result = Status::no("muSPARSE: SDDMM scratch allocation failed");
    if (result.ok) result = native(musparseSDDMM_preprocess(
        vendor(), ma, mb,
        alpha, b, d, beta, mat, type, MUSPARSE_SDDMM_ALG_DEFAULT, scratch.p),
        "SDDMM preprocess");
    if (result.ok) result = timed([&] { return native(musparseSDDMM(
        vendor(), ma, mb,
        alpha, b, d, beta, mat, type, MUSPARSE_SDDMM_ALG_DEFAULT, scratch.p), "SDDMM");
    }, warmup, iters, out);
    if (b) musparseDestroyDnMat(b); if (d) musparseDestroyDnMat(d);
    musparseDestroySpMat(mat);
    return result;
}

Status spvv(const void* sparse_val, const void* sparse_idx, const void* dense,
            int64_t nnz, int64_t size, void* result, flagsparseDataType_t dtype,
            flagsparseDataType_t compute_dtype, flagsparseOperation_t op,
            int warmup, int iters, Timing* out) {
    musaDataType_t type, compute; musparseOperation_t mop;
    if (!map_dtype(dtype, &type) || !map_dtype(compute_dtype, &compute) || !map_op(op, &mop))
        return Status::no("muSPARSE: unsupported SpVV dtype or operation");
    musparseSpVecDescr_t x = nullptr; musparseDnVecDescr_t y = nullptr;
    musparseStatus_t st = musparseCreateSpVec(&x, size, nnz, const_cast<void*>(sparse_idx),
                                               const_cast<void*>(sparse_val), MUSPARSE_INDEX_32I,
                                               MUSPARSE_INDEX_BASE_ZERO, type);
    if (st == MUSPARSE_STATUS_SUCCESS)
        st = musparseCreateDnVec(&y, size, const_cast<void*>(dense), type);
    std::size_t bytes = 0;
    if (st == MUSPARSE_STATUS_SUCCESS)
        st = musparseSpVV(vendor(), mop, x, y, result, compute, &bytes, nullptr);
    Scratch scratch;
    Status status = native(st, "SpVV setup");
    if (status.ok && !scratch.grab(bytes)) status = Status::no("muSPARSE: SpVV scratch allocation failed");
    if (status.ok) status = timed([&] { return native(musparseSpVV(
        vendor(), mop, x, y, result, compute, &bytes, scratch.p), "SpVV"); }, warmup, iters, out);
    if (x) musparseDestroySpVec(x);
    if (y) musparseDestroyDnVec(y);
    return status;
}

void free_csr(BaselineCsrOut* c) {
    if (!c) return;
    for (void** p : {&c->indptr, &c->indices, &c->values}) {
        if (*p) ad::device_free(reinterpret_cast<ad::DevicePtr>(*p));
        *p = nullptr;
    }
    c->rows = c->cols = c->nnz = 0;
}

Status spgemm_csr_impl(const DeviceCsr& A, const void* alpha, const void* beta,
                       int warmup, int iters, Timing* out, BaselineCsrOut* result) {
    if (A.rows != A.cols) return Status::no("A*A needs a square A");
    if (A.is_coo()) return Status::no("muSPARSE SpGEMM is CSR only");
    musaDataType_t type;
    if (!map_dtype(A.dtype, &type)) return Status::no("muSPARSE: SpGEMM dtype unsupported");

    musparseSpMatDescr_t a = nullptr, b = nullptr, c = nullptr;
    if (Status s = make_matrix(A, &a); !s.ok) return s;
    if (Status s = make_matrix(A, &b); !s.ok) {
        musparseDestroySpMat(a);
        return s;
    }
    Scratch c_ptr, c_ind, c_val, scratch;
    if (!c_ptr.grab(static_cast<std::size_t>(A.rows + 1) * sizeof(std::int32_t))) {
        musparseDestroySpMat(a); musparseDestroySpMat(b);
        return Status::no("muSPARSE: SpGEMM row-pointer allocation failed");
    }
    musparseStatus_t st = musparseCreateCsr(
        &c, A.rows, A.cols, 0, c_ptr.p, nullptr, nullptr, MUSPARSE_INDEX_32I,
        MUSPARSE_INDEX_32I, MUSPARSE_INDEX_BASE_ZERO, type);
    auto cleanup = [&](Status status) {
        if (a) musparseDestroySpMat(a);
        if (b) musparseDestroySpMat(b);
        if (c) musparseDestroySpMat(c);
        return status;
    };
    if (st != MUSPARSE_STATUS_SUCCESS) return cleanup(native(st, "CreateCsr(C)"));

    const auto NT = MUSPARSE_OPERATION_NON_TRANSPOSE;
    std::size_t bytes = 0;
    st = musparseSpGEMM(vendor(), NT, NT, alpha, a, b, beta, c, c, type,
                        MUSPARSE_SPGEMM_ALG_DEFAULT,
                        MUSPARSE_SPGEMM_STAGE_BUFFER_SIZE, &bytes, nullptr);
    if (st != MUSPARSE_STATUS_SUCCESS) return cleanup(native(st, "SpGEMM buffer size"));
    if (!scratch.grab(bytes)) return cleanup(Status::no("muSPARSE: SpGEMM scratch allocation failed"));
    st = musparseSpGEMM(vendor(), NT, NT, alpha, a, b, beta, c, c, type,
                        MUSPARSE_SPGEMM_ALG_DEFAULT, MUSPARSE_SPGEMM_STAGE_NNZ,
                        &bytes, scratch.p);
    if (st != MUSPARSE_STATUS_SUCCESS) return cleanup(native(st, "SpGEMM nnz"));

    int64_t rows = 0, cols = 0, nnz = 0;
    st = musparseSpMatGetSize(c, &rows, &cols, &nnz);
    if (st != MUSPARSE_STATUS_SUCCESS) return cleanup(native(st, "SpMatGetSize(C)"));
    const std::size_t value_bytes =
        A.dtype == FLAGSPARSE_R_64F ? sizeof(double) :
        A.dtype == FLAGSPARSE_C_32F ? 2 * sizeof(float) :
        A.dtype == FLAGSPARSE_C_64F ? 2 * sizeof(double) : sizeof(float);
    if (!c_ind.grab(static_cast<std::size_t>(nnz) * sizeof(std::int32_t)) ||
        !c_val.grab(static_cast<std::size_t>(nnz) * value_bytes)) {
        return cleanup(Status::no("muSPARSE: SpGEMM result allocation failed"));
    }
    st = musparseCsrSetPointers(c, c_ptr.p, c_ind.p, c_val.p);
    if (st != MUSPARSE_STATUS_SUCCESS) return cleanup(native(st, "CsrSetPointers(C)"));

    auto compute = [&]() {
        return native(musparseSpGEMM(
            vendor(), NT, NT, alpha, a, b, beta, c, c, type,
            MUSPARSE_SPGEMM_ALG_DEFAULT, MUSPARSE_SPGEMM_STAGE_COMPUTE, &bytes,
            scratch.p), "SpGEMM compute");
    };
    Status status = timed(compute, warmup, iters, out);
    if (result && status.ok) {
        free_csr(result);
        result->rows = rows;
        result->cols = cols;
        result->nnz = nnz;
        result->indptr = c_ptr.release();
        result->indices = c_ind.release();
        result->values = c_val.release();
    }
    return cleanup(status);
}

Status spgemm_csr(const DeviceCsr& A, const void* alpha, const void* beta,
                  int warmup, int iters, Timing* out, BaselineCsrOut* result) {
    try {
        return spgemm_csr_impl(A, alpha, beta, warmup, iters, out, result);
    } catch (musparseStatus_t status) {
        // SDK 4.3.5 throws its status enum for some unsupported SpGEMM
        // configurations instead of returning it. Keep the benchmark process
        // alive and expose the exact vendor status in the row detail.
        return Status::no(fail("SpGEMM threw status", status));
    } catch (const std::exception& exc) {
        return Status::no(std::string("muSPARSE SpGEMM threw: ") + exc.what());
    } catch (...) {
        return Status::no("muSPARSE SpGEMM threw an unknown exception");
    }
}

Status spsv_csr(const DeviceCsr& A, const void* x, void* y, const void* alpha,
                flagsparseFillMode_t fill, flagsparseDiagType_t diag,
                flagsparseOperation_t op, int warmup, int iters, Timing* out) {
    musaDataType_t type; musparseOperation_t mop;
    if (!map_dtype(A.dtype, &type) || !map_op(op, &mop))
        return Status::no("muSPARSE: unsupported SpSV type or operation");
    musparseSpMatDescr_t mat = nullptr; musparseDnVecDescr_t vx = nullptr, vy = nullptr;
    if (Status s = make_matrix(A, &mat); !s.ok) return s;
    Status result = set_triangle(mat, fill, diag);
    musparseStatus_t st = result.ok ? musparseCreateDnVec(&vx, A.cols, const_cast<void*>(x), type)
                                    : MUSPARSE_STATUS_INVALID_VALUE;
    if (result.ok && st == MUSPARSE_STATUS_SUCCESS) st = musparseCreateDnVec(&vy, A.rows, y, type);
    std::size_t bytes = 0;
    if (result.ok && st == MUSPARSE_STATUS_SUCCESS) st = musparseSpSV(
        vendor(), mop, alpha, mat, vx, vy, type, MUSPARSE_SPSV_ALG_DEFAULT,
        MUSPARSE_SPSV_STAGE_BUFFER_SIZE, &bytes, nullptr);
    if (result.ok) result = native(st, "SpSV buffer size");
    Scratch scratch;
    if (result.ok && !scratch.grab(bytes)) result = Status::no("muSPARSE: SpSV scratch allocation failed");
    if (result.ok) result = native(musparseSpSV(
        vendor(), mop, alpha, mat, vx, vy, type, MUSPARSE_SPSV_ALG_DEFAULT,
        MUSPARSE_SPSV_STAGE_PREPROCESS, nullptr, scratch.p), "SpSV preprocess");
    if (result.ok) result = timed([&] { return native(musparseSpSV(
        vendor(), mop, alpha, mat, vx, vy, type, MUSPARSE_SPSV_ALG_DEFAULT,
        MUSPARSE_SPSV_STAGE_COMPUTE, nullptr, scratch.p), "SpSV compute"); }, warmup, iters, out);
    if (vx) musparseDestroyDnVec(vx); if (vy) musparseDestroyDnVec(vy);
    musparseDestroySpMat(mat);
    return result;
}

Status spsm_csr(const DeviceCsr& A, const void* B, int64_t n, int64_t ldb, void* C,
                int64_t ldc, const void* alpha, flagsparseFillMode_t fill,
                flagsparseDiagType_t diag, flagsparseOperation_t op, int warmup,
                int iters, Timing* out) {
    musaDataType_t type; musparseOperation_t mop;
    if (!map_dtype(A.dtype, &type) || !map_op(op, &mop))
        return Status::no("muSPARSE: unsupported SpSM type or operation");
    musparseSpMatDescr_t mat = nullptr; musparseDnMatDescr_t b = nullptr, c = nullptr;
    if (Status s = make_matrix(A, &mat); !s.ok) return s;
    Status result = set_triangle(mat, fill, diag);
    musparseStatus_t st = result.ok ? musparseCreateDnMat(
        &b, A.cols, n, ldb, const_cast<void*>(B), type, MUSPARSE_ORDER_COLUMN)
        : MUSPARSE_STATUS_INVALID_VALUE;
    if (result.ok && st == MUSPARSE_STATUS_SUCCESS) st = musparseCreateDnMat(
        &c, A.rows, n, ldc, C, type, MUSPARSE_ORDER_COLUMN);
    std::size_t bytes = 0;
    if (result.ok && st == MUSPARSE_STATUS_SUCCESS) st = musparseSpSM(
        vendor(), mop, MUSPARSE_OPERATION_NON_TRANSPOSE, alpha, mat, b, c, type,
        MUSPARSE_SPSM_ALG_DEFAULT, MUSPARSE_SPSM_STAGE_BUFFER_SIZE, &bytes, nullptr);
    if (result.ok) result = native(st, "SpSM buffer size");
    Scratch scratch;
    if (result.ok && !scratch.grab(bytes)) result = Status::no("muSPARSE: SpSM scratch allocation failed");
    if (result.ok) result = native(musparseSpSM(
        vendor(), mop, MUSPARSE_OPERATION_NON_TRANSPOSE, alpha, mat, b, c, type,
        MUSPARSE_SPSM_ALG_DEFAULT, MUSPARSE_SPSM_STAGE_PREPROCESS, nullptr, scratch.p),
        "SpSM preprocess");
    if (result.ok) result = timed([&] { return native(musparseSpSM(
        vendor(), mop, MUSPARSE_OPERATION_NON_TRANSPOSE, alpha, mat, b, c, type,
        MUSPARSE_SPSM_ALG_DEFAULT, MUSPARSE_SPSM_STAGE_COMPUTE, nullptr, scratch.p),
        "SpSM compute"); }, warmup, iters, out);
    if (b) musparseDestroyDnMat(b); if (c) musparseDestroyDnMat(c);
    musparseDestroySpMat(mat);
    return result;
}

Status gather(const void* dense, void* values, const void* indices, int64_t nnz,
              int64_t size, flagsparseDataType_t dtype, int warmup, int iters,
              Timing* out) {
    musaDataType_t type;
    if (!map_dtype(dtype, &type)) return Status::no("muSPARSE: Gather dtype unsupported");
    musparseDnVecDescr_t d = nullptr; musparseSpVecDescr_t s = nullptr;
    musparseStatus_t st = musparseCreateDnVec(&d, size, const_cast<void*>(dense), type);
    if (st == MUSPARSE_STATUS_SUCCESS) st = musparseCreateSpVec(
        &s, size, nnz, const_cast<void*>(indices), values, MUSPARSE_INDEX_32I,
        MUSPARSE_INDEX_BASE_ZERO, type);
    Status result = native(st, "Gather setup");
    if (result.ok) result = timed([&] { return native(musparseGather(vendor(), d, s), "Gather"); },
                                   warmup, iters, out);
    if (d) musparseDestroyDnVec(d); if (s) musparseDestroySpVec(s);
    return result;
}

Status scatter(void* dense, const void* values, const void* indices, int64_t nnz,
               int64_t size, flagsparseDataType_t dtype, int warmup, int iters,
               Timing* out) {
    musaDataType_t type;
    if (!map_dtype(dtype, &type)) return Status::no("muSPARSE: Scatter dtype unsupported");
    musparseDnVecDescr_t d = nullptr; musparseSpVecDescr_t s = nullptr;
    musparseStatus_t st = musparseCreateDnVec(&d, size, dense, type);
    if (st == MUSPARSE_STATUS_SUCCESS) st = musparseCreateSpVec(
        &s, size, nnz, const_cast<void*>(indices), const_cast<void*>(values),
        MUSPARSE_INDEX_32I, MUSPARSE_INDEX_BASE_ZERO, type);
    Status result = native(st, "Scatter setup");
    if (result.ok) result = timed([&] { return native(musparseScatter(vendor(), s, d), "Scatter"); },
                                   warmup, iters, out);
    if (d) musparseDestroyDnVec(d); if (s) musparseDestroySpVec(s);
    return result;
}

Status gemvi(flagsparseOperation_t op, int m, int n, const void* alpha, const void* A,
             int lda, int nnz, const void* x_val, const void* x_ind, const void* beta,
             void* y, flagsparseDataType_t dtype, int warmup, int iters, Timing* out) {
    if (op != FLAGSPARSE_OPERATION_NON_TRANSPOSE)
        return Status::no("muSPARSE: GEMVI transpose compute is not implemented");
    if (dtype != FLAGSPARSE_R_32F && dtype != FLAGSPARSE_C_32F)
        return Status::no("muSPARSE: GEMVI dtype not verified");
    out->method = "musparse<t>gemvi";
    std::size_t bytes = 0;
    const auto mop = MUSPARSE_OPERATION_NON_TRANSPOSE;
    auto st = dtype == FLAGSPARSE_R_32F
        ? musparseSgemvi_bufferSize(vendor(), mop, m, n, nnz, &bytes)
        : musparseCgemvi_bufferSize(vendor(), mop, m, n, nnz, &bytes);
    if (Status s = native(st, "gemvi buffer size"); !s.ok) return s;
    Scratch scratch;
    if (!scratch.grab(bytes)) return Status::no("muSPARSE: GEMVI scratch allocation failed");
    // Preserve beta*y semantics across repetitions; the delivery beta is zero.
    const auto* b = static_cast<const float*>(beta);
    const bool reset = b[0] != 0 || (dtype == FLAGSPARSE_C_32F && b[1] != 0);
    std::vector<unsigned char> initial(reset ? m * value_bytes(dtype) : 0);
    if (!download(y, initial)) return Status::no("muSPARSE: GEMVI initial y download failed");
    std::function<Status()> prepare;
    if (reset) prepare = [&] {
        return ad::memcpy_h2d(reinterpret_cast<ad::DevicePtr>(y), initial.data(), initial.size())
            == FLAGSPARSE_STATUS_SUCCESS ? Status::good() : Status::no("GEMVI y reset failed");
    };
    return timed([&] {
        if (dtype == FLAGSPARSE_R_32F)
            return native(musparseSgemvi(vendor(), mop, m, n, static_cast<const float*>(alpha),
                static_cast<const float*>(A), lda, nnz, static_cast<const float*>(x_val),
                static_cast<const int*>(x_ind), static_cast<const float*>(beta),
                static_cast<float*>(y), MUSPARSE_INDEX_BASE_ZERO, scratch.p), "Sgemvi");
        return native(musparseCgemvi(vendor(), mop, m, n, static_cast<const muComplex*>(alpha),
            static_cast<const muComplex*>(A), lda, nnz, static_cast<const muComplex*>(x_val),
            static_cast<const int*>(x_ind), static_cast<const muComplex*>(beta),
            static_cast<muComplex*>(y), MUSPARSE_INDEX_BASE_ZERO, scratch.p), "Cgemvi");
    }, warmup, iters, out, prepare);
}

Status csrgeam2(const DeviceCsr& A, const DeviceCsr& B, const void* alpha, const void* beta,
                void* c_ptr, void* c_ind, void* c_val, int64_t capacity,
                int warmup, int iters, Timing* out) {
    if (A.is_coo() || A.is_csc() || B.is_coo() || B.is_csc() ||
        A.rows != B.rows || A.cols != B.cols || A.dtype != B.dtype ||
        (A.dtype != FLAGSPARSE_R_32F && A.dtype != FLAGSPARSE_C_32F))
        return Status::no("muSPARSE: GEAM requires matching f32/c32 CSR inputs");
    const auto max_index = std::numeric_limits<int>::max();
    if (A.rows > max_index || A.cols > max_index || A.nnz > max_index || B.nnz > max_index)
        return Status::no("muSPARSE: legacy GEAM requires int32 dimensions");
    out->method = "musparseXcsrgeam_nnz + musparse<t>csrgeam (both timed)";
    struct Descr {
        musparseMatDescr_t p = nullptr;
        ~Descr() { if (p) musparseDestroyMatDescr(p); }
    } a, b, c;
    for (auto* d : {&a, &b, &c}) {
        if (Status s = native(musparseCreateMatDescr(&d->p), "CreateMatDescr"); !s.ok) return s;
    }
    return timed([&] {
        int nnz = 0;
        Status s = native(musparseXcsrgeam_nnz(vendor(), A.rows, A.cols,
            a.p, A.nnz, static_cast<const int*>(A.indptr), static_cast<const int*>(A.indices),
            b.p, B.nnz, static_cast<const int*>(B.indptr), static_cast<const int*>(B.indices),
            c.p, static_cast<int*>(c_ptr), &nnz), "Xcsrgeam_nnz");
        if (!s.ok) return s;
        if (nnz > capacity) return Status::no("muSPARSE: GEAM output capacity exceeded");
        if (A.dtype == FLAGSPARSE_R_32F)
            return native(musparseScsrgeam(vendor(), A.rows, A.cols,
                static_cast<const float*>(alpha), a.p, A.nnz, static_cast<const float*>(A.values),
                static_cast<const int*>(A.indptr), static_cast<const int*>(A.indices),
                static_cast<const float*>(beta), b.p, B.nnz, static_cast<const float*>(B.values),
                static_cast<const int*>(B.indptr), static_cast<const int*>(B.indices), c.p,
                static_cast<float*>(c_val), static_cast<int*>(c_ptr), static_cast<int*>(c_ind)), "Scsrgeam");
        return native(musparseCcsrgeam(vendor(), A.rows, A.cols,
            static_cast<const muComplex*>(alpha), a.p, A.nnz, static_cast<const muComplex*>(A.values),
            static_cast<const int*>(A.indptr), static_cast<const int*>(A.indices),
            static_cast<const muComplex*>(beta), b.p, B.nnz, static_cast<const muComplex*>(B.values),
            static_cast<const int*>(B.indptr), static_cast<const int*>(B.indices), c.p,
            static_cast<muComplex*>(c_val), static_cast<int*>(c_ptr), static_cast<int*>(c_ind)), "Ccsrgeam");
    }, warmup, iters, out);
}

Status axpby(const void* sparse_val, const void* sparse_idx, void* dense,
             int64_t nnz, int64_t size, const void* alpha, const void* beta,
             flagsparseDataType_t dtype, int warmup, int iters, Timing* out) {
    musaDataType_t type;
    if (!map_dtype(dtype, &type)) return Status::no("muSPARSE: Axpby dtype unsupported");
    out->method = "musparseAxpby; initial y restored outside each timed call";
    std::vector<unsigned char> initial(size * value_bytes(dtype));
    if (!download(dense, initial)) return Status::no("muSPARSE: Axpby initial y download failed");
    musparseSpVecDescr_t x = nullptr;
    musparseDnVecDescr_t y = nullptr;
    auto st = musparseCreateSpVec(&x, size, nnz, const_cast<void*>(sparse_idx),
        const_cast<void*>(sparse_val), MUSPARSE_INDEX_32I, MUSPARSE_INDEX_BASE_ZERO, type);
    if (st == MUSPARSE_STATUS_SUCCESS) st = musparseCreateDnVec(&y, size, dense, type);
    Status s = native(st, "Axpby descriptors");
    if (s.ok) s = timed([&] { return native(musparseAxpby(vendor(), alpha, x, beta, y), "Axpby"); },
        warmup, iters, out, [&] {
            return ad::memcpy_h2d(reinterpret_cast<ad::DevicePtr>(dense), initial.data(), initial.size())
                == FLAGSPARSE_STATUS_SUCCESS ? Status::good() : Status::no("Axpby y reset failed");
        });
    if (x) musparseDestroySpVec(x);
    if (y) musparseDestroyDnVec(y);
    return s;
}

}  // namespace fstest::baseline
