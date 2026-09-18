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

// muSPARSE 4.3.5 baseline.  Its generic descriptors have the same broad shape
// as cuSPARSE, but SpMM/SpSV/SpSM/SpGEMM use stage arguments rather than the
// cuSPARSE split functions.  This is intentionally a native implementation,
// not a prefix-table instantiation: treating the two APIs as call-for-call gave
// a buildable-looking but invalid baseline.

#include <musparse.h>

#include <algorithm>
#include <chrono>
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
    const musparseStatus_t s = A.is_coo()
        ? musparseCreateCoo(out, A.rows, A.cols, A.nnz, A.coo_rows, A.indices,
                            A.values, MUSPARSE_INDEX_32I, MUSPARSE_INDEX_BASE_ZERO,
                            type)
        : musparseCreateCsr(out, A.rows, A.cols, A.nnz, A.indptr, A.indices,
                            A.values, MUSPARSE_INDEX_32I, MUSPARSE_INDEX_32I,
                            MUSPARSE_INDEX_BASE_ZERO, type);
    return s == MUSPARSE_STATUS_SUCCESS ? Status::good() : Status::no(fail("CreateSpMat", s));
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
Status timed(F&& once, int warmup, int iters, Timing* out) {
    if (!vendor()) return Status::no("muSPARSE Create failed");
    if (Status s = once(); !s.ok) return s;  // JIT/setup outside measurements.
    for (int i = 0; i < warmup; ++i) {
        if (Status s = once(); !s.ok) return s;
    }
    ad::synchronize();
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iters));
    for (int i = 0; i < iters; ++i) {
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

}  // namespace

bool available() { return true; }
const char* name() { return "muSPARSE 4.3.5"; }

Status spmv_csr(const DeviceCsr& A, const void* x, void* y, const void* alpha,
                const void* beta, flagsparseOperation_t op, int warmup, int iters,
                Timing* out) {
    musaDataType_t type; musparseOperation_t mop;
    if (!map_dtype(A.dtype, &type) || !map_op(op, &mop))
        return Status::no("muSPARSE: unsupported SpMV type or operation");
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

Status spmm_csr(const DeviceCsr& A, const void* B, int64_t n, int64_t ldb, void* C,
                int64_t ldc, const void* alpha, const void* beta,
                flagsparseOperation_t opA, flagsparseOperation_t opB, int warmup,
                int iters, Timing* out) {
    musaDataType_t type; musparseOperation_t ma, mb;
    if (!map_dtype(A.dtype, &type) || !map_op(opA, &ma) || !map_op(opB, &mb))
        return Status::no("muSPARSE: unsupported SpMM type or operation");
    musparseSpMatDescr_t mat = nullptr; musparseDnMatDescr_t b = nullptr, c = nullptr;
    if (Status s = make_matrix(A, &mat); !s.ok) return s;
    const int64_t k = opA == FLAGSPARSE_OPERATION_NON_TRANSPOSE ? A.cols : A.rows;
    const int64_t m = opA == FLAGSPARSE_OPERATION_NON_TRANSPOSE ? A.rows : A.cols;
    musparseStatus_t st = musparseCreateDnMat(&b, k, n, ldb, const_cast<void*>(B), type,
                                               MUSPARSE_ORDER_COLUMN);
    if (st == MUSPARSE_STATUS_SUCCESS) st = musparseCreateDnMat(&c, m, n, ldc, C, type,
                                                                  MUSPARSE_ORDER_COLUMN);
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
                 int warmup, int iters, Timing* out) {
    musaDataType_t type;
    if (!map_dtype(A.dtype, &type)) return Status::no("muSPARSE: unsupported SDDMM dtype");
    musparseSpMatDescr_t mat = nullptr; musparseDnMatDescr_t b = nullptr, d = nullptr;
    if (Status s = make_matrix(A, &mat); !s.ok) return s;
    musparseStatus_t st = musparseCreateDnMat(&b, A.rows, k, ldb, const_cast<void*>(B),
                                               type, MUSPARSE_ORDER_ROW);
    if (st == MUSPARSE_STATUS_SUCCESS) st = musparseCreateDnMat(
        &d, k, A.cols, ldd, const_cast<void*>(D), type, MUSPARSE_ORDER_ROW);
    std::size_t bytes = 0;
    if (st == MUSPARSE_STATUS_SUCCESS) st = musparseSDDMM_bufferSize(
        vendor(), MUSPARSE_OPERATION_NON_TRANSPOSE, MUSPARSE_OPERATION_NON_TRANSPOSE,
        alpha, b, d, beta, mat, type, MUSPARSE_SDDMM_ALG_DEFAULT, &bytes);
    Scratch scratch;
    Status result = native(st, "SDDMM buffer size");
    if (result.ok && !scratch.grab(bytes)) result = Status::no("muSPARSE: SDDMM scratch allocation failed");
    if (result.ok) result = native(musparseSDDMM_preprocess(
        vendor(), MUSPARSE_OPERATION_NON_TRANSPOSE, MUSPARSE_OPERATION_NON_TRANSPOSE,
        alpha, b, d, beta, mat, type, MUSPARSE_SDDMM_ALG_DEFAULT, scratch.p),
        "SDDMM preprocess");
    if (result.ok) result = timed([&] { return native(musparseSDDMM(
        vendor(), MUSPARSE_OPERATION_NON_TRANSPOSE, MUSPARSE_OPERATION_NON_TRANSPOSE,
        alpha, b, d, beta, mat, type, MUSPARSE_SDDMM_ALG_DEFAULT, scratch.p), "SDDMM");
    }, warmup, iters, out);
    if (b) musparseDestroyDnMat(b); if (d) musparseDestroyDnMat(d);
    musparseDestroySpMat(mat);
    return result;
}

void free_csr(BaselineCsrOut* c) {
    if (!c) return;
    for (void** p : {&c->indptr, &c->indices, &c->values}) {
        if (*p) ad::device_free(reinterpret_cast<ad::DevicePtr>(*p));
        *p = nullptr;
    }
    c->rows = c->cols = c->nnz = 0;
}

Status spgemm_csr(const DeviceCsr& A, const void* alpha, const void* beta,
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

}  // namespace fstest::baseline
