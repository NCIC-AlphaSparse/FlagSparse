// Shared body for the vendor baselines whose sparse library mirrors cuSPARSE's
// generic (descriptor) API. Included by baseline/<name>/baseline.cpp after
// defining the six macros below. cuSPARSE and hipSPARSE differ only in the
// symbol prefix and the data-type enum, so keeping one body stops the two from
// drifting -- the same reason src/adaptor/backend/_template.inc exists.
//
//   FS_SP_HEADER      the vendor header to include, e.g. <cusparse.h>
//   FS_SP(sym)        function prefix:      FS_SP(SpMV)      -> cusparseSpMV
//   FS_SP_E(sym)      enum prefix:          FS_SP_E(INDEX_32I) -> CUSPARSE_INDEX_32I
//   FS_DT(sym)        data-type enum:       FS_DT(R_32F)     -> CUDA_R_32F
//   FS_DT_TYPE        the data-type type,   e.g. cudaDataType_t / hipDataType
//   FS_SP_HANDLE      the handle type,      e.g. cusparseHandle_t
//   FS_SP_NAME        what to print,        e.g. "cuSPARSE"
//
// EVERY CALL IS CHECKED. A vendor status that is not SUCCESS becomes a Status::no
// naming the call site and the vendor's own status number, because the failure
// mode that matters here is silent: a baseline that quietly no-ops finishes in
// microseconds and reports a spectacular speedup for us. A baseline that errors
// must produce a blank speedup and a reason, never a fast one.

#include <algorithm>
#include <chrono>
#include <string>
#include <vector>

#include FS_SP_HEADER

#include "adaptor/adaptor.hpp"
#include "baseline/baseline.hpp"

namespace fstest::baseline {
namespace {

namespace ad = flagsparse::adaptor;

struct VendorHandle {
    FS_SP_HANDLE h = nullptr;
    VendorHandle() { FS_SP(Create)(&h); }
    ~VendorHandle() { if (h) FS_SP(Destroy)(h); }
};

// One handle for the process. Creating one per measurement would put library
// initialisation inside the timed region on the first call of every case.
FS_SP_HANDLE vendor() {
    static VendorHandle g;
    return g.h;
}

std::string fail(const char* call, int status) {
    return std::string(FS_SP_NAME) + " " + call + " returned status " +
           std::to_string(status);
}

// Vendor scratch. Owned here and freed on scope exit; the allocation is made
// before the timed region so the allocator never lands in a sample.
struct Scratch {
    void* p = nullptr;
    ~Scratch() { if (p) ad::device_free(reinterpret_cast<ad::DevicePtr>(p)); }
    // Give the pointer up without freeing it -- used when C is handed to the
    // caller for verification instead of being destroyed here.
    void* release() { void* q = p; p = nullptr; return q; }
    bool grab(std::size_t bytes) {
        if (bytes == 0) return true;
        ad::DevicePtr d = 0;
        if (ad::device_malloc(&d, bytes) != FLAGSPARSE_STATUS_SUCCESS) return false;
        p = reinterpret_cast<void*>(d);
        return true;
    }
};

bool map_dtype(flagsparseDataType_t t, FS_DT_TYPE* out) {
    switch (t) {
        // Half is mapped for the operators that take it (gather/scatter move
        // bytes and do no arithmetic). An operator whose vendor entry point
        // rejects fp16 returns its own status, which becomes a blank speedup
        // with that reason -- better than declining here and reporting the
        // dtype as unmappable when it is the operator, not the type.
        case FLAGSPARSE_R_16F: *out = FS_DT(R_16F); return true;
        case FLAGSPARSE_R_32F: *out = FS_DT(R_32F); return true;
        case FLAGSPARSE_R_64F: *out = FS_DT(R_64F); return true;
        case FLAGSPARSE_C_32F: *out = FS_DT(C_32F); return true;
        case FLAGSPARSE_C_64F: *out = FS_DT(C_64F); return true;
        default: return false;
    }
}

// The compute type cuSPARSE wants is the operand type for these operators; it is
// a separate argument only because mixed-precision modes exist, which we do not
// exercise here.
bool map_op(flagsparseOperation_t op, FS_SP_OPERATION* out) {
    switch (op) {
        case FLAGSPARSE_OPERATION_NON_TRANSPOSE:
            *out = FS_SP_E(OPERATION_NON_TRANSPOSE); return true;
        case FLAGSPARSE_OPERATION_TRANSPOSE:
            *out = FS_SP_E(OPERATION_TRANSPOSE); return true;
        case FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
            *out = FS_SP_E(OPERATION_CONJUGATE_TRANSPOSE); return true;
        default: return false;
    }
}

// Builds a CSR or a COO descriptor from the same operand struct; A.is_coo()
// decides which. Both formats carry column indices in A.indices.
Status make_csr(const DeviceCsr& A, FS_SP_SPMAT* out) {
    FS_DT_TYPE vt;
    if (!map_dtype(A.dtype, &vt)) return Status::no("baseline: unmapped value type");
    if (A.is_coo()) {
        const auto s = FS_SP(CreateCoo)(out, A.rows, A.cols, A.nnz, A.coo_rows,
                                        A.indices, A.values, FS_SP_E(INDEX_32I),
                                        FS_SP_E(INDEX_BASE_ZERO), vt);
        if (s != FS_SP_E(STATUS_SUCCESS)) return Status::no(fail("CreateCoo", s));
        return Status::good();
    }
    const auto s = FS_SP(CreateCsr)(
        out, A.rows, A.cols, A.nnz, A.indptr, A.indices, A.values,
        FS_SP_E(INDEX_32I), FS_SP_E(INDEX_32I), FS_SP_E(INDEX_BASE_ZERO), vt);
    if (s != FS_SP_E(STATUS_SUCCESS)) return Status::no(fail("CreateCsr", s));
    return Status::good();
}

// Warm up, then take `iters` samples and return the median -- the same shape as
// BenchReport::measure, so the two numbers can be divided.
template <typename F>
Status timed(F&& once, int warmup, int iters, Timing* out) {
    if (Status s = once(); !s.ok) return s;
    for (int i = 0; i < warmup; ++i) once();
    ad::synchronize();
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        if (Status s = once(); !s.ok) return s;
        ad::synchronize();
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    out->median_ms = samples[samples.size() / 2];
    return Status::good();
}

}  // namespace

bool available() { return true; }
const char* name() { return FS_SP_NAME; }

// ------------------------------------------------------------------- SpMV

Status spmv_csr(const DeviceCsr& A, const void* x, void* y, const void* alpha,
                const void* beta, flagsparseOperation_t op, int warmup, int iters,
                Timing* out) {
    FS_DT_TYPE vt;
    if (!map_dtype(A.dtype, &vt)) return Status::no("baseline: unmapped value type");
    FS_SP_OPERATION vop;
    if (!map_op(op, &vop)) return Status::no("baseline: unmapped operation");

    FS_SP_SPMAT mat = nullptr;
    if (Status s = make_csr(A, &mat); !s.ok) return s;
    const int64_t xlen = (op == FLAGSPARSE_OPERATION_NON_TRANSPOSE) ? A.cols : A.rows;
    const int64_t ylen = (op == FLAGSPARSE_OPERATION_NON_TRANSPOSE) ? A.rows : A.cols;

    FS_SP_DNVEC vx = nullptr, vy = nullptr;
    FS_SP(CreateDnVec)(&vx, xlen, const_cast<void*>(x), vt);
    FS_SP(CreateDnVec)(&vy, ylen, y, vt);

    std::size_t bytes = 0;
    auto st = FS_SP(SpMV_bufferSize)(vendor(), vop, alpha, mat, vx, beta, vy, vt,
                                     FS_SP_E(SPMV_ALG_DEFAULT), &bytes);
    Scratch buf;
    Status result = Status::good();
    if (st != FS_SP_E(STATUS_SUCCESS)) {
        result = Status::no(fail("SpMV_bufferSize", st));
    } else if (!buf.grab(bytes)) {
        result = Status::no("baseline: SpMV scratch allocation failed");
    } else {
        result = timed(
            [&]() -> Status {
                const auto s = FS_SP(SpMV)(vendor(), vop, alpha, mat, vx, beta, vy,
                                           vt, FS_SP_E(SPMV_ALG_DEFAULT), buf.p);
                return s == FS_SP_E(STATUS_SUCCESS) ? Status::good()
                                                    : Status::no(fail("SpMV", s));
            },
            warmup, iters, out);
    }
    FS_SP(DestroyDnVec)(vx);
    FS_SP(DestroyDnVec)(vy);
    FS_SP(DestroySpMat)(mat);
    return result;
}

// ------------------------------------------------------------------- SpMM

Status spmm_csr(const DeviceCsr& A, const void* B, int64_t n, int64_t ldb, void* C,
                int64_t ldc, const void* alpha, const void* beta,
                flagsparseOperation_t opA, flagsparseOperation_t opB, int warmup,
                int iters, Timing* out) {
    FS_DT_TYPE vt;
    if (!map_dtype(A.dtype, &vt)) return Status::no("baseline: unmapped value type");
    FS_SP_OPERATION va, vb;
    if (!map_op(opA, &va) || !map_op(opB, &vb))
        return Status::no("baseline: unmapped operation");

    FS_SP_SPMAT mat = nullptr;
    if (Status s = make_csr(A, &mat); !s.ok) return s;
    const int64_t k = (opA == FLAGSPARSE_OPERATION_NON_TRANSPOSE) ? A.cols : A.rows;
    const int64_t m = (opA == FLAGSPARSE_OPERATION_NON_TRANSPOSE) ? A.rows : A.cols;

    FS_SP_DNMAT mb = nullptr, mc = nullptr;
    FS_SP(CreateDnMat)(&mb, k, n, ldb, const_cast<void*>(B), vt, FS_SP_E(ORDER_COL));
    FS_SP(CreateDnMat)(&mc, m, n, ldc, C, vt, FS_SP_E(ORDER_COL));

    std::size_t bytes = 0;
    auto st = FS_SP(SpMM_bufferSize)(vendor(), va, vb, alpha, mat, mb, beta, mc, vt,
                                     FS_SP_E(SPMM_ALG_DEFAULT), &bytes);
    Scratch buf;
    Status result = Status::good();
    if (st != FS_SP_E(STATUS_SUCCESS)) {
        result = Status::no(fail("SpMM_bufferSize", st));
    } else if (!buf.grab(bytes)) {
        result = Status::no("baseline: SpMM scratch allocation failed");
    } else {
        result = timed(
            [&]() -> Status {
                const auto s = FS_SP(SpMM)(vendor(), va, vb, alpha, mat, mb, beta,
                                           mc, vt, FS_SP_E(SPMM_ALG_DEFAULT), buf.p);
                return s == FS_SP_E(STATUS_SUCCESS) ? Status::good()
                                                    : Status::no(fail("SpMM", s));
            },
            warmup, iters, out);
    }
    FS_SP(DestroyDnMat)(mb);
    FS_SP(DestroyDnMat)(mc);
    FS_SP(DestroySpMat)(mat);
    return result;
}

// ------------------------------------------------------------------ SDDMM
//
// C_sparse = alpha * (B @ D) .* spy(A) + beta * C_sparse. A carries the sparsity
// pattern AND receives the result, which is why it is both operands here.

Status sddmm_csr(const DeviceCsr& A, const void* Bd, int64_t k, int64_t ldb,
                 const void* Cd, int64_t ldc, const void* alpha, const void* beta,
                 int warmup, int iters, Timing* out) {
    FS_DT_TYPE vt;
    if (!map_dtype(A.dtype, &vt)) return Status::no("baseline: unmapped value type");

    FS_SP_SPMAT mat = nullptr;
    if (Status s = make_csr(A, &mat); !s.ok) return s;

    FS_SP_DNMAT mb = nullptr, md = nullptr;
    FS_SP(CreateDnMat)(&mb, A.rows, k, ldb, const_cast<void*>(Bd), vt,
                       FS_SP_E(ORDER_ROW));
    FS_SP(CreateDnMat)(&md, k, A.cols, ldc, const_cast<void*>(Cd), vt,
                       FS_SP_E(ORDER_ROW));

    std::size_t bytes = 0;
    auto st = FS_SP(SDDMM_bufferSize)(vendor(), FS_SP_E(OPERATION_NON_TRANSPOSE),
                                      FS_SP_E(OPERATION_NON_TRANSPOSE), alpha, mb, md,
                                      beta, mat, vt, FS_SP_E(SDDMM_ALG_DEFAULT),
                                      &bytes);
    Scratch buf;
    Status result = Status::good();
    if (st != FS_SP_E(STATUS_SUCCESS)) {
        result = Status::no(fail("SDDMM_bufferSize", st));
    } else if (!buf.grab(bytes)) {
        result = Status::no("baseline: SDDMM scratch allocation failed");
    } else {
        // preprocess is analysis, not part of the per-call cost: cuSPARSE expects
        // it once per pattern. Timing it every iteration would charge SDDMM for
        // work a real caller pays once.
        FS_SP(SDDMM_preprocess)(vendor(), FS_SP_E(OPERATION_NON_TRANSPOSE),
                                FS_SP_E(OPERATION_NON_TRANSPOSE), alpha, mb, md, beta,
                                mat, vt, FS_SP_E(SDDMM_ALG_DEFAULT), buf.p);
        result = timed(
            [&]() -> Status {
                const auto s = FS_SP(SDDMM)(
                    vendor(), FS_SP_E(OPERATION_NON_TRANSPOSE),
                    FS_SP_E(OPERATION_NON_TRANSPOSE), alpha, mb, md, beta, mat, vt,
                    FS_SP_E(SDDMM_ALG_DEFAULT), buf.p);
                return s == FS_SP_E(STATUS_SUCCESS) ? Status::good()
                                                    : Status::no(fail("SDDMM", s));
            },
            warmup, iters, out);
    }
    FS_SP(DestroyDnMat)(mb);
    FS_SP(DestroyDnMat)(md);
    FS_SP(DestroySpMat)(mat);
    return result;
}

// ----------------------------------------------------------------- SpGEMM
//
// C = alpha * A * A. C's dimensions are discovered, not known up front, so this
// owns C's three buffers and frees them before returning. The timed region is
// `compute` + `copy`: workEstimation and the first compute pass size the
// scratch, which a caller pays once per pattern, while copy materialises the
// result and is unavoidable per product.

void free_csr(BaselineCsrOut* c) {
    if (!c) return;
    for (void** p : {&c->indptr, &c->indices, &c->values}) {
        if (*p) ad::device_free(reinterpret_cast<ad::DevicePtr>(*p));
        *p = nullptr;
    }
    c->nnz = 0;
}

Status spgemm_csr(const DeviceCsr& A, const void* alpha, const void* beta,
                  int warmup, int iters, Timing* out, BaselineCsrOut* result) {
    if (A.rows != A.cols) return Status::no("A*A needs a square A");
    if (A.is_coo()) return Status::no(std::string(FS_SP_NAME) + " SpGEMM is CSR only");
    FS_DT_TYPE vt;
    if (!map_dtype(A.dtype, &vt)) return Status::no("baseline: unmapped value type");

    const auto NT = FS_SP_E(OPERATION_NON_TRANSPOSE);
    FS_SP_SPMAT a = nullptr, b = nullptr, c = nullptr;
    if (Status s = make_csr(A, &a); !s.ok) return s;
    if (Status s = make_csr(A, &b); !s.ok) { FS_SP(DestroySpMat)(a); return s; }

    // C starts empty: cuSPARSE fills its row pointers during compute and its
    // column/value arrays only once we have allocated them at the size it reports.
    Scratch c_ptr;
    if (!c_ptr.grab(static_cast<std::size_t>(A.rows + 1) * sizeof(int32_t))) {
        FS_SP(DestroySpMat)(a); FS_SP(DestroySpMat)(b);
        return Status::no("baseline: SpGEMM row-pointer allocation failed");
    }
    FS_SP(CreateCsr)(&c, A.rows, A.cols, 0, c_ptr.p, nullptr, nullptr,
                     FS_SP_E(INDEX_32I), FS_SP_E(INDEX_32I), FS_SP_E(INDEX_BASE_ZERO),
                     vt);

    FS_SP_SPGEMM_DESCR descr = nullptr;
    FS_SP(SpGEMM_createDescr)(&descr);

    auto cleanup = [&](Status s) {
        FS_SP(SpGEMM_destroyDescr)(descr);
        FS_SP(DestroySpMat)(a); FS_SP(DestroySpMat)(b); FS_SP(DestroySpMat)(c);
        return s;
    };

    std::size_t b1 = 0, b2 = 0;
    auto st = FS_SP(SpGEMM_workEstimation)(vendor(), NT, NT, alpha, a, b, beta, c, vt,
                                           FS_SP_E(SPGEMM_DEFAULT), descr, &b1,
                                           nullptr);
    if (st != FS_SP_E(STATUS_SUCCESS)) return cleanup(Status::no(fail("SpGEMM_workEstimation", st)));
    Scratch s1;
    if (!s1.grab(b1)) return cleanup(Status::no("baseline: SpGEMM scratch 1 allocation failed"));
    st = FS_SP(SpGEMM_workEstimation)(vendor(), NT, NT, alpha, a, b, beta, c, vt,
                                      FS_SP_E(SPGEMM_DEFAULT), descr, &b1, s1.p);
    if (st != FS_SP_E(STATUS_SUCCESS)) return cleanup(Status::no(fail("SpGEMM_workEstimation", st)));

    st = FS_SP(SpGEMM_compute)(vendor(), NT, NT, alpha, a, b, beta, c, vt,
                               FS_SP_E(SPGEMM_DEFAULT), descr, &b2, nullptr);
    if (st != FS_SP_E(STATUS_SUCCESS)) return cleanup(Status::no(fail("SpGEMM_compute", st)));
    Scratch s2;
    if (!s2.grab(b2)) return cleanup(Status::no("baseline: SpGEMM scratch 2 allocation failed"));
    st = FS_SP(SpGEMM_compute)(vendor(), NT, NT, alpha, a, b, beta, c, vt,
                               FS_SP_E(SPGEMM_DEFAULT), descr, &b2, s2.p);
    if (st != FS_SP_E(STATUS_SUCCESS)) return cleanup(Status::no(fail("SpGEMM_compute", st)));

    int64_t cr = 0, cc = 0, cnnz = 0;
    FS_SP(SpMatGetSize)(c, &cr, &cc, &cnnz);
    Scratch c_ind, c_val;
    if (!c_ind.grab(static_cast<std::size_t>(cnnz) * sizeof(int32_t)) ||
        !c_val.grab(static_cast<std::size_t>(cnnz) *
                    (A.dtype == FLAGSPARSE_R_64F   ? 8
                     : A.dtype == FLAGSPARSE_C_32F ? 8
                     : A.dtype == FLAGSPARSE_C_64F ? 16
                                                   : 4))) {
        return cleanup(Status::no("baseline: SpGEMM result allocation failed (" +
                                  std::to_string(cnnz) + " nonzeros)"));
    }
    FS_SP(CsrSetPointers)(c, c_ptr.p, c_ind.p, c_val.p);

    // One copy outside the clock, for the same reason as the operator side.
    st = FS_SP(SpGEMM_copy)(vendor(), NT, NT, alpha, a, b, beta, c, vt,
                            FS_SP_E(SPGEMM_DEFAULT), descr);
    if (st != FS_SP_E(STATUS_SUCCESS)) return cleanup(Status::no(fail("SpGEMM_copy", st)));

    Status result_status = timed(
        [&]() -> Status {
            const auto s = FS_SP(SpGEMM_compute)(vendor(), NT, NT, alpha, a, b, beta,
                                                 c, vt, FS_SP_E(SPGEMM_DEFAULT),
                                                 descr, &b2, s2.p);
            return s == FS_SP_E(STATUS_SUCCESS) ? Status::good()
                                                : Status::no(fail("SpGEMM_compute", s));
        },
        warmup, iters, out);

    // Hand C over instead of freeing it, when the caller asked. Scratch's
    // destructor must not run on these, so release them first.
    if (result && result_status.ok) {
        result->rows = cr;
        result->cols = cc;
        result->nnz = cnnz;
        result->indptr = c_ptr.release();
        result->indices = c_ind.release();
        result->values = c_val.release();
    }
    return cleanup(result_status);
}

// ------------------------------------------------------------------- SpSV
//
// analysis is NOT timed. It is the expensive part (a level schedule over the
// pattern) and cuSPARSE's API exists precisely so a caller pays it once and
// solves many right-hand sides. Timing it per call would compare our solve
// against their analysis+solve, which is not the same operation.

Status spsv_csr(const DeviceCsr& A, const void* x, void* y, const void* alpha,
                flagsparseFillMode_t fill, flagsparseDiagType_t diag,
                flagsparseOperation_t op, int warmup, int iters, Timing* out) {
    FS_DT_TYPE vt;
    if (!map_dtype(A.dtype, &vt)) return Status::no("baseline: unmapped value type");
    FS_SP_OPERATION vop;
    if (!map_op(op, &vop)) return Status::no("baseline: unmapped operation");

    FS_SP_SPMAT mat = nullptr;
    if (Status s = make_csr(A, &mat); !s.ok) return s;

    auto vfill = (fill == FLAGSPARSE_FILL_MODE_LOWER) ? FS_SP_E(FILL_MODE_LOWER)
                                                      : FS_SP_E(FILL_MODE_UPPER);
    auto vdiag = (diag == FLAGSPARSE_DIAG_TYPE_UNIT) ? FS_SP_E(DIAG_TYPE_UNIT)
                                                     : FS_SP_E(DIAG_TYPE_NON_UNIT);
    FS_SP(SpMatSetAttribute)(mat, FS_SP_E(SPMAT_FILL_MODE), &vfill, sizeof(vfill));
    FS_SP(SpMatSetAttribute)(mat, FS_SP_E(SPMAT_DIAG_TYPE), &vdiag, sizeof(vdiag));

    FS_SP_DNVEC vx = nullptr, vy = nullptr;
    FS_SP(CreateDnVec)(&vx, A.cols, const_cast<void*>(x), vt);
    FS_SP(CreateDnVec)(&vy, A.rows, y, vt);

    FS_SP_SPSV_DESCR descr = nullptr;
    FS_SP(SpSV_createDescr)(&descr);
    auto cleanup = [&](Status s) {
        FS_SP(SpSV_destroyDescr)(descr);
        FS_SP(DestroyDnVec)(vx); FS_SP(DestroyDnVec)(vy); FS_SP(DestroySpMat)(mat);
        return s;
    };

    std::size_t bytes = 0;
    auto st = FS_SP(SpSV_bufferSize)(vendor(), vop, alpha, mat, vx, vy, vt,
                                     FS_SP_E(SPSV_ALG_DEFAULT), descr, &bytes);
    if (st != FS_SP_E(STATUS_SUCCESS)) return cleanup(Status::no(fail("SpSV_bufferSize", st)));
    Scratch buf;
    if (!buf.grab(bytes)) return cleanup(Status::no("baseline: SpSV scratch allocation failed"));

    st = FS_SP(SpSV_analysis)(vendor(), vop, alpha, mat, vx, vy, vt,
                              FS_SP_E(SPSV_ALG_DEFAULT), descr, buf.p);
    if (st != FS_SP_E(STATUS_SUCCESS)) return cleanup(Status::no(fail("SpSV_analysis", st)));

    return cleanup(timed(
        [&]() -> Status {
            const auto s = FS_SP(SpSV_solve)(vendor(), vop, alpha, mat, vx, vy, vt,
                                             FS_SP_E(SPSV_ALG_DEFAULT), descr);
            return s == FS_SP_E(STATUS_SUCCESS) ? Status::good()
                                                : Status::no(fail("SpSV_solve", s));
        },
        warmup, iters, out));
}

// ------------------------------------------------------------------- SpSM

Status spsm_csr(const DeviceCsr& A, const void* B, int64_t n, int64_t ldb, void* C,
                int64_t ldc, const void* alpha, flagsparseFillMode_t fill,
                flagsparseDiagType_t diag, flagsparseOperation_t op, int warmup,
                int iters, Timing* out) {
    FS_DT_TYPE vt;
    if (!map_dtype(A.dtype, &vt)) return Status::no("baseline: unmapped value type");
    FS_SP_OPERATION vop;
    if (!map_op(op, &vop)) return Status::no("baseline: unmapped operation");

    FS_SP_SPMAT mat = nullptr;
    if (Status s = make_csr(A, &mat); !s.ok) return s;

    auto vfill = (fill == FLAGSPARSE_FILL_MODE_LOWER) ? FS_SP_E(FILL_MODE_LOWER)
                                                      : FS_SP_E(FILL_MODE_UPPER);
    auto vdiag = (diag == FLAGSPARSE_DIAG_TYPE_UNIT) ? FS_SP_E(DIAG_TYPE_UNIT)
                                                     : FS_SP_E(DIAG_TYPE_NON_UNIT);
    FS_SP(SpMatSetAttribute)(mat, FS_SP_E(SPMAT_FILL_MODE), &vfill, sizeof(vfill));
    FS_SP(SpMatSetAttribute)(mat, FS_SP_E(SPMAT_DIAG_TYPE), &vdiag, sizeof(vdiag));

    FS_SP_DNMAT mb = nullptr, mc = nullptr;
    FS_SP(CreateDnMat)(&mb, A.cols, n, ldb, const_cast<void*>(B), vt, FS_SP_E(ORDER_COL));
    FS_SP(CreateDnMat)(&mc, A.rows, n, ldc, C, vt, FS_SP_E(ORDER_COL));

    FS_SP_SPSM_DESCR descr = nullptr;
    FS_SP(SpSM_createDescr)(&descr);
    auto cleanup = [&](Status s) {
        FS_SP(SpSM_destroyDescr)(descr);
        FS_SP(DestroyDnMat)(mb); FS_SP(DestroyDnMat)(mc); FS_SP(DestroySpMat)(mat);
        return s;
    };

    std::size_t bytes = 0;
    auto st = FS_SP(SpSM_bufferSize)(vendor(), vop, FS_SP_E(OPERATION_NON_TRANSPOSE),
                                     alpha, mat, mb, mc, vt, FS_SP_E(SPSM_ALG_DEFAULT),
                                     descr, &bytes);
    if (st != FS_SP_E(STATUS_SUCCESS)) return cleanup(Status::no(fail("SpSM_bufferSize", st)));
    Scratch buf;
    if (!buf.grab(bytes)) return cleanup(Status::no("baseline: SpSM scratch allocation failed"));

    st = FS_SP(SpSM_analysis)(vendor(), vop, FS_SP_E(OPERATION_NON_TRANSPOSE), alpha,
                              mat, mb, mc, vt, FS_SP_E(SPSM_ALG_DEFAULT), descr, buf.p);
    if (st != FS_SP_E(STATUS_SUCCESS)) return cleanup(Status::no(fail("SpSM_analysis", st)));

    return cleanup(timed(
        [&]() -> Status {
            const auto s = FS_SP(SpSM_solve)(vendor(), vop,
                                             FS_SP_E(OPERATION_NON_TRANSPOSE), alpha,
                                             mat, mb, mc, vt,
                                             FS_SP_E(SPSM_ALG_DEFAULT), descr);
            return s == FS_SP_E(STATUS_SUCCESS) ? Status::good()
                                                : Status::no(fail("SpSM_solve", s));
        },
        warmup, iters, out));
}

// -------------------------------------------------------- gather / scatter

namespace {
Status make_spvec(int64_t nnz, int64_t size, const void* idx, void* val,
                  flagsparseDataType_t dtype, FS_SP_SPVEC* out) {
    FS_DT_TYPE vt;
    if (!map_dtype(dtype, &vt)) return Status::no("baseline: unmapped value type");
    const auto s = FS_SP(CreateSpVec)(out, size, nnz, const_cast<void*>(idx), val,
                                      FS_SP_E(INDEX_32I), FS_SP_E(INDEX_BASE_ZERO), vt);
    if (s != FS_SP_E(STATUS_SUCCESS)) return Status::no(fail("CreateSpVec", s));
    return Status::good();
}
}  // namespace

Status gather(const void* dense, void* sparse_val, const void* sparse_idx,
              int64_t nnz, int64_t size, flagsparseDataType_t dtype, int warmup,
              int iters, Timing* out) {
    FS_DT_TYPE vt;
    if (!map_dtype(dtype, &vt)) return Status::no("baseline: unmapped value type");
    FS_SP_SPVEC sv = nullptr;
    if (Status s = make_spvec(nnz, size, sparse_idx, sparse_val, dtype, &sv); !s.ok)
        return s;
    FS_SP_DNVEC dv = nullptr;
    FS_SP(CreateDnVec)(&dv, size, const_cast<void*>(dense), vt);

    Status result = timed(
        [&]() -> Status {
            const auto s = FS_SP(Gather)(vendor(), dv, sv);
            return s == FS_SP_E(STATUS_SUCCESS) ? Status::good()
                                                : Status::no(fail("Gather", s));
        },
        warmup, iters, out);
    FS_SP(DestroyDnVec)(dv);
    FS_SP(DestroySpVec)(sv);
    return result;
}

Status scatter(void* dense, const void* sparse_val, const void* sparse_idx,
               int64_t nnz, int64_t size, flagsparseDataType_t dtype, int warmup,
               int iters, Timing* out) {
    FS_DT_TYPE vt;
    if (!map_dtype(dtype, &vt)) return Status::no("baseline: unmapped value type");
    FS_SP_SPVEC sv = nullptr;
    if (Status s = make_spvec(nnz, size, sparse_idx, const_cast<void*>(sparse_val),
                              dtype, &sv); !s.ok)
        return s;
    FS_SP_DNVEC dv = nullptr;
    FS_SP(CreateDnVec)(&dv, size, dense, vt);

    Status result = timed(
        [&]() -> Status {
            const auto s = FS_SP(Scatter)(vendor(), sv, dv);
            return s == FS_SP_E(STATUS_SUCCESS) ? Status::good()
                                                : Status::no(fail("Scatter", s));
        },
        warmup, iters, out);
    FS_SP(DestroyDnVec)(dv);
    FS_SP(DestroySpVec)(sv);
    return result;
}

}  // namespace fstest::baseline
