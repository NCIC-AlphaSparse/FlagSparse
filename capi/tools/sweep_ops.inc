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


// One operator variant per function, all of them run against one loaded matrix.
// Included by matrix_sweep.cpp, which owns the helpers these use.

namespace {

constexpr int64_t kDenseCols = 32;   // SpMM / SDDMM width
constexpr int64_t kRhs = 8;          // SpSM right-hand sides

const flagsparseDataType_t kAllDtypes[4] = {
    FLAGSPARSE_R_32F, FLAGSPARSE_R_64F, FLAGSPARSE_C_32F, FLAGSPARSE_C_64F};
const flagsparseDataType_t kRealDtypes[2] = {FLAGSPARSE_R_32F, FLAGSPARSE_R_64F};

bool wanted(const std::vector<std::string>& only, const char* op) {
    return only.empty() || std::find(only.begin(), only.end(), op) != only.end();
}

// Scalars at the compute type's width. Reading a double's first four bytes as a
// float gives 0, which SpGEMM (alpha == 1 only) would refuse for the wrong
// reason -- so the width has to follow the dtype, not the host's convenience.
struct Scalars {
    double d[2];
    float f[2];
    const void* of(flagsparseDataType_t t) const {
        return (t == FLAGSPARSE_R_64F || t == FLAGSPARSE_C_64F)
                   ? static_cast<const void*>(d) : static_cast<const void*>(f);
    }
};
Scalars make_scalar(double re, double im) {
    Scalars s;
    s.d[0] = re; s.d[1] = im;
    s.f[0] = static_cast<float>(re); s.f[1] = static_cast<float>(im);
    return s;
}

// ------------------------------------------------------------ gather/scatter

void sweep_gather_scatter(flagsparseHandle_t h, const std::string& name,
                          const mm::Csr& A, const std::vector<std::string>& only) {
    const bool g = wanted(only, "gather"), s = wanted(only, "scatter");
    if (!g && !s) return;
    // The matrix supplies the index pattern: its column indices are exactly the
    // kind of scattered, structured access a real gather sees.
    //
    // A sparse vector cannot hold more entries than its length, and a corpus
    // matrix has far more nonzeros than columns, so the count is capped. The
    // indices still come from the matrix, which is the point -- a synthetic
    // random pattern would not have its locality.
    const int64_t dense = std::max<int64_t>(A.cols, 1);
    const int64_t nnz = std::min<int64_t>(A.nnz, dense);
    for (flagsparseDataType_t t : kAllDtypes) {
        const double bytes = static_cast<double>(nnz + dense) * value_bytes(t) * 2;
        for (int which = 0; which < 2; ++which) {
            if ((which == 0 && !g) || (which == 1 && !s)) continue;
            Row row = base(name, A, which == 0 ? "gather" : "scatter", "spvec", t);
            if (over_budget(bytes)) { skip(row, "skipped_memory", "vectors exceed budget"); continue; }
            Buf d_dense = filled(static_cast<std::size_t>(dense), t, 1.0);
            Buf d_sparse = filled(static_cast<std::size_t>(nnz), t, 1.0);
            Buf d_idx = upload(std::vector<int32_t>(
                A.indices.begin(), A.indices.begin() + static_cast<long>(nnz)));
            if (!d_dense.valid() || !d_sparse.valid() || !d_idx.valid()) {
                skip(row, "skipped_memory", "cudaMalloc failed"); continue;
            }
            flagsparseDnVecDescr_t dn = nullptr;
            flagsparseSpVecDescr_t sp = nullptr;
            flagsparseCreateDnVec(&dn, dense, d_dense.get(), t);
            flagsparseCreateSpVec(&sp, dense, nnz, d_idx.get(), d_sparse.get(),
                                  FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO, t);
            record(std::move(row),
                   [&] {
                       return which == 0 ? flagsparseGather(h, dn, sp)
                                         : flagsparseScatter(h, sp, dn);
                   },
                   0.0, h);
            flagsparseDestroySpVec(sp);
            flagsparseDestroyDnVec(dn);
        }
    }
}

// ------------------------------------------------------------------- SpMV

void sweep_spmv(flagsparseHandle_t h, const std::string& name, const mm::Csr& A,
                const std::vector<std::string>& only) {
    if (!wanted(only, "spmv")) return;
    const Scalars alpha = make_scalar(1.0, 0.0), beta = make_scalar(0.0, 0.0);
    for (int fmt = 0; fmt < 2; ++fmt) {
        const char* fname = fmt ? "coo" : "csr";
        for (flagsparseDataType_t t : kAllDtypes) {
            Row row = base(name, A, "spmv", fname, t);
            const double bytes = static_cast<double>(A.nnz) * value_bytes(t) +
                                 static_cast<double>(A.rows + A.cols) * value_bytes(t);
            if (over_budget(bytes)) { skip(row, "skipped_memory", "operands exceed budget"); continue; }
            Buf d_val = values_for(A.values, t);
            Buf d_col = upload(A.indices);
            Buf d_ptr = upload(A.indptr);
            Buf d_row = upload(coo_rows_of(A));
            Buf d_x = filled(static_cast<std::size_t>(A.cols), t, 1.0);
            Buf d_y = filled(static_cast<std::size_t>(A.rows), t, 0.0);
            if (!d_val.valid() || !d_x.valid() || !d_y.valid() || !d_row.valid()) {
                skip(row, "skipped_memory", "cudaMalloc failed"); continue;
            }
            flagsparseSpMatDescr_t mat = nullptr;
            const flagsparseStatus_t made =
                fmt ? flagsparseCreateCoo(&mat, A.rows, A.cols, A.nnz, d_row.get(),
                                          d_col.get(), d_val.get(), FLAGSPARSE_INDEX_32I,
                                          FLAGSPARSE_INDEX_BASE_ZERO, t)
                    : flagsparseCreateCsr(&mat, A.rows, A.cols, A.nnz, d_ptr.get(),
                                          d_col.get(), d_val.get(), FLAGSPARSE_INDEX_32I,
                                          FLAGSPARSE_INDEX_32I,
                                          FLAGSPARSE_INDEX_BASE_ZERO, t);
            if (made != FLAGSPARSE_STATUS_SUCCESS) { skip(row, "failed", "descriptor"); continue; }
            flagsparseDnVecDescr_t x = nullptr, y = nullptr;
            flagsparseCreateDnVec(&x, A.cols, d_x.get(), t);
            flagsparseCreateDnVec(&y, A.rows, d_y.get(), t);
            const auto NT = FLAGSPARSE_OPERATION_NON_TRANSPOSE;
            const auto ALG = FLAGSPARSE_SPMV_ALG_DEFAULT;
            std::size_t bs = 0;
            flagsparseSpMV_bufferSize(h, NT, alpha.of(t), mat, x, beta.of(t), y, t, ALG, &bs);
            Buf scratch(bs);
            // Preprocess outside the clock: it is the per-matrix step, and for
            // COO it builds the row-offsets array the segment route runs on.
            flagsparseSpMV_preprocess(h, NT, alpha.of(t), mat, x, beta.of(t), y, t, ALG,
                                      scratch.get());
            record(std::move(row),
                   [&] {
                       return flagsparseSpMV(h, NT, alpha.of(t), mat, x, beta.of(t), y, t,
                                             ALG, scratch.get());
                   },
                   2.0 * static_cast<double>(A.nnz), h);
            flagsparseDestroyDnVec(y); flagsparseDestroyDnVec(x);
            flagsparseDestroySpMat(mat);
        }
    }
}

// ------------------------------------------------------------------- SpMM

void sweep_spmm(flagsparseHandle_t h, const std::string& name, const mm::Csr& A,
                const std::vector<std::string>& only) {
    if (!wanted(only, "spmm")) return;
    const Scalars alpha = make_scalar(1.0, 0.0), beta = make_scalar(0.0, 0.0);
    for (int fmt = 0; fmt < 2; ++fmt) {
        const char* fname = fmt ? "coo" : "csr";
        for (flagsparseDataType_t t : kAllDtypes) {
            Row row = base(name, A, "spmm", fname, t);
            const double dense_bytes =
                static_cast<double>(A.rows + A.cols) * kDenseCols * value_bytes(t);
            const double bytes = dense_bytes + static_cast<double>(A.nnz) * value_bytes(t);
            if (over_budget(bytes)) { skip(row, "skipped_memory", "dense operands exceed budget"); continue; }
            Buf d_val = values_for(A.values, t);
            Buf d_col = upload(A.indices);
            Buf d_ptr = upload(A.indptr);
            Buf d_row = upload(coo_rows_of(A));
            Buf d_b = filled(static_cast<std::size_t>(A.cols * kDenseCols), t, 1.0);
            Buf d_c = filled(static_cast<std::size_t>(A.rows * kDenseCols), t, 0.0);
            if (!d_val.valid() || !d_b.valid() || !d_c.valid() || !d_row.valid()) {
                skip(row, "skipped_memory", "cudaMalloc failed"); continue;
            }
            flagsparseSpMatDescr_t mat = nullptr;
            const flagsparseStatus_t made =
                fmt ? flagsparseCreateCoo(&mat, A.rows, A.cols, A.nnz, d_row.get(),
                                          d_col.get(), d_val.get(), FLAGSPARSE_INDEX_32I,
                                          FLAGSPARSE_INDEX_BASE_ZERO, t)
                    : flagsparseCreateCsr(&mat, A.rows, A.cols, A.nnz, d_ptr.get(),
                                          d_col.get(), d_val.get(), FLAGSPARSE_INDEX_32I,
                                          FLAGSPARSE_INDEX_32I,
                                          FLAGSPARSE_INDEX_BASE_ZERO, t);
            if (made != FLAGSPARSE_STATUS_SUCCESS) { skip(row, "failed", "descriptor"); continue; }
            flagsparseDnMatDescr_t B = nullptr, C = nullptr;
            flagsparseCreateDnMat(&B, A.cols, kDenseCols, kDenseCols, d_b.get(), t,
                                  FLAGSPARSE_ORDER_ROW);
            flagsparseCreateDnMat(&C, A.rows, kDenseCols, kDenseCols, d_c.get(), t,
                                  FLAGSPARSE_ORDER_ROW);
            const auto NT = FLAGSPARSE_OPERATION_NON_TRANSPOSE;
            const auto ALG = FLAGSPARSE_SPMM_ALG_DEFAULT;
            std::size_t bs = 0;
            flagsparseSpMM_bufferSize(h, NT, NT, alpha.of(t), mat, B, beta.of(t), C, t, ALG, &bs);
            Buf scratch(bs);
            flagsparseSpMM_preprocess(h, NT, NT, alpha.of(t), mat, B, beta.of(t), C, t, ALG,
                                      scratch.get());
            record(std::move(row),
                   [&] {
                       return flagsparseSpMM(h, NT, NT, alpha.of(t), mat, B, beta.of(t), C,
                                             t, ALG, scratch.get());
                   },
                   2.0 * static_cast<double>(A.nnz) * kDenseCols, h);
            flagsparseDestroyDnMat(C); flagsparseDestroyDnMat(B);
            flagsparseDestroySpMat(mat);
        }
    }
}

// --------------------------------------------------------------- SpSV/SpSM

void sweep_solves(flagsparseHandle_t h, const std::string& name, const mm::Csr& A,
                  const std::vector<std::string>& only) {
    const bool want_sv = wanted(only, "spsv"), want_sm = wanted(only, "spsm");
    if (!want_sv && !want_sm) return;
    const mm::Csr L = lower_triangle(A);
    const Scalars alpha = make_scalar(1.0, 0.0);

    Buf d_col = upload(L.indices);
    Buf d_ptr = upload(L.indptr);
    Buf d_row = upload(coo_rows_of(L));

    for (flagsparseDataType_t t : kAllDtypes) {
        Buf d_val = values_for(L.values, t);
        for (int fmt = 0; fmt < 2; ++fmt) {
            const char* fname = fmt ? "coo" : "csr";
            if (want_sv) {
                Row row = base(name, L, "spsv", fname, t);
                const double bytes = static_cast<double>(L.nnz + 2 * L.rows) * value_bytes(t);
                if (over_budget(bytes) || !d_val.valid()) {
                    skip(row, "skipped_memory", "operands exceed budget");
                } else {
                    Buf d_x = filled(static_cast<std::size_t>(L.rows), t, 1.0);
                    Buf d_y = filled(static_cast<std::size_t>(L.rows), t, 0.0);
                    flagsparseSpMatDescr_t mat = nullptr;
                    const flagsparseStatus_t made =
                        fmt ? flagsparseCreateCoo(&mat, L.rows, L.cols, L.nnz, d_row.get(),
                                                  d_col.get(), d_val.get(),
                                                  FLAGSPARSE_INDEX_32I,
                                                  FLAGSPARSE_INDEX_BASE_ZERO, t)
                            : flagsparseCreateCsr(&mat, L.rows, L.cols, L.nnz, d_ptr.get(),
                                                  d_col.get(), d_val.get(),
                                                  FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                                  FLAGSPARSE_INDEX_BASE_ZERO, t);
                    if (made != FLAGSPARSE_STATUS_SUCCESS) {
                        skip(std::move(row), "failed", "descriptor");
                    } else {
                        const flagsparseFillMode_t lo = FLAGSPARSE_FILL_MODE_LOWER;
                        const flagsparseDiagType_t nu = FLAGSPARSE_DIAG_TYPE_NON_UNIT;
                        flagsparseSpMatSetAttribute(mat, FLAGSPARSE_SPMAT_FILL_MODE, &lo, sizeof lo);
                        flagsparseSpMatSetAttribute(mat, FLAGSPARSE_SPMAT_DIAG_TYPE, &nu, sizeof nu);
                        flagsparseDnVecDescr_t x = nullptr, y = nullptr;
                        flagsparseCreateDnVec(&x, L.rows, d_x.get(), t);
                        flagsparseCreateDnVec(&y, L.rows, d_y.get(), t);
                        flagsparseSpSVDescr_t descr = nullptr;
                        flagsparseSpSV_createDescr(&descr);
                        const auto NT = FLAGSPARSE_OPERATION_NON_TRANSPOSE;
                        const auto ALG = FLAGSPARSE_SPSV_ALG_DEFAULT;
                        std::size_t bs = 0;
                        flagsparseSpSV_bufferSize(h, NT, alpha.of(t), mat, x, y, t, ALG, descr, &bs);
                        Buf scratch(bs);
                        const flagsparseStatus_t an =
                            flagsparseSpSV_analysis(h, NT, alpha.of(t), mat, x, y, t, ALG,
                                                    descr, scratch.get());
                        if (an != FLAGSPARSE_STATUS_SUCCESS) {
                            const char* nm = "?"; flagsparseGetErrorName(an, &nm);
                            skip(std::move(row),
                                 an == FLAGSPARSE_STATUS_NOT_SUPPORTED ? "not_supported" : "failed",
                                 std::string("analysis: ") + nm);
                        } else {
                            record(std::move(row),
                                   [&] {
                                       return flagsparseSpSV_solve(h, NT, alpha.of(t), mat, x,
                                                                   y, t, ALG, descr);
                                   },
                                   2.0 * static_cast<double>(L.nnz), h);
                        }
                        flagsparseSpSV_destroyDescr(descr);
                        flagsparseDestroyDnVec(y); flagsparseDestroyDnVec(x);
                        flagsparseDestroySpMat(mat);
                    }
                }
            }
            // SpSM is listed for CSR and real dtypes only.
            if (want_sm && fmt == 0 && !is_cplx(t)) {
                Row row = base(name, L, "spsm", "csr", t);
                const double bytes = static_cast<double>(L.nnz) * value_bytes(t) +
                                     static_cast<double>(L.rows) * kRhs * value_bytes(t) * 3;
                if (over_budget(bytes) || !d_val.valid()) {
                    skip(row, "skipped_memory", "dense operands exceed budget");
                } else {
                    Buf d_b = filled(static_cast<std::size_t>(L.rows * kRhs), t, 1.0);
                    Buf d_c = filled(static_cast<std::size_t>(L.rows * kRhs), t, 0.0);
                    flagsparseSpMatDescr_t mat = nullptr;
                    flagsparseCreateCsr(&mat, L.rows, L.cols, L.nnz, d_ptr.get(), d_col.get(),
                                        d_val.get(), FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                        FLAGSPARSE_INDEX_BASE_ZERO, t);
                    const flagsparseFillMode_t lo = FLAGSPARSE_FILL_MODE_LOWER;
                    const flagsparseDiagType_t nu = FLAGSPARSE_DIAG_TYPE_NON_UNIT;
                    flagsparseSpMatSetAttribute(mat, FLAGSPARSE_SPMAT_FILL_MODE, &lo, sizeof lo);
                    flagsparseSpMatSetAttribute(mat, FLAGSPARSE_SPMAT_DIAG_TYPE, &nu, sizeof nu);
                    flagsparseDnMatDescr_t B = nullptr, C = nullptr;
                    flagsparseCreateDnMat(&B, L.rows, kRhs, kRhs, d_b.get(), t, FLAGSPARSE_ORDER_ROW);
                    flagsparseCreateDnMat(&C, L.rows, kRhs, kRhs, d_c.get(), t, FLAGSPARSE_ORDER_ROW);
                    flagsparseSpSMDescr_t descr = nullptr;
                    flagsparseSpSM_createDescr(&descr);
                    const auto NT = FLAGSPARSE_OPERATION_NON_TRANSPOSE;
                    const auto ALG = FLAGSPARSE_SPSM_ALG_DEFAULT;
                    std::size_t bs = 0;
                    flagsparseSpSM_bufferSize(h, NT, NT, alpha.of(t), mat, B, C, t, ALG, descr, &bs);
                    Buf scratch(bs);
                    const flagsparseStatus_t an =
                        flagsparseSpSM_analysis(h, NT, NT, alpha.of(t), mat, B, C, t, ALG,
                                                descr, scratch.get());
                    if (an != FLAGSPARSE_STATUS_SUCCESS) {
                        const char* nm = "?"; flagsparseGetErrorName(an, &nm);
                        skip(std::move(row),
                             an == FLAGSPARSE_STATUS_NOT_SUPPORTED ? "not_supported" : "failed",
                             std::string("analysis: ") + nm);
                    } else {
                        record(std::move(row),
                               [&] {
                                   return flagsparseSpSM_solve(h, NT, NT, alpha.of(t), mat, B,
                                                               C, t, ALG, descr);
                               },
                               2.0 * static_cast<double>(L.nnz) * kRhs, h);
                    }
                    flagsparseSpSM_destroyDescr(descr);
                    flagsparseDestroyDnMat(C); flagsparseDestroyDnMat(B);
                    flagsparseDestroySpMat(mat);
                }
            }
        }
    }
}

// ------------------------------------------------------------------ SDDMM

void sweep_sddmm(flagsparseHandle_t h, const std::string& name, const mm::Csr& A,
                 const std::vector<std::string>& only) {
    if (!wanted(only, "sddmm")) return;
    const Scalars alpha = make_scalar(1.0, 0.0), beta = make_scalar(0.0, 0.0);
    for (flagsparseDataType_t t : kRealDtypes) {
        Row row = base(name, A, "sddmm", "csr", t);
        const double bytes = static_cast<double>(A.nnz) * value_bytes(t) +
                             static_cast<double>(A.rows + A.cols) * kDenseCols * value_bytes(t);
        if (over_budget(bytes)) { skip(row, "skipped_memory", "dense operands exceed budget"); continue; }
        Buf d_val = values_for(A.values, t);
        Buf d_col = upload(A.indices);
        Buf d_ptr = upload(A.indptr);
        Buf d_x = filled(static_cast<std::size_t>(A.rows * kDenseCols), t, 0.5);
        Buf d_y = filled(static_cast<std::size_t>(kDenseCols * A.cols), t, 0.25);
        if (!d_val.valid() || !d_x.valid() || !d_y.valid()) {
            skip(row, "skipped_memory", "cudaMalloc failed"); continue;
        }
        flagsparseSpMatDescr_t C = nullptr;
        if (flagsparseCreateCsr(&C, A.rows, A.cols, A.nnz, d_ptr.get(), d_col.get(),
                                d_val.get(), FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                FLAGSPARSE_INDEX_BASE_ZERO, t) != FLAGSPARSE_STATUS_SUCCESS) {
            skip(row, "failed", "descriptor"); continue;
        }
        flagsparseDnMatDescr_t X = nullptr, Y = nullptr;
        flagsparseCreateDnMat(&X, A.rows, kDenseCols, kDenseCols, d_x.get(), t, FLAGSPARSE_ORDER_ROW);
        flagsparseCreateDnMat(&Y, kDenseCols, A.cols, A.cols, d_y.get(), t, FLAGSPARSE_ORDER_ROW);
        const auto NT = FLAGSPARSE_OPERATION_NON_TRANSPOSE;
        const auto ALG = FLAGSPARSE_SDDMM_ALG_DEFAULT;
        std::size_t bs = 0;
        flagsparseSDDMM_bufferSize(h, NT, NT, alpha.of(t), X, Y, beta.of(t), C, t, ALG, &bs);
        Buf scratch(bs);
        flagsparseSDDMM_preprocess(h, NT, NT, alpha.of(t), X, Y, beta.of(t), C, t, ALG, scratch.get());
        record(std::move(row),
               [&] {
                   return flagsparseSDDMM(h, NT, NT, alpha.of(t), X, Y, beta.of(t), C, t,
                                          ALG, scratch.get());
               },
               2.0 * static_cast<double>(A.nnz) * kDenseCols, h);
        flagsparseDestroyDnMat(Y); flagsparseDestroyDnMat(X);
        flagsparseDestroySpMat(C);
    }
}

// ------------------------------------------------------------------ SpGEMM

void sweep_spgemm(flagsparseHandle_t h, const std::string& name, const mm::Csr& A,
                  const std::vector<std::string>& only) {
    if (!wanted(only, "spgemm")) return;
    const Scalars one = make_scalar(1.0, 0.0), zero = make_scalar(0.0, 0.0);
    for (flagsparseDataType_t t : kRealDtypes) {
        Row row = base(name, A, "spgemm", "csr", t);
        if (A.rows != A.cols) { skip(row, "skipped_shape", "A*A needs a square A"); continue; }
        Buf d_val = values_for(A.values, t);
        Buf d_col = upload(A.indices);
        Buf d_ptr = upload(A.indptr);
        Buf c_ptr(static_cast<std::size_t>(A.rows + 1) * sizeof(int32_t));
        if (!d_val.valid() || !c_ptr.valid()) { skip(row, "skipped_memory", "cudaMalloc failed"); continue; }
        flagsparseSpMatDescr_t Am = nullptr, Cm = nullptr;
        flagsparseCreateCsr(&Am, A.rows, A.cols, A.nnz, d_ptr.get(), d_col.get(),
                            d_val.get(), FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                            FLAGSPARSE_INDEX_BASE_ZERO, t);
        flagsparseCreateCsr(&Cm, A.rows, A.cols, 0, c_ptr.get(), nullptr, nullptr,
                            FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                            FLAGSPARSE_INDEX_BASE_ZERO, t);
        flagsparseSpGEMMDescr_t descr = nullptr;
        flagsparseSpGEMM_createDescr(&descr);
        const auto NT = FLAGSPARSE_OPERATION_NON_TRANSPOSE;
        const auto ALG = FLAGSPARSE_SPGEMM_DEFAULT;
        std::size_t s1 = 0, s2 = 0;
        flagsparseSpGEMM_workEstimation(h, NT, NT, one.of(t), Am, Am, zero.of(t), Cm, t,
                                        ALG, descr, &s1, nullptr);
        Buf b1(s1);
        flagsparseStatus_t st = b1.valid()
            ? flagsparseSpGEMM_workEstimation(h, NT, NT, one.of(t), Am, Am, zero.of(t), Cm,
                                              t, ALG, descr, &s1, b1.get())
            : FLAGSPARSE_STATUS_ALLOC_FAILED;
        if (st == FLAGSPARSE_STATUS_SUCCESS) {
            flagsparseSpGEMM_compute(h, NT, NT, one.of(t), Am, Am, zero.of(t), Cm, t, ALG,
                                     descr, &s2, nullptr);
        }
        Buf b2(s2);
        // The whole flow is timed as one unit: the size of C is discovered by
        // compute, so the phases cannot be repeated independently here the way
        // the per-operator benchmark does it.
        if (st == FLAGSPARSE_STATUS_SUCCESS && b2.valid()) {
            st = flagsparseSpGEMM_compute(h, NT, NT, one.of(t), Am, Am, zero.of(t), Cm, t,
                                          ALG, descr, &s2, b2.get());
        }
        if (st != FLAGSPARSE_STATUS_SUCCESS) {
            const char* nm = "?"; flagsparseGetErrorName(st, &nm);
            const char* detail = ""; flagsparseGetLastErrorString(h, &detail);
            std::string d = std::string(nm) + (detail && *detail ? std::string(": ") + detail : "");
            if (d.size() > 160) d.resize(160);
            skip(std::move(row), st == FLAGSPARSE_STATUS_NOT_SUPPORTED ? "not_supported" : "failed", d);
        } else {
            int64_t cr = 0, cc = 0, cn = 0;
            flagsparseSpMatGetSize(Cm, &cr, &cc, &cn);
            Buf c_col(static_cast<std::size_t>(std::max<int64_t>(cn, 1)) * sizeof(int32_t));
            Buf c_val(static_cast<std::size_t>(std::max<int64_t>(cn, 1)) * value_bytes(t));
            if (!c_col.valid() || !c_val.valid()) {
                skip(std::move(row), "skipped_memory", "result C exceeds memory");
            } else {
                flagsparseCsrSetPointers(Cm, c_ptr.get(), c_col.get(), c_val.get());
                row.nnz = cn;   // report the DISCOVERED size, which is the result
                record(std::move(row),
                       [&] {
                           return flagsparseSpGEMM_copy(h, NT, NT, one.of(t), Am, Am,
                                                        zero.of(t), Cm, t, ALG, descr);
                       },
                       0.0, h);
            }
        }
        flagsparseSpGEMM_destroyDescr(descr);
        flagsparseDestroySpMat(Cm); flagsparseDestroySpMat(Am);
    }
}

void run_all_ops(flagsparseHandle_t h, const std::string& name, const mm::Csr& A,
                 const std::vector<std::string>& only) {
    sweep_gather_scatter(h, name, A, only);
    sweep_spmv(h, name, A, only);
    sweep_spmm(h, name, A, only);
    sweep_solves(h, name, A, only);
    sweep_sddmm(h, name, A, only);
    sweep_spgemm(h, name, A, only);
}

}  // namespace
