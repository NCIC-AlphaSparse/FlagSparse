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


// Accuracy tests for flagsparseSpMM, against an fp64 host reference.
//
// The dense operands are the point of this operator, so the layout axes are
// swept explicitly: row- and column-major, a padded leading dimension, and
// opB = TRANSPOSE. All four are the same kernel with different strides, which
// is exactly the claim worth testing.

#include <gtest/gtest.h>

#include <complex>
#include <vector>

#include "common.hpp"

using namespace fstest;

namespace {

struct Handle {
    flagsparseHandle_t h = nullptr;
    Handle() { flagsparseCreate(&h); }
    ~Handle() { if (h) flagsparseDestroy(h); }
};

struct RunResult {
    flagsparseStatus_t status = FLAGSPARSE_STATUS_SUCCESS;
    double strict_ratio = 0.0;
    double relaxed_ratio = 0.0;
    bool relaxed_used = false;
};

// How a dense matrix of logical extent rows x cols is laid out in memory.
struct Layout {
    flagsparseOrder_t order = FLAGSPARSE_ORDER_ROW;
    int64_t pad = 0;   // extra leading dimension beyond the minimum

    int64_t ld(int64_t rows, int64_t cols) const {
        return (order == FLAGSPARSE_ORDER_ROW ? cols : rows) + pad;
    }
    std::size_t elems(int64_t rows, int64_t cols) const {
        return static_cast<std::size_t>(
            (order == FLAGSPARSE_ORDER_ROW ? rows : cols) * ld(rows, cols));
    }
    std::size_t at(int64_t i, int64_t j, int64_t rows, int64_t cols) const {
        const int64_t l = ld(rows, cols);
        return static_cast<std::size_t>(order == FLAGSPARSE_ORDER_ROW ? i * l + j
                                                                      : i + j * l);
    }
};

struct Case {
    int64_t n = 32;                 // dense columns of C
    double alpha = 1.0;
    double beta = 0.0;
    Layout lb{};                    // layout of the B descriptor
    Layout lc{};                    // layout of the C descriptor
    flagsparseOperation_t opB = FLAGSPARSE_OPERATION_NON_TRANSPOSE;
    flagsparseSpMMAlg_t alg = FLAGSPARSE_SPMM_ALG_DEFAULT;
    flagsparseIndexType_t index_type = FLAGSPARSE_INDEX_32I;
    flagsparseFormat_t format = FLAGSPARSE_FORMAT_CSR;
    uint32_t seed = 7;
};

// CsrMatrix stores i32; the descriptor may be asked for i64, which is a
// different buffer, not a different code path (§4.5.1: index width never
// appears in a function name).
DeviceBuffer upload_indices(const std::vector<int32_t>& idx, flagsparseIndexType_t t) {
    if (t == FLAGSPARSE_INDEX_32I) return DeviceBuffer::from(idx);
    return DeviceBuffer::from(std::vector<int64_t>(idx.begin(), idx.end()));
}

// random_csr emits rows in order and columns ascending within a row, so this
// expansion is row-sorted -- which is what both cuSPARSE and this implementation
// require of a COO matrix.
std::vector<int32_t> coo_row_indices(const CsrMatrix& A) {
    std::vector<int32_t> row;
    row.reserve(static_cast<std::size_t>(A.nnz));
    for (int64_t r = 0; r < A.rows; ++r) {
        for (int32_t p = A.indptr[static_cast<std::size_t>(r)];
             p < A.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
            row.push_back(static_cast<int32_t>(r));
        }
    }
    return row;
}

// The B descriptor's extents: op(B) is always k x n, so a transposed opB means
// the descriptor itself is n x k.
void b_descr_extent(const Case& c, int64_t k, int64_t* rows, int64_t* cols) {
    const bool t = (c.opB != FLAGSPARSE_OPERATION_NON_TRANSPOSE);
    *rows = t ? c.n : k;
    *cols = t ? k : c.n;
}

// ------------------------------------------------------------------ real ---

template <typename T>
RunResult run_spmm(flagsparseHandle_t handle, const CsrMatrix& A,
                   flagsparseDataType_t dtype, const Case& c) {
    RunResult out;
    const int64_t m = A.rows, k = A.cols, n = c.n;
    int64_t b_rows = 0, b_cols = 0;
    b_descr_extent(c, k, &b_rows, &b_cols);

    std::mt19937 rng(c.seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    // The reference operands are row-major fp64; the device copies are the dtype
    // and layout under test, filled from the same numbers.
    std::vector<double> b_ref(static_cast<std::size_t>(k * n));
    std::vector<double> c_ref(static_cast<std::size_t>(m * n));
    for (auto& v : b_ref) v = dist(rng);
    for (auto& v : c_ref) v = dist(rng);

    std::vector<T> b_dev(c.lb.elems(b_rows, b_cols), T(0));
    std::vector<T> c_dev(c.lc.elems(m, n), T(0));
    for (int64_t i = 0; i < k; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            // Transposed opB stores the same logical element at (j, i).
            const bool t = (c.opB != FLAGSPARSE_OPERATION_NON_TRANSPOSE);
            const std::size_t slot = t ? c.lb.at(j, i, b_rows, b_cols)
                                       : c.lb.at(i, j, b_rows, b_cols);
            b_dev[slot] = static_cast<T>(b_ref[static_cast<std::size_t>(i * n + j)]);
        }
    }
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            c_dev[c.lc.at(i, j, m, n)] =
                static_cast<T>(c_ref[static_cast<std::size_t>(i * n + j)]);
        }
    }

    const std::vector<T> values(A.values.begin(), A.values.end());
    DeviceBuffer d_val = DeviceBuffer::from(values);
    DeviceBuffer d_col = upload_indices(A.indices, c.index_type);
    DeviceBuffer d_ptr = upload_indices(A.indptr, c.index_type);
    DeviceBuffer d_b   = DeviceBuffer::from(b_dev);
    DeviceBuffer d_c   = DeviceBuffer::from(c_dev);

    // COO carries one row index per nonzero where CSR carries one offset per row;
    // everything else about the call is identical, which is the claim being tested.
    DeviceBuffer d_row = upload_indices(coo_row_indices(A), c.index_type);

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnMatDescr_t matB = nullptr, matC = nullptr;
    out.status =
        (c.format == FLAGSPARSE_FORMAT_COO)
            ? flagsparseCreateCoo(&matA, m, k, A.nnz, d_row.get(), d_col.get(),
                                  d_val.get(), c.index_type,
                                  FLAGSPARSE_INDEX_BASE_ZERO, dtype)
            : flagsparseCreateCsr(&matA, m, k, A.nnz, d_ptr.get(), d_col.get(),
                                  d_val.get(), c.index_type, c.index_type,
                                  FLAGSPARSE_INDEX_BASE_ZERO, dtype);
    if (out.status != FLAGSPARSE_STATUS_SUCCESS) return out;
    flagsparseCreateDnMat(&matB, b_rows, b_cols, c.lb.ld(b_rows, b_cols), d_b.get(),
                          dtype, c.lb.order);
    flagsparseCreateDnMat(&matC, m, n, c.lc.ld(m, n), d_c.get(), dtype, c.lc.order);

    const T alpha_t = static_cast<T>(c.alpha);
    const T beta_t  = static_cast<T>(c.beta);

    // CSR reports 0 (its indptr already is the row-offsets array); COO reports
    // rows + 1 int32 and the caller owns that scratch.
    size_t buffer_size = 0;
    out.status = flagsparseSpMM_bufferSize(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                           c.opB, &alpha_t, matA, matB, &beta_t, matC,
                                           dtype, c.alg, &buffer_size);
    DeviceBuffer scratch(buffer_size);
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.status = flagsparseSpMM_preprocess(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                               c.opB, &alpha_t, matA, matB, &beta_t, matC,
                                               dtype, c.alg, scratch.get());
    }
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.status = flagsparseSpMM(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE, c.opB,
                                    &alpha_t, matA, matB, &beta_t, matC, dtype, c.alg,
                                    scratch.get());
    }
    dev_sync();

    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        const std::vector<T> got = d_c.download<T>(c_dev.size());
        std::vector<double> actual(static_cast<std::size_t>(m * n));
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                actual[static_cast<std::size_t>(i * n + j)] =
                    static_cast<double>(got[c.lc.at(i, j, m, n)]);
            }
        }
        const std::vector<double> ref =
            spmm_reference(A, b_ref, n, c.alpha, c.beta, c_ref);
        out.strict_ratio = max_error_ratio(actual, ref, default_tolerance(dtype));
        if (out.strict_ratio > 1.0) {
            out.relaxed_ratio = max_error_ratio(actual, ref, relaxed_tolerance(dtype));
            out.relaxed_used = true;
        }
    }

    flagsparseDestroyDnMat(matC);
    flagsparseDestroyDnMat(matB);
    flagsparseDestroySpMat(matA);
    return out;
}

// --------------------------------------------------------------- complex ---

// The complex reference lives here rather than in common.cpp: CsrMatrix carries
// real values, and SpMM is so far the only operator that multiplies complex
// numbers rather than moving them.
template <typename R>
RunResult run_spmm_complex(flagsparseHandle_t handle, const CsrMatrix& A,
                           flagsparseDataType_t dtype, const Case& c) {
    using C64 = std::complex<double>;
    RunResult out;
    const int64_t m = A.rows, k = A.cols, n = c.n;
    int64_t b_rows = 0, b_cols = 0;
    b_descr_extent(c, k, &b_rows, &b_cols);

    std::mt19937 rng(c.seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    std::vector<C64> a_vals(static_cast<std::size_t>(A.nnz));
    for (std::size_t i = 0; i < a_vals.size(); ++i) a_vals[i] = C64(A.values[i], dist(rng));
    std::vector<C64> b_ref(static_cast<std::size_t>(k * n));
    std::vector<C64> c_ref(static_cast<std::size_t>(m * n));
    for (auto& v : b_ref) v = C64(dist(rng), dist(rng));
    for (auto& v : c_ref) v = C64(dist(rng), dist(rng));
    const C64 alpha(c.alpha, c.beta == 0.0 ? 0.5 : -0.25);
    const C64 beta(c.beta, c.beta == 0.0 ? 0.0 : 0.75);

    // Interleaved real/imag, which is how the C API takes complex buffers.
    std::vector<R> a_dev(a_vals.size() * 2);
    for (std::size_t i = 0; i < a_vals.size(); ++i) {
        a_dev[i * 2]     = static_cast<R>(a_vals[i].real());
        a_dev[i * 2 + 1] = static_cast<R>(a_vals[i].imag());
    }
    std::vector<R> b_dev(c.lb.elems(b_rows, b_cols) * 2, R(0));
    std::vector<R> c_dev(c.lc.elems(m, n) * 2, R(0));
    const bool t = (c.opB != FLAGSPARSE_OPERATION_NON_TRANSPOSE);
    for (int64_t i = 0; i < k; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            const std::size_t slot = 2 * (t ? c.lb.at(j, i, b_rows, b_cols)
                                            : c.lb.at(i, j, b_rows, b_cols));
            const C64 v = b_ref[static_cast<std::size_t>(i * n + j)];
            b_dev[slot]     = static_cast<R>(v.real());
            b_dev[slot + 1] = static_cast<R>(v.imag());
        }
    }
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            const std::size_t slot = 2 * c.lc.at(i, j, m, n);
            const C64 v = c_ref[static_cast<std::size_t>(i * n + j)];
            c_dev[slot]     = static_cast<R>(v.real());
            c_dev[slot + 1] = static_cast<R>(v.imag());
        }
    }

    DeviceBuffer d_val = DeviceBuffer::from(a_dev);
    DeviceBuffer d_col = upload_indices(A.indices, c.index_type);
    DeviceBuffer d_ptr = upload_indices(A.indptr, c.index_type);
    DeviceBuffer d_b   = DeviceBuffer::from(b_dev);
    DeviceBuffer d_c   = DeviceBuffer::from(c_dev);

    // COO carries one row index per nonzero where CSR carries one offset per row;
    // everything else about the call is identical, which is the claim being tested.
    DeviceBuffer d_row = upload_indices(coo_row_indices(A), c.index_type);

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnMatDescr_t matB = nullptr, matC = nullptr;
    out.status =
        (c.format == FLAGSPARSE_FORMAT_COO)
            ? flagsparseCreateCoo(&matA, m, k, A.nnz, d_row.get(), d_col.get(),
                                  d_val.get(), c.index_type,
                                  FLAGSPARSE_INDEX_BASE_ZERO, dtype)
            : flagsparseCreateCsr(&matA, m, k, A.nnz, d_ptr.get(), d_col.get(),
                                  d_val.get(), c.index_type, c.index_type,
                                  FLAGSPARSE_INDEX_BASE_ZERO, dtype);
    if (out.status != FLAGSPARSE_STATUS_SUCCESS) return out;
    flagsparseCreateDnMat(&matB, b_rows, b_cols, c.lb.ld(b_rows, b_cols), d_b.get(),
                          dtype, c.lb.order);
    flagsparseCreateDnMat(&matC, m, n, c.lc.ld(m, n), d_c.get(), dtype, c.lc.order);

    const R alpha_t[2] = {static_cast<R>(alpha.real()), static_cast<R>(alpha.imag())};
    const R beta_t[2]  = {static_cast<R>(beta.real()),  static_cast<R>(beta.imag())};
    size_t buffer_size = 0;
    out.status = flagsparseSpMM_bufferSize(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                           c.opB, alpha_t, matA, matB, beta_t, matC,
                                           dtype, c.alg, &buffer_size);
    DeviceBuffer scratch(buffer_size);
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.status = flagsparseSpMM_preprocess(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                               c.opB, alpha_t, matA, matB, beta_t, matC,
                                               dtype, c.alg, scratch.get());
    }
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.status = flagsparseSpMM(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE, c.opB,
                                    alpha_t, matA, matB, beta_t, matC, dtype, c.alg,
                                    scratch.get());
    }
    dev_sync();

    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        const std::vector<R> got = d_c.download<R>(c_dev.size());
        // Real and imaginary parts are compared as one flat vector: a complex
        // result is wrong if either component is.
        std::vector<double> actual, ref;
        actual.reserve(static_cast<std::size_t>(m * n * 2));
        ref.reserve(static_cast<std::size_t>(m * n * 2));
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                C64 acc(0.0, 0.0);
                for (int32_t p = A.indptr[static_cast<std::size_t>(i)];
                     p < A.indptr[static_cast<std::size_t>(i) + 1]; ++p) {
                    const int64_t col = A.indices[static_cast<std::size_t>(p)];
                    acc += a_vals[static_cast<std::size_t>(p)] *
                           b_ref[static_cast<std::size_t>(col * n + j)];
                }
                C64 want = alpha * acc;
                if (beta != C64(0.0, 0.0)) {
                    want += beta * c_ref[static_cast<std::size_t>(i * n + j)];
                }
                const std::size_t slot = 2 * c.lc.at(i, j, m, n);
                actual.push_back(static_cast<double>(got[slot]));
                actual.push_back(static_cast<double>(got[slot + 1]));
                ref.push_back(want.real());
                ref.push_back(want.imag());
            }
        }
        out.strict_ratio = max_error_ratio(actual, ref, default_tolerance(dtype));
        if (out.strict_ratio > 1.0) {
            out.relaxed_ratio = max_error_ratio(actual, ref, relaxed_tolerance(dtype));
            out.relaxed_used = true;
        }
    }

    flagsparseDestroyDnMat(matC);
    flagsparseDestroyDnMat(matB);
    flagsparseDestroySpMat(matA);
    return out;
}

// Report the ratio, not just PASS/FAIL -- spec §6.3.1 asks for the number so
// precision work can be tracked over time.
void expect_close(const RunResult& r, const char* label, flagsparseHandle_t handle) {
    const char* detail = "";
    flagsparseGetLastErrorString(handle, &detail);
    ASSERT_EQ(r.status, FLAGSPARSE_STATUS_SUCCESS)
        << label << ": " << status_name(r.status) << " -- " << detail;
    if (!r.relaxed_used) {
        EXPECT_LE(r.strict_ratio, 1.0) << label << " max_error_ratio=" << r.strict_ratio;
        std::cout << "[   RATIO   ] " << label
                  << " max_error_ratio=" << r.strict_ratio << std::endl;
        return;
    }
    EXPECT_LE(r.relaxed_ratio, 1.0)
        << label << " failed even relaxed: strict=" << r.strict_ratio
        << " relaxed=" << r.relaxed_ratio;
    if (r.relaxed_ratio <= 1.0) {
        std::cout << "[ PASS(relaxed) ] " << label
                  << " strict_ratio=" << r.strict_ratio
                  << " relaxed_ratio=" << r.relaxed_ratio << std::endl;
    }
}

class SpMMAccuracy : public ::testing::Test {
  protected:
    Handle handle;
    void SetUp() override {
        if (handle.h == nullptr) GTEST_SKIP() << "no accelerator available";
        static bool announced = false;
        if (!announced) { print_backend_banner(); announced = true; }
    }
};

// n is swept across the warp/factor thresholds of the launch heuristic (4, 8,
// 16, 32, 64, >64), because each picks a different BLOCK_N / BLOCK_NNZ pair.
TEST_F(SpMMAccuracy, Float32AcrossBlockThresholds) {
    const CsrMatrix A = random_csr(96, 128, 0.05, 1234);
    for (int64_t n : {1, 4, 9, 17, 32, 33, 64, 65, 130}) {
        Case c; c.n = n; c.seed = static_cast<uint32_t>(n);
        const RunResult r = run_spmm<float>(handle.h, A, FLAGSPARSE_R_32F, c);
        expect_close(r, ("fp32_n" + std::to_string(n)).c_str(), handle.h);
    }
}

TEST_F(SpMMAccuracy, Float64MatchesHostReference) {
    for (auto shape : {std::pair<int64_t, int64_t>{64, 96}, {257, 129}, {1, 32}}) {
        const CsrMatrix A = random_csr(shape.first, shape.second, 0.05, 99);
        Case c; c.n = 48;
        const RunResult r = run_spmm<double>(handle.h, A, FLAGSPARSE_R_64F, c);
        expect_close(r, "fp64", handle.h);
    }
}

TEST_F(SpMMAccuracy, AlphaBetaAreApplied) {
    const CsrMatrix A = random_csr(128, 160, 0.08, 5);
    Case c; c.n = 40; c.alpha = -2.5; c.beta = 0.75;
    const RunResult r = run_spmm<double>(handle.h, A, FLAGSPARSE_R_64F, c);
    expect_close(r, "fp64_alpha_beta", handle.h);
}

// Column-major and a padded leading dimension are the same kernel with
// different strides. If that claim is wrong, it is wrong here.
TEST_F(SpMMAccuracy, DenseLayoutSweep) {
    const CsrMatrix A = random_csr(70, 90, 0.07, 21);
    const Layout row{FLAGSPARSE_ORDER_ROW, 0};
    const Layout col{FLAGSPARSE_ORDER_COL, 0};
    const Layout row_pad{FLAGSPARSE_ORDER_ROW, 5};
    const Layout col_pad{FLAGSPARSE_ORDER_COL, 3};
    struct Combo { Layout b, c; const char* name; };
    for (const Combo& combo : {Combo{row, row, "B_row_C_row"},
                               Combo{row, col, "B_row_C_col"},
                               Combo{col, row, "B_col_C_row"},
                               Combo{col, col, "B_col_C_col"},
                               Combo{row_pad, col_pad, "B_row_pad_C_col_pad"},
                               Combo{col_pad, row_pad, "B_col_pad_C_row_pad"}}) {
        Case c; c.n = 36; c.lb = combo.b; c.lc = combo.c; c.alpha = 1.5; c.beta = -0.5;
        const RunResult r = run_spmm<double>(handle.h, A, FLAGSPARSE_R_64F, c);
        expect_close(r, combo.name, handle.h);
    }
}

// op(B) = B^T is the two B strides swapped -- no transpose is materialised.
TEST_F(SpMMAccuracy, TransposedBMatchesUntransposed) {
    const CsrMatrix A = random_csr(80, 100, 0.06, 33);
    for (flagsparseOrder_t order : {FLAGSPARSE_ORDER_ROW, FLAGSPARSE_ORDER_COL}) {
        Case c;
        c.n = 24;
        c.opB = FLAGSPARSE_OPERATION_TRANSPOSE;
        c.lb = Layout{order, 0};
        const RunResult r = run_spmm<double>(handle.h, A, FLAGSPARSE_R_64F, c);
        expect_close(r, order == FLAGSPARSE_ORDER_ROW ? "opB_T_row" : "opB_T_col",
                     handle.h);
    }
}

TEST_F(SpMMAccuracy, Complex64MatchesHostReference) {
    const CsrMatrix A = random_csr(64, 80, 0.06, 41);
    Case c; c.n = 28;
    const RunResult r = run_spmm_complex<float>(handle.h, A, FLAGSPARSE_C_32F, c);
    expect_close(r, "c64", handle.h);
}

TEST_F(SpMMAccuracy, Complex128WithComplexAlphaBeta) {
    const CsrMatrix A = random_csr(48, 64, 0.08, 43);
    Case c; c.n = 20; c.alpha = 1.25; c.beta = -0.5;   // imaginary parts added inside
    const RunResult r = run_spmm_complex<double>(handle.h, A, FLAGSPARSE_C_64F, c);
    expect_close(r, "c128_alpha_beta", handle.h);

    Case cc = c; cc.lb = Layout{FLAGSPARSE_ORDER_COL, 0}; cc.lc = Layout{FLAGSPARSE_ORDER_COL, 2};
    const RunResult r2 = run_spmm_complex<double>(handle.h, A, FLAGSPARSE_C_64F, cc);
    expect_close(r2, "c128_col_major", handle.h);
}

// A matrix with no nonzeros at all still has to run: beta must reach every
// element of C, and alpha*0 must clear it when beta is zero.
TEST_F(SpMMAccuracy, EmptyRowsStillApplyBeta) {
    CsrMatrix A = random_csr(64, 64, 0.0, 17);
    ASSERT_EQ(A.nnz, 0);
    Case c; c.n = 16; c.alpha = 1.0; c.beta = 2.0;
    expect_close(run_spmm<double>(handle.h, A, FLAGSPARSE_R_64F, c), "empty_beta",
                 handle.h);
    c.beta = 0.0;
    expect_close(run_spmm<double>(handle.h, A, FLAGSPARSE_R_64F, c), "empty_no_beta",
                 handle.h);
}

TEST_F(SpMMAccuracy, RejectsMismatchedDimensions) {
    const CsrMatrix A = random_csr(32, 48, 0.1, 2);
    const std::vector<float> values(A.values.begin(), A.values.end());
    DeviceBuffer d_val = DeviceBuffer::from(values);
    DeviceBuffer d_col = DeviceBuffer::from(A.indices);
    DeviceBuffer d_ptr = DeviceBuffer::from(A.indptr);
    DeviceBuffer d_b   = DeviceBuffer::from(std::vector<float>((A.cols + 1) * 16, 1.0f));
    DeviceBuffer d_c   = DeviceBuffer::from(std::vector<float>(A.rows * 16, 0.0f));

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnMatDescr_t matB = nullptr, matC = nullptr;
    ASSERT_EQ(flagsparseCreateCsr(&matA, A.rows, A.cols, A.nnz, d_ptr.get(), d_col.get(),
                                  d_val.get(), FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F),
              FLAGSPARSE_STATUS_SUCCESS);
    // B has one row too many for A's column count.
    ASSERT_EQ(flagsparseCreateDnMat(&matB, A.cols + 1, 16, 16, d_b.get(),
                                    FLAGSPARSE_R_32F, FLAGSPARSE_ORDER_ROW),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnMat(&matC, A.rows, 16, 16, d_c.get(),
                                    FLAGSPARSE_R_32F, FLAGSPARSE_ORDER_ROW),
              FLAGSPARSE_STATUS_SUCCESS);

    const float one = 1.0f, zero = 0.0f;
    EXPECT_EQ(flagsparseSpMM(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                             FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, matB,
                             &zero, matC, FLAGSPARSE_R_32F, FLAGSPARSE_SPMM_ALG_DEFAULT,
                             nullptr),
              FLAGSPARSE_STATUS_INVALID_VALUE);

    // Unported paths must say NOT_SUPPORTED, never crash (spec §4.4).
    EXPECT_EQ(flagsparseSpMM(handle.h, FLAGSPARSE_OPERATION_TRANSPOSE,
                             FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, matB,
                             &zero, matC, FLAGSPARSE_R_32F, FLAGSPARSE_SPMM_ALG_DEFAULT,
                             nullptr),
              FLAGSPARSE_STATUS_NOT_SUPPORTED);
    // A COO algorithm id names a format this descriptor is not in.
    EXPECT_EQ(flagsparseSpMM(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                             FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, matB,
                             &zero, matC, FLAGSPARSE_R_32F, FLAGSPARSE_SPMM_COO_ALG1,
                             nullptr),
              FLAGSPARSE_STATUS_NOT_SUPPORTED);

    flagsparseDestroyDnMat(matC);
    flagsparseDestroyDnMat(matB);
    flagsparseDestroySpMat(matA);
}

// Index width lives in the descriptor, never in the function name (§4.5.1), so
// i64 must give bit-identical results to i32 on the same matrix -- the only
// thing that changes is one letter of the Triton signature.
TEST_F(SpMMAccuracy, Int64IndicesMatchInt32) {
    const CsrMatrix A = random_csr(72, 88, 0.07, 77);
    Case c32; c32.n = 40; c32.alpha = 1.5; c32.beta = -0.5;
    Case c64 = c32; c64.index_type = FLAGSPARSE_INDEX_64I;
    expect_close(run_spmm<double>(handle.h, A, FLAGSPARSE_R_64F, c32), "i32", handle.h);
    expect_close(run_spmm<double>(handle.h, A, FLAGSPARSE_R_64F, c64), "i64", handle.h);

    // Complex too: its kernel indexes the interleaved values array off the same
    // column indices.
    Case cc = c32; cc.n = 24; cc.index_type = FLAGSPARSE_INDEX_64I;
    expect_close(run_spmm_complex<double>(handle.h, A, FLAGSPARSE_C_64F, cc),
                 "c128_i64", handle.h);
}

// Every CSR algorithm id is answered by the same deterministic kernel, so they
// must all produce the same numbers, not merely all succeed.
TEST_F(SpMMAccuracy, CsrAlgorithmIdsAgree) {
    const CsrMatrix A = random_csr(96, 96, 0.05, 61);
    for (flagsparseSpMMAlg_t alg : {FLAGSPARSE_SPMM_CSR_ALG1, FLAGSPARSE_SPMM_CSR_ALG2,
                                    FLAGSPARSE_SPMM_CSR_ALG3}) {
        Case c; c.n = 32; c.alg = alg;
        const RunResult r = run_spmm<double>(handle.h, A, FLAGSPARSE_R_64F, c);
        expect_close(r, ("alg_" + std::to_string(static_cast<int>(alg))).c_str(),
                     handle.h);
    }
}

// ------------------------------------------------------------------- COO ---

// Same matrix, same dense operands, same reference -- only the sparse format
// differs. Both routes are deterministic but chunk the row differently
// (BLOCK_NNZ 4 for COO against the warp width for CSR), so they are compared to
// the fp64 reference rather than to each other bit-for-bit.
TEST_F(SpMMAccuracy, CooMatchesHostReference) {
    const CsrMatrix A = random_csr(96, 128, 0.05, 1234);
    for (int64_t n : {1, 9, 32, 65, 130}) {
        Case c; c.n = n; c.format = FLAGSPARSE_FORMAT_COO; c.seed = static_cast<uint32_t>(n);
        expect_close(run_spmm<float>(handle.h, A, FLAGSPARSE_R_32F, c),
                     ("coo_fp32_n" + std::to_string(n)).c_str(), handle.h);
    }
    Case c64; c64.n = 48; c64.format = FLAGSPARSE_FORMAT_COO;
    c64.alpha = -2.5; c64.beta = 0.75;
    expect_close(run_spmm<double>(handle.h, A, FLAGSPARSE_R_64F, c64),
                 "coo_fp64_alpha_beta", handle.h);
}

// The dense side is format-independent, so it must behave identically here.
TEST_F(SpMMAccuracy, CooDenseLayoutAndTranspose) {
    const CsrMatrix A = random_csr(70, 90, 0.07, 21);
    const Layout col_pad{FLAGSPARSE_ORDER_COL, 3};
    Case c; c.n = 36; c.format = FLAGSPARSE_FORMAT_COO; c.alpha = 1.5; c.beta = -0.5;
    c.lb = col_pad; c.lc = Layout{FLAGSPARSE_ORDER_ROW, 5};
    expect_close(run_spmm<double>(handle.h, A, FLAGSPARSE_R_64F, c), "coo_col_pad",
                 handle.h);

    Case t; t.n = 24; t.format = FLAGSPARSE_FORMAT_COO;
    t.opB = FLAGSPARSE_OPERATION_TRANSPOSE;
    expect_close(run_spmm<double>(handle.h, A, FLAGSPARSE_R_64F, t), "coo_opB_T",
                 handle.h);
}

TEST_F(SpMMAccuracy, CooComplexAndInt64) {
    const CsrMatrix A = random_csr(64, 80, 0.06, 41);
    Case c; c.n = 28; c.format = FLAGSPARSE_FORMAT_COO;
    expect_close(run_spmm_complex<float>(handle.h, A, FLAGSPARSE_C_32F, c), "coo_c64",
                 handle.h);

    Case cc; cc.n = 20; cc.format = FLAGSPARSE_FORMAT_COO;
    cc.alpha = 1.25; cc.beta = -0.5; cc.index_type = FLAGSPARSE_INDEX_64I;
    expect_close(run_spmm_complex<double>(handle.h, A, FLAGSPARSE_C_64F, cc),
                 "coo_c128_i64", handle.h);

    Case r; r.n = 40; r.format = FLAGSPARSE_FORMAT_COO;
    r.index_type = FLAGSPARSE_INDEX_64I; r.alpha = 1.5; r.beta = -0.5;
    expect_close(run_spmm<double>(handle.h, A, FLAGSPARSE_R_64F, r), "coo_fp64_i64",
                 handle.h);
}

// A row with no entries gets no COO nonzero at all, so it would never be written
// by a route that only visits runs of equal row ids. Under SEG_IS_ROW it still
// runs, and beta * C has to reach it.
TEST_F(SpMMAccuracy, CooEmptyRowsStillApplyBeta) {
    CsrMatrix A = random_csr(64, 64, 0.0, 17);
    ASSERT_EQ(A.nnz, 0);
    Case c; c.n = 16; c.format = FLAGSPARSE_FORMAT_COO; c.alpha = 1.0; c.beta = 2.0;
    expect_close(run_spmm<double>(handle.h, A, FLAGSPARSE_R_64F, c), "coo_empty_beta",
                 handle.h);
    c.beta = 0.0;
    expect_close(run_spmm<double>(handle.h, A, FLAGSPARSE_R_64F, c), "coo_empty_no_beta",
                 handle.h);

    // Interior empty rows, not just an all-empty matrix: density 0.02 on 128
    // columns leaves plenty of rows with nothing in them.
    const CsrMatrix sparse_rows = random_csr(256, 128, 0.02, 91);
    Case c2; c2.n = 24; c2.format = FLAGSPARSE_FORMAT_COO; c2.alpha = 2.0; c2.beta = -1.5;
    expect_close(run_spmm<double>(handle.h, sparse_rows, FLAGSPARSE_R_64F, c2),
                 "coo_interior_empty_rows", handle.h);
}

// Skipping preprocess must still be correct: the solve builds the offsets itself
// the first time it sees a buffer. It is only a performance step.
TEST_F(SpMMAccuracy, CooWorksWithoutPreprocess) {
    const CsrMatrix A = random_csr(80, 96, 0.06, 55);
    const int64_t n = 32;
    std::mt19937 rng(3);
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> b_ref(static_cast<std::size_t>(A.cols * n));
    for (auto& v : b_ref) v = dist(rng);

    const std::vector<double> values(A.values.begin(), A.values.end());
    DeviceBuffer d_val = DeviceBuffer::from(values);
    DeviceBuffer d_row = DeviceBuffer::from(coo_row_indices(A));
    DeviceBuffer d_col = DeviceBuffer::from(A.indices);
    DeviceBuffer d_b   = DeviceBuffer::from(b_ref);
    DeviceBuffer d_c   = DeviceBuffer::from(std::vector<double>(
        static_cast<std::size_t>(A.rows * n), 0.0));

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnMatDescr_t matB = nullptr, matC = nullptr;
    ASSERT_EQ(flagsparseCreateCoo(&matA, A.rows, A.cols, A.nnz, d_row.get(), d_col.get(),
                                  d_val.get(), FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_64F),
              FLAGSPARSE_STATUS_SUCCESS);
    flagsparseCreateDnMat(&matB, A.cols, n, n, d_b.get(), FLAGSPARSE_R_64F,
                          FLAGSPARSE_ORDER_ROW);
    flagsparseCreateDnMat(&matC, A.rows, n, n, d_c.get(), FLAGSPARSE_R_64F,
                          FLAGSPARSE_ORDER_ROW);

    const double one = 1.0, zero = 0.0;
    size_t buffer_size = 0;
    ASSERT_EQ(flagsparseSpMM_bufferSize(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                        FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA,
                                        matB, &zero, matC, FLAGSPARSE_R_64F,
                                        FLAGSPARSE_SPMM_ALG_DEFAULT, &buffer_size),
              FLAGSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(buffer_size, static_cast<size_t>(A.rows + 1) * sizeof(int32_t));
    DeviceBuffer scratch(buffer_size);

    // No preprocess call at all.
    ASSERT_EQ(flagsparseSpMM(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                             FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, matB, &zero,
                             matC, FLAGSPARSE_R_64F, FLAGSPARSE_SPMM_ALG_DEFAULT,
                             scratch.get()),
              FLAGSPARSE_STATUS_SUCCESS);
    dev_sync();
    const std::vector<double> got = d_c.download<double>(
        static_cast<std::size_t>(A.rows * n));
    const std::vector<double> ref = spmm_reference(
        A, b_ref, n, 1.0, 0.0, std::vector<double>(got.size(), 0.0));
    EXPECT_LE(max_error_ratio(got, ref, default_tolerance(FLAGSPARSE_R_64F)), 1.0);

    // Omitting the buffer is a caller error, not a silent wrong answer.
    EXPECT_EQ(flagsparseSpMM(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                             FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, matB, &zero,
                             matC, FLAGSPARSE_R_64F, FLAGSPARSE_SPMM_ALG_DEFAULT,
                             nullptr),
              FLAGSPARSE_STATUS_INVALID_VALUE);

    // A CSR algorithm id names a format this descriptor is not in.
    EXPECT_EQ(flagsparseSpMM(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                             FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, matB, &zero,
                             matC, FLAGSPARSE_R_64F, FLAGSPARSE_SPMM_CSR_ALG1, nullptr),
              FLAGSPARSE_STATUS_NOT_SUPPORTED);

    flagsparseDestroyDnMat(matC);
    flagsparseDestroyDnMat(matB);
    flagsparseDestroySpMat(matA);
}

// cuSPARSE requires COO sorted by row. Saying so beats computing nonsense: an
// unsorted matrix would give every duplicated row run a partial result, with the
// last writer winning.
TEST_F(SpMMAccuracy, CooRejectsUnsortedRows) {
    const CsrMatrix A = random_csr(32, 32, 0.2, 13);
    ASSERT_GT(A.nnz, 4);
    std::vector<int32_t> row = coo_row_indices(A);
    // Move the last nonzero to the front: now the row ids descend at entry 1.
    std::swap(row.front(), row.back());
    ASSERT_NE(row.front(), row[1]);

    const std::vector<float> values(A.values.begin(), A.values.end());
    DeviceBuffer d_val = DeviceBuffer::from(values);
    DeviceBuffer d_row = DeviceBuffer::from(row);
    DeviceBuffer d_col = DeviceBuffer::from(A.indices);
    DeviceBuffer d_b   = DeviceBuffer::from(std::vector<float>(A.cols * 16, 1.0f));
    DeviceBuffer d_c   = DeviceBuffer::from(std::vector<float>(A.rows * 16, 0.0f));

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnMatDescr_t matB = nullptr, matC = nullptr;
    flagsparseCreateCoo(&matA, A.rows, A.cols, A.nnz, d_row.get(), d_col.get(),
                        d_val.get(), FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO,
                        FLAGSPARSE_R_32F);
    flagsparseCreateDnMat(&matB, A.cols, 16, 16, d_b.get(), FLAGSPARSE_R_32F,
                          FLAGSPARSE_ORDER_ROW);
    flagsparseCreateDnMat(&matC, A.rows, 16, 16, d_c.get(), FLAGSPARSE_R_32F,
                          FLAGSPARSE_ORDER_ROW);

    const float one = 1.0f, zero = 0.0f;
    DeviceBuffer scratch(static_cast<size_t>(A.rows + 1) * sizeof(int32_t));
    EXPECT_EQ(flagsparseSpMM_preprocess(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                        FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA,
                                        matB, &zero, matC, FLAGSPARSE_R_32F,
                                        FLAGSPARSE_SPMM_ALG_DEFAULT, scratch.get()),
              FLAGSPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(flagsparseSpMM(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                             FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, matB, &zero,
                             matC, FLAGSPARSE_R_32F, FLAGSPARSE_SPMM_ALG_DEFAULT,
                             scratch.get()),
              FLAGSPARSE_STATUS_INVALID_VALUE);

    flagsparseDestroyDnMat(matC);
    flagsparseDestroyDnMat(matB);
    flagsparseDestroySpMat(matA);
}

}  // namespace
