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

template <typename In, typename Out>
struct MixedRunResult {
    flagsparseStatus_t status = FLAGSPARSE_STATUS_SUCCESS;
    size_t buffer_size = 0;
    std::vector<Out> values;
};

// A small exact matrix is more useful than random floating-point input here:
// it exercises the input/output dtype split without conflating it with host
// rounding. The same operands cover CSR and COO.
template <typename In, typename Out>
MixedRunResult<In, Out> run_spmm_mixed(flagsparseHandle_t handle,
                                       flagsparseFormat_t format,
                                       flagsparseDataType_t input_type,
                                       flagsparseDataType_t output_type,
                                       const std::vector<In>& a_values,
                                       const std::vector<In>& b_values) {
    constexpr int64_t m = 3, k = 4, n = 3;
    const std::vector<int32_t> indptr{0, 2, 3, 5};
    const std::vector<int32_t> rows{0, 0, 1, 2, 2};
    const std::vector<int32_t> cols{0, 2, 1, 0, 3};
    std::vector<Out> c_values(static_cast<std::size_t>(m * n), static_cast<Out>(37));

    DeviceBuffer d_val = DeviceBuffer::from(a_values);
    DeviceBuffer d_ptr = DeviceBuffer::from(indptr);
    DeviceBuffer d_row = DeviceBuffer::from(rows);
    DeviceBuffer d_col = DeviceBuffer::from(cols);
    DeviceBuffer d_b = DeviceBuffer::from(b_values);
    DeviceBuffer d_c = DeviceBuffer::from(c_values);

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnMatDescr_t matB = nullptr, matC = nullptr;
    MixedRunResult<In, Out> out;
    out.status = format == FLAGSPARSE_FORMAT_COO
                     ? flagsparseCreateCoo(&matA, m, k, 5, d_row.get(), d_col.get(),
                                           d_val.get(), FLAGSPARSE_INDEX_32I,
                                           FLAGSPARSE_INDEX_BASE_ZERO, input_type)
                     : flagsparseCreateCsr(&matA, m, k, 5, d_ptr.get(), d_col.get(),
                                           d_val.get(), FLAGSPARSE_INDEX_32I,
                                           FLAGSPARSE_INDEX_32I,
                                           FLAGSPARSE_INDEX_BASE_ZERO, input_type);
    if (out.status != FLAGSPARSE_STATUS_SUCCESS) return out;
    out.status = flagsparseCreateDnMat(&matB, k, n, n, d_b.get(), input_type,
                                       FLAGSPARSE_ORDER_ROW);
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.status = flagsparseCreateDnMat(&matC, m, n, n, d_c.get(), output_type,
                                           FLAGSPARSE_ORDER_ROW);
    }

    const Out one = static_cast<Out>(1), zero = static_cast<Out>(0);
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.status = flagsparseSpMM_bufferSize(
            handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
            FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, matB, &zero, matC,
            output_type, FLAGSPARSE_SPMM_ALG_DEFAULT, &out.buffer_size);
    }
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.status = flagsparseSpMM_preprocess(
            handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
            FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, matB, &zero, matC,
            output_type, FLAGSPARSE_SPMM_ALG_DEFAULT, nullptr);
    }
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.status = flagsparseSpMM(
            handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
            FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, matB, &zero, matC,
            output_type, FLAGSPARSE_SPMM_ALG_DEFAULT, nullptr);
    }
    dev_sync();
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.values = d_c.download<Out>(c_values.size());
    }

    if (matC != nullptr) flagsparseDestroyDnMat(matC);
    if (matB != nullptr) flagsparseDestroyDnMat(matB);
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

TEST_F(SpMMAccuracy, CscFloat32RowMajorIdentity) {
    // A = [[1, 0, 2], [0, 3, 0]], encoded by CSC columns.
    const std::vector<int32_t> colptr{0, 1, 2, 3};
    const std::vector<int32_t> rows{0, 1, 0};
    const std::vector<float> values{1.0f, 3.0f, 2.0f};
    const std::vector<float> b{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const std::vector<float> expected{11.0f, 14.0f, 9.0f, 12.0f};
    DeviceBuffer d_ptr = DeviceBuffer::from(colptr);
    DeviceBuffer d_rows = DeviceBuffer::from(rows);
    DeviceBuffer d_values = DeviceBuffer::from(values);
    DeviceBuffer d_b = DeviceBuffer::from(b);
    DeviceBuffer d_c = DeviceBuffer::from(std::vector<float>(4, -1.0f));
    flagsparseSpMatDescr_t a = nullptr;
    flagsparseDnMatDescr_t mat_b = nullptr, mat_c = nullptr;
    ASSERT_EQ(flagsparseCreateCsc(&a, 2, 3, 3, d_ptr.get(), d_rows.get(), d_values.get(),
                                  FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F), FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnMat(&mat_b, 3, 2, 2, d_b.get(), FLAGSPARSE_R_32F,
                                    FLAGSPARSE_ORDER_ROW), FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnMat(&mat_c, 2, 2, 2, d_c.get(), FLAGSPARSE_R_32F,
                                    FLAGSPARSE_ORDER_ROW), FLAGSPARSE_STATUS_SUCCESS);
    const float one = 1.0f, zero = 0.0f;
    size_t bytes = 1;
    ASSERT_EQ(flagsparseSpMM_bufferSize(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                        FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, a, mat_b,
                                        &zero, mat_c, FLAGSPARSE_R_32F,
                                        FLAGSPARSE_SPMM_ALG_DEFAULT, &bytes), FLAGSPARSE_STATUS_SUCCESS);
    DeviceBuffer scratch(bytes);
    ASSERT_EQ(flagsparseSpMM_preprocess(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
        FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, a, mat_b, &zero, mat_c,
        FLAGSPARSE_R_32F, FLAGSPARSE_SPMM_ALG_DEFAULT, scratch.get()), FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseSpMM(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                             FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, a, mat_b, &zero, mat_c,
                             FLAGSPARSE_R_32F, FLAGSPARSE_SPMM_ALG_DEFAULT, scratch.get()), FLAGSPARSE_STATUS_SUCCESS);
    dev_sync();
    EXPECT_EQ(d_c.download<float>(4), expected);
    flagsparseDestroyDnMat(mat_c); flagsparseDestroyDnMat(mat_b); flagsparseDestroySpMat(a);
}

TEST_F(SpMMAccuracy, CscFloat16RowMajorIdentity) {
    const std::vector<int32_t> colptr{0, 1, 2, 3};
    const std::vector<int32_t> rows{0, 1, 0};
    const std::vector<Half> values{Half(1), Half(3), Half(2)};
    const std::vector<Half> b{Half(1), Half(2), Half(3), Half(4), Half(5), Half(6)};
    const std::vector<Half> expected{Half(11), Half(14), Half(9), Half(12)};
    DeviceBuffer d_ptr = DeviceBuffer::from(colptr), d_rows = DeviceBuffer::from(rows);
    DeviceBuffer d_values = DeviceBuffer::from(values), d_b = DeviceBuffer::from(b);
    DeviceBuffer d_c = DeviceBuffer::from(std::vector<Half>(4, Half(-1)));
    flagsparseSpMatDescr_t a = nullptr; flagsparseDnMatDescr_t mat_b = nullptr, mat_c = nullptr;
    ASSERT_EQ(flagsparseCreateCsc(&a, 2, 3, 3, d_ptr.get(), d_rows.get(), d_values.get(),
                                  FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_16F), FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnMat(&mat_b, 3, 2, 2, d_b.get(), FLAGSPARSE_R_16F,
                                    FLAGSPARSE_ORDER_ROW), FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnMat(&mat_c, 2, 2, 2, d_c.get(), FLAGSPARSE_R_16F,
                                    FLAGSPARSE_ORDER_ROW), FLAGSPARSE_STATUS_SUCCESS);
    const Half one(1), zero(0);
    ASSERT_EQ(flagsparseSpMM(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                             FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, a, mat_b, &zero, mat_c,
                             FLAGSPARSE_R_16F, FLAGSPARSE_SPMM_ALG_DEFAULT, nullptr), FLAGSPARSE_STATUS_SUCCESS);
    dev_sync(); EXPECT_EQ(d_c.download<Half>(4), expected);
    flagsparseDestroyDnMat(mat_c); flagsparseDestroyDnMat(mat_b); flagsparseDestroySpMat(a);
}

TEST_F(SpMMAccuracy, CscComplex64RowMajorIdentity) {
    using C = std::complex<float>;
    const std::vector<int32_t> colptr{0, 1, 2, 3};
    const std::vector<int32_t> rows{0, 1, 0};
    const std::vector<C> values{C(1, 1), C(3, -1), C(2, 0)};
    const std::vector<C> b{C(1, 1), C(2, 0), C(3, 0), C(4, -1), C(5, 0), C(6, 0)};
    const std::vector<C> expected{C(10, 2), C(14, 2), C(9, -3), C(11, -7)};
    DeviceBuffer d_ptr = DeviceBuffer::from(colptr), d_rows = DeviceBuffer::from(rows);
    DeviceBuffer d_values = DeviceBuffer::from(values), d_b = DeviceBuffer::from(b);
    DeviceBuffer d_c = DeviceBuffer::from(std::vector<C>(4, C(-1, 1)));
    flagsparseSpMatDescr_t a = nullptr; flagsparseDnMatDescr_t mat_b = nullptr, mat_c = nullptr;
    ASSERT_EQ(flagsparseCreateCsc(&a, 2, 3, 3, d_ptr.get(), d_rows.get(), d_values.get(),
                                  FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_C_32F), FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnMat(&mat_b, 3, 2, 2, d_b.get(), FLAGSPARSE_C_32F,
                                    FLAGSPARSE_ORDER_ROW), FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnMat(&mat_c, 2, 2, 2, d_c.get(), FLAGSPARSE_C_32F,
                                    FLAGSPARSE_ORDER_ROW), FLAGSPARSE_STATUS_SUCCESS);
    const C one(1, 0), zero(0, 0);
    size_t bytes = 0;
    ASSERT_EQ(flagsparseSpMM_bufferSize(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
        FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, a, mat_b, &zero, mat_c,
        FLAGSPARSE_C_32F, FLAGSPARSE_SPMM_ALG_DEFAULT, &bytes), FLAGSPARSE_STATUS_SUCCESS);
    DeviceBuffer scratch(bytes);
    // Lazy preparation with genuinely complex values, followed by a live-value update.
    ASSERT_EQ(flagsparseSpMM(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                             FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, a, mat_b, &zero, mat_c,
                             FLAGSPARSE_C_32F, FLAGSPARSE_SPMM_ALG_DEFAULT, scratch.get()), FLAGSPARSE_STATUS_SUCCESS);
    dev_sync(); EXPECT_EQ(d_c.download<C>(4), expected);
    auto updated = values;
    for (auto& v : updated) v *= 2.f;
    ASSERT_EQ(to_device(d_values.get(), updated.data(), updated.size() * sizeof(C)), FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseSpMM(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
        FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, a, mat_b, &zero, mat_c,
        FLAGSPARSE_C_32F, FLAGSPARSE_SPMM_ALG_DEFAULT, scratch.get()), FLAGSPARSE_STATUS_SUCCESS);
    dev_sync();
    const auto got = d_c.download<C>(4);
    for (size_t i = 0; i < got.size(); ++i) EXPECT_EQ(got[i], 2.f * expected[i]);
    flagsparseDestroyDnMat(mat_c); flagsparseDestroyDnMat(mat_b); flagsparseDestroySpMat(a);
}

TEST_F(SpMMAccuracy, CsrInt8ToInt32MixedPrecision) {
    const std::vector<int8_t> a{2, -1, 3, 4, -2};
    const std::vector<int8_t> b{1, 2, -1, 3, 0, 2, -2, 1, 4, 5, -3, 1};
    const auto r = run_spmm_mixed<int8_t, int32_t>(
        handle.h, FLAGSPARSE_FORMAT_CSR, FLAGSPARSE_R_8I, FLAGSPARSE_R_32I, a, b);
    ASSERT_EQ(r.status, FLAGSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r.buffer_size, 0u);
    EXPECT_EQ(r.values, (std::vector<int32_t>{4, 3, -6, 9, 0, 6, -6, 14, -6}));
}

TEST_F(SpMMAccuracy, CsrFloat16ToFloat32MixedPrecision) {
    const std::vector<Half> a{Half(0.5), Half(-1.0), Half(2.0), Half(1.5), Half(-0.5)};
    const std::vector<Half> b{
        Half(1.0), Half(2.0), Half(-1.0), Half(3.0), Half(0.0), Half(2.0),
        Half(-2.0), Half(1.0), Half(4.0), Half(5.0), Half(-3.0), Half(1.0)};
    const auto r = run_spmm_mixed<Half, float>(
        handle.h, FLAGSPARSE_FORMAT_CSR, FLAGSPARSE_R_16F, FLAGSPARSE_R_32F, a, b);
    ASSERT_EQ(r.status, FLAGSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r.buffer_size, 0u);
    EXPECT_EQ(r.values, (std::vector<float>{2.5f, 0.0f, -4.5f, 6.0f, 0.0f, 4.0f,
                                            -1.0f, 4.5f, -2.0f}));
}

TEST_F(SpMMAccuracy, CooInt8ToInt32MixedPrecision) {
    const std::vector<int8_t> a{2, -1, 3, 4, -2};
    const std::vector<int8_t> b{1, 2, -1, 3, 0, 2, -2, 1, 4, 5, -3, 1};
    const auto r = run_spmm_mixed<int8_t, int32_t>(
        handle.h, FLAGSPARSE_FORMAT_COO, FLAGSPARSE_R_8I, FLAGSPARSE_R_32I, a, b);
    ASSERT_EQ(r.status, FLAGSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(r.buffer_size, 0u);
    EXPECT_EQ(r.values, (std::vector<int32_t>{4, 3, -6, 9, 0, 6, -6, 14, -6}));
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

TEST_F(SpMMAccuracy, CsrTransposeFloat32MatchesHostReference) {
    const int64_t m = 2, k = 3, n = 2;
    const std::vector<int32_t> indptr{0, 2, 3};
    const std::vector<int32_t> indices{0, 2, 1};
    const std::vector<float> values{1.0f, 2.0f, 3.0f};
    const std::vector<float> b_values{1.0f, 2.0f, 4.0f, 5.0f};
    const std::vector<float> c_initial{7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};

    DeviceBuffer d_values = DeviceBuffer::from(values);
    DeviceBuffer d_indices = DeviceBuffer::from(indices);
    DeviceBuffer d_indptr = DeviceBuffer::from(indptr);
    DeviceBuffer d_b = DeviceBuffer::from(b_values);
    DeviceBuffer d_c = DeviceBuffer::from(c_initial);

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnMatDescr_t matB = nullptr, matC = nullptr;
    ASSERT_EQ(flagsparseCreateCsr(
                  &matA, m, k, static_cast<int64_t>(values.size()), d_indptr.get(),
                  d_indices.get(), d_values.get(), FLAGSPARSE_INDEX_32I,
                  FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnMat(&matB, m, n, n, d_b.get(), FLAGSPARSE_R_32F,
                                    FLAGSPARSE_ORDER_ROW),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnMat(&matC, k, n, n, d_c.get(), FLAGSPARSE_R_32F,
                                    FLAGSPARSE_ORDER_ROW),
              FLAGSPARSE_STATUS_SUCCESS);

    const float alpha = 2.0f, beta = 0.5f;
    size_t buffer_size = 0;
    ASSERT_EQ(flagsparseSpMM_bufferSize(
                  handle.h, FLAGSPARSE_OPERATION_TRANSPOSE,
                  FLAGSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
                  FLAGSPARSE_R_32F, FLAGSPARSE_SPMM_ALG_DEFAULT, &buffer_size),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseSpMM_preprocess(
                  handle.h, FLAGSPARSE_OPERATION_TRANSPOSE,
                  FLAGSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
                  FLAGSPARSE_R_32F, FLAGSPARSE_SPMM_ALG_DEFAULT, nullptr),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseSpMM(
                  handle.h, FLAGSPARSE_OPERATION_TRANSPOSE,
                  FLAGSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, matB, &beta, matC,
                  FLAGSPARSE_R_32F, FLAGSPARSE_SPMM_ALG_DEFAULT, nullptr),
              FLAGSPARSE_STATUS_SUCCESS);
    dev_sync();

    const std::vector<float> got = d_c.download<float>(c_initial.size());
    const std::vector<float> expected{5.5f, 8.0f, 28.5f, 35.0f, 9.5f, 14.0f};
    ASSERT_EQ(got.size(), expected.size());
    for (std::size_t i = 0; i < expected.size(); ++i) EXPECT_NEAR(got[i], expected[i], 1e-4f);

    flagsparseDestroyDnMat(matC);
    flagsparseDestroyDnMat(matB);
    flagsparseDestroySpMat(matA);
}

TEST_F(SpMMAccuracy, CsrTransposeSkewedRowsAndLayouts) {
    constexpr int64_t m = 129, k = 97;
    for (const int64_t n : {8, 33}) {
        std::vector<int32_t> ptr(m + 1, 97), cols;
        ptr[0] = 0;
        ptr[m] = 100;
        for (int32_t i = 0; i < 97; ++i) cols.push_back(i);
        cols.insert(cols.end(), {0, 0, 96});  // duplicates and empty interior rows
        std::vector<float> values(100, 1.0f);
        for (auto index_type : {FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_64I}) {
            for (auto order : {FLAGSPARSE_ORDER_ROW, FLAGSPARSE_ORDER_COL}) {
                for (auto opB : {FLAGSPARSE_OPERATION_NON_TRANSPOSE, FLAGSPARSE_OPERATION_TRANSPOSE}) {
                    const int64_t br = opB == FLAGSPARSE_OPERATION_TRANSPOSE ? n : m;
                    const int64_t bc = opB == FLAGSPARSE_OPERATION_TRANSPOSE ? m : n;
                    Layout layout{order, 3};
                    std::vector<float> b(layout.elems(br, bc), 0.f);
                    for (int64_t r = 0; r < m; ++r) {
                        for (int64_t j = 0; j < n; ++j) {
                            b[opB == FLAGSPARSE_OPERATION_TRANSPOSE
                                  ? layout.at(j, r, br, bc) : layout.at(r, j, br, bc)] =
                                static_cast<float>((r + j) % 11 - 5);
                        }
                    }
                    std::vector<float> initial(layout.elems(k, n), 2.f), expected = initial;
                    for (int64_t c = 0; c < k; ++c)
                        for (int64_t j = 0; j < n; ++j) expected[layout.at(c, j, k, n)] *= 0.5f;
                    for (int64_t r = 0; r < m; ++r)
                        for (int32_t p = ptr[r]; p < ptr[r + 1]; ++p)
                            for (int64_t j = 0; j < n; ++j)
                                expected[layout.at(cols[p], j, k, n)] +=
                                    2.f * static_cast<float>((r + j) % 11 - 5);
                    auto dv = DeviceBuffer::from(values);
                    auto dp = upload_indices(ptr, index_type);
                    auto di = upload_indices(cols, index_type);
                    auto db = DeviceBuffer::from(b);
                    auto dc = DeviceBuffer::from(initial);
                    flagsparseSpMatDescr_t a = nullptr;
                    flagsparseDnMatDescr_t bd = nullptr, cd = nullptr;
                    ASSERT_EQ(flagsparseCreateCsr(&a, m, k, 100, dp.get(), di.get(), dv.get(),
                        index_type, index_type, FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F), FLAGSPARSE_STATUS_SUCCESS);
                    ASSERT_EQ(flagsparseCreateDnMat(&bd, br, bc, layout.ld(br, bc), db.get(),
                        FLAGSPARSE_R_32F, order), FLAGSPARSE_STATUS_SUCCESS);
                    ASSERT_EQ(flagsparseCreateDnMat(&cd, k, n, layout.ld(k, n), dc.get(),
                        FLAGSPARSE_R_32F, order), FLAGSPARSE_STATUS_SUCCESS);
                    const float alpha = 2.f, beta = 0.5f;
                    size_t workspace_bytes = 0;
                    ASSERT_EQ(flagsparseSpMM_bufferSize(handle.h, FLAGSPARSE_OPERATION_TRANSPOSE, opB,
                        &alpha, a, bd, &beta, cd, FLAGSPARSE_R_32F,
                        FLAGSPARSE_SPMM_ALG_DEFAULT, &workspace_bytes), FLAGSPARSE_STATUS_SUCCESS);
                    DeviceBuffer workspace(workspace_bytes);
                    ASSERT_EQ(flagsparseSpMM_preprocess(handle.h, FLAGSPARSE_OPERATION_TRANSPOSE, opB,
                        &alpha, a, bd, &beta, cd, FLAGSPARSE_R_32F,
                        FLAGSPARSE_SPMM_ALG_DEFAULT, workspace.get()), FLAGSPARSE_STATUS_SUCCESS);
                    ASSERT_EQ(flagsparseSpMM(handle.h, FLAGSPARSE_OPERATION_TRANSPOSE, opB,
                        &alpha, a, bd, &beta, cd, FLAGSPARSE_R_32F,
                        FLAGSPARSE_SPMM_ALG_DEFAULT, workspace.get()), FLAGSPARSE_STATUS_SUCCESS);
                    dev_sync();
                    const auto got = dc.download<float>(initial.size());
                    for (int64_t c = 0; c < k; ++c)
                        for (int64_t j = 0; j < n; ++j)
                            EXPECT_FLOAT_EQ(got[layout.at(c, j, k, n)], expected[layout.at(c, j, k, n)]);
                    // Reuse topology but change values in place: no stale value cache.
                    std::vector<float> updated_values(100, 2.f);
                    ASSERT_EQ(to_device(dv.get(), updated_values.data(), updated_values.size() * sizeof(float)),
                        FLAGSPARSE_STATUS_SUCCESS);
                    const float zero_beta = 0.f;
                    ASSERT_EQ(flagsparseSpMM(handle.h, FLAGSPARSE_OPERATION_TRANSPOSE, opB,
                        &alpha, a, bd, &zero_beta, cd, FLAGSPARSE_R_32F,
                        FLAGSPARSE_SPMM_ALG_DEFAULT, workspace.get()), FLAGSPARSE_STATUS_SUCCESS);
                    dev_sync();
                    const auto updated = dc.download<float>(initial.size());
                    for (int64_t c = 0; c < k; ++c)
                        for (int64_t j = 0; j < n; ++j) {
                            const auto pos = layout.at(c, j, k, n);
                            EXPECT_FLOAT_EQ(updated[pos], 2.f * (expected[pos] - 1.f));
                        }
                    flagsparseDestroyDnMat(cd);
                    flagsparseDestroyDnMat(bd);
                    flagsparseDestroySpMat(a);
                }
            }
        }
    }
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

TEST_F(SpMMAccuracy, Complex64NarrowRowsWithAlphaBetaAndLayouts) {
    const CsrMatrix A = random_csr(65, 81, 0.06, 41);
    for (auto order : {FLAGSPARSE_ORDER_ROW, FLAGSPARSE_ORDER_COL}) {
        for (auto opB : {FLAGSPARSE_OPERATION_NON_TRANSPOSE, FLAGSPARSE_OPERATION_TRANSPOSE}) {
            Case c; c.n = 8; c.alpha = 1.25; c.beta = -0.5;
            c.lb = Layout{order, 3}; c.lc = Layout{order, 2}; c.opB = opB;
            expect_close(run_spmm_complex<float>(handle.h, A, FLAGSPARSE_C_32F, c),
                "c32_n8_alpha_beta_layout", handle.h);
        }
    }
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

    // The transpose route is implemented for f32 CSR, so its mismatched B
    // extent is an argument error rather than an unsupported operation.
    EXPECT_EQ(flagsparseSpMM(handle.h, FLAGSPARSE_OPERATION_TRANSPOSE,
                             FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, matB,
                             &zero, matC, FLAGSPARSE_R_32F, FLAGSPARSE_SPMM_ALG_DEFAULT,
                             nullptr),
              FLAGSPARSE_STATUS_INVALID_VALUE);
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
