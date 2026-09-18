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


// Accuracy tests for flagsparseSpMV, against an fp64 host reference.

#include <gtest/gtest.h>

#include <complex>
#include <memory>

#include "common.hpp"

using namespace fstest;

namespace {

struct Handle {
    flagsparseHandle_t h = nullptr;
    Handle() { flagsparseCreate(&h); }
    ~Handle() { if (h) flagsparseDestroy(h); }
};

// One SpMV against the host reference, for a given value type and beta.
// Returns the error ratio at the strict tolerance; -1 when the call itself
// failed (which the caller turns into a test failure with the status name).
struct RunResult {
    flagsparseStatus_t status = FLAGSPARSE_STATUS_SUCCESS;
    double strict_ratio = 0.0;
    double relaxed_ratio = 0.0;
    bool relaxed_used = false;
};

template <typename T>
RunResult run_spmv(flagsparseHandle_t handle, const CsrMatrix& A,
                   flagsparseDataType_t dtype, double alpha, double beta,
                   uint32_t seed) {
    RunResult out;
    const auto n_rows = static_cast<std::size_t>(A.rows);
    const auto n_cols = static_cast<std::size_t>(A.cols);

    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> x64(n_cols), y64(n_rows);
    for (auto& v : x64) v = dist(rng);
    for (auto& v : y64) v = dist(rng);

    // The device copies are the dtype under test; the reference keeps fp64.
    std::vector<T> values(A.values.begin(), A.values.end());
    std::vector<T> x(x64.begin(), x64.end());
    std::vector<T> y(y64.begin(), y64.end());

    DeviceBuffer d_val = DeviceBuffer::from(values);
    DeviceBuffer d_col = DeviceBuffer::from(A.indices);
    DeviceBuffer d_ptr = DeviceBuffer::from(A.indptr);
    DeviceBuffer d_x   = DeviceBuffer::from(x);
    DeviceBuffer d_y   = DeviceBuffer::from(y);

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    out.status = flagsparseCreateCsr(&matA, A.rows, A.cols, A.nnz,
                                     d_ptr.get(), d_col.get(), d_val.get(),
                                     FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                     FLAGSPARSE_INDEX_BASE_ZERO, dtype);
    if (out.status != FLAGSPARSE_STATUS_SUCCESS) return out;
    flagsparseCreateDnVec(&vecX, A.cols, d_x.get(), dtype);
    flagsparseCreateDnVec(&vecY, A.rows, d_y.get(), dtype);

    const T alpha_t = static_cast<T>(alpha);
    const T beta_t  = static_cast<T>(beta);

    size_t buffer_size = 0;
    out.status = flagsparseSpMV_bufferSize(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_t, matA, vecX, &beta_t, vecY, dtype,
                                           FLAGSPARSE_SPMV_ALG_DEFAULT, &buffer_size);
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.status = flagsparseSpMV(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                    &alpha_t, matA, vecX, &beta_t, vecY, dtype,
                                    FLAGSPARSE_SPMV_ALG_DEFAULT, nullptr);
    }
    dev_sync();

    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        const std::vector<T> got = d_y.download<T>(n_rows);
        const std::vector<double> actual(got.begin(), got.end());
        const std::vector<double> ref = spmv_reference(A, x64, alpha, beta, y64);
        out.strict_ratio = max_error_ratio(actual, ref, default_tolerance(dtype));
        if (out.strict_ratio > 1.0) {
            out.relaxed_ratio = max_error_ratio(actual, ref, relaxed_tolerance(dtype));
            out.relaxed_used = true;
        }
    }

    flagsparseDestroyDnVec(vecY);
    flagsparseDestroyDnVec(vecX);
    flagsparseDestroySpMat(matA);
    return out;
}

// Report the ratio, not just PASS/FAIL -- spec §6.3.1 asks for the number so
// precision work can be tracked over time.
void expect_close(const RunResult& r, const char* label,
                  flagsparseHandle_t handle_for_detail) {
    const char* detail = "";
    flagsparseGetLastErrorString(handle_for_detail, &detail);
    ASSERT_EQ(r.status, FLAGSPARSE_STATUS_SUCCESS)
        << label << ": " << status_name(r.status) << " -- " << detail;
    if (!r.relaxed_used) {
        EXPECT_LE(r.strict_ratio, 1.0) << label << " max_error_ratio=" << r.strict_ratio;
        // Spec §6.3.1 wants the number reported, not just PASS/FAIL, so precision
        // work can be tracked. RecordProperty is a Test member, so print instead.
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

class SpMVAccuracy : public ::testing::Test {
  protected:
    Handle handle;
    void SetUp() override {
        if (handle.h == nullptr) GTEST_SKIP() << "no accelerator available";
        static bool announced = false;
        if (!announced) { print_backend_banner(); announced = true; }
    }
};

TEST_F(SpMVAccuracy, Float32MatchesHostReference) {
    for (auto shape : {std::pair<int64_t, int64_t>{64, 96},
                       {512, 512},
                       {1, 32}}) {
        const CsrMatrix A = random_csr(shape.first, shape.second, 0.05, 1234);
        const RunResult r = run_spmv<float>(handle.h, A, FLAGSPARSE_R_32F, 1.0, 0.0, 7);
        expect_close(r, "fp32", handle.h);
    }
}

TEST_F(SpMVAccuracy, Float64MatchesHostReference) {
    const CsrMatrix A = random_csr(256, 320, 0.05, 99);
    const RunResult r = run_spmv<double>(handle.h, A, FLAGSPARSE_R_64F, 1.0, 0.0, 11);
    expect_close(r, "fp64", handle.h);
}

TEST_F(SpMVAccuracy, AlphaBetaAreApplied) {
    const CsrMatrix A = random_csr(128, 160, 0.08, 5);
    const RunResult r = run_spmv<double>(handle.h, A, FLAGSPARSE_R_64F, -2.5, 0.75, 3);
    expect_close(r, "fp64_alpha_beta", handle.h);
}

// A matrix with empty rows still has to run: beta must reach every y entry.
TEST_F(SpMVAccuracy, EmptyRowsStillApplyBeta) {
    CsrMatrix A = random_csr(64, 64, 0.0, 17);   // density 0 -> no nonzeros at all
    ASSERT_EQ(A.nnz, 0);
    const RunResult r = run_spmv<double>(handle.h, A, FLAGSPARSE_R_64F, 1.0, 2.0, 4);
    expect_close(r, "fp64_empty_rows", handle.h);
}

TEST_F(SpMVAccuracy, RejectsMismatchedVectorLength) {
    const CsrMatrix A = random_csr(32, 48, 0.1, 2);
    DeviceBuffer d_val = DeviceBuffer::from(std::vector<float>(A.values.begin(), A.values.end()));
    DeviceBuffer d_col = DeviceBuffer::from(A.indices);
    DeviceBuffer d_ptr = DeviceBuffer::from(A.indptr);
    DeviceBuffer d_x   = DeviceBuffer::from(std::vector<float>(A.cols + 1, 1.0f));
    DeviceBuffer d_y   = DeviceBuffer::from(std::vector<float>(A.rows, 0.0f));

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    ASSERT_EQ(flagsparseCreateCsr(&matA, A.rows, A.cols, A.nnz, d_ptr.get(), d_col.get(),
                                  d_val.get(), FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F),
              FLAGSPARSE_STATUS_SUCCESS);
    // x is one element too long for A's column count.
    ASSERT_EQ(flagsparseCreateDnVec(&vecX, A.cols + 1, d_x.get(), FLAGSPARSE_R_32F),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vecY, A.rows, d_y.get(), FLAGSPARSE_R_32F),
              FLAGSPARSE_STATUS_SUCCESS);

    const float one = 1.0f, zero = 0.0f;
    EXPECT_EQ(flagsparseSpMV(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA,
                             vecX, &zero, vecY, FLAGSPARSE_R_32F,
                             FLAGSPARSE_SPMV_ALG_DEFAULT, nullptr),
              FLAGSPARSE_STATUS_INVALID_VALUE);

    flagsparseDestroyDnVec(vecY);
    flagsparseDestroyDnVec(vecX);
    flagsparseDestroySpMat(matA);
}

// Unported paths must say NOT_SUPPORTED, never crash (spec §4.4).
TEST_F(SpMVAccuracy, TransposeReportsNotSupported) {
    const CsrMatrix A = random_csr(32, 32, 0.1, 8);
    DeviceBuffer d_val = DeviceBuffer::from(std::vector<float>(A.values.begin(), A.values.end()));
    DeviceBuffer d_col = DeviceBuffer::from(A.indices);
    DeviceBuffer d_ptr = DeviceBuffer::from(A.indptr);
    DeviceBuffer d_x   = DeviceBuffer::from(std::vector<float>(A.rows, 1.0f));
    DeviceBuffer d_y   = DeviceBuffer::from(std::vector<float>(A.cols, 0.0f));

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    flagsparseCreateCsr(&matA, A.rows, A.cols, A.nnz, d_ptr.get(), d_col.get(), d_val.get(),
                        FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                        FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F);
    flagsparseCreateDnVec(&vecX, A.rows, d_x.get(), FLAGSPARSE_R_32F);
    flagsparseCreateDnVec(&vecY, A.cols, d_y.get(), FLAGSPARSE_R_32F);
    const float one = 1.0f, zero = 0.0f;
    EXPECT_EQ(flagsparseSpMV(handle.h, FLAGSPARSE_OPERATION_TRANSPOSE, &one, matA, vecX,
                             &zero, vecY, FLAGSPARSE_R_32F,
                             FLAGSPARSE_SPMV_ALG_DEFAULT, nullptr),
              FLAGSPARSE_STATUS_NOT_SUPPORTED);
    flagsparseDestroyDnVec(vecY);
    flagsparseDestroyDnVec(vecX);
    flagsparseDestroySpMat(matA);
}

// ------------------------------------------------------- COO and CSC ---

// random_csr emits rows in order, so expanding it gives a row-sorted COO --
// which is what both cuSPARSE and this implementation require.
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

// The same matrix in CSC: column offsets, row indices, values reordered.
struct CscMatrix {
    std::vector<int32_t> colptr, rowind;
    std::vector<double> values;
};

CscMatrix csr_to_csc(const CsrMatrix& A) {
    CscMatrix C;
    C.colptr.assign(static_cast<std::size_t>(A.cols) + 1, 0);
    for (int32_t c : A.indices) C.colptr[static_cast<std::size_t>(c) + 1]++;
    for (int64_t c = 0; c < A.cols; ++c) {
        C.colptr[static_cast<std::size_t>(c) + 1] += C.colptr[static_cast<std::size_t>(c)];
    }
    C.rowind.assign(static_cast<std::size_t>(A.nnz), 0);
    C.values.assign(static_cast<std::size_t>(A.nnz), 0.0);
    std::vector<int32_t> cursor(C.colptr.begin(), C.colptr.end() - 1);
    for (int64_t r = 0; r < A.rows; ++r) {
        for (int32_t p = A.indptr[static_cast<std::size_t>(r)];
             p < A.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
            const std::size_t slot =
                static_cast<std::size_t>(cursor[static_cast<std::size_t>(A.indices[p])]++);
            C.rowind[slot] = static_cast<int32_t>(r);
            C.values[slot] = A.values[static_cast<std::size_t>(p)];
        }
    }
    return C;
}

// y = alpha * A^T * x + beta * y, fp64 on the host.
std::vector<double> spmv_trans_reference(const CsrMatrix& A, const std::vector<double>& x,
                                         double alpha, double beta,
                                         const std::vector<double>& y_in) {
    std::vector<double> acc(static_cast<std::size_t>(A.cols), 0.0);
    for (int64_t r = 0; r < A.rows; ++r) {
        for (int32_t p = A.indptr[static_cast<std::size_t>(r)];
             p < A.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
            acc[static_cast<std::size_t>(A.indices[p])] +=
                A.values[static_cast<std::size_t>(p)] * x[static_cast<std::size_t>(r)];
        }
    }
    std::vector<double> y(acc.size(), 0.0);
    for (std::size_t i = 0; i < acc.size(); ++i) {
        y[i] = alpha * acc[i] + (beta == 0.0 ? 0.0 : beta * y_in[i]);
    }
    return y;
}

// One SpMV in a chosen format/direction, against the fp64 host reference. The
// dense operands are identical across formats on purpose: a disagreement is
// then unambiguously the sparse path.
template <typename T>
RunResult run_spmv_fmt(flagsparseHandle_t handle, const CsrMatrix& A,
                       flagsparseFormat_t format, flagsparseOperation_t opA,
                       flagsparseDataType_t dtype, double alpha, double beta,
                       uint32_t seed,
                       flagsparseSpMVAlg_t alg = FLAGSPARSE_SPMV_ALG_DEFAULT) {
    RunResult out;
    const bool trans = (opA != FLAGSPARSE_OPERATION_NON_TRANSPOSE);
    const int64_t n_x = trans ? A.rows : A.cols;
    const int64_t n_y = trans ? A.cols : A.rows;

    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> x64(static_cast<std::size_t>(n_x));
    std::vector<double> y64(static_cast<std::size_t>(n_y));
    for (auto& v : x64) v = dist(rng);
    for (auto& v : y64) v = dist(rng);
    const std::vector<T> x(x64.begin(), x64.end());
    const std::vector<T> y(y64.begin(), y64.end());

    const CscMatrix csc = csr_to_csc(A);
    const std::vector<double>& vals64 =
        (format == FLAGSPARSE_FORMAT_CSC) ? csc.values : A.values;
    const std::vector<T> values(vals64.begin(), vals64.end());

    DeviceBuffer d_val = DeviceBuffer::from(values);
    DeviceBuffer d_x   = DeviceBuffer::from(x);
    DeviceBuffer d_y   = DeviceBuffer::from(y);
    DeviceBuffer d_ptr = DeviceBuffer::from(
        format == FLAGSPARSE_FORMAT_CSC ? csc.colptr : A.indptr);
    DeviceBuffer d_idx = DeviceBuffer::from(
        format == FLAGSPARSE_FORMAT_CSC ? csc.rowind : A.indices);
    DeviceBuffer d_row = DeviceBuffer::from(coo_row_indices(A));

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    switch (format) {
        case FLAGSPARSE_FORMAT_COO:
            out.status = flagsparseCreateCoo(&matA, A.rows, A.cols, A.nnz, d_row.get(),
                                             d_idx.get(), d_val.get(),
                                             FLAGSPARSE_INDEX_32I,
                                             FLAGSPARSE_INDEX_BASE_ZERO, dtype);
            break;
        case FLAGSPARSE_FORMAT_CSC:
            out.status = flagsparseCreateCsc(&matA, A.rows, A.cols, A.nnz, d_ptr.get(),
                                             d_idx.get(), d_val.get(),
                                             FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                             FLAGSPARSE_INDEX_BASE_ZERO, dtype);
            break;
        default:
            out.status = flagsparseCreateCsr(&matA, A.rows, A.cols, A.nnz, d_ptr.get(),
                                             d_idx.get(), d_val.get(),
                                             FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                             FLAGSPARSE_INDEX_BASE_ZERO, dtype);
            break;
    }
    if (out.status != FLAGSPARSE_STATUS_SUCCESS) return out;
    flagsparseCreateDnVec(&vecX, n_x, d_x.get(), dtype);
    flagsparseCreateDnVec(&vecY, n_y, d_y.get(), dtype);

    const T alpha_t = static_cast<T>(alpha);
    const T beta_t  = static_cast<T>(beta);

    size_t buffer_size = 0;
    out.status = flagsparseSpMV_bufferSize(handle, opA, &alpha_t, matA, vecX, &beta_t,
                                           vecY, dtype, alg, &buffer_size);
    DeviceBuffer scratch(buffer_size);
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.status = flagsparseSpMV_preprocess(handle, opA, &alpha_t, matA, vecX,
                                               &beta_t, vecY, dtype, alg, scratch.get());
    }
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.status = flagsparseSpMV(handle, opA, &alpha_t, matA, vecX, &beta_t, vecY,
                                    dtype, alg, scratch.get());
    }
    dev_sync();

    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        const std::vector<T> got = d_y.download<T>(static_cast<std::size_t>(n_y));
        const std::vector<double> actual(got.begin(), got.end());
        const std::vector<double> ref =
            trans ? spmv_trans_reference(A, x64, alpha, beta, y64)
                  : spmv_reference(A, x64, alpha, beta, y64);
        out.strict_ratio = max_error_ratio(actual, ref, default_tolerance(dtype));
        if (out.strict_ratio > 1.0) {
            out.relaxed_ratio = max_error_ratio(actual, ref, relaxed_tolerance(dtype));
            out.relaxed_used = true;
        }
    }

    flagsparseDestroyDnVec(vecY);
    flagsparseDestroyDnVec(vecX);
    flagsparseDestroySpMat(matA);
    return out;
}

// Complex SpMV. A separate runner rather than a template parameter: complex
// operands are interleaved real/imag pairs of the COMPONENT dtype, and alpha and
// beta arrive split the same way -- there is no single T that covers both.
template <typename R>
RunResult run_spmv_complex(flagsparseHandle_t handle, const CsrMatrix& A,
                           flagsparseFormat_t format, flagsparseDataType_t dtype,
                           std::complex<double> alpha, std::complex<double> beta,
                           uint32_t seed,
                           flagsparseSpMVAlg_t alg = FLAGSPARSE_SPMV_ALG_DEFAULT) {
    using C64 = std::complex<double>;
    RunResult out;
    const auto m = static_cast<std::size_t>(A.rows);
    const auto n = static_cast<std::size_t>(A.cols);

    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<C64> a_vals(static_cast<std::size_t>(A.nnz));
    for (std::size_t i = 0; i < a_vals.size(); ++i) {
        a_vals[i] = C64(A.values[i], dist(rng));
    }
    std::vector<C64> x64(n), y64(m);
    for (auto& v : x64) v = C64(dist(rng), dist(rng));
    for (auto& v : y64) v = C64(dist(rng), dist(rng));

    std::vector<R> a_dev(a_vals.size() * 2), x_dev(n * 2), y_dev(m * 2);
    for (std::size_t i = 0; i < a_vals.size(); ++i) {
        a_dev[i * 2]     = static_cast<R>(a_vals[i].real());
        a_dev[i * 2 + 1] = static_cast<R>(a_vals[i].imag());
    }
    for (std::size_t i = 0; i < n; ++i) {
        x_dev[i * 2] = static_cast<R>(x64[i].real());
        x_dev[i * 2 + 1] = static_cast<R>(x64[i].imag());
    }
    for (std::size_t i = 0; i < m; ++i) {
        y_dev[i * 2] = static_cast<R>(y64[i].real());
        y_dev[i * 2 + 1] = static_cast<R>(y64[i].imag());
    }

    DeviceBuffer d_val = DeviceBuffer::from(a_dev);
    DeviceBuffer d_col = DeviceBuffer::from(A.indices);
    DeviceBuffer d_ptr = DeviceBuffer::from(A.indptr);
    DeviceBuffer d_row = DeviceBuffer::from(coo_row_indices_of(A));
    DeviceBuffer d_x   = DeviceBuffer::from(x_dev);
    DeviceBuffer d_y   = DeviceBuffer::from(y_dev);

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    out.status =
        (format == FLAGSPARSE_FORMAT_COO)
            ? flagsparseCreateCoo(&matA, A.rows, A.cols, A.nnz, d_row.get(),
                                  d_col.get(), d_val.get(), FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_BASE_ZERO, dtype)
            : flagsparseCreateCsr(&matA, A.rows, A.cols, A.nnz, d_ptr.get(),
                                  d_col.get(), d_val.get(), FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO,
                                  dtype);
    if (out.status != FLAGSPARSE_STATUS_SUCCESS) return out;
    flagsparseCreateDnVec(&vecX, A.cols, d_x.get(), dtype);
    flagsparseCreateDnVec(&vecY, A.rows, d_y.get(), dtype);

    const R alpha_t[2] = {static_cast<R>(alpha.real()), static_cast<R>(alpha.imag())};
    const R beta_t[2]  = {static_cast<R>(beta.real()),  static_cast<R>(beta.imag())};
    size_t buffer_size = 0;
    out.status = flagsparseSpMV_bufferSize(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                           alpha_t, matA, vecX, beta_t, vecY, dtype, alg,
                                           &buffer_size);
    DeviceBuffer scratch(buffer_size);
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.status = flagsparseSpMV_preprocess(handle,
                                               FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                               alpha_t, matA, vecX, beta_t, vecY, dtype,
                                               alg, scratch.get());
    }
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.status = flagsparseSpMV(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE, alpha_t,
                                    matA, vecX, beta_t, vecY, dtype, alg, scratch.get());
    }
    dev_sync();

    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        const std::vector<R> got = d_y.download<R>(m * 2);
        // Real and imaginary parts compared as one flat vector: the result is
        // wrong if either component is.
        std::vector<double> actual, ref;
        actual.reserve(m * 2); ref.reserve(m * 2);
        for (int64_t r = 0; r < A.rows; ++r) {
            C64 acc(0.0, 0.0);
            for (int32_t p = A.indptr[static_cast<std::size_t>(r)];
                 p < A.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
                acc += a_vals[static_cast<std::size_t>(p)] *
                       x64[static_cast<std::size_t>(A.indices[static_cast<std::size_t>(p)])];
            }
            C64 want = alpha * acc;
            if (beta != C64(0.0, 0.0)) want += beta * y64[static_cast<std::size_t>(r)];
            actual.push_back(static_cast<double>(got[static_cast<std::size_t>(r) * 2]));
            actual.push_back(static_cast<double>(got[static_cast<std::size_t>(r) * 2 + 1]));
            ref.push_back(want.real());
            ref.push_back(want.imag());
        }
        out.strict_ratio = max_error_ratio(actual, ref, default_tolerance(dtype));
        if (out.strict_ratio > 1.0) {
            out.relaxed_ratio = max_error_ratio(actual, ref, relaxed_tolerance(dtype));
            out.relaxed_used = true;
        }
    }
    flagsparseDestroyDnVec(vecY);
    flagsparseDestroyDnVec(vecX);
    flagsparseDestroySpMat(matA);
    return out;
}

// The four complex variants the operator registry lists: {csr, coo} x {c32, c64}.
TEST_F(SpMVAccuracy, ComplexCsrAndCoo) {
    for (auto shape : {std::pair<int64_t, int64_t>{64, 96}, {257, 129}}) {
        const CsrMatrix A = random_csr(shape.first, shape.second, 0.05, 1234);
        for (auto fmt : {FLAGSPARSE_FORMAT_CSR, FLAGSPARSE_FORMAT_COO}) {
            const char* f = (fmt == FLAGSPARSE_FORMAT_COO) ? "coo" : "csr";
            expect_close(run_spmv_complex<float>(handle.h, A, fmt, FLAGSPARSE_C_32F,
                                                 std::complex<double>(1.0, 0.0),
                                                 std::complex<double>(0.0, 0.0), 7),
                         (std::string(f) + "_c32").c_str(), handle.h);
            // Complex alpha AND beta: only their real halves reaching the kernel
            // would still look plausible without this.
            expect_close(run_spmv_complex<double>(handle.h, A, fmt, FLAGSPARSE_C_64F,
                                                  std::complex<double>(1.5, -0.75),
                                                  std::complex<double>(-0.5, 0.25), 11),
                         (std::string(f) + "_c64_alpha_beta").c_str(), handle.h);
        }
    }
}

// COO_ALG2 routes complex through the CSR kernel now, so the two COO routes must
// agree on complex the way they already do on real.
TEST_F(SpMVAccuracy, ComplexCooRoutesAgree) {
    const CsrMatrix A = random_csr(200, 150, 0.04, 31);
    for (auto alg : {FLAGSPARSE_SPMV_COO_ALG1, FLAGSPARSE_SPMV_COO_ALG2}) {
        expect_close(run_spmv_complex<double>(handle.h, A, FLAGSPARSE_FORMAT_COO,
                                              FLAGSPARSE_C_64F,
                                              std::complex<double>(1.75, 0.5),
                                              std::complex<double>(-1.0, 0.0), 13, alg),
                     alg == FLAGSPARSE_SPMV_COO_ALG2 ? "coo_c64_alg2" : "coo_c64_alg1",
                     handle.h);
    }
}

TEST_F(SpMVAccuracy, CooMatchesHostReference) {
    for (auto shape : {std::pair<int64_t, int64_t>{64, 96}, {257, 129}, {1, 32}}) {
        const CsrMatrix A = random_csr(shape.first, shape.second, 0.05, 1234);
        expect_close(run_spmv_fmt<float>(handle.h, A, FLAGSPARSE_FORMAT_COO,
                                         FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                         FLAGSPARSE_R_32F, 1.0, 0.0, 7),
                     "coo_fp32", handle.h);
        expect_close(run_spmv_fmt<double>(handle.h, A, FLAGSPARSE_FORMAT_COO,
                                          FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                          FLAGSPARSE_R_64F, -2.5, 0.75, 11),
                     "coo_fp64_alpha_beta", handle.h);
    }
}

// Interior rows with no nonzeros must still pick up beta * y -- the COO segment
// route only visits rows that appear, unless segments are whole rows.
TEST_F(SpMVAccuracy, CooEmptyRowsStillApplyBeta) {
    const CsrMatrix sparse_rows = random_csr(256, 64, 0.01, 91);
    expect_close(run_spmv_fmt<double>(handle.h, sparse_rows, FLAGSPARSE_FORMAT_COO,
                                      FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                      FLAGSPARSE_R_64F, 2.0, -1.5, 4),
                 "coo_interior_empty_rows", handle.h);
    CsrMatrix none = random_csr(64, 64, 0.0, 17);
    ASSERT_EQ(none.nnz, 0);
    expect_close(run_spmv_fmt<double>(handle.h, none, FLAGSPARSE_FORMAT_COO,
                                      FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                      FLAGSPARSE_R_64F, 1.0, 2.0, 4),
                 "coo_all_empty", handle.h);
}

// CSC gives BOTH directions. The transposed one is deterministic (one program
// per column); the non-transposed one scatters with atomics and takes its beta
// from the dense prologue, so it is the one that would break first if that
// prologue were wrong.
TEST_F(SpMVAccuracy, CscBothDirections) {
    for (auto shape : {std::pair<int64_t, int64_t>{64, 96}, {129, 257}}) {
        const CsrMatrix A = random_csr(shape.first, shape.second, 0.05, 55);
        expect_close(run_spmv_fmt<double>(handle.h, A, FLAGSPARSE_FORMAT_CSC,
                                          FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                          FLAGSPARSE_R_64F, 1.0, 0.0, 3),
                     "csc_non", handle.h);
        expect_close(run_spmv_fmt<double>(handle.h, A, FLAGSPARSE_FORMAT_CSC,
                                          FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                          FLAGSPARSE_R_64F, -2.5, 0.75, 3),
                     "csc_non_alpha_beta", handle.h);
        expect_close(run_spmv_fmt<double>(handle.h, A, FLAGSPARSE_FORMAT_CSC,
                                          FLAGSPARSE_OPERATION_TRANSPOSE,
                                          FLAGSPARSE_R_64F, 1.5, -0.5, 3),
                     "csc_trans_alpha_beta", handle.h);
        expect_close(run_spmv_fmt<float>(handle.h, A, FLAGSPARSE_FORMAT_CSC,
                                         FLAGSPARSE_OPERATION_TRANSPOSE,
                                         FLAGSPARSE_R_32F, 1.0, 0.0, 3),
                     "csc_trans_fp32", handle.h);
    }
}

// A real matrix has no imaginary part to conjugate, so CONJUGATE_TRANSPOSE must
// give exactly the TRANSPOSE answer rather than being refused.
TEST_F(SpMVAccuracy, CscConjugateTransposeOnRealIsTranspose) {
    const CsrMatrix A = random_csr(48, 72, 0.08, 61);
    expect_close(run_spmv_fmt<double>(handle.h, A, FLAGSPARSE_FORMAT_CSC,
                                      FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE,
                                      FLAGSPARSE_R_64F, 1.0, 0.0, 5),
                 "csc_conj_trans_real", handle.h);
}

// COO_ALG2 runs the CSR row-parallel kernel over the offsets the COO path had
// to build anyway -- the "COO to CSR" route, with nothing converted or copied.
// It must agree with the segment route, not merely succeed.
TEST_F(SpMVAccuracy, CooToCsrRouteAgrees) {
    const CsrMatrix A = random_csr(200, 150, 0.04, 31);
    for (double beta : {0.0, -1.5}) {
        expect_close(run_spmv_fmt<double>(handle.h, A, FLAGSPARSE_FORMAT_COO,
                                          FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                          FLAGSPARSE_R_64F, 1.75, beta, 13,
                                          FLAGSPARSE_SPMV_COO_ALG1),
                     "coo_alg1_segments", handle.h);
        expect_close(run_spmv_fmt<double>(handle.h, A, FLAGSPARSE_FORMAT_COO,
                                          FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                          FLAGSPARSE_R_64F, 1.75, beta, 13,
                                          FLAGSPARSE_SPMV_COO_ALG2),
                     "coo_alg2_tocsr", handle.h);
    }
    // Interior empty rows are the case the to-CSR route could get wrong if the
    // offsets were run-compressed rather than one entry per row.
    const CsrMatrix sparse_rows = random_csr(256, 64, 0.01, 91);
    expect_close(run_spmv_fmt<double>(handle.h, sparse_rows, FLAGSPARSE_FORMAT_COO,
                                      FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                      FLAGSPARSE_R_64F, 2.0, -1.5, 4,
                                      FLAGSPARSE_SPMV_COO_ALG2),
                 "coo_alg2_empty_rows", handle.h);
}

// ------------------------------------------------------------------- BSR ---

// A dense block matrix in BSR: brows x bcols blocks of block_dim x block_dim,
// stored row-major within a block. Built dense-first so the reference is
// obvious, then compressed by dropping all-zero blocks.
struct BsrMatrix {
    int64_t brows = 0, bcols = 0, block_dim = 0, bnnz = 0;
    std::vector<int32_t> indptr, indices;
    std::vector<double> values;          // bnnz * block_dim * block_dim
    std::vector<double> dense;           // (brows*bd) x (bcols*bd), row-major
    int64_t rows() const { return brows * block_dim; }
    int64_t cols() const { return bcols * block_dim; }
};

BsrMatrix random_bsr(int64_t brows, int64_t bcols, int64_t block_dim, double density,
                     uint32_t seed) {
    BsrMatrix B;
    B.brows = brows; B.bcols = bcols; B.block_dim = block_dim;
    B.dense.assign(static_cast<std::size_t>(B.rows() * B.cols()), 0.0);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double> value(0.0, 1.0);
    B.indptr.assign(static_cast<std::size_t>(brows) + 1, 0);
    for (int64_t br = 0; br < brows; ++br) {
        for (int64_t bc = 0; bc < bcols; ++bc) {
            if (unit(rng) >= density) continue;
            B.indices.push_back(static_cast<int32_t>(bc));
            for (int64_t i = 0; i < block_dim; ++i) {
                for (int64_t j = 0; j < block_dim; ++j) {
                    const double v = value(rng);
                    B.values.push_back(v);
                    B.dense[static_cast<std::size_t>((br * block_dim + i) * B.cols() +
                                                     bc * block_dim + j)] = v;
                }
            }
        }
        B.indptr[static_cast<std::size_t>(br) + 1] = static_cast<int32_t>(B.indices.size());
    }
    B.bnnz = static_cast<int64_t>(B.indices.size());
    return B;
}

std::vector<double> dense_spmv_reference(const std::vector<double>& A, int64_t m,
                                         int64_t n, const std::vector<double>& x,
                                         bool trans, double alpha, double beta,
                                         const std::vector<double>& y_in) {
    const int64_t out = trans ? n : m;
    std::vector<double> y(static_cast<std::size_t>(out), 0.0);
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            const double a = A[static_cast<std::size_t>(i * n + j)];
            if (trans) y[static_cast<std::size_t>(j)] += a * x[static_cast<std::size_t>(i)];
            else       y[static_cast<std::size_t>(i)] += a * x[static_cast<std::size_t>(j)];
        }
    }
    for (std::size_t i = 0; i < y.size(); ++i) {
        y[i] = alpha * y[i] + (beta == 0.0 ? 0.0 : beta * y_in[i]);
    }
    return y;
}

template <typename T>
RunResult run_spmv_bsr(flagsparseHandle_t handle, const BsrMatrix& B,
                       flagsparseOperation_t opA, flagsparseDataType_t dtype,
                       double alpha, double beta, uint32_t seed) {
    RunResult out;
    const bool trans = (opA != FLAGSPARSE_OPERATION_NON_TRANSPOSE);
    const int64_t n_x = trans ? B.rows() : B.cols();
    const int64_t n_y = trans ? B.cols() : B.rows();

    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> x64(static_cast<std::size_t>(n_x));
    std::vector<double> y64(static_cast<std::size_t>(n_y));
    for (auto& v : x64) v = dist(rng);
    for (auto& v : y64) v = dist(rng);

    const std::vector<T> values(B.values.begin(), B.values.end());
    const std::vector<T> x(x64.begin(), x64.end());
    const std::vector<T> y(y64.begin(), y64.end());
    DeviceBuffer d_val = DeviceBuffer::from(values);
    DeviceBuffer d_ptr = DeviceBuffer::from(B.indptr);
    DeviceBuffer d_idx = DeviceBuffer::from(B.indices);
    DeviceBuffer d_x   = DeviceBuffer::from(x);
    DeviceBuffer d_y   = DeviceBuffer::from(y);

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    out.status = flagsparseCreateBsr(&matA, B.brows, B.bcols, B.bnnz, B.block_dim,
                                     B.block_dim, d_ptr.get(), d_idx.get(), d_val.get(),
                                     FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                     FLAGSPARSE_INDEX_BASE_ZERO, dtype,
                                     FLAGSPARSE_ORDER_ROW);
    if (out.status != FLAGSPARSE_STATUS_SUCCESS) return out;
    flagsparseCreateDnVec(&vecX, n_x, d_x.get(), dtype);
    flagsparseCreateDnVec(&vecY, n_y, d_y.get(), dtype);

    const T alpha_t = static_cast<T>(alpha);
    const T beta_t  = static_cast<T>(beta);
    out.status = flagsparseSpMV(handle, opA, &alpha_t, matA, vecX, &beta_t, vecY, dtype,
                                FLAGSPARSE_SPMV_ALG_DEFAULT, nullptr);
    dev_sync();

    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        const std::vector<T> got = d_y.download<T>(static_cast<std::size_t>(n_y));
        const std::vector<double> actual(got.begin(), got.end());
        const std::vector<double> ref = dense_spmv_reference(
            B.dense, B.rows(), B.cols(), x64, trans, alpha, beta, y64);
        out.strict_ratio = max_error_ratio(actual, ref, default_tolerance(dtype));
        if (out.strict_ratio > 1.0) {
            out.relaxed_ratio = max_error_ratio(actual, ref, relaxed_tolerance(dtype));
            out.relaxed_used = true;
        }
    }
    flagsparseDestroyDnVec(vecY);
    flagsparseDestroyDnVec(vecX);
    flagsparseDestroySpMat(matA);
    return out;
}

TEST_F(SpMVAccuracy, BsrBothDirections) {
    for (int64_t bd : {1, 2, 4, 8}) {
        const BsrMatrix B = random_bsr(12, 9, bd, 0.3, 71);
        expect_close(run_spmv_bsr<double>(handle.h, B, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                          FLAGSPARSE_R_64F, 1.0, 0.0, 5),
                     ("bsr_non_bd" + std::to_string(bd)).c_str(), handle.h);
        expect_close(run_spmv_bsr<double>(handle.h, B, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                          FLAGSPARSE_R_64F, -2.5, 0.75, 5),
                     ("bsr_non_ab_bd" + std::to_string(bd)).c_str(), handle.h);
        expect_close(run_spmv_bsr<double>(handle.h, B, FLAGSPARSE_OPERATION_TRANSPOSE,
                                          FLAGSPARSE_R_64F, 1.5, -0.5, 5),
                     ("bsr_trans_bd" + std::to_string(bd)).c_str(), handle.h);
    }
}

// More block rows than fit one BLOCK_NNZ segment, so the grid's z dimension is
// exercised -- that is the axis SEG_FROM_GRID replaced a host-side loop on.
TEST_F(SpMVAccuracy, BsrMultipleSegments) {
    const BsrMatrix B = random_bsr(8, 80, 4, 0.9, 73);
    ASSERT_GT(B.bnnz, 8 * 32);   // some block row must need more than one segment
    expect_close(run_spmv_bsr<double>(handle.h, B, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                      FLAGSPARSE_R_64F, 1.25, -0.5, 9),
                 "bsr_multi_segment", handle.h);
    expect_close(run_spmv_bsr<float>(handle.h, B, FLAGSPARSE_OPERATION_TRANSPOSE,
                                     FLAGSPARSE_R_32F, 1.0, 0.0, 9),
                 "bsr_multi_segment_trans_fp32", handle.h);
}

// Non-square blocks and column-major blocks would be read wrongly, so they are
// refused rather than silently mis-indexed.
TEST_F(SpMVAccuracy, BsrRejectsUnsupportedBlockShapes) {
    const BsrMatrix B = random_bsr(4, 4, 2, 0.5, 77);
    const std::vector<double> values(B.values.begin(), B.values.end());
    DeviceBuffer d_val = DeviceBuffer::from(values);
    DeviceBuffer d_ptr = DeviceBuffer::from(B.indptr);
    DeviceBuffer d_idx = DeviceBuffer::from(B.indices);
    DeviceBuffer d_x   = DeviceBuffer::from(std::vector<double>(B.cols(), 1.0));
    DeviceBuffer d_y   = DeviceBuffer::from(std::vector<double>(B.rows(), 0.0));

    const double one = 1.0, zero = 0.0;
    struct Bad { int64_t rbd, cbd; flagsparseOrder_t order; };
    for (const Bad& bad : {Bad{2, 4, FLAGSPARSE_ORDER_ROW},
                           Bad{2, 2, FLAGSPARSE_ORDER_COL}}) {
        flagsparseSpMatDescr_t matA = nullptr;
        flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
        ASSERT_EQ(flagsparseCreateBsr(&matA, B.brows, B.bcols, B.bnnz, bad.rbd, bad.cbd,
                                      d_ptr.get(), d_idx.get(), d_val.get(),
                                      FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                      FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_64F,
                                      bad.order),
                  FLAGSPARSE_STATUS_SUCCESS);
        flagsparseCreateDnVec(&vecX, B.bcols * bad.cbd, d_x.get(), FLAGSPARSE_R_64F);
        flagsparseCreateDnVec(&vecY, B.brows * bad.rbd, d_y.get(), FLAGSPARSE_R_64F);
        EXPECT_EQ(flagsparseSpMV(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA,
                                 vecX, &zero, vecY, FLAGSPARSE_R_64F,
                                 FLAGSPARSE_SPMV_ALG_DEFAULT, nullptr),
                  FLAGSPARSE_STATUS_NOT_SUPPORTED);
        flagsparseDestroyDnVec(vecY);
        flagsparseDestroyDnVec(vecX);
        flagsparseDestroySpMat(matA);
    }
}

}  // namespace
