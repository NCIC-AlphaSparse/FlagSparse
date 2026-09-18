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


// Accuracy tests for flagsparseSpSV: op(A) y = alpha x, A triangular.
//
// A solve is checked by substitution rather than against a host solve: build a
// triangular A and a known y, compute x = A y on the host in fp64, then ask the
// library for y back. That keeps the reference exact and makes a wrong fill
// mode or a skipped term show up immediately.

#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <cmath>
#include <vector>

#include "common.hpp"

using namespace fstest;

namespace {

struct Handle {
    flagsparseHandle_t h = nullptr;
    Handle() { flagsparseCreate(&h); }
    ~Handle() { if (h) flagsparseDestroy(h); }
};

// A triangular CSR matrix with a well-conditioned diagonal, columns sorted
// ascending within each row (which SpSV requires).

// x = A y on the host in fp64, with the implicit unit diagonal added in.
std::vector<double> tri_apply(const TriMatrix& T, const std::vector<double>& y) {
    std::vector<double> x(static_cast<std::size_t>(T.n), 0.0);
    for (int64_t r = 0; r < T.n; ++r) {
        double acc = T.unit_diag ? y[static_cast<std::size_t>(r)] : 0.0;
        for (int32_t p = T.indptr[static_cast<std::size_t>(r)];
             p < T.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
            acc += T.values[static_cast<std::size_t>(p)] *
                   y[static_cast<std::size_t>(T.indices[static_cast<std::size_t>(p)])];
        }
        x[static_cast<std::size_t>(r)] = acc;
    }
    return x;
}

struct RunResult {
    flagsparseStatus_t status = FLAGSPARSE_STATUS_SUCCESS;
    double ratio = 0.0;
};

template <typename T>
RunResult run_spsv(flagsparseHandle_t handle, const TriMatrix& tri,
                   flagsparseDataType_t dtype, double alpha, uint32_t seed,
                   bool skip_analysis = false,
                   flagsparseFormat_t format = FLAGSPARSE_FORMAT_CSR) {
    RunResult out;
    const auto n = static_cast<std::size_t>(tri.n);

    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> y_want(n);
    for (auto& v : y_want) v = dist(rng);
    // Solve op(A) y = alpha x, so feed x = A y / alpha to get y_want back.
    std::vector<double> x64 = tri_apply(tri, y_want);
    for (auto& v : x64) v /= alpha;

    const std::vector<T> values(tri.values.begin(), tri.values.end());
    const std::vector<T> x(x64.begin(), x64.end());
    DeviceBuffer d_val = DeviceBuffer::from(values);
    DeviceBuffer d_col = DeviceBuffer::from(tri.indices);
    DeviceBuffer d_ptr = DeviceBuffer::from(tri.indptr);
    DeviceBuffer d_x   = DeviceBuffer::from(x);
    DeviceBuffer d_y   = DeviceBuffer::from(std::vector<T>(n, T(0)));

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    flagsparseSpSVDescr_t descr = nullptr;
    // The COO form is the same matrix with one row index per nonzero; lexical
    // (row, col) order is exactly what the CSR expansion produces.
    std::vector<int32_t> coo_rows;
    coo_rows.reserve(tri.indices.size());
    for (int64_t r = 0; r < tri.n; ++r) {
        for (int32_t p = tri.indptr[static_cast<std::size_t>(r)];
             p < tri.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
            coo_rows.push_back(static_cast<int32_t>(r));
        }
    }
    DeviceBuffer d_row = DeviceBuffer::from(coo_rows);
    out.status =
        (format == FLAGSPARSE_FORMAT_COO)
            ? flagsparseCreateCoo(&matA, tri.n, tri.n,
                                  static_cast<int64_t>(tri.indices.size()), d_row.get(),
                                  d_col.get(), d_val.get(), FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_BASE_ZERO, dtype)
            : flagsparseCreateCsr(&matA, tri.n, tri.n,
                                  static_cast<int64_t>(tri.indices.size()),
                                  d_ptr.get(), d_col.get(), d_val.get(),
                                  FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_BASE_ZERO, dtype);
    if (out.status != FLAGSPARSE_STATUS_SUCCESS) return out;
    const flagsparseFillMode_t fill =
        tri.lower ? FLAGSPARSE_FILL_MODE_LOWER : FLAGSPARSE_FILL_MODE_UPPER;
    const flagsparseDiagType_t diag =
        tri.unit_diag ? FLAGSPARSE_DIAG_TYPE_UNIT : FLAGSPARSE_DIAG_TYPE_NON_UNIT;
    flagsparseSpMatSetAttribute(matA, FLAGSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill));
    flagsparseSpMatSetAttribute(matA, FLAGSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag));
    flagsparseCreateDnVec(&vecX, tri.n, d_x.get(), dtype);
    flagsparseCreateDnVec(&vecY, tri.n, d_y.get(), dtype);
    flagsparseSpSV_createDescr(&descr);

    const T alpha_t = static_cast<T>(alpha);
    size_t buffer_size = 0;
    out.status = flagsparseSpSV_bufferSize(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_t, matA, vecX, vecY, dtype,
                                           FLAGSPARSE_SPSV_ALG_DEFAULT, descr,
                                           &buffer_size);
    DeviceBuffer scratch(buffer_size);
    if (out.status == FLAGSPARSE_STATUS_SUCCESS && !skip_analysis) {
        out.status = flagsparseSpSV_analysis(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                             &alpha_t, matA, vecX, vecY, dtype,
                                             FLAGSPARSE_SPSV_ALG_DEFAULT, descr,
                                             scratch.get());
    }
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.status = flagsparseSpSV_solve(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha_t, matA, vecX, vecY, dtype,
                                          FLAGSPARSE_SPSV_ALG_DEFAULT, descr);
    }
    dev_sync();

    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        const std::vector<T> got = d_y.download<T>(n);
        const std::vector<double> actual(got.begin(), got.end());
        out.ratio = max_error_ratio(actual, y_want, relaxed_tolerance(dtype));
    }

    flagsparseSpSV_destroyDescr(descr);
    flagsparseDestroyDnVec(vecY);
    flagsparseDestroyDnVec(vecX);
    flagsparseDestroySpMat(matA);
    return out;
}

// A triangular solve amplifies error along the dependency chain, so this is
// reported at the spec's relaxed tolerance from the start rather than pretending
// the strict one is meaningful for a recurrence hundreds of rows deep.
void expect_solves(const RunResult& r, const char* label, flagsparseHandle_t handle) {
    const char* detail = "";
    flagsparseGetLastErrorString(handle, &detail);
    ASSERT_EQ(r.status, FLAGSPARSE_STATUS_SUCCESS)
        << label << ": " << status_name(r.status) << " -- " << detail;
    EXPECT_LE(r.ratio, 1.0) << label << " max_error_ratio=" << r.ratio;
    std::cout << "[   RATIO   ] " << label << " max_error_ratio=" << r.ratio << std::endl;
}

class SpSVAccuracy : public ::testing::Test {
  protected:
    Handle handle;
    void SetUp() override {
        if (handle.h == nullptr) GTEST_SKIP() << "no accelerator available";
        static bool announced = false;
        if (!announced) { print_backend_banner(); announced = true; }
    }
};

TEST_F(SpSVAccuracy, AllFourTriangleAndDiagonalCombinations) {
    for (bool lower : {true, false}) {
        for (bool unit : {true, false}) {
            const TriMatrix T = random_triangular(256, 0.03, lower, unit, 17);
            const std::string label = std::string(lower ? "lower" : "upper") +
                                      (unit ? "_unit" : "_nonunit");
            expect_solves(run_spsv<double>(handle.h, T, FLAGSPARSE_R_64F, 1.0, 5),
                          label.c_str(), handle.h);
        }
    }
}

TEST_F(SpSVAccuracy, AlphaScalesTheRightHandSide) {
    const TriMatrix T = random_triangular(192, 0.04, true, false, 23);
    for (double alpha : {2.5, -0.75}) {
        expect_solves(run_spsv<double>(handle.h, T, FLAGSPARSE_R_64F, alpha, 9),
                      "fp64_alpha", handle.h);
    }
}

TEST_F(SpSVAccuracy, Float32AndSizes) {
    for (int64_t n : {1, 2, 33, 512}) {
        const TriMatrix T = random_triangular(n, 0.05, true, false, 41);
        expect_solves(run_spsv<float>(handle.h, T, FLAGSPARSE_R_32F, 1.0, 3),
                      ("fp32_n" + std::to_string(n)).c_str(), handle.h);
    }
}

// A diagonal matrix is the shallowest possible dependency chain; a dense
// triangle is the deepest. Both must work, and the deep one is where a missed
// ready flag would hang or give garbage.
TEST_F(SpSVAccuracy, DependencyChainDepth) {
    const TriMatrix diagonal = random_triangular(1024, 0.0, true, false, 51);
    expect_solves(run_spsv<double>(handle.h, diagonal, FLAGSPARSE_R_64F, 1.0, 7),
                  "diagonal_only", handle.h);
    const TriMatrix dense = random_triangular(300, 1.0, false, false, 53);
    expect_solves(run_spsv<double>(handle.h, dense, FLAGSPARSE_R_64F, 1.0, 7),
                  "dense_upper_triangle", handle.h);
}

// A COO triangular matrix is the same solve: _analysis builds the row-offsets
// array into its half of the scratch and the CSR kernel runs over that view, so
// COO must agree with CSR rather than merely succeed.
TEST_F(SpSVAccuracy, CooMatchesCsr) {
    for (bool lower : {true, false}) {
        for (bool unit : {true, false}) {
            const TriMatrix T = random_triangular(192, 0.04, lower, unit, 29);
            const std::string label = std::string("coo_") + (lower ? "lower" : "upper") +
                                      (unit ? "_unit" : "_nonunit");
            expect_solves(run_spsv<double>(handle.h, T, FLAGSPARSE_R_64F, 1.5, 11, false,
                                           FLAGSPARSE_FORMAT_COO),
                          label.c_str(), handle.h);
        }
    }
    // Bigger scratch than CSR: flags + counter AND the offsets array.
    const TriMatrix T = random_triangular(64, 0.05, true, false, 31);
    expect_solves(run_spsv<float>(handle.h, T, FLAGSPARSE_R_32F, 1.0, 13, false,
                                  FLAGSPARSE_FORMAT_COO),
                  "coo_fp32", handle.h);
}

// The complex kernel is a separate function in the operator package, so it is
// wired separately and has to be shown separately. Complex alpha is the part
// that would silently be wrong if only its real half reached the kernel.
template <typename R>
RunResult run_spsv_complex(flagsparseHandle_t handle, const TriMatrix& tri,
                           flagsparseDataType_t dtype, std::complex<double> alpha,
                           uint32_t seed,
                           flagsparseFormat_t format = FLAGSPARSE_FORMAT_CSR) {
    using C64 = std::complex<double>;
    RunResult out;
    const auto n = static_cast<std::size_t>(tri.n);
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);

    // Complex matrix values: the real part is the well-conditioned triangle, the
    // imaginary part a small perturbation that keeps it so.
    std::vector<C64> a_vals(tri.values.size());
    for (std::size_t i = 0; i < a_vals.size(); ++i) {
        a_vals[i] = C64(tri.values[i], tri.values[i] * 0.25 * dist(rng));
    }
    std::vector<C64> y_want(n);
    for (auto& v : y_want) v = C64(dist(rng), dist(rng));

    // x = A y / alpha, so the solve returns y_want.
    std::vector<C64> x64(n, C64(0.0, 0.0));
    for (int64_t r = 0; r < tri.n; ++r) {
        C64 acc = tri.unit_diag ? y_want[static_cast<std::size_t>(r)] : C64(0.0, 0.0);
        for (int32_t p = tri.indptr[static_cast<std::size_t>(r)];
             p < tri.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
            acc += a_vals[static_cast<std::size_t>(p)] *
                   y_want[static_cast<std::size_t>(tri.indices[static_cast<std::size_t>(p)])];
        }
        x64[static_cast<std::size_t>(r)] = acc / alpha;
    }

    std::vector<R> a_dev(a_vals.size() * 2), x_dev(n * 2), y_dev(n * 2, R(0));
    for (std::size_t i = 0; i < a_vals.size(); ++i) {
        a_dev[i * 2]     = static_cast<R>(a_vals[i].real());
        a_dev[i * 2 + 1] = static_cast<R>(a_vals[i].imag());
    }
    for (std::size_t i = 0; i < n; ++i) {
        x_dev[i * 2]     = static_cast<R>(x64[i].real());
        x_dev[i * 2 + 1] = static_cast<R>(x64[i].imag());
    }

    DeviceBuffer d_val = DeviceBuffer::from(a_dev);
    DeviceBuffer d_col = DeviceBuffer::from(tri.indices);
    DeviceBuffer d_ptr = DeviceBuffer::from(tri.indptr);
    DeviceBuffer d_row = DeviceBuffer::from(coo_row_indices_of(tri));
    DeviceBuffer d_x   = DeviceBuffer::from(x_dev);
    DeviceBuffer d_y   = DeviceBuffer::from(y_dev);

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    flagsparseSpSVDescr_t descr = nullptr;
    const int64_t nnz = static_cast<int64_t>(tri.indices.size());
    out.status = (format == FLAGSPARSE_FORMAT_COO)
                     ? flagsparseCreateCoo(&matA, tri.n, tri.n, nnz, d_row.get(),
                                           d_col.get(), d_val.get(),
                                           FLAGSPARSE_INDEX_32I,
                                           FLAGSPARSE_INDEX_BASE_ZERO, dtype)
                     : flagsparseCreateCsr(&matA, tri.n, tri.n, nnz, d_ptr.get(),
                                           d_col.get(), d_val.get(),
                                           FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                           FLAGSPARSE_INDEX_BASE_ZERO, dtype);
    if (out.status != FLAGSPARSE_STATUS_SUCCESS) return out;
    const flagsparseFillMode_t fill =
        tri.lower ? FLAGSPARSE_FILL_MODE_LOWER : FLAGSPARSE_FILL_MODE_UPPER;
    const flagsparseDiagType_t diag =
        tri.unit_diag ? FLAGSPARSE_DIAG_TYPE_UNIT : FLAGSPARSE_DIAG_TYPE_NON_UNIT;
    flagsparseSpMatSetAttribute(matA, FLAGSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill));
    flagsparseSpMatSetAttribute(matA, FLAGSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag));
    flagsparseCreateDnVec(&vecX, tri.n, d_x.get(), dtype);
    flagsparseCreateDnVec(&vecY, tri.n, d_y.get(), dtype);
    flagsparseSpSV_createDescr(&descr);

    const R alpha_t[2] = {static_cast<R>(alpha.real()), static_cast<R>(alpha.imag())};
    size_t buffer_size = 0;
    out.status = flagsparseSpSV_bufferSize(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                           alpha_t, matA, vecX, vecY, dtype,
                                           FLAGSPARSE_SPSV_ALG_DEFAULT, descr,
                                           &buffer_size);
    DeviceBuffer scratch(buffer_size);
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.status = flagsparseSpSV_analysis(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                             alpha_t, matA, vecX, vecY, dtype,
                                             FLAGSPARSE_SPSV_ALG_DEFAULT, descr,
                                             scratch.get());
    }
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.status = flagsparseSpSV_solve(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                          alpha_t, matA, vecX, vecY, dtype,
                                          FLAGSPARSE_SPSV_ALG_DEFAULT, descr);
    }
    dev_sync();

    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        const std::vector<R> got = d_y.download<R>(n * 2);
        std::vector<double> actual, ref;
        actual.reserve(n * 2); ref.reserve(n * 2);
        for (std::size_t i = 0; i < n; ++i) {
            actual.push_back(static_cast<double>(got[i * 2]));
            actual.push_back(static_cast<double>(got[i * 2 + 1]));
            ref.push_back(y_want[i].real());
            ref.push_back(y_want[i].imag());
        }
        out.ratio = max_error_ratio(actual, ref, relaxed_tolerance(dtype));
    }

    flagsparseSpSV_destroyDescr(descr);
    flagsparseDestroyDnVec(vecY);
    flagsparseDestroyDnVec(vecX);
    flagsparseDestroySpMat(matA);
    return out;
}

TEST_F(SpSVAccuracy, ComplexWithComplexAlpha) {
    for (bool lower : {true, false}) {
        const TriMatrix T = random_triangular(128, 0.05, lower, false, 37);
        expect_solves(run_spsv_complex<double>(handle.h, T, FLAGSPARSE_C_64F,
                                               std::complex<double>(1.5, -0.75), 19),
                      lower ? "c128_lower" : "c128_upper", handle.h);
    }
    const TriMatrix U = random_triangular(96, 0.06, true, true, 39);
    expect_solves(run_spsv_complex<float>(handle.h, U, FLAGSPARSE_C_32F,
                                          std::complex<double>(1.0, 0.5), 21),
                  "c64_unit_diag", handle.h);
}

// COO carries the other two complex variants the registry lists. It runs the
// same kernel over the offsets analysis builds, so it must agree with CSR rather
// than merely succeed.
TEST_F(SpSVAccuracy, ComplexCoo) {
    for (bool lower : {true, false}) {
        const TriMatrix T = random_triangular(128, 0.05, lower, false, 37);
        const std::string tag = std::string("coo_c128_") + (lower ? "lower" : "upper");
        expect_solves(run_spsv_complex<double>(handle.h, T, FLAGSPARSE_C_64F,
                                               std::complex<double>(1.5, -0.75), 19,
                                               FLAGSPARSE_FORMAT_COO),
                      tag.c_str(), handle.h);
    }
    const TriMatrix U = random_triangular(96, 0.06, true, true, 39);
    expect_solves(run_spsv_complex<float>(handle.h, U, FLAGSPARSE_C_32F,
                                          std::complex<double>(1.0, 0.5), 21,
                                          FLAGSPARSE_FORMAT_COO),
                  "coo_c64_unit_diag", handle.h);
}

// ------------------------------------------------------------------ SELL ---

template <typename T>
RunResult run_spsv_sell(flagsparseHandle_t handle, const SellMatrix& S,
                        flagsparseDataType_t dtype, double alpha,
                        flagsparseSpSVAlg_t alg, uint32_t seed) {
    RunResult out;
    const auto n = static_cast<std::size_t>(S.n);
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> y_want(n);
    for (auto& v : y_want) v = dist(rng);

    // x = A y / alpha, so the solve must return y_want.
    std::vector<double> x64(n, 0.0);
    for (int64_t r = 0; r < S.n; ++r) {
        double acc = 0.0;
        for (int64_t c = 0; c <= r; ++c) {
            acc += S.dense[static_cast<std::size_t>(r * S.n + c)] *
                   y_want[static_cast<std::size_t>(c)];
        }
        x64[static_cast<std::size_t>(r)] = acc / alpha;
    }

    const std::vector<T> values(S.values.begin(), S.values.end());
    const std::vector<T> x(x64.begin(), x64.end());
    DeviceBuffer d_val = DeviceBuffer::from(values);
    DeviceBuffer d_col = DeviceBuffer::from(S.cols);
    DeviceBuffer d_off = DeviceBuffer::from(S.offsets);
    DeviceBuffer d_x   = DeviceBuffer::from(x);
    DeviceBuffer d_y   = DeviceBuffer::from(std::vector<T>(n, T(0)));

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    flagsparseSpSVDescr_t descr = nullptr;
    out.status = flagsparseCreateSlicedEll(&matA, S.n, S.n, S.slice_size, d_off.get(),
                                           d_col.get(), d_val.get(),
                                           FLAGSPARSE_INDEX_32I,
                                           FLAGSPARSE_INDEX_BASE_ZERO, dtype);
    if (out.status != FLAGSPARSE_STATUS_SUCCESS) return out;
    const flagsparseFillMode_t fill = FLAGSPARSE_FILL_MODE_LOWER;
    const flagsparseDiagType_t diag = FLAGSPARSE_DIAG_TYPE_NON_UNIT;
    flagsparseSpMatSetAttribute(matA, FLAGSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill));
    flagsparseSpMatSetAttribute(matA, FLAGSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag));
    flagsparseCreateDnVec(&vecX, S.n, d_x.get(), dtype);
    flagsparseCreateDnVec(&vecY, S.n, d_y.get(), dtype);
    flagsparseSpSV_createDescr(&descr);

    const T alpha_t = static_cast<T>(alpha);
    size_t buffer_size = 0;
    out.status = flagsparseSpSV_bufferSize(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha_t, matA, vecX, vecY, dtype, alg,
                                           descr, &buffer_size);
    DeviceBuffer scratch(buffer_size);
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.status = flagsparseSpSV_analysis(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                             &alpha_t, matA, vecX, vecY, dtype, alg,
                                             descr, scratch.get());
    }
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.status = flagsparseSpSV_solve(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                          &alpha_t, matA, vecX, vecY, dtype, alg, descr);
    }
    dev_sync();

    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        const std::vector<T> got = d_y.download<T>(n);
        const std::vector<double> actual(got.begin(), got.end());
        out.ratio = max_error_ratio(actual, y_want, relaxed_tolerance(dtype));
    }
    flagsparseSpSV_destroyDescr(descr);
    flagsparseDestroyDnVec(vecY);
    flagsparseDestroyDnVec(vecX);
    flagsparseDestroySpMat(matA);
    return out;
}

// Both SELL routes solve the same system, so they must agree with the reference
// and with each other -- ALG2's per-lane slot state is the part that could
// diverge, and only on a slice whose rows have different lengths.
TEST_F(SpSVAccuracy, SellBothAlgorithms) {
    for (int64_t slice : {4, 8, 32}) {
        const SellMatrix S = random_sell_lower(256, slice, 0.04, 71);
        for (auto alg : {FLAGSPARSE_SPSV_ALG_DEFAULT, FLAGSPARSE_SPSV_SELL_ALG1,
                         FLAGSPARSE_SPSV_SELL_ALG2}) {
            const char* tag = (alg == FLAGSPARSE_SPSV_SELL_ALG2) ? "alg2"
                            : (alg == FLAGSPARSE_SPSV_SELL_ALG1) ? "alg1" : "default";
            expect_solves(run_spsv_sell<double>(handle.h, S, FLAGSPARSE_R_64F, 1.75,
                                                alg, 5),
                          ("sell_s" + std::to_string(slice) + "_" + tag).c_str(),
                          handle.h);
        }
    }
}

TEST_F(SpSVAccuracy, SellSizesAndDtype) {
    for (int64_t n : {1, 7, 64, 1024}) {
        const SellMatrix S = random_sell_lower(n, 8, 0.05, 73);
        expect_solves(run_spsv_sell<float>(handle.h, S, FLAGSPARSE_R_32F, 1.0,
                                           FLAGSPARSE_SPSV_SELL_ALG1, 3),
                      ("sell_fp32_n" + std::to_string(n)).c_str(), handle.h);
    }
    // A row longer than one slot in every slice: the padding path is only
    // exercised when rows within a slice differ in length.
    const SellMatrix ragged = random_sell_lower(512, 16, 0.10, 77);
    expect_solves(run_spsv_sell<double>(handle.h, ragged, FLAGSPARSE_R_64F, -0.5,
                                        FLAGSPARSE_SPSV_SELL_ALG2, 9),
                  "sell_ragged_alg2", handle.h);
}

// The SELL kernels have no fill-mode switch, and a SELL algorithm id on a CSR
// matrix names a format that descriptor is not in. Both are refused.
TEST_F(SpSVAccuracy, SellRejectsUpperAndCrossFormatAlg) {
    const SellMatrix S = random_sell_lower(64, 8, 0.06, 79);
    const std::vector<double> values(S.values.begin(), S.values.end());
    DeviceBuffer d_val = DeviceBuffer::from(values);
    DeviceBuffer d_col = DeviceBuffer::from(S.cols);
    DeviceBuffer d_off = DeviceBuffer::from(S.offsets);
    DeviceBuffer d_x   = DeviceBuffer::from(std::vector<double>(S.n, 1.0));
    DeviceBuffer d_y   = DeviceBuffer::from(std::vector<double>(S.n, 0.0));

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    flagsparseSpSVDescr_t descr = nullptr;
    flagsparseCreateSlicedEll(&matA, S.n, S.n, S.slice_size, d_off.get(), d_col.get(),
                              d_val.get(), FLAGSPARSE_INDEX_32I,
                              FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_64F);
    const flagsparseFillMode_t upper = FLAGSPARSE_FILL_MODE_UPPER;
    flagsparseSpMatSetAttribute(matA, FLAGSPARSE_SPMAT_FILL_MODE, &upper, sizeof(upper));
    flagsparseCreateDnVec(&vecX, S.n, d_x.get(), FLAGSPARSE_R_64F);
    flagsparseCreateDnVec(&vecY, S.n, d_y.get(), FLAGSPARSE_R_64F);
    flagsparseSpSV_createDescr(&descr);
    const double one = 1.0;
    size_t buf = 0;
    EXPECT_EQ(flagsparseSpSV_bufferSize(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                        &one, matA, vecX, vecY, FLAGSPARSE_R_64F,
                                        FLAGSPARSE_SPSV_SELL_ALG1, descr, &buf),
              FLAGSPARSE_STATUS_NOT_SUPPORTED);
    flagsparseSpSV_destroyDescr(descr);
    flagsparseDestroyDnVec(vecY);
    flagsparseDestroyDnVec(vecX);
    flagsparseDestroySpMat(matA);

    // A SELL alg id on a CSR matrix.
    const TriMatrix T = random_triangular(32, 0.1, true, false, 81);
    const std::vector<double> tvals(T.values.begin(), T.values.end());
    DeviceBuffer t_val = DeviceBuffer::from(tvals);
    DeviceBuffer t_col = DeviceBuffer::from(T.indices);
    DeviceBuffer t_ptr = DeviceBuffer::from(T.indptr);
    flagsparseSpMatDescr_t csr = nullptr;
    flagsparseDnVecDescr_t cx = nullptr, cy = nullptr;
    flagsparseSpSVDescr_t cdescr = nullptr;
    flagsparseCreateCsr(&csr, T.n, T.n, T.nnz(), t_ptr.get(), t_col.get(), t_val.get(),
                        FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                        FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_64F);
    const flagsparseFillMode_t lower = FLAGSPARSE_FILL_MODE_LOWER;
    flagsparseSpMatSetAttribute(csr, FLAGSPARSE_SPMAT_FILL_MODE, &lower, sizeof(lower));
    flagsparseCreateDnVec(&cx, T.n, d_x.get(), FLAGSPARSE_R_64F);
    flagsparseCreateDnVec(&cy, T.n, d_y.get(), FLAGSPARSE_R_64F);
    flagsparseSpSV_createDescr(&cdescr);
    EXPECT_EQ(flagsparseSpSV_bufferSize(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                        &one, csr, cx, cy, FLAGSPARSE_R_64F,
                                        FLAGSPARSE_SPSV_SELL_ALG1, cdescr, &buf),
              FLAGSPARSE_STATUS_NOT_SUPPORTED);
    flagsparseSpSV_destroyDescr(cdescr);
    flagsparseDestroyDnVec(cy);
    flagsparseDestroyDnVec(cx);
    flagsparseDestroySpMat(csr);
}

TEST_F(SpSVAccuracy, RejectsMisuse) {
    const TriMatrix T = random_triangular(64, 0.05, true, false, 61);
    const std::vector<double> values(T.values.begin(), T.values.end());
    DeviceBuffer d_val = DeviceBuffer::from(values);
    DeviceBuffer d_col = DeviceBuffer::from(T.indices);
    DeviceBuffer d_ptr = DeviceBuffer::from(T.indptr);
    DeviceBuffer d_x   = DeviceBuffer::from(std::vector<double>(T.n, 1.0));
    DeviceBuffer d_y   = DeviceBuffer::from(std::vector<double>(T.n, 0.0));

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    flagsparseSpSVDescr_t descr = nullptr;
    flagsparseCreateCsr(&matA, T.n, T.n, static_cast<int64_t>(T.indices.size()),
                        d_ptr.get(), d_col.get(), d_val.get(), FLAGSPARSE_INDEX_32I,
                        FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO,
                        FLAGSPARSE_R_64F);
    flagsparseCreateDnVec(&vecX, T.n, d_x.get(), FLAGSPARSE_R_64F);
    flagsparseCreateDnVec(&vecY, T.n, d_y.get(), FLAGSPARSE_R_64F);
    ASSERT_EQ(flagsparseSpSV_createDescr(&descr), FLAGSPARSE_STATUS_SUCCESS);
    const double one = 1.0;
    size_t buf = 0;

    // Fill mode is an attribute with no safe default: without it, which triangle
    // to solve is a guess.
    EXPECT_EQ(flagsparseSpSV_bufferSize(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                        &one, matA, vecX, vecY, FLAGSPARSE_R_64F,
                                        FLAGSPARSE_SPSV_ALG_DEFAULT, descr, &buf),
              FLAGSPARSE_STATUS_INVALID_VALUE);

    const flagsparseFillMode_t fill = FLAGSPARSE_FILL_MODE_LOWER;
    flagsparseSpMatSetAttribute(matA, FLAGSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill));
    ASSERT_EQ(flagsparseSpSV_bufferSize(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                        &one, matA, vecX, vecY, FLAGSPARSE_R_64F,
                                        FLAGSPARSE_SPSV_ALG_DEFAULT, descr, &buf),
              FLAGSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(buf, static_cast<size_t>(T.n + 1) * sizeof(int32_t));

    // Solving before analysis has no buffer to work with; say so rather than
    // dereference a null.
    EXPECT_EQ(flagsparseSpSV_solve(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                   matA, vecX, vecY, FLAGSPARSE_R_64F,
                                   FLAGSPARSE_SPSV_ALG_DEFAULT, descr),
              FLAGSPARSE_STATUS_INVALID_VALUE);

    // op(A) = A^T is a different traversal, not wired up yet.
    DeviceBuffer scratch(buf);
    EXPECT_EQ(flagsparseSpSV_analysis(handle.h, FLAGSPARSE_OPERATION_TRANSPOSE, &one,
                                      matA, vecX, vecY, FLAGSPARSE_R_64F,
                                      FLAGSPARSE_SPSV_ALG_DEFAULT, descr, scratch.get()),
              FLAGSPARSE_STATUS_NOT_SUPPORTED);

    flagsparseSpSV_destroyDescr(descr);
    flagsparseDestroyDnVec(vecY);
    flagsparseDestroyDnVec(vecX);
    flagsparseDestroySpMat(matA);
}

// Unsorted columns would make the row scan stop early and drop terms, so the
// analysis pass rejects them instead of returning a plausible wrong answer.
TEST_F(SpSVAccuracy, AnalysisRejectsUnsortedColumns) {
    TriMatrix T = random_triangular(64, 0.3, true, false, 67);
    // Find a row with at least two entries and swap them out of order.
    int64_t victim = -1;
    for (int64_t r = 0; r < T.n; ++r) {
        if (T.indptr[static_cast<std::size_t>(r) + 1] -
                T.indptr[static_cast<std::size_t>(r)] >= 2) { victim = r; break; }
    }
    ASSERT_GE(victim, 0);
    std::swap(T.indices[static_cast<std::size_t>(T.indptr[static_cast<std::size_t>(victim)])],
              T.indices[static_cast<std::size_t>(
                  T.indptr[static_cast<std::size_t>(victim)] + 1)]);

    const std::vector<double> values(T.values.begin(), T.values.end());
    DeviceBuffer d_val = DeviceBuffer::from(values);
    DeviceBuffer d_col = DeviceBuffer::from(T.indices);
    DeviceBuffer d_ptr = DeviceBuffer::from(T.indptr);
    DeviceBuffer d_x   = DeviceBuffer::from(std::vector<double>(T.n, 1.0));
    DeviceBuffer d_y   = DeviceBuffer::from(std::vector<double>(T.n, 0.0));

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    flagsparseSpSVDescr_t descr = nullptr;
    flagsparseCreateCsr(&matA, T.n, T.n, static_cast<int64_t>(T.indices.size()),
                        d_ptr.get(), d_col.get(), d_val.get(), FLAGSPARSE_INDEX_32I,
                        FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO,
                        FLAGSPARSE_R_64F);
    const flagsparseFillMode_t fill = FLAGSPARSE_FILL_MODE_LOWER;
    flagsparseSpMatSetAttribute(matA, FLAGSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill));
    flagsparseCreateDnVec(&vecX, T.n, d_x.get(), FLAGSPARSE_R_64F);
    flagsparseCreateDnVec(&vecY, T.n, d_y.get(), FLAGSPARSE_R_64F);
    flagsparseSpSV_createDescr(&descr);
    const double one = 1.0;
    DeviceBuffer scratch(static_cast<size_t>(T.n + 1) * sizeof(int32_t));
    EXPECT_EQ(flagsparseSpSV_analysis(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one,
                                      matA, vecX, vecY, FLAGSPARSE_R_64F,
                                      FLAGSPARSE_SPSV_ALG_DEFAULT, descr, scratch.get()),
              FLAGSPARSE_STATUS_INVALID_VALUE);
    flagsparseSpSV_destroyDescr(descr);
    flagsparseDestroyDnVec(vecY);
    flagsparseDestroyDnVec(vecX);
    flagsparseDestroySpMat(matA);
}

}  // namespace
