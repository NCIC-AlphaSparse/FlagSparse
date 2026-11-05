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

// flagsparseGemvi against a host complex128 oracle: every value type, both
// directions, non-trivial alpha/beta (complex ones for complex types), a padded
// lda, and a shape large enough to take the split-and-reduce route.

#include <gtest/gtest.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include "common.hpp"

using namespace fstest;

namespace {

using cd = std::complex<double>;

struct Handle {
    flagsparseHandle_t h = nullptr;
    Handle() { flagsparseCreate(&h); }
    ~Handle() { if (h) flagsparseDestroy(h); }
};

std::string last_error(flagsparseHandle_t h) {
    const char* s = nullptr;
    return flagsparseGetLastErrorString(h, &s) == FLAGSPARSE_STATUS_SUCCESS && s ? s : "";
}

class GemviAccuracy : public ::testing::Test {
  protected:
    Handle handle;
    void SetUp() override {
        if (handle.h == nullptr) GTEST_SKIP() << "no accelerator available";
        static bool announced = false;
        if (!announced) { print_backend_banner(); announced = true; }
    }
};

template <typename T> T cast_to(cd v);
template <> float cast_to<float>(cd v) { return static_cast<float>(v.real()); }
template <> double cast_to<double>(cd v) { return v.real(); }
template <> std::complex<float> cast_to<std::complex<float>>(cd v) {
    return {static_cast<float>(v.real()), static_cast<float>(v.imag())};
}
template <> std::complex<double> cast_to<std::complex<double>>(cd v) { return v; }

template <typename T> cd widen(T v) { return cd(std::real(v), std::imag(v)); }

// Deterministic values with a non-zero imaginary part for complex types.
template <typename T> T value(int64_t i, int64_t salt) {
    const double re = 0.25 * static_cast<double>((i * 7 + salt * 3) % 9) - 1.0;
    const double im = 0.125 * static_cast<double>((i * 5 + salt) % 7) - 0.375;
    return cast_to<T>(cd(re, im));
}

template <typename T>
void check(flagsparseHandle_t h, flagsparseDataType_t dt, flagsparseOperation_t op, int m,
           int n, int nnz, int pad, cd alpha, cd beta, double tol) {
    const bool trans = op == FLAGSPARSE_OPERATION_TRANSPOSE;
    const int lda = m + pad;
    const int n_in = trans ? m : n, n_out = trans ? n : m;
    std::vector<T> A(static_cast<std::size_t>(lda) * n);
    for (std::size_t i = 0; i < A.size(); ++i) A[i] = value<T>(static_cast<int64_t>(i), 1);
    std::vector<int32_t> idx(static_cast<std::size_t>(nnz));
    // An odd stride walks every index of a power-of-two range exactly once.
    for (int k = 0; k < nnz; ++k) idx[k] = static_cast<int32_t>((int64_t{k} * 7919) % n_in);
    std::vector<T> x(static_cast<std::size_t>(nnz)), y(static_cast<std::size_t>(n_out));
    for (int k = 0; k < nnz; ++k) x[k] = value<T>(k, 2);
    for (int r = 0; r < n_out; ++r) y[r] = value<T>(r, 3);

    std::vector<cd> ref(static_cast<std::size_t>(n_out));
    for (int r = 0; r < n_out; ++r) {
        cd acc = 0;
        for (int k = 0; k < nnz; ++k) {
            const int64_t i = trans ? idx[k] : r, j = trans ? r : idx[k];
            acc += widen(A[static_cast<std::size_t>(i + j * lda)]) * widen(x[k]);
        }
        ref[r] = alpha * acc + beta * widen(y[r]);
    }

    DeviceBuffer dA = DeviceBuffer::from(A), dx = DeviceBuffer::from(x),
                 di = DeviceBuffer::from(idx), dy = DeviceBuffer::from(y);
    const T a = cast_to<T>(alpha), b = cast_to<T>(beta);
    size_t bytes = 0;
    ASSERT_EQ(flagsparseGemvi_bufferSize(h, op, m, n, nnz, dt, &bytes), FLAGSPARSE_STATUS_SUCCESS);
    DeviceBuffer scratch(bytes ? bytes : 1);
    ASSERT_EQ(flagsparseGemvi(h, op, m, n, &a, dA.get(), lda, nnz, dx.get(),
                              static_cast<const int*>(di.get()), &b, dy.get(),
                              FLAGSPARSE_INDEX_BASE_ZERO, dt, bytes ? scratch.get() : nullptr),
              FLAGSPARSE_STATUS_SUCCESS)
        << last_error(h);
    const std::vector<T> got = dy.download<T>(y.size());
    double scale = 1.0;
    for (const cd& v : ref) scale = std::max(scale, std::abs(v));
    double err = 0.0;
    for (int r = 0; r < n_out; ++r) err = std::max(err, std::abs(widen(got[r]) - ref[r]));
    EXPECT_LE(err, tol * scale) << "m=" << m << " n=" << n << " nnz=" << nnz
                                << " trans=" << trans << " bytes=" << bytes;
}

TEST_F(GemviAccuracy, EveryTypeBothDirections) {
    for (auto op : {FLAGSPARSE_OPERATION_NON_TRANSPOSE, FLAGSPARSE_OPERATION_TRANSPOSE}) {
        check<float>(handle.h, FLAGSPARSE_R_32F, op, 131, 77, 40, 5, 1.5, -0.5, 1e-5);
        check<double>(handle.h, FLAGSPARSE_R_64F, op, 131, 77, 40, 5, 1.5, -0.5, 1e-12);
        check<std::complex<float>>(handle.h, FLAGSPARSE_C_32F, op, 131, 77, 40, 5,
                                   cd(1.5, -0.25), cd(0.5, 1.0), 1e-5);
        check<std::complex<double>>(handle.h, FLAGSPARSE_C_64F, op, 131, 77, 40, 5,
                                    cd(1.5, -0.25), cd(0.5, 1.0), 1e-12);
    }
}

TEST_F(GemviAccuracy, BetaZeroNeverReadsY) {
    // beta == 0 must overwrite y, NaN included, rather than scale it.
    const int m = 64, n = 32, nnz = 8;
    std::vector<float> A(static_cast<std::size_t>(m) * n, 1.0f);
    std::vector<int32_t> idx{0, 3, 5, 7, 11, 13, 17, 19};
    std::vector<float> x(nnz, 2.0f), y(m, std::nanf(""));
    DeviceBuffer dA = DeviceBuffer::from(A), dx = DeviceBuffer::from(x),
                 di = DeviceBuffer::from(idx), dy = DeviceBuffer::from(y);
    const float a = 1.0f, b = 0.0f;
    ASSERT_EQ(flagsparseGemvi(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE, m, n, &a, dA.get(), m,
                              nnz, dx.get(), static_cast<const int*>(di.get()), &b, dy.get(),
                              FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F, nullptr),
              FLAGSPARSE_STATUS_SUCCESS);
    for (float v : dy.download<float>(y.size())) EXPECT_FLOAT_EQ(v, 16.0f);
}

TEST_F(GemviAccuracy, SplitRouteMatchesTheOracle) {
    // Few outputs and many entries: more than 16 MB of A to read over 4 output
    // blocks, which is the split-and-reduce route; bufferSize must say so.
    const int m = 256, n = 65536, nnz = 32768;  // 32 MB of A at float32
    size_t bytes = 0;
    ASSERT_EQ(flagsparseGemvi_bufferSize(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz,
                                         FLAGSPARSE_R_32F, &bytes),
              FLAGSPARSE_STATUS_SUCCESS);
    EXPECT_GT(bytes, 0u);
    check<float>(handle.h, FLAGSPARSE_R_32F, FLAGSPARSE_OPERATION_NON_TRANSPOSE, m, n, nnz, 0,
                 2.0, 0.5, 1e-4);
    check<std::complex<float>>(handle.h, FLAGSPARSE_C_32F, FLAGSPARSE_OPERATION_TRANSPOSE, n, m,
                               nnz, 0, cd(1.0, 0.5), cd(0.0, 0.0), 1e-4);
}

TEST_F(GemviAccuracy, ComplexNonTransposeSplitWithPaddedLdaAndBeta) {
    check<std::complex<float>>(handle.h, FLAGSPARSE_C_32F, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                               4096, 2048, 1024, 5, cd(1.25, -0.5), cd(-0.25, 0.75), 1e-4);
}

TEST_F(GemviAccuracy, RejectsWhatCusparseGemviDoesNot) {
    size_t bytes = 0;
    EXPECT_EQ(flagsparseGemvi_bufferSize(handle.h, FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE, 4, 4,
                                         2, FLAGSPARSE_C_32F, &bytes),
              FLAGSPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(flagsparseGemvi_bufferSize(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE, 4, 4, 2,
                                         FLAGSPARSE_R_16F, &bytes),
              FLAGSPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(flagsparseGemvi_bufferSize(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE, 4, 4, 5,
                                         FLAGSPARSE_R_32F, &bytes),
              FLAGSPARSE_STATUS_INVALID_VALUE);  // more entries than x has slots
}

}  // namespace
