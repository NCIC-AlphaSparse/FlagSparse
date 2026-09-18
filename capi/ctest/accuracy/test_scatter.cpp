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


// Accuracy tests for flagsparseScatter:  Y[X_indices[i]] = X_values[i].
// Exact, like gather -- no arithmetic, so no tolerance.

#include <gtest/gtest.h>

#include <complex>
#include <numeric>
#include <random>

#include "common.hpp"

using namespace fstest;

namespace {

struct Handle {
    flagsparseHandle_t h = nullptr;
    Handle() { flagsparseCreate(&h); }
    ~Handle() { if (h) flagsparseDestroy(h); }
};

template <typename I>
std::vector<I> distinct_indices(int64_t dense_size, int64_t nnz, uint32_t seed) {
    std::vector<I> all(static_cast<std::size_t>(dense_size));
    std::iota(all.begin(), all.end(), static_cast<I>(0));
    std::shuffle(all.begin(), all.end(), std::mt19937(seed));
    all.resize(static_cast<std::size_t>(nnz));
    return all;
}

template <typename T> T sample(std::mt19937& rng);
template <> float sample<float>(std::mt19937& rng) {
    return std::normal_distribution<float>(0.f, 1.f)(rng);
}
template <> double sample<double>(std::mt19937& rng) {
    return std::normal_distribution<double>(0.0, 1.0)(rng);
}
template <> std::complex<float> sample<std::complex<float>>(std::mt19937& rng) {
    std::normal_distribution<float> d(0.f, 1.f);
    return {d(rng), d(rng)};
}
template <> std::complex<double> sample<std::complex<double>>(std::mt19937& rng) {
    std::normal_distribution<double> d(0.0, 1.0);
    return {d(rng), d(rng)};
}

template <typename T>
std::vector<T> random_vec(int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<T> v(static_cast<std::size_t>(n));
    for (auto& e : v) e = sample<T>(rng);
    return v;
}

template <typename T, typename I>
void check_scatter(flagsparseHandle_t handle, flagsparseDataType_t vtype,
                   flagsparseIndexType_t itype, int64_t dense_size, int64_t nnz,
                   const char* label) {
    const std::vector<T> dense_init = random_vec<T>(dense_size, 91);
    const std::vector<T> sparse = random_vec<T>(nnz, 17);
    const std::vector<I> idx = distinct_indices<I>(dense_size, nnz, 43);

    DeviceBuffer d_dense  = DeviceBuffer::from(dense_init);
    DeviceBuffer d_idx    = DeviceBuffer::from(idx);
    DeviceBuffer d_sparse = DeviceBuffer::from(sparse);

    flagsparseSpVecDescr_t vecX = nullptr;
    flagsparseDnVecDescr_t vecY = nullptr;
    ASSERT_EQ(flagsparseCreateSpVec(&vecX, dense_size, nnz, d_idx.get(), d_sparse.get(),
                                    itype, FLAGSPARSE_INDEX_BASE_ZERO, vtype),
              FLAGSPARSE_STATUS_SUCCESS) << label;
    ASSERT_EQ(flagsparseCreateDnVec(&vecY, dense_size, d_dense.get(), vtype),
              FLAGSPARSE_STATUS_SUCCESS) << label;

    const flagsparseStatus_t st = flagsparseScatter(handle, vecX, vecY);
    dev_sync();
    const char* detail = "";
    flagsparseGetLastErrorString(handle, &detail);
    ASSERT_EQ(st, FLAGSPARSE_STATUS_SUCCESS)
        << label << ": " << status_name(st) << " -- " << detail;

    // Expectation: scattered positions take the sparse value, every OTHER
    // position keeps what it had. Checking only the written ones would miss a
    // kernel that clobbers the rest of the dense vector.
    std::vector<T> expect = dense_init;
    for (int64_t i = 0; i < nnz; ++i) {
        expect[static_cast<std::size_t>(idx[static_cast<std::size_t>(i)])] =
            sparse[static_cast<std::size_t>(i)];
    }
    const std::vector<T> got = d_dense.download<T>(static_cast<std::size_t>(dense_size));
    for (int64_t i = 0; i < dense_size; ++i) {
        ASSERT_EQ(got[static_cast<std::size_t>(i)], expect[static_cast<std::size_t>(i)])
            << label << " at i=" << i;
    }
    flagsparseDestroyDnVec(vecY);
    flagsparseDestroySpVec(vecX);
}

class ScatterAccuracy : public ::testing::Test {
  protected:
    Handle handle;
    void SetUp() override {
        if (handle.h == nullptr) GTEST_SKIP() << "no accelerator available";
        static bool announced = false;
        if (!announced) { print_backend_banner(); announced = true; }
    }
};

TEST_F(ScatterAccuracy, Float32) {
    check_scatter<float, int32_t>(handle.h, FLAGSPARSE_R_32F, FLAGSPARSE_INDEX_32I,
                                  4096, 1024, "fp32/i32");
}
TEST_F(ScatterAccuracy, Float64) {
    check_scatter<double, int32_t>(handle.h, FLAGSPARSE_R_64F, FLAGSPARSE_INDEX_32I,
                                   4096, 1024, "fp64/i32");
}
TEST_F(ScatterAccuracy, Complex64) {
    check_scatter<std::complex<float>, int32_t>(handle.h, FLAGSPARSE_C_32F,
                                                FLAGSPARSE_INDEX_32I, 2048, 512, "c64/i32");
}
TEST_F(ScatterAccuracy, Complex128) {
    check_scatter<std::complex<double>, int32_t>(handle.h, FLAGSPARSE_C_64F,
                                                 FLAGSPARSE_INDEX_32I, 2048, 512, "c128/i32");
}
TEST_F(ScatterAccuracy, Int64Indices) {
    check_scatter<float, int64_t>(handle.h, FLAGSPARSE_R_32F, FLAGSPARSE_INDEX_64I,
                                  4096, 1024, "fp32/i64");
    check_scatter<std::complex<double>, int64_t>(handle.h, FLAGSPARSE_C_64F,
                                                 FLAGSPARSE_INDEX_64I, 1024, 256, "c128/i64");
}
TEST_F(ScatterAccuracy, LargeNnz) {
    check_scatter<float, int32_t>(handle.h, FLAGSPARSE_R_32F, FLAGSPARSE_INDEX_32I,
                                  1 << 20, 1 << 19, "fp32/large");
}

TEST_F(ScatterAccuracy, ZeroNnzLeavesDenseUntouched) {
    const std::vector<float> dense = random_vec<float>(16, 2);
    DeviceBuffer d_dense = DeviceBuffer::from(dense);
    flagsparseSpVecDescr_t vecX = nullptr;
    flagsparseDnVecDescr_t vecY = nullptr;
    ASSERT_EQ(flagsparseCreateSpVec(&vecX, 16, 0, nullptr, nullptr, FLAGSPARSE_INDEX_32I,
                                    FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F),
              FLAGSPARSE_STATUS_SUCCESS);
    flagsparseCreateDnVec(&vecY, 16, d_dense.get(), FLAGSPARSE_R_32F);
    EXPECT_EQ(flagsparseScatter(handle.h, vecX, vecY), FLAGSPARSE_STATUS_SUCCESS);
    dev_sync();
    EXPECT_EQ(d_dense.download<float>(16), dense);
    flagsparseDestroyDnVec(vecY);
    flagsparseDestroySpVec(vecX);
}

// Gather then scatter through the same index set must reproduce the original
// values at those positions -- the two operators are each other's inverse on a
// distinct index set, which no single-operator test can check.
TEST_F(ScatterAccuracy, RoundTripThroughGather) {
    const int64_t n = 2048, nnz = 512;
    const std::vector<double> dense = random_vec<double>(n, 77);
    const std::vector<int32_t> idx = distinct_indices<int32_t>(n, nnz, 13);

    DeviceBuffer d_dense  = DeviceBuffer::from(dense);
    DeviceBuffer d_idx    = DeviceBuffer::from(idx);
    DeviceBuffer d_sparse = DeviceBuffer::from(std::vector<double>(nnz, 0.0));
    DeviceBuffer d_out    = DeviceBuffer::from(std::vector<double>(n, 0.0));

    flagsparseDnVecDescr_t vecY = nullptr, vecOut = nullptr;
    flagsparseSpVecDescr_t vecX = nullptr;
    flagsparseCreateDnVec(&vecY, n, d_dense.get(), FLAGSPARSE_R_64F);
    flagsparseCreateDnVec(&vecOut, n, d_out.get(), FLAGSPARSE_R_64F);
    flagsparseCreateSpVec(&vecX, n, nnz, d_idx.get(), d_sparse.get(), FLAGSPARSE_INDEX_32I,
                          FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_64F);

    ASSERT_EQ(flagsparseGather(handle.h, vecY, vecX), FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseScatter(handle.h, vecX, vecOut), FLAGSPARSE_STATUS_SUCCESS);
    dev_sync();

    const std::vector<double> out = d_out.download<double>(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < nnz; ++i) {
        const auto pos = static_cast<std::size_t>(idx[static_cast<std::size_t>(i)]);
        EXPECT_EQ(out[pos], dense[pos]) << "round trip at " << pos;
    }
    flagsparseDestroySpVec(vecX);
    flagsparseDestroyDnVec(vecOut);
    flagsparseDestroyDnVec(vecY);
}

TEST_F(ScatterAccuracy, SizeMismatchIsRejected) {
    DeviceBuffer d_dense = DeviceBuffer::from(std::vector<float>(32, 1.f));
    DeviceBuffer d_idx   = DeviceBuffer::from(std::vector<int32_t>(4, 0));
    DeviceBuffer d_val   = DeviceBuffer::from(std::vector<float>(4, 0.f));
    flagsparseSpVecDescr_t vecX = nullptr;
    flagsparseDnVecDescr_t vecY = nullptr;
    // Sparse vector claims a logical size the dense vector does not have.
    flagsparseCreateSpVec(&vecX, 64, 4, d_idx.get(), d_val.get(), FLAGSPARSE_INDEX_32I,
                          FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F);
    flagsparseCreateDnVec(&vecY, 32, d_dense.get(), FLAGSPARSE_R_32F);
    EXPECT_EQ(flagsparseScatter(handle.h, vecX, vecY), FLAGSPARSE_STATUS_INVALID_VALUE);
    flagsparseDestroyDnVec(vecY);
    flagsparseDestroySpVec(vecX);
}

}  // namespace
