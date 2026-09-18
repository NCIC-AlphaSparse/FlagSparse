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


// Accuracy tests for flagsparseGather:  X_values[i] = Y[X_indices[i]].
//
// Gather and scatter are exact -- no arithmetic happens, so the reference is
// bit-for-bit and a tolerance would only hide a real defect.

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

// Distinct indices in random order. Gather tolerates duplicates, but using
// distinct ones keeps this test's expectation unambiguous; duplicates get their
// own case below.
template <typename I>
std::vector<I> random_indices(int64_t dense_size, int64_t nnz, uint32_t seed) {
    std::vector<I> all(static_cast<std::size_t>(dense_size));
    std::iota(all.begin(), all.end(), static_cast<I>(0));
    std::shuffle(all.begin(), all.end(), std::mt19937(seed));
    all.resize(static_cast<std::size_t>(nnz));
    return all;
}

template <typename T>
std::vector<T> random_values(int64_t n, uint32_t seed);

template <>
std::vector<float> random_values<float>(int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.f, 1.f);
    std::vector<float> v(static_cast<std::size_t>(n));
    for (auto& e : v) e = d(rng);
    return v;
}
template <>
std::vector<double> random_values<double>(int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> d(0.0, 1.0);
    std::vector<double> v(static_cast<std::size_t>(n));
    for (auto& e : v) e = d(rng);
    return v;
}
template <>
std::vector<std::complex<float>> random_values<std::complex<float>>(int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.f, 1.f);
    std::vector<std::complex<float>> v(static_cast<std::size_t>(n));
    for (auto& e : v) e = {d(rng), d(rng)};
    return v;
}
template <>
std::vector<std::complex<double>> random_values<std::complex<double>>(int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<double> d(0.0, 1.0);
    std::vector<std::complex<double>> v(static_cast<std::size_t>(n));
    for (auto& e : v) e = {d(rng), d(rng)};
    return v;
}

// One gather, compared exactly against the host.
template <typename T, typename I>
void check_gather(flagsparseHandle_t handle, flagsparseDataType_t vtype,
                  flagsparseIndexType_t itype, int64_t dense_size, int64_t nnz,
                  const char* label) {
    const std::vector<T> dense = random_values<T>(dense_size, 31);
    const std::vector<I> idx = random_indices<I>(dense_size, nnz, 57);
    // Poison the destination so a kernel that writes nothing cannot pass.
    const std::vector<T> sparse_init(static_cast<std::size_t>(nnz), T(-12345));

    DeviceBuffer d_dense  = DeviceBuffer::from(dense);
    DeviceBuffer d_idx    = DeviceBuffer::from(idx);
    DeviceBuffer d_sparse = DeviceBuffer::from(sparse_init);

    flagsparseDnVecDescr_t vecY = nullptr;
    flagsparseSpVecDescr_t vecX = nullptr;
    ASSERT_EQ(flagsparseCreateDnVec(&vecY, dense_size, d_dense.get(), vtype),
              FLAGSPARSE_STATUS_SUCCESS) << label;
    ASSERT_EQ(flagsparseCreateSpVec(&vecX, dense_size, nnz, d_idx.get(), d_sparse.get(),
                                    itype, FLAGSPARSE_INDEX_BASE_ZERO, vtype),
              FLAGSPARSE_STATUS_SUCCESS) << label;

    const flagsparseStatus_t st = flagsparseGather(handle, vecY, vecX);
    dev_sync();
    const char* detail = "";
    flagsparseGetLastErrorString(handle, &detail);
    ASSERT_EQ(st, FLAGSPARSE_STATUS_SUCCESS)
        << label << ": " << status_name(st) << " -- " << detail;

    const std::vector<T> got = d_sparse.download<T>(static_cast<std::size_t>(nnz));
    for (int64_t i = 0; i < nnz; ++i) {
        EXPECT_EQ(got[static_cast<std::size_t>(i)],
                  dense[static_cast<std::size_t>(idx[static_cast<std::size_t>(i)])])
            << label << " at i=" << i;
    }
    flagsparseDestroySpVec(vecX);
    flagsparseDestroyDnVec(vecY);
}

class GatherAccuracy : public ::testing::Test {
  protected:
    Handle handle;
    void SetUp() override {
        if (handle.h == nullptr) GTEST_SKIP() << "no accelerator available";
        static bool announced = false;
        if (!announced) { print_backend_banner(); announced = true; }
    }
};

TEST_F(GatherAccuracy, Float32) {
    check_gather<float, int32_t>(handle.h, FLAGSPARSE_R_32F, FLAGSPARSE_INDEX_32I,
                                 4096, 1024, "fp32/i32");
}
TEST_F(GatherAccuracy, Float64) {
    check_gather<double, int32_t>(handle.h, FLAGSPARSE_R_64F, FLAGSPARSE_INDEX_32I,
                                  4096, 1024, "fp64/i32");
}
TEST_F(GatherAccuracy, Complex64) {
    check_gather<std::complex<float>, int32_t>(handle.h, FLAGSPARSE_C_32F,
                                               FLAGSPARSE_INDEX_32I, 2048, 512, "c64/i32");
}
TEST_F(GatherAccuracy, Complex128) {
    check_gather<std::complex<double>, int32_t>(handle.h, FLAGSPARSE_C_64F,
                                                FLAGSPARSE_INDEX_32I, 2048, 512, "c128/i32");
}

// Index width is a descriptor property, not part of the function name, and both
// widths must work (spec §4.5.1).
TEST_F(GatherAccuracy, Int64Indices) {
    check_gather<float, int64_t>(handle.h, FLAGSPARSE_R_32F, FLAGSPARSE_INDEX_64I,
                                 4096, 1024, "fp32/i64");
    check_gather<std::complex<double>, int64_t>(handle.h, FLAGSPARSE_C_64F,
                                                FLAGSPARSE_INDEX_64I, 1024, 256, "c128/i64");
}

// The grid is capped at what fills the device, so a large nnz exercises the
// grid-stride loop taking several tiles per program -- the path that a
// hardcoded cap once got badly wrong.
TEST_F(GatherAccuracy, LargeNnzExercisesGridStride) {
    check_gather<float, int32_t>(handle.h, FLAGSPARSE_R_32F, FLAGSPARSE_INDEX_32I,
                                 1 << 20, 1 << 19, "fp32/large");
}

TEST_F(GatherAccuracy, DuplicateIndicesAreAllowed) {
    const int64_t dense_size = 64, nnz = 32;
    const std::vector<float> dense = random_values<float>(dense_size, 5);
    const std::vector<int32_t> idx(static_cast<std::size_t>(nnz), 7);  // all the same
    const std::vector<float> init(static_cast<std::size_t>(nnz), -1.f);

    DeviceBuffer d_dense = DeviceBuffer::from(dense);
    DeviceBuffer d_idx = DeviceBuffer::from(idx);
    DeviceBuffer d_sparse = DeviceBuffer::from(init);

    flagsparseDnVecDescr_t vecY = nullptr;
    flagsparseSpVecDescr_t vecX = nullptr;
    flagsparseCreateDnVec(&vecY, dense_size, d_dense.get(), FLAGSPARSE_R_32F);
    flagsparseCreateSpVec(&vecX, dense_size, nnz, d_idx.get(), d_sparse.get(),
                          FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO,
                          FLAGSPARSE_R_32F);
    ASSERT_EQ(flagsparseGather(handle.h, vecY, vecX), FLAGSPARSE_STATUS_SUCCESS);
    dev_sync();
    const std::vector<float> got = d_sparse.download<float>(static_cast<std::size_t>(nnz));
    for (float v : got) EXPECT_EQ(v, dense[7]);
    flagsparseDestroySpVec(vecX);
    flagsparseDestroyDnVec(vecY);
}

TEST_F(GatherAccuracy, ZeroNnzIsANoOp) {
    const std::vector<float> dense = random_values<float>(16, 1);
    DeviceBuffer d_dense = DeviceBuffer::from(dense);
    flagsparseDnVecDescr_t vecY = nullptr;
    flagsparseSpVecDescr_t vecX = nullptr;
    flagsparseCreateDnVec(&vecY, 16, d_dense.get(), FLAGSPARSE_R_32F);
    // nnz == 0 legitimately has null indices/values.
    ASSERT_EQ(flagsparseCreateSpVec(&vecX, 16, 0, nullptr, nullptr, FLAGSPARSE_INDEX_32I,
                                    FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F),
              FLAGSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(flagsparseGather(handle.h, vecY, vecX), FLAGSPARSE_STATUS_SUCCESS);
    flagsparseDestroySpVec(vecX);
    flagsparseDestroyDnVec(vecY);
}

TEST_F(GatherAccuracy, MismatchedDtypeIsRejected) {
    DeviceBuffer d_dense = DeviceBuffer::from(std::vector<float>(64, 1.f));
    DeviceBuffer d_idx   = DeviceBuffer::from(std::vector<int32_t>(8, 0));
    DeviceBuffer d_val   = DeviceBuffer::from(std::vector<double>(8, 0.0));
    flagsparseDnVecDescr_t vecY = nullptr;
    flagsparseSpVecDescr_t vecX = nullptr;
    flagsparseCreateDnVec(&vecY, 64, d_dense.get(), FLAGSPARSE_R_32F);
    flagsparseCreateSpVec(&vecX, 64, 8, d_idx.get(), d_val.get(), FLAGSPARSE_INDEX_32I,
                          FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_64F);
    EXPECT_EQ(flagsparseGather(handle.h, vecY, vecX), FLAGSPARSE_STATUS_NOT_SUPPORTED);
    flagsparseDestroySpVec(vecX);
    flagsparseDestroyDnVec(vecY);
}

// One-based indexing is a cuSPARSE feature this build does not implement; it
// must say so rather than silently reading one element off.
TEST_F(GatherAccuracy, OneBasedIndexingReportsNotSupported) {
    DeviceBuffer d_dense = DeviceBuffer::from(std::vector<float>(64, 1.f));
    DeviceBuffer d_idx   = DeviceBuffer::from(std::vector<int32_t>(8, 1));
    DeviceBuffer d_val   = DeviceBuffer::from(std::vector<float>(8, 0.f));
    flagsparseDnVecDescr_t vecY = nullptr;
    flagsparseSpVecDescr_t vecX = nullptr;
    flagsparseCreateDnVec(&vecY, 64, d_dense.get(), FLAGSPARSE_R_32F);
    flagsparseCreateSpVec(&vecX, 64, 8, d_idx.get(), d_val.get(), FLAGSPARSE_INDEX_32I,
                          FLAGSPARSE_INDEX_BASE_ONE, FLAGSPARSE_R_32F);
    EXPECT_EQ(flagsparseGather(handle.h, vecY, vecX), FLAGSPARSE_STATUS_NOT_SUPPORTED);
    flagsparseDestroySpVec(vecX);
    flagsparseDestroyDnVec(vecY);
}

}  // namespace
