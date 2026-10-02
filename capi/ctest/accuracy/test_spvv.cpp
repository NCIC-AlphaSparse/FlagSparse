// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstdint>
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

template <typename In, typename Out>
Out run_spvv(flagsparseHandle_t handle, flagsparseDataType_t input_type,
             flagsparseDataType_t output_type, const std::vector<In>& sparse,
             const std::vector<int32_t>& indices, const std::vector<In>& dense,
             flagsparseOperation_t op = FLAGSPARSE_OPERATION_NON_TRANSPOSE) {
    DeviceBuffer d_sparse = DeviceBuffer::from(sparse);
    DeviceBuffer d_indices = DeviceBuffer::from(indices);
    DeviceBuffer d_dense = DeviceBuffer::from(dense);

    flagsparseSpVecDescr_t vecX = nullptr;
    flagsparseDnVecDescr_t vecY = nullptr;
    EXPECT_EQ(flagsparseCreateSpVec(&vecX, static_cast<int64_t>(dense.size()),
                                    static_cast<int64_t>(sparse.size()),
                                    d_indices.get(), d_sparse.get(),
                                    FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO,
                                    input_type),
              FLAGSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(flagsparseCreateDnVec(&vecY, static_cast<int64_t>(dense.size()),
                                    d_dense.get(), input_type),
              FLAGSPARSE_STATUS_SUCCESS);

    Out result{};
    size_t buffer_size = 0;
    EXPECT_EQ(flagsparseSpVV_bufferSize(handle, op,
                                        vecX, vecY, &result, output_type, &buffer_size),
              FLAGSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(buffer_size, sizeof(Out));
    DeviceBuffer scratch(buffer_size);
    EXPECT_EQ(flagsparseSpVV(handle, op, vecX, vecY,
                             &result, output_type, scratch.get()),
              FLAGSPARSE_STATUS_SUCCESS);

    flagsparseDestroyDnVec(vecY);
    flagsparseDestroySpVec(vecX);
    return result;
}

class SpVVAccuracy : public ::testing::Test {
  protected:
    Handle handle;
    void SetUp() override {
        if (handle.h == nullptr) GTEST_SKIP() << "no accelerator available";
        static bool announced = false;
        if (!announced) { print_backend_banner(); announced = true; }
    }
};

TEST_F(SpVVAccuracy, Int8ToInt32MixedPrecision) {
    const std::vector<int8_t> sparse{2, -1, 4};
    const std::vector<int32_t> indices{0, 2, 3};
    const std::vector<int8_t> dense{3, 1, 5, -2};
    EXPECT_EQ((run_spvv<int8_t, int32_t>(handle.h, FLAGSPARSE_R_8I,
                                         FLAGSPARSE_R_32I, sparse, indices, dense)),
              -7);
}

TEST_F(SpVVAccuracy, Float16ToFloat32MixedPrecision) {
    const std::vector<Half> sparse{Half(0.5), Half(-1.5), Half(4.0)};
    const std::vector<int32_t> indices{0, 2, 3};
    const std::vector<Half> dense{Half(2.0), Half(7.0), Half(-2.0), Half(0.25)};
    EXPECT_FLOAT_EQ((run_spvv<Half, float>(handle.h, FLAGSPARSE_R_16F,
                                           FLAGSPARSE_R_32F, sparse, indices, dense)),
                    5.0f);
}

TEST_F(SpVVAccuracy, Complex64Conjugate) {
    using C = std::complex<float>;
    const std::vector<C> sparse{C(2, 1), C(-1, 3)};
    const std::vector<int32_t> indices{0, 2};
    const std::vector<C> dense{C(3, -2), C(0, 0), C(4, 1)};
    EXPECT_EQ((run_spvv<C, C>(handle.h, FLAGSPARSE_C_32F, FLAGSPARSE_C_32F,
                               sparse, indices, dense,
                               FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE)),
              C(3, -20));
}

}  // namespace
