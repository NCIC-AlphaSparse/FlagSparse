// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "common.hpp"

using namespace fstest;

TEST(AxpbyAccuracy, Float16InPlace) {
    flagsparseHandle_t handle = nullptr;
    ASSERT_EQ(flagsparseCreate(&handle), FLAGSPARSE_STATUS_SUCCESS);
    const std::vector<Half> values{Half(2.0), Half(-1.0), Half(4.0)};
    const std::vector<int32_t> indices{0, 2, 3};
    const std::vector<Half> initial{Half(1.0), Half(2.0), Half(3.0), Half(-2.0)};
    DeviceBuffer d_values = DeviceBuffer::from(values);
    DeviceBuffer d_indices = DeviceBuffer::from(indices);
    DeviceBuffer d_y = DeviceBuffer::from(initial);
    flagsparseSpVecDescr_t x = nullptr;
    flagsparseDnVecDescr_t y = nullptr;
    ASSERT_EQ(flagsparseCreateSpVec(&x, 4, 3, d_indices.get(), d_values.get(),
                                    FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO,
                                    FLAGSPARSE_R_16F), FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&y, 4, d_y.get(), FLAGSPARSE_R_16F), FLAGSPARSE_STATUS_SUCCESS);
    const Half alpha(1.5), beta(-0.5);
    ASSERT_EQ(flagsparseAxpby(handle, &alpha, x, &beta, y), FLAGSPARSE_STATUS_SUCCESS);
    dev_sync();
    const auto out = d_y.download<Half>(4);
    EXPECT_DOUBLE_EQ(static_cast<double>(out[0]), 2.5);
    EXPECT_DOUBLE_EQ(static_cast<double>(out[1]), -1.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(out[2]), -3.0);
    EXPECT_DOUBLE_EQ(static_cast<double>(out[3]), 7.0);
    flagsparseDestroyDnVec(y);
    flagsparseDestroySpVec(x);
    flagsparseDestroy(handle);
}
