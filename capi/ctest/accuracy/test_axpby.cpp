// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <complex>
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

TEST(AxpbyAccuracy, Complex64WithComplexScalars) {
    // y = alpha x + beta y with alpha = 1+i, beta = 0.5-i, on 3 of 4 entries.
    using C = std::complex<float>;
    flagsparseHandle_t handle = nullptr;
    if (flagsparseCreate(&handle) != FLAGSPARSE_STATUS_SUCCESS) GTEST_SKIP() << "no accelerator";
    const std::vector<int32_t> idx{0, 2, 3};
    const std::vector<C> xv{C(2, 1), C(-1, 0), C(0, 4)};
    const std::vector<C> yv{C(1, 1), C(2, 0), C(3, -2), C(-2, 0.5f)};
    const C alpha(1, 1), beta(0.5f, -1);
    std::vector<C> expect(yv.size());
    for (std::size_t i = 0; i < yv.size(); ++i) expect[i] = beta * yv[i];
    for (std::size_t k = 0; k < idx.size(); ++k) expect[idx[k]] += alpha * xv[k];
    DeviceBuffer d_idx = DeviceBuffer::from(idx), d_x = DeviceBuffer::from(xv),
                 d_y = DeviceBuffer::from(yv);
    flagsparseSpVecDescr_t x = nullptr;
    flagsparseDnVecDescr_t y = nullptr;
    ASSERT_EQ(flagsparseCreateSpVec(&x, 4, 3, d_idx.get(), d_x.get(), FLAGSPARSE_INDEX_32I,
                                    FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_C_32F),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&y, 4, d_y.get(), FLAGSPARSE_C_32F), FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseAxpby(handle, &alpha, x, &beta, y), FLAGSPARSE_STATUS_SUCCESS);
    const auto out = d_y.download<C>(4);
    for (std::size_t i = 0; i < out.size(); ++i) {
        EXPECT_NEAR(out[i].real(), expect[i].real(), 1e-6) << i;
        EXPECT_NEAR(out[i].imag(), expect[i].imag(), 1e-6) << i;
    }
    flagsparseDestroyDnVec(y);
    flagsparseDestroySpVec(x);
    flagsparseDestroy(handle);
}
