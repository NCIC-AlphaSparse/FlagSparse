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
#include <cstring>
#include <limits>
#include <memory>

#include "common.hpp"
#include "adaptor/adaptor.hpp"

#if defined(FLAGSPARSE_MUSA_BASELINE_EXTENSIONS)
#include <musa.h>
#endif

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

// spmv_csr_i8i32_int_non / spmv_csr_i8f32_int_non: int8 matrix and x,
// widened int32 or float32 y -- capi/src/ops/spmv.cpp's run_csr_mixed, a
// separate path from the rest of this file (Operands/launch_csr_rowpar
// assume A/X/Y share one dtype, which this deliberately does not). Hand-built
// small integer data, not random_csr's continuous doubles: int8 has no
// fractional part, so a rounding-derived reference would just be testing the
// rounding, not the kernel.
TEST_F(SpMVAccuracy, Int8ToInt32MixedPrecision) {
    // A (3x4), CSR: row0=[2,0,-3,0] row1=[0,0,1,4] row2=[-1,2,0,0]
    const std::vector<int8_t> data = {2, -3, 1, 4, -1, 2};
    const std::vector<int32_t> indices = {0, 2, 2, 3, 0, 1};
    const std::vector<int32_t> indptr = {0, 2, 4, 6};
    const std::vector<int8_t> x = {3, -2, 5, -1};
    // expected: row0=2*3-3*5=-9, row1=1*5+4*-1=1, row2=-1*3+2*-2=-7
    const std::vector<int32_t> expect = {-9, 1, -7};

    DeviceBuffer d_data = DeviceBuffer::from(data);
    DeviceBuffer d_idx = DeviceBuffer::from(indices);
    DeviceBuffer d_ptr = DeviceBuffer::from(indptr);
    DeviceBuffer d_x = DeviceBuffer::from(x);
    DeviceBuffer d_y(3 * sizeof(int32_t));

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    ASSERT_EQ(flagsparseCreateCsr(&matA, 3, 4, 6, d_ptr.get(), d_idx.get(), d_data.get(),
                                  FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_8I),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vecX, 4, d_x.get(), FLAGSPARSE_R_8I),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vecY, 3, d_y.get(), FLAGSPARSE_R_32I),
              FLAGSPARSE_STATUS_SUCCESS);

    const int32_t one = 1, zero = 0;
    size_t bufsz = 12345;  // must come back 0 for this path -- catches a stale value.
    ASSERT_EQ(flagsparseSpMV_bufferSize(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                        &one, matA, vecX, &zero, vecY,
                                        FLAGSPARSE_R_32I, FLAGSPARSE_SPMV_ALG_DEFAULT,
                                        &bufsz),
              FLAGSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bufsz, 0u);
    ASSERT_EQ(flagsparseSpMV(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA,
                             vecX, &zero, vecY, FLAGSPARSE_R_32I,
                             FLAGSPARSE_SPMV_ALG_DEFAULT, nullptr),
              FLAGSPARSE_STATUS_SUCCESS);
    dev_sync();
    const std::vector<int32_t> got = d_y.download<int32_t>(3);
    for (int i = 0; i < 3; ++i) EXPECT_EQ(got[i], expect[i]) << "row " << i;

    flagsparseDestroyDnVec(vecX);
    flagsparseDestroyDnVec(vecY);
    flagsparseDestroySpMat(matA);
}

TEST_F(SpMVAccuracy, Int8ToFloat32MixedPrecision) {
    const std::vector<int8_t> data = {2, -3, 1, 4, -1, 2};
    const std::vector<int32_t> indices = {0, 2, 2, 3, 0, 1};
    const std::vector<int32_t> indptr = {0, 2, 4, 6};
    const std::vector<int8_t> x = {3, -2, 5, -1};
    const std::vector<float> expect = {-9.0f, 1.0f, -7.0f};

    DeviceBuffer d_data = DeviceBuffer::from(data);
    DeviceBuffer d_idx = DeviceBuffer::from(indices);
    DeviceBuffer d_ptr = DeviceBuffer::from(indptr);
    DeviceBuffer d_x = DeviceBuffer::from(x);
    DeviceBuffer d_y(3 * sizeof(float));

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    ASSERT_EQ(flagsparseCreateCsr(&matA, 3, 4, 6, d_ptr.get(), d_idx.get(), d_data.get(),
                                  FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_8I),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vecX, 4, d_x.get(), FLAGSPARSE_R_8I),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vecY, 3, d_y.get(), FLAGSPARSE_R_32F),
              FLAGSPARSE_STATUS_SUCCESS);

    const float one = 1.0f, zero = 0.0f;
    ASSERT_EQ(flagsparseSpMV(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA,
                             vecX, &zero, vecY, FLAGSPARSE_R_32F,
                             FLAGSPARSE_SPMV_ALG_DEFAULT, nullptr),
              FLAGSPARSE_STATUS_SUCCESS);
    dev_sync();
    const std::vector<float> got = d_y.download<float>(3);
    for (int i = 0; i < 3; ++i) EXPECT_FLOAT_EQ(got[i], expect[i]) << "row " << i;

    flagsparseDestroyDnVec(vecX);
    flagsparseDestroyDnVec(vecY);
    flagsparseDestroySpMat(matA);
}

// spmv_csr_f16f32_int_non: fp16 matrix/x, widened float32 y. Same
// run_csr_mixed path as the int8 tests above (ACC=float32 either way); values
// kept exactly fp16-representable (halves) so there is no rounding to reason
// about, isolating this test to "does the mixed dispatch route correctly."
TEST_F(SpMVAccuracy, Float16ToFloat32MixedPrecision) {
    const std::vector<Half> data = {Half(2.5), Half(-1.5), Half(0.5), Half(4.0)};
    const std::vector<int32_t> indices = {0, 1, 1, 2};
    const std::vector<int32_t> indptr = {0, 2, 3, 4};
    const std::vector<Half> x = {Half(2.0), Half(-2.0), Half(1.0)};
    // row0=2.5*2 + -1.5*-2 = 8; row1=0.5*-2=-1; row2=4*1=4
    const std::vector<float> expect = {8.0f, -1.0f, 4.0f};

    DeviceBuffer d_data = DeviceBuffer::from(data);
    DeviceBuffer d_idx = DeviceBuffer::from(indices);
    DeviceBuffer d_ptr = DeviceBuffer::from(indptr);
    DeviceBuffer d_x = DeviceBuffer::from(x);
    DeviceBuffer d_y(3 * sizeof(float));

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    ASSERT_EQ(flagsparseCreateCsr(&matA, 3, 3, 4, d_ptr.get(), d_idx.get(), d_data.get(),
                                  FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_16F),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vecX, 3, d_x.get(), FLAGSPARSE_R_16F),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vecY, 3, d_y.get(), FLAGSPARSE_R_32F),
              FLAGSPARSE_STATUS_SUCCESS);

    const float one = 1.0f, zero = 0.0f;
    ASSERT_EQ(flagsparseSpMV(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA,
                             vecX, &zero, vecY, FLAGSPARSE_R_32F,
                             FLAGSPARSE_SPMV_ALG_DEFAULT, nullptr),
              FLAGSPARSE_STATUS_SUCCESS);
    dev_sync();
    const std::vector<float> got = d_y.download<float>(3);
    for (int i = 0; i < 3; ++i) EXPECT_FLOAT_EQ(got[i], expect[i]) << "row " << i;

    flagsparseDestroyDnVec(vecX);
    flagsparseDestroyDnVec(vecY);
    flagsparseDestroySpMat(matA);
}

TEST_F(SpMVAccuracy, RealFloat32ByComplex64MixedPrecision) {
    // A = [[2, 0, -1], [0, 3, 4]], x = [1+2i, -2+i, 0.5-3i].
    const std::vector<float> data{2.0f, -1.0f, 3.0f, 4.0f};
    const std::vector<int32_t> indices{0, 2, 1, 2};
    const std::vector<int32_t> indptr{0, 2, 4};
    const std::vector<float> x_ri{1.0f, 2.0f, -2.0f, 1.0f, 0.5f, -3.0f};
    const std::vector<float> expect_ri{1.5f, 7.0f, -4.0f, -9.0f};

    DeviceBuffer d_data = DeviceBuffer::from(data);
    DeviceBuffer d_idx = DeviceBuffer::from(indices);
    DeviceBuffer d_ptr = DeviceBuffer::from(indptr);
    DeviceBuffer d_x = DeviceBuffer::from(x_ri);
    DeviceBuffer d_y = DeviceBuffer::from(std::vector<float>(4, 99.0f));

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    ASSERT_EQ(flagsparseCreateCsr(&matA, 2, 3, 4, d_ptr.get(), d_idx.get(), d_data.get(),
                                  FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vecX, 3, d_x.get(), FLAGSPARSE_C_32F),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vecY, 2, d_y.get(), FLAGSPARSE_C_32F),
              FLAGSPARSE_STATUS_SUCCESS);

    const float one[2] = {1.0f, 0.0f};
    const float zero[2] = {0.0f, 0.0f};
    size_t bufsz = 12345;
    ASSERT_EQ(flagsparseSpMV_bufferSize(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                        one, matA, vecX, zero, vecY,
                                        FLAGSPARSE_C_32F, FLAGSPARSE_SPMV_ALG_DEFAULT,
                                        &bufsz),
              FLAGSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bufsz, 0u);
    ASSERT_EQ(flagsparseSpMV(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE, one, matA,
                             vecX, zero, vecY, FLAGSPARSE_C_32F,
                             FLAGSPARSE_SPMV_ALG_DEFAULT, nullptr),
              FLAGSPARSE_STATUS_SUCCESS);
    dev_sync();
    EXPECT_EQ(d_y.download<float>(4), expect_ri);

    flagsparseDestroyDnVec(vecY);
    flagsparseDestroyDnVec(vecX);
    flagsparseDestroySpMat(matA);
}

TEST_F(SpMVAccuracy, SellInt8ToInt32MixedPrecision) {
    // Slice size 2. A = [[2,0,-1], [0,3,4], [-2,0,0]]; the final entry pads
    // the incomplete second slice and must be ignored through col=-1.
    const std::vector<int32_t> offsets{0, 4, 6};
    const std::vector<int32_t> cols{0, 1, 2, 2, 0, -1};
    const std::vector<int8_t> values{2, 3, -1, 4, -2, 0};
    const std::vector<int8_t> x{3, -2, 5};
    const std::vector<int32_t> expected{1, 14, -6};

    DeviceBuffer d_offsets = DeviceBuffer::from(offsets);
    DeviceBuffer d_cols = DeviceBuffer::from(cols);
    DeviceBuffer d_values = DeviceBuffer::from(values);
    DeviceBuffer d_x = DeviceBuffer::from(x);
    DeviceBuffer d_y = DeviceBuffer::from(std::vector<int32_t>(3, 99));

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    ASSERT_EQ(flagsparseCreateSlicedEll(
                  &matA, 3, 3, 2, d_offsets.get(), d_cols.get(), d_values.get(),
                  FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_8I),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vecX, 3, d_x.get(), FLAGSPARSE_R_8I),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vecY, 3, d_y.get(), FLAGSPARSE_R_32I),
              FLAGSPARSE_STATUS_SUCCESS);

    const int32_t one = 1, zero = 0;
    size_t buffer_size = 12345;
    ASSERT_EQ(flagsparseSpMV_bufferSize(
                  handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecX,
                  &zero, vecY, FLAGSPARSE_R_32I, FLAGSPARSE_SPMV_SELL_ALG1,
                  &buffer_size),
              FLAGSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(buffer_size, 0u);
    ASSERT_EQ(flagsparseSpMV(
                  handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, vecX,
                  &zero, vecY, FLAGSPARSE_R_32I, FLAGSPARSE_SPMV_SELL_ALG1, nullptr),
              FLAGSPARSE_STATUS_SUCCESS);
    dev_sync();
    EXPECT_EQ(d_y.download<int32_t>(3), expected);

    flagsparseDestroyDnVec(vecY);
    flagsparseDestroyDnVec(vecX);
    flagsparseDestroySpMat(matA);
}

TEST_F(SpMVAccuracy, SellFloat32MatchesHostReference) {
    const std::vector<int32_t> offsets{0, 4, 6};
    const std::vector<int32_t> cols{0, 1, 2, 2, 0, -1};
    const std::vector<float> values{2.0f, 3.0f, -1.0f, 4.0f, -2.0f, 0.0f};
    const std::vector<float> x{3.0f, -2.0f, 5.0f};
    const std::vector<float> expected{1.0f, 14.0f, -6.0f};
    DeviceBuffer d_offsets = DeviceBuffer::from(offsets);
    DeviceBuffer d_cols = DeviceBuffer::from(cols);
    DeviceBuffer d_values = DeviceBuffer::from(values);
    DeviceBuffer d_x = DeviceBuffer::from(x);
    DeviceBuffer d_y = DeviceBuffer::from(std::vector<float>(3, 99.0f));
    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    ASSERT_EQ(flagsparseCreateSlicedEll(&matA, 3, 3, 2, d_offsets.get(), d_cols.get(),
                                        d_values.get(), FLAGSPARSE_INDEX_32I,
                                        FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vecX, 3, d_x.get(), FLAGSPARSE_R_32F), FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vecY, 3, d_y.get(), FLAGSPARSE_R_32F), FLAGSPARSE_STATUS_SUCCESS);
    const float one = 1.0f, zero = 0.0f;
    size_t buffer = 0;
    ASSERT_EQ(flagsparseSpMV_bufferSize(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                        &one, matA, vecX, &zero, vecY, FLAGSPARSE_R_32F,
                                        FLAGSPARSE_SPMV_SELL_ALG1, &buffer), FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseSpMV(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                             &one, matA, vecX, &zero, vecY, FLAGSPARSE_R_32F,
                             FLAGSPARSE_SPMV_SELL_ALG1, nullptr), FLAGSPARSE_STATUS_SUCCESS);
    dev_sync();
    EXPECT_EQ(d_y.download<float>(3), expected);
    flagsparseDestroyDnVec(vecY); flagsparseDestroyDnVec(vecX); flagsparseDestroySpMat(matA);
}

TEST_F(SpMVAccuracy, SellComplex64MatchesHostReference) {
    using C = std::complex<float>;
    const std::vector<int32_t> offsets{0, 4, 6};
    const std::vector<int32_t> cols{0, 1, 2, 2, 0, -1};
    const std::vector<C> values{C(2, 1), C(3, -1), C(-1, 2), C(4, 0), C(-2, 1), C(0, 0)};
    const std::vector<C> x{C(3, 1), C(-2, 0), C(5, -1)};
    const std::vector<C> expected{C(2, 16), C(14, -2), C(-7, 1)};
    DeviceBuffer d_offsets = DeviceBuffer::from(offsets);
    DeviceBuffer d_cols = DeviceBuffer::from(cols);
    DeviceBuffer d_values = DeviceBuffer::from(values);
    DeviceBuffer d_x = DeviceBuffer::from(x);
    DeviceBuffer d_y = DeviceBuffer::from(std::vector<C>(3, C(99, 99)));
    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    ASSERT_EQ(flagsparseCreateSlicedEll(&matA, 3, 3, 2, d_offsets.get(), d_cols.get(),
                                        d_values.get(), FLAGSPARSE_INDEX_32I,
                                        FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_C_32F), FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vecX, 3, d_x.get(), FLAGSPARSE_C_32F), FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vecY, 3, d_y.get(), FLAGSPARSE_C_32F), FLAGSPARSE_STATUS_SUCCESS);
    const C one(1, 0), zero(0, 0);
    size_t buffer = 0;
    ASSERT_EQ(flagsparseSpMV_bufferSize(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                        &one, matA, vecX, &zero, vecY, FLAGSPARSE_C_32F,
                                        FLAGSPARSE_SPMV_SELL_ALG1, &buffer), FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseSpMV(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                             &one, matA, vecX, &zero, vecY, FLAGSPARSE_C_32F,
                             FLAGSPARSE_SPMV_SELL_ALG1, nullptr), FLAGSPARSE_STATUS_SUCCESS);
    dev_sync();
    EXPECT_EQ(d_y.download<C>(3), expected);
    flagsparseDestroyDnVec(vecY); flagsparseDestroyDnVec(vecX); flagsparseDestroySpMat(matA);
}

TEST_F(SpMVAccuracy, SellFloat16MatchesHostReference) {
    const std::vector<int32_t> offsets{0, 4, 6};
    const std::vector<int32_t> cols{0, 1, 2, 2, 0, -1};
    const std::vector<Half> values{2.0, 3.0, -1.0, 4.0, -2.0, 0.0};
    const std::vector<Half> x{3.0, -2.0, 5.0};
    const std::vector<Half> expected{1.0, 14.0, -6.0};
    DeviceBuffer d_offsets = DeviceBuffer::from(offsets);
    DeviceBuffer d_cols = DeviceBuffer::from(cols);
    DeviceBuffer d_values = DeviceBuffer::from(values);
    DeviceBuffer d_x = DeviceBuffer::from(x);
    DeviceBuffer d_y = DeviceBuffer::from(std::vector<Half>(3, Half(99.0)));
    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    ASSERT_EQ(flagsparseCreateSlicedEll(&matA, 3, 3, 2, d_offsets.get(), d_cols.get(),
                                        d_values.get(), FLAGSPARSE_INDEX_32I,
                                        FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_16F), FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vecX, 3, d_x.get(), FLAGSPARSE_R_16F), FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vecY, 3, d_y.get(), FLAGSPARSE_R_16F), FLAGSPARSE_STATUS_SUCCESS);
    const Half one(1.0), zero(0.0);
    size_t buffer = 0;
    ASSERT_EQ(flagsparseSpMV_bufferSize(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                        &one, matA, vecX, &zero, vecY, FLAGSPARSE_R_16F,
                                        FLAGSPARSE_SPMV_SELL_ALG1, &buffer), FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseSpMV(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                             &one, matA, vecX, &zero, vecY, FLAGSPARSE_R_16F,
                             FLAGSPARSE_SPMV_SELL_ALG1, nullptr), FLAGSPARSE_STATUS_SUCCESS);
    dev_sync();
    EXPECT_EQ(d_y.download<Half>(3), expected);
    flagsparseDestroyDnVec(vecY); flagsparseDestroyDnVec(vecX); flagsparseDestroySpMat(matA);
}

// spmv_coo_i8i32_int_non / spmv_coo_i8f32_int_non: same dtype-pair table
// as the CSR mixed tests above (mixed_narrow_wide_pair is shared), but routed
// through run_coo_mixed's nnz-parallel atomic_add kernel instead of a
// row-tiled one -- same (row,col) pairs as the CSR tests' (indptr,indices),
// just spelled as COO coordinates, so the expected sums are identical.
TEST_F(SpMVAccuracy, CooInt8ToInt32MixedPrecision) {
    const std::vector<int8_t> data = {2, -3, 1, 4, -1, 2};
    const std::vector<int32_t> rows = {0, 0, 1, 1, 2, 2};
    const std::vector<int32_t> cols = {0, 2, 2, 3, 0, 1};
    const std::vector<int8_t> x = {3, -2, 5, -1};
    const std::vector<int32_t> expect = {-9, 1, -7};

    DeviceBuffer d_data = DeviceBuffer::from(data);
    DeviceBuffer d_rows = DeviceBuffer::from(rows);
    DeviceBuffer d_cols = DeviceBuffer::from(cols);
    DeviceBuffer d_x = DeviceBuffer::from(x);
    DeviceBuffer d_y(3 * sizeof(int32_t));

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    ASSERT_EQ(flagsparseCreateCoo(&matA, 3, 4, 6, d_rows.get(), d_cols.get(), d_data.get(),
                                  FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO,
                                  FLAGSPARSE_R_8I),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vecX, 4, d_x.get(), FLAGSPARSE_R_8I),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vecY, 3, d_y.get(), FLAGSPARSE_R_32I),
              FLAGSPARSE_STATUS_SUCCESS);

    const int32_t one = 1, zero = 0;
    size_t bufsz = 12345;  // must come back 0 -- the atomic route needs no scratch.
    ASSERT_EQ(flagsparseSpMV_bufferSize(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                        &one, matA, vecX, &zero, vecY,
                                        FLAGSPARSE_R_32I, FLAGSPARSE_SPMV_ALG_DEFAULT,
                                        &bufsz),
              FLAGSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(bufsz, 0u);
    ASSERT_EQ(flagsparseSpMV(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA,
                             vecX, &zero, vecY, FLAGSPARSE_R_32I,
                             FLAGSPARSE_SPMV_ALG_DEFAULT, nullptr),
              FLAGSPARSE_STATUS_SUCCESS);
    dev_sync();
    const std::vector<int32_t> got = d_y.download<int32_t>(3);
    for (int i = 0; i < 3; ++i) EXPECT_EQ(got[i], expect[i]) << "row " << i;

    flagsparseDestroyDnVec(vecX);
    flagsparseDestroyDnVec(vecY);
    flagsparseDestroySpMat(matA);
}

TEST_F(SpMVAccuracy, CooFloat16ToFloat32MixedPrecision) {
    const std::vector<Half> data = {Half(2.5), Half(-1.5), Half(0.5), Half(4.0)};
    const std::vector<int32_t> rows = {0, 0, 1, 2};
    const std::vector<int32_t> cols = {0, 1, 1, 2};
    const std::vector<Half> x = {Half(2.0), Half(-2.0), Half(1.0)};
    // row0=2.5*2 + -1.5*-2 = 8; row1=0.5*-2=-1; row2=4*1=4
    const std::vector<float> expect = {8.0f, -1.0f, 4.0f};

    DeviceBuffer d_data = DeviceBuffer::from(data);
    DeviceBuffer d_rows = DeviceBuffer::from(rows);
    DeviceBuffer d_cols = DeviceBuffer::from(cols);
    DeviceBuffer d_x = DeviceBuffer::from(x);
    DeviceBuffer d_y(3 * sizeof(float));

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
    ASSERT_EQ(flagsparseCreateCoo(&matA, 3, 3, 4, d_rows.get(), d_cols.get(), d_data.get(),
                                  FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO,
                                  FLAGSPARSE_R_16F),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vecX, 3, d_x.get(), FLAGSPARSE_R_16F),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vecY, 3, d_y.get(), FLAGSPARSE_R_32F),
              FLAGSPARSE_STATUS_SUCCESS);

    const float one = 1.0f, zero = 0.0f;
    ASSERT_EQ(flagsparseSpMV(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA,
                             vecX, &zero, vecY, FLAGSPARSE_R_32F,
                             FLAGSPARSE_SPMV_ALG_DEFAULT, nullptr),
              FLAGSPARSE_STATUS_SUCCESS);
    dev_sync();
    const std::vector<float> got = d_y.download<float>(3);
    for (int i = 0; i < 3; ++i) EXPECT_FLOAT_EQ(got[i], expect[i]) << "row " << i;

    flagsparseDestroyDnVec(vecX);
    flagsparseDestroyDnVec(vecY);
    flagsparseDestroySpMat(matA);
}

// spmv_csr_f16_int_non: same dispatch as fp32/fp64, just fp16 storage.
// Row density kept modest (mean nnz/row ~5, matching the smaller of the
// fp32 shapes above): fp16 error is input-quantization noise that grows with
// how many nonzeros a row sums, same reason sddmm's fp16 ctest picked a
// modest k (see ctest/accuracy/test_sddmm.cpp's Float16 test).
TEST_F(SpMVAccuracy, Float16MatchesHostReference) {
    const CsrMatrix A = random_csr(64, 96, 0.03, 1234);
    const RunResult r = run_spmv<Half>(handle.h, A, FLAGSPARSE_R_16F, 1.0, 0.0, 7);
    expect_close(r, "fp16", handle.h);
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

// Types outside the fp32/c32 transpose kernels must still report the
// capability boundary instead of launching an incompatible atomic kernel.
TEST_F(SpMVAccuracy, Float64TransposeReportsNotSupported) {
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
                             &zero, vecY, FLAGSPARSE_R_64F,
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
                           flagsparseSpMVAlg_t alg = FLAGSPARSE_SPMV_ALG_DEFAULT,
                           flagsparseOperation_t opA = FLAGSPARSE_OPERATION_NON_TRANSPOSE) {
    using C64 = std::complex<double>;
    RunResult out;
    const bool trans = opA != FLAGSPARSE_OPERATION_NON_TRANSPOSE;
    const auto m = static_cast<std::size_t>(A.rows);
    const auto n = static_cast<std::size_t>(A.cols);
    const auto x_size = trans ? m : n;
    const auto y_size = trans ? n : m;

    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<C64> a_vals(static_cast<std::size_t>(A.nnz));
    for (std::size_t i = 0; i < a_vals.size(); ++i) {
        a_vals[i] = C64(A.values[i], dist(rng));
    }
    std::vector<C64> x64(x_size), y64(y_size);
    for (auto& v : x64) v = C64(dist(rng), dist(rng));
    for (auto& v : y64) v = C64(dist(rng), dist(rng));

    std::vector<R> a_dev(a_vals.size() * 2), x_dev(x_size * 2), y_dev(y_size * 2);
    for (std::size_t i = 0; i < a_vals.size(); ++i) {
        a_dev[i * 2]     = static_cast<R>(a_vals[i].real());
        a_dev[i * 2 + 1] = static_cast<R>(a_vals[i].imag());
    }
    for (std::size_t i = 0; i < x_size; ++i) {
        x_dev[i * 2] = static_cast<R>(x64[i].real());
        x_dev[i * 2 + 1] = static_cast<R>(x64[i].imag());
    }
    for (std::size_t i = 0; i < y_size; ++i) {
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
    flagsparseCreateDnVec(&vecX, static_cast<int64_t>(x_size), d_x.get(), dtype);
    flagsparseCreateDnVec(&vecY, static_cast<int64_t>(y_size), d_y.get(), dtype);

    const R alpha_t[2] = {static_cast<R>(alpha.real()), static_cast<R>(alpha.imag())};
    const R beta_t[2]  = {static_cast<R>(beta.real()),  static_cast<R>(beta.imag())};
    size_t buffer_size = 0;
    out.status = flagsparseSpMV_bufferSize(handle, opA,
                                           alpha_t, matA, vecX, beta_t, vecY, dtype, alg,
                                           &buffer_size);
    DeviceBuffer scratch(buffer_size);
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.status = flagsparseSpMV_preprocess(handle, opA,
                                               alpha_t, matA, vecX, beta_t, vecY, dtype,
                                               alg, scratch.get());
    }
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.status = flagsparseSpMV(handle, opA, alpha_t,
                                    matA, vecX, beta_t, vecY, dtype, alg, scratch.get());
    }
    dev_sync();

    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        const std::vector<R> got = d_y.download<R>(y_size * 2);
        // Real and imaginary parts compared as one flat vector: the result is
        // wrong if either component is.
        std::vector<double> actual, ref;
        actual.reserve(y_size * 2); ref.reserve(y_size * 2);
        std::vector<C64> want = y64;
        for (auto& v : want) v = beta * v;
        for (int64_t r = 0; r < A.rows; ++r) {
            for (int32_t p = A.indptr[static_cast<std::size_t>(r)];
                 p < A.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
                C64 value = a_vals[static_cast<std::size_t>(p)];
                if (opA == FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE) value = std::conj(value);
                const std::size_t in = static_cast<std::size_t>(
                    trans ? r : A.indices[static_cast<std::size_t>(p)]);
                const std::size_t out_index = static_cast<std::size_t>(
                    trans ? A.indices[static_cast<std::size_t>(p)] : r);
                want[out_index] += alpha * value * x64[in];
            }
        }
        for (std::size_t i = 0; i < y_size; ++i) {
            actual.push_back(static_cast<double>(got[i * 2]));
            actual.push_back(static_cast<double>(got[i * 2 + 1]));
            ref.push_back(want[i].real());
            ref.push_back(want[i].imag());
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

TEST_F(SpMVAccuracy, CsrAndCooFloat32TransposeMatchHostReference) {
    const CsrMatrix A = random_csr(73, 91, 0.07, 71);
    for (auto format : {FLAGSPARSE_FORMAT_CSR, FLAGSPARSE_FORMAT_COO}) {
        expect_close(run_spmv_fmt<float>(handle.h, A, format,
                                         FLAGSPARSE_OPERATION_TRANSPOSE,
                                         FLAGSPARSE_R_32F, 1.25, -0.5, 19),
                     format == FLAGSPARSE_FORMAT_CSR ? "csr_f32_trans" : "coo_f32_trans",
                     handle.h);
    }
}

#if defined(FLAGSPARSE_MUSA_BASELINE_EXTENSIONS)
TEST_F(SpMVAccuracy, MusaComplexGatherCancellation) {
    using C = std::complex<float>;
    const std::vector<C> values{C(0.07468984f, -0.07468984f),
                                C(-37.027058f, 37.027058f),
                                C(37.027058f, -37.027058f)};
    const std::vector<int32_t> ptr{0, 1, 2, 3}, indices{0, 0, 0};
    const std::vector<C> x{C(1.5f, 0), C(2, 0), C(2, 0)};
    DeviceBuffer dv = DeviceBuffer::from(values), dp = DeviceBuffer::from(ptr);
    DeviceBuffer di = DeviceBuffer::from(indices), dx = DeviceBuffer::from(x);
    DeviceBuffer dy(sizeof(C));
    flagsparseSpMatDescr_t a = nullptr;
    flagsparseDnVecDescr_t vx = nullptr, vy = nullptr;
    ASSERT_EQ(flagsparseCreateCsr(&a, 3, 1, 3, dp.get(), di.get(), dv.get(),
                                  FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_C_32F),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vx, 3, dx.get(), FLAGSPARSE_C_32F), FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&vy, 1, dy.get(), FLAGSPARSE_C_32F), FLAGSPARSE_STATUS_SUCCESS);
    const C one(1, 0), zero(0, 0);
    size_t bytes = 0;
    const auto op = FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    ASSERT_EQ(flagsparseSpMV_bufferSize(handle.h, op, &one, a, vx, &zero, vy,
                                        FLAGSPARSE_C_32F, FLAGSPARSE_SPMV_ALG_DEFAULT, &bytes),
              FLAGSPARSE_STATUS_SUCCESS);
    DeviceBuffer scratch(bytes);
    ASSERT_EQ(flagsparseSpMV_preprocess(handle.h, op, &one, a, vx, &zero, vy,
                                        FLAGSPARSE_C_32F, FLAGSPARSE_SPMV_ALG_DEFAULT, scratch.get()),
              FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseSpMV(handle.h, op, &one, a, vx, &zero, vy, FLAGSPARSE_C_32F,
                             FLAGSPARSE_SPMV_ALG_DEFAULT, scratch.get()), FLAGSPARSE_STATUS_SUCCESS);
    dev_sync();
    const C actual = dy.download<C>(1)[0];
    const double expected = static_cast<double>(values[0].real()) * 1.5;
    const Tolerance tol = default_tolerance(FLAGSPARSE_C_32F);
    EXPECT_LE(max_error_ratio({actual.real(), actual.imag()}, {expected, expected}, tol), 1.0);
    flagsparseDestroyDnVec(vy);
    flagsparseDestroyDnVec(vx);
    flagsparseDestroySpMat(a);
}
#endif

TEST_F(SpMVAccuracy, CsrAndCooComplex64ConjugateTransposeMatchHostReference) {
    const CsrMatrix A = random_csr(67, 89, 0.08, 73);
    for (auto format : {FLAGSPARSE_FORMAT_CSR, FLAGSPARSE_FORMAT_COO}) {
        expect_close(run_spmv_complex<float>(handle.h, A, format, FLAGSPARSE_C_32F,
                                              std::complex<double>(1.25, -0.75),
                                              std::complex<double>(-0.5, 0.25), 23,
                                              FLAGSPARSE_SPMV_ALG_DEFAULT,
                                              FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE),
                     format == FLAGSPARSE_FORMAT_CSR ? "csr_c32_conj" : "coo_c32_conj",
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

// spmv_coo_f16_int_non. Same modest-density reasoning as the CSR fp16
// test above (mean nnz/row ~5).
TEST_F(SpMVAccuracy, CooFloat16MatchesHostReference) {
    const CsrMatrix A = random_csr(64, 96, 0.03, 1234);
    expect_close(run_spmv_fmt<Half>(handle.h, A, FLAGSPARSE_FORMAT_COO,
                                    FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                    FLAGSPARSE_R_16F, 1.0, 0.0, 7),
                 "coo_fp16", handle.h);
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
TEST_F(SpMVAccuracy, GatherComplexDuplicatesEmptyRowsAndLiveTopology) {
    using C = std::complex<float>;
    // Same compressed structure denotes CSR(4x5)^H or CSC(5x4).
    const std::vector<int32_t> ptr{0, 2, 2, 3, 4}, indices{0, 0, 2, 1};
    const std::vector<C> values{C(1, 2), C(-2, 1), C(3, -1), C(2, 4)};
    const std::vector<C> x{C(2, 1), C(-1, 3), C(1, -2), C(4, 1)};
    const std::vector<C> initial(5, C(1, -1));
    const C alpha(1.25f, -0.75f), beta(-0.5f, 0.25f);
    for (auto format : {FLAGSPARSE_FORMAT_CSR, FLAGSPARSE_FORMAT_CSC}) {
        const bool csr = format == FLAGSPARSE_FORMAT_CSR;
        const auto op = csr ? FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE
                            : FLAGSPARSE_OPERATION_NON_TRANSPOSE;
        auto dp = DeviceBuffer::from(ptr), di = DeviceBuffer::from(indices);
        auto dv = DeviceBuffer::from(values), dx = DeviceBuffer::from(x);
        auto dy = DeviceBuffer::from(initial);
        flagsparseSpMatDescr_t a = nullptr;
        flagsparseDnVecDescr_t xd = nullptr, yd = nullptr;
        if (csr) {
            ASSERT_EQ(flagsparseCreateCsr(&a, 4, 5, 4, dp.get(), di.get(), dv.get(),
                FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO,
                FLAGSPARSE_C_32F), FLAGSPARSE_STATUS_SUCCESS);
        } else {
            ASSERT_EQ(flagsparseCreateCsc(&a, 5, 4, 4, dp.get(), di.get(), dv.get(),
                FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO,
                FLAGSPARSE_C_32F), FLAGSPARSE_STATUS_SUCCESS);
        }
        ASSERT_EQ(flagsparseCreateDnVec(&xd, 4, dx.get(), FLAGSPARSE_C_32F), FLAGSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(flagsparseCreateDnVec(&yd, 5, dy.get(), FLAGSPARSE_C_32F), FLAGSPARSE_STATUS_SUCCESS);
        size_t bytes = 0;
        ASSERT_EQ(flagsparseSpMV_bufferSize(handle.h, op, &alpha, a, xd, &beta, yd,
            FLAGSPARSE_C_32F, FLAGSPARSE_SPMV_ALG_DEFAULT, &bytes), FLAGSPARSE_STATUS_SUCCESS);
        DeviceBuffer scratch(bytes), second(bytes);
        ASSERT_EQ(flagsparseSpMV_preprocess(handle.h, op, &alpha, a, xd, &beta, yd,
            FLAGSPARSE_C_32F, FLAGSPARSE_SPMV_ALG_DEFAULT, scratch.get()), FLAGSPARSE_STATUS_SUCCESS);
        auto current_ptr = ptr;
        auto current_values = values;
        for (int step = 0; step < 3; ++step) {
            // New scratch must prepare lazily; replacing CSR pointers invalidates topology.
            if (step == 1) {
                for (auto& v : current_values) v *= C(0.5f, 0.25f);
                ASSERT_EQ(to_device(dv.get(), current_values.data(), current_values.size() * sizeof(C)),
                    FLAGSPARSE_STATUS_SUCCESS);
            }
            if (step == 2 && csr) {
                current_ptr = {0, 1, 1, 2, 4};
                ASSERT_EQ(to_device(dp.get(), current_ptr.data(), current_ptr.size() * sizeof(int32_t)),
                    FLAGSPARSE_STATUS_SUCCESS);
                ASSERT_EQ(flagsparseCsrSetPointers(a, dp.get(), di.get(), dv.get()), FLAGSPARSE_STATUS_SUCCESS);
            }
            ASSERT_EQ(to_device(dy.get(), initial.data(), initial.size() * sizeof(C)), FLAGSPARSE_STATUS_SUCCESS);
            ASSERT_EQ(flagsparseSpMV(handle.h, op, &alpha, a, xd, &beta, yd,
                FLAGSPARSE_C_32F, FLAGSPARSE_SPMV_ALG_DEFAULT,
                step == 0 ? scratch.get() : second.get()), FLAGSPARSE_STATUS_SUCCESS);
            dev_sync();
            std::vector<C> expected(5, beta * initial[0]);
            for (int r = 0; r < 4; ++r)
                for (int p = current_ptr[r]; p < current_ptr[r + 1]; ++p)
                    expected[indices[p]] += alpha * (csr ? std::conj(current_values[p]) : current_values[p]) * x[r];
            const auto got = dy.download<C>(5);
            for (size_t i = 0; i < got.size(); ++i) {
                EXPECT_NEAR(got[i].real(), expected[i].real(), 1e-5f);
                EXPECT_NEAR(got[i].imag(), expected[i].imag(), 1e-5f);
            }
        }
        flagsparseDestroyDnVec(yd); flagsparseDestroyDnVec(xd); flagsparseDestroySpMat(a);
    }
}

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

// The real-by-complex scatter routes: COO A x and A^T x, and CSR A^T x, with
// A = [[2, 0, -1], [0, 3, 4]] and complex x of non-zero imaginary part. A^T x
// takes a 2-vector, A x a 3-vector, as op(A)'s shape says.
TEST_F(SpMVAccuracy, RealFloat32ByComplex64ScatterRoutes) {
    const std::vector<float> data{2.0f, -1.0f, 3.0f, 4.0f};
    const std::vector<int32_t> rows{0, 0, 1, 1}, cols{0, 2, 1, 2}, indptr{0, 2, 4};
    struct Case { bool coo; flagsparseOperation_t op; std::vector<float> x, expect; };
    const std::vector<Case> cases{
        // A x, x = [1+2i, -2+i, 0.5-3i]
        {true, FLAGSPARSE_OPERATION_NON_TRANSPOSE, {1, 2, -2, 1, 0.5f, -3}, {1.5f, 7, -4, -9}},
        // A^T x, x = [1+2i, -1+0.5i]: [2+4i, -3+1.5i, -5]
        {true, FLAGSPARSE_OPERATION_TRANSPOSE, {1, 2, -1, 0.5f}, {2, 4, -3, 1.5f, -5, 0}},
        {false, FLAGSPARSE_OPERATION_TRANSPOSE, {1, 2, -1, 0.5f}, {2, 4, -3, 1.5f, -5, 0}},
        {false, FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE, {1, 2, -1, 0.5f},
         {2, 4, -3, 1.5f, -5, 0}},
    };
    const float one[2] = {1.0f, 0.0f}, zero[2] = {0.0f, 0.0f};
    for (const Case& c : cases) {
        DeviceBuffer d_data = DeviceBuffer::from(data), d_rows = DeviceBuffer::from(rows),
                     d_cols = DeviceBuffer::from(cols), d_ptr = DeviceBuffer::from(indptr),
                     d_x = DeviceBuffer::from(c.x),
                     d_y = DeviceBuffer::from(std::vector<float>(c.expect.size(), 99.0f));
        flagsparseSpMatDescr_t matA = nullptr;
        flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
        ASSERT_EQ(c.coo ? flagsparseCreateCoo(&matA, 2, 3, 4, d_rows.get(), d_cols.get(),
                                              d_data.get(), FLAGSPARSE_INDEX_32I,
                                              FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F)
                        : flagsparseCreateCsr(&matA, 2, 3, 4, d_ptr.get(), d_cols.get(),
                                              d_data.get(), FLAGSPARSE_INDEX_32I,
                                              FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO,
                                              FLAGSPARSE_R_32F),
                  FLAGSPARSE_STATUS_SUCCESS);
        flagsparseCreateDnVec(&vecX, static_cast<int64_t>(c.x.size() / 2), d_x.get(),
                              FLAGSPARSE_C_32F);
        flagsparseCreateDnVec(&vecY, static_cast<int64_t>(c.expect.size() / 2), d_y.get(),
                              FLAGSPARSE_C_32F);
        size_t bytes = 1;
        ASSERT_EQ(flagsparseSpMV_bufferSize(handle.h, c.op, one, matA, vecX, zero, vecY,
                                            FLAGSPARSE_C_32F, FLAGSPARSE_SPMV_ALG_DEFAULT, &bytes),
                  FLAGSPARSE_STATUS_SUCCESS);
        const bool musa_csr = !c.coo &&
            std::string(flagsparse::adaptor::backend_name()) == "musa";
        if (musa_csr) EXPECT_GT(bytes, 0u);
        else EXPECT_EQ(bytes, 0u);
        DeviceBuffer scratch(bytes);
        // Preprocessing is optional: the MUSA CSR gather builds lazily here.
        ASSERT_EQ(flagsparseSpMV(handle.h, c.op, one, matA, vecX, zero, vecY, FLAGSPARSE_C_32F,
                                 FLAGSPARSE_SPMV_ALG_DEFAULT, scratch.get()),
                  FLAGSPARSE_STATUS_SUCCESS);
        dev_sync();
        EXPECT_EQ(d_y.download<float>(c.expect.size()), c.expect)
            << (c.coo ? "coo" : "csr") << " op=" << c.op;
        flagsparseDestroyDnVec(vecY);
        flagsparseDestroyDnVec(vecX);
        flagsparseDestroySpMat(matA);
    }
}



template <typename Index>
void check_musa_real_complex_transpose(flagsparseHandle_t handle) {
    using C = std::complex<float>;
    const auto index_type = sizeof(Index) == 4 ? FLAGSPARSE_INDEX_32I : FLAGSPARSE_INDEX_64I;
    constexpr int m = 337, n = 131;
    for (auto op : {FLAGSPARSE_OPERATION_TRANSPOSE, FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE}) {
        SCOPED_TRACE(static_cast<int>(op));
        std::vector<Index> ptr(m + 1, 0), cols;
        std::vector<float> values;
        for (int row = 0; row < m; ++row) {
            auto add = [&](int col, float value) { cols.push_back(col); values.push_back(value); };
            if (row == 0) add(130, 0.0746898427605629f);
            if (row == 1) { add(0, 2.f); add(0, -1.f); }  // duplicate column
            if (row == 2) add(130, -37.02705764770508f);
            if (row == m - 1) add(130, 37.02705764770508f);
            if (row == 111) {
                for (int i = 0; i < 131; ++i) add(7, (i % 3 - 1) * .01f);
            }
            ptr[row + 1] = static_cast<Index>(cols.size());
        }
        std::vector<C> rhs(m, C(2.f, -2.f));
        rhs[0] = C(1.5f, -1.5f);
        const std::vector<C> initial(n, C(std::numeric_limits<float>::quiet_NaN(), 99.f));
        auto dp = DeviceBuffer::from(ptr), di = DeviceBuffer::from(cols);
        auto dv = DeviceBuffer::from(values), dx = DeviceBuffer::from(rhs), dy = DeviceBuffer::from(initial);
        flagsparseSpMatDescr_t a = nullptr;
        flagsparseDnVecDescr_t x = nullptr, y = nullptr;
        ASSERT_EQ(flagsparseCreateCsr(&a, m, n, values.size(), dp.get(), di.get(), dv.get(),
            index_type, index_type, FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F), FLAGSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(flagsparseCreateDnVec(&x, m, dx.get(), FLAGSPARSE_C_32F), FLAGSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(flagsparseCreateDnVec(&y, n, dy.get(), FLAGSPARSE_C_32F), FLAGSPARSE_STATUS_SUCCESS);
        const C one(1, 0), zero(0, 0), two(2, 0);
        size_t bytes = 0;
        ASSERT_EQ(flagsparseSpMV_bufferSize(handle, op, &one, a, x, &zero, y, FLAGSPARSE_C_32F,
            FLAGSPARSE_SPMV_ALG_DEFAULT, &bytes), FLAGSPARSE_STATUS_SUCCESS);
        ASSERT_GT(bytes, 0u);
        DeviceBuffer scratch_a(bytes), scratch_b(bytes);
        ASSERT_EQ(flagsparseSpMV_preprocess(handle, op, &one, a, x, &zero, y, FLAGSPARSE_C_32F,
            FLAGSPARSE_SPMV_ALG_DEFAULT, nullptr), FLAGSPARSE_STATUS_INVALID_VALUE);
        ASSERT_EQ(flagsparseSpMV(handle, op, &one, a, x, &zero, y, FLAGSPARSE_C_32F,
            FLAGSPARSE_SPMV_ALG_DEFAULT, nullptr), FLAGSPARSE_STATUS_INVALID_VALUE);
        ASSERT_EQ(flagsparseSpMV(handle, op, &two, a, x, &zero, y, FLAGSPARSE_C_32F,
            FLAGSPARSE_SPMV_ALG_DEFAULT, scratch_a.get()), FLAGSPARSE_STATUS_NOT_SUPPORTED);
        ASSERT_EQ(flagsparseSpMV_preprocess(handle, op, &one, a, x, &zero, y, FLAGSPARSE_C_32F,
            FLAGSPARSE_SPMV_ALG_DEFAULT, scratch_a.get()), FLAGSPARSE_STATUS_SUCCESS);
        for (int step = 0; step < 6; ++step) {
            SCOPED_TRACE(step);
            if (step == 1) {
                for (auto& value : values) value *= .5f;
                for (auto& value : rhs) value += C(.25f, -.5f);
                ASSERT_EQ(to_device(dv.get(), values.data(), values.size() * sizeof(float)), FLAGSPARSE_STATUS_SUCCESS);
                ASSERT_EQ(to_device(dx.get(), rhs.data(), rhs.size() * sizeof(C)), FLAGSPARSE_STATUS_SUCCESS);
            }
            if (step == 2 || step == 3) {
                for (auto& col : cols) if (col == (step == 2 ? 130 : 129)) --col;
                ASSERT_EQ(to_device(di.get(), cols.data(), cols.size() * sizeof(Index)), FLAGSPARSE_STATUS_SUCCESS);
                if (step == 2) {
                    ASSERT_EQ(flagsparseSpMV_preprocess(handle, op, &one, a, x, &zero, y, FLAGSPARSE_C_32F,
                        FLAGSPARSE_SPMV_ALG_DEFAULT, scratch_a.get()), FLAGSPARSE_STATUS_SUCCESS);
                } else {
                    ASSERT_EQ(flagsparseCsrSetPointers(a, dp.get(), di.get(), dv.get()), FLAGSPARSE_STATUS_SUCCESS);
                }
            }
            std::vector<std::complex<double>> ref(n);
            for (int row = 0; row < m; ++row) {
                for (Index p = ptr[row]; p < ptr[row + 1]; ++p)
                    ref[cols[p]] += static_cast<double>(values[p]) * std::complex<double>(rhs[row]);
            }
            std::vector<double> expected;
            for (const auto& value : ref) {
                expected.push_back(static_cast<float>(value.real()));
                expected.push_back(static_cast<float>(value.imag()));
            }
            void* scratch = step == 4 ? scratch_b.get() : scratch_a.get();
            std::vector<C> first;
            for (int repeat = 0; repeat < 16; ++repeat) {
                ASSERT_EQ(to_device(dy.get(), initial.data(), initial.size() * sizeof(C)), FLAGSPARSE_STATUS_SUCCESS);
                ASSERT_EQ(flagsparseSpMV(handle, op, &one, a, x, &zero, y, FLAGSPARSE_C_32F,
                    FLAGSPARSE_SPMV_ALG_DEFAULT, scratch), FLAGSPARSE_STATUS_SUCCESS);
                dev_sync();
                const auto got = dy.download<C>(n);
                std::vector<double> actual;
                for (const auto& value : got) { actual.push_back(value.real()); actual.push_back(value.imag()); }
                ASSERT_LE(max_error_ratio(actual, expected, default_tolerance(FLAGSPARSE_C_32F)), 1.0);
                if (repeat == 0) first = got;
                else EXPECT_EQ(std::memcmp(first.data(), got.data(), n * sizeof(C)), 0);
            }
        }
        flagsparseDestroyDnVec(y); flagsparseDestroyDnVec(x); flagsparseDestroySpMat(a);
    }
}

template <typename Index>
void check_musa_csr_fp32_transpose(flagsparseHandle_t handle) {
    const auto index_type = sizeof(Index) == 4 ? FLAGSPARSE_INDEX_32I : FLAGSPARSE_INDEX_64I;
    constexpr int m = 1033;
    for (int n : {4, 512}) for (auto op : {FLAGSPARSE_OPERATION_TRANSPOSE, FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE}) {
        std::vector<Index> ptr(m + 1), cols(m, 0);
        std::vector<float> values(m, 1.f), rhs(m, 1.f);
        for (int row = 0; row <= m; ++row) ptr[row] = Index(row);
        values.front() = 1e8f;
        values.back() = -1e8f;
        const std::vector<float> initial(n, 2.f);
        auto dp = DeviceBuffer::from(ptr), di = DeviceBuffer::from(cols), dv = DeviceBuffer::from(values);
        auto dx = DeviceBuffer::from(rhs), dy = DeviceBuffer::from(initial);
        flagsparseSpMatDescr_t a = nullptr;
        flagsparseDnVecDescr_t x = nullptr, y = nullptr;
        ASSERT_EQ(flagsparseCreateCsr(&a, m, n, values.size(), dp.get(), di.get(), dv.get(),
            index_type, index_type, FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F), FLAGSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(flagsparseCreateDnVec(&x, m, dx.get(), FLAGSPARSE_R_32F), FLAGSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(flagsparseCreateDnVec(&y, n, dy.get(), FLAGSPARSE_R_32F), FLAGSPARSE_STATUS_SUCCESS);
        size_t bytes = 0;
        const float one = 1.f, zero = 0.f;
        ASSERT_EQ(flagsparseSpMV_bufferSize(handle, op, &one, a, x, &zero, y, FLAGSPARSE_R_32F,
            FLAGSPARSE_SPMV_ALG_DEFAULT, &bytes), FLAGSPARSE_STATUS_SUCCESS);
        DeviceBuffer scratch_a(bytes), scratch_b(bytes);
        for (int step = 0; step < 5; ++step) {
            const float alpha = step == 0 ? 1.f : 2.f, beta = step == 0 ? 0.f : -.5f;
            if (step == 1) { values[1] = 2.f; rhs[1] = 2.f; }
            if (step == 2) cols[2] = Index(3);
            ASSERT_EQ(to_device(dv.get(), values.data(), values.size() * sizeof(float)), FLAGSPARSE_STATUS_SUCCESS);
            ASSERT_EQ(to_device(dx.get(), rhs.data(), rhs.size() * sizeof(float)), FLAGSPARSE_STATUS_SUCCESS);
            ASSERT_EQ(to_device(di.get(), cols.data(), cols.size() * sizeof(Index)), FLAGSPARSE_STATUS_SUCCESS);
            if (step == 3)
                ASSERT_EQ(flagsparseCsrSetPointers(a, dp.get(), di.get(), dv.get()), FLAGSPARSE_STATUS_SUCCESS);
            void* scratch = step == 4 ? scratch_b.get() : scratch_a.get();
            if (step == 0 || step == 2)
                ASSERT_EQ(flagsparseSpMV_preprocess(handle, op, &alpha, a, x, &beta, y,
                    FLAGSPARSE_R_32F, FLAGSPARSE_SPMV_ALG_DEFAULT, scratch), FLAGSPARSE_STATUS_SUCCESS);
            std::vector<double> ref(n, 0.);
            for (int row = 0; row < m; ++row) ref[cols[row]] += double(values[row]) * rhs[row];
            std::vector<double> expected(n);
            for (int col = 0; col < n; ++col) expected[col] = float(double(alpha) * ref[col] + double(beta) * initial[col]);
            for (int repeat = 0; repeat < 8; ++repeat) {
                auto input = initial;
                if (beta == 0.f) std::fill(input.begin(), input.end(), std::numeric_limits<float>::quiet_NaN());
                ASSERT_EQ(to_device(dy.get(), input.data(), input.size() * sizeof(float)), FLAGSPARSE_STATUS_SUCCESS);
                ASSERT_EQ(flagsparseSpMV(handle, op, &alpha, a, x, &beta, y, FLAGSPARSE_R_32F,
                    FLAGSPARSE_SPMV_ALG_DEFAULT, scratch), FLAGSPARSE_STATUS_SUCCESS);
                dev_sync();
                auto got = dy.download<float>(n);
                std::vector<double> actual(got.begin(), got.end());
                ASSERT_LE(max_error_ratio(actual, expected, default_tolerance(FLAGSPARSE_R_32F)), 1.0)
                    << "n=" << n << " step=" << step << " repeat=" << repeat;
            }
        }
        flagsparseDestroyDnVec(y); flagsparseDestroyDnVec(x); flagsparseDestroySpMat(a);
    }
}

TEST_F(SpMVAccuracy, MusaCsrFp32TransposeCancellationAndLiveInputs) {
    if (std::string(flagsparse::adaptor::backend_name()) != "musa") GTEST_SKIP();
    check_musa_csr_fp32_transpose<int32_t>(handle.h);
    check_musa_csr_fp32_transpose<int64_t>(handle.h);
}

TEST_F(SpMVAccuracy, MusaCsrFp32TransposeEmptyInputs) {
    if (std::string(flagsparse::adaptor::backend_name()) != "musa") GTEST_SKIP();
    for (auto shape : {std::pair<int, int>{0, 5}, {7, 5}, {7, 0}}) {
        const int m = shape.first, n = shape.second;
        auto dp = DeviceBuffer::from(std::vector<int32_t>(m + 1, 0));
        auto dx = DeviceBuffer::from(std::vector<float>(m, 1.f));
        auto dy = DeviceBuffer::from(std::vector<float>(n, 2.f));
        flagsparseSpMatDescr_t a = nullptr;
        flagsparseDnVecDescr_t x = nullptr, y = nullptr;
        ASSERT_EQ(flagsparseCreateCsr(&a, m, n, 0, dp.get(), nullptr, nullptr, FLAGSPARSE_INDEX_32I,
            FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F), FLAGSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(flagsparseCreateDnVec(&x, m, dx.get(), FLAGSPARSE_R_32F), FLAGSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(flagsparseCreateDnVec(&y, n, dy.get(), FLAGSPARSE_R_32F), FLAGSPARSE_STATUS_SUCCESS);
        for (float beta : {0.f, -.5f}) {
            const float alpha = 2.f;
            size_t bytes = 0;
            ASSERT_EQ(flagsparseSpMV_bufferSize(handle.h, FLAGSPARSE_OPERATION_TRANSPOSE,
                &alpha, a, x, &beta, y, FLAGSPARSE_R_32F, FLAGSPARSE_SPMV_ALG_DEFAULT,
                &bytes), FLAGSPARSE_STATUS_SUCCESS);
            DeviceBuffer scratch(bytes);
            std::vector<float> initial(n, beta == 0.f ? std::numeric_limits<float>::quiet_NaN() : 2.f);
            ASSERT_EQ(to_device(dy.get(), initial.data(), initial.size() * sizeof(float)), FLAGSPARSE_STATUS_SUCCESS);
            ASSERT_EQ(flagsparseSpMV(handle.h, FLAGSPARSE_OPERATION_TRANSPOSE, &alpha, a, x,
                &beta, y, FLAGSPARSE_R_32F, FLAGSPARSE_SPMV_ALG_DEFAULT, scratch.get()),
                FLAGSPARSE_STATUS_SUCCESS);
            dev_sync();
            EXPECT_EQ(dy.download<float>(n), std::vector<float>(n, beta * 2.f));
        }
        flagsparseDestroyDnVec(y); flagsparseDestroyDnVec(x); flagsparseDestroySpMat(a);
    }
}

template <typename Index>
void check_musa_coo_transpose(flagsparseHandle_t handle) {
    const auto index_type = sizeof(Index) == 4 ? FLAGSPARSE_INDEX_32I : FLAGSPARSE_INDEX_64I;
    constexpr int base = 0;  // SpMV currently supports zero-based indices.
    for (int n : {4, 512}) for (auto op : {FLAGSPARSE_OPERATION_TRANSPOSE, FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE}) {
        std::vector<Index> rows = {Index(base + 2), Index(base), Index(base + 1), Index(base + 1)};
        std::vector<Index> cols = {Index(base), Index(base), Index(base), Index(base + 2)};
        std::vector<float> values = {-1e8f, 1.f, 1e8f, 3.f}, rhs = {1.f, 1.f, 1.f};
        // Cross multiple NNZ blocks, including a prefix containing large terms.
        for (int i = 0; i < 1029; ++i) {
            rows.push_back(Index(base + i % 3));
            cols.push_back(Index(base + 2));
            values.push_back(i == 0 ? 1e8f : i == 1028 ? -1e8f : 1.f);
        }
        std::vector<float> initial(n, 2.f);
        auto dr = DeviceBuffer::from(rows), dc = DeviceBuffer::from(cols), dv = DeviceBuffer::from(values);
        auto dx = DeviceBuffer::from(rhs), dy = DeviceBuffer::from(initial);
        flagsparseSpMatDescr_t a = nullptr;
        flagsparseDnVecDescr_t x = nullptr, y = nullptr;
        ASSERT_EQ(flagsparseCreateCoo(&a, 3, n, values.size(), dr.get(), dc.get(), dv.get(), index_type,
            base ? FLAGSPARSE_INDEX_BASE_ONE : FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F), FLAGSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(flagsparseCreateDnVec(&x, 3, dx.get(), FLAGSPARSE_R_32F), FLAGSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(flagsparseCreateDnVec(&y, n, dy.get(), FLAGSPARSE_R_32F), FLAGSPARSE_STATUS_SUCCESS);
        for (int step = 0; step < 4; ++step) {
            const float alpha = step == 0 ? 1.f : 2.f, beta = step == 0 ? 0.f : -.5f;
            if (step == 1) { values[1] = 2.f; rhs[0] = 2.f; }
            if (step == 2) cols[3] = Index(base + 3);
            ASSERT_EQ(to_device(dv.get(), values.data(), values.size() * sizeof(float)), FLAGSPARSE_STATUS_SUCCESS);
            ASSERT_EQ(to_device(dx.get(), rhs.data(), rhs.size() * sizeof(float)), FLAGSPARSE_STATUS_SUCCESS);
            ASSERT_EQ(to_device(dc.get(), cols.data(), cols.size() * sizeof(Index)), FLAGSPARSE_STATUS_SUCCESS);
            size_t bytes = 0;
            ASSERT_EQ(flagsparseSpMV_bufferSize(handle, op, &alpha, a, x, &beta, y, FLAGSPARSE_R_32F,
                FLAGSPARSE_SPMV_ALG_DEFAULT, &bytes), FLAGSPARSE_STATUS_SUCCESS);
            DeviceBuffer scratch(bytes);
            ASSERT_GT(bytes, 0u);
            if (step != 3) ASSERT_EQ(flagsparseSpMV_preprocess(handle, op, &alpha, a, x, &beta, y,
                FLAGSPARSE_R_32F, FLAGSPARSE_SPMV_ALG_DEFAULT, scratch.get()), FLAGSPARSE_STATUS_SUCCESS);
            std::vector<double> expected(n, beta * 2.);
            for (size_t p = 0; p < values.size(); ++p)
                expected[cols[p] - base] += double(alpha) * values[p] * rhs[rows[p] - base];
            for (int repeat = 0; repeat < 8; ++repeat) {
                auto input = initial;
                if (beta == 0.f) std::fill(input.begin(), input.end(), std::numeric_limits<float>::quiet_NaN());
                ASSERT_EQ(to_device(dy.get(), input.data(), input.size() * sizeof(float)), FLAGSPARSE_STATUS_SUCCESS);
                ASSERT_EQ(flagsparseSpMV(handle, op, &alpha, a, x, &beta, y, FLAGSPARSE_R_32F,
                    FLAGSPARSE_SPMV_ALG_DEFAULT, scratch.get()), FLAGSPARSE_STATUS_SUCCESS);
                dev_sync();
                auto got = dy.download<float>(n);
                for (int i = 0; i < n; ++i) EXPECT_FLOAT_EQ(got[i], static_cast<float>(expected[i]));
            }
        }
        flagsparseDestroyDnVec(y); flagsparseDestroyDnVec(x); flagsparseDestroySpMat(a);
    }
}

TEST_F(SpMVAccuracy, MusaCooTransposeCancellationAndLiveInputs) {
    if (std::string(flagsparse::adaptor::backend_name()) != "musa") GTEST_SKIP();
    check_musa_coo_transpose<int32_t>(handle.h);
    check_musa_coo_transpose<int64_t>(handle.h);
}

TEST_F(SpMVAccuracy, MusaCooTransposeEmptyInputs) {
    if (std::string(flagsparse::adaptor::backend_name()) != "musa") GTEST_SKIP();
    for (auto shape : {std::pair<int, int>{0, 5}, {7, 5}, {7, 0}}) {
        const int m = shape.first, n = shape.second;
        auto dx = DeviceBuffer::from(std::vector<float>(m, 1.f));
        auto dy = DeviceBuffer::from(std::vector<float>(n, 2.f));
        flagsparseSpMatDescr_t a = nullptr;
        flagsparseDnVecDescr_t x = nullptr, y = nullptr;
        ASSERT_EQ(flagsparseCreateCoo(&a, m, n, 0, nullptr, nullptr, nullptr,
            FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F),
            FLAGSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(flagsparseCreateDnVec(&x, m, dx.get(), FLAGSPARSE_R_32F), FLAGSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(flagsparseCreateDnVec(&y, n, dy.get(), FLAGSPARSE_R_32F), FLAGSPARSE_STATUS_SUCCESS);
        for (float beta : {0.f, -.5f}) {
            const float alpha = 2.f;
            size_t bytes = 0;
            ASSERT_EQ(flagsparseSpMV_bufferSize(handle.h, FLAGSPARSE_OPERATION_TRANSPOSE,
                &alpha, a, x, &beta, y, FLAGSPARSE_R_32F, FLAGSPARSE_SPMV_ALG_DEFAULT,
                &bytes), FLAGSPARSE_STATUS_SUCCESS);
            DeviceBuffer scratch(bytes);
            ASSERT_EQ(flagsparseSpMV_preprocess(handle.h, FLAGSPARSE_OPERATION_TRANSPOSE,
                &alpha, a, x, &beta, y, FLAGSPARSE_R_32F, FLAGSPARSE_SPMV_ALG_DEFAULT,
                scratch.get()), FLAGSPARSE_STATUS_SUCCESS);
            std::vector<float> initial(n, beta == 0.f ? std::numeric_limits<float>::quiet_NaN() : 2.f);
            ASSERT_EQ(to_device(dy.get(), initial.data(), n * sizeof(float)), FLAGSPARSE_STATUS_SUCCESS);
            ASSERT_EQ(flagsparseSpMV(handle.h, FLAGSPARSE_OPERATION_TRANSPOSE, &alpha, a, x,
                &beta, y, FLAGSPARSE_R_32F, FLAGSPARSE_SPMV_ALG_DEFAULT, scratch.get()),
                FLAGSPARSE_STATUS_SUCCESS);
            EXPECT_EQ(dy.download<float>(n), std::vector<float>(n, beta * 2.f));
        }
        flagsparseDestroyDnVec(y); flagsparseDestroyDnVec(x); flagsparseDestroySpMat(a);
    }
}

TEST_F(SpMVAccuracy, MusaRealComplexTransposeCancellationAndLiveTopology) {
    if (std::string(flagsparse::adaptor::backend_name()) != "musa") GTEST_SKIP();
    check_musa_real_complex_transpose<int32_t>(handle.h);
    check_musa_real_complex_transpose<int64_t>(handle.h);
}

#if defined(FLAGSPARSE_MUSA_BASELINE_EXTENSIONS)
TEST_F(SpMVAccuracy, MusaRealComplexTransposeNondefaultStream) {
    struct Stream {
        MUstream value = nullptr;
        ~Stream() { if (value) muStreamDestroy(value); }
    } stream;
    ASSERT_EQ(muStreamCreate(&stream.value, MU_STREAM_NON_BLOCKING), MUSA_SUCCESS);
    ASSERT_EQ(flagsparseSetStream(handle.h, reinterpret_cast<flagsparseStream_t>(stream.value)),
              FLAGSPARSE_STATUS_SUCCESS);
    check_musa_real_complex_transpose<int32_t>(handle.h);
    check_musa_csr_fp32_transpose<int32_t>(handle.h);
    check_musa_csr_fp32_transpose<int64_t>(handle.h);
    check_musa_coo_transpose<int32_t>(handle.h);
    check_musa_coo_transpose<int64_t>(handle.h);
    ASSERT_EQ(flagsparseSetStream(handle.h, nullptr), FLAGSPARSE_STATUS_SUCCESS);
}
#endif

TEST_F(SpMVAccuracy, MusaRealComplexTransposeEmptyInputs) {
    if (std::string(flagsparse::adaptor::backend_name()) != "musa") GTEST_SKIP();
    using C = std::complex<float>;
    for (const auto& shape : {std::pair<int, int>{0, 5}, {7, 5}, {7, 0}}) {
        const int m = shape.first, n = shape.second;
        auto dp = DeviceBuffer::from(std::vector<int32_t>(m + 1, 0));
        auto dx = DeviceBuffer::from(std::vector<C>(m, C(1, 2)));
        auto dy = DeviceBuffer::from(std::vector<C>(n, C(99, 99)));
        flagsparseSpMatDescr_t a = nullptr;
        flagsparseDnVecDescr_t x = nullptr, y = nullptr;
        ASSERT_EQ(flagsparseCreateCsr(&a, m, n, 0, dp.get(), nullptr, nullptr, FLAGSPARSE_INDEX_32I,
            FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F), FLAGSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(flagsparseCreateDnVec(&x, m, dx.get(), FLAGSPARSE_C_32F), FLAGSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(flagsparseCreateDnVec(&y, n, dy.get(), FLAGSPARSE_C_32F), FLAGSPARSE_STATUS_SUCCESS);
        const C one(1, 0), zero(0, 0);
        size_t bytes = 0;
        ASSERT_EQ(flagsparseSpMV_bufferSize(handle.h, FLAGSPARSE_OPERATION_TRANSPOSE, &one, a, x,
            &zero, y, FLAGSPARSE_C_32F, FLAGSPARSE_SPMV_ALG_DEFAULT, &bytes), FLAGSPARSE_STATUS_SUCCESS);
        DeviceBuffer scratch(bytes);
        ASSERT_EQ(flagsparseSpMV(handle.h, FLAGSPARSE_OPERATION_TRANSPOSE, &one, a, x, &zero, y,
            FLAGSPARSE_C_32F, FLAGSPARSE_SPMV_ALG_DEFAULT, scratch.get()), FLAGSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(dy.download<C>(n), std::vector<C>(n, C(0, 0)));
        flagsparseDestroyDnVec(y); flagsparseDestroyDnVec(x); flagsparseDestroySpMat(a);
    }
}

template <typename Index>
void check_musa_coo_complex_transpose(flagsparseHandle_t handle) {
    using C = std::complex<float>;
    const auto index_type = sizeof(Index) == 4 ? FLAGSPARSE_INDEX_32I : FLAGSPARSE_INDEX_64I;
    constexpr int m = 7, nnz = 1033;
    for (int n : {4, 513}) for (auto op : {FLAGSPARSE_OPERATION_TRANSPOSE, FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE}) {
        std::vector<Index> rows(nnz), cols(nnz, 0);
        std::vector<C> values(nnz, C(1, 2)), rhs(m), initial(n, C(2, -1));
        for (int i = 0; i < nnz; ++i) rows[i] = Index(i % m);
        rows.front() = rows.back() = 0;
        values.front() = C(1e8f, -1e8f); values.back() = -values.front();
        for (int i = 0; i < m; ++i) rhs[i] = C(1.f + .25f * i, .5f - .25f * i);
        auto dr = DeviceBuffer::from(rows), dc = DeviceBuffer::from(cols), dv = DeviceBuffer::from(values);
        auto dx = DeviceBuffer::from(rhs), dy = DeviceBuffer::from(initial);
        flagsparseSpMatDescr_t a = nullptr;
        flagsparseDnVecDescr_t x = nullptr, y = nullptr;
        ASSERT_EQ(flagsparseCreateCoo(&a, m, n, nnz, dr.get(), dc.get(), dv.get(), index_type,
            FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_C_32F), FLAGSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(flagsparseCreateDnVec(&x, m, dx.get(), FLAGSPARSE_C_32F), FLAGSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(flagsparseCreateDnVec(&y, n, dy.get(), FLAGSPARSE_C_32F), FLAGSPARSE_STATUS_SUCCESS);
        for (int step = 0; step < 4; ++step) {
            const C alpha = step == 0 ? C(1, 0) : C(1.25f, -.75f);
            const C beta = step == 0 ? C(0, 0) : C(-.5f, .25f);
            if (step == 1) { values[1] = C(3, -2); rhs[1] = C(2, -1); }
            if (step == 2) { cols[2] = Index(n - 1); rows[3] = Index(5); }
            ASSERT_EQ(to_device(dr.get(), rows.data(), rows.size() * sizeof(Index)), FLAGSPARSE_STATUS_SUCCESS);
            ASSERT_EQ(to_device(dc.get(), cols.data(), cols.size() * sizeof(Index)), FLAGSPARSE_STATUS_SUCCESS);
            ASSERT_EQ(to_device(dv.get(), values.data(), values.size() * sizeof(C)), FLAGSPARSE_STATUS_SUCCESS);
            ASSERT_EQ(to_device(dx.get(), rhs.data(), rhs.size() * sizeof(C)), FLAGSPARSE_STATUS_SUCCESS);
            size_t bytes = 0;
            ASSERT_EQ(flagsparseSpMV_bufferSize(handle, op, &alpha, a, x, &beta, y, FLAGSPARSE_C_32F,
                FLAGSPARSE_SPMV_ALG_DEFAULT, &bytes), FLAGSPARSE_STATUS_SUCCESS);
            DeviceBuffer scratch(bytes);
            if (step != 3)
                ASSERT_EQ(flagsparseSpMV_preprocess(handle, op, &alpha, a, x, &beta, y,
                    FLAGSPARSE_C_32F, FLAGSPARSE_SPMV_ALG_DEFAULT, scratch.get()), FLAGSPARSE_STATUS_SUCCESS);
            std::vector<std::complex<double>> sum(n, 0.);
            for (int p = 0; p < nnz; ++p) {
                std::complex<double> v(values[p]);
                if (op == FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE) v = std::conj(v);
                sum[cols[p]] += v * std::complex<double>(rhs[rows[p]]);
            }
            std::vector<double> expected;
            for (int col = 0; col < n; ++col) {
                const auto value = std::complex<double>(alpha) * sum[col] + std::complex<double>(beta) * std::complex<double>(initial[col]);
                expected.push_back(float(value.real())); expected.push_back(float(value.imag()));
            }
            for (int repeat = 0; repeat < 8; ++repeat) {
                auto input = initial;
                if (beta == C(0, 0)) std::fill(input.begin(), input.end(), C(NAN, NAN));
                ASSERT_EQ(to_device(dy.get(), input.data(), input.size() * sizeof(C)), FLAGSPARSE_STATUS_SUCCESS);
                ASSERT_EQ(flagsparseSpMV(handle, op, &alpha, a, x, &beta, y, FLAGSPARSE_C_32F,
                    FLAGSPARSE_SPMV_ALG_DEFAULT, scratch.get()), FLAGSPARSE_STATUS_SUCCESS);
                dev_sync();
                std::vector<double> actual;
                for (const auto v : dy.download<C>(n)) { actual.push_back(v.real()); actual.push_back(v.imag()); }
                ASSERT_LE(max_error_ratio(actual, expected, default_tolerance(FLAGSPARSE_C_32F)), 1.)
                    << "n=" << n << " op=" << int(op) << " step=" << step;
            }
        }
        flagsparseDestroyDnVec(y); flagsparseDestroyDnVec(x); flagsparseDestroySpMat(a);
    }
}

template <typename Index>
void check_musa_csc_fp32_transpose(flagsparseHandle_t handle) {
    const auto index_type = sizeof(Index) == 4 ? FLAGSPARSE_INDEX_32I : FLAGSPARSE_INDEX_64I;
    constexpr int m = 1033;
    for (int n : {4, 67, 129}) for (auto op : {FLAGSPARSE_OPERATION_TRANSPOSE, FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE}) {
        std::vector<Index> ptr(n + 1, Index(m)), rows(m);
        ptr[0] = 0;
        std::vector<float> values(m, 1.f), rhs(m, 1.f), initial(n, 2.f);
        for (int row = 0; row < m; ++row) rows[row] = Index(row);
        values.front() = 1e8f; values.back() = -1e8f;
        auto dp = DeviceBuffer::from(ptr), di = DeviceBuffer::from(rows), dv = DeviceBuffer::from(values);
        auto dx = DeviceBuffer::from(rhs), dy = DeviceBuffer::from(initial);
        flagsparseSpMatDescr_t a = nullptr;
        flagsparseDnVecDescr_t x = nullptr, y = nullptr;
        ASSERT_EQ(flagsparseCreateCsc(&a, m, n, m, dp.get(), di.get(), dv.get(), index_type,
            index_type, FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F), FLAGSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(flagsparseCreateDnVec(&x, m, dx.get(), FLAGSPARSE_R_32F), FLAGSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(flagsparseCreateDnVec(&y, n, dy.get(), FLAGSPARSE_R_32F), FLAGSPARSE_STATUS_SUCCESS);
        for (int step = 0; step < 3; ++step) {
            const float alpha = step == 0 ? 1.f : 1.25f, beta = step == 0 ? 0.f : -.5f;
            if (step == 1) { values[1] = 2.f; rhs[1] = 2.f; }
            if (step == 2) rows[2] = Index(1);
            ASSERT_EQ(to_device(di.get(), rows.data(), rows.size() * sizeof(Index)), FLAGSPARSE_STATUS_SUCCESS);
            ASSERT_EQ(to_device(dv.get(), values.data(), values.size() * sizeof(float)), FLAGSPARSE_STATUS_SUCCESS);
            ASSERT_EQ(to_device(dx.get(), rhs.data(), rhs.size() * sizeof(float)), FLAGSPARSE_STATUS_SUCCESS);
            size_t bytes = 0;
            ASSERT_EQ(flagsparseSpMV_bufferSize(handle, op, &alpha, a, x, &beta, y, FLAGSPARSE_R_32F,
                FLAGSPARSE_SPMV_ALG_DEFAULT, &bytes), FLAGSPARSE_STATUS_SUCCESS);
            DeviceBuffer scratch(bytes);
            if (step == 0)
                ASSERT_EQ(flagsparseSpMV_preprocess(handle, op, &alpha, a, x, &beta, y, FLAGSPARSE_R_32F,
                    FLAGSPARSE_SPMV_ALG_DEFAULT, scratch.get()), FLAGSPARSE_STATUS_SUCCESS);
            std::vector<double> expected(n, beta * 2.);
            for (int col = 0; col < n; ++col) {
                double sum = 0.;
                for (Index p = ptr[col]; p < ptr[col + 1]; ++p) sum += double(values[p]) * rhs[rows[p]];
                expected[col] = float(double(alpha) * sum + double(beta) * initial[col]);
            }
            for (int repeat = 0; repeat < 4; ++repeat) {
                auto input = initial;
                if (beta == 0.f) std::fill(input.begin(), input.end(), NAN);
                ASSERT_EQ(to_device(dy.get(), input.data(), input.size() * sizeof(float)), FLAGSPARSE_STATUS_SUCCESS);
                ASSERT_EQ(flagsparseSpMV(handle, op, &alpha, a, x, &beta, y, FLAGSPARSE_R_32F,
                    FLAGSPARSE_SPMV_ALG_DEFAULT, scratch.get()), FLAGSPARSE_STATUS_SUCCESS);
                dev_sync();
                const auto got = dy.download<float>(n);
                ASSERT_LE(max_error_ratio(std::vector<double>(got.begin(), got.end()), expected,
                    default_tolerance(FLAGSPARSE_R_32F)), 1.) << "n=" << n << " step=" << step;
            }
        }
        flagsparseDestroyDnVec(y); flagsparseDestroyDnVec(x); flagsparseDestroySpMat(a);
    }
}

TEST_F(SpMVAccuracy, MusaCooComplexTransposeStrictCancellation) {
    if (std::string(flagsparse::adaptor::backend_name()) != "musa") GTEST_SKIP();
    check_musa_coo_complex_transpose<int32_t>(handle.h);
    check_musa_coo_complex_transpose<int64_t>(handle.h);
}

TEST_F(SpMVAccuracy, MusaCscFp32TransposeStrictCancellation) {
    if (std::string(flagsparse::adaptor::backend_name()) != "musa") GTEST_SKIP();
    check_musa_csc_fp32_transpose<int32_t>(handle.h);
    check_musa_csc_fp32_transpose<int64_t>(handle.h);
}

TEST_F(SpMVAccuracy, MusaRemainingTransposeStrictEmptyInputs) {
    if (std::string(flagsparse::adaptor::backend_name()) != "musa") GTEST_SKIP();
    for (bool coo_complex : {false, true}) for (auto shape : {std::pair<int, int>{0, 5}, {7, 5}, {7, 0}}) {
        const int m = shape.first, n = shape.second, unit = coo_complex ? 2 : 1;
        const auto dtype = coo_complex ? FLAGSPARSE_C_32F : FLAGSPARSE_R_32F;
        auto dp = DeviceBuffer::from(std::vector<int32_t>(n + 1, 0));
        auto dx = DeviceBuffer::from(std::vector<float>(m * unit, 1.f));
        auto dy = DeviceBuffer::from(std::vector<float>(n * unit, 2.f));
        flagsparseSpMatDescr_t a = nullptr;
        flagsparseDnVecDescr_t x = nullptr, y = nullptr;
        ASSERT_EQ(coo_complex ? flagsparseCreateCoo(&a, m, n, 0, nullptr, nullptr, nullptr,
            FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO, dtype) :
            flagsparseCreateCsc(&a, m, n, 0, dp.get(), nullptr, nullptr, FLAGSPARSE_INDEX_32I,
                FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO, dtype), FLAGSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(flagsparseCreateDnVec(&x, m, dx.get(), dtype), FLAGSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(flagsparseCreateDnVec(&y, n, dy.get(), dtype), FLAGSPARSE_STATUS_SUCCESS);
        for (bool has_beta : {false, true}) {
            const float alpha[2] = {1.25f, -.75f}, beta[2] = {has_beta ? -.5f : 0.f, has_beta ? .25f : 0.f};
            size_t bytes = 0;
            const auto op = FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
            ASSERT_EQ(flagsparseSpMV_bufferSize(handle.h, op, alpha, a, x, beta, y, dtype,
                FLAGSPARSE_SPMV_ALG_DEFAULT, &bytes), FLAGSPARSE_STATUS_SUCCESS);
            DeviceBuffer scratch(bytes);
            std::vector<float> initial(n * unit, has_beta ? 2.f : NAN);
            ASSERT_EQ(to_device(dy.get(), initial.data(), initial.size() * sizeof(float)), FLAGSPARSE_STATUS_SUCCESS);
            ASSERT_EQ(flagsparseSpMV(handle.h, op, alpha, a, x, beta, y, dtype,
                FLAGSPARSE_SPMV_ALG_DEFAULT, scratch.get()), FLAGSPARSE_STATUS_SUCCESS);
            dev_sync();
            std::vector<float> expected(n * unit, has_beta ? -1.f : 0.f);
            if (coo_complex && has_beta) for (int i = 0; i < n; ++i) {
                expected[2 * i] = -1.5f; expected[2 * i + 1] = -.5f;
            }
            EXPECT_EQ(dy.download<float>(n * unit), expected);
        }
        flagsparseDestroyDnVec(y); flagsparseDestroyDnVec(x); flagsparseDestroySpMat(a);
    }
}

#if defined(FLAGSPARSE_MUSA_BASELINE_EXTENSIONS)
TEST_F(SpMVAccuracy, MusaRemainingTransposeStrictNondefaultStream) {
    struct Stream {
        MUstream value = nullptr;
        ~Stream() { if (value) muStreamDestroy(value); }
    } stream;
    ASSERT_EQ(muStreamCreate(&stream.value, MU_STREAM_NON_BLOCKING), MUSA_SUCCESS);
    ASSERT_EQ(flagsparseSetStream(handle.h, reinterpret_cast<flagsparseStream_t>(stream.value)), FLAGSPARSE_STATUS_SUCCESS);
    check_musa_coo_complex_transpose<int32_t>(handle.h);
    check_musa_csc_fp32_transpose<int64_t>(handle.h);
    ASSERT_EQ(flagsparseSetStream(handle.h, nullptr), FLAGSPARSE_STATUS_SUCCESS);
}
#endif

TEST_F(SpMVAccuracy, MusaCscConjugateDirectReadsLiveValues) {
    if (std::string(flagsparse::adaptor::backend_name()) != "musa") GTEST_SKIP();
    using C = std::complex<float>;
    const std::vector<int32_t> ptr{0, 2, 2, 3, 4}, idx{0, 0, 2, 1};
    std::vector<C> values{C(1, 2), C(-2, 1), C(3, -1), C(2, 4)};
    std::vector<C> rhs{C(2, 1), C(-1, 3), C(1, -2), C(4, 1), C(2, -3)};
    const std::vector<C> initial(4, C(1, -1));
    const C alpha(1.25f, -0.75f), beta(-0.5f, 0.25f);
    auto dp = DeviceBuffer::from(ptr), di = DeviceBuffer::from(idx), dv = DeviceBuffer::from(values);
    auto dx = DeviceBuffer::from(rhs), dy = DeviceBuffer::from(initial);
    flagsparseSpMatDescr_t a = nullptr;
    flagsparseDnVecDescr_t x = nullptr, y = nullptr;
    ASSERT_EQ(flagsparseCreateCsc(&a, 5, 4, 4, dp.get(), di.get(), dv.get(), FLAGSPARSE_INDEX_32I,
        FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_C_32F), FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&x, 5, dx.get(), FLAGSPARSE_C_32F), FLAGSPARSE_STATUS_SUCCESS);
    ASSERT_EQ(flagsparseCreateDnVec(&y, 4, dy.get(), FLAGSPARSE_C_32F), FLAGSPARSE_STATUS_SUCCESS);
    for (int step = 0; step < 2; ++step) {
        if (step) {
            for (auto& v : values) v *= C(0.5f, 0.25f);
            for (auto& v : rhs) v += C(2, -1);
            ASSERT_EQ(to_device(dv.get(), values.data(), values.size() * sizeof(C)), FLAGSPARSE_STATUS_SUCCESS);
            ASSERT_EQ(to_device(dx.get(), rhs.data(), rhs.size() * sizeof(C)), FLAGSPARSE_STATUS_SUCCESS);
        }
        ASSERT_EQ(to_device(dy.get(), initial.data(), initial.size() * sizeof(C)), FLAGSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(flagsparseSpMV(handle.h, FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE, &alpha, a, x,
            &beta, y, FLAGSPARSE_C_32F, FLAGSPARSE_SPMV_ALG_DEFAULT, nullptr), FLAGSPARSE_STATUS_SUCCESS);
        auto got = dy.download<C>(4);
        for (int col = 0; col < 4; ++col) {
            C expected = beta * initial[col];
            for (int p = ptr[col]; p < ptr[col + 1]; ++p) expected += alpha * std::conj(values[p]) * rhs[idx[p]];
            EXPECT_NEAR(got[col].real(), expected.real(), 1e-5);
            EXPECT_NEAR(got[col].imag(), expected.imag(), 1e-5);
        }
    }
    flagsparseDestroyDnVec(y); flagsparseDestroyDnVec(x); flagsparseDestroySpMat(a);
}

}  // namespace
