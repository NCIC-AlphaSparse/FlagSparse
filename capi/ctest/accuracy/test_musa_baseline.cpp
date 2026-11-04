// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// These exercise the vendor adapters, independently of FlagSparse's kernels.
#include <gtest/gtest.h>
#include <complex>
#include "baseline/baseline.hpp"
#include "common.hpp"

using namespace fstest;
namespace {
using C = std::complex<float>;
constexpr auto NT = FLAGSPARSE_OPERATION_NON_TRANSPOSE;
constexpr auto CT = FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
constexpr auto TR = FLAGSPARSE_OPERATION_TRANSPOSE;

class MusaBaseline : public ::testing::Test {
    flagsparseHandle_t handle = nullptr;
    void SetUp() override { ASSERT_EQ(flagsparseCreate(&handle), FLAGSPARSE_STATUS_SUCCESS); }
    void TearDown() override { if (handle) flagsparseDestroy(handle); }
};

template <typename T> constexpr flagsparseDataType_t dtype() { return FLAGSPARSE_R_32F; }
template <> constexpr flagsparseDataType_t dtype<C>() { return FLAGSPARSE_C_32F; }
template <typename T> T val(float real, float imag) { return T(real); }
template <> C val<C>(float real, float imag) { return {real, imag}; }

template <typename T> void check_axpby() {
    const std::vector<T> xv{val<T>(2, 1), val<T>(-1, 2)};
    const std::vector<T> initial{val<T>(1, 2), val<T>(3, -1), val<T>(-2, 1), val<T>(4, 3)};
    const T alpha = val<T>(2, -1), beta = val<T>(-1, 1);
    auto idx = DeviceBuffer::from(std::vector<int>{0, 2});
    auto x = DeviceBuffer::from(xv), y = DeviceBuffer::from(initial);
    baseline::Timing t;
    auto s = baseline::axpby(x.get(), idx.get(), y.get(), 2, 4, &alpha, &beta, dtype<T>(), 2, 3, &t);
    ASSERT_TRUE(s.ok) << s.reason;
    auto got = y.template download<T>(4);
    for (int i = 0; i < 4; ++i) {
        T ref = beta * initial[i];
        if (i == 0) ref += alpha * xv[0];
        if (i == 2) ref += alpha * xv[1];
        EXPECT_LT(std::abs(got[i] - ref), 1e-5) << i;
    }
}
TEST_F(MusaBaseline, AxpbyFloatBetaScalesAllY) { check_axpby<float>(); }
TEST_F(MusaBaseline, AxpbyComplexBetaScalesAllY) { check_axpby<C>(); }

template <typename T> void check_gemvi() {
    const std::vector<T> a{val<T>(1, 2), val<T>(2, -1), val<T>(3, 1),
                            val<T>(4, -2), val<T>(5, 1), val<T>(6, 3)};
    const std::vector<T> xv{val<T>(2, -1)}, initial{val<T>(1, 1), val<T>(-1, 2), T(999)};
    auto A = DeviceBuffer::from(a), x = DeviceBuffer::from(xv), y = DeviceBuffer::from(initial);
    auto idx = DeviceBuffer::from(std::vector<int>{2});
    const T alpha = val<T>(2, 1), beta = val<T>(-1, 1);
    baseline::Timing t;
    auto s = baseline::gemvi(NT, 2, 3, &alpha, A.get(), 2, 1, x.get(), idx.get(), &beta,
                              y.get(), dtype<T>(), 2, 3, &t);
    ASSERT_TRUE(s.ok) << s.reason;
    auto got = y.template download<T>(3);
    for (int i = 0; i < 2; ++i)
        EXPECT_LT(std::abs(got[i] - (alpha * a[4 + i] * xv[0] + beta * initial[i])), 1e-5);
    EXPECT_EQ(got[2], T(999));
    EXPECT_FALSE(baseline::gemvi(TR, 2, 3, &alpha, A.get(), 2, 1, x.get(), idx.get(),
                                 &beta, y.get(), dtype<T>(), 0, 1, &t).ok);
}
TEST_F(MusaBaseline, GemviFloatNonzeroBetaAndGuard) { check_gemvi<float>(); }
TEST_F(MusaBaseline, GemviComplexNonzeroBetaAndGuard) { check_gemvi<C>(); }

template <typename T> void check_geam() {
    // Different structures, overlapping (0,2), a cancellation retained in C.
    auto ap = DeviceBuffer::from(std::vector<int>{0, 2, 3});
    auto ai = DeviceBuffer::from(std::vector<int>{0, 2, 1});
    auto av = DeviceBuffer::from(std::vector<T>{val<T>(1, 1), val<T>(2, 1), val<T>(3, -1)});
    auto bp = DeviceBuffer::from(std::vector<int>{0, 2, 3});
    auto bi = DeviceBuffer::from(std::vector<int>{1, 2, 0});
    auto bv = DeviceBuffer::from(std::vector<T>{val<T>(4, 2), val<T>(2, 1), val<T>(5, -2)});
    baseline::DeviceCsr A{ap.get(), ai.get(), av.get(), 2, 3, 3, dtype<T>()};
    baseline::DeviceCsr B{bp.get(), bi.get(), bv.get(), 2, 3, 3, dtype<T>()};
    auto cp = DeviceBuffer::from(std::vector<int>(3, -1));
    auto ci = DeviceBuffer::from(std::vector<int>(6, 999));
    auto cv = DeviceBuffer::from(std::vector<T>(6, T(999)));
    const T alpha(1), beta(-1);
    baseline::Timing t;
    auto s = baseline::csrgeam2(A, B, &alpha, &beta, cp.get(), ci.get(), cv.get(), 5, 2, 3, &t);
    ASSERT_TRUE(s.ok) << s.reason;
    EXPECT_EQ(cp.download<int>(3), (std::vector<int>{0, 3, 5}));
    EXPECT_EQ(ci.download<int>(6), (std::vector<int>{0, 1, 2, 0, 1, 999}));
    const std::vector<T> ref{val<T>(1, 1), val<T>(-4, -2), T(0), val<T>(-5, 2), val<T>(3, -1), T(999)};
    EXPECT_EQ(cv.template download<T>(6), ref);
    // Capacity failure must happen before any column/value writes.
    std::vector<T> poison(6, T(999));
    to_device(cv.get(), poison.data(), poison.size() * sizeof(T));
    s = baseline::csrgeam2(A, B, &alpha, &beta, cp.get(), ci.get(), cv.get(), 4, 0, 1, &t);
    EXPECT_FALSE(s.ok);
    EXPECT_EQ(cv.template download<T>(6), poison);
}
TEST_F(MusaBaseline, GeamFloatUnionAndCapacity) { check_geam<float>(); }
TEST_F(MusaBaseline, GeamComplexUnionAndCapacity) { check_geam<C>(); }

TEST_F(MusaBaseline, CscConjugateRectangularGuardAndFreshValues) {
    // A is 3x2. A^H x has length 2, with a guard at index 2.
    auto ptr = DeviceBuffer::from(std::vector<int>{0, 2, 3});
    auto idx = DeviceBuffer::from(std::vector<int>{0, 2, 1});
    std::vector<C> values{{1, 2}, {3, -1}, {2, 4}};
    const std::vector<C> xv{{2, 1}, {1, -2}, {-1, 3}};
    auto av = DeviceBuffer::from(values), x = DeviceBuffer::from(xv);
    auto y = DeviceBuffer::from(std::vector<C>{C(0), C(0), C(999)});
    baseline::DeviceCsr A{ptr.get(), idx.get(), av.get(), 3, 2, 3, FLAGSPARSE_C_32F};
    A.format = baseline::DeviceCsr::Format::Csc;
    const C alpha(1), beta(0);
    for (int repeat = 0; repeat < 2; ++repeat) {
        baseline::Timing t;
        auto s = baseline::spmv_csr(A, x.get(), y.get(), &alpha, &beta, CT, 1, 2, &t);
        ASSERT_TRUE(s.ok) << s.reason;
        auto got = y.download<C>(3);
        EXPECT_EQ(got[0], std::conj(values[0]) * xv[0] + std::conj(values[1]) * xv[2]);
        EXPECT_EQ(got[1], std::conj(values[2]) * xv[1]);
        EXPECT_EQ(got[2], C(999));
        EXPECT_FALSE(t.method.empty());
        EXPECT_GT(t.prepare_ms, 0);
        values[0] += C(2, 3);
        to_device(av.get(), values.data(), values.size() * sizeof(C));
    }
}

TEST_F(MusaBaseline, CooTransposeRectangularAndFreshValues) {
    auto rows = DeviceBuffer::from(std::vector<int>{0, 0, 1});
    auto cols = DeviceBuffer::from(std::vector<int>{0, 2, 1});
    std::vector<float> values{1, 3, 2};
    auto av = DeviceBuffer::from(values);
    auto b = DeviceBuffer::from(std::vector<float>{1, 2, 3, 4}); // 2x2 row-major
    auto c = DeviceBuffer::from(std::vector<float>(7, 999));
    baseline::DeviceCsr A{nullptr, cols.get(), av.get(), 2, 3, 3, FLAGSPARSE_R_32F};
    A.coo_rows = rows.get();
    A.format = baseline::DeviceCsr::Format::Coo;
    const float alpha = 1, beta = 0;
    for (int repeat = 0; repeat < 2; ++repeat) {
        baseline::Timing t;
        auto s = baseline::spmm_csr(A, b.get(), 2, 2, c.get(), 2, &alpha, &beta, TR, NT,
                                    1, 2, &t, FLAGSPARSE_ORDER_ROW, FLAGSPARSE_ORDER_ROW);
        ASSERT_TRUE(s.ok) << s.reason;
        EXPECT_EQ(c.download<float>(7), (std::vector<float>{values[0], 2 * values[0],
                    3 * values[2], 4 * values[2], values[1], 2 * values[1], 999}));
        EXPECT_FALSE(t.method.empty());
        EXPECT_GT(t.prepare_ms, 0);
        values[0] = 7;
        to_device(av.get(), values.data(), values.size() * sizeof(float));
    }
}

TEST_F(MusaBaseline, UnsupportedTypesDoNotBecomeFloat32Baselines) {
    auto ptr = DeviceBuffer::from(std::vector<int>{0, 1});
    auto idx = DeviceBuffer::from(std::vector<int>{0});
    auto av = DeviceBuffer::from(std::vector<float>{2});
    auto x = DeviceBuffer::from(std::vector<C>{C(1, 2)});
    auto y = DeviceBuffer::from(std::vector<C>{C(999)});
    baseline::DeviceCsr A{ptr.get(), idx.get(), av.get(), 1, 1, 1, FLAGSPARSE_R_32F};
    const C alpha(1), beta(0);
    baseline::Timing t;
    EXPECT_FALSE(baseline::spmv_mixed(A, x.get(), FLAGSPARSE_C_32F, y.get(), FLAGSPARSE_C_32F,
        FLAGSPARSE_C_32F, &alpha, &beta, NT, 0, 1, &t).ok);
    A.dtype = FLAGSPARSE_R_16F;
    EXPECT_FALSE(baseline::spmv_csr(A, x.get(), y.get(), &alpha, &beta, NT, 0, 1, &t).ok);
    EXPECT_FALSE(baseline::gemvi(NT, 1, 1, &alpha, av.get(), 1, 1, x.get(), idx.get(),
                                 &beta, y.get(), FLAGSPARSE_R_16F, 0, 1, &t).ok);
    EXPECT_EQ(y.download<C>(1), (std::vector<C>{C(999)}));
}
}  // namespace
