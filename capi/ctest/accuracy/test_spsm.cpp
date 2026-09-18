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


// Accuracy tests for flagsparseSpSM: op(A) Y = alpha op(B), A triangular.
//
// Checked by substitution, like SpSV: build a triangular A and a known Y,
// compute B = A Y on the host in fp64, then ask the library for Y back.

#include <gtest/gtest.h>

#include <algorithm>
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


// B = A Y, row-major n x n_rhs, fp64 on the host.
std::vector<double> tri_apply(const TriMatrix& T, const std::vector<double>& Y,
                              int64_t n_rhs) {
    std::vector<double> B(static_cast<std::size_t>(T.n * n_rhs), 0.0);
    for (int64_t r = 0; r < T.n; ++r) {
        for (int64_t j = 0; j < n_rhs; ++j) {
            double acc = T.unit_diag ? Y[static_cast<std::size_t>(r * n_rhs + j)] : 0.0;
            for (int32_t p = T.indptr[static_cast<std::size_t>(r)];
                 p < T.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
                acc += T.values[static_cast<std::size_t>(p)] *
                       Y[static_cast<std::size_t>(
                           T.indices[static_cast<std::size_t>(p)] * n_rhs + j)];
            }
            B[static_cast<std::size_t>(r * n_rhs + j)] = acc;
        }
    }
    return B;
}

struct Layout {
    flagsparseOrder_t order = FLAGSPARSE_ORDER_ROW;
    int64_t pad = 0;
    int64_t ld(int64_t rows, int64_t cols) const {
        return (order == FLAGSPARSE_ORDER_ROW ? cols : rows) + pad;
    }
    std::size_t elems(int64_t rows, int64_t cols) const {
        return static_cast<std::size_t>(
            (order == FLAGSPARSE_ORDER_ROW ? rows : cols) * ld(rows, cols));
    }
    std::size_t at(int64_t i, int64_t j, int64_t rows, int64_t cols) const {
        const int64_t l = ld(rows, cols);
        return static_cast<std::size_t>(order == FLAGSPARSE_ORDER_ROW ? i * l + j
                                                                      : i + j * l);
    }
};

struct Case {
    int64_t n_rhs = 8;
    double alpha = 1.0;
    Layout lb{}, lc{};
    flagsparseOperation_t opB = FLAGSPARSE_OPERATION_NON_TRANSPOSE;
    flagsparseFormat_t format = FLAGSPARSE_FORMAT_CSR;
    bool skip_analysis = false;
    uint32_t seed = 7;
};

struct RunResult {
    flagsparseStatus_t status = FLAGSPARSE_STATUS_SUCCESS;
    double ratio = 0.0;
};

template <typename T>
RunResult run_spsm(flagsparseHandle_t handle, const TriMatrix& tri,
                   flagsparseDataType_t dtype, const Case& c) {
    RunResult out;
    const int64_t n = tri.n, k = c.n_rhs;
    const bool tb = (c.opB != FLAGSPARSE_OPERATION_NON_TRANSPOSE);

    std::mt19937 rng(c.seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> y_want(static_cast<std::size_t>(n * k));
    for (auto& v : y_want) v = dist(rng);
    std::vector<double> b64 = tri_apply(tri, y_want, k);
    for (auto& v : b64) v /= c.alpha;

    const int64_t bdr = tb ? k : n, bdc = tb ? n : k;
    std::vector<T> b_dev(c.lb.elems(bdr, bdc), T(0));
    std::vector<T> c_dev(c.lc.elems(n, k), T(0));
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < k; ++j) {
            const std::size_t slot = tb ? c.lb.at(j, i, bdr, bdc)
                                        : c.lb.at(i, j, bdr, bdc);
            b_dev[slot] = static_cast<T>(b64[static_cast<std::size_t>(i * k + j)]);
        }
    }

    const std::vector<T> values(tri.values.begin(), tri.values.end());
    std::vector<int32_t> coo_rows;
    for (int64_t r = 0; r < n; ++r) {
        for (int32_t p = tri.indptr[static_cast<std::size_t>(r)];
             p < tri.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
            coo_rows.push_back(static_cast<int32_t>(r));
        }
    }
    DeviceBuffer d_val = DeviceBuffer::from(values);
    DeviceBuffer d_col = DeviceBuffer::from(tri.indices);
    DeviceBuffer d_ptr = DeviceBuffer::from(tri.indptr);
    DeviceBuffer d_row = DeviceBuffer::from(coo_rows);
    DeviceBuffer d_b   = DeviceBuffer::from(b_dev);
    DeviceBuffer d_c   = DeviceBuffer::from(c_dev);

    flagsparseSpMatDescr_t matA = nullptr;
    flagsparseDnMatDescr_t matB = nullptr, matC = nullptr;
    flagsparseSpSMDescr_t descr = nullptr;
    const int64_t nnz = static_cast<int64_t>(tri.indices.size());
    out.status = (c.format == FLAGSPARSE_FORMAT_COO)
                     ? flagsparseCreateCoo(&matA, n, n, nnz, d_row.get(), d_col.get(),
                                           d_val.get(), FLAGSPARSE_INDEX_32I,
                                           FLAGSPARSE_INDEX_BASE_ZERO, dtype)
                     : flagsparseCreateCsr(&matA, n, n, nnz, d_ptr.get(), d_col.get(),
                                           d_val.get(), FLAGSPARSE_INDEX_32I,
                                           FLAGSPARSE_INDEX_32I,
                                           FLAGSPARSE_INDEX_BASE_ZERO, dtype);
    if (out.status != FLAGSPARSE_STATUS_SUCCESS) return out;
    const flagsparseFillMode_t fill =
        tri.lower ? FLAGSPARSE_FILL_MODE_LOWER : FLAGSPARSE_FILL_MODE_UPPER;
    const flagsparseDiagType_t diag =
        tri.unit_diag ? FLAGSPARSE_DIAG_TYPE_UNIT : FLAGSPARSE_DIAG_TYPE_NON_UNIT;
    flagsparseSpMatSetAttribute(matA, FLAGSPARSE_SPMAT_FILL_MODE, &fill, sizeof(fill));
    flagsparseSpMatSetAttribute(matA, FLAGSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof(diag));
    flagsparseCreateDnMat(&matB, bdr, bdc, c.lb.ld(bdr, bdc), d_b.get(), dtype,
                          c.lb.order);
    flagsparseCreateDnMat(&matC, n, k, c.lc.ld(n, k), d_c.get(), dtype, c.lc.order);
    flagsparseSpSM_createDescr(&descr);

    const T alpha_t = static_cast<T>(c.alpha);
    size_t buffer_size = 0;
    out.status = flagsparseSpSM_bufferSize(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                           c.opB, &alpha_t, matA, matB, matC, dtype,
                                           FLAGSPARSE_SPSM_ALG_DEFAULT, descr,
                                           &buffer_size);
    DeviceBuffer scratch(buffer_size);
    if (out.status == FLAGSPARSE_STATUS_SUCCESS && !c.skip_analysis) {
        out.status = flagsparseSpSM_analysis(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                             c.opB, &alpha_t, matA, matB, matC, dtype,
                                             FLAGSPARSE_SPSM_ALG_DEFAULT, descr,
                                             scratch.get());
    }
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.status = flagsparseSpSM_solve(handle, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                          c.opB, &alpha_t, matA, matB, matC, dtype,
                                          FLAGSPARSE_SPSM_ALG_DEFAULT, descr);
    }
    dev_sync();

    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        const std::vector<T> got = d_c.download<T>(c_dev.size());
        std::vector<double> actual(static_cast<std::size_t>(n * k));
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < k; ++j) {
                actual[static_cast<std::size_t>(i * k + j)] =
                    static_cast<double>(got[c.lc.at(i, j, n, k)]);
            }
        }
        out.ratio = max_error_ratio(actual, y_want, relaxed_tolerance(dtype));
    }

    flagsparseSpSM_destroyDescr(descr);
    flagsparseDestroyDnMat(matC);
    flagsparseDestroyDnMat(matB);
    flagsparseDestroySpMat(matA);
    return out;
}

void expect_solves(const RunResult& r, const char* label, flagsparseHandle_t handle) {
    const char* detail = "";
    flagsparseGetLastErrorString(handle, &detail);
    ASSERT_EQ(r.status, FLAGSPARSE_STATUS_SUCCESS)
        << label << ": " << status_name(r.status) << " -- " << detail;
    EXPECT_LE(r.ratio, 1.0) << label << " max_error_ratio=" << r.ratio;
    std::cout << "[   RATIO   ] " << label << " max_error_ratio=" << r.ratio << std::endl;
}

class SpSMAccuracy : public ::testing::Test {
  protected:
    Handle handle;
    void SetUp() override {
        if (handle.h == nullptr) GTEST_SKIP() << "no accelerator available";
        static bool announced = false;
        if (!announced) { print_backend_banner(); announced = true; }
    }
};

TEST_F(SpSMAccuracy, AllFourTriangleAndDiagonalCombinations) {
    for (bool lower : {true, false}) {
        for (bool unit : {true, false}) {
            const TriMatrix T = random_triangular(192, 0.04, lower, unit, 17);
            Case c; c.n_rhs = 8; c.alpha = 1.5;
            const std::string label = std::string(lower ? "lower" : "upper") +
                                      (unit ? "_unit" : "_nonunit");
            expect_solves(run_spsm<double>(handle.h, T, FLAGSPARSE_R_64F, c),
                          label.c_str(), handle.h);
        }
    }
}

// n_rhs straddles the RHS tile: the tile is the next power of two capped at
// 1024, so more than that means several tiles and therefore several done-flag
// planes, which is where an off-by-one in the flag base would show.
TEST_F(SpSMAccuracy, RhsWidthSweep) {
    const TriMatrix T = random_triangular(128, 0.05, true, false, 23);
    for (int64_t k : {1, 3, 32, 64, 100, 1024, 1500}) {
        Case c; c.n_rhs = k; c.seed = static_cast<uint32_t>(k);
        expect_solves(run_spsm<double>(handle.h, T, FLAGSPARSE_R_64F, c),
                      ("nrhs_" + std::to_string(k)).c_str(), handle.h);
    }
}

// The solver works in place on a packed row-major work array, so every layout
// and op(B) goes through the strided copy on the way in and out.
TEST_F(SpSMAccuracy, DenseLayoutAndTransposeSweep) {
    const TriMatrix T = random_triangular(96, 0.06, true, false, 31);
    const Layout row{FLAGSPARSE_ORDER_ROW, 0};
    const Layout col{FLAGSPARSE_ORDER_COL, 0};
    const Layout row_pad{FLAGSPARSE_ORDER_ROW, 5};
    const Layout col_pad{FLAGSPARSE_ORDER_COL, 3};
    struct Combo { Layout b, c; flagsparseOperation_t ob; const char* name; };
    for (const Combo& x : {
             Combo{row, row, FLAGSPARSE_OPERATION_NON_TRANSPOSE, "B_row_C_row"},
             Combo{col, row, FLAGSPARSE_OPERATION_NON_TRANSPOSE, "B_col_C_row"},
             Combo{row, col, FLAGSPARSE_OPERATION_NON_TRANSPOSE, "B_row_C_col"},
             Combo{col_pad, row_pad, FLAGSPARSE_OPERATION_NON_TRANSPOSE, "padded"},
             Combo{row, row, FLAGSPARSE_OPERATION_TRANSPOSE, "opB_T"},
             Combo{col, col, FLAGSPARSE_OPERATION_TRANSPOSE, "opB_T_col"}}) {
        Case c; c.n_rhs = 12; c.alpha = -0.75; c.lb = x.b; c.lc = x.c; c.opB = x.ob;
        expect_solves(run_spsm<double>(handle.h, T, FLAGSPARSE_R_64F, c), x.name,
                      handle.h);
    }
}

TEST_F(SpSMAccuracy, CooMatchesCsr) {
    for (bool lower : {true, false}) {
        const TriMatrix T = random_triangular(128, 0.05, lower, false, 29);
        Case c; c.n_rhs = 16; c.alpha = 1.25; c.format = FLAGSPARSE_FORMAT_COO;
        expect_solves(run_spsm<double>(handle.h, T, FLAGSPARSE_R_64F, c),
                      lower ? "coo_lower" : "coo_upper", handle.h);
    }
}

TEST_F(SpSMAccuracy, Float32) {
    const TriMatrix T = random_triangular(256, 0.04, true, false, 41);
    Case c; c.n_rhs = 24;
    expect_solves(run_spsm<float>(handle.h, T, FLAGSPARSE_R_32F, c), "fp32", handle.h);
}

TEST_F(SpSMAccuracy, RejectsMisuse) {
    const TriMatrix T = random_triangular(48, 0.08, true, false, 61);
    Case c; c.n_rhs = 8; c.skip_analysis = true;
    // Solving before analysis has no diagonal extracted and no buffer recorded.
    const RunResult r = run_spsm<double>(handle.h, T, FLAGSPARSE_R_64F, c);
    EXPECT_EQ(r.status, FLAGSPARSE_STATUS_INVALID_VALUE);
}

}  // namespace
