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


// Accuracy tests for flagsparseSDDMM, against an fp64 host reference.
//
// The axes that actually change the generated code are swept explicitly: k
// either side of the BLOCK_K=32 threshold, mean row length either side of the
// 16 that switches BLOCK_P between 64 and 512, both dense layouts, and both
// transpose flags.

#include <gtest/gtest.h>

#include <vector>

#include "common.hpp"

using namespace fstest;

namespace {

struct Handle {
    flagsparseHandle_t h = nullptr;
    Handle() { flagsparseCreate(&h); }
    ~Handle() { if (h) flagsparseDestroy(h); }
};

struct RunResult {
    flagsparseStatus_t status = FLAGSPARSE_STATUS_SUCCESS;
    double strict_ratio = 0.0;
    double relaxed_ratio = 0.0;
    bool relaxed_used = false;
};

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
    int64_t k = 64;
    double alpha = 1.0;
    double beta = 0.0;
    Layout la{}, lb{};
    flagsparseOperation_t opA = FLAGSPARSE_OPERATION_NON_TRANSPOSE;
    flagsparseOperation_t opB = FLAGSPARSE_OPERATION_NON_TRANSPOSE;
    bool skip_preprocess = false;
    uint32_t seed = 7;
};

// Fill a descriptor-shaped buffer so that logical element (i, j) of an r x c
// matrix lands where the descriptor's order/ld put it, transposing if the
// descriptor holds op(M)'s transpose.
template <typename T>
void place(std::vector<T>& buf, const std::vector<double>& logical, int64_t r,
           int64_t c, const Layout& layout, bool transposed) {
    const int64_t dr = transposed ? c : r;
    const int64_t dc = transposed ? r : c;
    for (int64_t i = 0; i < r; ++i) {
        for (int64_t j = 0; j < c; ++j) {
            const std::size_t slot = transposed ? layout.at(j, i, dr, dc)
                                                : layout.at(i, j, dr, dc);
            buf[slot] = static_cast<T>(logical[static_cast<std::size_t>(i * c + j)]);
        }
    }
}

template <typename T>
RunResult run_sddmm(flagsparseHandle_t handle, const CsrMatrix& C,
                    flagsparseDataType_t dtype, const Case& c) {
    RunResult out;
    const int64_t m = C.rows, n = C.cols, k = c.k;
    const bool ta = (c.opA != FLAGSPARSE_OPERATION_NON_TRANSPOSE);
    const bool tb = (c.opB != FLAGSPARSE_OPERATION_NON_TRANSPOSE);

    std::mt19937 rng(c.seed);
    std::normal_distribution<double> dist(0.0, 1.0);
    std::vector<double> a64(static_cast<std::size_t>(m * k));
    std::vector<double> b64(static_cast<std::size_t>(k * n));
    for (auto& v : a64) v = dist(rng);
    for (auto& v : b64) v = dist(rng);

    const int64_t adr = ta ? k : m, adc = ta ? m : k;
    const int64_t bdr = tb ? n : k, bdc = tb ? k : n;
    std::vector<T> a_dev(c.la.elems(adr, adc), T(0));
    std::vector<T> b_dev(c.lb.elems(bdr, bdc), T(0));
    place(a_dev, a64, m, k, c.la, ta);
    place(b_dev, b64, k, n, c.lb, tb);

    const std::vector<T> c_in(C.values.begin(), C.values.end());
    DeviceBuffer d_a   = DeviceBuffer::from(a_dev);
    DeviceBuffer d_b   = DeviceBuffer::from(b_dev);
    DeviceBuffer d_val = DeviceBuffer::from(c_in);
    DeviceBuffer d_col = DeviceBuffer::from(C.indices);
    DeviceBuffer d_ptr = DeviceBuffer::from(C.indptr);

    flagsparseDnMatDescr_t matA = nullptr, matB = nullptr;
    flagsparseSpMatDescr_t matC = nullptr;
    flagsparseCreateDnMat(&matA, adr, adc, c.la.ld(adr, adc), d_a.get(), dtype, c.la.order);
    flagsparseCreateDnMat(&matB, bdr, bdc, c.lb.ld(bdr, bdc), d_b.get(), dtype, c.lb.order);
    out.status = flagsparseCreateCsr(&matC, m, n, C.nnz, d_ptr.get(), d_col.get(),
                                     d_val.get(), FLAGSPARSE_INDEX_32I,
                                     FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO,
                                     dtype);
    if (out.status != FLAGSPARSE_STATUS_SUCCESS) return out;

    const T alpha_t = static_cast<T>(c.alpha);
    const T beta_t  = static_cast<T>(c.beta);
    size_t buffer_size = 0;
    out.status = flagsparseSDDMM_bufferSize(handle, c.opA, c.opB, &alpha_t, matA, matB,
                                            &beta_t, matC, dtype,
                                            FLAGSPARSE_SDDMM_ALG_DEFAULT, &buffer_size);
    DeviceBuffer scratch(buffer_size);
    if (out.status == FLAGSPARSE_STATUS_SUCCESS && !c.skip_preprocess) {
        out.status = flagsparseSDDMM_preprocess(handle, c.opA, c.opB, &alpha_t, matA,
                                                matB, &beta_t, matC, dtype,
                                                FLAGSPARSE_SDDMM_ALG_DEFAULT,
                                                scratch.get());
    }
    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        out.status = flagsparseSDDMM(handle, c.opA, c.opB, &alpha_t, matA, matB, &beta_t,
                                     matC, dtype, FLAGSPARSE_SDDMM_ALG_DEFAULT,
                                     scratch.get());
    }
    dev_sync();

    if (out.status == FLAGSPARSE_STATUS_SUCCESS) {
        const std::vector<T> got = d_val.download<T>(static_cast<std::size_t>(C.nnz));
        const std::vector<double> actual(got.begin(), got.end());
        std::vector<double> ref(static_cast<std::size_t>(C.nnz), 0.0);
        for (int64_t r = 0; r < m; ++r) {
            for (int32_t p = C.indptr[static_cast<std::size_t>(r)];
                 p < C.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
                const int64_t col = C.indices[static_cast<std::size_t>(p)];
                double acc = 0.0;
                for (int64_t kk = 0; kk < k; ++kk) {
                    acc += a64[static_cast<std::size_t>(r * k + kk)] *
                           b64[static_cast<std::size_t>(kk * n + col)];
                }
                ref[static_cast<std::size_t>(p)] =
                    c.alpha * acc +
                    (c.beta == 0.0 ? 0.0
                                   : c.beta * C.values[static_cast<std::size_t>(p)]);
            }
        }
        out.strict_ratio = max_error_ratio(actual, ref, default_tolerance(dtype));
        if (out.strict_ratio > 1.0) {
            out.relaxed_ratio = max_error_ratio(actual, ref, relaxed_tolerance(dtype));
            out.relaxed_used = true;
        }
    }

    flagsparseDestroySpMat(matC);
    flagsparseDestroyDnMat(matB);
    flagsparseDestroyDnMat(matA);
    return out;
}

void expect_close(const RunResult& r, const char* label, flagsparseHandle_t handle) {
    const char* detail = "";
    flagsparseGetLastErrorString(handle, &detail);
    ASSERT_EQ(r.status, FLAGSPARSE_STATUS_SUCCESS)
        << label << ": " << status_name(r.status) << " -- " << detail;
    if (!r.relaxed_used) {
        EXPECT_LE(r.strict_ratio, 1.0) << label << " max_error_ratio=" << r.strict_ratio;
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

class SDDMMAccuracy : public ::testing::Test {
  protected:
    Handle handle;
    void SetUp() override {
        if (handle.h == nullptr) GTEST_SKIP() << "no accelerator available";
        static bool announced = false;
        if (!announced) { print_backend_banner(); announced = true; }
    }
};

// k straddles the BLOCK_K = 32 threshold; below it the tile is next_pow2(k).
TEST_F(SDDMMAccuracy, KSweep) {
    const CsrMatrix C = random_csr(96, 128, 0.05, 1234);
    for (int64_t k : {1, 7, 16, 32, 33, 64, 200}) {
        Case c; c.k = k; c.seed = static_cast<uint32_t>(k);
        expect_close(run_sddmm<float>(handle.h, C, FLAGSPARSE_R_32F, c),
                     ("fp32_k" + std::to_string(k)).c_str(), handle.h);
    }
}

TEST_F(SDDMMAccuracy, Float64AndAlphaBeta) {
    const CsrMatrix C = random_csr(128, 96, 0.06, 99);
    Case c; c.k = 48;
    expect_close(run_sddmm<double>(handle.h, C, FLAGSPARSE_R_64F, c), "fp64", handle.h);
    c.alpha = -2.5; c.beta = 0.75;
    expect_close(run_sddmm<double>(handle.h, C, FLAGSPARSE_R_64F, c), "fp64_alpha_beta",
                 handle.h);
}

// Mean row length picks BLOCK_P: >= 16 takes the wide 512 config (non-fp64),
// below it the narrow 64. Both must be right, not just the default one.
TEST_F(SDDMMAccuracy, BlockPConfigsAgree) {
    const CsrMatrix sparse_rows = random_csr(256, 256, 0.02, 21);   // mean ~5
    const CsrMatrix dense_rows  = random_csr(256, 256, 0.20, 21);   // mean ~51
    Case c; c.k = 64; c.alpha = 1.5; c.beta = -0.5;
    expect_close(run_sddmm<float>(handle.h, sparse_rows, FLAGSPARSE_R_32F, c),
                 "block_p_64", handle.h);
    expect_close(run_sddmm<float>(handle.h, dense_rows, FLAGSPARSE_R_32F, c),
                 "block_p_512", handle.h);
    // fp64 always takes the narrow config even with long rows.
    expect_close(run_sddmm<double>(handle.h, dense_rows, FLAGSPARSE_R_64F, c),
                 "block_p_64_fp64", handle.h);
}

// Both dense layouts, a padded ld, and both transpose flags -- all of which are
// stride choices, nothing materialised.
TEST_F(SDDMMAccuracy, DenseLayoutAndTransposeSweep) {
    const CsrMatrix C = random_csr(70, 90, 0.07, 33);
    const Layout row{FLAGSPARSE_ORDER_ROW, 0};
    const Layout col{FLAGSPARSE_ORDER_COL, 0};
    const Layout row_pad{FLAGSPARSE_ORDER_ROW, 5};
    const Layout col_pad{FLAGSPARSE_ORDER_COL, 3};
    struct Combo { Layout a, b; flagsparseOperation_t oa, ob; const char* name; };
    for (const Combo& x : {
             Combo{row, row, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                   FLAGSPARSE_OPERATION_NON_TRANSPOSE, "A_row_B_row"},
             Combo{col, row, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                   FLAGSPARSE_OPERATION_NON_TRANSPOSE, "A_col_B_row"},
             Combo{row, col, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                   FLAGSPARSE_OPERATION_NON_TRANSPOSE, "A_row_B_col"},
             Combo{row_pad, col_pad, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                   FLAGSPARSE_OPERATION_NON_TRANSPOSE, "padded_lds"},
             Combo{row, row, FLAGSPARSE_OPERATION_TRANSPOSE,
                   FLAGSPARSE_OPERATION_NON_TRANSPOSE, "opA_T"},
             Combo{row, row, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                   FLAGSPARSE_OPERATION_TRANSPOSE, "opB_T"},
             Combo{col, col, FLAGSPARSE_OPERATION_TRANSPOSE,
                   FLAGSPARSE_OPERATION_TRANSPOSE, "both_T_col"}}) {
        Case c; c.k = 40; c.la = x.a; c.lb = x.b; c.opA = x.oa; c.opB = x.ob;
        c.alpha = 1.5; c.beta = -0.5;
        expect_close(run_sddmm<double>(handle.h, C, FLAGSPARSE_R_64F, c), x.name,
                     handle.h);
    }
}

// Rows with no nonzeros contribute nothing, but the binary search that assigns
// a row to each nonzero has to skip them correctly -- an off-by-one there gives
// every following nonzero the wrong row.
TEST_F(SDDMMAccuracy, EmptyRowsInPattern) {
    const CsrMatrix C = random_csr(300, 64, 0.01, 91);
    Case c; c.k = 32; c.alpha = 2.0; c.beta = -1.5;
    expect_close(run_sddmm<double>(handle.h, C, FLAGSPARSE_R_64F, c), "empty_rows",
                 handle.h);
}

TEST_F(SDDMMAccuracy, WorksWithoutPreprocess) {
    const CsrMatrix C = random_csr(80, 96, 0.06, 55);
    Case c; c.k = 32; c.skip_preprocess = true; c.alpha = 1.25;
    expect_close(run_sddmm<double>(handle.h, C, FLAGSPARSE_R_64F, c), "no_preprocess",
                 handle.h);
}

TEST_F(SDDMMAccuracy, RejectsUnsupportedAndMalformed) {
    const CsrMatrix C = random_csr(32, 48, 0.1, 2);
    const std::vector<float> vals(C.values.begin(), C.values.end());
    DeviceBuffer d_val = DeviceBuffer::from(vals);
    DeviceBuffer d_col = DeviceBuffer::from(C.indices);
    DeviceBuffer d_ptr = DeviceBuffer::from(C.indptr);
    DeviceBuffer d_a   = DeviceBuffer::from(std::vector<float>(32 * 16, 1.0f));
    DeviceBuffer d_b   = DeviceBuffer::from(std::vector<float>(16 * 48, 1.0f));

    flagsparseDnMatDescr_t matA = nullptr, matB = nullptr;
    flagsparseSpMatDescr_t matC = nullptr;
    flagsparseCreateDnMat(&matA, 32, 16, 16, d_a.get(), FLAGSPARSE_R_32F,
                          FLAGSPARSE_ORDER_ROW);
    flagsparseCreateDnMat(&matB, 16, 48, 48, d_b.get(), FLAGSPARSE_R_32F,
                          FLAGSPARSE_ORDER_ROW);
    ASSERT_EQ(flagsparseCreateCsr(&matC, C.rows, C.cols, C.nnz, d_ptr.get(), d_col.get(),
                                  d_val.get(), FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_32F),
              FLAGSPARSE_STATUS_SUCCESS);

    const float one = 1.0f, zero = 0.0f;
    size_t buf = 0;
    ASSERT_EQ(flagsparseSDDMM_bufferSize(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                         FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA,
                                         matB, &zero, matC, FLAGSPARSE_R_32F,
                                         FLAGSPARSE_SDDMM_ALG_DEFAULT, &buf),
              FLAGSPARSE_STATUS_SUCCESS);
    EXPECT_EQ(buf, static_cast<size_t>(C.nnz) * sizeof(int32_t));

    // Omitting the buffer is a caller error, not a silent wrong answer.
    EXPECT_EQ(flagsparseSDDMM(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                              FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, matB, &zero,
                              matC, FLAGSPARSE_R_32F, FLAGSPARSE_SDDMM_ALG_DEFAULT,
                              nullptr),
              FLAGSPARSE_STATUS_INVALID_VALUE);

    // A k that does not match between the two dense operands.
    flagsparseDnMatDescr_t badB = nullptr;
    flagsparseCreateDnMat(&badB, 8, 48, 48, d_b.get(), FLAGSPARSE_R_32F,
                          FLAGSPARSE_ORDER_ROW);
    DeviceBuffer scratch(buf);
    EXPECT_EQ(flagsparseSDDMM(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                              FLAGSPARSE_OPERATION_NON_TRANSPOSE, &one, matA, badB, &zero,
                              matC, FLAGSPARSE_R_32F, FLAGSPARSE_SDDMM_ALG_DEFAULT,
                              scratch.get()),
              FLAGSPARSE_STATUS_INVALID_VALUE);
    flagsparseDestroyDnMat(badB);

    // Complex has no kernel in the operator package; say so rather than invent one.
    flagsparseSpMatDescr_t cplxC = nullptr;
    flagsparseDnMatDescr_t cplxA = nullptr, cplxB = nullptr;
    flagsparseCreateDnMat(&cplxA, 32, 16, 16, d_a.get(), FLAGSPARSE_C_32F,
                          FLAGSPARSE_ORDER_ROW);
    flagsparseCreateDnMat(&cplxB, 16, 48, 48, d_b.get(), FLAGSPARSE_C_32F,
                          FLAGSPARSE_ORDER_ROW);
    flagsparseCreateCsr(&cplxC, C.rows, C.cols, C.nnz, d_ptr.get(), d_col.get(),
                        d_val.get(), FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                        FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_C_32F);
    const float cone[2] = {1.0f, 0.0f}, czero[2] = {0.0f, 0.0f};
    EXPECT_EQ(flagsparseSDDMM(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                              FLAGSPARSE_OPERATION_NON_TRANSPOSE, cone, cplxA, cplxB,
                              czero, cplxC, FLAGSPARSE_C_32F,
                              FLAGSPARSE_SDDMM_ALG_DEFAULT, scratch.get()),
              FLAGSPARSE_STATUS_NOT_SUPPORTED);
    flagsparseDestroySpMat(cplxC);
    flagsparseDestroyDnMat(cplxB);
    flagsparseDestroyDnMat(cplxA);

    flagsparseDestroySpMat(matC);
    flagsparseDestroyDnMat(matB);
    flagsparseDestroyDnMat(matA);
}

}  // namespace
