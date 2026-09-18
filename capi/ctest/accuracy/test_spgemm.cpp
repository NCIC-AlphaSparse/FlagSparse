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


// Accuracy tests for the SpGEMM descriptor flow: C = A * B, both sparse.
//
// The result's SIZE is discovered, not given, so these check the discovered
// structure as well as the numbers: the nnz, the row offsets, the column
// indices (which must come out sorted, as cuSPARSE guarantees) and the values,
// all against a dense fp64 product on the host.

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <vector>

#include "common.hpp"

using namespace fstest;

namespace {

struct Handle {
    flagsparseHandle_t h = nullptr;
    Handle() { flagsparseCreate(&h); }
    ~Handle() { if (h) flagsparseDestroy(h); }
};

// A CSR matrix on the device, owning its buffers.
struct DeviceCsr {
    DeviceBuffer ptr, col, val;
    flagsparseSpMatDescr_t descr = nullptr;
    ~DeviceCsr() { if (descr) flagsparseDestroySpMat(descr); }
};

template <typename T>
void upload_csr(const CsrMatrix& M, flagsparseDataType_t dtype, DeviceCsr* out) {
    const std::vector<T> values(M.values.begin(), M.values.end());
    out->ptr = DeviceBuffer::from(M.indptr);
    out->col = DeviceBuffer::from(M.indices);
    out->val = DeviceBuffer::from(values);
    ASSERT_EQ(flagsparseCreateCsr(&out->descr, M.rows, M.cols, M.nnz, out->ptr.get(),
                                  out->col.get(), out->val.get(), FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO,
                                  dtype),
              FLAGSPARSE_STATUS_SUCCESS);
}

// Dense fp64 product, as the reference for both structure and values.
std::vector<double> dense_product(const CsrMatrix& A, const CsrMatrix& B) {
    std::vector<double> C(static_cast<std::size_t>(A.rows * B.cols), 0.0);
    for (int64_t r = 0; r < A.rows; ++r) {
        for (int32_t p = A.indptr[static_cast<std::size_t>(r)];
             p < A.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
            const int64_t k = A.indices[static_cast<std::size_t>(p)];
            const double a = A.values[static_cast<std::size_t>(p)];
            for (int32_t q = B.indptr[static_cast<std::size_t>(k)];
                 q < B.indptr[static_cast<std::size_t>(k) + 1]; ++q) {
                C[static_cast<std::size_t>(r * B.cols + B.indices[static_cast<std::size_t>(q)])] +=
                    a * B.values[static_cast<std::size_t>(q)];
            }
        }
    }
    return C;
}

class SpGEMMAccuracy : public ::testing::Test {
  protected:
    Handle handle;
    void SetUp() override {
        if (handle.h == nullptr) GTEST_SKIP() << "no accelerator available";
        static bool announced = false;
        if (!announced) { print_backend_banner(); announced = true; }
    }

    // The full five-call flow, ending with the structure and values checked
    // against the dense reference.
    template <typename T>
    void run_and_check(const CsrMatrix& A, const CsrMatrix& B,
                       flagsparseDataType_t dtype, const char* label) {
        DeviceCsr da, db;
        upload_csr<T>(A, dtype, &da);
        upload_csr<T>(B, dtype, &db);

        flagsparseSpMatDescr_t matC = nullptr;
        // C starts empty: its size is what the flow discovers.
        DeviceBuffer c_ptr_probe(static_cast<size_t>(A.rows + 1) * sizeof(int32_t));
        ASSERT_EQ(flagsparseCreateCsr(&matC, A.rows, B.cols, 0, c_ptr_probe.get(),
                                      nullptr, nullptr, FLAGSPARSE_INDEX_32I,
                                      FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO,
                                      dtype),
                  FLAGSPARSE_STATUS_SUCCESS);

        flagsparseSpGEMMDescr_t descr = nullptr;
        ASSERT_EQ(flagsparseSpGEMM_createDescr(&descr), FLAGSPARSE_STATUS_SUCCESS);
        const T one = static_cast<T>(1), zero = static_cast<T>(0);
        const auto NT = FLAGSPARSE_OPERATION_NON_TRANSPOSE;
        const auto ALG = FLAGSPARSE_SPGEMM_DEFAULT;

        size_t size1 = 0;
        ASSERT_EQ(flagsparseSpGEMM_workEstimation(handle.h, NT, NT, &one, da.descr,
                                                  db.descr, &zero, matC, dtype, ALG,
                                                  descr, &size1, nullptr),
                  FLAGSPARSE_STATUS_SUCCESS) << label;
        DeviceBuffer buf1(size1);
        ASSERT_EQ(flagsparseSpGEMM_workEstimation(handle.h, NT, NT, &one, da.descr,
                                                  db.descr, &zero, matC, dtype, ALG,
                                                  descr, &size1, buf1.get()),
                  FLAGSPARSE_STATUS_SUCCESS) << label;

        size_t size2 = 0;
        ASSERT_EQ(flagsparseSpGEMM_compute(handle.h, NT, NT, &one, da.descr, db.descr,
                                           &zero, matC, dtype, ALG, descr, &size2,
                                           nullptr),
                  FLAGSPARSE_STATUS_SUCCESS) << label;
        DeviceBuffer buf2(size2);
        const char* detail = "";
        const flagsparseStatus_t st =
            flagsparseSpGEMM_compute(handle.h, NT, NT, &one, da.descr, db.descr, &zero,
                                     matC, dtype, ALG, descr, &size2, buf2.get());
        flagsparseGetLastErrorString(handle.h, &detail);
        ASSERT_EQ(st, FLAGSPARSE_STATUS_SUCCESS) << label << ": " << detail;

        int64_t c_rows = 0, c_cols = 0, c_nnz = 0;
        ASSERT_EQ(flagsparseSpMatGetSize(matC, &c_rows, &c_cols, &c_nnz),
                  FLAGSPARSE_STATUS_SUCCESS);
        EXPECT_EQ(c_rows, A.rows);
        EXPECT_EQ(c_cols, B.cols);

        const std::vector<double> ref = dense_product(A, B);
        int64_t want_nnz = 0;
        for (double v : ref) if (v != 0.0) ++want_nnz;
        EXPECT_EQ(c_nnz, want_nnz) << label << ": discovered nnz";

        DeviceBuffer c_ptr(static_cast<size_t>(A.rows + 1) * sizeof(int32_t));
        DeviceBuffer c_col(static_cast<size_t>(std::max<int64_t>(c_nnz, 1)) * sizeof(int32_t));
        DeviceBuffer c_val(static_cast<size_t>(std::max<int64_t>(c_nnz, 1)) * sizeof(T));
        ASSERT_EQ(flagsparseCsrSetPointers(matC, c_ptr.get(), c_col.get(), c_val.get()),
                  FLAGSPARSE_STATUS_SUCCESS);
        ASSERT_EQ(flagsparseSpGEMM_copy(handle.h, NT, NT, &one, da.descr, db.descr,
                                        &zero, matC, dtype, ALG, descr),
                  FLAGSPARSE_STATUS_SUCCESS) << label;
        dev_sync();

        const std::vector<int32_t> got_ptr = c_ptr.download<int32_t>(
            static_cast<std::size_t>(A.rows + 1));
        const std::vector<int32_t> got_col = c_col.download<int32_t>(
            static_cast<std::size_t>(c_nnz));
        const std::vector<T> got_val = c_val.download<T>(static_cast<std::size_t>(c_nnz));

        std::vector<double> actual, expect;
        for (int64_t r = 0; r < A.rows; ++r) {
            // Columns must come out sorted ascending: cuSPARSE guarantees it and
            // every downstream operator here assumes it.
            for (int32_t p = got_ptr[static_cast<std::size_t>(r)] + 1;
                 p < got_ptr[static_cast<std::size_t>(r) + 1]; ++p) {
                ASSERT_LT(got_col[static_cast<std::size_t>(p - 1)],
                          got_col[static_cast<std::size_t>(p)])
                    << label << ": row " << r << " columns not sorted";
            }
            // Every stored entry must match, and every nonzero of the reference
            // must be stored -- checked by walking the two in step.
            int32_t p = got_ptr[static_cast<std::size_t>(r)];
            for (int64_t c = 0; c < B.cols; ++c) {
                const double want = ref[static_cast<std::size_t>(r * B.cols + c)];
                if (want == 0.0) continue;
                ASSERT_LT(p, got_ptr[static_cast<std::size_t>(r) + 1])
                    << label << ": missing entry at (" << r << "," << c << ")";
                ASSERT_EQ(got_col[static_cast<std::size_t>(p)], c)
                    << label << ": column mismatch in row " << r;
                actual.push_back(static_cast<double>(got_val[static_cast<std::size_t>(p)]));
                expect.push_back(want);
                ++p;
            }
            ASSERT_EQ(p, got_ptr[static_cast<std::size_t>(r) + 1])
                << label << ": extra entries in row " << r;
        }
        const double ratio = max_error_ratio(actual, expect, relaxed_tolerance(dtype));
        EXPECT_LE(ratio, 1.0) << label << " max_error_ratio=" << ratio;
        std::cout << "[   RATIO   ] " << label << " nnz=" << c_nnz
                  << " max_error_ratio=" << ratio << std::endl;

        flagsparseSpGEMM_destroyDescr(descr);
        flagsparseDestroySpMat(matC);
    }
};

TEST_F(SpGEMMAccuracy, SquareAndRectangular) {
    for (auto shape : {std::array<int64_t, 3>{64, 48, 80},
                       {128, 128, 128},
                       {33, 17, 41}}) {
        const CsrMatrix A = random_csr(shape[0], shape[1], 0.06, 11);
        const CsrMatrix B = random_csr(shape[1], shape[2], 0.06, 13);
        run_and_check<double>(A, B, FLAGSPARSE_R_64F, "fp64");
    }
}

TEST_F(SpGEMMAccuracy, Float32) {
    const CsrMatrix A = random_csr(96, 64, 0.05, 21);
    const CsrMatrix B = random_csr(64, 96, 0.05, 23);
    run_and_check<float>(A, B, FLAGSPARSE_R_32F, "fp32");
}

// Rows of A with no entries produce empty rows of C; the discovered indptr has
// to stay monotone through them.
TEST_F(SpGEMMAccuracy, EmptyRowsAndEmptyResult) {
    const CsrMatrix sparse_a = random_csr(128, 64, 0.008, 31);
    const CsrMatrix B = random_csr(64, 64, 0.05, 33);
    run_and_check<double>(sparse_a, B, FLAGSPARSE_R_64F, "sparse_rows");

    // Must share B's row count to be a legal product; an all-empty A gives an
    // all-empty C, which is the degenerate case of the discovered indptr.
    const CsrMatrix empty = random_csr(32, 64, 0.0, 37);
    ASSERT_EQ(empty.nnz, 0);
    run_and_check<double>(empty, B, FLAGSPARSE_R_64F, "empty_a");
}

// Denser rows push the hash table into the larger buckets.
TEST_F(SpGEMMAccuracy, WiderRowsUseLargerHashBuckets) {
    const CsrMatrix A = random_csr(64, 64, 0.30, 41);
    const CsrMatrix B = random_csr(64, 64, 0.30, 43);
    run_and_check<double>(A, B, FLAGSPARSE_R_64F, "dense_rows");
}

TEST_F(SpGEMMAccuracy, RejectsMisuseAndUnsupported) {
    const CsrMatrix A = random_csr(32, 32, 0.1, 51);
    const CsrMatrix B = random_csr(32, 32, 0.1, 53);
    DeviceCsr da, db;
    upload_csr<double>(A, FLAGSPARSE_R_64F, &da);
    upload_csr<double>(B, FLAGSPARSE_R_64F, &db);
    DeviceBuffer c_ptr(static_cast<size_t>(A.rows + 1) * sizeof(int32_t));
    flagsparseSpMatDescr_t matC = nullptr;
    flagsparseCreateCsr(&matC, A.rows, B.cols, 0, c_ptr.get(), nullptr, nullptr,
                        FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                        FLAGSPARSE_INDEX_BASE_ZERO, FLAGSPARSE_R_64F);
    flagsparseSpGEMMDescr_t descr = nullptr;
    flagsparseSpGEMM_createDescr(&descr);
    const double one = 1.0, zero = 0.0, two = 2.0;
    const auto NT = FLAGSPARSE_OPERATION_NON_TRANSPOSE;
    const auto ALG = FLAGSPARSE_SPGEMM_DEFAULT;
    size_t s1 = 0, s2 = 0;

    // alpha != 1 / beta != 0: cuSPARSE does not define them either.
    EXPECT_EQ(flagsparseSpGEMM_workEstimation(handle.h, NT, NT, &two, da.descr, db.descr,
                                              &zero, matC, FLAGSPARSE_R_64F, ALG, descr,
                                              &s1, nullptr),
              FLAGSPARSE_STATUS_NOT_SUPPORTED);
    EXPECT_EQ(flagsparseSpGEMM_workEstimation(handle.h, NT, NT, &one, da.descr, db.descr,
                                              &one, matC, FLAGSPARSE_R_64F, ALG, descr,
                                              &s1, nullptr),
              FLAGSPARSE_STATUS_NOT_SUPPORTED);

    // compute before a completed workEstimation, and copy before compute.
    EXPECT_EQ(flagsparseSpGEMM_compute(handle.h, NT, NT, &one, da.descr, db.descr, &zero,
                                       matC, FLAGSPARSE_R_64F, ALG, descr, &s2, nullptr),
              FLAGSPARSE_STATUS_INVALID_VALUE);
    EXPECT_EQ(flagsparseSpGEMM_copy(handle.h, NT, NT, &one, da.descr, db.descr, &zero,
                                    matC, FLAGSPARSE_R_64F, ALG, descr),
              FLAGSPARSE_STATUS_INVALID_VALUE);

    // op(A) = A^T would need A transposed first.
    EXPECT_EQ(flagsparseSpGEMM_workEstimation(handle.h, FLAGSPARSE_OPERATION_TRANSPOSE,
                                              NT, &one, da.descr, db.descr, &zero, matC,
                                              FLAGSPARSE_R_64F, ALG, descr, &s1, nullptr),
              FLAGSPARSE_STATUS_NOT_SUPPORTED);

    flagsparseSpGEMM_destroyDescr(descr);
    flagsparseDestroySpMat(matC);
}

}  // namespace
