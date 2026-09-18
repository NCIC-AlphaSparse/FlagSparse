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

// SpMM over the real-matrix corpus, against the vendor baseline.
//
// n (the dense width) is swept as well as the dtype, because SpMM's cost is not
// linear in it: a narrow B is bandwidth-bound on A, a wide one starts to reuse
// the A row it loaded and the ratio against a vendor kernel moves accordingly.
// Reporting one n would hide whichever end we happen to lose.

#include <gtest/gtest.h>

#include <vector>

#include "baseline/baseline.hpp"
#include "sweep.hpp"

using namespace fstest;

namespace {

BenchReport g_report("spmm");

// Column-major B and C, which is what ORDER_COL means to both libraries: ld is
// the row count of the operand.
constexpr int64_t kWidths[] = {8, 32, 128};

}  // namespace

TEST(SpmmBenchmark, CsrOverCorpus) {
    Handle handle;
    ASSERT_NE(handle.h, nullptr);
    report_corpus_failures(g_report, "csr");

    const Scalars sc;
    // Variant list from conf/operators.yaml via the generated registry.
    const auto declared = variants_of("spmm");

    for (const auto& entry : corpus()) {
        const CsrMatrix& A = entry.A;
        // Row-sorted COO, expanded from the row pointer, so both formats measure
        // the same matrix rather than two different orderings of it.
        const std::vector<int32_t> coo_rows = coo_row_indices_of(A);
        for (const int64_t n : kWidths) {
            // B column-major (cols x n); the oracle wants it row-major (k x n),
            // so the reference is built from the same numbers in the other order.
            const std::size_t bcount = static_cast<std::size_t>(A.cols) *
                                       static_cast<std::size_t>(n);
            const std::vector<double> b_col = dense_pattern(bcount);
            std::vector<double> b_row(bcount);
            for (int64_t c = 0; c < A.cols; ++c) {
                for (int64_t j = 0; j < n; ++j) {
                    b_row[static_cast<std::size_t>(c) * n + j] =
                        b_col[static_cast<std::size_t>(j) * A.cols + c];
                }
            }
            const std::vector<double> ref_row = spmm_reference(
                A, b_row, n, 1.0, 0.0,
                std::vector<double>(static_cast<std::size_t>(A.rows) * n, 0.0));
            // Our C is column-major; transpose the oracle to match so the two are
            // compared element for element rather than by luck of the layout.
            std::vector<double> ref(ref_row.size());
            for (int64_t r = 0; r < A.rows; ++r) {
                for (int64_t j = 0; j < n; ++j) {
                    ref[static_cast<std::size_t>(j) * A.rows + r] =
                        ref_row[static_cast<std::size_t>(r) * n + j];
                }
            }

            for (const registry::Variant* v : declared) {
                const bool is_coo = std::string(v->format) == "coo";
                const bool is_csr = std::string(v->format) == "csr";
                if (!is_csr && !is_coo) {
                    if (n == kWidths[0]) {   // one row per variant, not per width
                        const std::string why =
                            std::string("benchmark/test_spmm.cpp has no ") +
                            v->format + " operand builder yet";
                        report_unimplemented(g_report, *v, why.c_str());
                    }
                    continue;
                }
                const auto dt = v->dt;
                BenchRow row;
                row.name = std::string("spmm_") + v->format + "_" + v->dtype +
                           "_n" + std::to_string(n) + "_" + entry.name;
                row.tag("operator", v->op)
                   .tag("matrix", entry.name).tag("format", v->format)
                   .tag("dtype", v->dtype).tag("corpus", corpus_tag()).tag("reporting", v->reporting)
                   .num("rows", static_cast<double>(A.rows))
                   .num("cols", static_cast<double>(A.cols))
                   .num("nnz", static_cast<double>(A.nnz))
                   .num("n", static_cast<double>(n));
                trace("spmm", entry.name, v->dtype, A);

                DeviceBuffer indptr = DeviceBuffer::from(A.indptr);
                DeviceBuffer indices = DeviceBuffer::from(A.indices);
                DeviceBuffer rowind = DeviceBuffer::from(coo_rows);
                DeviceBuffer values = upload_as(A.values, dt);
                DeviceBuffer B = upload_as(b_col, dt);
                DeviceBuffer C(static_cast<std::size_t>(A.rows) *
                               static_cast<std::size_t>(n) * elem_bytes(dt));
                if (!indptr.get() || !indices.get() || !rowind.get() ||
                    !values.get() || !B.get() || !C.get()) {
                    g_report.skip(std::move(row), "skipped_memory",
                                  "device allocation failed for this matrix at n=" +
                                      std::to_string(n));
                    continue;
                }

                flagsparseSpMatDescr_t matA = nullptr;
                flagsparseDnMatDescr_t matB = nullptr, matC = nullptr;
                const flagsparseStatus_t cs =
                    is_coo
                        ? flagsparseCreateCoo(&matA, A.rows, A.cols, A.nnz,
                                              rowind.get(), indices.get(), values.get(),
                                              FLAGSPARSE_INDEX_32I,
                                              FLAGSPARSE_INDEX_BASE_ZERO, dt)
                        : flagsparseCreateCsr(&matA, A.rows, A.cols, A.nnz, indptr.get(),
                                              indices.get(), values.get(),
                                              FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                              FLAGSPARSE_INDEX_BASE_ZERO, dt);
                if (cs != FLAGSPARSE_STATUS_SUCCESS) {
                    g_report.skip(std::move(row), "failed",
                                  std::string("descriptor creation failed for ") +
                                      v->format);
                    continue;
                }
                flagsparseCreateDnMat(&matB, A.cols, n, A.cols, B.get(), dt,
                                      FLAGSPARSE_ORDER_COL);
                flagsparseCreateDnMat(&matC, A.rows, n, A.rows, C.get(), dt,
                                      FLAGSPARSE_ORDER_COL);

                std::size_t bufsz = 0;
                flagsparseSpMM_bufferSize(handle.h,
                                          FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                          FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                          sc.alpha(dt), matA, matB, sc.beta(dt), matC,
                                          dt, FLAGSPARSE_SPMM_ALG_DEFAULT, &bufsz);
                DeviceBuffer scratch(bufsz ? bufsz : 1);
                if (!scratch.get()) {
                    flagsparseDestroyDnMat(matB); flagsparseDestroyDnMat(matC);
                    flagsparseDestroySpMat(matA);
                    g_report.skip(std::move(row), "skipped_memory",
                                  "SpMM scratch allocation failed");
                    continue;
                }

                baseline::DeviceCsr bA{indptr.get(), indices.get(), values.get(),
                                       A.rows, A.cols, A.nnz, dt,
                                       is_coo ? rowind.get() : nullptr};
                g_report.measure_vs_baseline(
                    std::move(row),
                    [&]() {
                        return flagsparseSpMM(handle.h,
                                              FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                              FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                              sc.alpha(dt), matA, matB, sc.beta(dt),
                                              matC, dt, FLAGSPARSE_SPMM_ALG_DEFAULT,
                                              scratch.get());
                    },
                    [&](bool relaxed) { return ratio_against(C.get(), ref, dt, relaxed); },
                    [&](baseline::Timing* t) {
                        return baseline::spmm_csr(
                            bA, B.get(), n, A.cols, C.get(), A.rows, sc.alpha(dt),
                            sc.beta(dt), FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                            FLAGSPARSE_OPERATION_NON_TRANSPOSE, BenchReport::kWarmup,
                            BenchReport::kIters, t);
                    },
                    2.0 * static_cast<double>(A.nnz) * static_cast<double>(n));

                flagsparseDestroyDnMat(matB);
                flagsparseDestroyDnMat(matC);
                flagsparseDestroySpMat(matA);
            }
        }
    }
    EXPECT_GT(g_report.size(), 0u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    g_report.write();
    return rc;
}
