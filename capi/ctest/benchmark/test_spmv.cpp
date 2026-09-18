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

// SpMV over the real-matrix corpus, against the vendor baseline.
//
// One row per (matrix, dtype): our median, the vendor's median, the error ratio
// of our answer against a CPU fp64 oracle, and -- only when that answer passed --
// the speedup. See ctest/corpus.hpp for where the matrices come from and
// ctest/baseline/baseline.hpp for what the denominator is.

#include <gtest/gtest.h>

#include <vector>

#include "baseline/baseline.hpp"
#include "sweep.hpp"

using namespace fstest;

namespace {

BenchReport g_report("spmv");

}  // namespace

TEST(SpmvBenchmark, CsrOverCorpus) {
    Handle handle;
    ASSERT_NE(handle.h, nullptr);
    report_corpus_failures(g_report, "csr");

    const Scalars sc;
    // The variant list comes from conf/operators.yaml via the generated
    // registry, not from a table in this file.
    const auto declared = variants_of("spmv");

    for (const auto& entry : corpus()) {
        const CsrMatrix& A = entry.A;
        const std::vector<double> x_host = dense_pattern(static_cast<std::size_t>(A.cols));
        const std::vector<double> ref =
            spmv_reference(A, x_host, 1.0, 0.0,
                           std::vector<double>(static_cast<std::size_t>(A.rows), 0.0));

        // random_csr and the corpus both emit rows in order, so expanding the
        // row pointer gives a row-sorted COO -- which is what the COO kernels
        // assume and what makes the two formats comparable on one matrix.
        const std::vector<int32_t> coo_rows = coo_row_indices_of(A);

        for (const registry::Variant* v : declared) {
            const bool is_coo = std::string(v->format) == "coo";
            const bool is_csr = std::string(v->format) == "csr";
            if (!is_csr && !is_coo) {
                // CSC and BSR are declared implemented and ARE covered by
                // ctest/accuracy/test_spmv.cpp, but this sweep has no operand
                // builder for them yet. Saying so beats omitting the row.
                const std::string why =
                    std::string("benchmark/test_spmv.cpp has no ") + v->format +
                    " operand builder yet (accuracy coverage exists)";
                report_unimplemented(g_report, *v, why.c_str());
                continue;
            }
            const auto dt = v->dt;
            BenchRow row;
            row.name = std::string("spmv_") + v->format + "_" + v->dtype + "_" +
                       entry.name;
            row.tag("operator", v->op)
               .tag("matrix", entry.name).tag("format", v->format).tag("dtype", v->dtype)
               .tag("corpus", corpus_tag()).tag("reporting", v->reporting)
               .num("rows", static_cast<double>(A.rows))
               .num("cols", static_cast<double>(A.cols))
               .num("nnz", static_cast<double>(A.nnz));
            trace("spmv", entry.name, v->dtype, A);

            DeviceBuffer indptr = DeviceBuffer::from(A.indptr);
            DeviceBuffer indices = DeviceBuffer::from(A.indices);
            DeviceBuffer rowind = DeviceBuffer::from(coo_rows);
            DeviceBuffer values = upload_as(A.values, dt);
            DeviceBuffer x = upload_as(x_host, dt);
            DeviceBuffer y(static_cast<std::size_t>(A.rows) * elem_bytes(dt));
            if (!indptr.get() || !indices.get() || !rowind.get() || !values.get() ||
                !x.get() || !y.get()) {
                // Out of memory on this matrix is a fact about the matrix; the
                // next one may well fit, so record and carry on.
                g_report.skip(std::move(row), "skipped_memory",
                              "device allocation failed for this matrix");
                continue;
            }

            flagsparseSpMatDescr_t matA = nullptr;
            flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
            const flagsparseStatus_t cs =
                is_coo
                    ? flagsparseCreateCoo(&matA, A.rows, A.cols, A.nnz, rowind.get(),
                                          indices.get(), values.get(),
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
            flagsparseCreateDnVec(&vecX, A.cols, x.get(), dt);
            flagsparseCreateDnVec(&vecY, A.rows, y.get(), dt);

            // Scratch comes from bufferSize, never a guess, and is allocated
            // outside the timed region so the allocator never lands in a sample.
            std::size_t bufsz = 0;
            flagsparseSpMV_bufferSize(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                      sc.alpha(dt), matA, vecX, sc.beta(dt), vecY, dt,
                                      FLAGSPARSE_SPMV_ALG_DEFAULT, &bufsz);
            DeviceBuffer scratch(bufsz ? bufsz : 1);
            if (!scratch.get()) {
                flagsparseDestroyDnVec(vecX); flagsparseDestroyDnVec(vecY);
                flagsparseDestroySpMat(matA);
                g_report.skip(std::move(row), "skipped_memory",
                              "SpMV scratch allocation failed");
                continue;
            }

            baseline::DeviceCsr bA{indptr.get(), indices.get(), values.get(),
                                   A.rows, A.cols, A.nnz, dt,
                                   is_coo ? rowind.get() : nullptr};
            g_report.measure_vs_baseline(
                std::move(row),
                [&]() {
                    return flagsparseSpMV(handle.h, FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                          sc.alpha(dt), matA, vecX, sc.beta(dt), vecY,
                                          dt, FLAGSPARSE_SPMV_ALG_DEFAULT, scratch.get());
                },
                [&](bool relaxed) { return ratio_against(y.get(), ref, dt, relaxed); },
                [&](baseline::Timing* t) {
                    return baseline::spmv_csr(bA, x.get(), y.get(), sc.alpha(dt),
                                              sc.beta(dt),
                                              FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                              BenchReport::kWarmup,
                                              BenchReport::kIters, t);
                },
                2.0 * static_cast<double>(A.nnz));

            flagsparseDestroyDnVec(vecX);
            flagsparseDestroyDnVec(vecY);
            flagsparseDestroySpMat(matA);
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
