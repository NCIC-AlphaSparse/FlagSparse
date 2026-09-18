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

// SpSM over the real-matrix corpus, against the vendor baseline.
//
// Same operand construction and the same analysis-excluded timing as SpSV (see
// benchmark/test_spsv.cpp); what this adds is the right-hand-side count. On this
// box n=1 -> 32 measured as nearly free, which is the interesting property of a
// triangular solve: the level schedule is the cost and it is shared across
// columns, so the ratio against a vendor kernel should move with n. Sweeping n
// is how that claim stays checked rather than remembered.

#include <gtest/gtest.h>

#include <vector>

#include "baseline/baseline.hpp"
#include "sweep.hpp"

using namespace fstest;

namespace {

BenchReport g_report("spsm");

constexpr int64_t kRhs[] = {1, 8, 32};

}  // namespace

TEST(SpsmBenchmark, CsrOverCorpus) {
    Handle handle;
    ASSERT_NE(handle.h, nullptr);
    report_corpus_failures(g_report, "csr");

    const Scalars sc;
    const auto declared = variants_of("spsm");

    for (const auto& entry : corpus()) {
        const CsrMatrix L = lower_triangle(entry.A);
        if (L.rows <= 0) {
            g_report.skip(BenchRow{}.tag("matrix", entry.name).tag("format", "csr"),
                          "skipped_shape", "empty triangle");
            continue;
        }
        for (const int64_t n : kRhs) {
            // B column-major (rows x n). Each column is solved independently, so
            // the oracle is the per-column forward substitution.
            const std::size_t count = static_cast<std::size_t>(L.rows) *
                                      static_cast<std::size_t>(n);
            const std::vector<double> b_col = dense_pattern(count);
            std::vector<double> ref(count);
            for (int64_t j = 0; j < n; ++j) {
                std::vector<double> col(b_col.begin() + static_cast<long>(j * L.rows),
                                        b_col.begin() +
                                            static_cast<long>((j + 1) * L.rows));
                const std::vector<double> sol = trsv_reference(L, col);
                for (int64_t r = 0; r < L.rows; ++r) {
                    ref[static_cast<std::size_t>(j) * L.rows + r] =
                        sol[static_cast<std::size_t>(r)];
                }
            }

            for (const registry::Variant* v : declared) {
                if (std::string(v->format) != "csr") {
                    if (n == kRhs[0]) {
                        const std::string why =
                            std::string("benchmark/test_spsm.cpp has no ") +
                            v->format + " operand builder yet "
                            "(accuracy coverage exists)";
                        report_unimplemented(g_report, *v, why.c_str());
                    }
                    continue;
                }
                const auto dt = v->dt;
                BenchRow row;
                row.name = std::string("spsm_csr_") + v->dtype + "_n" +
                           std::to_string(n) + "_" + entry.name;
                row.tag("operator", v->op)
                   .tag("matrix", entry.name).tag("format", "csr")
                   .tag("dtype", v->dtype).tag("corpus", corpus_tag()).tag("reporting", v->reporting)
                   .tag("fill", "lower").tag("diag", "non_unit")
                   .num("rows", static_cast<double>(L.rows))
                   .num("nnz", static_cast<double>(L.nnz))
                   .num("n", static_cast<double>(n));
                trace("spsm", entry.name, v->dtype, L);

                DeviceBuffer indptr = DeviceBuffer::from(L.indptr);
                DeviceBuffer indices = DeviceBuffer::from(L.indices);
                DeviceBuffer values = upload_as(L.values, dt);
                DeviceBuffer B = upload_as(b_col, dt);
                DeviceBuffer C(count * elem_bytes(dt));
                if (!indptr.get() || !indices.get() || !values.get() || !B.get() ||
                    !C.get()) {
                    g_report.skip(std::move(row), "skipped_memory",
                                  "device allocation failed at n=" + std::to_string(n));
                    continue;
                }

                flagsparseSpMatDescr_t matA = nullptr;
                flagsparseDnMatDescr_t matB = nullptr, matC = nullptr;
                if (flagsparseCreateCsr(&matA, L.rows, L.cols, L.nnz, indptr.get(),
                                        indices.get(), values.get(),
                                        FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                        FLAGSPARSE_INDEX_BASE_ZERO, dt) !=
                    FLAGSPARSE_STATUS_SUCCESS) {
                    g_report.skip(std::move(row), "failed", "flagsparseCreateCsr failed");
                    continue;
                }
                const flagsparseFillMode_t fill = FLAGSPARSE_FILL_MODE_LOWER;
                const flagsparseDiagType_t diag = FLAGSPARSE_DIAG_TYPE_NON_UNIT;
                flagsparseSpMatSetAttribute(matA, FLAGSPARSE_SPMAT_FILL_MODE, &fill,
                                            sizeof(fill));
                flagsparseSpMatSetAttribute(matA, FLAGSPARSE_SPMAT_DIAG_TYPE, &diag,
                                            sizeof(diag));
                flagsparseCreateDnMat(&matB, L.rows, n, L.rows, B.get(), dt,
                                      FLAGSPARSE_ORDER_COL);
                flagsparseCreateDnMat(&matC, L.rows, n, L.rows, C.get(), dt,
                                      FLAGSPARSE_ORDER_COL);

                const auto NT = FLAGSPARSE_OPERATION_NON_TRANSPOSE;
                flagsparseSpSMDescr_t descr = nullptr;
                flagsparseSpSM_createDescr(&descr);
                std::size_t bufsz = 0;
                flagsparseSpSM_bufferSize(handle.h, NT, NT, sc.alpha(dt), matA, matB,
                                          matC, dt, FLAGSPARSE_SPSM_ALG_DEFAULT, descr,
                                          &bufsz);
                DeviceBuffer scratch(bufsz ? bufsz : 1);
                auto teardown = [&]() {
                    flagsparseSpSM_destroyDescr(descr);
                    flagsparseDestroyDnMat(matB); flagsparseDestroyDnMat(matC);
                    flagsparseDestroySpMat(matA);
                };
                if (!scratch.get()) {
                    teardown();
                    g_report.skip(std::move(row), "skipped_memory",
                                  "SpSM scratch allocation failed");
                    continue;
                }
                const flagsparseStatus_t an = flagsparseSpSM_analysis(
                    handle.h, NT, NT, sc.alpha(dt), matA, matB, matC, dt,
                    FLAGSPARSE_SPSM_ALG_DEFAULT, descr, scratch.get());
                if (an != FLAGSPARSE_STATUS_SUCCESS) {
                    teardown();
                    g_report.skip(std::move(row),
                                  an == FLAGSPARSE_STATUS_NOT_SUPPORTED
                                      ? "not_supported" : "failed",
                                  "SpSM_analysis declined this matrix");
                    continue;
                }

                baseline::DeviceCsr bA{indptr.get(), indices.get(), values.get(),
                                       L.rows, L.cols, L.nnz, dt};
                g_report.measure_vs_baseline(
                    std::move(row),
                    [&]() {
                        return flagsparseSpSM_solve(handle.h, NT, NT, sc.alpha(dt),
                                                    matA, matB, matC, dt,
                                                    FLAGSPARSE_SPSM_ALG_DEFAULT, descr);
                    },
                    [&](bool relaxed) { return ratio_against(C.get(), ref, dt, relaxed); },
                    [&](baseline::Timing* t) {
                        return baseline::spsm_csr(bA, B.get(), n, L.rows, C.get(),
                                                  L.rows, sc.alpha(dt), fill, diag, NT,
                                                  BenchReport::kWarmup,
                                                  BenchReport::kIters, t);
                    },
                    2.0 * static_cast<double>(L.nnz) * static_cast<double>(n));
                teardown();
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
