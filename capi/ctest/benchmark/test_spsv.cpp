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

// SpSV over the real-matrix corpus, against the vendor baseline.
//
// The operand is each corpus matrix's LOWER TRIANGLE with a dominant diagonal
// (sweep.hpp::lower_triangle) -- a triangular solve on a general matrix is not
// defined, and on a real matrix's own diagonal it diverges.
//
// ANALYSIS IS NOT IN THE TIMED REGION, on either side. cuSPARSE splits analysis
// from solve precisely so a caller pays the level schedule once and solves many
// right-hand sides against it; timing ours against their analysis+solve would
// compare two different operations. Both sides here time the solve alone.

#include <gtest/gtest.h>

#include <vector>

#include "baseline/baseline.hpp"
#include "sweep.hpp"

using namespace fstest;

namespace {

BenchReport g_report("spsv");

}  // namespace

TEST(SpsvBenchmark, CsrOverCorpus) {
    Handle handle;
    ASSERT_NE(handle.h, nullptr);
    report_corpus_failures(g_report, "csr");

    const Scalars sc;
    const auto declared = variants_of("spsv");

    for (const auto& entry : corpus()) {
        const CsrMatrix L = lower_triangle(entry.A);
        if (L.rows <= 0) {
            g_report.skip(BenchRow{}.tag("matrix", entry.name).tag("format", "csr"),
                          "skipped_shape", "empty triangle");
            continue;
        }
        const std::vector<double> b = dense_pattern(static_cast<std::size_t>(L.rows));
        const std::vector<double> ref = trsv_reference(L, b);
        const std::vector<int32_t> coo_rows = coo_row_indices_of(L);

        for (const registry::Variant* v : declared) {
            const bool is_coo = std::string(v->format) == "coo";
            const bool is_csr = std::string(v->format) == "csr";
            if (!is_csr && !is_coo) {
                // SELL is declared implemented and ctest/accuracy/test_spsv.cpp
                // covers it; this sweep has no SELL operand builder yet.
                const std::string why =
                    std::string("benchmark/test_spsv.cpp has no ") + v->format +
                    " operand builder yet (accuracy coverage exists)";
                report_unimplemented(g_report, *v, why.c_str());
                continue;
            }
            const auto dt = v->dt;
            BenchRow row;
            row.name = std::string("spsv_") + v->format + "_" + v->dtype + "_" +
                       entry.name;
            row.tag("operator", v->op)
               .tag("matrix", entry.name).tag("format", v->format)
               .tag("dtype", v->dtype)
               .tag("corpus", corpus_tag()).tag("reporting", v->reporting).tag("fill", "lower").tag("diag", "non_unit")
               .num("rows", static_cast<double>(L.rows))
               .num("nnz", static_cast<double>(L.nnz));
            trace("spsv", entry.name, v->dtype, L);

            DeviceBuffer indptr = DeviceBuffer::from(L.indptr);
            DeviceBuffer indices = DeviceBuffer::from(L.indices);
            DeviceBuffer rowind = DeviceBuffer::from(coo_rows);
            DeviceBuffer values = upload_as(L.values, dt);
            DeviceBuffer x = upload_as(b, dt);
            DeviceBuffer y(static_cast<std::size_t>(L.rows) * elem_bytes(dt));
            if (!indptr.get() || !indices.get() || !rowind.get() || !values.get() ||
                !x.get() || !y.get()) {
                g_report.skip(std::move(row), "skipped_memory",
                              "device allocation failed for this matrix");
                continue;
            }

            flagsparseSpMatDescr_t matA = nullptr;
            flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
            const flagsparseStatus_t cs =
                is_coo
                    ? flagsparseCreateCoo(&matA, L.rows, L.cols, L.nnz, rowind.get(),
                                          indices.get(), values.get(),
                                          FLAGSPARSE_INDEX_32I,
                                          FLAGSPARSE_INDEX_BASE_ZERO, dt)
                    : flagsparseCreateCsr(&matA, L.rows, L.cols, L.nnz, indptr.get(),
                                          indices.get(), values.get(),
                                          FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                          FLAGSPARSE_INDEX_BASE_ZERO, dt);
            if (cs != FLAGSPARSE_STATUS_SUCCESS) {
                g_report.skip(std::move(row), "failed",
                              std::string("descriptor creation failed for ") +
                                  v->format);
                continue;
            }
            const flagsparseFillMode_t fill = FLAGSPARSE_FILL_MODE_LOWER;
            const flagsparseDiagType_t diag = FLAGSPARSE_DIAG_TYPE_NON_UNIT;
            flagsparseSpMatSetAttribute(matA, FLAGSPARSE_SPMAT_FILL_MODE, &fill,
                                        sizeof(fill));
            flagsparseSpMatSetAttribute(matA, FLAGSPARSE_SPMAT_DIAG_TYPE, &diag,
                                        sizeof(diag));
            flagsparseCreateDnVec(&vecX, L.cols, x.get(), dt);
            flagsparseCreateDnVec(&vecY, L.rows, y.get(), dt);

            const auto NT = FLAGSPARSE_OPERATION_NON_TRANSPOSE;
            flagsparseSpSVDescr_t descr = nullptr;
            flagsparseSpSV_createDescr(&descr);
            std::size_t bufsz = 0;
            flagsparseSpSV_bufferSize(handle.h, NT, sc.alpha(dt), matA, vecX, vecY, dt,
                                      FLAGSPARSE_SPSV_ALG_DEFAULT, descr, &bufsz);
            DeviceBuffer scratch(bufsz ? bufsz : 1);
            auto teardown = [&]() {
                flagsparseSpSV_destroyDescr(descr);
                flagsparseDestroyDnVec(vecX); flagsparseDestroyDnVec(vecY);
                flagsparseDestroySpMat(matA);
            };
            if (!scratch.get()) {
                teardown();
                g_report.skip(std::move(row), "skipped_memory",
                              "SpSV scratch allocation failed");
                continue;
            }
            const flagsparseStatus_t an =
                flagsparseSpSV_analysis(handle.h, NT, sc.alpha(dt), matA, vecX, vecY,
                                        dt, FLAGSPARSE_SPSV_ALG_DEFAULT, descr,
                                        scratch.get());
            if (an != FLAGSPARSE_STATUS_SUCCESS) {
                teardown();
                g_report.skip(std::move(row),
                              an == FLAGSPARSE_STATUS_NOT_SUPPORTED ? "not_supported"
                                                                    : "failed",
                              "SpSV_analysis declined this matrix");
                continue;
            }

            // cuSPARSE 12.5 accepts a COO operand here (measured, not assumed).
            // A vendor that does not will return its own status, which becomes a
            // blank speedup carrying that reason rather than a missing row.
            baseline::DeviceCsr bA{indptr.get(), indices.get(), values.get(),
                                   L.rows, L.cols, L.nnz, dt,
                                   is_coo ? rowind.get() : nullptr};
            g_report.measure_vs_baseline(
                std::move(row),
                [&]() {
                    return flagsparseSpSV_solve(handle.h, NT, sc.alpha(dt), matA, vecX,
                                                vecY, dt, FLAGSPARSE_SPSV_ALG_DEFAULT,
                                                descr);
                },
                [&](bool relaxed) { return ratio_against(y.get(), ref, dt, relaxed); },
                [&](baseline::Timing* t) {
                    return baseline::spsv_csr(bA, x.get(), y.get(), sc.alpha(dt), fill,
                                              diag, NT, BenchReport::kWarmup,
                                              BenchReport::kIters, t);
                },
                2.0 * static_cast<double>(L.nnz));
            teardown();
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
