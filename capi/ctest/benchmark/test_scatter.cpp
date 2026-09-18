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

// Scatter over the real-matrix corpus, against the vendor baseline.
//
// The mirror of gather, and NOT a copy of it: scatter writes the sparse values
// into the dense vector, so the buffer to check is the dense one and the oracle
// is the starting dense vector with the scattered entries applied.
//
// DUPLICATE INDICES WOULD MAKE THIS UNCHECKABLE. A matrix row can repeat a
// column index across rows, and two writes to one dense slot have no defined
// winner, so the index slice is deduplicated before use. Without that the error
// ratio would be a race, failing at random on some matrices and passing on
// others -- the worst kind of flaky test, because it looks like a precision bug.
//
// As with gather there is no arithmetic, so the measurement column is
// bytes_moved rather than GFLOPS.

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "baseline/baseline.hpp"
#include "sweep.hpp"

using namespace fstest;

namespace {

BenchReport g_report("scatter");

}  // namespace

TEST(ScatterBenchmark, SpVecOverCorpus) {
    Handle handle;
    ASSERT_NE(handle.h, nullptr);
    report_corpus_failures(g_report, "spvec");

    const auto declared = variants_of("scatter");

    for (const auto& entry : corpus()) {
        const CsrMatrix& A = entry.A;
        const int64_t dense = A.cols;
        if (dense <= 0) {
            g_report.skip(BenchRow{}.tag("matrix", entry.name).tag("format", "spvec"),
                          "skipped_shape", "empty matrix");
            continue;
        }
        // Deduplicate, then truncate: nnz <= size is a descriptor requirement.
        std::vector<int32_t> idx(A.indices.begin(),
                                 A.indices.begin() +
                                     std::min<int64_t>(A.nnz, dense));
        std::sort(idx.begin(), idx.end());
        idx.erase(std::unique(idx.begin(), idx.end()), idx.end());
        const int64_t nnz = static_cast<int64_t>(idx.size());
        if (nnz <= 0) {
            g_report.skip(BenchRow{}.tag("matrix", entry.name).tag("format", "spvec"),
                          "skipped_shape", "no usable indices");
            continue;
        }

        const std::vector<double> dense_h =
            dense_pattern(static_cast<std::size_t>(dense));
        const std::vector<double> val_h =
            dense_pattern(static_cast<std::size_t>(nnz), 5.0);
        std::vector<double> ref = dense_h;
        for (int64_t i = 0; i < nnz; ++i) {
            ref[static_cast<std::size_t>(idx[static_cast<std::size_t>(i)])] =
                val_h[static_cast<std::size_t>(i)];
        }

        for (const registry::Variant* v : declared) {
            const auto dt = v->dt;
            BenchRow row;
            row.name = std::string("scatter_spvec_") + v->dtype + "_" + entry.name;
            // One scattered element reads a value and an index and writes a value.
            const double bytes = static_cast<double>(nnz) *
                                 (2.0 * static_cast<double>(elem_bytes(dt)) +
                                  static_cast<double>(sizeof(int32_t)));
            row.tag("matrix", entry.name).tag("format", "spvec").tag("operator", v->op).tag("dtype", v->dtype)
               .tag("corpus", corpus_tag()).tag("reporting", v->reporting)
               .num("nnz", static_cast<double>(nnz))
               .num("size", static_cast<double>(dense))
               .num("bytes_moved", bytes);
            trace("scatter", entry.name, v->dtype, A);

            DeviceBuffer d_idx = DeviceBuffer::from(idx);
            DeviceBuffer d_dense = upload_as(dense_h, dt);
            DeviceBuffer d_val = upload_as(val_h, dt);
            if (!d_idx.get() || !d_dense.get() || !d_val.get()) {
                g_report.skip(std::move(row), "skipped_memory",
                              "device allocation failed for this matrix");
                continue;
            }

            flagsparseSpVecDescr_t vecX = nullptr;
            flagsparseDnVecDescr_t vecY = nullptr;
            if (flagsparseCreateSpVec(&vecX, dense, nnz, d_idx.get(), d_val.get(),
                                      FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO,
                                      dt) != FLAGSPARSE_STATUS_SUCCESS) {
                g_report.skip(std::move(row), "failed", "flagsparseCreateSpVec failed");
                continue;
            }
            flagsparseCreateDnVec(&vecY, dense, d_dense.get(), dt);

            g_report.measure_vs_baseline(
                std::move(row),
                [&]() { return flagsparseScatter(handle.h, vecX, vecY); },
                [&](bool relaxed) { return ratio_against(d_dense.get(), ref, dt, relaxed); },
                [&](baseline::Timing* t) {
                    return baseline::scatter(d_dense.get(), d_val.get(), d_idx.get(),
                                             nnz, dense, dt, BenchReport::kWarmup,
                                             BenchReport::kIters, t);
                },
                0.0);

            flagsparseDestroyDnVec(vecY);
            flagsparseDestroySpVec(vecX);
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
