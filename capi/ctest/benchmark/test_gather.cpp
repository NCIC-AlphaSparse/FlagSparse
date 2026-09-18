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

// Gather over the real-matrix corpus, against the vendor baseline.
//
// Gather does no arithmetic, so GFLOPS is meaningless for it and the column that
// carries the measurement is bytes_moved / GB_per_s instead. Reporting a zero
// GFLOPS and nothing else -- which an earlier sweep of mine did -- makes these
// rows look unmeasured when they are in fact the cleanest bandwidth numbers here.
//
// The index set is the matrix's own column indices, truncated to the dense
// vector's length. That is the point of sweeping a corpus for a gather: real
// column indices carry real locality, and a random permutation of the same
// length measures a different machine (~635 GB/s random vs ~1061 GB/s sequential
// on this box).

#include <gtest/gtest.h>

#include <algorithm>
#include <vector>

#include "baseline/baseline.hpp"
#include "sweep.hpp"

using namespace fstest;

namespace {

BenchReport g_report("gather");

}  // namespace

TEST(GatherBenchmark, SpVecOverCorpus) {
    Handle handle;
    ASSERT_NE(handle.h, nullptr);
    report_corpus_failures(g_report, "spvec");

    const auto declared = variants_of("gather");

    for (const auto& entry : corpus()) {
        const CsrMatrix& A = entry.A;
        // A sparse vector of nnz entries indexing a dense vector of `dense`
        // elements: the descriptor requires nnz <= size, so a matrix with more
        // nonzeros than columns contributes a truncated index slice rather than
        // an INVALID_VALUE row.
        const int64_t dense = A.cols;
        const int64_t nnz = std::min<int64_t>(A.nnz, dense);
        if (nnz <= 0) {
            g_report.skip(BenchRow{}.tag("matrix", entry.name).tag("format", "spvec"),
                          "skipped_shape", "empty matrix");
            continue;
        }
        std::vector<int32_t> idx(A.indices.begin(), A.indices.begin() + nnz);
        const std::vector<double> dense_h =
            dense_pattern(static_cast<std::size_t>(dense));
        std::vector<double> ref(static_cast<std::size_t>(nnz));
        for (int64_t i = 0; i < nnz; ++i) {
            ref[static_cast<std::size_t>(i)] =
                dense_h[static_cast<std::size_t>(idx[static_cast<std::size_t>(i)])];
        }

        for (const registry::Variant* v : declared) {
            const auto dt = v->dt;
            BenchRow row;
            row.name = std::string("gather_spvec_") + v->dtype + "_" + entry.name;
            // One gathered element reads a value and an index and writes a value.
            const double bytes = static_cast<double>(nnz) *
                                 (2.0 * static_cast<double>(elem_bytes(dt)) +
                                  static_cast<double>(sizeof(int32_t)));
            row.tag("matrix", entry.name).tag("format", "spvec").tag("operator", v->op).tag("dtype", v->dtype)
               .tag("corpus", corpus_tag()).tag("reporting", v->reporting)
               .num("nnz", static_cast<double>(nnz))
               .num("size", static_cast<double>(dense))
               .num("bytes_moved", bytes);
            trace("gather", entry.name, v->dtype, A);

            DeviceBuffer d_idx = DeviceBuffer::from(idx);
            DeviceBuffer d_dense = upload_as(dense_h, dt);
            DeviceBuffer d_val(static_cast<std::size_t>(nnz) * elem_bytes(dt));
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
                [&]() { return flagsparseGather(handle.h, vecY, vecX); },
                [&](bool relaxed) { return ratio_against(d_val.get(), ref, dt, relaxed); },
                [&](baseline::Timing* t) {
                    return baseline::gather(d_dense.get(), d_val.get(), d_idx.get(),
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
