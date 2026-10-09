// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <vector>

#include "sweep.hpp"

using namespace fstest;

namespace {
BenchReport g_report("axpby");
}

TEST(AxpbyBenchmark, SparseVector) {
    Handle handle;
    ASSERT_NE(handle.h, nullptr);
    const auto declared = variants_of("axpby");
    const std::vector<int32_t> indices{0, 2, 3};
    const std::vector<double> sparse_h{2.0, -1.0, 4.0};
    const std::vector<double> initial_h{1.0, 2.0, 3.0, -2.0};
    const std::vector<double> ref{2.5, -1.0, -3.0, 7.0};
    const uint16_t alpha = float_to_fp16(1.5f);
    const uint16_t beta = float_to_fp16(-0.5f);

    for (const registry::Variant* v : declared) {
        if (!benchmark_variant_selected(*v)) continue;
        const auto dt = v->dt;
        BenchRow row;
        row.name = std::string("axpby_spvec_") + v->dtype;
        row.tag("operator", v->op).tag("format", v->format).tag("dtype", v->dtype)
           .tag("reporting", v->reporting).num("nnz", 3).num("size", 4)
           .num("bytes_moved", 3.0 * elem_bytes(dt) + 4.0 * elem_bytes(dt));
        if (v->variant_id) row.tag("variant", v->variant_id);

        DeviceBuffer d_idx = DeviceBuffer::from(indices);
        DeviceBuffer d_x = upload_as(sparse_h, dt);
        DeviceBuffer d_y = upload_as(initial_h, dt);
        if (!d_idx.get() || !d_x.get() || !d_y.get()) {
            g_report.skip(std::move(row), "skipped_memory", "device allocation failed");
            continue;
        }
        flagsparseSpVecDescr_t x = nullptr;
        flagsparseDnVecDescr_t y = nullptr;
        if (flagsparseCreateSpVec(&x, 4, 3, d_idx.get(), d_x.get(), FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_BASE_ZERO, dt) != FLAGSPARSE_STATUS_SUCCESS ||
            flagsparseCreateDnVec(&y, 4, d_y.get(), dt) != FLAGSPARSE_STATUS_SUCCESS) {
            if (y) flagsparseDestroyDnVec(y);
            if (x) flagsparseDestroySpVec(x);
            g_report.skip(std::move(row), "failed", "descriptor creation failed");
            continue;
        }
        g_report.measure_vs_baseline(
            std::move(row),
            [&]() { return flagsparseAxpby(handle.h, &alpha, x, &beta, y); },
            [&](bool relaxed) { return ratio_against(d_y.get(), ref, dt, relaxed); },
            [&](baseline::Timing*) {
                return baseline::Status::no("no matching cuSPARSE Axpby baseline in harness");
            },
            0.0);
        flagsparseDestroyDnVec(y);
        flagsparseDestroySpVec(x);
    }
    EXPECT_GT(g_report.size(), 0u);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    const int rc = RUN_ALL_TESTS();
    g_report.write();
    return rc;
}
