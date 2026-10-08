// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>

#include <cstring>
#include <cstdint>
#include <vector>

#include "baseline/baseline.hpp"
#include "sweep.hpp"

using namespace fstest;

namespace {
BenchReport g_report("spvv");
}

TEST(SpvvBenchmark, SparseVectorDot) {
    Handle handle;
    ASSERT_NE(handle.h, nullptr);
    const auto declared = variants_of("spvv");
    const std::vector<int32_t> indices{0, 2, 3};
    const std::vector<double> sparse_h{2.0, -1.0, 4.0};
    const std::vector<double> dense_h{3.0, 1.0, 5.0, -2.0};

    for (const registry::Variant* v : declared) {
        if (!benchmark_variant_selected(*v)) continue;
        const auto dt = v->dt;
        const auto out_dt = variant_output_dtype(*v);
        const std::string q = v->variant_id ? v->variant_id : "";
        const flagsparseOperation_t op = q.find("_int_conj") != std::string::npos
            ? FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE
            : FLAGSPARSE_OPERATION_NON_TRANSPOSE;
        const std::vector<double> ref{-7.0};
        BenchRow row;
        row.name = std::string("spvv_") + v->dtype;
        row.tag("operator", v->op).tag("format", v->format).tag("dtype", v->dtype)
           .tag("reporting", v->reporting).num("nnz", 3).num("size", 4);
        if (v->variant_id) row.tag("variant", v->variant_id);

        DeviceBuffer d_idx = DeviceBuffer::from(indices);
        DeviceBuffer d_sparse = upload_as(sparse_h, dt);
        DeviceBuffer d_dense = upload_as(dense_h, dt);
        std::vector<unsigned char> result(elem_bytes(out_dt), 0);
        if (!d_idx.get() || !d_sparse.get() || !d_dense.get()) {
            g_report.skip(std::move(row), "skipped_memory", "device allocation failed");
            continue;
        }
        flagsparseSpVecDescr_t x = nullptr;
        flagsparseDnVecDescr_t y = nullptr;
        if (flagsparseCreateSpVec(&x, 4, 3, d_idx.get(), d_sparse.get(), FLAGSPARSE_INDEX_32I,
                                  FLAGSPARSE_INDEX_BASE_ZERO, dt) != FLAGSPARSE_STATUS_SUCCESS ||
            flagsparseCreateDnVec(&y, 4, d_dense.get(), dt) != FLAGSPARSE_STATUS_SUCCESS) {
            if (y) flagsparseDestroyDnVec(y);
            if (x) flagsparseDestroySpVec(x);
            g_report.skip(std::move(row), "failed", "descriptor creation failed");
            continue;
        }
        std::size_t bytes = 0;
        const flagsparseStatus_t bs = flagsparseSpVV_bufferSize(
            handle.h, op, x, y, result.data(), out_dt, &bytes);
        if (bs != FLAGSPARSE_STATUS_SUCCESS) {
            g_report.skip(std::move(row), "failed", "SpVV bufferSize failed");
            flagsparseDestroyDnVec(y); flagsparseDestroySpVec(x);
            continue;
        }
        DeviceBuffer scratch(bytes ? bytes : 1);
        g_report.measure_vs_baseline(
            std::move(row),
            [&]() { return flagsparseSpVV(handle.h, op, x, y, result.data(), out_dt,
                                          scratch.get()); },
            [&](bool relaxed) {
                double got = 0.0;
                if (out_dt == FLAGSPARSE_R_32I) {
                    std::int32_t v = 0; std::memcpy(&v, result.data(), sizeof(v)); got = v;
                } else if (out_dt == FLAGSPARSE_R_32F) {
                    float v = 0.0f; std::memcpy(&v, result.data(), sizeof(v)); got = v;
                } else {
                    float v = 0.0f; std::memcpy(&v, result.data(), sizeof(v)); got = v;
                }
                return max_error_ratio(std::vector<double>{got}, ref,
                                       relaxed ? relaxed_tolerance(out_dt)
                                               : default_tolerance(out_dt));
            },
            [&](baseline::Timing* t) {
#if defined(FLAGSPARSE_MUSA_BASELINE_EXTENSIONS)
                return baseline::spvv(d_sparse.get(), d_idx.get(), d_dense.get(), 3, 4,
                                      result.data(), dt, out_dt, op,
                                      BenchReport::kWarmup, BenchReport::kIters, t);
#else
                (void)t;
                return baseline::Status::no("no matching cuSPARSE SpVV baseline in harness");
#endif
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
