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

// gemvi over synthetic (m, n, nnz) cases -- a dense matrix and a sparse vector,
// so no matrix file applies -- the same four tests/test_gemvi.py measures,
// against the vendor's legacy <t>gemvi. A is column-major, as gemvi requires.

#include <gtest/gtest.h>

#include <complex>
#include <cstdint>
#include <string>
#include <vector>

#include "baseline/baseline.hpp"
#include "sweep.hpp"

using namespace fstest;

namespace {

BenchReport g_report("gemvi");

struct Case { int m, n, nnz; };
constexpr Case kCases[] = {{8192, 4096, 256}, {16384, 8192, 2048},
                           {65536, 4096, 1024}, {4096, 65536, 4096}};

double a_value(int64_t i, int64_t j) { return 0.5 + 0.25 * static_cast<double>((i * 7 + j * 13) % 5); }

// A as the device stores it: real-valued, complex ones with a zero imaginary
// part like every other sweep's complex operands (read_back compares the real
// part; the accuracy suite checks complex arithmetic).
template <typename T>
DeviceBuffer upload_dense(int m, int n) {
    std::vector<T> host(static_cast<std::size_t>(m) * n);
    for (int64_t j = 0; j < n; ++j)
        for (int64_t i = 0; i < m; ++i)
            host[static_cast<std::size_t>(i + j * m)] = static_cast<T>(a_value(i, j));
    return DeviceBuffer::from(host);
}

}  // namespace

TEST(GemviBenchmark, SyntheticCases) {
    Handle handle;
    ASSERT_NE(handle.h, nullptr);
    const Scalars sc;
    for (const registry::Variant* v : variants_of("gemvi")) {
        if (!benchmark_variant_selected(*v)) continue;
        const auto dt = v->dt;
        const std::string vid = v->variant_id ? v->variant_id : "";
        const flagsparseOperation_t op = vid.find("_int_trans") != std::string::npos
            ? FLAGSPARSE_OPERATION_TRANSPOSE : FLAGSPARSE_OPERATION_NON_TRANSPOSE;
        const bool trans = op == FLAGSPARSE_OPERATION_TRANSPOSE;
        for (const Case& c : kCases) {
            const int n_in = trans ? c.m : c.n, n_out = trans ? c.n : c.m;
            BenchRow row;
            const std::string shape = "m" + std::to_string(c.m) + "_n" + std::to_string(c.n) +
                                      "_nnz" + std::to_string(c.nnz);
            row.name = std::string("gemvi_") + v->dtype + (trans ? "_trans_" : "_non_") + shape;
            row.tag("operator", v->op).tag("matrix", shape).tag("format", v->format)
               .tag("dtype", v->dtype).tag("corpus", "synthetic").tag("reporting", v->reporting)
               .num("rows", c.m).num("cols", c.n).num("nnz", c.nnz);
            if (v->variant_id) row.tag("variant", v->variant_id);

            std::vector<int32_t> idx(static_cast<std::size_t>(c.nnz));
            // An odd stride visits each index of a power-of-two range once.
            for (int k = 0; k < c.nnz; ++k) idx[k] = static_cast<int32_t>((int64_t{k} * 7919) % n_in);
            const std::vector<double> x_host = dense_pattern(static_cast<std::size_t>(c.nnz));
            std::vector<double> ref(static_cast<std::size_t>(n_out), 0.0);
            for (int r = 0; r < n_out; ++r)
                for (int k = 0; k < c.nnz; ++k)
                    ref[r] += (trans ? a_value(idx[k], r) : a_value(r, idx[k])) * x_host[k];

            DeviceBuffer A = dt == FLAGSPARSE_C_32F   ? upload_dense<std::complex<float>>(c.m, c.n)
                             : dt == FLAGSPARSE_C_64F ? upload_dense<std::complex<double>>(c.m, c.n)
                             : dt == FLAGSPARSE_R_64F ? upload_dense<double>(c.m, c.n)
                                                      : upload_dense<float>(c.m, c.n);
            DeviceBuffer d_idx = DeviceBuffer::from(idx);
            DeviceBuffer x = upload_as(x_host, dt);
            DeviceBuffer y(static_cast<std::size_t>(n_out) * elem_bytes(dt));
            if (!A.get() || !d_idx.get() || !x.get() || !y.get()) {
                g_report.skip(std::move(row), "skipped_memory", "device allocation failed");
                continue;
            }
            std::size_t bytes = 0;
            flagsparseGemvi_bufferSize(handle.h, op, c.m, c.n, c.nnz, dt, &bytes);
            DeviceBuffer scratch(bytes ? bytes : 1);
            const auto* ind = static_cast<const int*>(d_idx.get());
            g_report.measure_vs_baseline(
                std::move(row),
                [&]() {
                    return flagsparseGemvi(handle.h, op, c.m, c.n, sc.alpha(dt), A.get(), c.m,
                                           c.nnz, x.get(), ind, sc.beta(dt), y.get(),
                                           FLAGSPARSE_INDEX_BASE_ZERO, dt,
                                           bytes ? scratch.get() : nullptr);
                },
                [&](bool relaxed) { return ratio_against(y.get(), ref, dt, relaxed); },
                [&](baseline::Timing* t) {
                    return baseline::gemvi(op, c.m, c.n, sc.alpha(dt), A.get(), c.m, c.nnz,
                                           x.get(), ind, sc.beta(dt), y.get(), dt,
                                           BenchReport::kWarmup, BenchReport::kIters, t);
                },
                2.0 * static_cast<double>(n_out) * static_cast<double>(c.nnz));
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
