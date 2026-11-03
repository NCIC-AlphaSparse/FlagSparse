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

// csrgeam2 (C = A + B) over the real-matrix corpus, against the vendor's legacy
// <t>csrgeam2. B is A^T for a square matrix -- the symmetrisation csrgeam2 is
// usually there for -- and otherwise A's pattern with other values, as
// tests/test_geam_csr.py does. Each timed call is Nnz + csrgeam2 on both sides,
// with C allocated once from the first Nnz.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

#include "baseline/baseline.hpp"
#include "sweep.hpp"

using namespace fstest;

namespace {

BenchReport g_report("geam");

// csrgeam2 wants sorted, duplicate-free rows; sort a copy rather than trust the file.
CsrMatrix sorted_rows(const CsrMatrix& A) {
    CsrMatrix S = A;
    for (int64_t r = 0; r < A.rows; ++r) {
        const int32_t b = A.indptr[r], e = A.indptr[r + 1];
        std::vector<int32_t> order(static_cast<std::size_t>(e - b));
        std::iota(order.begin(), order.end(), b);
        std::sort(order.begin(), order.end(),
                  [&](int32_t x, int32_t y) { return A.indices[x] < A.indices[y]; });
        for (int32_t i = 0; i < e - b; ++i) {
            S.indices[b + i] = A.indices[order[i]];
            S.values[b + i] = A.values[order[i]];
        }
    }
    return S;
}

CsrMatrix transpose(const CsrMatrix& A) {
    CsrMatrix T;
    T.rows = A.cols;
    T.cols = A.rows;
    T.nnz = A.nnz;
    T.indptr.assign(static_cast<std::size_t>(A.cols) + 1, 0);
    for (int32_t c : A.indices) ++T.indptr[static_cast<std::size_t>(c) + 1];
    for (int64_t c = 0; c < A.cols; ++c) T.indptr[c + 1] += T.indptr[c];
    T.indices.assign(static_cast<std::size_t>(A.nnz), 0);
    T.values.assign(static_cast<std::size_t>(A.nnz), 0.0);
    std::vector<int32_t> cursor(T.indptr.begin(), T.indptr.end() - 1);
    for (int64_t r = 0; r < A.rows; ++r) {
        for (int32_t p = A.indptr[r]; p < A.indptr[r + 1]; ++p) {
            const int32_t slot = cursor[A.indices[p]]++;
            T.indices[slot] = static_cast<int32_t>(r);
            T.values[slot] = A.values[p];
        }
    }
    return T;  // rows come out sorted: source rows are visited in order
}

// alpha * A + beta * B as a CSR merge, in fp64.
CsrMatrix merge(const CsrMatrix& A, const CsrMatrix& B, double alpha, double beta) {
    CsrMatrix C;
    C.rows = A.rows;
    C.cols = A.cols;
    C.indptr.push_back(0);
    for (int64_t r = 0; r < A.rows; ++r) {
        int32_t p = A.indptr[r], q = B.indptr[r];
        const int32_t pe = A.indptr[r + 1], qe = B.indptr[r + 1];
        while (p < pe || q < qe) {
            const int32_t ca = p < pe ? A.indices[p] : INT32_MAX;
            const int32_t cb = q < qe ? B.indices[q] : INT32_MAX;
            const int32_t c = std::min(ca, cb);
            double v = 0.0;
            if (ca == c) v += alpha * A.values[p++];
            if (cb == c) v += beta * B.values[q++];
            C.indices.push_back(c);
            C.values.push_back(v);
        }
        C.indptr.push_back(static_cast<int32_t>(C.indices.size()));
    }
    C.nnz = static_cast<int64_t>(C.indices.size());
    return C;
}

}  // namespace

TEST(GeamBenchmark, CsrOverCorpus) {
    Handle handle;
    ASSERT_NE(handle.h, nullptr);
    report_corpus_failures(g_report, "csr");
    const Scalars sc;
    for (const auto& entry : corpus()) {
        const CsrMatrix A = sorted_rows(entry.A);
        CsrMatrix B;
        if (A.rows == A.cols) {
            B = transpose(A);
        } else {
            B = A;
            for (std::size_t i = 0; i < B.values.size(); ++i) B.values[i] = 0.5 + 0.25 * (i % 7);
        }
        const CsrMatrix ref = merge(A, B, 1.0, 1.0);
        for (const registry::Variant* v : variants_of("geam")) {
            if (!benchmark_variant_selected(*v)) continue;
            const auto dt = v->dt;
            BenchRow row;
            row.name = std::string("geam_csr_") + v->dtype + "_" + entry.name;
            row.tag("operator", v->op).tag("matrix", entry.name).tag("format", v->format)
               .tag("dtype", v->dtype).tag("corpus", corpus_tag()).tag("reporting", v->reporting)
               .num("rows", static_cast<double>(A.rows)).num("cols", static_cast<double>(A.cols))
               .num("nnz", static_cast<double>(A.nnz));
            if (v->variant_id) row.tag("variant", v->variant_id);
            trace("geam", entry.name, v->dtype, A);

            DeviceBuffer a_ptr = DeviceBuffer::from(A.indptr), a_idx = DeviceBuffer::from(A.indices);
            DeviceBuffer b_ptr = DeviceBuffer::from(B.indptr), b_idx = DeviceBuffer::from(B.indices);
            DeviceBuffer a_val = upload_as(A.values, dt), b_val = upload_as(B.values, dt);
            DeviceBuffer c_ptr(static_cast<std::size_t>(A.rows + 1) * sizeof(int32_t));
            DeviceBuffer c_idx(static_cast<std::size_t>(std::max<int64_t>(ref.nnz, 1)) * sizeof(int32_t));
            DeviceBuffer c_val(static_cast<std::size_t>(std::max<int64_t>(ref.nnz, 1)) * elem_bytes(dt));
            if (!a_ptr.get() || !a_idx.get() || !b_ptr.get() || !b_idx.get() || !a_val.get() ||
                !b_val.get() || !c_ptr.get() || !c_idx.get() || !c_val.get()) {
                g_report.skip(std::move(row), "skipped_memory", "device allocation failed");
                continue;
            }
            const int m = static_cast<int>(A.rows), n = static_cast<int>(A.cols);
            const int nnz_a = static_cast<int>(A.nnz), nnz_b = static_cast<int>(B.nnz);
            const auto* ap = static_cast<const int*>(a_ptr.get());
            const auto* ai = static_cast<const int*>(a_idx.get());
            const auto* bp = static_cast<const int*>(b_ptr.get());
            const auto* bi = static_cast<const int*>(b_idx.get());
            std::size_t bytes = 0;
            flagsparseCsrgeam2_bufferSizeExt(handle.h, m, n, sc.alpha(dt), nnz_a, a_val.get(), ap,
                                             ai, sc.alpha(dt), nnz_b, b_val.get(), bp, bi,
                                             c_val.get(), static_cast<const int*>(c_ptr.get()),
                                             static_cast<const int*>(c_idx.get()), dt, &bytes);
            DeviceBuffer work(bytes ? bytes : 1);

            baseline::DeviceCsr bA{a_ptr.get(), a_idx.get(), a_val.get(), A.rows, A.cols, A.nnz, dt};
            baseline::DeviceCsr bB{b_ptr.get(), b_idx.get(), b_val.get(), B.rows, B.cols, B.nnz, dt};
            g_report.measure_vs_baseline(
                std::move(row),
                [&]() {
                    int nnz_c = 0;
                    flagsparseStatus_t st = flagsparseXcsrgeam2Nnz(
                        handle.h, m, n, nnz_a, ap, ai, nnz_b, bp, bi,
                        static_cast<int*>(c_ptr.get()), &nnz_c, work.get());
                    if (st != FLAGSPARSE_STATUS_SUCCESS) return st;
                    if (nnz_c != ref.nnz) return FLAGSPARSE_STATUS_EXECUTION_FAILED;
                    return flagsparseCsrgeam2(handle.h, m, n, sc.alpha(dt), nnz_a, a_val.get(), ap,
                                              ai, sc.alpha(dt), nnz_b, b_val.get(), bp, bi,
                                              c_val.get(), static_cast<const int*>(c_ptr.get()),
                                              static_cast<int*>(c_idx.get()), dt, work.get());
                },
                [&](bool relaxed) {
                    // Structure first: a wrong pattern has no meaningful value error.
                    const auto ptr = c_ptr.download<int32_t>(static_cast<std::size_t>(A.rows + 1));
                    const auto idx = c_idx.download<int32_t>(static_cast<std::size_t>(ref.nnz));
                    if (ptr != ref.indptr || idx != ref.indices) return 1e30;
                    return ratio_against(c_val.get(), ref.values, dt, relaxed);
                },
                [&](baseline::Timing* t) {
                    return baseline::csrgeam2(bA, bB, sc.alpha(dt), sc.alpha(dt), c_ptr.get(),
                                              c_idx.get(), c_val.get(), ref.nnz,
                                              BenchReport::kWarmup, BenchReport::kIters, t);
                },
                static_cast<double>(A.nnz + B.nnz));
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
