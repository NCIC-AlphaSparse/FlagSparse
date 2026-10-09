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

// SDDMM over the real-matrix corpus, against the vendor baseline.
//
// C_sparse = alpha * (B @ D) .* spy(A) + beta * C_sparse: A supplies the pattern
// and receives the result, so the output to check is A's value array, not a dense
// buffer. k is swept because SDDMM's arithmetic intensity is linear in it while
// its sparse traffic is not.
//
// REAL DTYPES ONLY. FlagSparse has no complex SDDMM kernel yet, so c32/c64 are
// absent here rather than present and failing -- an operator that was never
// implemented is not a regression, and a `failed` row would read as one. See
// docs/README.md for the standing gap list.

#include <gtest/gtest.h>

#include <vector>

#include "baseline/baseline.hpp"
#include "sweep.hpp"

using namespace fstest;

namespace {

BenchReport g_report("sddmm");

constexpr int64_t kInner[] = {16, 64};

// The fp64 oracle for the nonzeros A holds: for each stored (r, c), the dot
// product of B's row r with D's column c.
std::vector<double> sddmm_reference(const CsrMatrix& A, const std::vector<double>& B,
                                    const std::vector<double>& D, int64_t k) {
    std::vector<double> out(static_cast<std::size_t>(A.nnz), 0.0);
    for (int64_t r = 0; r < A.rows; ++r) {
        for (int32_t p = A.indptr[static_cast<std::size_t>(r)];
             p < A.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
            const int32_t c = A.indices[static_cast<std::size_t>(p)];
            double acc = 0.0;
            for (int64_t i = 0; i < k; ++i) {
                acc += B[static_cast<std::size_t>(r) * k + i] *
                       D[static_cast<std::size_t>(i) * A.cols + c];
            }
            out[static_cast<std::size_t>(p)] = acc;
        }
    }
    return out;
}

std::vector<double> pack_dense(const std::vector<double>& logical, int64_t rows,
                               int64_t cols, bool transpose, flagsparseOrder_t order) {
    const int64_t dr = transpose ? cols : rows;
    const int64_t dc = transpose ? rows : cols;
    const int64_t ld = order == FLAGSPARSE_ORDER_ROW ? dc : dr;
    std::vector<double> packed(static_cast<std::size_t>(dr * dc), 0.0);
    for (int64_t r = 0; r < rows; ++r) {
        for (int64_t c = 0; c < cols; ++c) {
            const int64_t pr = transpose ? c : r;
            const int64_t pc = transpose ? r : c;
            const std::size_t dst = static_cast<std::size_t>(
                order == FLAGSPARSE_ORDER_ROW ? pr * ld + pc : pr + pc * ld);
            packed[dst] = logical[static_cast<std::size_t>(r * cols + c)];
        }
    }
    return packed;
}

}  // namespace

TEST(SddmmBenchmark, CsrOverCorpus) {
    Handle handle;
    ASSERT_NE(handle.h, nullptr);
    report_corpus_failures(g_report, "csr");

    const Scalars sc;
    const auto declared = variants_of("sddmm");
    for (const auto& entry : corpus()) {
        const CsrMatrix& A = entry.A;
        for (const int64_t k : kInner) {
            // B is row-major rows x k, D row-major k x cols -- the layout the
            // stride arguments already encode, so no transpose is needed.
            const std::vector<double> Bh =
                dense_pattern(static_cast<std::size_t>(A.rows) * k);
            const std::vector<double> Dh =
                dense_pattern(static_cast<std::size_t>(k) * A.cols, 3.0);
            const std::vector<double> ref = sddmm_reference(A, Bh, Dh, k);

            for (const registry::Variant* v : declared) {
                if (!benchmark_variant_selected(*v)) continue;
                if (std::string(v->format) != "csr") {
                    if (k == kInner[0]) {
                        const std::string why =
                            std::string("benchmark/test_sddmm.cpp has no ") +
                            v->format + " operand builder yet";
                        report_unimplemented(g_report, *v, why.c_str());
                    }
                    continue;
                }
                const auto dt = v->dt;
                const std::string vid = v->variant_id ? v->variant_id : "";
                const bool row_order = vid.find("_row") != std::string::npos;
                const bool trans_a = vid.find("_trans_non_") != std::string::npos;
                const bool trans_b = vid.find("_non_trans_") != std::string::npos;
                const flagsparseOperation_t op_a = trans_a
                    ? FLAGSPARSE_OPERATION_TRANSPOSE : FLAGSPARSE_OPERATION_NON_TRANSPOSE;
                const flagsparseOperation_t op_b = trans_b
                    ? FLAGSPARSE_OPERATION_TRANSPOSE : FLAGSPARSE_OPERATION_NON_TRANSPOSE;
                BenchRow row;
                row.name = std::string("sddmm_csr_") + v->dtype + "_k" +
                           std::to_string(k) + "_" + entry.name;
                row.tag("operator", v->op)
                   .tag("matrix", entry.name).tag("format", "csr").tag("dtype", v->dtype)
                   .tag("corpus", corpus_tag()).tag("reporting", v->reporting)
                   .num("rows", static_cast<double>(A.rows))
                   .num("cols", static_cast<double>(A.cols))
                   .num("nnz", static_cast<double>(A.nnz))
                   .num("k", static_cast<double>(k));
                if (v->variant_id) row.tag("variant", v->variant_id);
                trace("sddmm", entry.name, v->dtype, A);

                DeviceBuffer indptr = DeviceBuffer::from(A.indptr);
                DeviceBuffer indices = DeviceBuffer::from(A.indices);
                DeviceBuffer values(static_cast<std::size_t>(A.nnz) * elem_bytes(dt));
                const flagsparseOrder_t order = !v->variant_id || row_order
                    ? FLAGSPARSE_ORDER_ROW : FLAGSPARSE_ORDER_COL;
                const std::vector<double> Bpacked = pack_dense(Bh, A.rows, k, trans_a, order);
                const std::vector<double> Dpacked = pack_dense(Dh, k, A.cols, trans_b, order);
                DeviceBuffer B = upload_as(v->variant_id ? Bpacked : Bh, dt);
                DeviceBuffer D = upload_as(v->variant_id ? Dpacked : Dh, dt);
                if (!indptr.get() || !indices.get() || !values.get() || !B.get() ||
                    !D.get()) {
                    g_report.skip(std::move(row), "skipped_memory",
                                  "device allocation failed at k=" + std::to_string(k));
                    continue;
                }

                flagsparseSpMatDescr_t matA = nullptr;
                flagsparseDnMatDescr_t matB = nullptr, matD = nullptr;
                if (flagsparseCreateCsr(&matA, A.rows, A.cols, A.nnz, indptr.get(),
                                        indices.get(), values.get(),
                                        FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                        FLAGSPARSE_INDEX_BASE_ZERO, dt) !=
                    FLAGSPARSE_STATUS_SUCCESS) {
                    g_report.skip(std::move(row), "failed", "flagsparseCreateCsr failed");
                    continue;
                }
                const int64_t b_rows = trans_a ? k : A.rows;
                const int64_t b_cols = trans_a ? A.rows : k;
                const int64_t d_rows = trans_b ? A.cols : k;
                const int64_t d_cols = trans_b ? k : A.cols;
                // `order`, not `row_order`: an untagged sweep row is row-major too.
                const int64_t b_ld = order == FLAGSPARSE_ORDER_ROW ? b_cols : b_rows;
                const int64_t d_ld = order == FLAGSPARSE_ORDER_ROW ? d_cols : d_rows;
                flagsparseCreateDnMat(&matB, b_rows, b_cols, b_ld, B.get(), dt, order);
                flagsparseCreateDnMat(&matD, d_rows, d_cols, d_ld, D.get(), dt, order);

                std::size_t bufsz = 0;
                flagsparseSDDMM_bufferSize(handle.h, op_a, op_b, sc.alpha(dt), matB, matD,
                                           sc.beta(dt), matA, dt,
                                           FLAGSPARSE_SDDMM_ALG_DEFAULT, &bufsz);
                DeviceBuffer scratch(bufsz ? bufsz : 1);
                if (!scratch.get()) {
                    flagsparseDestroyDnMat(matB); flagsparseDestroyDnMat(matD);
                    flagsparseDestroySpMat(matA);
                    g_report.skip(std::move(row), "skipped_memory",
                                  "SDDMM scratch allocation failed");
                    continue;
                }

                baseline::DeviceCsr bA{indptr.get(), indices.get(), values.get(),
                                       A.rows, A.cols, A.nnz, dt};
                g_report.measure_vs_baseline(
                    std::move(row),
                    [&]() {
                        return flagsparseSDDMM(handle.h, op_a, op_b, sc.alpha(dt), matB,
                                               matD, sc.beta(dt), matA, dt,
                                               FLAGSPARSE_SDDMM_ALG_DEFAULT,
                                               scratch.get());
                    },
                    [&](bool relaxed) { return ratio_against(values.get(), ref, dt, relaxed); },
                    [&](baseline::Timing* t) {
#if defined(FLAGSPARSE_MUSA_BASELINE_EXTENSIONS)
                        return baseline::sddmm_csr(bA, B.get(), k, b_ld, D.get(), d_ld,
                                                   sc.alpha(dt), sc.beta(dt),
                                                   op_a, op_b, order, order,
                                                   BenchReport::kWarmup, BenchReport::kIters, t);
#else
                        // This baseline entry point takes neither op nor order: it
                        // computes non/non with row-major B and D, so only that
                        // configuration has a vendor time to compare against.
                        if (trans_a || trans_b || order != FLAGSPARSE_ORDER_ROW) {
                            return baseline::Status::no(
                                "baseline harness has no matching SDDMM op/order path");
                        }
                        return baseline::sddmm_csr(bA, B.get(), k, b_ld, D.get(), d_ld,
                                                   sc.alpha(dt), sc.beta(dt),
                                                   BenchReport::kWarmup, BenchReport::kIters, t);
#endif
                    },
                    2.0 * static_cast<double>(A.nnz) * static_cast<double>(k));

                flagsparseDestroyDnMat(matB);
                flagsparseDestroyDnMat(matD);
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
