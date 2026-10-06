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

// SpMM over the real-matrix corpus, against the vendor baseline.
//
// n (the dense width) is swept as well as the dtype, because SpMM's cost is not
// linear in it: a narrow B is bandwidth-bound on A, a wide one starts to reuse
// the A row it loaded and the ratio against a vendor kernel moves accordingly.
// Reporting one n would hide whichever end we happen to lose.

#include <gtest/gtest.h>

#include <cstdlib>
#include <vector>

#include "baseline/baseline.hpp"
#include "sweep.hpp"

using namespace fstest;

namespace {

BenchReport g_report("spmm");

// Column-major B and C, which is what ORDER_COL means to both libraries: ld is
// the row count of the operand.
constexpr int64_t kWidths[] = {8, 32, 128};

// MUSA's delivery scope is n=8 only. It skips the wider cases rather than
// narrowing the array, which would drop them on every other backend too.
bool musa_backend() {
    static const bool is_musa = std::string(flagsparseGetBackendName()) == "musa";
    return is_musa;
}

std::vector<double> pack_dense(const std::vector<double>& src, int64_t rows, int64_t cols,
                               bool transpose, flagsparseOrder_t order) {
    const int64_t dr = transpose ? cols : rows, dc = transpose ? rows : cols;
    const int64_t ld = order == FLAGSPARSE_ORDER_ROW ? dc : dr;
    std::vector<double> dst(static_cast<std::size_t>(dr * dc), 0.0);
    for (int64_t r = 0; r < rows; ++r) for (int64_t c = 0; c < cols; ++c) {
        const int64_t pr = transpose ? c : r, pc = transpose ? r : c;
        dst[static_cast<std::size_t>(order == FLAGSPARSE_ORDER_ROW ? pr * ld + pc
                                                                     : pr + pc * ld)] =
            src[static_cast<std::size_t>(r * cols + c)];
    }
    return dst;
}

struct CscMatrix {
    std::vector<int32_t> colptr, rowind;
    std::vector<double> values;
};

CscMatrix csr_to_csc(const CsrMatrix& A) {
    CscMatrix out;
    out.colptr.assign(static_cast<std::size_t>(A.cols) + 1, 0);
    for (int32_t c : A.indices) ++out.colptr[static_cast<std::size_t>(c) + 1];
    for (int64_t c = 0; c < A.cols; ++c)
        out.colptr[static_cast<std::size_t>(c) + 1] += out.colptr[static_cast<std::size_t>(c)];
    out.rowind.assign(static_cast<std::size_t>(A.nnz), 0);
    out.values.assign(static_cast<std::size_t>(A.nnz), 0.0);
    std::vector<int32_t> cursor(out.colptr.begin(), out.colptr.end() - 1);
    for (int64_t r = 0; r < A.rows; ++r) {
        for (int32_t p = A.indptr[static_cast<std::size_t>(r)];
             p < A.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
            const int32_t c = A.indices[static_cast<std::size_t>(p)];
            const std::size_t slot = static_cast<std::size_t>(cursor[c]++);
            out.rowind[slot] = static_cast<int32_t>(r);
            out.values[slot] = A.values[static_cast<std::size_t>(p)];
        }
    }
    return out;
}

std::vector<double> typed_spmm_reference(const CsrMatrix& A,
                                         const std::vector<double>& B, int64_t n,
                                         flagsparseDataType_t input_dt,
                                         flagsparseDataType_t accum_dt, bool csc) {
    std::vector<double> C(static_cast<std::size_t>(A.rows * n), 0.0);
    const CscMatrix S = csr_to_csc(A);
    auto add = [&](int64_t row, int64_t col, double term) {
        const std::size_t idx = static_cast<std::size_t>(row * n + col);
        C[idx] = accumulate_typed(C[idx], term, accum_dt);
    };
    if (csc) {
        for (int64_t k = 0; k < A.cols; ++k) {
            for (int32_t p = S.colptr[static_cast<std::size_t>(k)];
                 p < S.colptr[static_cast<std::size_t>(k) + 1]; ++p) {
                const int64_t row = S.rowind[static_cast<std::size_t>(p)];
                const double a = quantize_scalar(S.values[static_cast<std::size_t>(p)], input_dt);
                for (int64_t j = 0; j < n; ++j) {
                    add(row, j, a * quantize_scalar(B[static_cast<std::size_t>(k * n + j)], input_dt));
                }
            }
        }
    } else {
        for (int64_t row = 0; row < A.rows; ++row) {
            for (int32_t p = A.indptr[static_cast<std::size_t>(row)];
                 p < A.indptr[static_cast<std::size_t>(row) + 1]; ++p) {
                const int64_t k = A.indices[static_cast<std::size_t>(p)];
                const double a = quantize_scalar(A.values[static_cast<std::size_t>(p)], input_dt);
                for (int64_t j = 0; j < n; ++j) {
                    add(row, j, a * quantize_scalar(B[static_cast<std::size_t>(k * n + j)], input_dt));
                }
            }
        }
    }
    return C;
}

}  // namespace

TEST(SpmmBenchmark, CsrOverCorpus) {
    Handle handle;
    ASSERT_NE(handle.h, nullptr);
    report_corpus_failures(g_report, "csr");

    const Scalars sc;
    // Variant list from conf/operators.yaml via the generated registry.
    const auto declared = variants_of("spmm");
    const bool delivery_csr_only = std::getenv("FLAGSPARSE_DELIVERY_CSR_ONLY") != nullptr;

    for (const auto& entry : corpus()) {
        const CsrMatrix& A = entry.A;
        // Row-sorted COO, expanded from the row pointer, so both formats measure
        // the same matrix rather than two different orderings of it.
        const std::vector<int32_t> coo_rows = coo_row_indices_of(A);
        for (const int64_t n : kWidths) {
            if (musa_backend() && n != kWidths[0]) continue;
            // B column-major (cols x n); the oracle wants it row-major (k x n),
            // so the reference is built from the same numbers in the other order.
            const std::size_t bcount = static_cast<std::size_t>(A.cols) *
                                       static_cast<std::size_t>(n);
            const std::vector<double> b_col = dense_pattern(bcount);
            std::vector<double> b_row(bcount);
            for (int64_t c = 0; c < A.cols; ++c) {
                for (int64_t j = 0; j < n; ++j) {
                    b_row[static_cast<std::size_t>(c) * n + j] =
                        b_col[static_cast<std::size_t>(j) * A.cols + c];
                }
            }
            const std::vector<double> ref_row = spmm_reference(
                A, b_row, n, 1.0, 0.0,
                std::vector<double>(static_cast<std::size_t>(A.rows) * n, 0.0));
            // Our C is column-major; transpose the oracle to match so the two are
            // compared element for element rather than by luck of the layout.
            std::vector<double> ref(ref_row.size());
            for (int64_t r = 0; r < A.rows; ++r) {
                for (int64_t j = 0; j < n; ++j) {
                    ref[static_cast<std::size_t>(j) * A.rows + r] =
                        ref_row[static_cast<std::size_t>(r) * n + j];
                }
            }

            for (const registry::Variant* v : declared) {
            if (!benchmark_variant_selected(*v)) continue;
                if (delivery_csr_only &&
                    (std::string(v->format) != "csr" ||
                     (std::string(v->dtype) != "f32" && std::string(v->dtype) != "f64"))) {
                    continue;
                }
                const bool is_coo = std::string(v->format) == "coo";
                const bool is_csr = std::string(v->format) == "csr";
                const bool is_csc = std::string(v->format) == "csc";
                if (!is_csr && !is_coo && !is_csc) {
                    if (n == kWidths[0]) report_unimplemented(
                        g_report, *v, "benchmark/test_spmm.cpp has no operand builder");
                    continue;
                }
                const auto dt = v->dt;
                const auto out_dt = q4_output_dtype(*v);
                const bool mixed = q4_is_mixed(*v);
                const std::string vid = v->q4_variant ? v->q4_variant : "";
                const bool row_layout = vid.find("_row") != std::string::npos;
                const bool trans_a = vid.find("_int_trans_non_") != std::string::npos;
                const bool trans_b = vid.find("_int_non_trans_") != std::string::npos;
                if (trans_a && !is_csr) continue;
                const int64_t b_logical_rows = trans_a ? A.rows : A.cols;
                const int64_t out_rows = trans_a ? A.cols : A.rows;
                const std::vector<double> b_row_variant =
                    trans_a ? dense_pattern(static_cast<std::size_t>(b_logical_rows * n))
                            : b_row;
                std::vector<double> ref_row_variant;
                if (trans_a) {
                    ref_row_variant.assign(static_cast<std::size_t>(out_rows * n), 0.0);
                    for (int64_t r = 0; r < A.rows; ++r) {
                        for (int32_t p = A.indptr[static_cast<std::size_t>(r)];
                             p < A.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
                            const int64_t c = A.indices[static_cast<std::size_t>(p)];
                            for (int64_t j = 0; j < n; ++j) {
                                ref_row_variant[static_cast<std::size_t>(c * n + j)] +=
                                    A.values[static_cast<std::size_t>(p)] *
                                    b_row_variant[static_cast<std::size_t>(r * n + j)];
                            }
                        }
                    }
                } else {
                    if (is_csc || (dtype_is_half(dt) && !mixed)) {
                        const flagsparseDataType_t accum_dt =
                            out_dt == FLAGSPARSE_R_32I ? FLAGSPARSE_R_32I :
                            // Native fp16 SpMM widens inputs and accumulates in
                            // fp32 before storing the fp16 output.
                            (dtype_is_half(dt) ? FLAGSPARSE_R_32F :
                             (dtype_is_64(out_dt) ? FLAGSPARSE_R_64F : FLAGSPARSE_R_32F));
                        ref_row_variant = typed_spmm_reference(
                            A, b_row_variant, n, dt, accum_dt, is_csc);
                        quantize_vector_inplace(&ref_row_variant, out_dt);
                    } else if (mixed) {
                        CsrMatrix Aq = A;
                        for (double& value : Aq.values) value = quantize_scalar(value, dt);
                        std::vector<double> bq = b_row_variant;
                        for (double& value : bq) value = quantize_scalar(value, dt);
                        ref_row_variant = spmm_reference(
                            Aq, bq, n, 1.0, 0.0,
                            std::vector<double>(static_cast<std::size_t>(A.rows * n), 0.0));
                    } else if (dtype_is_half(dt) || dtype_is_int8(dt)) {
                        CsrMatrix Aq = A;
                        for (double& value : Aq.values) value = quantize_scalar(value, dt);
                        std::vector<double> bq = b_row_variant;
                        for (double& value : bq) value = quantize_scalar(value, dt);
                        ref_row_variant = spmm_reference(
                            Aq, bq, n, 1.0, 0.0,
                            std::vector<double>(static_cast<std::size_t>(A.rows * n), 0.0));
                        quantize_vector_inplace(&ref_row_variant, out_dt);
                    } else {
                        ref_row_variant = ref_row;
                    }
                }
                BenchRow row;
                row.name = std::string("spmm_") + v->format + "_" + v->dtype +
                           "_n" + std::to_string(n) + "_" + entry.name;
                row.tag("operator", v->op)
                   .tag("matrix", entry.name).tag("format", v->format)
                   .tag("dtype", v->dtype).tag("corpus", corpus_tag()).tag("reporting", v->reporting)
                   .num("rows", static_cast<double>(A.rows))
                   .num("cols", static_cast<double>(A.cols))
                   .num("nnz", static_cast<double>(A.nnz))
                   .num("n", static_cast<double>(n));
                if (v->q4_variant) row.tag("q4_variant", v->q4_variant);
                trace("spmm", entry.name, v->dtype, A);

                const CscMatrix csc = csr_to_csc(A);
                DeviceBuffer indptr = DeviceBuffer::from(is_csc ? csc.colptr : A.indptr);
                DeviceBuffer indices = DeviceBuffer::from(is_csc ? csc.rowind : A.indices);
                DeviceBuffer rowind = DeviceBuffer::from(coo_rows);
                DeviceBuffer values = upload_as(is_csc ? csc.values : A.values, dt);
                const flagsparseOrder_t order = row_layout ? FLAGSPARSE_ORDER_ROW
                                                            : FLAGSPARSE_ORDER_COL;
                const auto q4_b = pack_dense(b_row_variant, b_logical_rows, n, trans_b, order);
                const auto q4_ref = pack_dense(ref_row_variant, out_rows, n, false, order);
                DeviceBuffer B = upload_as(v->q4_variant ? q4_b : b_col, dt);
                DeviceBuffer C(static_cast<std::size_t>(out_rows) *
                               static_cast<std::size_t>(n) * elem_bytes(out_dt));
                if (!indptr.get() || !indices.get() ||
                    (!is_csc && !rowind.get()) ||
                    !values.get() || !B.get() || !C.get()) {
                    g_report.skip(std::move(row), "skipped_memory",
                                  "device allocation failed for this matrix at n=" +
                                      std::to_string(n));
                    continue;
                }

                flagsparseSpMatDescr_t matA = nullptr;
                flagsparseDnMatDescr_t matB = nullptr, matC = nullptr;
                const flagsparseStatus_t cs =
                    is_coo
                        ? flagsparseCreateCoo(&matA, A.rows, A.cols, A.nnz,
                                              rowind.get(), indices.get(), values.get(),
                                              FLAGSPARSE_INDEX_32I,
                                              FLAGSPARSE_INDEX_BASE_ZERO, dt)
                        : is_csc
                        ? flagsparseCreateCsc(&matA, A.rows, A.cols, A.nnz, indptr.get(),
                                              indices.get(), values.get(),
                                              FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
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
                const int64_t b_rows = trans_b ? n : b_logical_rows;
                const int64_t b_cols = trans_b ? b_logical_rows : n;
                const int64_t b_ld = order == FLAGSPARSE_ORDER_ROW ? b_cols : b_rows;
                flagsparseCreateDnMat(&matB, b_rows, b_cols, b_ld, B.get(), dt, order);
                flagsparseCreateDnMat(&matC, out_rows, n, row_layout ? n : out_rows, C.get(), out_dt,
                                      order);

                std::size_t bufsz = 0;
                flagsparseSpMM_bufferSize(handle.h,
                                          trans_a ? FLAGSPARSE_OPERATION_TRANSPOSE : FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                          trans_b ? FLAGSPARSE_OPERATION_TRANSPOSE : FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                          sc.alpha(out_dt), matA, matB, sc.beta(out_dt), matC,
                                          out_dt, FLAGSPARSE_SPMM_ALG_DEFAULT, &bufsz);
                DeviceBuffer scratch(bufsz ? bufsz : 1);
                if (!scratch.get()) {
                    flagsparseDestroyDnMat(matB); flagsparseDestroyDnMat(matC);
                    flagsparseDestroySpMat(matA);
                    g_report.skip(std::move(row), "skipped_memory",
                                  "SpMM scratch allocation failed");
                    continue;
                }

                baseline::DeviceCsr bA{indptr.get(), indices.get(), values.get(),
                                       A.rows, A.cols, A.nnz, dt,
                                       is_coo ? rowind.get() : nullptr};
#if defined(FLAGSPARSE_MUSA_BASELINE_EXTENSIONS)
                if (is_csc) bA.format = baseline::DeviceCsr::Format::Csc;
#endif
                g_report.measure_vs_baseline(
                    std::move(row),
                    [&]() {
                        return flagsparseSpMM(handle.h,
                                              trans_a ? FLAGSPARSE_OPERATION_TRANSPOSE : FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                              trans_b ? FLAGSPARSE_OPERATION_TRANSPOSE : FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                                              sc.alpha(out_dt), matA, matB, sc.beta(out_dt),
                                              matC, out_dt, FLAGSPARSE_SPMM_ALG_DEFAULT,
                                              scratch.get());
                    },
                    [&](bool relaxed) {
                        return out_dt == FLAGSPARSE_R_16F
                            ? benchmark_ratio_against(C.get(), v->q4_variant ? q4_ref : ref,
                                                      out_dt, relaxed)
                            : ratio_against(C.get(), v->q4_variant ? q4_ref : ref,
                                            out_dt, relaxed);
                    },
                    [&](baseline::Timing* t) {
                        if (mixed
#if !defined(FLAGSPARSE_MUSA_BASELINE_EXTENSIONS)
                            || is_csc
#endif
                        ) {
                            return baseline::Status::no(
                                mixed ? "mixed-precision SpMM has no matching vendor baseline"
                                      : "no matching cuSPARSE CSC SpMM baseline in harness");
                        }
                        // Pass the exact descriptors and layouts to the vendor
                        // library so its result overwrites the verify() buffer.
                        const int64_t b_ld = order == FLAGSPARSE_ORDER_ROW ? b_cols : b_rows;
                        const int64_t c_ld = order == FLAGSPARSE_ORDER_ROW ? n : out_rows;
                        return baseline::spmm_csr(
                            bA, B.get(), n, b_ld, C.get(), c_ld, sc.alpha(out_dt),
                            sc.beta(out_dt),
                            trans_a ? FLAGSPARSE_OPERATION_TRANSPOSE
                                    : FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                            trans_b ? FLAGSPARSE_OPERATION_TRANSPOSE
                                     : FLAGSPARSE_OPERATION_NON_TRANSPOSE,
                            BenchReport::kWarmup, BenchReport::kIters, t, order, order);
                    },
                    2.0 * static_cast<double>(A.nnz) * static_cast<double>(n));

                flagsparseDestroyDnMat(matB);
                flagsparseDestroyDnMat(matC);
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
