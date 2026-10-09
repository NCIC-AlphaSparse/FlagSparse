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

// SpMV over the real-matrix corpus, against the vendor baseline.
//
// One row per (matrix, dtype): our median, the vendor's median, the error ratio
// of our answer against a CPU fp64 oracle, and -- only when that answer passed --
// the speedup. See ctest/corpus.hpp for where the matrices come from and
// ctest/baseline/baseline.hpp for what the denominator is.

#include <gtest/gtest.h>

#include <cstdlib>
#include <vector>

#include "baseline/baseline.hpp"
#include "sweep.hpp"

using namespace fstest;

namespace {

BenchReport g_report("spmv");

struct CscMatrix {
    std::vector<int32_t> colptr, rowind;
    std::vector<double> values;
};

struct SellView {
    std::vector<int32_t> offsets, cols;
    std::vector<double> values;
};

SellView csr_to_sell(const CsrMatrix& A, int64_t slice_size) {
    SellView out;
    const int64_t slices = (A.rows + slice_size - 1) / slice_size;
    out.offsets.assign(static_cast<std::size_t>(slices) + 1, 0);
    for (int64_t s = 0; s < slices; ++s) {
        std::size_t width = 0;
        for (int64_t lane = 0; lane < slice_size; ++lane) {
            const int64_t r = s * slice_size + lane;
            if (r < A.rows) {
                width = std::max(width, static_cast<std::size_t>(
                    A.indptr[static_cast<std::size_t>(r) + 1] - A.indptr[static_cast<std::size_t>(r)]));
            }
        }
        for (std::size_t slot = 0; slot < width; ++slot) {
            for (int64_t lane = 0; lane < slice_size; ++lane) {
                const int64_t r = s * slice_size + lane;
                const int32_t begin = r < A.rows ? A.indptr[static_cast<std::size_t>(r)] : 0;
                const int32_t end = r < A.rows ? A.indptr[static_cast<std::size_t>(r) + 1] : 0;
                const bool present = r < A.rows && begin + static_cast<int32_t>(slot) < end;
                out.cols.push_back(present ? A.indices[static_cast<std::size_t>(begin + slot)] : -1);
                out.values.push_back(present ? A.values[static_cast<std::size_t>(begin + slot)] : 0.0);
            }
        }
        out.offsets[static_cast<std::size_t>(s) + 1] = static_cast<int32_t>(out.cols.size());
    }
    return out;
}

CscMatrix csr_to_csc(const CsrMatrix& A) {
    CscMatrix C;
    C.colptr.assign(static_cast<std::size_t>(A.cols) + 1, 0);
    for (int32_t c : A.indices) ++C.colptr[static_cast<std::size_t>(c) + 1];
    for (int64_t c = 0; c < A.cols; ++c)
        C.colptr[static_cast<std::size_t>(c) + 1] += C.colptr[static_cast<std::size_t>(c)];
    C.rowind.assign(static_cast<std::size_t>(A.nnz), 0);
    C.values.assign(static_cast<std::size_t>(A.nnz), 0.0);
    std::vector<int32_t> cursor(C.colptr.begin(), C.colptr.end() - 1);
    for (int64_t r = 0; r < A.rows; ++r) {
        for (int32_t p = A.indptr[static_cast<std::size_t>(r)];
             p < A.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
            const int32_t c = A.indices[static_cast<std::size_t>(p)];
            const std::size_t slot = static_cast<std::size_t>(cursor[c]++);
            C.rowind[slot] = static_cast<int32_t>(r);
            C.values[slot] = A.values[static_cast<std::size_t>(p)];
        }
    }
    return C;
}

std::vector<double> typed_spmv_reference(const CsrMatrix& A, const std::vector<double>& x,
                                         flagsparseDataType_t input_dt,
                                         flagsparseDataType_t accum_dt,
                                         bool csc, bool sell) {
    std::vector<double> y(static_cast<std::size_t>(A.rows), 0.0);
    const CscMatrix C = csr_to_csc(A);
    const SellView S = csr_to_sell(A, 8);
    if (csc) {
        for (int64_t col = 0; col < A.cols; ++col) {
            for (int32_t p = C.colptr[static_cast<std::size_t>(col)];
                 p < C.colptr[static_cast<std::size_t>(col) + 1]; ++p) {
                const int32_t row = C.rowind[static_cast<std::size_t>(p)];
                const double term = quantize_scalar(C.values[static_cast<std::size_t>(p)], input_dt) *
                                    quantize_scalar(x[static_cast<std::size_t>(col)], input_dt);
                y[static_cast<std::size_t>(row)] = accumulate_typed(
                    y[static_cast<std::size_t>(row)], term, accum_dt);
            }
        }
    } else if (sell) {
        const int64_t slices = (A.rows + 7) / 8;
        for (int64_t s = 0; s < slices; ++s) {
            const int32_t begin = S.offsets[static_cast<std::size_t>(s)];
            const int32_t end = S.offsets[static_cast<std::size_t>(s) + 1];
            for (int32_t p = begin; p < end; ++p) {
                const int64_t lane = (p - begin) % 8;
                const int64_t row = s * 8 + lane;
                const int32_t col = S.cols[static_cast<std::size_t>(p)];
                if (row >= A.rows || col < 0) continue;
                const double term = quantize_scalar(S.values[static_cast<std::size_t>(p)], input_dt) *
                                    quantize_scalar(x[static_cast<std::size_t>(col)], input_dt);
                y[static_cast<std::size_t>(row)] = accumulate_typed(
                    y[static_cast<std::size_t>(row)], term, accum_dt);
            }
        }
    } else {
        for (int64_t row = 0; row < A.rows; ++row) {
            double acc = 0.0;
            for (int32_t p = A.indptr[static_cast<std::size_t>(row)];
                 p < A.indptr[static_cast<std::size_t>(row) + 1]; ++p) {
                const int32_t col = A.indices[static_cast<std::size_t>(p)];
                const double term = quantize_scalar(A.values[static_cast<std::size_t>(p)], input_dt) *
                                    quantize_scalar(x[static_cast<std::size_t>(col)], input_dt);
                acc = accumulate_typed(acc, term, accum_dt);
            }
            y[static_cast<std::size_t>(row)] = acc;
        }
    }
    return y;
}

std::vector<double> transpose_reference(const CsrMatrix& A, const std::vector<double>& x) {
    std::vector<double> y(static_cast<std::size_t>(A.cols), 0.0);
    for (int64_t r = 0; r < A.rows; ++r) {
        for (int32_t p = A.indptr[static_cast<std::size_t>(r)];
             p < A.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
            y[static_cast<std::size_t>(A.indices[static_cast<std::size_t>(p)])] +=
                A.values[static_cast<std::size_t>(p)] * x[static_cast<std::size_t>(r)];
        }
    }
    return y;
}

}  // namespace

TEST(SpmvBenchmark, CsrOverCorpus) {
    Handle handle;
    ASSERT_NE(handle.h, nullptr);
    report_corpus_failures(g_report, "csr");

    const Scalars sc;
    // The variant list comes from conf/operators.yaml via the generated
    // registry, not from a table in this file.
    const auto declared = variants_of("spmv");
    const bool delivery_csr_only = std::getenv("FLAGSPARSE_DELIVERY_CSR_ONLY") != nullptr;

    for (const auto& entry : corpus()) {
        const CsrMatrix& A = entry.A;
        // random_csr and the corpus both emit rows in order, so expanding the
        // row pointer gives a row-sorted COO -- which is what the COO kernels
        // assume and what makes the two formats comparable on one matrix.
        const std::vector<int32_t> coo_rows = coo_row_indices_of(A);
        const CsrMatrix A_half = unit_scaled(A);

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
            const bool is_sell = std::string(v->format) == "sell" ||
                                 std::string(v->format) == "sliced_ell";
            if (!is_csr && !is_coo && !is_csc && !is_sell) {
                report_unimplemented(g_report, *v, "benchmark/test_spmv.cpp has no operand builder");
                continue;
            }
            const auto dt = v->dt;
            // Shadows the outer A: half-precision variants run on the scaled copy.
            const CsrMatrix& A = dtype_is_half(dt) ? A_half : entry.A;
            const auto out_dt = variant_output_dtype(*v);
            const auto x_dt = variant_vector_input_dtype(*v);
            const bool mixed = variant_is_mixed(*v);
            const std::string vid = v->variant_id ? v->variant_id : "";
            const flagsparseOperation_t op_a = vid.find("_int_conj") != std::string::npos
                ? FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE
                : (vid.find("_int_trans") != std::string::npos
                    ? FLAGSPARSE_OPERATION_TRANSPOSE : FLAGSPARSE_OPERATION_NON_TRANSPOSE);
            const bool trans = op_a != FLAGSPARSE_OPERATION_NON_TRANSPOSE;
            if (is_sell && trans) continue;
            const std::vector<double> x_host = dense_pattern(
                static_cast<std::size_t>(trans ? A.rows : A.cols));
            const std::vector<double> ref = trans
                ? transpose_reference(A, x_host)
                : spmv_reference(A, x_host, 1.0, 0.0,
                                 std::vector<double>(static_cast<std::size_t>(A.rows), 0.0));
            std::vector<double> ref_variant = ref;
            if (!trans && (is_csc || is_sell || dtype_is_half(dt)) && !variant_is_mixed(*v)) {
                const flagsparseDataType_t accum_dt =
                    out_dt == FLAGSPARSE_R_32I ? FLAGSPARSE_R_32I :
                    // The native fp16 kernels widen products and accumulate in
                    // fp32; only the stored output is rounded back to fp16.
                    (dtype_is_half(dt) ? FLAGSPARSE_R_32F :
                     (dtype_is_64(out_dt) ? FLAGSPARSE_R_64F : FLAGSPARSE_R_32F));
                ref_variant = typed_spmv_reference(A, x_host, x_dt, accum_dt, is_csc, is_sell);
                quantize_vector_inplace(&ref_variant, out_dt);
            } else if (!trans && dtype_is_int8(dt)) {
                CsrMatrix Aq = A;
                for (double& value : Aq.values) value = quantize_scalar(value, dt);
                std::vector<double> xq = x_host;
                for (double& value : xq) value = quantize_scalar(value, x_dt);
                ref_variant = spmv_reference(
                    Aq, xq, 1.0, 0.0,
                    std::vector<double>(static_cast<std::size_t>(A.rows), 0.0));
                quantize_vector_inplace(&ref_variant, out_dt);
            }
            if (!trans && variant_is_mixed(*v) && !is_csc && !is_sell) {
                CsrMatrix Aq = A;
                for (double& value : Aq.values) value = quantize_scalar(value, dt);
                std::vector<double> xq = x_host;
                for (double& value : xq) value = quantize_scalar(value, x_dt);
                ref_variant = spmv_reference(
                    Aq, xq, 1.0, 0.0,
                    std::vector<double>(static_cast<std::size_t>(A.rows), 0.0));
            }
            BenchRow row;
            row.name = std::string("spmv_") + v->format + "_" + v->dtype + "_" +
                       entry.name;
            row.tag("operator", v->op)
               .tag("matrix", entry.name).tag("format", v->format).tag("dtype", v->dtype)
               .tag("corpus", corpus_tag()).tag("reporting", v->reporting)
               .num("rows", static_cast<double>(A.rows))
               .num("cols", static_cast<double>(A.cols))
               .num("nnz", static_cast<double>(A.nnz));
            if (v->variant_id) row.tag("variant", v->variant_id);
            trace("spmv", entry.name, v->dtype, A);

            const CscMatrix csc = csr_to_csc(A);
            const SellView sell = csr_to_sell(A, 8);
            DeviceBuffer indptr = DeviceBuffer::from(is_csc ? csc.colptr :
                                                     (is_sell ? sell.offsets : A.indptr));
            DeviceBuffer indices = DeviceBuffer::from(is_csc ? csc.rowind :
                                                      (is_sell ? sell.cols : A.indices));
            DeviceBuffer rowind = DeviceBuffer::from(coo_rows);
            DeviceBuffer values = upload_as(is_csc ? csc.values :
                                            (is_sell ? sell.values : A.values), dt);
            DeviceBuffer x = upload_as(x_host, x_dt);
            DeviceBuffer y(static_cast<std::size_t>(trans ? A.cols : A.rows) * elem_bytes(out_dt));
            if (!indptr.get() || !indices.get() || (!is_sell && !rowind.get()) || !values.get() ||
                !x.get() || !y.get()) {
                // Out of memory on this matrix is a fact about the matrix; the
                // next one may well fit, so record and carry on.
                g_report.skip(std::move(row), "skipped_memory",
                              "device allocation failed for this matrix");
                continue;
            }

            flagsparseSpMatDescr_t matA = nullptr;
            flagsparseDnVecDescr_t vecX = nullptr, vecY = nullptr;
            flagsparseStatus_t cs = is_coo
                ? flagsparseCreateCoo(&matA, A.rows, A.cols, A.nnz, rowind.get(),
                                      indices.get(), values.get(), FLAGSPARSE_INDEX_32I,
                                      FLAGSPARSE_INDEX_BASE_ZERO, dt)
                : is_csc
                ? flagsparseCreateCsc(&matA, A.rows, A.cols, A.nnz, indptr.get(),
                                      indices.get(), values.get(), FLAGSPARSE_INDEX_32I,
                                      FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO, dt)
                : is_sell
                ? flagsparseCreateSlicedEll(&matA, A.rows, A.cols, 8, indptr.get(),
                                            indices.get(), values.get(), FLAGSPARSE_INDEX_32I,
                                            FLAGSPARSE_INDEX_BASE_ZERO, dt)
                : flagsparseCreateCsr(&matA, A.rows, A.cols, A.nnz, indptr.get(),
                                       indices.get(), values.get(), FLAGSPARSE_INDEX_32I,
                                       FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO, dt);
            if (cs != FLAGSPARSE_STATUS_SUCCESS) {
                g_report.skip(std::move(row), "failed",
                              std::string("descriptor creation failed for ") +
                                  v->format);
                continue;
            }
            flagsparseCreateDnVec(&vecX, trans ? A.rows : A.cols, x.get(), x_dt);
            flagsparseCreateDnVec(&vecY, trans ? A.cols : A.rows, y.get(), out_dt);

            // Scratch comes from bufferSize, never a guess, and is allocated
            // outside the timed region so the allocator never lands in a sample.
            std::size_t bufsz = 0;
            flagsparseSpMV_bufferSize(handle.h, op_a,
                                      sc.alpha(out_dt), matA, vecX, sc.beta(out_dt), vecY, out_dt,
                                      FLAGSPARSE_SPMV_ALG_DEFAULT, &bufsz);
            DeviceBuffer scratch(bufsz ? bufsz : 1);
            if (!scratch.get()) {
                flagsparseDestroyDnVec(vecX); flagsparseDestroyDnVec(vecY);
                flagsparseDestroySpMat(matA);
                g_report.skip(std::move(row), "skipped_memory",
                              "SpMV scratch allocation failed");
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
                    return flagsparseSpMV(handle.h, op_a,
                                          sc.alpha(out_dt), matA, vecX, sc.beta(out_dt), vecY,
                                          out_dt, is_sell ? FLAGSPARSE_SPMV_SELL_ALG1 :
                                          FLAGSPARSE_SPMV_ALG_DEFAULT, scratch.get());
                },
                [&](bool relaxed) {
                    return out_dt == FLAGSPARSE_R_16F
                        ? benchmark_ratio_against(y.get(), ref_variant, out_dt, relaxed)
                        : ratio_against(y.get(), ref_variant, out_dt, relaxed);
                },
                [&](baseline::Timing* t) {
                    if (is_sell || mixed
#if !defined(FLAGSPARSE_MUSA_BASELINE_EXTENSIONS)
                        || is_csc
#endif
                    ) {
                        return baseline::Status::no(
                            is_sell ? "no matching cuSPARSE SELL SpMV baseline"
                            : (mixed ? "mixed-precision SpMV has no matching vendor baseline"
                                     : "CSC SpMV baseline is not wired in harness"));
                    }
                    return baseline::spmv_csr(bA, x.get(), y.get(), sc.alpha(out_dt),
                                              sc.beta(out_dt), op_a, BenchReport::kWarmup,
                                              BenchReport::kIters, t);
                },
                2.0 * static_cast<double>(A.nnz));

            flagsparseDestroyDnVec(vecX);
            flagsparseDestroyDnVec(vecY);
            flagsparseDestroySpMat(matA);
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
