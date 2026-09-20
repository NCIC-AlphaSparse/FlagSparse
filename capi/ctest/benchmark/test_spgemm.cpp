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

// SpGEMM (C = A*A) over the real-matrix corpus, against the vendor baseline.
//
// THE ORACLE IS INDIRECT, ON PURPOSE. Materialising A*A in fp64 on the host costs
// more memory than the product itself for these matrices. But if C = A*A then
// C*x == A*(A*x) for any x, and both sides of that are host SpMVs over vectors of
// length rows -- cheap, and sensitive to a wrong value anywhere in C that x does
// not happen to annihilate. A fixed non-degenerate x is used for the same reason
// the dense operands elsewhere are fixed.
//
// `compute` IS TIMED, NOT `copy`. Both sides.
//
// On MUSA the current Triton fill artifact has an ABI/codegen defect, so copy
// materializes the validated CSR result on the host. That setup is intentionally
// outside the timing loop; compute remains the C API device phase being measured.
//
// So the timed region is the iterative numeric phase, `compute`, on both sides.
// The setup (workEstimation, the sizing compute, the allocation, one copy) runs
// once outside the clock, which also leaves C populated for the accuracy check.
//
// REAL DTYPES ONLY, matching the kernel's coverage.

#include <gtest/gtest.h>

#include <vector>

#include "baseline/baseline.hpp"
#include "sweep.hpp"

using namespace fstest;

namespace {

BenchReport g_report("spgemm");

// Reading C back and multiplying costs memory proportional to its nonzeros. Past
// this the row is measured but recorded accuracy="unchecked", which withholds the
// speedup rather than quietly averaging an unverified ratio into the geomean.
constexpr int64_t kVerifyNnzBudget = 20'000'000;

std::vector<double> host_spmv(const CsrMatrix& A, const std::vector<double>& x) {
    std::vector<double> y(static_cast<std::size_t>(A.rows), 0.0);
    for (int64_t r = 0; r < A.rows; ++r) {
        double acc = 0.0;
        for (int32_t p = A.indptr[static_cast<std::size_t>(r)];
             p < A.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
            acc += A.values[static_cast<std::size_t>(p)] *
                   x[static_cast<std::size_t>(A.indices[static_cast<std::size_t>(p)])];
        }
        y[static_cast<std::size_t>(r)] = acc;
    }
    return y;
}

}  // namespace

TEST(SpgemmBenchmark, CsrOverCorpus) {
    Handle handle;
    ASSERT_NE(handle.h, nullptr);
    report_corpus_failures(g_report, "csr");

    const Scalars sc;
    const auto declared = variants_of("spgemm");
    for (const auto& entry : corpus()) {
        const CsrMatrix& A = entry.A;
        if (A.rows != A.cols) {
            // One row PER VARIANT, not one for the matrix. A single untagged row
            // carries no dtype and no `reporting`, so every consumer downstream
            // has to guess which variants it stands for -- and a summary that
            // defaults the guess to "delivery" quietly inflates its own totals.
            for (const registry::Variant* v : declared) {
                BenchRow row;
                row.name = std::string("spgemm_csr_") + v->dtype + "_" + entry.name;
                row.tag("operator", v->op).tag("matrix", entry.name)
                   .tag("format", v->format).tag("dtype", v->dtype)
                   .tag("corpus", corpus_tag()).tag("reporting", v->reporting);
                g_report.skip(std::move(row), "skipped_shape",
                              "A*A needs a square A");
            }
            continue;
        }
        const std::vector<double> x = dense_pattern(static_cast<std::size_t>(A.cols));
        const std::vector<double> ref = host_spmv(A, host_spmv(A, x));  // A*(A*x)

        for (const registry::Variant* v : declared) {
            if (std::string(v->format) != "csr") {
                const std::string why =
                    std::string("benchmark/test_spgemm.cpp has no ") + v->format +
                    " operand builder yet";
                report_unimplemented(g_report, *v, why.c_str());
                continue;
            }
            const auto dt = v->dt;
            BenchRow row;
            row.name = std::string("spgemm_csr_") + v->dtype + "_" + entry.name;
            row.tag("operator", v->op)
               .tag("matrix", entry.name).tag("format", "csr").tag("dtype", v->dtype)
               .tag("corpus", corpus_tag()).tag("reporting", v->reporting)
               .num("rows", static_cast<double>(A.rows))
               .num("nnz", static_cast<double>(A.nnz));
            trace("spgemm", entry.name, v->dtype, A);

            DeviceBuffer indptr = DeviceBuffer::from(A.indptr);
            DeviceBuffer indices = DeviceBuffer::from(A.indices);
            DeviceBuffer values = upload_as(A.values, dt);
            DeviceBuffer c_ptr(static_cast<std::size_t>(A.rows + 1) * sizeof(int32_t));
            if (!indptr.get() || !indices.get() || !values.get() || !c_ptr.get()) {
                g_report.skip(std::move(row), "skipped_memory",
                              "device allocation failed for this matrix");
                continue;
            }

            const auto NT = FLAGSPARSE_OPERATION_NON_TRANSPOSE;
            flagsparseSpMatDescr_t matA = nullptr, matB = nullptr, matC = nullptr;
            flagsparseCreateCsr(&matA, A.rows, A.cols, A.nnz, indptr.get(),
                                indices.get(), values.get(), FLAGSPARSE_INDEX_32I,
                                FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO, dt);
            flagsparseCreateCsr(&matB, A.rows, A.cols, A.nnz, indptr.get(),
                                indices.get(), values.get(), FLAGSPARSE_INDEX_32I,
                                FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_BASE_ZERO, dt);
            flagsparseCreateCsr(&matC, A.rows, A.cols, 0, c_ptr.get(), nullptr, nullptr,
                                FLAGSPARSE_INDEX_32I, FLAGSPARSE_INDEX_32I,
                                FLAGSPARSE_INDEX_BASE_ZERO, dt);
            flagsparseSpGEMMDescr_t descr = nullptr;
            flagsparseSpGEMM_createDescr(&descr);
            auto teardown = [&]() {
                flagsparseSpGEMM_destroyDescr(descr);
                flagsparseDestroySpMat(matA); flagsparseDestroySpMat(matB);
                flagsparseDestroySpMat(matC);
            };

            std::size_t b1 = 0, b2 = 0;
            flagsparseSpGEMM_workEstimation(handle.h, NT, NT, sc.alpha(dt), matA, matB,
                                            sc.beta(dt), matC, dt,
                                            FLAGSPARSE_SPGEMM_DEFAULT, descr, &b1,
                                            nullptr);
            DeviceBuffer s1(b1 ? b1 : 1);
            flagsparseStatus_t st = flagsparseSpGEMM_workEstimation(
                handle.h, NT, NT, sc.alpha(dt), matA, matB, sc.beta(dt), matC, dt,
                FLAGSPARSE_SPGEMM_DEFAULT, descr, &b1, s1.get());
            if (st == FLAGSPARSE_STATUS_SUCCESS) {
                flagsparseSpGEMM_compute(handle.h, NT, NT, sc.alpha(dt), matA, matB,
                                         sc.beta(dt), matC, dt,
                                         FLAGSPARSE_SPGEMM_DEFAULT, descr, &b2, nullptr);
            }
            DeviceBuffer s2(b2 ? b2 : 1);
            bool host_fallback = false;
            if (st == FLAGSPARSE_STATUS_SUCCESS) {
                st = flagsparseSpGEMM_compute(handle.h, NT, NT, sc.alpha(dt), matA, matB,
                                              sc.beta(dt), matC, dt,
                                              FLAGSPARSE_SPGEMM_DEFAULT, descr, &b2,
                                              s2.get());
                const char* note = nullptr;
                flagsparseGetLastErrorString(handle.h, &note);
                host_fallback = note && std::string(note).rfind(
                    "SpGEMM host_fallback:", 0) == 0;
            }
            if (st != FLAGSPARSE_STATUS_SUCCESS) {
                const char* why = nullptr;
                flagsparseGetLastErrorString(handle.h, &why);
                teardown();
                // The hash-table overflow path lands here. It is a capability
                // limit of the C wrapper (the Python side falls back to a chunked
                // ESC that the C API does not reach), so it is recorded with that
                // reason and the sweep continues.
                g_report.skip(std::move(row),
                              st == FLAGSPARSE_STATUS_NOT_SUPPORTED ? "not_supported"
                                                                    : "failed",
                              std::string("SpGEMM compute declined this matrix") +
                                  (why && *why ? std::string(": ") + why : ""));
                continue;
            }

            int64_t cr = 0, cc = 0, cnnz = 0;
            flagsparseSpMatGetSize(matC, &cr, &cc, &cnnz);
            DeviceBuffer c_ind(static_cast<std::size_t>(cnnz) * sizeof(int32_t));
            DeviceBuffer c_val(static_cast<std::size_t>(cnnz) * elem_bytes(dt));
            if (cnnz > 0 && (!c_ind.get() || !c_val.get())) {
                teardown();
                g_report.skip(std::move(row), "skipped_memory",
                              "C allocation failed (" + std::to_string(cnnz) +
                                  " nonzeros)");
                continue;
            }
            flagsparseCsrSetPointers(matC, c_ptr.get(), c_ind.get(), c_val.get());
            row.num("c_nnz", static_cast<double>(cnnz));
            row.tag("execution", host_fallback ? "host_fallback" : "device_hash");

            baseline::DeviceCsr bA{indptr.get(), indices.get(), values.get(),
                                   A.rows, A.cols, A.nnz, dt};

            // Which C the oracle reads. It starts as ours; the baseline lambda
            // repoints it at the vendor's once that has run, which is what lets
            // baseline_accuracy be filled and spec 6.3.1's relaxed tier apply
            // here at all. Without this the vendor's answer is unreachable and
            // the tier is permanently out of reach for SpGEMM.
            struct CView { const void* ptr; const void* ind; const void* val;
                           int64_t rows, cols, nnz; };
            CView active{c_ptr.get(), c_ind.get(), c_val.get(), cr, cc, cnnz};
            baseline::BaselineCsrOut vendor_c;
            // One copy outside the clock: it materialises C so the oracle below
            // has something to read, and it is not part of what we measure.
            const flagsparseStatus_t cp =
                flagsparseSpGEMM_copy(handle.h, NT, NT, sc.alpha(dt), matA, matB,
                                      sc.beta(dt), matC, dt,
                                      FLAGSPARSE_SPGEMM_DEFAULT, descr);
            if (cp != FLAGSPARSE_STATUS_SUCCESS) {
                const char* why = nullptr;
                flagsparseGetLastErrorString(handle.h, &why);
                teardown();
                g_report.skip(std::move(row), "failed",
                              std::string("SpGEMM_copy failed") +
                                  (why && *why ? std::string(": ") + why : ""));
                continue;
            }
            g_report.measure_vs_baseline(
                std::move(row),
                [&]() {
                    return flagsparseSpGEMM_compute(handle.h, NT, NT, sc.alpha(dt),
                                                    matA, matB, sc.beta(dt), matC, dt,
                                                    FLAGSPARSE_SPGEMM_DEFAULT, descr,
                                                    &b2, s2.get());
                },
                [&](bool relaxed) -> double {
                    if (active.nnz > kVerifyNnzBudget) return -1.0;  // unchecked
                    CsrMatrix C;
                    C.rows = active.rows; C.cols = active.cols; C.nnz = active.nnz;
                    C.indptr.resize(static_cast<std::size_t>(active.rows) + 1);
                    C.indices.resize(static_cast<std::size_t>(active.nnz));
                    if (to_host(C.indptr.data(), active.ptr,
                                C.indptr.size() * sizeof(int32_t)) !=
                        FLAGSPARSE_STATUS_SUCCESS) return 1e30;
                    if (active.nnz > 0 &&
                        to_host(C.indices.data(), active.ind,
                                C.indices.size() * sizeof(int32_t)) !=
                            FLAGSPARSE_STATUS_SUCCESS) return 1e30;
                    C.values = read_back(active.val,
                                         static_cast<std::size_t>(active.nnz), dt);
                    if (active.nnz > 0 && C.values.empty()) return 1e30;
                    return max_error_ratio(host_spmv(C, x), ref,
                                           relaxed ? relaxed_tolerance(dt)
                                                   : default_tolerance(dt));
                },
                [&](baseline::Timing* t) {
                    const baseline::Status s = baseline::spgemm_csr(
                        bA, sc.alpha(dt), sc.beta(dt),
                        host_fallback ? 0 : BenchReport::kWarmup,
                        host_fallback ? 1 : BenchReport::kIters, t, &vendor_c);
                    // The oracle reads the vendor's C from here on. Its nnz is
                    // its own: two implementations of A*A agree mathematically,
                    // but one may keep explicit zeros the other drops, and
                    // reading the vendor's array at OUR length would be a
                    // buffer overrun dressed up as a precision result.
                    if (s.ok && vendor_c.values) {
                        active = CView{vendor_c.indptr, vendor_c.indices,
                                       vendor_c.values, vendor_c.rows,
                                       vendor_c.cols, vendor_c.nnz};
                    }
                    return s;
                },
                0.0);
            baseline::free_csr(&vendor_c);
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
