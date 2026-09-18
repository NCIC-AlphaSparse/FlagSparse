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

// The vendor baseline a speedup column is measured against.
//
// WHAT THIS IS FOR. A speedup number is only worth printing if the thing in the
// denominator is the library a user would otherwise have called. So the baseline
// is the vendor's own sparse library -- cuSPARSE on CUDA, hipSPARSE on DCU --
// driven through its generic (descriptor) API, the same one flagsparse's C
// header is modelled on.
//
// WHAT IT IS NOT FOR. The baseline is a TIMING reference, never the correctness
// oracle. Accuracy is judged against a CPU fp64 evaluation of the same matrix
// (see spmv_reference/spmm_reference in common.hpp), because a vendor kernel and
// ours can be wrong in the same direction and agreeing with each other would
// then read as a pass. The baseline's own agreement with that oracle IS checked,
// and recorded separately as `baseline_accurate`: a speedup over a baseline that
// did not compute the right answer is not a speedup, and the JSON says so
// instead of quietly averaging it in.
//
// AVAILABILITY IS A RESULT, NOT AN ERROR. Every entry point returns Status, and
// `unavailable` carries a reason string that names what is missing. Three things
// legitimately produce it:
//   - the backend has no vendor generic-sparse library wired (see CMakeLists)
//   - the library has it but declines this configuration (hipSPARSE's SpMM is
//     op=non only, so the transposed direction has no baseline on DCU)
//   - the matrix does not fit the operator (A*A needs a square A)
// None of these is a test failure. They produce a row with a blank speedup and a
// reason, which is the honest rendering of "not comparable here".

#ifndef FLAGSPARSE_CTEST_BASELINE_HPP
#define FLAGSPARSE_CTEST_BASELINE_HPP

#include <cstdint>
#include <string>
#include <vector>

#include "flagsparse.h"

namespace fstest::baseline {

// Whether a vendor baseline is compiled in at all, and what it is called.
// `name()` is what the JSON and the console banner print in the baseline column,
// e.g. "cuSPARSE 12.5" or "none".
bool available();
const char* name();

struct Status {
    bool ok = false;
    // Set when !ok. Names what is missing, in the terms of the thing that is
    // missing -- "hipsparseSpMM: op(A)=transpose is not implemented" beats
    // "unsupported".
    std::string reason;

    static Status good() { return {true, {}}; }
    static Status no(std::string why) { return {false, std::move(why)}; }
};

// A CSR operand already resident on the device. The baseline never allocates or
// frees these: the caller owns them, which keeps the allocator out of the timed
// region for both sides of the comparison.
struct DeviceCsr {
    void* indptr = nullptr;   // int32, rows + 1  (CSR only)
    void* indices = nullptr;  // int32, nnz -- column indices in both formats
    void* values = nullptr;   // `dtype`, nnz
    int64_t rows = 0, cols = 0, nnz = 0;
    flagsparseDataType_t dtype = FLAGSPARSE_R_32F;

    // COO. When non-null the operand is built as a COO descriptor from
    // (coo_rows, indices, values) and `indptr` is ignored.
    //
    // ONE STRUCT RATHER THAN TWO because the eight entry points differ only in
    // how the operand descriptor is created; duplicating each signature per
    // format would double an interface to express a property of one field. A
    // vendor that does not implement an operator on COO -- cuSPARSE's SpSV is
    // CSR and sliced-ELL only -- simply returns its own status, which becomes a
    // blank speedup with that reason rather than a missing row.
    void* coo_rows = nullptr;  // int32, nnz

    bool is_coo() const { return coo_rows != nullptr; }
};

// One timed run. `iters` samples are taken after `warmup` untimed ones and the
// median is returned, matching BenchReport so the two numbers are commensurable.
struct Timing {
    double median_ms = 0.0;
};

// ---------------------------------------------------------------- entry points
//
// Each mirrors the flagsparse call it is the denominator for. `y`/`C` are both
// the output and, on return, hold the baseline's result so the caller can check
// it against the CPU oracle.
//
// alpha/beta are read at the operand's own width, as cuSPARSE requires: a double
// passed to an fp32 op reads as garbage, which is how a baseline silently
// computes alpha=0 and looks 1000x faster than it is.

Status spmv_csr(const DeviceCsr& A, const void* x, void* y, const void* alpha,
                const void* beta, flagsparseOperation_t op, int warmup, int iters,
                Timing* out);

Status spmm_csr(const DeviceCsr& A, const void* B, int64_t n, int64_t ldb, void* C,
                int64_t ldc, const void* alpha, const void* beta,
                flagsparseOperation_t opA, flagsparseOperation_t opB, int warmup,
                int iters, Timing* out);

Status sddmm_csr(const DeviceCsr& A, const void* Bd, int64_t k, int64_t ldb,
                 const void* Cd, int64_t ldc, const void* alpha, const void* beta,
                 int warmup, int iters, Timing* out);

// C's arrays as the baseline produced them. Allocated by the baseline because
// their size is DISCOVERED -- the caller cannot size them in advance, which is
// what makes SpGEMM different from every other entry point here.
//
// Ownership passes to the caller when `result` is non-null below; free it with
// free_csr(). The point of handing it over is accuracy: without it the vendor's
// answer is unreachable, `baseline_accuracy` stays "unchecked" forever, and spec
// 6.3.1's relaxed tier -- which requires evidence that the VENDOR also missed the
// strict tolerance -- can never be awarded for this operator.
struct BaselineCsrOut {
    void* indptr = nullptr;
    void* indices = nullptr;
    void* values = nullptr;
    int64_t rows = 0, cols = 0, nnz = 0;
};

void free_csr(BaselineCsrOut* c);

// C = A * A. Pass `result` to take ownership of C and check the vendor's answer;
// pass nullptr to have the baseline free it (only the timing escapes).
Status spgemm_csr(const DeviceCsr& A, const void* alpha, const void* beta,
                  int warmup, int iters, Timing* out,
                  BaselineCsrOut* result = nullptr);

Status spsv_csr(const DeviceCsr& A, const void* x, void* y, const void* alpha,
                flagsparseFillMode_t fill, flagsparseDiagType_t diag,
                flagsparseOperation_t op, int warmup, int iters, Timing* out);

Status spsm_csr(const DeviceCsr& A, const void* B, int64_t n, int64_t ldb, void* C,
                int64_t ldc, const void* alpha, flagsparseFillMode_t fill,
                flagsparseDiagType_t diag, flagsparseOperation_t op, int warmup,
                int iters, Timing* out);

// Sparse-vector gather/scatter. `nnz` indices into a dense vector of `size`.
Status gather(const void* dense, void* sparse_val, const void* sparse_idx,
              int64_t nnz, int64_t size, flagsparseDataType_t dtype, int warmup,
              int iters, Timing* out);

Status scatter(void* dense, const void* sparse_val, const void* sparse_idx,
               int64_t nnz, int64_t size, flagsparseDataType_t dtype, int warmup,
               int iters, Timing* out);

}  // namespace fstest::baseline

#endif  // FLAGSPARSE_CTEST_BASELINE_HPP
