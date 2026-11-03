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

// csrgeam2: C = alpha * A + beta * B, all CSR, cuSPARSE's three-call shape.
//
// The algorithm is src/flagsparse/sparse_operations/geam_csr.py's, with the same
// probe and place kernels; what that function does with torch between them (two
// prefix sums and the row lengths) runs here as the device kernels its C API
// section adds. Nothing is sorted: every slot of C comes from ranks.
//
//   Nnz       probe B's entries against A's rows -> scan the duplicate flags ->
//             C's row lengths -> scan them into csrRowPtrC, read the total back
//   csrgeam2  place A's entries, then B's (a duplicate adds into A's slot)
//
// The workspace carries the probe results from the first call to the second,
// which is why cuSPARSE asks for the same buffer in both.

#include <algorithm>
#include <climits>
#include <cstdint>
#include <string>
#include <vector>

#include "adaptor/adaptor.hpp"
#include "core/internal.hpp"
#include "core/jit.hpp"

using namespace flagsparse;

namespace {

constexpr int kScanBlock = 1024;
// The block totals are scanned by one program of at most this many lanes.
constexpr int64_t kMaxScanBlocks = 65536;
constexpr int kRowBlock = 256;

int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }
std::size_t align_up(std::size_t v) { return (v + 255u) & ~static_cast<std::size_t>(255u); }

int next_pow2(int64_t v) {
    int out = 16;
    while (out < v) out *= 2;
    return out;
}

// Iterations of a binary search over a row of at most n columns.
int search_iters(int64_t n) {
    int bits = 0;
    while ((int64_t{1} << bits) <= n) ++bits;
    return bits;
}

// geam_csr.py's _row_tile.
void row_tile(int64_t nnz, int64_t rows, int* rows_per, int* block) {
    const double mean = rows > 0 ? static_cast<double>(nnz) / static_cast<double>(rows) : 0.0;
    if (mean <= 8.0)       { *rows_per = 32; *block = 8; }
    else if (mean <= 32.0) { *rows_per = 8;  *block = 32; }
    else                   { *rows_per = 2;  *block = 128; }
}

struct Workspace {
    int32_t* lb_a = nullptr;        // nnzB: A columns smaller than each B entry's
    int32_t* dup = nullptr;         // nnzB: 1 where B's column is also in A
    int32_t* dup_prefix = nullptr;  // nnzB + 1: exclusive scan of dup
    int32_t* sums = nullptr;        // scan block totals (+ the grand total)
    int32_t* c_len = nullptr;       // m: C's row lengths
};

std::size_t sums_count(int64_t m, int64_t nnz_b) {
    return static_cast<std::size_t>(cdiv(std::max<int64_t>({m, nnz_b, 1}), kScanBlock)) + 1;
}

std::size_t workspace_bytes(int64_t m, int64_t nnz_b) {
    const std::size_t i = sizeof(int32_t);
    return align_up(static_cast<std::size_t>(nnz_b) * i) * 2 +
           align_up(static_cast<std::size_t>(nnz_b + 1) * i) +
           align_up(sums_count(m, nnz_b) * i) + align_up(static_cast<std::size_t>(m) * i);
}

Workspace carve(void* buffer, int64_t m, int64_t nnz_b) {
    auto* p = static_cast<unsigned char*>(buffer);
    const std::size_t i = sizeof(int32_t);
    Workspace w;
    w.lb_a = reinterpret_cast<int32_t*>(p);
    p += align_up(static_cast<std::size_t>(nnz_b) * i);
    w.dup = reinterpret_cast<int32_t*>(p);
    p += align_up(static_cast<std::size_t>(nnz_b) * i);
    w.dup_prefix = reinterpret_cast<int32_t*>(p);
    p += align_up(static_cast<std::size_t>(nnz_b + 1) * i);
    w.sums = reinterpret_cast<int32_t*>(p);
    p += align_up(sums_count(m, nnz_b) * i);
    w.c_len = reinterpret_cast<int32_t*>(p);
    return w;
}

bool supported_type(flagsparseDataType_t t) {
    return t == FLAGSPARSE_R_32F || t == FLAGSPARSE_R_64F || t == FLAGSPARSE_C_32F ||
           t == FLAGSPARSE_C_64F;
}

jit::Arg ptr(const void* p) {
    return jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(const_cast<void*>(p)));
}

flagsparseStatus_t launch(flagsparseHandle_t handle, const char* kernel, const std::string& sig,
                          int64_t grid, const std::vector<jit::Arg>& args) {
    std::string err;
    const flagsparseStatus_t st = jit::launch(jit::codegen_module("geam_csr.py"), kernel, sig,
                                              ctx(handle)->stream, grid, 1, 1, /*num_warps=*/4,
                                              /*num_stages=*/2, args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}

// out[0..n] = exclusive prefix sum of x[0..n), out[n] = the total.
flagsparseStatus_t exclusive_scan(flagsparseHandle_t handle, const int32_t* x, int32_t* out,
                                  int64_t n, int32_t* sums) {
    if (n == 0) {
        return adaptor::memset_device(reinterpret_cast<adaptor::DevicePtr>(out), 0,
                                      sizeof(int32_t));
    }
    const int64_t blocks = cdiv(n, kScanBlock);
    if (blocks > kMaxScanBlocks) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    const std::string b = std::to_string(kScanBlock);
    if (auto s = launch(handle, "_geam_scan_blocks_kernel", "*i32:16,*i32:16,*i32:16,i32," + b,
                        blocks, {ptr(x), ptr(out), ptr(sums), jit::Arg::i(static_cast<int32_t>(n))}))
        return s;
    if (auto s = launch(handle, "_geam_scan_sums_kernel",
                        "*i32:16,i32," + std::to_string(next_pow2(blocks)), 1,
                        {ptr(sums), jit::Arg::i(static_cast<int32_t>(blocks))}))
        return s;
    return launch(handle, "_geam_scan_add_kernel", "*i32:16,*i32:16,i32,i32," + b, blocks,
                  {ptr(out), ptr(sums), jit::Arg::i(static_cast<int32_t>(n)),
                   jit::Arg::i(static_cast<int32_t>(blocks))});
}

flagsparseStatus_t check_shape(flagsparseHandle_t handle, int m, int n, int nnzA, int nnzB) {
    if (handle == nullptr) return FLAGSPARSE_STATUS_NOT_INITIALIZED;
    if (m < 0 || n < 0 || nnzA < 0 || nnzB < 0) return FLAGSPARSE_STATUS_INVALID_VALUE;
    // C's nnz is at most nnzA + nnzB and has to fit csrRowPtrC's int32.
    if (static_cast<int64_t>(nnzA) + nnzB > INT32_MAX) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    return FLAGSPARSE_STATUS_SUCCESS;
}

// Host scalars in the value type; complex ones are (re, im) pairs.
void read_scalar(flagsparseDataType_t t, const void* p, double* re, double* im) {
    *im = 0.0;
    if (component_dtype(t) == FLAGSPARSE_R_64F) {
        const auto* d = static_cast<const double*>(p);
        *re = d[0];
        if (is_complex(t)) *im = d[1];
    } else {
        const auto* f = static_cast<const float*>(p);
        *re = f[0];
        if (is_complex(t)) *im = f[1];
    }
}

}  // namespace

extern "C" {

flagsparseStatus_t flagsparseCsrgeam2_bufferSizeExt(
    flagsparseHandle_t handle, int m, int n, const void* alpha, int nnzA,
    const void* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const void* beta, int nnzB, const void* csrValB, const int* csrRowPtrB,
    const int* csrColIndB, const void* csrValC, const int* csrRowPtrC,
    const int* csrColIndC, flagsparseDataType_t valueType, size_t* bufferSize) {
    (void)alpha; (void)csrValA; (void)csrRowPtrA; (void)csrColIndA; (void)beta;
    (void)csrValB; (void)csrRowPtrB; (void)csrColIndB; (void)csrValC; (void)csrRowPtrC;
    (void)csrColIndC;
    if (bufferSize == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *bufferSize = 0;
    return guard(handle, [&]() -> flagsparseStatus_t {
        if (auto s = check_shape(handle, m, n, nnzA, nnzB)) return s;
        if (!supported_type(valueType)) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
        *bufferSize = workspace_bytes(m, nnzB);
        return FLAGSPARSE_STATUS_SUCCESS;
    });
}

flagsparseStatus_t flagsparseXcsrgeam2Nnz(
    flagsparseHandle_t handle, int m, int n, int nnzA, const int* csrRowPtrA,
    const int* csrColIndA, int nnzB, const int* csrRowPtrB, const int* csrColIndB,
    int* csrRowPtrC, int* nnzTotalDevHostPtr, void* workspace) {
    return guard(handle, [&]() -> flagsparseStatus_t {
        if (auto s = check_shape(handle, m, n, nnzA, nnzB)) return s;
        if (csrRowPtrC == nullptr || nnzTotalDevHostPtr == nullptr || workspace == nullptr ||
            (m > 0 && (csrRowPtrA == nullptr || csrRowPtrB == nullptr))) {
            return FLAGSPARSE_STATUS_INVALID_VALUE;
        }
        const Workspace w = carve(workspace, m, nnzB);
        if (m > 0 && nnzB > 0) {
            int rows_per = 32, block = 8;
            row_tile(nnzB, m, &rows_per, &block);
            const std::string sig = "*i32:16,*i32:16,*i32:16,*i32:16,*i32:16,*i32:16,i32,i32," +
                                    std::to_string(rows_per) + "," + std::to_string(block);
            if (auto s = launch(handle, "_geam_probe_kernel", sig, cdiv(m, rows_per),
                                {ptr(csrColIndB), ptr(csrRowPtrB), ptr(csrColIndA), ptr(csrRowPtrA),
                                 ptr(w.lb_a), ptr(w.dup), jit::Arg::i(m),
                                 jit::Arg::i(search_iters(n))}))
                return s;
        }
        if (auto s = exclusive_scan(handle, w.dup, w.dup_prefix, nnzB, w.sums)) return s;
        if (m > 0) {
            if (auto s = launch(handle, "_geam_row_nnz_kernel",
                                "*i32:16,*i32:16,*i32:16,*i32:16,i32," + std::to_string(kRowBlock),
                                cdiv(m, kRowBlock),
                                {ptr(csrRowPtrA), ptr(csrRowPtrB), ptr(w.dup_prefix), ptr(w.c_len),
                                 jit::Arg::i(m)}))
                return s;
        }
        if (auto s = exclusive_scan(handle, w.c_len, csrRowPtrC, m, w.sums)) return s;

        // csrRowPtrC[m] is C's nnz; the caller sizes C's arrays from it.
        adaptor::synchronize();
        int32_t total = 0;
        if (auto s = adaptor::memcpy_d2h(&total, reinterpret_cast<adaptor::DevicePtr>(
                                                     csrRowPtrC + m), sizeof(total)))
            return s;
        if (ctx(handle)->pointer_mode == FLAGSPARSE_POINTER_MODE_DEVICE) {
            return adaptor::memcpy_h2d(reinterpret_cast<adaptor::DevicePtr>(nnzTotalDevHostPtr),
                                       &total, sizeof(total));
        }
        *nnzTotalDevHostPtr = total;
        return FLAGSPARSE_STATUS_SUCCESS;
    });
}

flagsparseStatus_t flagsparseCsrgeam2(
    flagsparseHandle_t handle, int m, int n, const void* alpha, int nnzA,
    const void* csrValA, const int* csrRowPtrA, const int* csrColIndA,
    const void* beta, int nnzB, const void* csrValB, const int* csrRowPtrB,
    const int* csrColIndB, void* csrValC, const int* csrRowPtrC, int* csrColIndC,
    flagsparseDataType_t valueType, void* pBuffer) {
    return guard(handle, [&]() -> flagsparseStatus_t {
        if (auto s = check_shape(handle, m, n, nnzA, nnzB)) return s;
        if (!supported_type(valueType)) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
        if (alpha == nullptr || beta == nullptr || pBuffer == nullptr || csrRowPtrC == nullptr) {
            return FLAGSPARSE_STATUS_INVALID_VALUE;
        }
        if (ctx(handle)->pointer_mode != FLAGSPARSE_POINTER_MODE_HOST) {
            return FLAGSPARSE_STATUS_NOT_SUPPORTED;
        }
        if (m == 0 || nnzA + nnzB == 0) return FLAGSPARSE_STATUS_SUCCESS;
        const Workspace w = carve(pBuffer, m, nnzB);
        double a_re, a_im, b_re, b_im;
        read_scalar(valueType, alpha, &a_re, &a_im);
        read_scalar(valueType, beta, &b_re, &b_im);
        const bool complex = is_complex(valueType);
        const bool fp64 = component_dtype(valueType) == FLAGSPARSE_R_64F;
        const std::string vt = fp64 ? "fp64" : "fp32";   // complex: interleaved components
        const auto scalar = [&](double v) {
            return fp64 ? jit::Arg::d(v) : jit::Arg::f(static_cast<float>(v));
        };
        const std::string flag = complex ? "True" : "False";

        if (nnzA > 0) {
            int rows_per = 32, block = 8;
            row_tile(nnzA, m, &rows_per, &block);
            const std::string sig = "*i32:16,*i32:16,*" + vt + ":16,*i32:16,*i32:16,*i32:16,"
                                    "*i32:16,*i32:16,*" + vt + ":16," + vt + "," + vt +
                                    ",i32,i32," + flag + "," + std::to_string(rows_per) + "," +
                                    std::to_string(block);
            if (auto s = launch(handle, "_geam_place_a_kernel", sig, cdiv(m, rows_per),
                                {ptr(csrColIndA), ptr(csrRowPtrA), ptr(csrValA), ptr(csrColIndB),
                                 ptr(csrRowPtrB), ptr(w.dup_prefix), ptr(csrRowPtrC),
                                 ptr(csrColIndC), ptr(csrValC), scalar(a_re), scalar(a_im),
                                 jit::Arg::i(m), jit::Arg::i(search_iters(n))}))
                return s;
        }
        if (nnzB > 0) {
            int rows_per = 32, block = 8;
            row_tile(nnzB, m, &rows_per, &block);
            const std::string sig = "*i32:16,*i32:16,*" + vt + ":16,*i32:16,*i32:16,*i32:16,"
                                    "*i32:16,*i32:16,*" + vt + ":16," + vt + "," + vt + ",i32," +
                                    flag + "," + std::to_string(rows_per) + "," +
                                    std::to_string(block);
            if (auto s = launch(handle, "_geam_place_b_kernel", sig, cdiv(m, rows_per),
                                {ptr(csrColIndB), ptr(csrRowPtrB), ptr(csrValB), ptr(w.lb_a),
                                 ptr(w.dup), ptr(w.dup_prefix), ptr(csrRowPtrC), ptr(csrColIndC),
                                 ptr(csrValC), scalar(b_re), scalar(b_im), jit::Arg::i(m)}))
                return s;
        }
        return FLAGSPARSE_STATUS_SUCCESS;
    });
}

}  // extern "C"
