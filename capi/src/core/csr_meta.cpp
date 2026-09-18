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


// Matrix-level facts that can only be had by reading the index arrays back to
// the host. Shared rather than per-operator: SpMV sizes its segment loop from
// the longest row and SpMM tunes its launch from it, and two copies of a
// readback would be two places for the caching to go wrong.

#include <algorithm>
#include <climits>
#include <cstdint>
#include <vector>

#include "adaptor/adaptor.hpp"
#include "core/internal.hpp"

namespace flagsparse {

flagsparseStatus_t ensure_max_row_nnz(SpMatDescr* A, int64_t* out) {
    if (A == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    if (A->max_row_nnz >= 0) { *out = A->max_row_nnz; return FLAGSPARSE_STATUS_SUCCESS; }
    // The offsets array indexes rows for CSR and COLUMNS for CSC, so the count
    // is not always A->rows. Reading rows+1 entries out of a cols+1 array is how
    // a CSC matrix with more rows than columns walks off the end.
    const int64_t n = (A->format == FLAGSPARSE_FORMAT_CSC) ? A->cols : A->rows;
    if (n <= 0 || A->nnz == 0) { A->max_row_nnz = 0; *out = 0; return FLAGSPARSE_STATUS_SUCCESS; }

    const std::size_t isize = index_size(A->offsets_type);
    if (isize == 0) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    std::vector<unsigned char> host(static_cast<std::size_t>(n + 1) * isize);
    const flagsparseStatus_t st = adaptor::memcpy_d2h(
        host.data(), reinterpret_cast<adaptor::DevicePtr>(A->offsets), host.size());
    if (st != FLAGSPARSE_STATUS_SUCCESS) return st;

    int64_t longest = 0;
    if (A->offsets_type == FLAGSPARSE_INDEX_32I) {
        const auto* p = reinterpret_cast<const std::int32_t*>(host.data());
        for (int64_t r = 0; r < n; ++r) longest = std::max<int64_t>(longest, p[r + 1] - p[r]);
    } else {
        const auto* p = reinterpret_cast<const std::int64_t*>(host.data());
        for (int64_t r = 0; r < n; ++r) longest = std::max<int64_t>(longest, p[r + 1] - p[r]);
    }
    A->max_row_nnz = longest;
    *out = longest;
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t build_coo_row_offsets(SpMatDescr* A, void* buffer) {
    if (A == nullptr || buffer == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    const int64_t m = A->rows;
    const int64_t nnz = A->nnz;
    if (m < 0) return FLAGSPARSE_STATUS_INVALID_VALUE;

    std::vector<std::int32_t> offsets(static_cast<std::size_t>(m + 1), 0);
    if (nnz > 0) {
        const std::size_t isize = index_size(A->indices_type);
        if (isize == 0) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
        // The kernels index the value array with int32 arithmetic, so an nnz
        // past that range would wrap rather than merely be slow.
        if (nnz > static_cast<int64_t>(INT32_MAX)) return FLAGSPARSE_STATUS_NOT_SUPPORTED;

        std::vector<unsigned char> host(static_cast<std::size_t>(nnz) * isize);
        const flagsparseStatus_t st = adaptor::memcpy_d2h(
            host.data(), reinterpret_cast<adaptor::DevicePtr>(A->row_ind), host.size());
        if (st != FLAGSPARSE_STATUS_SUCCESS) return st;

        int64_t previous = 0;
        for (int64_t k = 0; k < nnz; ++k) {
            const int64_t r = (A->indices_type == FLAGSPARSE_INDEX_32I)
                                  ? reinterpret_cast<const std::int32_t*>(host.data())[k]
                                  : reinterpret_cast<const std::int64_t*>(host.data())[k];
            if (r < 0 || r >= m) return FLAGSPARSE_STATUS_INVALID_VALUE;
            if (r < previous) return FLAGSPARSE_STATUS_INVALID_VALUE;  // not row-sorted
            previous = r;
            ++offsets[static_cast<std::size_t>(r) + 1];
        }
        for (int64_t r = 0; r < m; ++r) {
            offsets[static_cast<std::size_t>(r) + 1] += offsets[static_cast<std::size_t>(r)];
        }
        A->max_row_nnz = 0;
        for (int64_t r = 0; r < m; ++r) {
            A->max_row_nnz = std::max<int64_t>(
                A->max_row_nnz,
                offsets[static_cast<std::size_t>(r) + 1] - offsets[static_cast<std::size_t>(r)]);
        }
    } else {
        A->max_row_nnz = 0;
    }

    const flagsparseStatus_t st = adaptor::memcpy_h2d(
        reinterpret_cast<adaptor::DevicePtr>(buffer), offsets.data(),
        offsets.size() * sizeof(std::int32_t));
    if (st != FLAGSPARSE_STATUS_SUCCESS) return st;
    A->coo_offsets_buffer = buffer;
    return FLAGSPARSE_STATUS_SUCCESS;
}

// The triangular-solve kernels walk a row and stop at the diagonal, so an
// unsorted row stops early and silently drops terms. cuSPARSE requires sorted
// CSR too; checking costs one readback in analysis, which is once per matrix.
flagsparseStatus_t check_csr_columns_sorted(const SpMatDescr* A) {
    if (A == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    const int64_t n = A->rows;
    if (n <= 0 || A->nnz == 0) return FLAGSPARSE_STATUS_SUCCESS;

    const std::size_t osize = index_size(A->offsets_type);
    const std::size_t isize = index_size(A->indices_type);
    if (osize == 0 || isize == 0) return FLAGSPARSE_STATUS_NOT_SUPPORTED;

    std::vector<unsigned char> off(static_cast<std::size_t>(n + 1) * osize);
    std::vector<unsigned char> idx(static_cast<std::size_t>(A->nnz) * isize);
    if (flagsparseStatus_t st = adaptor::memcpy_d2h(
            off.data(), reinterpret_cast<adaptor::DevicePtr>(A->offsets), off.size())) {
        return st;
    }
    if (flagsparseStatus_t st = adaptor::memcpy_d2h(
            idx.data(), reinterpret_cast<adaptor::DevicePtr>(A->indices), idx.size())) {
        return st;
    }

    const auto offset_at = [&](int64_t i) -> int64_t {
        return (A->offsets_type == FLAGSPARSE_INDEX_32I)
                   ? reinterpret_cast<const std::int32_t*>(off.data())[i]
                   : reinterpret_cast<const std::int64_t*>(off.data())[i];
    };
    const auto column_at = [&](int64_t p) -> int64_t {
        return (A->indices_type == FLAGSPARSE_INDEX_32I)
                   ? reinterpret_cast<const std::int32_t*>(idx.data())[p]
                   : reinterpret_cast<const std::int64_t*>(idx.data())[p];
    };

    for (int64_t r = 0; r < n; ++r) {
        const int64_t begin = offset_at(r), end = offset_at(r + 1);
        if (begin < 0 || end > A->nnz || end < begin) return FLAGSPARSE_STATUS_INVALID_VALUE;
        for (int64_t p = begin; p < end; ++p) {
            const int64_t c = column_at(p);
            if (c < 0 || c >= A->cols) return FLAGSPARSE_STATUS_INVALID_VALUE;
            if (p > begin && c <= column_at(p - 1)) return FLAGSPARSE_STATUS_INVALID_VALUE;
        }
    }
    return FLAGSPARSE_STATUS_SUCCESS;
}

}  // namespace flagsparse
