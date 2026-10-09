// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <algorithm>
#include <climits>
#include <cstring>
#include <string>
#include <vector>
#include "adaptor/adaptor.hpp"
#include "core/internal.hpp"
#include "core/jit.hpp"
#include "core/prologue.hpp"

namespace flagsparse {
// CSC(A) and CSR(A^T) have the same compressed source layout. Scratch stores
// output-row pointers, original value positions, source rows and native nnz
// source IDs, never values:
// updates to A/B and alpha/beta are visible on every solve.
inline int64_t gather_rows(const SpMatDescr* A) {
    return A->format == FLAGSPARSE_FORMAT_CSC ? A->rows : A->cols;
}
// Leave room for padded tiles and interleaved component addressing.
inline bool gather_i32(const SpMatDescr* A) {
    return A->nnz <= INT32_MAX / 32 && A->rows <= INT32_MAX / 32 && A->cols <= INT32_MAX / 32;
}
inline size_t gather_index_bytes(const SpMatDescr* A) { return gather_i32(A) ? 4 : 8; }
inline size_t gather_ptr_count(const SpMatDescr* A) {
    return static_cast<size_t>((gather_rows(A) + 4) / 4 * 4);
}
inline size_t gather_nnz_count(const SpMatDescr* A) {
    return static_cast<size_t>((A->nnz + 3) / 4 * 4);
}
inline size_t gather_bytes(const SpMatDescr* A) {
    return (gather_ptr_count(A) + 3 * gather_nnz_count(A)) * gather_index_bytes(A);
}
inline bool musa_gather_type(const SpMatDescr* A) {
    return std::strcmp(adaptor::backend_name(), "musa") == 0 &&
        (A->value_type == FLAGSPARSE_R_32F || A->value_type == FLAGSPARSE_C_32F);
}
inline flagsparseStatus_t prepare_sparse_gather(SpMatDescr* A, void* buffer) {
    if (!buffer) return FLAGSPARSE_STATUS_INVALID_VALUE;
    const int64_t source_rows = A->format == FLAGSPARSE_FORMAT_CSC ? A->cols : A->rows;
    const int64_t output_rows = gather_rows(A);
    const size_t os = index_size(A->offsets_type), is = index_size(A->indices_type);
    std::vector<unsigned char> offsets(static_cast<size_t>(source_rows + 1) * os);
    std::vector<unsigned char> indices(static_cast<size_t>(A->nnz) * is);
    if (auto s = adaptor::memcpy_d2h(offsets.data(), reinterpret_cast<adaptor::DevicePtr>(A->offsets), offsets.size())) return s;
    if (!indices.empty()) {
        if (auto s = adaptor::memcpy_d2h(indices.data(), reinterpret_cast<adaptor::DevicePtr>(A->indices), indices.size())) return s;
    }
    auto at = [](const std::vector<unsigned char>& v, size_t size, int64_t pos) -> int64_t {
        return size == 4 ? reinterpret_cast<const int32_t*>(v.data())[pos]
                         : reinterpret_cast<const int64_t*>(v.data())[pos];
    };
    const int64_t base = A->idx_base == FLAGSPARSE_INDEX_BASE_ONE ? 1 : 0;
    const size_t pc = gather_ptr_count(A), nc = gather_nnz_count(A);
    std::vector<int64_t> topology(pc + 3 * nc, 0);
    auto* ptr = topology.data();
    auto* order = ptr + pc;
    auto* rows = order + nc;
    auto* native_rows = rows + nc;
    for (int64_t p = 0; p < A->nnz; ++p) {
        const int64_t col = at(indices, is, p) - base;
        if (col < 0 || col >= output_rows) return FLAGSPARSE_STATUS_INVALID_VALUE;
        ++ptr[col + 1];
    }
    for (int64_t r = 0; r < output_rows; ++r) ptr[r + 1] += ptr[r];
    std::vector<int64_t> cursor(ptr, ptr + output_rows);
    if (at(offsets, os, 0) != base || at(offsets, os, source_rows) != A->nnz + base)
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    for (int64_t r = 0; r < source_rows; ++r) {
        const int64_t begin = at(offsets, os, r) - base, end = at(offsets, os, r + 1) - base;
        if (begin < 0 || end < begin || end > A->nnz) return FLAGSPARSE_STATUS_INVALID_VALUE;
        for (int64_t p = begin; p < end; ++p) {
            const int64_t dst = cursor[at(indices, is, p) - base]++;
            order[dst] = p;
            rows[dst] = r;
            native_rows[p] = r;
        }
    }
    if (gather_i32(A)) {
        std::vector<int32_t> compact(topology.begin(), topology.end());
        if (auto s = adaptor::memcpy_h2d(reinterpret_cast<adaptor::DevicePtr>(buffer), compact.data(), compact.size() * sizeof(int32_t))) return s;
    } else {
        if (auto s = adaptor::memcpy_h2d(reinterpret_cast<adaptor::DevicePtr>(buffer), topology.data(), topology.size() * sizeof(int64_t))) return s;
    }
    A->sparse_gather_buffer = buffer;
    return FLAGSPARSE_STATUS_SUCCESS;
}
// nnz-balanced f32 scatter uses native source IDs prepared once, avoiding the
// source_rows * max_segments launch of the original CSC/transpose route.
inline flagsparseStatus_t launch_sparse_scatter(flagsparseHandle_t handle, SpMatDescr* A,
    void* B, void* C, int64_t m, double alpha, double beta, void* buffer) {
    if (auto s = scale_dense(handle, C, FLAGSPARSE_R_32F, false, 1, m, 0, 1, beta, 0.)) return s;
    if (A->nnz == 0) return FLAGSPARSE_STATUS_SUCCESS;
    auto* native = static_cast<unsigned char*>(buffer) +
        (gather_ptr_count(A) + 2 * gather_nnz_count(A)) * gather_index_bytes(A);
    std::string sig = "*fp32:16,*";
    sig += gather_i32(A) ? "i32" : "i64";
    sig += ":16,*";
    sig += triton_index_dtype(A->indices_type);
    sig += ":16,*fp32:16,*fp32:16,fp32,i64,256";
    std::vector<jit::Arg> args;
    for (void* p : {A->values, static_cast<void*>(native), A->indices, B, C})
        args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(p)));
    args.push_back(jit::Arg::f(static_cast<float>(alpha)));
    args.push_back(jit::Arg::i64v(A->nnz));
    std::string err;
    auto s = jit::launch(jit::codegen_module("sparse_gather.py"), "sparse_scatter_spmv_f32",
        sig, ctx(handle)->stream, (A->nnz + 255) / 256, 1, 1, 4, 1, args, &err);
    if (s != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return s;
}
inline flagsparseStatus_t launch_sparse_gather(flagsparseHandle_t handle, SpMatDescr* A,
    void* B, void* C, int64_t m, int64_t n, int64_t sbk, int64_t sbn,
    int64_t scm, int64_t scn, double ar, double ai, double br, double bi,
    bool conjugate, void* buffer, bool indirect) {
    if (indirect && A->sparse_gather_buffer != buffer) {
        if (auto s = prepare_sparse_gather(A, buffer)) return s;
    }
    const bool complex = A->value_type == FLAGSPARSE_C_32F;
    if (n == 1 && indirect && !complex)
        return launch_sparse_scatter(handle, A, B, C, m, ar, br, buffer);
    const double average = static_cast<double>(A->nnz) / std::max<int64_t>(m, 1);
    const bool stream = n == 8;
    const int r = n == 1 ? (average <= 4. ? 128 : 16) : (stream ? (average <= 8. ? 8 : 16) : 4);
    const int k = n == 1 ? (average <= 4. ? 4 : 8) : (stream ? 1 : 16);
    const int warps = stream && r == 8 ? 2 : 4;
    int bn = 1;
    while (bn < n && bn < 32) bn *= 2;
    auto* ptr = static_cast<unsigned char*>(buffer);
    const size_t ib = gather_index_bytes(A);
    const char* topo_type = gather_i32(A) ? "i32" : "i64";
    void* offsets = indirect ? static_cast<void*>(ptr) : A->offsets;
    void* order = indirect ? static_cast<void*>(ptr + gather_ptr_count(A) * ib) : A->indices;
    void* rows = indirect ? static_cast<void*>(ptr + (gather_ptr_count(A) + gather_nnz_count(A)) * ib) : A->indices;
    std::string sig = "*fp32:16,*";
    sig += indirect ? topo_type : triton_index_dtype(A->offsets_type);
    sig += ":16,*";
    sig += indirect ? topo_type : triton_index_dtype(A->indices_type);
    sig += ":16,*";
    sig += indirect ? topo_type : triton_index_dtype(A->indices_type);
    sig += ":16,*fp32:16,*fp32:16,fp32,fp32,fp32,fp32,i64,i64,i64,i64,i64,i64,";
    sig += std::to_string(r) + "," + std::to_string(k) + "," + std::to_string(bn) + ",";
    sig += complex ? "True," : "False,";
    sig += conjugate ? "True," : "False,";
    sig += indirect ? "True," : "False,";
    sig += (br != 0. || bi != 0.) ? "True," : "False,";
    sig += gather_i32(A) ? "False" : "True";
    std::vector<jit::Arg> args;
    for (void* p : {A->values, offsets, order, rows, B, C})
        args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(p)));
    for (double x : {ar, ai, br, bi}) args.push_back(jit::Arg::f(static_cast<float>(x)));
    for (int64_t x : {m, n, sbk, sbn, scm, scn}) args.push_back(jit::Arg::i64v(x));
    std::string err;
    auto s = jit::launch(jit::codegen_module("sparse_gather.py"), n == 1 ? "sparse_gather_spmv_f32" : (stream ? "sparse_gather_spmm_stream_f32" : "sparse_gather_f32"),
        sig, ctx(handle)->stream, (m + r - 1) / r, (n + bn - 1) / bn,
        1, warps, 1, args, &err);
    if (s != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return s;
}
} // namespace flagsparse
