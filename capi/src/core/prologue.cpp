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

#include "core/prologue.hpp"

#include <string>
#include <vector>

#include "adaptor/adaptor.hpp"
#include "core/internal.hpp"
#include "core/jit.hpp"

namespace flagsparse {

namespace {
constexpr int kBlockN = 256;
constexpr int kNumWarps = 4;
constexpr int kNumStages = 1;
}  // namespace

flagsparseStatus_t scale_dense(flagsparseHandle_t handle, void* values,
                               flagsparseDataType_t component_type, bool is_complex,
                               int64_t n_rows, int64_t n_cols,
                               int64_t stride_m, int64_t stride_n,
                               double beta_re, double beta_im) {
    if (values == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    if (n_rows <= 0 || n_cols <= 0) return FLAGSPARSE_STATUS_SUCCESS;

    const char* vt = triton_dtype(component_type);
    if (vt[0] == '\0') return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    const bool is_fp64 = (component_type == FLAGSPARSE_R_64F);
    const bool has_beta = (beta_re != 0.0 || beta_im != 0.0);

    std::string sig;
    sig.reserve(96);
    sig += "*"; sig += vt; sig += ":16,";
    sig += vt; sig += ",";                 // beta_re
    sig += vt; sig += ",";                 // beta_im
    sig += "i32,i32,";                     // n_rows, n_cols
    sig += "i64,i64,i64,";                 // stride_m, stride_n, stride_r
    sig += std::to_string(kBlockN) + ",";
    sig += is_complex ? "True," : "False,";
    sig += has_beta ? "True" : "False";

    std::vector<jit::Arg> args;
    args.reserve(8);
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(values)));
    if (is_fp64) {
        args.push_back(jit::Arg::d(beta_re));
        args.push_back(jit::Arg::d(beta_im));
    } else {
        args.push_back(jit::Arg::f(static_cast<float>(beta_re)));
        args.push_back(jit::Arg::f(static_cast<float>(beta_im)));
    }
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(n_rows)));
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(n_cols)));
    args.push_back(jit::Arg::i64v(stride_m));
    args.push_back(jit::Arg::i64v(stride_n));
    args.push_back(jit::Arg::i64v(1));     // stride_r, interleaved real/imag

    const int64_t grid_n = (n_cols + kBlockN - 1) / kBlockN;
    std::string err;
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("_dense.py"), "_dense_scale_kernel", sig,
        ctx(handle)->stream, n_rows, grid_n, 1, kNumWarps, kNumStages, args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}

flagsparseStatus_t copy_dense(flagsparseHandle_t handle, const void* src, void* dst,
                              flagsparseDataType_t component_type, bool is_complex,
                              int64_t n_rows, int64_t n_cols,
                              int64_t src_stride_m, int64_t src_stride_n,
                              int64_t dst_stride_m, int64_t dst_stride_n) {
    if (src == nullptr || dst == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    if (n_rows <= 0 || n_cols <= 0) return FLAGSPARSE_STATUS_SUCCESS;

    const char* vt = triton_dtype(component_type);
    if (vt[0] == '\0') return FLAGSPARSE_STATUS_NOT_SUPPORTED;

    std::string sig;
    sig.reserve(112);
    sig += "*"; sig += vt; sig += ":16,";
    sig += "*"; sig += vt; sig += ":16,";
    sig += "i32,i32,";                     // n_rows, n_cols
    sig += "i64,i64,i64,";                 // src strides (m, n, r)
    sig += "i64,i64,i64,";                 // dst strides (m, n, r)
    sig += std::to_string(kBlockN) + ",";
    sig += is_complex ? "True" : "False";

    std::vector<jit::Arg> args;
    args.reserve(10);
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(
        const_cast<void*>(src))));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(dst)));
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(n_rows)));
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(n_cols)));
    args.push_back(jit::Arg::i64v(src_stride_m));
    args.push_back(jit::Arg::i64v(src_stride_n));
    args.push_back(jit::Arg::i64v(1));     // src_stride_r, interleaved real/imag
    args.push_back(jit::Arg::i64v(dst_stride_m));
    args.push_back(jit::Arg::i64v(dst_stride_n));
    args.push_back(jit::Arg::i64v(1));     // dst_stride_r

    const int64_t grid_n = (n_cols + kBlockN - 1) / kBlockN;
    std::string err;
    const flagsparseStatus_t st = jit::launch(
        jit::codegen_module("_dense.py"), "_dense_copy_kernel", sig,
        ctx(handle)->stream, n_rows, grid_n, 1, kNumWarps, kNumStages, args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}

}  // namespace flagsparse
