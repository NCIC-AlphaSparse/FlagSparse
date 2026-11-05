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

// gemvi: y = alpha * op(A) * x + beta * y, A dense column-major, x sparse.
//
// The kernels are src/flagsparse/sparse_operations/gemvi.py's, through the
// ACC_IS_FP64 wrappers in flagsparse_codegen/gemvi.py, and so is the plan: one
// program per BLOCK_OUT outputs, and when that leaves the device idle on a
// large enough product, x's entries cut into runs whose partial sums (in the
// caller's buffer) a reduction kernel finishes together with alpha and beta.

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

constexpr int kBlockOut = 64;
constexpr int kBlockK = 32;
constexpr int kReduceBlock = 256;
// Below this many bytes of A to read a call is launch-bound and a second kernel
// costs more than it saves (gemvi.py's _SPLIT_MIN_BYTES).
constexpr std::size_t kSplitMinBytes = std::size_t{16} << 20;

int64_t cdiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

struct Plan {
    int64_t n_out = 0, n_in = 0;
    int64_t n_split = 1, k_chunk = 0;
    bool complex = false, fp64 = false, musa_col = false;
    int block_out = kBlockOut, block_k = kBlockK, warps = 4;
    std::size_t part_bytes = 0;
};

bool supported_type(flagsparseDataType_t t) {
    return t == FLAGSPARSE_R_32F || t == FLAGSPARSE_R_64F || t == FLAGSPARSE_C_32F ||
           t == FLAGSPARSE_C_64F;
}

int gemvi_sm_count(int device) {
    if (std::string(adaptor::backend_name()) != "musa")
        return adaptor::multiprocessor_count(device);
    // Hardware properties are immutable. MUSA's query costs more than a
    // small GEMVI kernel; cache by device without retaining any operands.
    thread_local std::vector<std::pair<int, int>> counts;
    for (const auto& entry : counts)
        if (entry.first == device) return entry.second;
    const int count = adaptor::multiprocessor_count(device);
    counts.emplace_back(device, count);
    return count;
}

// gemvi.py's _split_count: ~8 programs per SM, each run several BLOCK_K steps long.
int64_t split_count(int64_t n_out, int64_t nnz, std::size_t elem, int sm_count) {
    if (static_cast<std::size_t>(n_out) * static_cast<std::size_t>(nnz) * elem <= kSplitMinBytes) {
        return 1;
    }
    const int64_t out_blocks = cdiv(n_out, kBlockOut);
    const int64_t target = 8 * std::max(1, sm_count);
    if (out_blocks >= target) return 1;
    return std::max<int64_t>(1, std::min<int64_t>(target / out_blocks, nnz / (4 * kBlockK)));
}

flagsparseStatus_t make_plan(flagsparseHandle_t handle, flagsparseOperation_t transA,
                             int m, int n, int nnz, flagsparseDataType_t valueType,
                             Plan* plan) {
    if (handle == nullptr) return FLAGSPARSE_STATUS_NOT_INITIALIZED;
    if (transA == FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        ctx(handle)->last_error = "gemvi has no conjugate transpose, as cusparse<t>gemvi has none";
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (transA != FLAGSPARSE_OPERATION_NON_TRANSPOSE &&
        transA != FLAGSPARSE_OPERATION_TRANSPOSE) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    if (!supported_type(valueType)) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    if (m < 0 || n < 0 || nnz < 0) return FLAGSPARSE_STATUS_INVALID_VALUE;
    const bool trans = transA == FLAGSPARSE_OPERATION_TRANSPOSE;
    plan->n_out = trans ? n : m;
    plan->n_in = trans ? m : n;
    if (nnz > plan->n_in) return FLAGSPARSE_STATUS_INVALID_VALUE;
    plan->complex = is_complex(valueType);
    plan->fp64 = component_dtype(valueType) == FLAGSPARSE_R_64F;
    plan->n_split = split_count(plan->n_out, nnz, dtype_size(valueType),
                                gemvi_sm_count(ctx(handle)->device_index));
    plan->musa_col = std::string(adaptor::backend_name()) == "musa" && !trans && !plan->fp64;
    if (plan->musa_col) {
        // Dense rows are contiguous. More independent output lanes avoid
        // repeated cross-lane reductions on MUSA's 32-lane warps.
        plan->block_out = plan->complex ? 64 : 128;
        plan->block_k = plan->complex ? 4 : 16;
        if (plan->n_out <= 8192) {
            plan->block_out = plan->complex ? 32 : 64;
            plan->block_k = 16;
        }
        plan->n_split = static_cast<std::size_t>(plan->n_out) * nnz * dtype_size(valueType)
            > kSplitMinBytes && plan->n_out < 32768 ? 4 : 1;
    }
    // Runs are whole sparse tiles; recount them after rounding.
    plan->k_chunk = cdiv(cdiv(std::max(1, nnz), plan->n_split), plan->block_k) * plan->block_k;
    plan->n_split = cdiv(std::max(1, nnz), plan->k_chunk);
    plan->part_bytes = plan->n_split > 1
        ? static_cast<std::size_t>(plan->n_split * plan->n_out) * dtype_size(valueType)
        : 0;
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

jit::Arg scalar_arg(bool fp64, double v) {
    return fp64 ? jit::Arg::d(v) : jit::Arg::f(static_cast<float>(v));
}

}  // namespace

extern "C" {

flagsparseStatus_t flagsparseGemvi_bufferSize(
    flagsparseHandle_t handle, flagsparseOperation_t transA, int m, int n, int nnz,
    flagsparseDataType_t valueType, size_t* bufferSize) {
    if (bufferSize == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *bufferSize = 0;
    return guard(handle, [&]() -> flagsparseStatus_t {
        Plan plan;
        if (flagsparseStatus_t s = make_plan(handle, transA, m, n, nnz, valueType, &plan)) {
            return s;
        }
        *bufferSize = plan.part_bytes;
        return FLAGSPARSE_STATUS_SUCCESS;
    });
}

flagsparseStatus_t flagsparseGemvi(
    flagsparseHandle_t handle, flagsparseOperation_t transA, int m, int n,
    const void* alpha, const void* A, int lda, int nnz, const void* xVal,
    const int* xInd, const void* beta, void* y, flagsparseIndexBase_t idxBase,
    flagsparseDataType_t valueType, void* pBuffer) {
    return guard(handle, [&]() -> flagsparseStatus_t {
        Plan plan;
        if (flagsparseStatus_t s = make_plan(handle, transA, m, n, nnz, valueType, &plan)) {
            return s;
        }
        if (idxBase != FLAGSPARSE_INDEX_BASE_ZERO) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
        if (alpha == nullptr || beta == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
        if (ctx(handle)->pointer_mode != FLAGSPARSE_POINTER_MODE_HOST) {
            return FLAGSPARSE_STATUS_NOT_SUPPORTED;
        }
        if (lda < std::max(1, m)) return FLAGSPARSE_STATUS_INVALID_VALUE;
        if (plan.n_out == 0) return FLAGSPARSE_STATUS_SUCCESS;
        if ((nnz > 0 && (xVal == nullptr || xInd == nullptr)) || y == nullptr ||
            (m > 0 && n > 0 && A == nullptr)) {
            return FLAGSPARSE_STATUS_INVALID_VALUE;
        }
        if (plan.n_split > 1 && pBuffer == nullptr) {
            ctx(handle)->last_error =
                "this gemvi splits x's entries and needs the pBuffer reported by "
                "flagsparseGemvi_bufferSize";
            return FLAGSPARSE_STATUS_INVALID_VALUE;
        }

        double a_re, a_im, b_re, b_im;
        read_scalar(valueType, alpha, &a_re, &a_im);
        read_scalar(valueType, beta, &b_re, &b_im);
        const bool has_beta = b_re != 0.0 || b_im != 0.0;
        const bool split = plan.n_split > 1;
        const bool trans = transA == FLAGSPARSE_OPERATION_TRANSPOSE;
        // Column-major A: element (i, j) at i + j * lda. op(A)'s outputs walk
        // A's rows for op = A and its columns for op = A^T.
        const int64_t unit = plan.complex ? 2 : 1;
        const int64_t stride_out = (trans ? lda : 1) * unit;
        const int64_t stride_k = (trans ? 1 : lda) * unit;
        const char* vt = plan.fp64 ? "fp64" : "fp32";   // a component
        const std::string fp = vt;
        void* first_out = split ? pBuffer : y;
        const int64_t part_stride = split ? plan.n_out * unit : 0;

        auto integer_sig = [&](const char* type, int64_t value) {
            return std::string(type) + (plan.musa_col && value % 16 == 0 ? ":16," : ",");
        };
        std::string sig = "*" + fp + ":16,*" + fp + ":16,*" + fp + ":16,*i32:16,";
        sig += integer_sig("i32", plan.n_out) + integer_sig("i32", nnz) +
               integer_sig("i64", stride_out) + integer_sig("i64", stride_k);
        std::vector<jit::Arg> args;
        args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(first_out)));
        args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(const_cast<void*>(A))));
        args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(const_cast<void*>(xVal))));
        args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(const_cast<int*>(xInd))));
        args.push_back(jit::Arg::i(static_cast<std::int32_t>(plan.n_out)));
        args.push_back(jit::Arg::i(nnz));
        args.push_back(jit::Arg::i64v(stride_out));
        args.push_back(jit::Arg::i64v(stride_k));
        // The first pass writes alpha * sum and beta * y itself unless it only
        // writes partial sums, which the reduction scales.
        const double fa_re = split ? 1.0 : a_re, fa_im = split ? 0.0 : a_im;
        const double fb_re = split ? 0.0 : b_re, fb_im = split ? 0.0 : b_im;
        if (plan.complex || plan.musa_col) {
            sig += fp + "," + fp + "," + fp + "," + fp + ",";
            args.push_back(scalar_arg(plan.fp64, fa_re));
            args.push_back(scalar_arg(plan.fp64, fa_im));
            args.push_back(scalar_arg(plan.fp64, fb_re));
            args.push_back(scalar_arg(plan.fp64, fb_im));
        } else {
            sig += fp + "," + fp + ",";
            args.push_back(scalar_arg(plan.fp64, fa_re));
            args.push_back(scalar_arg(plan.fp64, fb_re));
        }
        sig += integer_sig("i32", plan.k_chunk) + integer_sig("i64", part_stride);
        args.push_back(jit::Arg::i(static_cast<std::int32_t>(plan.k_chunk)));
        args.push_back(jit::Arg::i64v(part_stride));
        sig += (!split && has_beta) ? "True," : "False,";
        sig += (plan.musa_col ? plan.complex : plan.fp64) ? "True," : "False,";
        sig += std::to_string(plan.block_out) + "," + std::to_string(plan.block_k);
        // OUT_INNER (real only): outputs adjacent in memory, i.e. op = A here.
        if (!plan.complex && !plan.musa_col) sig += stride_out == 1 ? ",True" : ",False";

        std::string err;
        flagsparseStatus_t st = jit::launch(
            jit::codegen_module("gemvi.py"), plan.musa_col ? "gemvi_musa_col" : (plan.complex ? "gemvi_complex" : "gemvi_real"),
            sig, ctx(handle)->stream, cdiv(plan.n_out, plan.block_out), plan.n_split, 1,
            /*num_warps=*/plan.warps, /*num_stages=*/plan.musa_col ? 1 : 2, args, &err);
        if (st != FLAGSPARSE_STATUS_SUCCESS) {
            ctx(handle)->last_error = err;
            return st;
        }
        if (!split) return FLAGSPARSE_STATUS_SUCCESS;

        std::string rsig = "*" + fp + ":16,*" + fp + ":16,i32,i32," + fp + "," + fp + "," + fp +
                           "," + fp + ",";
        rsig += has_beta ? "True," : "False,";
        rsig += plan.complex ? "True," : "False,";
        rsig += plan.fp64 ? "True," : "False,";
        rsig += std::to_string(kReduceBlock);
        std::vector<jit::Arg> rargs;
        rargs.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(y)));
        rargs.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(pBuffer)));
        rargs.push_back(jit::Arg::i(static_cast<std::int32_t>(plan.n_out)));
        rargs.push_back(jit::Arg::i(static_cast<std::int32_t>(plan.n_split)));
        rargs.push_back(scalar_arg(plan.fp64, a_re));
        rargs.push_back(scalar_arg(plan.fp64, a_im));
        rargs.push_back(scalar_arg(plan.fp64, b_re));
        rargs.push_back(scalar_arg(plan.fp64, b_im));
        st = jit::launch(jit::codegen_module("gemvi.py"), "gemvi_reduce", rsig,
                         ctx(handle)->stream, cdiv(plan.n_out, kReduceBlock), 1, 1,
                         /*num_warps=*/4, /*num_stages=*/2, rargs, &err);
        if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
        return st;
    });
}

}  // extern "C"
