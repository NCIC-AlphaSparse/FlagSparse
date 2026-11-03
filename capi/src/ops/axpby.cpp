// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

#include <climits>
#include <cstdint>
#include <string>
#include <vector>

#include "adaptor/adaptor.hpp"
#include "core/internal.hpp"
#include "core/jit.hpp"

using namespace flagsparse;

namespace {
constexpr int kBlock = 256;

flagsparseStatus_t validate(flagsparseHandle_t handle, flagsparseConstSpVecDescr_t vecX,
                            flagsparseConstDnVecDescr_t vecY, const void* alpha,
                            const void* beta) {
    if (handle == nullptr) return FLAGSPARSE_STATUS_NOT_INITIALIZED;
    if (vecX == nullptr || vecY == nullptr || alpha == nullptr || beta == nullptr) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    if (ctx(handle)->pointer_mode != FLAGSPARSE_POINTER_MODE_HOST) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    const SpVecDescr* X = spvec(vecX);
    const DnVecDescr* Y = dnvec(vecY);
    const bool supported = X->value_type == FLAGSPARSE_R_16F ||
                           X->value_type == FLAGSPARSE_R_32F || X->value_type == FLAGSPARSE_C_32F;
    if (!supported || Y->value_type != X->value_type ||
        X->idx_base != FLAGSPARSE_INDEX_BASE_ZERO || X->size != Y->size ||
        X->nnz < 0 || X->nnz > X->size || X->nnz > INT32_MAX ||
        triton_index_dtype(X->idx_type)[0] == '\0') {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    return FLAGSPARSE_STATUS_SUCCESS;
}

// Host scalars in the value type: float16 bits, a float, or a (re, im) float pair.
void read_scalar(flagsparseDataType_t type, const void* p, float* re, float* im) {
    *im = 0.0f;
    if (type == FLAGSPARSE_R_16F) {
        *re = static_cast<float>(fp16_to_double(*static_cast<const uint16_t*>(p)));
    } else {
        const auto* f = static_cast<const float*>(p);
        *re = f[0];
        if (type == FLAGSPARSE_C_32F) *im = f[1];
    }
}

flagsparseStatus_t launch(flagsparseHandle_t handle, const void* alpha,
                           const SpVecDescr* X, const void* beta, DnVecDescr* Y) {
    float a_re, a_im, b_re, b_im;
    read_scalar(X->value_type, alpha, &a_re, &a_im);
    read_scalar(X->value_type, beta, &b_re, &b_im);
    const bool complex = X->value_type == FLAGSPARSE_C_32F;
    // Complex buffers are addressed as interleaved float32 pairs.
    const std::string vt = X->value_type == FLAGSPARSE_R_16F ? "fp16" : "fp32";
    const char* it = triton_index_dtype(X->idx_type);
    std::string err;
    std::vector<jit::Arg> scale{
        jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(Y->values)),
        jit::Arg::i(static_cast<int32_t>(Y->size)), jit::Arg::f(b_re)};
    if (complex) scale.push_back(jit::Arg::f(b_im));
    std::string scale_sig = "*" + vt + ":16,i32,fp32," + (complex ? "fp32," : "") +
                            std::to_string(kBlock);
    const char* scale_kernel = complex ? "_scale_complex_kernel"
        : (X->value_type == FLAGSPARSE_R_16F ? "axpby_scale_f16" : "axpby_scale_f32");
    flagsparseStatus_t st = jit::launch(jit::codegen_module("axpby.py"), scale_kernel,
                                         scale_sig, ctx(handle)->stream,
                                         (Y->size + kBlock - 1) / kBlock, 1, 1, 4, 1,
                                         scale, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) { ctx(handle)->last_error = err; return st; }
    if (X->nnz == 0) return FLAGSPARSE_STATUS_SUCCESS;
    std::vector<jit::Arg> add{
        jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(Y->values)),
        jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(X->values)),
        jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(X->indices)),
        jit::Arg::i(static_cast<int32_t>(X->nnz)), jit::Arg::f(a_re)};
    if (complex) add.push_back(jit::Arg::f(a_im));
    std::string add_sig = "*" + vt + ":16,*" + vt + ":16,*" + std::string(it) + ":16,i32,fp32," +
                          (complex ? "fp32," : "") + std::to_string(kBlock);
    const char* add_kernel = complex ? "_axpy_complex_kernel"
        : (X->value_type == FLAGSPARSE_R_16F ? "axpby_add_f16" : "axpby_add_f32");
    st = jit::launch(jit::codegen_module("axpby.py"), add_kernel, add_sig,
                     ctx(handle)->stream, (X->nnz + kBlock - 1) / kBlock, 1, 1, 4, 1,
                     add, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) ctx(handle)->last_error = err;
    return st;
}
}  // namespace

extern "C" flagsparseStatus_t flagsparseAxpby(
    flagsparseHandle_t handle, const void* alpha, flagsparseConstSpVecDescr_t vecX,
    const void* beta, flagsparseDnVecDescr_t vecY) {
    return guard(handle, [&]() -> flagsparseStatus_t {
        if (flagsparseStatus_t s = validate(handle, vecX, vecY, alpha, beta)) return s;
        return launch(handle, alpha, spvec(vecX), beta, dnvec(vecY));
    });
}
