// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: Apache-2.0

// Mixed-precision sparse-vector dot product. The public shape mirrors
// cusparseSpVV; the first C API slice covers q4's f16->f32 and i8->i32 cases.

#include <climits>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "adaptor/adaptor.hpp"
#include "core/internal.hpp"
#include "core/jit.hpp"

using namespace flagsparse;

namespace {

constexpr int kBlock = 256;

flagsparseStatus_t validate(flagsparseHandle_t handle, flagsparseOperation_t opX,
                            flagsparseConstSpVecDescr_t vecX,
                            flagsparseConstDnVecDescr_t vecY, const void* result,
                            flagsparseDataType_t computeType) {
    if (handle == nullptr) return FLAGSPARSE_STATUS_NOT_INITIALIZED;
    if (vecX == nullptr || vecY == nullptr || result == nullptr) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    if (opX != FLAGSPARSE_OPERATION_NON_TRANSPOSE &&
        opX != FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    const SpVecDescr* X = spvec(vecX);
    const DnVecDescr* Y = dnvec(vecY);
    if (X->idx_base != FLAGSPARSE_INDEX_BASE_ZERO) {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    if (X->size != Y->size || X->nnz < 0 || X->nnz > X->size) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    if (X->nnz > static_cast<int64_t>(INT32_MAX) ||
        triton_index_dtype(X->idx_type)[0] == '\0') {
        return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
    const bool f16f32 = X->value_type == FLAGSPARSE_R_16F &&
                        Y->value_type == FLAGSPARSE_R_16F &&
                        computeType == FLAGSPARSE_R_32F;
    const bool i8i32 = X->value_type == FLAGSPARSE_R_8I &&
                       Y->value_type == FLAGSPARSE_R_8I &&
                       computeType == FLAGSPARSE_R_32I;
    const bool c32 = X->value_type == FLAGSPARSE_C_32F &&
                     Y->value_type == FLAGSPARSE_C_32F &&
                     computeType == FLAGSPARSE_C_32F;
    return (f16f32 || i8i32 || c32) ? FLAGSPARSE_STATUS_SUCCESS
                             : FLAGSPARSE_STATUS_NOT_SUPPORTED;
}

flagsparseStatus_t store_zero(flagsparseHandle_t handle, void* result,
                              flagsparseDataType_t computeType) {
    const std::size_t bytes = dtype_size(computeType);
    if (ctx(handle)->pointer_mode == FLAGSPARSE_POINTER_MODE_DEVICE) {
        return adaptor::memset_device(reinterpret_cast<adaptor::DevicePtr>(result), 0, bytes);
    }
    std::memset(result, 0, bytes);
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t run(flagsparseHandle_t handle, flagsparseOperation_t opX,
                       const SpVecDescr* X,
                       const DnVecDescr* Y, void* result,
                       flagsparseDataType_t computeType, void* externalBuffer) {
    if (X->nnz == 0) return store_zero(handle, result, computeType);

    const bool result_on_device =
        ctx(handle)->pointer_mode == FLAGSPARSE_POINTER_MODE_DEVICE;
    void* device_result = result_on_device ? result : externalBuffer;
    if (device_result == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;

    const bool c32 = computeType == FLAGSPARSE_C_32F;
    const char* in = triton_dtype(component_dtype(X->value_type));
    const char* out = triton_dtype(component_dtype(computeType));
    const char* it = triton_index_dtype(X->idx_type);
    std::string sig;
    sig.reserve(96);
    sig += "*"; sig += out; sig += ":16,";
    sig += "*"; sig += in; sig += ":16,";
    sig += "*"; sig += in; sig += ":16,";
    sig += "*"; sig += it; sig += ":16,i32,";
    if (c32) {
        sig += opX == FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE ? "True," : "False,";
    }
    sig += std::to_string(kBlock);

    std::vector<jit::Arg> args;
    args.reserve(5);
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(device_result)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(X->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(Y->values)));
    args.push_back(jit::Arg::ptr(reinterpret_cast<adaptor::DevicePtr>(X->indices)));
    args.push_back(jit::Arg::i(static_cast<std::int32_t>(X->nnz)));

    std::string err;
    flagsparseStatus_t st = jit::launch(
        jit::codegen_module("spvv.py"), c32 ? "spvv_c32" :
        (computeType == FLAGSPARSE_R_32I ? "spvv_mixed_i32acc" : "spvv_mixed_f32acc"),
        sig, ctx(handle)->stream, 1, 1, 1, /*num_warps=*/4, /*num_stages=*/2,
        args, &err);
    if (st != FLAGSPARSE_STATUS_SUCCESS) {
        ctx(handle)->last_error = err;
        return st;
    }
    if (!result_on_device) {
        // The runtime copies are synchronous and do not carry the handle's
        // stream, so finish the launch before copying the scalar to host.
        adaptor::synchronize();
        st = adaptor::memcpy_d2h(result,
                                 reinterpret_cast<adaptor::DevicePtr>(device_result),
                                 dtype_size(computeType));
    }
    return st;
}

}  // namespace

extern "C" {

flagsparseStatus_t flagsparseSpVV_bufferSize(
    flagsparseHandle_t handle, flagsparseOperation_t opX,
    flagsparseConstSpVecDescr_t vecX, flagsparseConstDnVecDescr_t vecY,
    const void* result, flagsparseDataType_t computeType, size_t* bufferSize) {
    if (bufferSize == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *bufferSize = 0;
    return guard(handle, [&]() -> flagsparseStatus_t {
        if (flagsparseStatus_t s = validate(handle, opX, vecX, vecY, result, computeType)) {
            return s;
        }
        if (ctx(handle)->pointer_mode == FLAGSPARSE_POINTER_MODE_HOST &&
            spvec(vecX)->nnz != 0) {
            *bufferSize = dtype_size(computeType);
        }
        return FLAGSPARSE_STATUS_SUCCESS;
    });
}

flagsparseStatus_t flagsparseSpVV(
    flagsparseHandle_t handle, flagsparseOperation_t opX,
    flagsparseConstSpVecDescr_t vecX, flagsparseConstDnVecDescr_t vecY,
    void* result, flagsparseDataType_t computeType, void* externalBuffer) {
    return guard(handle, [&]() -> flagsparseStatus_t {
        if (flagsparseStatus_t s = validate(handle, opX, vecX, vecY, result, computeType)) {
            return s;
        }
        return run(handle, opX, spvec(vecX), dnvec(vecY), result, computeType, externalBuffer);
    });
}

}  // extern "C"
