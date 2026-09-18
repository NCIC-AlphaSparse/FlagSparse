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


#include <new>

#include "adaptor/adaptor.hpp"
#include "core/internal.hpp"

using namespace flagsparse;

extern "C" {

flagsparseStatus_t flagsparseCreate(flagsparseHandle_t* handle) {
    if (handle == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *handle = nullptr;
    auto* c = new (std::nothrow) Context();
    if (c == nullptr) return FLAGSPARSE_STATUS_ALLOC_FAILED;
    // Device context init happens here, not lazily at first launch: the spec
    // requires a status code rather than a crash when the backend is absent.
    const flagsparseStatus_t st = adaptor::ensure_device(c->device_index, c->device_arch);
    if (st != FLAGSPARSE_STATUS_SUCCESS) {
        delete c;
        return st;
    }
    *handle = reinterpret_cast<flagsparseHandle_t>(c);
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseDestroy(flagsparseHandle_t handle) {
    if (handle == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    delete ctx(handle);
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseSetStream(flagsparseHandle_t handle, flagsparseStream_t stream) {
    if (handle == nullptr) return FLAGSPARSE_STATUS_NOT_INITIALIZED;
    ctx(handle)->stream = stream;   // nullptr means the backend default stream
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseGetStream(flagsparseHandle_t handle, flagsparseStream_t* stream) {
    if (handle == nullptr) return FLAGSPARSE_STATUS_NOT_INITIALIZED;
    if (stream == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *stream = ctx(handle)->stream;
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseSetPointerMode(flagsparseHandle_t handle,
                                            flagsparsePointerMode_t mode) {
    if (handle == nullptr) return FLAGSPARSE_STATUS_NOT_INITIALIZED;
    if (mode != FLAGSPARSE_POINTER_MODE_HOST && mode != FLAGSPARSE_POINTER_MODE_DEVICE) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    ctx(handle)->pointer_mode = mode;
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseGetPointerMode(flagsparseHandle_t handle,
                                            flagsparsePointerMode_t* mode) {
    if (handle == nullptr) return FLAGSPARSE_STATUS_NOT_INITIALIZED;
    if (mode == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *mode = ctx(handle)->pointer_mode;
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseGetVersion(flagsparseHandle_t handle, int* version) {
    if (handle == nullptr) return FLAGSPARSE_STATUS_NOT_INITIALIZED;
    if (version == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *version = FLAGSPARSE_VERSION;
    return FLAGSPARSE_STATUS_SUCCESS;
}

const char* flagsparseGetBackendName(void) { return adaptor::backend_name(); }

flagsparseStatus_t flagsparseGetLastErrorString(flagsparseHandle_t handle, const char** str) {
    if (handle == nullptr) return FLAGSPARSE_STATUS_NOT_INITIALIZED;
    if (str == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *str = ctx(handle)->last_error.c_str();
    return FLAGSPARSE_STATUS_SUCCESS;
}

// Must assign through `str` and return a STATUS: returning the string would
// silently reinterpret a pointer as an enum on a compiler less strict than this one.
#define FS_STATUS_CASE(name) \
    case name: *str = #name; return FLAGSPARSE_STATUS_SUCCESS;

flagsparseStatus_t flagsparseGetErrorName(flagsparseStatus_t status, const char** str) {
    if (str == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    switch (status) {
        FS_STATUS_CASE(FLAGSPARSE_STATUS_SUCCESS)
        FS_STATUS_CASE(FLAGSPARSE_STATUS_NOT_INITIALIZED)
        FS_STATUS_CASE(FLAGSPARSE_STATUS_ALLOC_FAILED)
        FS_STATUS_CASE(FLAGSPARSE_STATUS_INVALID_VALUE)
        FS_STATUS_CASE(FLAGSPARSE_STATUS_ARCH_MISMATCH)
        FS_STATUS_CASE(FLAGSPARSE_STATUS_MAPPING_ERROR)
        FS_STATUS_CASE(FLAGSPARSE_STATUS_EXECUTION_FAILED)
        FS_STATUS_CASE(FLAGSPARSE_STATUS_INTERNAL_ERROR)
        FS_STATUS_CASE(FLAGSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED)
        FS_STATUS_CASE(FLAGSPARSE_STATUS_ZERO_PIVOT)
        FS_STATUS_CASE(FLAGSPARSE_STATUS_NOT_SUPPORTED)
        FS_STATUS_CASE(FLAGSPARSE_STATUS_INSUFFICIENT_RESOURCES)
    }
    *str = "FLAGSPARSE_STATUS_UNKNOWN";
    return FLAGSPARSE_STATUS_INVALID_VALUE;
}

#undef FS_STATUS_CASE

flagsparseStatus_t flagsparseGetErrorString(flagsparseStatus_t status, const char** str) {
    if (str == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    switch (status) {
        case FLAGSPARSE_STATUS_SUCCESS:
            *str = "the operation completed successfully"; break;
        case FLAGSPARSE_STATUS_NOT_INITIALIZED:
            *str = "the handle was not created with flagsparseCreate"; break;
        case FLAGSPARSE_STATUS_ALLOC_FAILED:
            *str = "resource allocation failed"; break;
        case FLAGSPARSE_STATUS_INVALID_VALUE:
            *str = "an argument had an invalid value"; break;
        case FLAGSPARSE_STATUS_ARCH_MISMATCH:
            *str = "the device architecture is not supported by this build"; break;
        case FLAGSPARSE_STATUS_MAPPING_ERROR:
            *str = "a memory mapping operation failed"; break;
        case FLAGSPARSE_STATUS_EXECUTION_FAILED:
            *str = "the kernel launch failed"; break;
        case FLAGSPARSE_STATUS_INTERNAL_ERROR:
            *str = "an internal error occurred"; break;
        case FLAGSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
            *str = "the matrix type is not supported by this routine"; break;
        case FLAGSPARSE_STATUS_ZERO_PIVOT:
            *str = "a zero pivot was encountered"; break;
        case FLAGSPARSE_STATUS_NOT_SUPPORTED:
            *str = "the operation is not supported on this backend or build"; break;
        case FLAGSPARSE_STATUS_INSUFFICIENT_RESOURCES:
            *str = "insufficient resources for the requested operation"; break;
        default:
            *str = "unknown status";
            return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    return FLAGSPARSE_STATUS_SUCCESS;
}

}  // extern "C"
