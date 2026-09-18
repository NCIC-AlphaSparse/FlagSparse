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

#include "core/internal.hpp"

namespace flagsparse {

std::size_t dtype_size(flagsparseDataType_t dtype) {
    switch (dtype) {
        case FLAGSPARSE_R_8I:   return 1;
        case FLAGSPARSE_R_16F:
        case FLAGSPARSE_R_16BF: return 2;
        case FLAGSPARSE_R_32F:
        case FLAGSPARSE_R_32I:  return 4;
        case FLAGSPARSE_R_64F:
        case FLAGSPARSE_C_32F:  return 8;
        case FLAGSPARSE_C_64F:  return 16;
        default:                return 0;  // caller -> NOT_SUPPORTED
    }
}

const char* triton_dtype(flagsparseDataType_t dtype) {
    switch (dtype) {
        case FLAGSPARSE_R_16F:  return "fp16";
        case FLAGSPARSE_R_16BF: return "bf16";
        case FLAGSPARSE_R_32F:  return "fp32";
        case FLAGSPARSE_R_64F:  return "fp64";
        case FLAGSPARSE_R_8I:   return "i8";
        case FLAGSPARSE_R_32I:  return "i32";
        // Triton has no complex type. The kernels consume interleaved real/imag
        // pairs of the COMPONENT dtype instead, which is how the Python side
        // already does it (view_as_real). Callers must pass the component type.
        case FLAGSPARSE_C_32F:
        case FLAGSPARSE_C_64F:
        default:                return "";
    }
}

bool is_complex(flagsparseDataType_t dtype) {
    return dtype == FLAGSPARSE_C_32F || dtype == FLAGSPARSE_C_64F;
}

flagsparseDataType_t component_dtype(flagsparseDataType_t dtype) {
    switch (dtype) {
        case FLAGSPARSE_C_32F: return FLAGSPARSE_R_32F;
        case FLAGSPARSE_C_64F: return FLAGSPARSE_R_64F;
        default:               return dtype;
    }
}

std::size_t index_size(flagsparseIndexType_t idx) {
    switch (idx) {
        case FLAGSPARSE_INDEX_16U: return 2;
        case FLAGSPARSE_INDEX_32I: return 4;
        case FLAGSPARSE_INDEX_64I: return 8;
        default:                   return 0;
    }
}

const char* triton_index_dtype(flagsparseIndexType_t idx) {
    switch (idx) {
        case FLAGSPARSE_INDEX_32I: return "i32";
        case FLAGSPARSE_INDEX_64I: return "i64";
        default:                   return "";
    }
}

}  // namespace flagsparse
