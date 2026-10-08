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

#include <cstring>

namespace flagsparse {

double fp16_to_double(std::uint16_t h) {
    const std::uint32_t sign = static_cast<std::uint32_t>(h & 0x8000u) << 16;
    const std::uint32_t exp = (h >> 10) & 0x1fu;
    const std::uint32_t mant = h & 0x3ffu;
    std::uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign;
        } else {
            std::int32_t e = -1;
            std::uint32_t m = mant;
            do { m <<= 1; ++e; } while ((m & 0x400u) == 0);
            bits = sign | (static_cast<std::uint32_t>(127 - 15 - e) << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return static_cast<double>(f);
}

std::uint16_t double_to_fp16(double d) {
    float f = static_cast<float>(d);
    std::uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const std::uint32_t sign = (x >> 16) & 0x8000u;
    std::int32_t exp = static_cast<std::int32_t>((x >> 23) & 0xffu) - 127 + 15;
    std::uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return static_cast<std::uint16_t>(sign);
        mant |= 0x800000u;
        const std::int32_t shift = 14 - exp;
        const std::uint32_t sub = mant >> shift;
        const std::uint32_t rem = mant & ((1u << shift) - 1u);
        const std::uint32_t half = 1u << (shift - 1);
        std::uint32_t rounded = sub + ((rem > half || (rem == half && (sub & 1u))) ? 1u : 0u);
        return static_cast<std::uint16_t>(sign | rounded);
    }
    if (exp >= 31) return static_cast<std::uint16_t>(sign | 0x7c00u);
    const std::uint32_t round = (mant & 0x1fffu) > 0x1000u ||
                                ((mant & 0x1fffu) == 0x1000u && ((mant >> 13) & 1u));
    std::uint16_t out = static_cast<std::uint16_t>(sign | (static_cast<std::uint32_t>(exp) << 10) |
                                                   (mant >> 13));
    return static_cast<std::uint16_t>(out + round);
}

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
