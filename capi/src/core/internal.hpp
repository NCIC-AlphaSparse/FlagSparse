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

// Internal types shared by the C++ dispatch layer. NOT installed: the public
// surface is include/flagsparse.h and nothing else.

#pragma once

#include <cstdint>
#include <string>

#include "flagsparse.h"

namespace flagsparse {

// ---------------------------------------------------------------- dtype ---

// Bytes per scalar. 0 means "not a value type this build understands", which
// callers must turn into FLAGSPARSE_STATUS_NOT_SUPPORTED rather than guessing.
std::size_t dtype_size(flagsparseDataType_t dtype);

// Triton's spelling of a dtype, for the JIT signature string ("fp32", "i32"...).
// Empty when the dtype has no Triton equivalent.
const char* triton_dtype(flagsparseDataType_t dtype);

bool is_complex(flagsparseDataType_t dtype);

// Real component of a complex dtype (C_32F -> R_32F). Triton has no complex
// type; the kernels take interleaved real/imag pairs of THIS dtype, with twice
// the element count. Returns the input unchanged for a real dtype.
flagsparseDataType_t component_dtype(flagsparseDataType_t dtype);

std::size_t index_size(flagsparseIndexType_t idx);
const char* triton_index_dtype(flagsparseIndexType_t idx);

// ---------------------------------------------------------------- handle ---

struct Context {
    flagsparseStream_t stream = nullptr;
    flagsparsePointerMode_t pointer_mode = FLAGSPARSE_POINTER_MODE_HOST;
    int device_index = 0;
    std::string device_arch;
    // Set when a call fails deep inside; surfaced through the status code only.
    std::string last_error;
};

// --------------------------------------------------------- descriptors ---

struct SpMatDescr {
    flagsparseFormat_t format = FLAGSPARSE_FORMAT_CSR;
    int64_t rows = 0, cols = 0, nnz = 0;

    // CSR/CSC: offsets + indices. COO: row_ind + col_ind. BSR: block offsets.
    void* offsets = nullptr;   // csrRowOffsets / cscColOffsets / bsrRowOffsets
    void* indices = nullptr;   // csrColInd / cscRowInd / bsrColInd / ellColInd
    void* row_ind = nullptr;   // COO only
    void* values  = nullptr;

    flagsparseIndexType_t offsets_type = FLAGSPARSE_INDEX_32I;
    flagsparseIndexType_t indices_type = FLAGSPARSE_INDEX_32I;
    flagsparseIndexBase_t idx_base = FLAGSPARSE_INDEX_BASE_ZERO;
    flagsparseDataType_t value_type = FLAGSPARSE_R_32F;

    // BSR / ELL / SELL
    int64_t row_block_dim = 0, col_block_dim = 0;
    int64_t ell_block_size = 0, ell_cols = 0, slice_size = 0;
    flagsparseOrder_t order = FLAGSPARSE_ORDER_ROW;

    // Longest row, in nonzeros. Needed to size the kernel's segment loop and
    // only obtainable by reading the offsets, so it is computed once by
    // *_preprocess (or lazily on first use) and cached here -- the same reason
    // cuSPARSE has a preprocess step at all. -1 means "not computed yet".
    int64_t max_row_nnz = -1;

    // COO SpMM builds a row-offsets array (a CSR indptr, really) into the
    // caller's externalBuffer. Remembering WHICH buffer it went into is what
    // lets a repeated solve skip the rebuild; a caller that hands over a
    // different buffer, or reuses one buffer across matrices, gets it rebuilt.
    void* coo_offsets_buffer = nullptr;

    // SDDMM expands the CSR pattern to one row id per nonzero, into the caller's
    // externalBuffer. Same contract as coo_offsets_buffer: remembering which
    // buffer it went into is what lets a repeated solve skip the rebuild.
    void* sddmm_row_ids_buffer = nullptr;

    // Set through flagsparseSpMatSetAttribute; consumed by SpSV/SpSM.
    flagsparseFillMode_t fill_mode = FLAGSPARSE_FILL_MODE_LOWER;
    flagsparseDiagType_t diag_type = FLAGSPARSE_DIAG_TYPE_NON_UNIT;
    bool fill_mode_set = false;
};

struct DnMatDescr {
    int64_t rows = 0, cols = 0, ld = 0;
    void* values = nullptr;
    flagsparseDataType_t value_type = FLAGSPARSE_R_32F;
    flagsparseOrder_t order = FLAGSPARSE_ORDER_COL;
};

struct DnVecDescr {
    int64_t size = 0;
    void* values = nullptr;
    flagsparseDataType_t value_type = FLAGSPARSE_R_32F;
};

struct SpVecDescr {
    int64_t size = 0, nnz = 0;
    void* indices = nullptr;
    void* values = nullptr;
    flagsparseIndexType_t idx_type = FLAGSPARSE_INDEX_32I;
    flagsparseIndexBase_t idx_base = FLAGSPARSE_INDEX_BASE_ZERO;
    flagsparseDataType_t value_type = FLAGSPARSE_R_32F;
};

// ------------------------------------------------------- matrix metadata ---

// Longest row in nonzeros -- longest COLUMN for a CSC descriptor, which is what
// its offsets array describes -- read back once and cached on the descriptor
// (see SpMatDescr::max_row_nnz). SpMV needs it for correctness, its segment
// count being a constexpr; SpMM only for launch tuning. Same cached answer.
flagsparseStatus_t ensure_max_row_nnz(SpMatDescr* A, int64_t* out);

// Build a row-offsets array (length rows + 1, int32) for a row-sorted COO matrix
// and upload it to `buffer`, which the caller owns and sized from
// flagsparseSpMM_bufferSize. Returns INVALID_VALUE when the COO is not sorted by
// row or an index is out of range -- cuSPARSE requires sorted COO, and the
// alternative to checking is silently wrong output.
flagsparseStatus_t build_coo_row_offsets(SpMatDescr* A, void* buffer);

// Verify a CSR matrix's column indices are sorted ascending within each row and
// in range. Required by the triangular solves, which scan a row up to the
// diagonal and would otherwise stop early.
flagsparseStatus_t check_csr_columns_sorted(const SpMatDescr* A);

// ------------------------------------------------------------- plumbing ---

// Every extern "C" entry point wraps its body in this: a C boundary must never
// let an exception escape, and the spec forbids aborting on an unsupported
// request. Anything unexpected becomes INTERNAL_ERROR with the text retained.
template <typename Fn>
flagsparseStatus_t guard(flagsparseHandle_t handle, Fn&& fn) noexcept {
    try {
        return fn();
    } catch (const std::exception& exc) {
        if (handle != nullptr) {
            reinterpret_cast<Context*>(handle)->last_error = exc.what();
        }
        return FLAGSPARSE_STATUS_INTERNAL_ERROR;
    } catch (...) {
        return FLAGSPARSE_STATUS_INTERNAL_ERROR;
    }
}

inline Context* ctx(flagsparseHandle_t h) { return reinterpret_cast<Context*>(h); }
inline const SpMatDescr* spmat(flagsparseConstSpMatDescr_t d) {
    return reinterpret_cast<const SpMatDescr*>(d);
}
inline SpMatDescr* spmat(flagsparseSpMatDescr_t d) {
    return reinterpret_cast<SpMatDescr*>(d);
}
inline const DnVecDescr* dnvec(flagsparseConstDnVecDescr_t d) {
    return reinterpret_cast<const DnVecDescr*>(d);
}
inline DnVecDescr* dnvec(flagsparseDnVecDescr_t d) {
    return reinterpret_cast<DnVecDescr*>(d);
}
inline const DnMatDescr* dnmat(flagsparseConstDnMatDescr_t d) {
    return reinterpret_cast<const DnMatDescr*>(d);
}
inline DnMatDescr* dnmat(flagsparseDnMatDescr_t d) {
    return reinterpret_cast<DnMatDescr*>(d);
}
inline const SpVecDescr* spvec(flagsparseConstSpVecDescr_t d) {
    return reinterpret_cast<const SpVecDescr*>(d);
}
inline SpVecDescr* spvec(flagsparseSpVecDescr_t d) {
    return reinterpret_cast<SpVecDescr*>(d);
}

}  // namespace flagsparse
