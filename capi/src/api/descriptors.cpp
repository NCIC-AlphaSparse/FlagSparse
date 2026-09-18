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

#include "core/internal.hpp"

using namespace flagsparse;

namespace {

// Shared entry validation. Index/value types are checked here so that a bad
// enum can never reach a kernel launch as a silently wrong element size.
flagsparseStatus_t check_common(int64_t rows, int64_t cols, int64_t nnz,
                               flagsparseIndexBase_t base, flagsparseDataType_t vtype) {
    if (rows < 0 || cols < 0 || nnz < 0) return FLAGSPARSE_STATUS_INVALID_VALUE;
    if (base != FLAGSPARSE_INDEX_BASE_ZERO && base != FLAGSPARSE_INDEX_BASE_ONE) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    if (dtype_size(vtype) == 0) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t check_index(flagsparseIndexType_t a, flagsparseIndexType_t b) {
    if (index_size(a) == 0 || index_size(b) == 0) return FLAGSPARSE_STATUS_INVALID_VALUE;
    return FLAGSPARSE_STATUS_SUCCESS;
}

}  // namespace

extern "C" {

flagsparseStatus_t flagsparseCreateCsr(
    flagsparseSpMatDescr_t* descr, int64_t rows, int64_t cols, int64_t nnz,
    void* csrRowOffsets, void* csrColInd, void* csrValues,
    flagsparseIndexType_t csrRowOffsetsType, flagsparseIndexType_t csrColIndType,
    flagsparseIndexBase_t idxBase, flagsparseDataType_t valueType) {
    if (descr == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *descr = nullptr;
    if (flagsparseStatus_t s = check_common(rows, cols, nnz, idxBase, valueType)) return s;
    if (flagsparseStatus_t s = check_index(csrRowOffsetsType, csrColIndType)) return s;
    // A zero-nnz matrix legitimately has null indices/values; a non-empty one
    // does not, and catching it here beats a null dereference inside a kernel.
    if (csrRowOffsets == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    if (nnz > 0 && (csrColInd == nullptr || csrValues == nullptr)) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    auto* d = new (std::nothrow) SpMatDescr();
    if (d == nullptr) return FLAGSPARSE_STATUS_ALLOC_FAILED;
    d->format = FLAGSPARSE_FORMAT_CSR;
    d->rows = rows; d->cols = cols; d->nnz = nnz;
    d->offsets = csrRowOffsets; d->indices = csrColInd; d->values = csrValues;
    d->offsets_type = csrRowOffsetsType; d->indices_type = csrColIndType;
    d->idx_base = idxBase; d->value_type = valueType;
    *descr = reinterpret_cast<flagsparseSpMatDescr_t>(d);
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseCreateCsc(
    flagsparseSpMatDescr_t* descr, int64_t rows, int64_t cols, int64_t nnz,
    void* cscColOffsets, void* cscRowInd, void* cscValues,
    flagsparseIndexType_t cscColOffsetsType, flagsparseIndexType_t cscRowIndType,
    flagsparseIndexBase_t idxBase, flagsparseDataType_t valueType) {
    if (descr == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *descr = nullptr;
    if (flagsparseStatus_t s = check_common(rows, cols, nnz, idxBase, valueType)) return s;
    if (flagsparseStatus_t s = check_index(cscColOffsetsType, cscRowIndType)) return s;
    if (cscColOffsets == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    if (nnz > 0 && (cscRowInd == nullptr || cscValues == nullptr)) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    auto* d = new (std::nothrow) SpMatDescr();
    if (d == nullptr) return FLAGSPARSE_STATUS_ALLOC_FAILED;
    d->format = FLAGSPARSE_FORMAT_CSC;
    d->rows = rows; d->cols = cols; d->nnz = nnz;
    d->offsets = cscColOffsets; d->indices = cscRowInd; d->values = cscValues;
    d->offsets_type = cscColOffsetsType; d->indices_type = cscRowIndType;
    d->idx_base = idxBase; d->value_type = valueType;
    *descr = reinterpret_cast<flagsparseSpMatDescr_t>(d);
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseCreateCoo(
    flagsparseSpMatDescr_t* descr, int64_t rows, int64_t cols, int64_t nnz,
    void* cooRowInd, void* cooColInd, void* cooValues,
    flagsparseIndexType_t cooIdxType, flagsparseIndexBase_t idxBase,
    flagsparseDataType_t valueType) {
    if (descr == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *descr = nullptr;
    if (flagsparseStatus_t s = check_common(rows, cols, nnz, idxBase, valueType)) return s;
    if (flagsparseStatus_t s = check_index(cooIdxType, cooIdxType)) return s;
    if (nnz > 0 && (cooRowInd == nullptr || cooColInd == nullptr || cooValues == nullptr)) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    auto* d = new (std::nothrow) SpMatDescr();
    if (d == nullptr) return FLAGSPARSE_STATUS_ALLOC_FAILED;
    d->format = FLAGSPARSE_FORMAT_COO;
    d->rows = rows; d->cols = cols; d->nnz = nnz;
    d->row_ind = cooRowInd; d->indices = cooColInd; d->values = cooValues;
    d->offsets_type = cooIdxType; d->indices_type = cooIdxType;
    d->idx_base = idxBase; d->value_type = valueType;
    *descr = reinterpret_cast<flagsparseSpMatDescr_t>(d);
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseCreateBsr(
    flagsparseSpMatDescr_t* descr, int64_t brows, int64_t bcols, int64_t bnnz,
    int64_t rowBlockDim, int64_t colBlockDim,
    void* bsrRowOffsets, void* bsrColInd, void* bsrValues,
    flagsparseIndexType_t bsrRowOffsetsType, flagsparseIndexType_t bsrColIndType,
    flagsparseIndexBase_t idxBase, flagsparseDataType_t valueType,
    flagsparseOrder_t order) {
    if (descr == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *descr = nullptr;
    if (flagsparseStatus_t s = check_common(brows, bcols, bnnz, idxBase, valueType)) return s;
    if (flagsparseStatus_t s = check_index(bsrRowOffsetsType, bsrColIndType)) return s;
    if (rowBlockDim <= 0 || colBlockDim <= 0) return FLAGSPARSE_STATUS_INVALID_VALUE;
    if (bsrRowOffsets == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    auto* d = new (std::nothrow) SpMatDescr();
    if (d == nullptr) return FLAGSPARSE_STATUS_ALLOC_FAILED;
    d->format = FLAGSPARSE_FORMAT_BSR;
    d->rows = brows; d->cols = bcols; d->nnz = bnnz;
    d->row_block_dim = rowBlockDim; d->col_block_dim = colBlockDim;
    d->offsets = bsrRowOffsets; d->indices = bsrColInd; d->values = bsrValues;
    d->offsets_type = bsrRowOffsetsType; d->indices_type = bsrColIndType;
    d->idx_base = idxBase; d->value_type = valueType; d->order = order;
    *descr = reinterpret_cast<flagsparseSpMatDescr_t>(d);
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseCreateBlockedEll(
    flagsparseSpMatDescr_t* descr, int64_t rows, int64_t cols,
    int64_t ellBlockSize, int64_t ellCols, void* ellColInd, void* ellValue,
    flagsparseIndexType_t ellIdxType, flagsparseIndexBase_t idxBase,
    flagsparseDataType_t valueType) {
    if (descr == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *descr = nullptr;
    if (flagsparseStatus_t s = check_common(rows, cols, 0, idxBase, valueType)) return s;
    if (flagsparseStatus_t s = check_index(ellIdxType, ellIdxType)) return s;
    if (ellBlockSize <= 0 || ellCols < 0) return FLAGSPARSE_STATUS_INVALID_VALUE;
    auto* d = new (std::nothrow) SpMatDescr();
    if (d == nullptr) return FLAGSPARSE_STATUS_ALLOC_FAILED;
    d->format = FLAGSPARSE_FORMAT_BLOCKED_ELL;
    d->rows = rows; d->cols = cols;
    d->ell_block_size = ellBlockSize; d->ell_cols = ellCols;
    d->indices = ellColInd; d->values = ellValue;
    d->offsets_type = ellIdxType; d->indices_type = ellIdxType;
    d->idx_base = idxBase; d->value_type = valueType;
    *descr = reinterpret_cast<flagsparseSpMatDescr_t>(d);
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseCreateSlicedEll(
    flagsparseSpMatDescr_t* descr, int64_t rows, int64_t cols, int64_t sliceSize,
    void* sellOffsets, void* sellColInd, void* sellValues,
    flagsparseIndexType_t sellIdxType, flagsparseIndexBase_t idxBase,
    flagsparseDataType_t valueType) {
    if (descr == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *descr = nullptr;
    if (flagsparseStatus_t s = check_common(rows, cols, 0, idxBase, valueType)) return s;
    if (flagsparseStatus_t s = check_index(sellIdxType, sellIdxType)) return s;
    if (sliceSize <= 0) return FLAGSPARSE_STATUS_INVALID_VALUE;
    if (sellOffsets == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    auto* d = new (std::nothrow) SpMatDescr();
    if (d == nullptr) return FLAGSPARSE_STATUS_ALLOC_FAILED;
    d->format = FLAGSPARSE_FORMAT_SLICED_ELL;
    d->rows = rows; d->cols = cols; d->slice_size = sliceSize;
    d->offsets = sellOffsets; d->indices = sellColInd; d->values = sellValues;
    d->offsets_type = sellIdxType; d->indices_type = sellIdxType;
    d->idx_base = idxBase; d->value_type = valueType;
    *descr = reinterpret_cast<flagsparseSpMatDescr_t>(d);
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseCsrSetPointers(flagsparseSpMatDescr_t descr,
                                           void* csrRowOffsets, void* csrColInd,
                                           void* csrValues) {
    if (descr == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    SpMatDescr* d = spmat(descr);
    if (d->format != FLAGSPARSE_FORMAT_CSR) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    if (csrRowOffsets == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    // A matrix with nonzeros needs all three; an empty one legitimately has no
    // column or value array to point at.
    if (d->nnz > 0 && (csrColInd == nullptr || csrValues == nullptr)) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    d->offsets = csrRowOffsets;
    d->indices = csrColInd;
    d->values = csrValues;
    // Anything cached from the previous arrays describes a different matrix now.
    d->max_row_nnz = -1;
    d->coo_offsets_buffer = nullptr;
    d->sddmm_row_ids_buffer = nullptr;
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseDestroySpMat(flagsparseConstSpMatDescr_t descr) {
    if (descr == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    // cuSPARSE's destroy takes the const descriptor; ownership is ours.
    delete const_cast<SpMatDescr*>(spmat(descr));
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseSpMatGetSize(flagsparseConstSpMatDescr_t descr,
                                          int64_t* rows, int64_t* cols, int64_t* nnz) {
    if (descr == nullptr || rows == nullptr || cols == nullptr || nnz == nullptr) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    const SpMatDescr* d = spmat(descr);
    *rows = d->rows; *cols = d->cols; *nnz = d->nnz;
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseSpMatGetFormat(flagsparseConstSpMatDescr_t descr,
                                            flagsparseFormat_t* format) {
    if (descr == nullptr || format == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *format = spmat(descr)->format;
    return FLAGSPARSE_STATUS_SUCCESS;
}

// Attribute ids follow cuSPARSE: 0 = FILL_MODE, 1 = DIAG_TYPE.
flagsparseStatus_t flagsparseSpMatSetAttribute(flagsparseSpMatDescr_t descr,
                                               flagsparseSpMatAttribute_t attribute,
                                               const void* data, size_t dataSize) {
    if (descr == nullptr || data == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    SpMatDescr* d = spmat(descr);
    switch (attribute) {
        case FLAGSPARSE_SPMAT_FILL_MODE: {
            if (dataSize != sizeof(flagsparseFillMode_t)) return FLAGSPARSE_STATUS_INVALID_VALUE;
            d->fill_mode = *static_cast<const flagsparseFillMode_t*>(data);
            d->fill_mode_set = true;
            return FLAGSPARSE_STATUS_SUCCESS;
        }
        case FLAGSPARSE_SPMAT_DIAG_TYPE: {
            if (dataSize != sizeof(flagsparseDiagType_t)) return FLAGSPARSE_STATUS_INVALID_VALUE;
            d->diag_type = *static_cast<const flagsparseDiagType_t*>(data);
            return FLAGSPARSE_STATUS_SUCCESS;
        }
        default:
            return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    }
}

flagsparseStatus_t flagsparseCreateDnVec(flagsparseDnVecDescr_t* descr, int64_t size,
                                         void* values, flagsparseDataType_t valueType) {
    if (descr == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *descr = nullptr;
    if (size < 0) return FLAGSPARSE_STATUS_INVALID_VALUE;
    if (dtype_size(valueType) == 0) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    if (size > 0 && values == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    auto* d = new (std::nothrow) DnVecDescr();
    if (d == nullptr) return FLAGSPARSE_STATUS_ALLOC_FAILED;
    d->size = size; d->values = values; d->value_type = valueType;
    *descr = reinterpret_cast<flagsparseDnVecDescr_t>(d);
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseDestroyDnVec(flagsparseConstDnVecDescr_t descr) {
    if (descr == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    delete const_cast<DnVecDescr*>(dnvec(descr));
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseCreateDnMat(flagsparseDnMatDescr_t* descr, int64_t rows,
                                         int64_t cols, int64_t ld, void* values,
                                         flagsparseDataType_t valueType,
                                         flagsparseOrder_t order) {
    if (descr == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *descr = nullptr;
    if (rows < 0 || cols < 0) return FLAGSPARSE_STATUS_INVALID_VALUE;
    if (dtype_size(valueType) == 0) return FLAGSPARSE_STATUS_NOT_SUPPORTED;
    if (order != FLAGSPARSE_ORDER_COL && order != FLAGSPARSE_ORDER_ROW) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    // Leading dimension must span the fastest-varying extent, or a kernel would
    // read past the end of every row/column.
    const int64_t need = (order == FLAGSPARSE_ORDER_COL) ? rows : cols;
    if (ld < need) return FLAGSPARSE_STATUS_INVALID_VALUE;
    if (rows > 0 && cols > 0 && values == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    auto* d = new (std::nothrow) DnMatDescr();
    if (d == nullptr) return FLAGSPARSE_STATUS_ALLOC_FAILED;
    d->rows = rows; d->cols = cols; d->ld = ld;
    d->values = values; d->value_type = valueType; d->order = order;
    *descr = reinterpret_cast<flagsparseDnMatDescr_t>(d);
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseDestroyDnMat(flagsparseConstDnMatDescr_t descr) {
    if (descr == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    delete const_cast<DnMatDescr*>(dnmat(descr));
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseCreateSpVec(flagsparseSpVecDescr_t* descr, int64_t size,
                                         int64_t nnz, void* indices, void* values,
                                         flagsparseIndexType_t idxType,
                                         flagsparseIndexBase_t idxBase,
                                         flagsparseDataType_t valueType) {
    if (descr == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *descr = nullptr;
    if (size < 0 || nnz < 0 || nnz > size) return FLAGSPARSE_STATUS_INVALID_VALUE;
    if (flagsparseStatus_t s = check_common(size, 1, nnz, idxBase, valueType)) return s;
    if (flagsparseStatus_t s = check_index(idxType, idxType)) return s;
    if (nnz > 0 && (indices == nullptr || values == nullptr)) {
        return FLAGSPARSE_STATUS_INVALID_VALUE;
    }
    auto* d = new (std::nothrow) SpVecDescr();
    if (d == nullptr) return FLAGSPARSE_STATUS_ALLOC_FAILED;
    d->size = size; d->nnz = nnz; d->indices = indices; d->values = values;
    d->idx_type = idxType; d->idx_base = idxBase; d->value_type = valueType;
    *descr = reinterpret_cast<flagsparseSpVecDescr_t>(d);
    return FLAGSPARSE_STATUS_SUCCESS;
}

flagsparseStatus_t flagsparseDestroySpVec(flagsparseConstSpVecDescr_t descr) {
    if (descr == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    delete const_cast<SpVecDescr*>(spvec(descr));
    return FLAGSPARSE_STATUS_SUCCESS;
}

}  // extern "C"
