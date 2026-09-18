/*
 * Copyright 2026 FlagOS Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * FlagSparse public C API -- the ONLY installed header.
 *
 * Mirrors the cuSPARSE Generic API: signatures, parameter order and semantics
 * match, so migrating from cuSPARSE is a prefix replacement
 * (cusparseXxx -> flagsparseXxx) with case and suffixes preserved.
 *
 * Hardware-specific types (stream, handle) are opaque pointers and never bind a
 * vendor type, so a new backend needs no header change.
 */

#ifndef FLAGSPARSE_H_
#define FLAGSPARSE_H_

#include <stddef.h>
#include <stdint.h>

#define FLAGSPARSE_VER_MAJOR 1
#define FLAGSPARSE_VER_MINOR 0
#define FLAGSPARSE_VER_PATCH 0
#define FLAGSPARSE_VERSION (FLAGSPARSE_VER_MAJOR * 1000 + \
                            FLAGSPARSE_VER_MINOR *  100 + \
                            FLAGSPARSE_VER_PATCH)

#if defined(__GNUC__) || defined(__clang__)
#define FLAGSPARSE_DEPRECATED __attribute__((deprecated))
#define FLAGSPARSE_EXPORT     __attribute__((visibility("default")))
#else
#define FLAGSPARSE_DEPRECATED
#define FLAGSPARSE_EXPORT
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------ status */
/* Values are cuSPARSE's, so an NVIDIA adaptation layer can map them straight
 * through. Never abort or throw across this boundary: an unsupported request
 * returns FLAGSPARSE_STATUS_NOT_SUPPORTED. */
typedef enum {
    FLAGSPARSE_STATUS_SUCCESS                   = 0,
    FLAGSPARSE_STATUS_NOT_INITIALIZED           = 1,
    FLAGSPARSE_STATUS_ALLOC_FAILED              = 2,
    FLAGSPARSE_STATUS_INVALID_VALUE             = 3,
    FLAGSPARSE_STATUS_ARCH_MISMATCH             = 4,
    FLAGSPARSE_STATUS_MAPPING_ERROR             = 5,
    FLAGSPARSE_STATUS_EXECUTION_FAILED          = 6,
    FLAGSPARSE_STATUS_INTERNAL_ERROR            = 7,
    FLAGSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED = 8,
    FLAGSPARSE_STATUS_ZERO_PIVOT                = 9,
    FLAGSPARSE_STATUS_NOT_SUPPORTED             = 10,
    FLAGSPARSE_STATUS_INSUFFICIENT_RESOURCES    = 11
} flagsparseStatus_t;

/* ------------------------------------------------------------------- enums */
typedef enum {
    FLAGSPARSE_POINTER_MODE_HOST   = 0,
    FLAGSPARSE_POINTER_MODE_DEVICE = 1
} flagsparsePointerMode_t;

typedef enum {
    FLAGSPARSE_OPERATION_NON_TRANSPOSE       = 0,
    FLAGSPARSE_OPERATION_TRANSPOSE           = 1,
    FLAGSPARSE_OPERATION_CONJUGATE_TRANSPOSE = 2
} flagsparseOperation_t;

typedef enum {
    FLAGSPARSE_FILL_MODE_LOWER = 0,
    FLAGSPARSE_FILL_MODE_UPPER = 1
} flagsparseFillMode_t;

typedef enum {
    FLAGSPARSE_DIAG_TYPE_NON_UNIT = 0,
    FLAGSPARSE_DIAG_TYPE_UNIT     = 1
} flagsparseDiagType_t;

typedef enum {
    FLAGSPARSE_INDEX_BASE_ZERO = 0,
    FLAGSPARSE_INDEX_BASE_ONE  = 1
} flagsparseIndexBase_t;

typedef enum {
    FLAGSPARSE_MATRIX_TYPE_GENERAL    = 0,
    FLAGSPARSE_MATRIX_TYPE_SYMMETRIC  = 1,
    FLAGSPARSE_MATRIX_TYPE_HERMITIAN  = 2,
    FLAGSPARSE_MATRIX_TYPE_TRIANGULAR = 3
} flagsparseMatrixType_t;

/* Index width is a DESCRIPTOR property, never part of a function name -- the
 * same rule cuSPARSE follows. Every operator accepts both 32I and 64I. */
typedef enum {
    FLAGSPARSE_INDEX_16U = 1,
    FLAGSPARSE_INDEX_32I = 2,
    FLAGSPARSE_INDEX_64I = 3
} flagsparseIndexType_t;

typedef enum {
    FLAGSPARSE_ORDER_COL = 1,
    FLAGSPARSE_ORDER_ROW = 2
} flagsparseOrder_t;

/* Value type, mirroring cudaDataType's naming. */
typedef enum {
    FLAGSPARSE_R_16F  = 2,
    FLAGSPARSE_R_32F  = 0,
    FLAGSPARSE_R_64F  = 1,
    FLAGSPARSE_C_32F  = 4,
    FLAGSPARSE_C_64F  = 5,
    FLAGSPARSE_R_16BF = 14,
    FLAGSPARSE_R_8I   = 3,
    FLAGSPARSE_R_32I  = 10
} flagsparseDataType_t;

typedef enum {
    FLAGSPARSE_FORMAT_CSR      = 1,
    FLAGSPARSE_FORMAT_CSC      = 2,
    FLAGSPARSE_FORMAT_COO      = 3,
    FLAGSPARSE_FORMAT_BSR      = 4,
    FLAGSPARSE_FORMAT_BLOCKED_ELL = 5,
    FLAGSPARSE_FORMAT_SLICED_ELL  = 6
} flagsparseFormat_t;

/* ---------------------------------------------------- handle and resources */
typedef struct flagsparseContext* flagsparseHandle_t;
typedef void*                     flagsparseStream_t;

FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseCreate(flagsparseHandle_t* handle);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseDestroy(flagsparseHandle_t handle);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSetStream(flagsparseHandle_t handle,
                                                         flagsparseStream_t stream);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseGetStream(flagsparseHandle_t handle,
                                                         flagsparseStream_t* stream);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSetPointerMode(flagsparseHandle_t handle,
                                                              flagsparsePointerMode_t mode);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseGetPointerMode(flagsparseHandle_t handle,
                                                              flagsparsePointerMode_t* mode);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseGetVersion(flagsparseHandle_t handle, int* version);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseGetErrorName(flagsparseStatus_t status,
                                                            const char** str);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseGetErrorString(flagsparseStatus_t status,
                                                              const char** str);
/* Backend the library was built for ("cuda", "musa", ...). Not in cuSPARSE;
 * FlagSparse-specific and additive, so it cannot break prefix migration. */
FLAGSPARSE_EXPORT const char* flagsparseGetBackendName(void);

/* Detail of the most recent failure on this handle, or "" when there is none.
 *
 * flagsparseGetErrorString only ever returns the generic text for a status
 * code; a JIT compile or launch failure carries a specific message (a Triton
 * diagnostic, a missing kernel module) that would otherwise be unreachable,
 * leaving the caller with a bare EXECUTION_FAILED and nothing to act on.
 *
 * The pointer stays valid until the next failing call on the same handle.
 * FlagSparse-specific and additive. */
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseGetLastErrorString(flagsparseHandle_t handle,
                                                                  const char** str);

/* -------------------------------------------------------------- descriptors */
typedef struct flagsparseSpMatDescr*       flagsparseSpMatDescr_t;
typedef const struct flagsparseSpMatDescr* flagsparseConstSpMatDescr_t;
typedef struct flagsparseDnMatDescr*       flagsparseDnMatDescr_t;
typedef const struct flagsparseDnMatDescr* flagsparseConstDnMatDescr_t;
typedef struct flagsparseDnVecDescr*       flagsparseDnVecDescr_t;
typedef const struct flagsparseDnVecDescr* flagsparseConstDnVecDescr_t;
typedef struct flagsparseSpVecDescr*       flagsparseSpVecDescr_t;
typedef const struct flagsparseSpVecDescr* flagsparseConstSpVecDescr_t;

FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseCreateCsr(
    flagsparseSpMatDescr_t* descr, int64_t rows, int64_t cols, int64_t nnz,
    void* csrRowOffsets, void* csrColInd, void* csrValues,
    flagsparseIndexType_t csrRowOffsetsType, flagsparseIndexType_t csrColIndType,
    flagsparseIndexBase_t idxBase, flagsparseDataType_t valueType);

FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseCreateCsc(
    flagsparseSpMatDescr_t* descr, int64_t rows, int64_t cols, int64_t nnz,
    void* cscColOffsets, void* cscRowInd, void* cscValues,
    flagsparseIndexType_t cscColOffsetsType, flagsparseIndexType_t cscRowIndType,
    flagsparseIndexBase_t idxBase, flagsparseDataType_t valueType);

FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseCreateCoo(
    flagsparseSpMatDescr_t* descr, int64_t rows, int64_t cols, int64_t nnz,
    void* cooRowInd, void* cooColInd, void* cooValues,
    flagsparseIndexType_t cooIdxType, flagsparseIndexBase_t idxBase,
    flagsparseDataType_t valueType);

FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseCreateBsr(
    flagsparseSpMatDescr_t* descr, int64_t brows, int64_t bcols, int64_t bnnz,
    int64_t rowBlockDim, int64_t colBlockDim,
    void* bsrRowOffsets, void* bsrColInd, void* bsrValues,
    flagsparseIndexType_t bsrRowOffsetsType, flagsparseIndexType_t bsrColIndType,
    flagsparseIndexBase_t idxBase, flagsparseDataType_t valueType,
    flagsparseOrder_t order);

FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseCreateBlockedEll(
    flagsparseSpMatDescr_t* descr, int64_t rows, int64_t cols,
    int64_t ellBlockSize, int64_t ellCols, void* ellColInd, void* ellValue,
    flagsparseIndexType_t ellIdxType, flagsparseIndexBase_t idxBase,
    flagsparseDataType_t valueType);

/* Sliced-ELL is a FlagSparse extension (SpSV SELL); cuSPARSE has no counterpart. */
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseCreateSlicedEll(
    flagsparseSpMatDescr_t* descr, int64_t rows, int64_t cols, int64_t sliceSize,
    void* sellOffsets, void* sellColInd, void* sellValues,
    flagsparseIndexType_t sellIdxType, flagsparseIndexBase_t idxBase,
    flagsparseDataType_t valueType);

/* Attach arrays to an existing CSR descriptor, cuSPARSE's cusparseCsrSetPointers.
   SpGEMM needs it: the size of C is only known after the compute step, so the
   descriptor is created empty, queried with flagsparseSpMatGetSize, and filled
   in here before flagsparseSpGEMM_copy writes the result. */
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseCsrSetPointers(flagsparseSpMatDescr_t descr,
                                                              void* csrRowOffsets,
                                                              void* csrColInd,
                                                              void* csrValues);

FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseDestroySpMat(flagsparseConstSpMatDescr_t descr);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpMatGetSize(flagsparseConstSpMatDescr_t descr,
                                                            int64_t* rows, int64_t* cols,
                                                            int64_t* nnz);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpMatGetFormat(flagsparseConstSpMatDescr_t descr,
                                                              flagsparseFormat_t* format);
/* Matrix attributes, cuSPARSE's cusparseSpMatAttribute_t. Both are consumed by
   the triangular solves (SpSV / SpSM), which have no safe default for either:
   which triangle to read and whether the diagonal is implicit change the
   answer, not the speed. */
typedef enum {
    FLAGSPARSE_SPMAT_FILL_MODE = 0,   /* data: flagsparseFillMode_t */
    FLAGSPARSE_SPMAT_DIAG_TYPE = 1    /* data: flagsparseDiagType_t */
} flagsparseSpMatAttribute_t;

FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpMatSetAttribute(flagsparseSpMatDescr_t descr,
                                                                 flagsparseSpMatAttribute_t attribute,
                                                                 const void* data,
                                                                 size_t dataSize);

FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseCreateDnVec(flagsparseDnVecDescr_t* descr,
                                                           int64_t size, void* values,
                                                           flagsparseDataType_t valueType);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseDestroyDnVec(flagsparseConstDnVecDescr_t descr);

FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseCreateDnMat(flagsparseDnMatDescr_t* descr,
                                                           int64_t rows, int64_t cols, int64_t ld,
                                                           void* values,
                                                           flagsparseDataType_t valueType,
                                                           flagsparseOrder_t order);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseDestroyDnMat(flagsparseConstDnMatDescr_t descr);

FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseCreateSpVec(flagsparseSpVecDescr_t* descr,
                                                           int64_t size, int64_t nnz,
                                                           void* indices, void* values,
                                                           flagsparseIndexType_t idxType,
                                                           flagsparseIndexBase_t idxBase,
                                                           flagsparseDataType_t valueType);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseDestroySpVec(flagsparseConstSpVecDescr_t descr);

/* ------------------------------------------------------------------- SpMV */
typedef enum {
    FLAGSPARSE_SPMV_ALG_DEFAULT = 0,
    FLAGSPARSE_SPMV_COO_ALG1    = 1,
    FLAGSPARSE_SPMV_CSR_ALG1    = 2,
    FLAGSPARSE_SPMV_CSR_ALG2    = 3,
    FLAGSPARSE_SPMV_COO_ALG2    = 4,
    FLAGSPARSE_SPMV_SELL_ALG1   = 5
} flagsparseSpMVAlg_t;

FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpMV_bufferSize(
    flagsparseHandle_t handle, flagsparseOperation_t opA, const void* alpha,
    flagsparseConstSpMatDescr_t matA, flagsparseConstDnVecDescr_t vecX,
    const void* beta, flagsparseDnVecDescr_t vecY,
    flagsparseDataType_t computeType, flagsparseSpMVAlg_t alg, size_t* bufferSize);

FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpMV(
    flagsparseHandle_t handle, flagsparseOperation_t opA, const void* alpha,
    flagsparseConstSpMatDescr_t matA, flagsparseConstDnVecDescr_t vecX,
    const void* beta, flagsparseDnVecDescr_t vecY,
    flagsparseDataType_t computeType, flagsparseSpMVAlg_t alg, void* externalBuffer);

FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpMV_preprocess(
    flagsparseHandle_t handle, flagsparseOperation_t opA, const void* alpha,
    flagsparseConstSpMatDescr_t matA, flagsparseConstDnVecDescr_t vecX,
    const void* beta, flagsparseDnVecDescr_t vecY,
    flagsparseDataType_t computeType, flagsparseSpMVAlg_t alg, void* externalBuffer);

/* ------------------------------------------------------------------- SpMM */
typedef enum {
    FLAGSPARSE_SPMM_ALG_DEFAULT = 0,
    FLAGSPARSE_SPMM_COO_ALG1    = 1,
    FLAGSPARSE_SPMM_COO_ALG2    = 2,
    FLAGSPARSE_SPMM_CSR_ALG1    = 4,
    FLAGSPARSE_SPMM_CSR_ALG2    = 5,
    FLAGSPARSE_SPMM_CSR_ALG3    = 12,
    FLAGSPARSE_SPMM_BLOCKED_ELL_ALG1 = 13
} flagsparseSpMMAlg_t;

FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpMM_bufferSize(
    flagsparseHandle_t handle, flagsparseOperation_t opA, flagsparseOperation_t opB,
    const void* alpha, flagsparseConstSpMatDescr_t matA, flagsparseConstDnMatDescr_t matB,
    const void* beta, flagsparseDnMatDescr_t matC,
    flagsparseDataType_t computeType, flagsparseSpMMAlg_t alg, size_t* bufferSize);

FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpMM_preprocess(
    flagsparseHandle_t handle, flagsparseOperation_t opA, flagsparseOperation_t opB,
    const void* alpha, flagsparseConstSpMatDescr_t matA, flagsparseConstDnMatDescr_t matB,
    const void* beta, flagsparseDnMatDescr_t matC,
    flagsparseDataType_t computeType, flagsparseSpMMAlg_t alg, void* externalBuffer);

FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpMM(
    flagsparseHandle_t handle, flagsparseOperation_t opA, flagsparseOperation_t opB,
    const void* alpha, flagsparseConstSpMatDescr_t matA, flagsparseConstDnMatDescr_t matB,
    const void* beta, flagsparseDnMatDescr_t matC,
    flagsparseDataType_t computeType, flagsparseSpMMAlg_t alg, void* externalBuffer);

/* ------------------------------------------------------------------- SpSV */
typedef enum {
    FLAGSPARSE_SPSV_ALG_DEFAULT = 0,
    /* Sliced-ELL routes. A FlagSparse extension: cuSPARSE has no SELL format,
       so these ids are ours and carry the same values the Python operator
       library uses. ALG1 is one program per row, ALG2 one per slice. */
    FLAGSPARSE_SPSV_SELL_ALG1   = 1,
    FLAGSPARSE_SPSV_SELL_ALG2   = 2
} flagsparseSpSVAlg_t;
typedef struct flagsparseSpSVDescr* flagsparseSpSVDescr_t;

FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpSV_createDescr(flagsparseSpSVDescr_t* descr);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpSV_destroyDescr(flagsparseSpSVDescr_t descr);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpSV_bufferSize(
    flagsparseHandle_t handle, flagsparseOperation_t opA, const void* alpha,
    flagsparseConstSpMatDescr_t matA, flagsparseConstDnVecDescr_t vecX,
    flagsparseDnVecDescr_t vecY, flagsparseDataType_t computeType,
    flagsparseSpSVAlg_t alg, flagsparseSpSVDescr_t spsvDescr, size_t* bufferSize);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpSV_analysis(
    flagsparseHandle_t handle, flagsparseOperation_t opA, const void* alpha,
    flagsparseConstSpMatDescr_t matA, flagsparseConstDnVecDescr_t vecX,
    flagsparseDnVecDescr_t vecY, flagsparseDataType_t computeType,
    flagsparseSpSVAlg_t alg, flagsparseSpSVDescr_t spsvDescr, void* externalBuffer);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpSV_solve(
    flagsparseHandle_t handle, flagsparseOperation_t opA, const void* alpha,
    flagsparseConstSpMatDescr_t matA, flagsparseConstDnVecDescr_t vecX,
    flagsparseDnVecDescr_t vecY, flagsparseDataType_t computeType,
    flagsparseSpSVAlg_t alg, flagsparseSpSVDescr_t spsvDescr);

/* ------------------------------------------------------------------- SpSM */
typedef enum { FLAGSPARSE_SPSM_ALG_DEFAULT = 0 } flagsparseSpSMAlg_t;
typedef struct flagsparseSpSMDescr* flagsparseSpSMDescr_t;

FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpSM_createDescr(flagsparseSpSMDescr_t* descr);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpSM_destroyDescr(flagsparseSpSMDescr_t descr);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpSM_bufferSize(
    flagsparseHandle_t handle, flagsparseOperation_t opA, flagsparseOperation_t opB,
    const void* alpha, flagsparseConstSpMatDescr_t matA, flagsparseConstDnMatDescr_t matB,
    flagsparseDnMatDescr_t matC, flagsparseDataType_t computeType,
    flagsparseSpSMAlg_t alg, flagsparseSpSMDescr_t spsmDescr, size_t* bufferSize);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpSM_analysis(
    flagsparseHandle_t handle, flagsparseOperation_t opA, flagsparseOperation_t opB,
    const void* alpha, flagsparseConstSpMatDescr_t matA, flagsparseConstDnMatDescr_t matB,
    flagsparseDnMatDescr_t matC, flagsparseDataType_t computeType,
    flagsparseSpSMAlg_t alg, flagsparseSpSMDescr_t spsmDescr, void* externalBuffer);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpSM_solve(
    flagsparseHandle_t handle, flagsparseOperation_t opA, flagsparseOperation_t opB,
    const void* alpha, flagsparseConstSpMatDescr_t matA, flagsparseConstDnMatDescr_t matB,
    flagsparseDnMatDescr_t matC, flagsparseDataType_t computeType,
    flagsparseSpSMAlg_t alg, flagsparseSpSMDescr_t spsmDescr);

/* ----------------------------------------------------------------- SpGEMM */
typedef enum { FLAGSPARSE_SPGEMM_DEFAULT = 0 } flagsparseSpGEMMAlg_t;
typedef struct flagsparseSpGEMMDescr* flagsparseSpGEMMDescr_t;

FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpGEMM_createDescr(flagsparseSpGEMMDescr_t* descr);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpGEMM_destroyDescr(flagsparseSpGEMMDescr_t descr);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpGEMM_workEstimation(
    flagsparseHandle_t handle, flagsparseOperation_t opA, flagsparseOperation_t opB,
    const void* alpha, flagsparseConstSpMatDescr_t matA, flagsparseConstSpMatDescr_t matB,
    const void* beta, flagsparseSpMatDescr_t matC, flagsparseDataType_t computeType,
    flagsparseSpGEMMAlg_t alg, flagsparseSpGEMMDescr_t spgemmDescr,
    size_t* bufferSize1, void* externalBuffer1);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpGEMM_compute(
    flagsparseHandle_t handle, flagsparseOperation_t opA, flagsparseOperation_t opB,
    const void* alpha, flagsparseConstSpMatDescr_t matA, flagsparseConstSpMatDescr_t matB,
    const void* beta, flagsparseSpMatDescr_t matC, flagsparseDataType_t computeType,
    flagsparseSpGEMMAlg_t alg, flagsparseSpGEMMDescr_t spgemmDescr,
    size_t* bufferSize2, void* externalBuffer2);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSpGEMM_copy(
    flagsparseHandle_t handle, flagsparseOperation_t opA, flagsparseOperation_t opB,
    const void* alpha, flagsparseConstSpMatDescr_t matA, flagsparseConstSpMatDescr_t matB,
    const void* beta, flagsparseSpMatDescr_t matC, flagsparseDataType_t computeType,
    flagsparseSpGEMMAlg_t alg, flagsparseSpGEMMDescr_t spgemmDescr);

/* ------------------------------------------------------------------ SDDMM */
typedef enum { FLAGSPARSE_SDDMM_ALG_DEFAULT = 0 } flagsparseSDDMMAlg_t;

FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSDDMM_bufferSize(
    flagsparseHandle_t handle, flagsparseOperation_t opA, flagsparseOperation_t opB,
    const void* alpha, flagsparseConstDnMatDescr_t matA, flagsparseConstDnMatDescr_t matB,
    const void* beta, flagsparseSpMatDescr_t matC, flagsparseDataType_t computeType,
    flagsparseSDDMMAlg_t alg, size_t* bufferSize);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSDDMM_preprocess(
    flagsparseHandle_t handle, flagsparseOperation_t opA, flagsparseOperation_t opB,
    const void* alpha, flagsparseConstDnMatDescr_t matA, flagsparseConstDnMatDescr_t matB,
    const void* beta, flagsparseSpMatDescr_t matC, flagsparseDataType_t computeType,
    flagsparseSDDMMAlg_t alg, void* externalBuffer);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseSDDMM(
    flagsparseHandle_t handle, flagsparseOperation_t opA, flagsparseOperation_t opB,
    const void* alpha, flagsparseConstDnMatDescr_t matA, flagsparseConstDnMatDescr_t matB,
    const void* beta, flagsparseSpMatDescr_t matC, flagsparseDataType_t computeType,
    flagsparseSDDMMAlg_t alg, void* externalBuffer);

/* -------------------------------------------------------- gather / scatter */
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseGather(flagsparseHandle_t handle,
                                                      flagsparseConstDnVecDescr_t vecY,
                                                      flagsparseSpVecDescr_t vecX);
FLAGSPARSE_EXPORT flagsparseStatus_t flagsparseScatter(flagsparseHandle_t handle,
                                                       flagsparseConstSpVecDescr_t vecX,
                                                       flagsparseDnVecDescr_t vecY);

#ifdef __cplusplus
}  /* extern "C" */
#endif

#endif  /* FLAGSPARSE_H_ */
