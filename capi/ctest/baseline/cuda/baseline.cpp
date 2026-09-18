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

// cuSPARSE baseline. The whole backend is the prefix table below; the body is
// baseline/_template.inc.

#include <cusparse.h>

#define FS_SP_HEADER   <cusparse.h>
#define FS_SP(sym)     cusparse##sym
#define FS_SP_E(sym)   CUSPARSE_##sym
#define FS_DT(sym)     CUDA_##sym
#define FS_DT_TYPE     cudaDataType_t
#define FS_SP_HANDLE   cusparseHandle_t
#define FS_SP_NAME     "cuSPARSE"

#define FS_SP_OPERATION    cusparseOperation_t
#define FS_SP_SPMAT        cusparseSpMatDescr_t
#define FS_SP_DNMAT        cusparseDnMatDescr_t
#define FS_SP_DNVEC        cusparseDnVecDescr_t
#define FS_SP_SPVEC        cusparseSpVecDescr_t
#define FS_SP_SPGEMM_DESCR cusparseSpGEMMDescr_t
#define FS_SP_SPSV_DESCR   cusparseSpSVDescr_t
#define FS_SP_SPSM_DESCR   cusparseSpSMDescr_t

#include "../_template.inc"
