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

// hipSPARSE baseline (DCU / ROCm). The whole backend is the prefix table below;
// the body is baseline/_template.inc.
//
// NOT COMPILED OR RUN ON THIS MACHINE -- there is no ROCm SDK here, so this is
// written against hipSPARSE's generic API, which is a direct clone of cuSPARSE's
// (same call names, same descriptor model, HIPSPARSE_/HIP_ enum prefixes). Two
// differences are known and handled elsewhere rather than hidden here:
//   - hipSPARSE's SpMM implements op(A)=non only, so the transposed direction
//     legitimately returns a vendor status and lands as a blank speedup with a
//     reason. See docs/DCU.md.
//   - the header lives at <hipsparse/hipsparse.h>, not at the top level.
// The first build on a DCU box is what turns this from plausible into verified.

#include <hipsparse/hipsparse.h>

#define FS_SP_HEADER   <hipsparse/hipsparse.h>
#define FS_SP(sym)     hipsparse##sym
#define FS_SP_E(sym)   HIPSPARSE_##sym
#define FS_DT(sym)     HIP_##sym
#define FS_DT_TYPE     hipDataType
#define FS_SP_HANDLE   hipsparseHandle_t
#define FS_SP_NAME     "hipSPARSE"

#define FS_SP_OPERATION    hipsparseOperation_t
#define FS_SP_SPMAT        hipsparseSpMatDescr_t
#define FS_SP_DNMAT        hipsparseDnMatDescr_t
#define FS_SP_DNVEC        hipsparseDnVecDescr_t
#define FS_SP_SPVEC        hipsparseSpVecDescr_t
#define FS_SP_SPGEMM_DESCR hipsparseSpGEMMDescr_t
#define FS_SP_SPSV_DESCR   hipsparseSpSVDescr_t
#define FS_SP_SPSM_DESCR   hipsparseSpSMDescr_t

#include "../_template.inc"
