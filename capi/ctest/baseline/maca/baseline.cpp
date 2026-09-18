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

// mcSPARSE baseline (沐曦 MetaX). The whole backend is the prefix table below;
// the body is baseline/_template.inc.
//
// NOT COMPILED ON THIS MACHINE -- no MACA SDK here. The names come from the
// vendor's CUDA-mirroring convention, which this repo already relies on elsewhere
// (mcruntime / mcr/mc_runtime.h in src/adaptor/CMakeLists.txt): the toolkit mirrors
// CUDA's API with a mc prefix, so the sparse library is mcsparse and its
// generic (descriptor) API matches cuSPARSE's call for call.
//
// A WRONG GUESS HERE IS NOT FATAL. ctest/baseline/CMakeLists.txt probes for the
// header and the library with find_path/find_library before selecting this slot;
// if either is absent the build falls back to baseline/none with a reason naming
// what was not found. That is why these are filled in rather than left blank --
// the failure mode that made guessing unacceptable (a link error naming neither
// the vendor nor the mistake) no longer applies.

#include <mcsparse.h>

// THREE TOKENS TO CHECK FIRST on a box that has the SDK. Everything else follows
// cuSPARSE call for call, so if the build gets past these three it will link:
//   1. the header path      <mcsparse.h>      -- may be namespaced, as hipSPARSE's is
//   2. the data-type enum   MC_R_32F        -- comes from the runtime, not the
//                                            sparse library (mcDataType_t)
//   3. the library name     libmcsparse.so
// ctest/baseline/CMakeLists.txt probes 1 and 3 and falls back to baseline/none
// with a reason when either is missing, so a wrong guess costs a blank speedup
// column, not a broken build.
#define FS_SP_HEADER   <mcsparse.h>
#define FS_SP(sym)     mcsparse##sym
#define FS_SP_E(sym)   MCSPARSE_##sym
#define FS_DT(sym)     MC_##sym
#define FS_DT_TYPE     mcDataType_t
#define FS_SP_HANDLE   mcsparseHandle_t
#define FS_SP_NAME     "mcSPARSE"

#define FS_SP_OPERATION    mcsparseOperation_t
#define FS_SP_SPMAT        mcsparseSpMatDescr_t
#define FS_SP_DNMAT        mcsparseDnMatDescr_t
#define FS_SP_DNVEC        mcsparseDnVecDescr_t
#define FS_SP_SPVEC        mcsparseSpVecDescr_t
#define FS_SP_SPGEMM_DESCR mcsparseSpGEMMDescr_t
#define FS_SP_SPSV_DESCR   mcsparseSpSVDescr_t
#define FS_SP_SPSM_DESCR   mcsparseSpSMDescr_t

#include "../_template.inc"
