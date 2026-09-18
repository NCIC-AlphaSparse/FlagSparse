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

// The baseline for a backend with no vendor generic-sparse library wired.
//
// This is a REAL ANSWER, not a stub that got left behind. Three backends land
// here for three different reasons, and the reason string says which, so a blank
// speedup column is never ambiguous:
//
//   MACA (MetaX), MUSA (Moore Threads)
//       Both vendors ship a sparse library, but its header path and library name
//       are not recorded anywhere this repo can read -- deps/libtriton_jit's
//       cmake/Backend*.cmake carry only the driver/runtime names, because the JIT
//       bridge never calls a sparse library. Guessing is what this repo refuses
//       to do (src/adaptor/CMakeLists.txt:26): a wrong name fails at link time
//       with a missing symbol that names neither the vendor nor the mistake.
//
//   XPU (Kunlunxin)
//       Kunlunxin does not ship a cuSPARSE-shaped generic sparse library at all.
//       Its math library is XDNN, whose sparse coverage is a handful of fixed
//       ops rather than a descriptor API, so there is nothing to bind to the
//       eight entry points here even with the SDK in hand.
//
//   Ascend (Huawei)
//       CANN has no generic sparse API either; the comparable numbers there come
//       from torch_npu/aclnn per operator, which is a different measurement shape
//       and lives in benchmark/benchmark_ascend_probe.py on the Python side.
//
// Filling one of these in is: add its entry to the table in CMakeLists.txt and
// drop in baseline/<name>/baseline.cpp with the prefix macros (35 lines -- see
// baseline/cuda/baseline.cpp). Nothing else changes.

#include <string>

#include "baseline/baseline.hpp"

namespace fstest::baseline {
namespace {
Status none() { return Status::no(std::string(FLAGSPARSE_BASELINE_REASON)); }
}  // namespace

bool available() { return false; }
const char* name() { return "none"; }

Status spmv_csr(const DeviceCsr&, const void*, void*, const void*, const void*,
                flagsparseOperation_t, int, int, Timing*) { return none(); }
Status spmm_csr(const DeviceCsr&, const void*, int64_t, int64_t, void*, int64_t,
                const void*, const void*, flagsparseOperation_t,
                flagsparseOperation_t, int, int, Timing*) { return none(); }
Status sddmm_csr(const DeviceCsr&, const void*, int64_t, int64_t, const void*,
                 int64_t, const void*, const void*, int, int, Timing*) { return none(); }
void free_csr(BaselineCsrOut*) {}   // nothing was ever allocated

Status spgemm_csr(const DeviceCsr&, const void*, const void*, int, int, Timing*,
                  BaselineCsrOut*) { return none(); }
Status spsv_csr(const DeviceCsr&, const void*, void*, const void*,
                flagsparseFillMode_t, flagsparseDiagType_t, flagsparseOperation_t,
                int, int, Timing*) { return none(); }
Status spsm_csr(const DeviceCsr&, const void*, int64_t, int64_t, void*, int64_t,
                const void*, flagsparseFillMode_t, flagsparseDiagType_t,
                flagsparseOperation_t, int, int, Timing*) { return none(); }
Status gather(const void*, void*, const void*, int64_t, int64_t,
              flagsparseDataType_t, int, int, Timing*) { return none(); }
Status scatter(void*, const void*, const void*, int64_t, int64_t,
               flagsparseDataType_t, int, int, Timing*) { return none(); }

}  // namespace fstest::baseline
