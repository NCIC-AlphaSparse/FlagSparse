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


// Entry points whose kernels have not been ported to the C++ dispatch layer yet.
//
// They exist, link, and return FLAGSPARSE_STATUS_NOT_SUPPORTED. That is a
// deliberate choice the spec requires (§4.4: an unsupported request returns a
// status, it does not abort) and it has a practical payoff: the ABI is complete
// from day one, so a caller can link against the final library and discover
// operator availability at runtime instead of at link time.
//
// Porting an operator means deleting its stub here and adding a src/ops/<op>.cpp
// shaped like src/ops/spmv.cpp. conf/operators.yaml tracks which is which.

#include "core/internal.hpp"

using namespace flagsparse;

namespace {
// Argument-eating no-op: keeps every stub honest about its signature without
// a pile of (void) casts, and guarantees the parameter list stays in sync with
// the header (a mismatched stub would fail to link, not silently diverge).
template <typename... Ts>
inline flagsparseStatus_t not_supported(Ts&&...) { return FLAGSPARSE_STATUS_NOT_SUPPORTED; }
}  // namespace

extern "C" {

#define FS_STUB(name, ...) \
    flagsparseStatus_t name(__VA_ARGS__)

/* SpSV: implemented in src/ops/spsv.cpp */

/* SpSM: implemented in src/ops/spsm.cpp */

/* SpGEMM: implemented in src/ops/spgemm.cpp */

/* SDDMM: implemented in src/ops/sddmm.cpp */

/* gather / scatter: implemented in src/ops/gather_scatter.cpp */

#undef FS_STUB

}  // extern "C"
