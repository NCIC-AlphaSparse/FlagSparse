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


// The dense prologue for accumulate-style routes.
//
// A route that builds its result with atomics (CSC op="non", BSR, the COO
// atomic variants) cannot fold `beta * y` into a store the way a
// one-program-per-row route can, because no single program owns the output
// element. Those routes get `y = beta * y` applied first, by this.

#pragma once

#include <cstdint>
#include <string>

#include "flagsparse.h"

namespace flagsparse {

// out = beta * out over an n_rows x n_cols strided block, in elements of
// `component_type`. A vector is n_rows = 1, stride_m = 0.
//
// beta == 0 stores a literal zero rather than multiplying: cuSPARSE defines it
// as "do not read the output", and an uninitialised buffer holding NaN would
// survive `0 * NaN`.
//
// Strides are in components, so a complex block doubles them; `stride_r` (the
// step from a real part to its imaginary part) is 1 for the interleaved layout
// every operand here uses.
flagsparseStatus_t scale_dense(flagsparseHandle_t handle, void* values,
                               flagsparseDataType_t component_type, bool is_complex,
                               int64_t n_rows, int64_t n_cols,
                               int64_t stride_m, int64_t stride_n,
                               double beta_re, double beta_im);

// dst = src over an n_rows x n_cols strided block, in elements of
// `component_type`. Both sides are strided, so any order / leading dimension
// combination works without a second code path.
//
// SpSM is what needs this: its solver works in place on a packed row-major work
// array, while cuSPARSE hands over a separate right-hand side and destination.
flagsparseStatus_t copy_dense(flagsparseHandle_t handle, const void* src, void* dst,
                              flagsparseDataType_t component_type, bool is_complex,
                              int64_t n_rows, int64_t n_cols,
                              int64_t src_stride_m, int64_t src_stride_n,
                              int64_t dst_stride_m, int64_t dst_stride_n);

}  // namespace flagsparse
