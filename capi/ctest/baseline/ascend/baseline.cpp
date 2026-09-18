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

// Ascend (Huawei CANN) baseline.
//
// THIS FILE DOES NOT USE _template.inc, AND THAT IS THE WHOLE POINT. The other
// four vendors ship a cuSPARSE-shaped generic API -- create-matrix, bufferSize,
// execute -- so their baselines are a 35-line prefix table over one shared body.
// CANN does not. It has sparse operators (aclnn*, and the torch_npu paths the
// Python probe uses), but no descriptor model to bind eight generic entry points
// to. Porting here means writing an op-by-op adaptor with a different shape.
//
// WHAT THIS FILE IS TODAY: the seam that adaptor will occupy, with each entry
// point naming the specific CANN operator a reviewer would bind it to, and
// returning `unavailable` with that name until someone does. That is more useful
// than routing Ascend to baseline/none, because `none` says "this backend has no
// baseline" while this says "this operator would map to THAT call, and nobody has
// written it yet" -- a different and much more actionable statement.
//
// WHY IT IS NOT JUST WRITTEN NOW. Binding these needs a CANN box to verify
// against: the aclnn sparse surface has changed across CANN releases, its
// operators take NZ/ND layout arguments with no cuSPARSE counterpart, and the
// per-op workspace call (aclnnXxxGetWorkspaceSize) has a different lifetime than
// this interface's bufferSize contract. Guessing all three from a machine with no
// CANN would produce a file that compiles nowhere and misleads everywhere.
//
// See docs/ASCEND.md for the problems to expect when doing it.

#include <string>

#include "baseline/baseline.hpp"

namespace fstest::baseline {
namespace {

// The CANN operator each entry point would bind to, named so the reason string
// is a starting point rather than a refusal. These are the aclnn surfaces the
// Python probe already exercises through torch_npu; a C baseline would call them
// directly (or through the aclnn C API) rather than through torch.
Status pending(const char* cann_op, const char* note = "") {
    std::string why = "Ascend: no cuSPARSE-shaped descriptor API; this would bind to ";
    why += cann_op;
    why += ", which has not been written";
    if (*note) {
        why += " (";
        why += note;
        why += ")";
    }
    return Status::no(why);
}

}  // namespace

bool available() { return false; }
const char* name() { return "none (CANN, op-by-op adaptor not written)"; }

Status spmv_csr(const DeviceCsr&, const void*, void*, const void*, const void*,
                flagsparseOperation_t, int, int, Timing*) {
    return pending("aclnnMm over a sparse-to-dense conversion, or torch_npu's "
                   "sparse mm",
                   "CANN has no dedicated SpMV; the Python probe multiplies by a "
                   "width-1 dense operand, which is SpMM n=1 and NOT comparable "
                   "to a real SpMV");
}

Status spmm_csr(const DeviceCsr&, const void*, int64_t, int64_t, void*, int64_t,
                const void*, const void*, flagsparseOperation_t,
                flagsparseOperation_t, int, int, Timing*) {
    return pending("aclnnSparseTensorDenseMatmul / torch_npu sparse mm");
}

Status sddmm_csr(const DeviceCsr&, const void*, int64_t, int64_t, const void*,
                 int64_t, const void*, const void*, int, int, Timing*) {
    return pending("no direct CANN counterpart",
                   "would be a dense matmul masked by the pattern, which measures "
                   "the dense library rather than a sparse kernel");
}

void free_csr(BaselineCsrOut*) {}   // nothing was ever allocated

Status spgemm_csr(const DeviceCsr&, const void*, const void*, int, int, Timing*,
                  BaselineCsrOut*) {
    return pending("no direct CANN counterpart");
}

Status spsv_csr(const DeviceCsr&, const void*, void*, const void*,
                flagsparseFillMode_t, flagsparseDiagType_t, flagsparseOperation_t,
                int, int, Timing*) {
    return pending("aclnnTriangularSolve over a densified operand",
                   "densifying changes the complexity class, so such a number is "
                   "not a sparse-solver baseline and must be labelled if used");
}

Status spsm_csr(const DeviceCsr&, const void*, int64_t, int64_t, void*, int64_t,
                const void*, flagsparseFillMode_t, flagsparseDiagType_t,
                flagsparseOperation_t, int, int, Timing*) {
    return pending("aclnnTriangularSolve (multi-RHS)",
                   "same densification caveat as SpSV");
}

Status gather(const void*, void*, const void*, int64_t, int64_t,
              flagsparseDataType_t, int, int, Timing*) {
    return pending("aclnnGather / aclnnIndexSelect");
}

Status scatter(void*, const void*, const void*, int64_t, int64_t,
               flagsparseDataType_t, int, int, Timing*) {
    return pending("aclnnScatter / aclnnIndexPut",
                   "index_put semantics differ on duplicate indices; the sweep "
                   "deduplicates, so a bound baseline must too or the two are not "
                   "measuring the same thing");
}

}  // namespace fstest::baseline
