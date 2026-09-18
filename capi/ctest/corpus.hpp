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

// The real-matrix corpus the benchmarks sweep.
//
// WHY REAL MATRICES. A synthetic matrix with uniform density gives every row the
// same length, which is the one property real sparse matrices never have. Row
// skew is what decides whether a row-per-thread kernel stalls behind its longest
// row, so a benchmark built on uniform rows measures the case that does not occur
// and misses the case that does. The corpus is Matrix Market files -- the
// SuiteSparse collection is the usual source -- pointed at by:
//
//     FLAGSPARSE_MATRIX_DIR=/path/to/mtx  ctest -R benchmark
//
// WHEN IT IS UNSET the benchmarks still run, on the synthetic shapes they used
// before, and every row is tagged corpus=synthetic. That is deliberate: a
// developer without the corpus gets a working `ctest`, and nobody reading the
// JSON can mistake those rows for measurements on real data.

#ifndef FLAGSPARSE_CTEST_CORPUS_HPP
#define FLAGSPARSE_CTEST_CORPUS_HPP

#include <string>
#include <vector>

#include "common.hpp"

namespace fstest {

struct CorpusEntry {
    std::string name;   // file stem, e.g. "ecology1"
    std::string path;
    CsrMatrix A;
};

// Every .mtx under $FLAGSPARSE_MATRIX_DIR, in name order, loaded once per process
// and shared by every case in that binary. Loading is the expensive part (a
// 5M-nonzero file is seconds of parsing), so doing it per case would dominate the
// suite's wall time.
//
// A file that fails to parse does NOT abort the sweep: it lands in failures()
// with the parser's reason and the sweep continues to the next file, because one
// malformed download should not cost the other 29 matrices.
const std::vector<CorpusEntry>& corpus();

// name -> why it could not be loaded. Benchmarks emit one `failed` row per entry
// so a matrix that never got measured is visible in the JSON rather than absent.
const std::vector<std::pair<std::string, std::string>>& corpus_failures();

// True when FLAGSPARSE_MATRIX_DIR was set and at least one matrix loaded.
bool corpus_is_real();

// What to tag rows with: "real" or "synthetic".
const char* corpus_tag();

}  // namespace fstest

#endif  // FLAGSPARSE_CTEST_CORPUS_HPP
