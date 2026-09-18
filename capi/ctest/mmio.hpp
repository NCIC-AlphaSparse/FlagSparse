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


// Matrix Market reader, enough of it for a benchmark corpus.
//
// Handles `coordinate` with field real / integer / pattern and symmetry
// general / symmetric / skew-symmetric. A stored-symmetric file holds only one
// triangle, so the mirror is generated -- a solver fed the half would be
// measured on half the work, which is the kind of quiet error a sweep exists to
// avoid.
//
// `array` (dense) and `complex` files are refused by name rather than guessed
// at: no corpus matrix uses them, and silently misreading one would be worse
// than not reading it.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace mm {

struct Csr {
    int64_t rows = 0, cols = 0, nnz = 0;
    std::vector<int32_t> indptr, indices;
    std::vector<double> values;
    bool ok = false;
    std::string error;
};

inline Csr read_csr(const std::string& path) {
    Csr out;
    std::FILE* f = std::fopen(path.c_str(), "r");
    if (f == nullptr) { out.error = "cannot open"; return out; }

    char line[4096];
    if (std::fgets(line, sizeof line, f) == nullptr) {
        out.error = "empty file"; std::fclose(f); return out;
    }
    std::string banner(line);
    auto has = [&](const char* w) { return banner.find(w) != std::string::npos; };
    if (!has("coordinate")) {
        out.error = "not a coordinate file"; std::fclose(f); return out;
    }
    if (has("complex")) {
        out.error = "complex Matrix Market files are not read here";
        std::fclose(f); return out;
    }
    const bool pattern = has("pattern");
    const bool symmetric = has("symmetric") || has("hermitian");
    const bool skew = has("skew");

    // Skip the comment block.
    do {
        if (std::fgets(line, sizeof line, f) == nullptr) {
            out.error = "no dimension line"; std::fclose(f); return out;
        }
    } while (line[0] == '%');

    long long m = 0, n = 0, entries = 0;
    if (std::sscanf(line, "%lld %lld %lld", &m, &n, &entries) != 3) {
        out.error = "bad dimension line"; std::fclose(f); return out;
    }
    out.rows = m; out.cols = n;

    struct Entry { int32_t r, c; double v; };
    std::vector<Entry> coo;
    coo.reserve(static_cast<std::size_t>(entries) * (symmetric ? 2 : 1));
    for (long long i = 0; i < entries; ++i) {
        if (std::fgets(line, sizeof line, f) == nullptr) {
            out.error = "truncated"; std::fclose(f); return out;
        }
        long long r = 0, c = 0; double v = 1.0;
        if (pattern) {
            if (std::sscanf(line, "%lld %lld", &r, &c) != 2) {
                out.error = "bad entry"; std::fclose(f); return out;
            }
        } else if (std::sscanf(line, "%lld %lld %lf", &r, &c, &v) != 3) {
            out.error = "bad entry"; std::fclose(f); return out;
        }
        // Matrix Market is 1-based.
        const int32_t rr = static_cast<int32_t>(r - 1), cc = static_cast<int32_t>(c - 1);
        if (rr < 0 || cc < 0 || rr >= m || cc >= n) {
            out.error = "index out of range"; std::fclose(f); return out;
        }
        coo.push_back({rr, cc, v});
        if (symmetric && rr != cc) coo.push_back({cc, rr, skew ? -v : v});
    }
    std::fclose(f);

    // Sort by (row, col) so the CSR comes out with ascending columns, which
    // every operator here assumes and the solves require.
    std::sort(coo.begin(), coo.end(), [](const Entry& a, const Entry& b) {
        return a.r != b.r ? a.r < b.r : a.c < b.c;
    });

    // Duplicates are adjacent after the sort, so summing them is a look at the
    // previous entry -- a canonical CSR has one entry per (row, column).
    out.indptr.assign(static_cast<std::size_t>(m) + 1, 0);
    out.indices.reserve(coo.size());
    out.values.reserve(coo.size());
    std::vector<int32_t> per_row(static_cast<std::size_t>(m), 0);
    for (std::size_t i = 0; i < coo.size(); ++i) {
        const Entry& e = coo[i];
        if (i > 0 && coo[i - 1].r == e.r && coo[i - 1].c == e.c) {
            out.values.back() += e.v;
            continue;
        }
        out.indices.push_back(e.c);
        out.values.push_back(e.v);
        per_row[static_cast<std::size_t>(e.r)]++;
    }
    for (int64_t r = 0; r < m; ++r) {
        out.indptr[static_cast<std::size_t>(r) + 1] =
            out.indptr[static_cast<std::size_t>(r)] + per_row[static_cast<std::size_t>(r)];
    }
    out.nnz = static_cast<int64_t>(out.indices.size());
    out.ok = true;
    return out;
}

}  // namespace mm
