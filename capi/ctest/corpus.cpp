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

#include "corpus.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <mutex>
#include <stdexcept>

#include "mmio.hpp"

namespace fstest {
namespace {

struct Loaded {
    std::vector<CorpusEntry> entries;
    std::vector<std::pair<std::string, std::string>> failures;
    bool real = false;
};

// The synthetic fallback. These are the shapes the benchmarks used before the
// corpus existed, kept so an unset FLAGSPARSE_MATRIX_DIR still produces a
// comparable sweep rather than an empty one.
void add_synthetic(Loaded& out) {
    struct Shape { const char* name; int64_t n; double density; };
    static constexpr Shape kShapes[] = {
        {"synthetic_8k_d0.001",   8192, 0.001},
        {"synthetic_32k_d0.0005", 32768, 0.0005},
        {"synthetic_128k_d0.0001", 131072, 0.0001},
    };
    uint32_t seed = 12345;
    for (const auto& s : kShapes) {
        CorpusEntry e;
        e.name = s.name;
        e.path = "(synthetic)";
        e.A = random_csr(s.n, s.n, s.density, seed++);
        out.entries.push_back(std::move(e));
    }
}

Loaded load_once() {
    Loaded out;
    const char* dir = std::getenv("FLAGSPARSE_MATRIX_DIR");
    if (!dir || !*dir) {
        std::cerr << "[corpus] FLAGSPARSE_MATRIX_DIR unset; using synthetic shapes. "
                     "Rows are tagged corpus=synthetic.\n";
        add_synthetic(out);
        return out;
    }

    std::error_code ec;
    std::vector<std::filesystem::path> files;
    for (const auto& de : std::filesystem::directory_iterator(dir, ec)) {
        if (de.is_regular_file() && de.path().extension() == ".mtx") {
            files.push_back(de.path());
        }
    }
    if (ec) {
        std::cerr << "[corpus] cannot read " << dir << ": " << ec.message()
                  << "; using synthetic shapes.\n";
        add_synthetic(out);
        return out;
    }
    std::sort(files.begin(), files.end());

    for (const auto& f : files) {
        const std::string stem = f.stem().string();
        try {
            mm::Csr m = mm::read_csr(f.string());
            CorpusEntry e;
            e.name = stem;
            e.path = f.string();
            e.A.rows = m.rows;
            e.A.cols = m.cols;
            e.A.nnz = m.nnz;
            e.A.indptr = std::move(m.indptr);
            e.A.indices = std::move(m.indices);
            e.A.values = std::move(m.values);
            out.entries.push_back(std::move(e));
        } catch (const std::exception& ex) {
            // One bad file must not cost the rest of the corpus.
            out.failures.emplace_back(stem, ex.what());
            std::cerr << "[corpus] " << stem << ": " << ex.what() << " (skipped)\n";
        }
    }

    if (out.entries.empty()) {
        std::cerr << "[corpus] no matrix loaded from " << dir
                  << "; using synthetic shapes.\n";
        add_synthetic(out);
        return out;
    }
    out.real = true;
    std::cerr << "[corpus] " << out.entries.size() << " matrices from " << dir;
    if (!out.failures.empty()) std::cerr << " (" << out.failures.size() << " unreadable)";
    std::cerr << "\n";
    return out;
}

const Loaded& loaded() {
    static const Loaded g = load_once();
    return g;
}

}  // namespace

const std::vector<CorpusEntry>& corpus() { return loaded().entries; }
const std::vector<std::pair<std::string, std::string>>& corpus_failures() {
    return loaded().failures;
}
bool corpus_is_real() { return loaded().real; }
const char* corpus_tag() { return loaded().real ? "real" : "synthetic"; }

}  // namespace fstest
