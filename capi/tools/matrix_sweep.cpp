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


// Every operator variant, over a corpus of real matrices.
//
// The per-operator benchmarks in ctest/benchmark/ sweep synthetic shapes to catch
// launch-config regressions. This sweeps the actual matrices, which is a
// different question: a random matrix has uniform rows, and the row-length skew
// of a real one is what decides whether a launch heuristic was a good guess.
//
// Every (variant, matrix) cell produces a row, and the status says what happened:
//
//   ok              measured
//   not_supported   the operator declined it on this backend -- recorded, not
//                   dropped, because a missing row reads the same as a pass
//   skipped_memory  the dense operands would not fit the budget; this is the
//                   sweep's limit, not the operator's, so it is named separately
//   skipped_shape   the matrix cannot feed this operator (SpGEMM needs A square)
//   failed          it ran and errored; the status name is kept

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "flagsparse.h"
#include "adaptor/adaptor.hpp"
#include "mmio.hpp"

namespace {

int g_warmup = 2;
int g_iters = 10;
double g_budget_mb = 8000.0;

struct Row {
    std::string matrix, op, format, dtype, status, detail;
    int64_t rows = 0, cols = 0, nnz = 0;
    double median_ms = 0.0, gflops = 0.0;
};
std::vector<Row> g_rows;

// ------------------------------------------------------------ device memory

struct Buf {
    void* p = nullptr;
    std::size_t bytes = 0;
    Buf() = default;
    explicit Buf(std::size_t n) : bytes(n) {
        flagsparse::adaptor::DevicePtr d = 0;
        if (n && flagsparse::adaptor::device_malloc(&d, n) == FLAGSPARSE_STATUS_SUCCESS)
            p = reinterpret_cast<void*>(d);
    }
    ~Buf() { if (p) flagsparse::adaptor::device_free(reinterpret_cast<std::uintptr_t>(p)); }
    Buf(const Buf&) = delete;
    Buf& operator=(const Buf&) = delete;
    Buf(Buf&& o) noexcept : p(o.p), bytes(o.bytes) { o.p = nullptr; o.bytes = 0; }
    Buf& operator=(Buf&& o) noexcept {
        if (this != &o) {
            if (p) flagsparse::adaptor::device_free(reinterpret_cast<std::uintptr_t>(p));
            p = o.p; bytes = o.bytes; o.p = nullptr; o.bytes = 0;
        }
        return *this;
    }
    bool valid() const { return bytes == 0 || p != nullptr; }
    void* get() const { return p; }
};

template <typename T>
Buf upload(const std::vector<T>& host) {
    Buf b(host.size() * sizeof(T));
    if (b.get() && !host.empty()) {
        if (flagsparse::adaptor::memcpy_h2d(reinterpret_cast<std::uintptr_t>(b.get()),
                                             host.data(), host.size() * sizeof(T)) !=
            FLAGSPARSE_STATUS_SUCCESS) {
            return Buf{};
        }
    }
    return b;
}

std::size_t value_bytes(flagsparseDataType_t t) {
    switch (t) {
        case FLAGSPARSE_R_32F: return 4;
        case FLAGSPARSE_R_64F:
        case FLAGSPARSE_C_32F: return 8;
        case FLAGSPARSE_C_64F: return 16;
        default: return 0;
    }
}
bool is_cplx(flagsparseDataType_t t) {
    return t == FLAGSPARSE_C_32F || t == FLAGSPARSE_C_64F;
}
const char* dtype_tag(flagsparseDataType_t t) {
    switch (t) {
        case FLAGSPARSE_R_32F: return "f32";
        case FLAGSPARSE_R_64F: return "f64";
        case FLAGSPARSE_C_32F: return "c32";
        default:               return "c64";
    }
}

// Values in the layout the API wants: one scalar per entry for a real dtype,
// interleaved real/imag for a complex one. The imaginary part is derived rather
// than random so a rerun measures the same thing.
Buf values_for(const std::vector<double>& v, flagsparseDataType_t t) {
    if (t == FLAGSPARSE_R_32F) return upload(std::vector<float>(v.begin(), v.end()));
    if (t == FLAGSPARSE_R_64F) return upload(v);
    std::vector<double> ri(v.size() * 2);
    for (std::size_t i = 0; i < v.size(); ++i) {
        ri[i * 2] = v[i];
        ri[i * 2 + 1] = v[i] * 0.25;
    }
    if (t == FLAGSPARSE_C_64F) return upload(ri);
    return upload(std::vector<float>(ri.begin(), ri.end()));
}

Buf filled(std::size_t count, flagsparseDataType_t t, double value) {
    const std::size_t scalars = count * (is_cplx(t) ? 2 : 1);
    if (t == FLAGSPARSE_R_64F || t == FLAGSPARSE_C_64F) {
        return upload(std::vector<double>(scalars, value));
    }
    return upload(std::vector<float>(scalars, static_cast<float>(value)));
}

// ------------------------------------------------------------------ timing

void record(Row row, const std::function<flagsparseStatus_t()>& once, double flops,
            flagsparseHandle_t handle) {
    const flagsparseStatus_t first = once();
    if (first != FLAGSPARSE_STATUS_SUCCESS) {
        const char* name = "?";
        flagsparseGetErrorName(first, &name);
        const char* detail = "";
        flagsparseGetLastErrorString(handle, &detail);
        row.status = (first == FLAGSPARSE_STATUS_NOT_SUPPORTED) ? "not_supported" : "failed";
        row.detail = std::string(name) + (detail && *detail ? std::string(": ") + detail : "");
        if (row.detail.size() > 160) row.detail.resize(160);
        g_rows.push_back(std::move(row));
        return;
    }
    for (int i = 0; i < g_warmup; ++i) once();
    flagsparse::adaptor::synchronize();
    std::vector<double> s;
    s.reserve(g_iters);
    for (int i = 0; i < g_iters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        once();
        flagsparse::adaptor::synchronize();
        s.push_back(std::chrono::duration<double, std::milli>(
                        std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(s.begin(), s.end());
    row.status = "ok";
    row.median_ms = s[s.size() / 2];
    row.gflops = (flops > 0 && row.median_ms > 0) ? flops / (row.median_ms * 1e6) : 0.0;
    g_rows.push_back(std::move(row));
}

void skip(Row row, const char* status, const std::string& detail) {
    row.status = status;
    row.detail = detail;
    g_rows.push_back(std::move(row));
}

Row base(const std::string& matrix, const mm::Csr& A, const char* op, const char* fmt,
         flagsparseDataType_t t) {
    Row r;
    r.matrix = matrix; r.op = op; r.format = fmt; r.dtype = dtype_tag(t);
    r.rows = A.rows; r.cols = A.cols; r.nnz = A.nnz;
    return r;
}

bool over_budget(double bytes) { return bytes / 1e6 > g_budget_mb; }

// A lower triangle with a diagonal that dominates its own row, built from the
// corpus matrix. Without the dominance a real matrix's triangle is often wildly
// ill-conditioned and the solve would measure the matrix, not the kernel.
mm::Csr lower_triangle(const mm::Csr& A) {
    mm::Csr L;
    L.rows = L.cols = std::min(A.rows, A.cols);
    L.indptr.assign(static_cast<std::size_t>(L.rows) + 1, 0);
    for (int64_t r = 0; r < L.rows; ++r) {
        double off = 0.0;
        const std::size_t diag_slot = L.indices.size();
        L.indices.push_back(static_cast<int32_t>(r));   // placeholder, fixed below
        L.values.push_back(0.0);
        for (int32_t p = A.indptr[static_cast<std::size_t>(r)];
             p < A.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
            const int32_t c = A.indices[static_cast<std::size_t>(p)];
            if (c >= r || c >= L.cols) continue;
            const double v = A.values[static_cast<std::size_t>(p)];
            L.indices.push_back(c);
            L.values.push_back(v);
            off += std::abs(v);
        }
        // The diagonal must come FIRST in storage order? No -- ascending columns
        // put it last for a lower row, which is what the scan expects. Move it.
        std::rotate(L.indices.begin() + static_cast<long>(diag_slot),
                    L.indices.begin() + static_cast<long>(diag_slot) + 1,
                    L.indices.end());
        std::rotate(L.values.begin() + static_cast<long>(diag_slot),
                    L.values.begin() + static_cast<long>(diag_slot) + 1,
                    L.values.end());
        L.indices.back() = static_cast<int32_t>(r);
        L.values.back() = 1.0 + off;
        L.indptr[static_cast<std::size_t>(r) + 1] = static_cast<int32_t>(L.indices.size());
    }
    L.nnz = static_cast<int64_t>(L.indices.size());
    L.ok = true;
    return L;
}

std::vector<int32_t> coo_rows_of(const mm::Csr& A) {
    std::vector<int32_t> row;
    row.reserve(static_cast<std::size_t>(A.nnz));
    for (int64_t r = 0; r < A.rows; ++r) {
        for (int32_t p = A.indptr[static_cast<std::size_t>(r)];
             p < A.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
            row.push_back(static_cast<int32_t>(r));
        }
    }
    return row;
}

}  // namespace

#include "sweep_ops.inc"

int main(int argc, char** argv) {
    std::string dir = "/workspace/matrix";
    std::string json_path = "matrix_sweep.json";
    std::vector<std::string> only_ops, only_matrices;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (a == "--matrix-dir") dir = next();
        else if (a == "--json") json_path = next();
        else if (a == "--warmup") g_warmup = std::atoi(next());
        else if (a == "--iters") g_iters = std::atoi(next());
        else if (a == "--budget-mb") g_budget_mb = std::atof(next());
        else if (a == "--op") only_ops.push_back(next());
        else if (a == "--matrix") only_matrices.push_back(next());
        else if (a == "--help") {
            std::cout << "usage: matrix_sweep [--matrix-dir D] [--json F] "
                         "[--warmup N] [--iters N] [--budget-mb M] "
                         "[--op NAME]... [--matrix NAME]...\n";
            return 0;
        }
    }

    flagsparseHandle_t handle = nullptr;
    if (flagsparseCreate(&handle) != FLAGSPARSE_STATUS_SUCCESS) {
        std::cerr << "no accelerator available\n";
        return 1;
    }
    std::cout << "backend: " << flagsparseGetBackendName() << "\n";

    std::vector<std::filesystem::path> files;
    for (const auto& e : std::filesystem::directory_iterator(dir)) {
        if (e.path().extension() == ".mtx") files.push_back(e.path());
    }
    std::sort(files.begin(), files.end());
    std::cout << "matrices: " << files.size() << "  in " << dir << "\n" << std::flush;

    for (const auto& path : files) {
        const std::string name = path.stem().string();
        if (!only_matrices.empty() &&
            std::find(only_matrices.begin(), only_matrices.end(), name) == only_matrices.end()) {
            continue;
        }
        const mm::Csr A = mm::read_csr(path.string());
        if (!A.ok) {
            std::cout << "  " << name << ": unreadable (" << A.error << ")\n" << std::flush;
            continue;
        }
        std::cout << "  " << name << "  " << A.rows << "x" << A.cols
                  << " nnz=" << A.nnz << std::flush;
        const std::size_t before = g_rows.size();
        run_all_ops(handle, name, A, only_ops);
        std::size_t ok = 0;
        for (std::size_t i = before; i < g_rows.size(); ++i) ok += g_rows[i].status == "ok";
        std::cout << "  -> " << ok << "/" << (g_rows.size() - before) << " ok\n" << std::flush;
    }

    std::ofstream out(json_path);
    out << "{\n  \"env\": {\"backend\": \"" << flagsparseGetBackendName()
        << "\", \"version\": " << FLAGSPARSE_VERSION << "},\n";
    out << "  \"config\": {\"warmup\": " << g_warmup << ", \"iters\": " << g_iters
        << ", \"statistic\": \"median\", \"budget_mb\": " << g_budget_mb << "},\n";
    out << "  \"result\": [\n";
    for (std::size_t i = 0; i < g_rows.size(); ++i) {
        const Row& r = g_rows[i];
        out << "    {\"matrix\": \"" << r.matrix << "\", \"op\": \"" << r.op
            << "\", \"format\": \"" << r.format << "\", \"dtype\": \"" << r.dtype
            << "\", \"status\": \"" << r.status << "\", \"rows\": " << r.rows
            << ", \"cols\": " << r.cols << ", \"nnz\": " << r.nnz;
        if (r.status == "ok") {
            out << ", \"median_ms\": " << std::setprecision(6) << r.median_ms
                << ", \"gflops\": " << r.gflops;
        }
        if (!r.detail.empty()) out << ", \"detail\": \"" << r.detail << "\"";
        out << ", \"baseline_ms\": null, \"speedup\": null}"
            << (i + 1 < g_rows.size() ? ",\n" : "\n");
    }
    std::size_t tally[5] = {0, 0, 0, 0, 0};
    for (const Row& r : g_rows) {
        if (r.status == "ok") tally[0]++;
        else if (r.status == "not_supported") tally[1]++;
        else if (r.status == "skipped_memory") tally[2]++;
        else if (r.status == "skipped_shape") tally[3]++;
        else tally[4]++;
    }
    out << "  ],\n  \"summary\": {\"rows\": " << g_rows.size()
        << ", \"ok\": " << tally[0] << ", \"not_supported\": " << tally[1]
        << ", \"skipped_memory\": " << tally[2] << ", \"skipped_shape\": " << tally[3]
        << ", \"failed\": " << tally[4] << "}\n}\n";
    std::cout << "\nrows " << g_rows.size() << "  ok " << tally[0]
              << "  not_supported " << tally[1] << "  skipped_memory " << tally[2]
              << "  skipped_shape " << tally[3] << "  failed " << tally[4]
              << "\nwrote " << json_path << "\n";
    flagsparseDestroy(handle);
    return 0;
}
