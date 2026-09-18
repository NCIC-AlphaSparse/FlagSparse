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


#include "common.hpp"

#include "corpus.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>

// The adaptor is internal, but the tests need device memory and the whole point
// of routing through it is that they stay vendor-neutral.
#include "adaptor/adaptor.hpp"

namespace fstest {

namespace ad = flagsparse::adaptor;

flagsparseStatus_t dev_alloc(void** ptr, std::size_t bytes) {
    ad::DevicePtr p = 0;
    const flagsparseStatus_t st = ad::device_malloc(&p, bytes);
    *ptr = reinterpret_cast<void*>(p);
    return st;
}
void dev_free(void* ptr) { ad::device_free(reinterpret_cast<ad::DevicePtr>(ptr)); }
flagsparseStatus_t to_device(void* dst, const void* src, std::size_t bytes) {
    return ad::memcpy_h2d(reinterpret_cast<ad::DevicePtr>(dst), src, bytes);
}
flagsparseStatus_t to_host(void* dst, const void* src, std::size_t bytes) {
    return ad::memcpy_d2h(dst, reinterpret_cast<ad::DevicePtr>(src), bytes);
}
void dev_sync() { ad::synchronize(); }

CsrMatrix random_csr(int64_t rows, int64_t cols, double density, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double> value(0.0, 1.0);

    CsrMatrix A;
    A.rows = rows; A.cols = cols;
    A.indptr.assign(static_cast<std::size_t>(rows) + 1, 0);
    for (int64_t r = 0; r < rows; ++r) {
        // Row-to-row variation on purpose: a uniform nnz per row would hide a
        // segment count sized from the average instead of the maximum.
        const double row_density = density * (0.25 + 1.5 * unit(rng));
        for (int64_t c = 0; c < cols; ++c) {
            if (unit(rng) < row_density) {
                A.indices.push_back(static_cast<int32_t>(c));
                A.values.push_back(value(rng));
            }
        }
        A.indptr[static_cast<std::size_t>(r) + 1] = static_cast<int32_t>(A.indices.size());
    }
    A.nnz = static_cast<int64_t>(A.indices.size());
    return A;
}

SellMatrix random_sell_lower(int64_t n, int64_t slice_size, double density,
                             uint32_t seed) {
    SellMatrix S;
    S.n = n; S.slice_size = slice_size;
    S.dense.assign(static_cast<std::size_t>(n * n), 0.0);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double> value(0.0, 1.0);
    const double expected = std::max(1.0, density * static_cast<double>(n));

    // Row contents first, so the slice width can be the max over its rows.
    std::vector<std::vector<std::pair<int32_t, double>>> rows(
        static_cast<std::size_t>(n));
    for (int64_t r = 0; r < n; ++r) {
        for (int64_t c = 0; c < r; ++c) {
            if (unit(rng) >= density) continue;
            const double v = value(rng) / (4.0 * expected);
            rows[static_cast<std::size_t>(r)].emplace_back(static_cast<int32_t>(c), v);
            S.dense[static_cast<std::size_t>(r * n + c)] = v;
        }
        const double d = 2.0 + std::abs(value(rng));
        rows[static_cast<std::size_t>(r)].emplace_back(static_cast<int32_t>(r), d);
        S.dense[static_cast<std::size_t>(r * n + r)] = d;
    }

    const int64_t n_slices = (n + slice_size - 1) / slice_size;
    S.offsets.assign(static_cast<std::size_t>(n_slices) + 1, 0);
    for (int64_t s = 0; s < n_slices; ++s) {
        std::size_t width = 0;
        for (int64_t lane = 0; lane < slice_size; ++lane) {
            const int64_t r = s * slice_size + lane;
            if (r < n) width = std::max(width, rows[static_cast<std::size_t>(r)].size());
        }
        // Column-major within a slice: slot-major, lane-minor, which is what
        // makes one slot of every row a coalesced load.
        for (std::size_t slot = 0; slot < width; ++slot) {
            for (int64_t lane = 0; lane < slice_size; ++lane) {
                const int64_t r = s * slice_size + lane;
                const bool present =
                    r < n && slot < rows[static_cast<std::size_t>(r)].size();
                S.cols.push_back(present
                                     ? rows[static_cast<std::size_t>(r)][slot].first
                                     : -1);
                S.values.push_back(present
                                       ? rows[static_cast<std::size_t>(r)][slot].second
                                       : 0.0);
            }
        }
        S.offsets[static_cast<std::size_t>(s) + 1] =
            static_cast<int32_t>(S.cols.size());
    }
    return S;
}

TriMatrix random_triangular(int64_t n, double density, bool lower, bool unit_diag,
                            uint32_t seed) {
    TriMatrix T;
    T.n = n; T.lower = lower; T.unit_diag = unit_diag;
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::normal_distribution<double> value(0.0, 1.0);
    const double expected = std::max(1.0, density * static_cast<double>(n));
    T.indptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (int64_t r = 0; r < n; ++r) {
        const int64_t begin = lower ? 0 : r;
        const int64_t end   = lower ? r + 1 : n;
        for (int64_t c = begin; c < end; ++c) {
            if (c == r) {
                // A unit diagonal is implicit: cuSPARSE does not store it.
                if (unit_diag) continue;
                T.indices.push_back(static_cast<int32_t>(c));
                T.values.push_back(2.0 + std::abs(value(rng)));
                continue;
            }
            if (unit(rng) >= density) continue;
            T.indices.push_back(static_cast<int32_t>(c));
            T.values.push_back(value(rng) / (4.0 * expected));
        }
        T.indptr[static_cast<std::size_t>(r) + 1] = static_cast<int32_t>(T.indices.size());
    }
    return T;
}

namespace {
template <typename M>
std::vector<int32_t> expand_rows(const M& m, int64_t rows) {
    std::vector<int32_t> row;
    row.reserve(static_cast<std::size_t>(m.indices.size()));
    for (int64_t r = 0; r < rows; ++r) {
        for (int32_t p = m.indptr[static_cast<std::size_t>(r)];
             p < m.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
            row.push_back(static_cast<int32_t>(r));
        }
    }
    return row;
}
}  // namespace

std::vector<int32_t> coo_row_indices_of(const CsrMatrix& A) {
    return expand_rows(A, A.rows);
}
std::vector<int32_t> coo_row_indices_of(const TriMatrix& T) {
    return expand_rows(T, T.n);
}

std::vector<double> spmv_reference(const CsrMatrix& A, const std::vector<double>& x,
                                   double alpha, double beta,
                                   const std::vector<double>& y_in) {
    std::vector<double> y(static_cast<std::size_t>(A.rows), 0.0);
    for (int64_t r = 0; r < A.rows; ++r) {
        double acc = 0.0;
        for (int32_t k = A.indptr[static_cast<std::size_t>(r)];
             k < A.indptr[static_cast<std::size_t>(r) + 1]; ++k) {
            acc += A.values[static_cast<std::size_t>(k)] *
                   x[static_cast<std::size_t>(A.indices[static_cast<std::size_t>(k)])];
        }
        // beta == 0 must ignore y entirely, matching cuSPARSE: an uninitialised
        // y would otherwise turn into NaN through 0 * NaN.
        y[static_cast<std::size_t>(r)] =
            alpha * acc + (beta == 0.0 ? 0.0 : beta * y_in[static_cast<std::size_t>(r)]);
    }
    return y;
}

std::vector<double> spmm_reference(const CsrMatrix& A, const std::vector<double>& B,
                                   int64_t n, double alpha, double beta,
                                   const std::vector<double>& c_in) {
    std::vector<double> C(static_cast<std::size_t>(A.rows * n), 0.0);
    for (int64_t r = 0; r < A.rows; ++r) {
        for (int64_t j = 0; j < n; ++j) {
            double acc = 0.0;
            for (int32_t p = A.indptr[static_cast<std::size_t>(r)];
                 p < A.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
                const int64_t col = A.indices[static_cast<std::size_t>(p)];
                acc += A.values[static_cast<std::size_t>(p)] *
                       B[static_cast<std::size_t>(col * n + j)];
            }
            const std::size_t idx = static_cast<std::size_t>(r * n + j);
            // beta == 0 must ignore C entirely, matching cuSPARSE.
            C[idx] = alpha * acc + (beta == 0.0 ? 0.0 : beta * c_in[idx]);
        }
    }
    return C;
}

Tolerance default_tolerance(flagsparseDataType_t dtype) {
    switch (dtype) {
        case FLAGSPARSE_R_16F:
        case FLAGSPARSE_R_16BF: return {1e-2, 1e-3};
        case FLAGSPARSE_R_32F:
        case FLAGSPARSE_C_32F:  return {1e-5, 1e-6};
        case FLAGSPARSE_R_64F:
        case FLAGSPARSE_C_64F:  return {1e-12, 1e-13};
        default:                return {1e-5, 1e-6};
    }
}

// Spec §6.3.1: fp32 sparse accumulation is order-dependent, so a matrix that
// fails at 1e-5 is not necessarily wrong. The caller retries here and reports
// PASS(relaxed) rather than turning a known numerical property into a failure.
Tolerance relaxed_tolerance(flagsparseDataType_t dtype) {
    switch (dtype) {
        case FLAGSPARSE_R_64F:
        case FLAGSPARSE_C_64F: return {1e-10, 1e-11};
        default:               return {1e-3, 1e-4};
    }
}

double max_error_ratio(const std::vector<double>& actual, const std::vector<double>& ref,
                       Tolerance tol) {
    if (actual.size() != ref.size()) return std::numeric_limits<double>::infinity();
    double worst = 0.0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(actual[i])) return std::numeric_limits<double>::infinity();
        const double allow = tol.atol + tol.rtol * std::abs(ref[i]);
        worst = std::max(worst, std::abs(actual[i] - ref[i]) / allow);
    }
    return worst;
}

void print_backend_banner() {
    std::cout << "[  BACKEND  ] " << flagsparseGetBackendName()
              << "  flagsparse " << FLAGSPARSE_VERSION << std::endl;
}

std::string device_arch() {
    return ad::device_architecture(0);
}

bool BenchReport::measure(BenchRow row, const std::function<flagsparseStatus_t()>& once,
                          double flops) {
    // The first call also pays JIT compilation; it decides the status and stays
    // out of the samples.
    const flagsparseStatus_t first = once();
    if (first == FLAGSPARSE_STATUS_NOT_SUPPORTED) {
        row.status = "not_supported";
        row.detail = "operator declined this configuration on this backend";
        rows_.push_back(std::move(row));
        return false;
    }
    if (first != FLAGSPARSE_STATUS_SUCCESS) {
        row.status = "failed";
        row.detail = status_name(first);
        rows_.push_back(std::move(row));
        return false;
    }
    for (int i = 0; i < kWarmup; ++i) once();
    dev_sync();

    std::vector<double> samples;
    samples.reserve(kIters);
    for (int i = 0; i < kIters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        once();
        dev_sync();
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    row.status = "ok";
    row.median_ms = samples[samples.size() / 2];
    row.gflops = (flops > 0.0 && row.median_ms > 0.0)
                     ? flops / (row.median_ms * 1e6)
                     : 0.0;
    rows_.push_back(std::move(row));
    return true;
}

namespace {

// Baseline reasons are vendor strings and carry quotes and backslashes; emitting
// them raw produces JSON that a reader silently truncates at the first quote.
std::string json_escape(const std::string& in) {
    std::string out;
    out.reserve(in.size() + 8);
    for (char c : in) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;
        }
    }
    return out;
}

}  // namespace

bool BenchReport::measure_vs_baseline(
    BenchRow row, const std::function<flagsparseStatus_t()>& once,
    const std::function<double(bool)>& verify,
    const std::function<baseline::Status(baseline::Timing*)>& base, double flops,
    bool baseline_writes_output) {
    // 1. Our side. The first call also pays JIT compilation; it decides the
    //    status and stays out of the samples.
    const flagsparseStatus_t first = once();
    if (first == FLAGSPARSE_STATUS_NOT_SUPPORTED) {
        row.status = "not_supported";
        row.detail = "operator declined this configuration on this backend";
        rows_.push_back(std::move(row));
        return false;
    }
    if (first != FLAGSPARSE_STATUS_SUCCESS) {
        row.status = "failed";
        row.detail = std::string("operator returned ") + status_name(first);
        rows_.push_back(std::move(row));
        return false;
    }

    // 2. BOTH ratios, before timing and before the baseline runs.
    //
    //    Before timing, because the timing loop re-runs the operator and an
    //    accumulating output would no longer hold the value the oracle was
    //    computed for. Before the baseline, because the baseline overwrites the
    //    same buffer -- and spec 6.3.1's relaxed verdict needs OUR relaxed ratio,
    //    which is unrecoverable once that happens.
    const double strict = verify(false);
    const bool checkable = strict >= 0;
    const double relaxed = checkable ? verify(true) : -1.0;
    row.error_ratio = checkable ? strict : 0.0;
    row.relaxed_error_ratio = (relaxed >= 0) ? relaxed : 0.0;

    // 3. Our timing. Done even when the answer is out of tolerance: a wrong
    //    result that is also slow is worth knowing about, and the row records
    //    both rather than hiding the timing behind the accuracy verdict.
    for (int i = 0; i < kWarmup; ++i) once();
    dev_sync();
    std::vector<double> samples;
    samples.reserve(kIters);
    for (int i = 0; i < kIters; ++i) {
        const auto t0 = std::chrono::steady_clock::now();
        once();
        dev_sync();
        const auto t1 = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    std::sort(samples.begin(), samples.end());
    row.median_ms = samples[samples.size() / 2];
    if (flops > 0 && row.median_ms > 0) row.gflops = flops / (row.median_ms * 1e6);

    // 4. The vendor. Same warmup/iters and the same median, so the two divide.
    baseline::Timing bt;
    const baseline::Status bs = base(&bt);
    if (!bs.ok) {
        row.baseline_status = baseline::available() ? "failed" : "unavailable";
        row.baseline_detail = bs.reason;
    } else {
        row.baseline_status = "ok";
        row.baseline_ms = bt.median_ms;
        // 5. The baseline's OWN answer, now sitting in the output buffer.
        if (baseline_writes_output && checkable) {
            const double b_strict = verify(false);
            row.baseline_error_ratio = b_strict >= 0 ? b_strict : 0.0;
            row.baseline_accuracy = (b_strict >= 0 && b_strict <= 1.0) ? "pass" : "fail";
        }
    }

    // 6. The verdict (spec 6.3, and 6.3.1 for the relaxed tier).
    if (!checkable) {
        row.accuracy = "unchecked";
    } else if (strict <= 1.0) {
        row.accuracy = "pass";
    } else if (row.baseline_accuracy == "fail" && relaxed >= 0 && relaxed <= 1.0) {
        // Spec 6.3.1: the vendor's kernel missed the strict tolerance on this
        // matrix too, which is the evidence that the matrix is ill-conditioned
        // rather than that we are wrong. Only then is the relaxed tier allowed.
        row.accuracy = "pass_relaxed";
        row.detail = "PASS(relaxed) per spec 6.3.1: strict ratio " +
                     std::to_string(strict) + ", relaxed " + std::to_string(relaxed) +
                     "; " + baseline::name() + " also failed strict (ratio " +
                     std::to_string(row.baseline_error_ratio) + ")";
    } else {
        row.accuracy = "fail";
        row.status = "failed";
        row.detail = "result outside tolerance: strict ratio " +
                     std::to_string(strict);
        if (relaxed >= 0 && relaxed <= 1.0) {
            // The relaxed tier would have passed, but 6.3.1 gates it on the
            // baseline failing too. Say which half is missing rather than
            // leaving a reviewer to guess why the softer tier was not applied.
            row.detail += "; relaxed would pass (" + std::to_string(relaxed) + ") but ";
            row.detail += (row.baseline_accuracy == "pass")
                              ? std::string(baseline::name()) + " passed strict, so "
                                "spec 6.3.1 does not apply"
                              : "there is no vendor baseline to confirm the matrix "
                                "is hard for both (spec 6.3.1 requires it)";
        }
        rows_.push_back(std::move(row));
        return false;
    }

    row.status = "ok";
    // The gate: a speedup is written only over an answer we checked and believed.
    // pass_relaxed counts -- 6.3.1 calls it a PASS -- but "unchecked" does not.
    if (row.baseline_status == "ok" && row.baseline_ms > 0 && row.median_ms > 0 &&
        (row.accuracy == "pass" || row.accuracy == "pass_relaxed")) {
        row.speedup = row.baseline_ms / row.median_ms;
    }
    rows_.push_back(std::move(row));
    return true;
}

void BenchReport::skip(BenchRow row, const std::string& status,
                       const std::string& detail) {
    row.status = status;
    row.detail = detail;
    rows_.push_back(std::move(row));
}

// The accuracy artifact: one row per (variant, matrix) with the ratio that
// decided it. Separate from the benchmark JSON because a precision reviewer
// wants the ratios without the timings, and because summary.json names a
// data_file per phase.
void write_accuracy_json(const std::string& op, const std::vector<BenchRow>& rows) {
    const char* dir = std::getenv("FLAGSPARSE_BENCH_OUT");
    const std::string path =
        std::string(dir ? dir : ".") + "/" + op + "_accuracy.json";
    std::ofstream out(path);
    if (!out) return;
    const std::time_t now = std::time(nullptr);
    char stamp[64];
    std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%S", std::gmtime(&now));

    out << "{\n  \"timestamp\": \"" << stamp << "\",\n";
    out << "  \"operator\": \"" << op << "\",\n";
    out << "  \"env\": {\"backend\": \"" << flagsparseGetBackendName()
        << "\", \"arch\": \"" << device_arch()
        << "\", \"version\": " << FLAGSPARSE_VERSION << "},\n";
    out << "  \"reference\": \"host_fp64\",\n";
    out << "  \"criterion\": \"max(|actual-ref|/(atol+rtol*|ref|)) <= 1  (spec 6.3)\",\n";
    out << "  \"result\": [\n";
    for (std::size_t i = 0; i < rows.size(); ++i) {
        const BenchRow& r = rows[i];
        out << "    {\"name\": \"" << r.name << "\"";
        for (const auto& kv : r.tags) {
            out << ", \"" << kv.first << "\": \"" << kv.second << "\"";
        }
        out << ", \"accuracy\": \"" << r.accuracy << "\"";
        if (r.accuracy != "unchecked") {
            out << ", \"error_ratio\": " << std::setprecision(6) << r.error_ratio
                << ", \"relaxed_error_ratio\": " << r.relaxed_error_ratio;
        } else {
            out << ", \"error_ratio\": null, \"relaxed_error_ratio\": null";
        }
        out << ", \"baseline_accuracy\": \"" << r.baseline_accuracy << "\"";
        if (r.baseline_accuracy != "unchecked") {
            out << ", \"baseline_error_ratio\": " << std::setprecision(6)
                << r.baseline_error_ratio;
        }
        out << ", \"status\": \"" << r.status << "\"";
        if (!r.detail.empty()) {
            out << ", \"detail\": \"" << json_escape(r.detail) << "\"";
        }
        out << "}" << (i + 1 < rows.size() ? ",\n" : "\n");
    }
    std::size_t passed = 0, failed = 0, unchecked = 0, skipped = 0, no_test = 0;
    for (const BenchRow& r : rows) {
        if (r.accuracy == "pass") ++passed;
        else if (r.accuracy == "fail") ++failed;
        else ++unchecked;
        if (r.status == "not_implemented_in_test") ++no_test;
        else if (r.status == "not_supported" ||
                 r.status.rfind("skipped", 0) == 0) ++skipped;
    }
    out << "  ],\n  \"summary\": {\"rows\": " << rows.size()
        << ", \"passed\": " << passed << ", \"failed\": " << failed
        << ", \"unchecked\": " << unchecked << ", \"skipped\": " << skipped
        << ", \"not_implemented_in_test\": " << no_test << "}\n}\n";
    std::cout << "wrote " << path << "  (" << passed << " passed, " << failed
              << " failed, " << unchecked << " unchecked)" << std::endl;
}

void BenchReport::write() const {
    const char* dir = std::getenv("FLAGSPARSE_BENCH_OUT");
    const std::string path =
        std::string(dir ? dir : ".") + "/" + op_ + "_benchmark.json";
    std::ofstream out(path);
    if (!out) return;
    const std::time_t now = std::time(nullptr);
    char stamp[64];
    std::strftime(stamp, sizeof(stamp), "%Y-%m-%dT%H:%M:%S", std::gmtime(&now));

    out << "{\n  \"timestamp\": \"" << stamp << "\",\n";
    out << "  \"operator\": \"" << op_ << "\",\n";
    // The backend and the device it ran on: without these a row from one chip
    // is indistinguishable from a row from another.
    out << "  \"env\": {\"backend\": \"" << flagsparseGetBackendName()
        << "\", \"arch\": \"" << device_arch()
        << "\", \"version\": " << FLAGSPARSE_VERSION << "},\n";
    out << "  \"config\": {\"warmup\": " << kWarmup << ", \"iters\": " << kIters
        << ", \"statistic\": \"median\"},\n";
    out << "  \"result\": [\n";
    for (std::size_t i = 0; i < rows_.size(); ++i) {
        const BenchRow& r = rows_[i];
        out << "    {\"name\": \"" << r.name << "\", \"status\": \"" << r.status
            << "\"";
        if (!r.detail.empty()) out << ", \"detail\": \"" << r.detail << "\"";
        for (const auto& kv : r.tags) {
            out << ", \"" << kv.first << "\": \"" << kv.second << "\"";
        }
        for (const auto& kv : r.numbers) {
            out << ", \"" << kv.first << "\": " << std::setprecision(10) << kv.second;
        }
        if (r.status == "ok") {
            out << ", \"median_ms\": " << std::setprecision(6) << r.median_ms
                << ", \"gflops\": " << r.gflops;
        }
        // The vendor comparison. Each of these is null rather than a stand-in
        // value: a fabricated baseline is worse than none, and a speedup over an
        // answer we did not check is worse still.
        out << ", \"accuracy\": \"" << r.accuracy << "\"";
        if (r.accuracy != "unchecked") {
            out << ", \"error_ratio\": " << std::setprecision(6) << r.error_ratio
                << ", \"relaxed_error_ratio\": " << r.relaxed_error_ratio;
        } else {
            out << ", \"error_ratio\": null, \"relaxed_error_ratio\": null";
        }
        out << ", \"baseline\": \"" << baseline::name() << "\"";
        out << ", \"baseline_accuracy\": \"" << r.baseline_accuracy << "\"";
        if (r.baseline_accuracy != "unchecked") {
            out << ", \"baseline_error_ratio\": " << std::setprecision(6)
                << r.baseline_error_ratio;
        } else {
            out << ", \"baseline_error_ratio\": null";
        }
        out << ", \"baseline_status\": \"" << r.baseline_status << "\"";
        if (!r.baseline_detail.empty()) {
            out << ", \"baseline_detail\": \"" << json_escape(r.baseline_detail) << "\"";
        }
        if (r.baseline_status == "ok") {
            out << ", \"baseline_ms\": " << std::setprecision(6) << r.baseline_ms;
        } else {
            out << ", \"baseline_ms\": null";
        }
        if (r.speedup > 0) {
            out << ", \"speedup\": " << std::setprecision(6) << r.speedup;
        } else {
            out << ", \"speedup\": null";
        }
        out << "}";
        out << (i + 1 < rows_.size() ? ",\n" : "\n");
    }
    // not_implemented_in_test is counted APART from failed. A variant the
    // manifest declares and this binary has no operand builder for is a gap in
    // the test, not a defect in the library, and folding it into `failed` would
    // report the library as broken for work it was never asked to do.
    std::size_t ok = 0, unsupported = 0, failed = 0, acc_fail = 0, no_test = 0,
                acc_relaxed = 0;
    // The speedup aggregate is built ONLY from rows whose answer passed, which is
    // the whole point of gating it. geomean, not mean: these are ratios, and a
    // single 40x on a tiny matrix would otherwise carry the average.
    std::vector<double> speedups;
    for (const BenchRow& r : rows_) {
        if (r.status == "ok") ++ok;
        else if (r.status == "not_supported") ++unsupported;
        else if (r.status == "not_implemented_in_test") ++no_test;
        else ++failed;
        if (r.accuracy == "fail") ++acc_fail;
        if (r.accuracy == "pass_relaxed") ++acc_relaxed;
        if (r.speedup > 0 && r.accuracy == "pass") speedups.push_back(r.speedup);
    }
    double geo = 0.0;
    if (!speedups.empty()) {
        double acc = 0.0;
        for (double v : speedups) acc += std::log(v);
        geo = std::exp(acc / static_cast<double>(speedups.size()));
    }

    out << "  ],\n  \"summary\": {\"rows\": " << rows_.size()
        << ", \"ok\": " << ok << ", \"not_supported\": " << unsupported
        << ", \"failed\": " << failed
        << ", \"not_implemented_in_test\": " << no_test
        << ", \"accuracy_failed\": " << acc_fail
        << ", \"accuracy_pass_relaxed\": " << acc_relaxed
        << ", \"baseline\": \"" << baseline::name() << "\""
        << ", \"corpus\": \"" << corpus_tag() << "\""
        // How many matrices the geomean is actually over. Without this the
        // headline ratio hides how much of the corpus it represents.
        << ", \"speedup_matrices\": " << speedups.size();
    if (!speedups.empty()) {
        out << ", \"speedup_geomean\": " << std::setprecision(6) << geo;
    } else {
        out << ", \"speedup_geomean\": null";
    }
    out << "}\n}\n";

    std::cout << "wrote " << path << "  (" << ok << " ok, " << unsupported
              << " not_supported, " << failed << " failed";
    if (no_test) std::cout << ", " << no_test << " declared but no test path";
    if (acc_relaxed) std::cout << ", " << acc_relaxed << " PASS(relaxed)";
    if (acc_fail) std::cout << ", " << acc_fail << " out of tolerance";
    std::cout << ")";
    if (!speedups.empty()) {
        std::cout << "  vs " << baseline::name() << ": geomean " << std::setprecision(4)
                  << geo << "x over " << speedups.size() << " matrices";
    } else if (!baseline::available()) {
        std::cout << "  (no vendor baseline on this backend)";
    }
    std::cout << std::endl;

    write_accuracy_json(op_, rows_);
}

std::size_t value_bytes(flagsparseDataType_t dtype) {
    switch (dtype) {
        case FLAGSPARSE_R_8I:   return 1;
        case FLAGSPARSE_R_16F:
        case FLAGSPARSE_R_16BF: return 2;
        case FLAGSPARSE_R_32F:
        case FLAGSPARSE_R_32I:  return 4;
        case FLAGSPARSE_R_64F:
        case FLAGSPARSE_C_32F:  return 8;
        case FLAGSPARSE_C_64F:  return 16;
        default:                return 0;
    }
}

std::size_t index_bytes(flagsparseIndexType_t idx) {
    switch (idx) {
        case FLAGSPARSE_INDEX_16U: return 2;
        case FLAGSPARSE_INDEX_32I: return 4;
        case FLAGSPARSE_INDEX_64I: return 8;
        default:                   return 0;
    }
}

const char* status_name(flagsparseStatus_t st) {
    const char* name = nullptr;
    flagsparseGetErrorName(st, &name);
    return name;
}

}  // namespace fstest
