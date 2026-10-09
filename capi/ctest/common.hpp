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


// Helpers shared by the accuracy and benchmark binaries.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <utility>
#include <random>
#include <string>
#include <vector>

#include "flagsparse.h"
#include "baseline/baseline.hpp"

namespace fstest {

// IEEE binary16, from/to float. Moved here (not just in sweep.hpp, which
// originally had the only copy) so ctest/accuracy/*.cpp can build the `Half`
// host type below without pulling in sweep.hpp's benchmark-oriented machinery.
inline uint16_t float_to_fp16(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    int32_t exp = static_cast<int32_t>((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        // Subnormal, not zero. fp16's smallest normal is ~6.1e-5 and its
        // smallest subnormal ~6e-8, so flushing this whole range to zero loses
        // four orders of magnitude -- a unit test caught 1e-5 becoming 0.
        if (exp < -10) return static_cast<uint16_t>(sign);   // truly below range
        mant |= 0x800000u;                                   // restore implicit 1
        const int32_t shift = 14 - exp;                      // 24-bit mant -> 10-bit
        const uint32_t sub = mant >> shift;
        // Round to nearest even on the discarded bits.
        const uint32_t rem = mant & ((1u << shift) - 1u);
        const uint32_t half = 1u << (shift - 1);
        uint32_t rounded = sub + ((rem > half || (rem == half && (sub & 1u))) ? 1u : 0u);
        return static_cast<uint16_t>(sign | rounded);
    }
    if (exp >= 31) return static_cast<uint16_t>(sign | 0x7c00u);  // overflow to inf
    // Round to nearest even on the 13 bits being discarded.
    const uint32_t round = (mant & 0x1fffu) > 0x1000u ||
                           ((mant & 0x1fffu) == 0x1000u && ((mant >> 13) & 1u));
    uint16_t out = static_cast<uint16_t>(sign | (static_cast<uint32_t>(exp) << 10) |
                                         (mant >> 13));
    return static_cast<uint16_t>(out + round);
}

inline float fp16_to_float(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) { bits = sign; }
        else {
            // Subnormal: normalise it into an fp32 exponent.
            int32_t e = -1;
            uint32_t m = mant;
            do { m <<= 1; ++e; } while ((m & 0x400u) == 0);
            bits = sign | (static_cast<uint32_t>(127 - 15 - e) << 23) |
                   ((m & 0x3ffu) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

// A host-side fp16 value that behaves like a float for template purposes
// (static_cast<Half>(double), implicit conversion back to double for error
// checks) while being bit-exact fp16 storage -- the same representation
// flagsparseCreateDnMat's FLAGSPARSE_R_16F expects on device. Needed because
// plain `float`/`double` are 4/8 bytes and DeviceBuffer<T>::from/download are
// sizeof(T)-based, so an accuracy test cannot use a native C++ type for fp16.
struct Half {
    uint16_t bits = 0;
    Half() = default;
    Half(double f) : bits(float_to_fp16(static_cast<float>(f))) {}  // NOLINT
    operator double() const { return static_cast<double>(fp16_to_float(bits)); }  // NOLINT
};
static_assert(sizeof(Half) == 2, "Half must be bit-exact fp16 storage");

// Device buffer with RAII, built on the C API's own backend so the tests need
// no vendor headers of their own.
flagsparseStatus_t dev_alloc(void** ptr, std::size_t bytes);
void dev_free(void* ptr);
flagsparseStatus_t to_device(void* dst, const void* src, std::size_t bytes);
flagsparseStatus_t to_host(void* dst, const void* src, std::size_t bytes);
void dev_sync();

class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t bytes) { dev_alloc(&ptr_, bytes); bytes_ = bytes; }
    ~DeviceBuffer() { dev_free(ptr_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& o) noexcept : ptr_(o.ptr_), bytes_(o.bytes_) {
        o.ptr_ = nullptr; o.bytes_ = 0;
    }
    DeviceBuffer& operator=(DeviceBuffer&& o) noexcept {
        if (this != &o) {
            dev_free(ptr_);
            ptr_ = o.ptr_; bytes_ = o.bytes_;
            o.ptr_ = nullptr; o.bytes_ = 0;
        }
        return *this;
    }
    void* get() const { return ptr_; }
    std::size_t size() const { return bytes_; }
    bool valid() const { return ptr_ != nullptr || bytes_ == 0; }

    template <typename T>
    static DeviceBuffer from(const std::vector<T>& host) {
        DeviceBuffer b(host.size() * sizeof(T));
        if (!host.empty()) to_device(b.ptr_, host.data(), host.size() * sizeof(T));
        return b;
    }
    template <typename T>
    std::vector<T> download(std::size_t count) const {
        std::vector<T> host(count);
        if (count) to_host(host.data(), ptr_, count * sizeof(T));
        return host;
    }

  private:
    void* ptr_ = nullptr;
    std::size_t bytes_ = 0;
};

// A host-side CSR matrix plus the dense reference it was generated from.
struct CsrMatrix {
    int64_t rows = 0, cols = 0, nnz = 0;
    std::vector<int32_t> indptr;
    std::vector<int32_t> indices;
    std::vector<double>  values;   // fp64 master copy; cast down per dtype
};

// Random CSR with a target density. Rows are deliberately uneven so the
// segment-loop sizing (max row nnz, not average) is actually exercised.
CsrMatrix random_csr(int64_t rows, int64_t cols, double density, uint32_t seed);

// A triangular CSR matrix with a dominant diagonal, columns sorted ascending
// within each row (which SpSV requires and SpSM tolerates).
//
// The off-diagonal magnitude is scaled by the expected row population, so the
// triangle stays diagonally dominant at ANY density. Without that a dense row's
// off-diagonal sum swamps the diagonal and the solve amplifies rounding by
// orders of magnitude -- which measures the matrix, not the kernel.
struct TriMatrix {
    int64_t n = 0;
    std::vector<int32_t> indptr, indices;
    std::vector<double> values;
    bool lower = true;
    bool unit_diag = false;
    int64_t nnz() const { return static_cast<int64_t>(indices.size()); }
};

TriMatrix random_triangular(int64_t n, double density, bool lower, bool unit_diag,
                            uint32_t seed);

// A lower-triangular matrix in sliced-ELL. The format's rules are structural,
// not advisory, so they are built in rather than hoped for: every row carries
// exactly one diagonal entry, padding is -1 and strictly trailing, and a slice
// is as wide as its longest row.
struct SellMatrix {
    int64_t n = 0, slice_size = 0;
    std::vector<int32_t> offsets, cols;
    std::vector<double> values;
    std::vector<double> dense;          // n x n row-major, the reference
};

SellMatrix random_sell_lower(int64_t n, int64_t slice_size, double density,
                             uint32_t seed);

// One row index per nonzero: the COO expansion of a CSR pattern. Row-sorted by
// construction, which is what both cuSPARSE and this library require of a COO.
std::vector<int32_t> coo_row_indices_of(const CsrMatrix& A);
std::vector<int32_t> coo_row_indices_of(const TriMatrix& T);

// y = alpha * A * x + beta * y, entirely in fp64 on the host. This is the
// golden reference: computing it on the accelerator would test the vendor's
// dense library rather than FlagSparse.
std::vector<double> spmv_reference(const CsrMatrix& A, const std::vector<double>& x,
                                   double alpha, double beta,
                                   const std::vector<double>& y_in);

// C = alpha * A * B + beta * C, entirely in fp64 on the host, with B and C
// row-major k x n and m x n. Same reasoning as spmv_reference: computing this
// on the accelerator would test the vendor's dense library instead.
std::vector<double> spmm_reference(const CsrMatrix& A, const std::vector<double>& B,
                                   int64_t n, double alpha, double beta,
                                   const std::vector<double>& c_in);

// Spec §6.3: error_ratio = max(|actual - ref| / (atol + rtol * |ref|)); PASS at <= 1.
struct Tolerance { double rtol; double atol; };
Tolerance default_tolerance(flagsparseDataType_t dtype);
Tolerance relaxed_tolerance(flagsparseDataType_t dtype);
double max_error_ratio(const std::vector<double>& actual, const std::vector<double>& ref,
                       Tolerance tol);

const char* status_name(flagsparseStatus_t st);

// Bytes per value / per index. The library has these internally, but the public
// surface is flagsparse.h and nothing else, so the tests carry their own -- and
// a memory-bound operator cannot report bytes moved without them.
std::size_t value_bytes(flagsparseDataType_t dtype);
std::size_t index_bytes(flagsparseIndexType_t idx);

// Print which backend and device the run landed on, once per binary.
//
// The harness has to cover CUDA and every domestic accelerator, and a result
// with no platform on it is not attributable: two runs of the same binary on two
// chips produce the same lines. The benchmark JSON already carries the backend;
// this puts it on the accuracy output too.
void print_backend_banner();

// Device architecture string, for the JSON env block. The tests compile their
// own adaptor copy, so this is available to them without widening the ABI.
std::string device_arch();

// ------------------------------------------------------------- benchmark ---
//
// One harness for every operator, because the report has to be readable across
// platforms. Six operators each writing their own JSON shape means six shapes to
// reconcile when comparing CUDA against an accelerator, and the first thing that
// gets lost is WHY a row is missing.
//
// So a row always exists and always carries a status:
//
//   ok             measured; median_ms and gflops are meaningful
//   not_supported  the operator declined this configuration on this backend --
//                  a fact about the port, recorded rather than dropped
//   failed         it tried and errored; the status name and message are kept
//
// A backend that has not wired an operator up therefore produces a JSON full of
// not_supported rows instead of an absent file, which reads the same as a pass.

struct BenchRow {
    std::string name;
    std::string status = "ok";
    std::string detail;
    // Free-form columns, written verbatim. The runner's CSV/JSON reader carries
    // unknown keys through, so an operator can record whatever identifies its
    // case without a schema change here.
    std::vector<std::pair<std::string, double>> numbers;
    std::vector<std::pair<std::string, std::string>> tags;
    double median_ms = 0.0;
    double gflops = 0.0;

    // ---- the vendor comparison. Filled by measure_vs_baseline, blank otherwise.
    //
    // `speedup` is baseline_ms / median_ms and is written ONLY when accuracy is
    // "pass". A speedup computed from a wrong answer is not a speedup, and the
    // rule the aggregate applies has to be visible on the row that fed it.
    double baseline_ms = 0.0;
    double speedup = 0.0;

    // "pass" | "pass_relaxed" | "fail" | "unchecked". Judged against the CPU fp64
    // oracle, never against the baseline -- see baseline/baseline.hpp.
    //
    // pass_relaxed is spec 6.3.1 and is NOT a softer pass handed out on request:
    // it requires that the VENDOR baseline also failed at the strict tolerance on
    // this matrix, which is the evidence that the matrix is ill-conditioned rather
    // than that we are wrong. Without a baseline it is never awarded.
    std::string accuracy = "unchecked";
    double error_ratio = 0.0;          // at the strict tolerance
    double relaxed_error_ratio = 0.0;  // at spec 6.3.1's relaxed tolerance

    // The baseline's own agreement with the same fp64 oracle: "pass" | "fail" |
    // "unchecked". A speedup over a baseline that did not compute the right
    // answer is not a speedup, and this is what makes that visible.
    std::string baseline_accuracy = "unchecked";
    double baseline_error_ratio = 0.0;

    // "ok" | "unavailable" | "failed". When not ok, baseline_detail says why and
    // the speedup column stays blank.
    std::string baseline_status = "unavailable";
    std::string baseline_detail;

    BenchRow& num(const std::string& key, double value) {
        numbers.emplace_back(key, value);
        return *this;
    }
    BenchRow& tag(const std::string& key, const std::string& value) {
        tags.emplace_back(key, value);
        return *this;
    }
};

class BenchReport {
  public:
    // `op` names the JSON: $FLAGSPARSE_BENCH_OUT/<op>_benchmark.json
    explicit BenchReport(std::string op) : op_(std::move(op)) {}

    // Warm up, time `once` kIters times, record the median. `flops` is the work
    // one call does, for the GFLOPS column; pass 0 to leave it blank.
    //
    // The first call is made outside the samples on purpose: it pays JIT
    // compilation, which on some backends is seconds and would otherwise land
    // in the median as a single enormous outlier.
    bool measure(BenchRow row, const std::function<flagsparseStatus_t()>& once,
                 double flops);

    // One matrix of a corpus sweep: time ours, check the answer, time the vendor,
    // and record all three on one row.
    //
    // NOTHING HERE ABORTS THE SWEEP. Every way this can go wrong -- the operator
    // declines, the operator errors, the answer is out of tolerance, the baseline
    // is unavailable, the baseline errors -- produces a row carrying the reason
    // and returns false, so the caller moves to the next matrix. A benchmark that
    // stops at the first bad matrix reports the corpus it got through, not the
    // corpus it was given.
    //
    //   once   : runs the operator (timed, and leaves its result in place)
    //   verify : error ratio of whatever is CURRENTLY in the output buffer,
    //            against the CPU fp64 oracle, at the strict tolerance (false) or
    //            spec 6.3.1's relaxed one (true). <= 1 passes. Return a negative
    //            value to declare the check not applicable, which records
    //            accuracy="unchecked" and withholds the speedup.
    //            It is called again AFTER the baseline runs, to judge the
    //            baseline's own answer -- hence "currently in the buffer".
    //   base   : runs the vendor baseline; may report unavailable, which is a
    //            recorded fact and not a failure.
    //   baseline_writes_output : false when the baseline does not leave its
    //            result where verify can see it (SpGEMM allocates its own C and
    //            frees it). Then baseline_accuracy stays "unchecked" instead of
    //            re-reading OUR result and calling the baseline correct.
    bool measure_vs_baseline(
        BenchRow row, const std::function<flagsparseStatus_t()>& once,
        const std::function<double(bool)>& verify,
        const std::function<baseline::Status(baseline::Timing*)>& base, double flops,
        bool baseline_writes_output = true);

    // Record a case that was never measured -- no descriptor, unsupported shape.
    void skip(BenchRow row, const std::string& status, const std::string& detail);

    // Writes BOTH artifacts from the one sweep:
    //   $FLAGSPARSE_BENCH_OUT/<op>_benchmark.json   timings and speedups
    //   $FLAGSPARSE_BENCH_OUT/<op>_accuracy.json    per-variant error ratios
    //
    // They are two files because two consumers want two things, not because
    // there were two runs. Every row here was checked against the host fp64
    // oracle BEFORE it was timed, so the accuracy artifact is not derived from
    // the timing one -- both are views of the same measurement, and each says so.
    void write() const;
    std::size_t size() const { return rows_.size(); }

    static constexpr int kWarmup = 10;
    static constexpr int kIters = 100;

  private:
    std::string op_;
    std::vector<BenchRow> rows_;
};

}  // namespace fstest
