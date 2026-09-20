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

// Shared machinery for a corpus sweep. Eight benchmarks do the same four things
// -- widen an fp64 master copy to the operand dtype, upload it, read the result
// back as fp64, judge it against the oracle -- and doing them eight times in
// eight files is how the eight drift apart.

#ifndef FLAGSPARSE_CTEST_SWEEP_HPP
#define FLAGSPARSE_CTEST_SWEEP_HPP

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

#include "common.hpp"
#include "corpus.hpp"

// Generated variant registry (namespace fstest::registry). Included before the
// namespace opens, for the reason given at variants_of() below.
#include "variants.inc"

namespace fstest {

struct Handle {
    flagsparseHandle_t h = nullptr;
    Handle() { flagsparseCreate(&h); }
    ~Handle() { if (h) flagsparseDestroy(h); }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
};

// Half-precision. Stored as 2 bytes, converted on the host because the C tests
// deliberately carry no vendor headers (no __half, no cuda_fp16.h): the bit
// twiddling below is the price of that, and it is worth paying -- pulling in a
// vendor half type would make ctest unbuildable on a backend whose SDK spells it
// differently.
inline bool dtype_is_half(flagsparseDataType_t t) {
    return t == FLAGSPARSE_R_16F || t == FLAGSPARSE_R_16BF;
}

// IEEE binary16 and bfloat16, both from a float. bf16 is just the top 16 bits of
// the fp32 pattern, with round-to-nearest-even; fp16 needs a real conversion.
inline uint16_t float_to_bf16(float f) {
    uint32_t bits;
    std::memcpy(&bits, &f, sizeof(bits));
    const uint32_t rounding = 0x7fffu + ((bits >> 16) & 1u);
    return static_cast<uint16_t>((bits + rounding) >> 16);
}

inline float bf16_to_float(uint16_t h) {
    const uint32_t bits = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

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

inline bool dtype_is_64(flagsparseDataType_t t) {
    return t == FLAGSPARSE_R_64F || t == FLAGSPARSE_C_64F;
}
inline bool dtype_is_complex(flagsparseDataType_t t) {
    return t == FLAGSPARSE_C_32F || t == FLAGSPARSE_C_64F;
}
// Scalars per element: a complex operand stores (re, im).
inline std::size_t dtype_components(flagsparseDataType_t t) {
    return dtype_is_complex(t) ? 2 : 1;
}

// alpha and beta, each in its OWN full-width slot.
//
// A complex128 operator reads 16 bytes from the alpha pointer. Packing alpha and
// beta into one double[2] and handing out &d[1] as beta therefore reads 8 bytes
// past the end: the vendor library gets a garbage beta, and on this box it was a
// segfault. Separate slots at the widest width any operator may read is the fix,
// and it is why this lives here instead of being retyped per benchmark.
struct Scalars {
    double da[2] = {1.0, 0.0};   // alpha = 1 (+0i)
    double db[2] = {0.0, 0.0};   // beta  = 0 (+0i)
    float  fa[2] = {1.0f, 0.0f};
    float  fb[2] = {0.0f, 0.0f};

    const void* alpha(flagsparseDataType_t t) const {
        return dtype_is_64(t) ? static_cast<const void*>(da)
                              : static_cast<const void*>(fa);
    }
    const void* beta(flagsparseDataType_t t) const {
        return dtype_is_64(t) ? static_cast<const void*>(db)
                              : static_cast<const void*>(fb);
    }
};

// An fp64 host vector uploaded at the operand's dtype. The imaginary part of a
// complex operand is left at zero so the fp64 oracle remains the right answer
// for it -- these sweeps measure throughput, and a complex operand that is real
// -valued still exercises the complex code path and its 2x traffic.
inline DeviceBuffer upload_as(const std::vector<double>& src,
                              flagsparseDataType_t dt) {
    const std::size_t comp = dtype_components(dt);
    if (dtype_is_half(dt)) {
        std::vector<uint16_t> h(src.size(), 0);
        for (std::size_t i = 0; i < src.size(); ++i) {
            const float f = static_cast<float>(src[i]);
            h[i] = (dt == FLAGSPARSE_R_16F) ? float_to_fp16(f) : float_to_bf16(f);
        }
        return DeviceBuffer::from(h);
    }
    if (dtype_is_64(dt)) {
        std::vector<double> h(src.size() * comp, 0.0);
        for (std::size_t i = 0; i < src.size(); ++i) h[i * comp] = src[i];
        return DeviceBuffer::from(h);
    }
    std::vector<float> h(src.size() * comp, 0.0f);
    for (std::size_t i = 0; i < src.size(); ++i)
        h[i * comp] = static_cast<float>(src[i]);
    return DeviceBuffer::from(h);
}

// Bytes one dense element occupies at this dtype.
inline std::size_t elem_bytes(flagsparseDataType_t dt) {
    if (dtype_is_half(dt)) return sizeof(uint16_t);
    return (dtype_is_64(dt) ? sizeof(double) : sizeof(float)) * dtype_components(dt);
}

// Read `count` elements back as fp64, taking the real part of a complex operand.
// Returns empty on a copy failure, which the caller turns into an out-of-range
// error ratio rather than a crash.
inline std::vector<double> read_back(const void* dev, std::size_t count,
                                     flagsparseDataType_t dt) {
    const std::size_t comp = dtype_components(dt);
    std::vector<double> out(count);
    if (dtype_is_half(dt)) {
        std::vector<uint16_t> h(count);
        if (to_host(h.data(), dev, h.size() * sizeof(uint16_t)) !=
            FLAGSPARSE_STATUS_SUCCESS) return {};
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = (dt == FLAGSPARSE_R_16F) ? fp16_to_float(h[i])
                                              : bf16_to_float(h[i]);
        }
        return out;
    }
    if (dtype_is_64(dt)) {
        std::vector<double> h(count * comp);
        if (to_host(h.data(), dev, h.size() * sizeof(double)) !=
            FLAGSPARSE_STATUS_SUCCESS) return {};
        for (std::size_t i = 0; i < count; ++i) out[i] = h[i * comp];
    } else {
        std::vector<float> h(count * comp);
        if (to_host(h.data(), dev, h.size() * sizeof(float)) !=
            FLAGSPARSE_STATUS_SUCCESS) return {};
        for (std::size_t i = 0; i < count; ++i) out[i] = h[i * comp];
    }
    return out;
}

// The error ratio of a device result against an fp64 oracle, at either the strict
// or the relaxed tolerance (spec 6.3 / 6.3.1). A failed read-back returns a value
// far above the pass threshold so the row records a failure instead of a silent
// pass on an empty comparison.
//
// BOTH RATIOS ARE NEEDED, not just the one that decides. Spec 6.3.1 only permits
// the relaxed tolerance when the VENDOR baseline fails at strict too, and that is
// not known until the baseline has run and overwritten the output buffer. So the
// relaxed ratio has to be computed up front, while our result is still there.
inline double ratio_against(const void* dev, const std::vector<double>& ref,
                            flagsparseDataType_t dt, bool relaxed = false) {
    const std::vector<double> got = read_back(dev, ref.size(), dt);
    if (got.empty()) return 1e30;
    return max_error_ratio(got, ref,
                           relaxed ? relaxed_tolerance(dt) : default_tolerance(dt));
}

// A deterministic dense operand. Fixed, not random: the oracle and the device
// must see bit-identical input for an error ratio to mean anything, and a seeded
// RNG that drifts between the two is a bug that looks like a precision problem.
inline std::vector<double> dense_pattern(std::size_t n, double phase = 0.0) {
    std::vector<double> v(n);
    for (std::size_t i = 0; i < n; ++i) {
        v[i] = 0.5 + 0.25 * static_cast<double>((i + static_cast<std::size_t>(phase)) % 7);
    }
    return v;
}


// The lower triangle of a corpus matrix, with a diagonal forced to dominate its
// row: diag = 1 + sum|off-diagonal|.
//
// BOTH PARTS ARE NECESSARY. A triangular solve on a real matrix's own diagonal
// diverges -- an early version of this measured an error ratio of 3e11 and the
// bug was the test, not the kernel. And the diagonal must be stored LAST in a
// lower row, because ascending column order puts it there and the solve's scan
// expects to meet it at the end of the row.
inline CsrMatrix lower_triangle(const CsrMatrix& A) {
    CsrMatrix L;
    L.rows = L.cols = std::min(A.rows, A.cols);
    L.indptr.assign(static_cast<std::size_t>(L.rows) + 1, 0);
    for (int64_t r = 0; r < L.rows; ++r) {
        double off = 0.0;
        for (int32_t p = A.indptr[static_cast<std::size_t>(r)];
             p < A.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
            const int32_t c = A.indices[static_cast<std::size_t>(p)];
            if (c >= r || c >= L.cols) continue;
            const double v = A.values[static_cast<std::size_t>(p)];
            L.indices.push_back(c);
            L.values.push_back(v);
            off += std::abs(v);
        }
        L.indices.push_back(static_cast<int32_t>(r));   // diagonal, last in the row
        L.values.push_back(1.0 + off);
        L.indptr[static_cast<std::size_t>(r) + 1] =
            static_cast<int32_t>(L.indices.size());
    }
    L.nnz = static_cast<int64_t>(L.indices.size());
    return L;
}

// Forward substitution on a lower-triangular CSR, entirely in fp64 on the host.
// The oracle for SpSV and, column by column, for SpSM.
inline std::vector<double> trsv_reference(const CsrMatrix& L,
                                          const std::vector<double>& b) {
    std::vector<double> x(static_cast<std::size_t>(L.rows), 0.0);
    for (int64_t r = 0; r < L.rows; ++r) {
        double acc = b[static_cast<std::size_t>(r)];
        double diag = 1.0;
        for (int32_t p = L.indptr[static_cast<std::size_t>(r)];
             p < L.indptr[static_cast<std::size_t>(r) + 1]; ++p) {
            const int32_t c = L.indices[static_cast<std::size_t>(p)];
            const double v = L.values[static_cast<std::size_t>(p)];
            if (c == r) diag = v;
            else acc -= v * x[static_cast<std::size_t>(c)];
        }
        x[static_cast<std::size_t>(r)] = acc / diag;
    }
    return x;
}

// The formats a corpus sweep covers for the operators that have more than one.
// This is a LIST, not a bool, because dropping a format from a benchmark is a
// silent coverage loss: an earlier rewrite of these files narrowed to CSR only
// and the JSON looked just as healthy with 26 variants as it had with 38.
struct FormatCase { const char* name; bool coo; };
inline const FormatCase* csr_and_coo(std::size_t* n) {
    static constexpr FormatCase kBoth[] = {{"csr", false}, {"coo", true}};
    *n = 2;
    return kBoth;
}

// ---------------------------------------------------------------- registry
//
// The variant list is generated from conf/operators.yaml (see
// tools/gen_variants.py) and included at the top of this file, OUTSIDE
// namespace fstest -- it declares fstest::registry itself, and including it in
// here would nest that into fstest::fstest::registry.
// Entries belonging to one benchmark binary.
inline std::vector<const registry::Variant*> variants_of(
    const char* family, const char* reporting = nullptr) {
    std::vector<const registry::Variant*> out;
    for (int i = 0; i < registry::kVariantCount; ++i) {
        const auto& variant = registry::kVariants[i];
        if (std::string(variant.family) == family &&
            (reporting == nullptr || std::string(variant.reporting) == reporting)) {
            out.push_back(&variant);
        }
    }
    return out;
}

// A declared variant this benchmark has no code path for.
//
// This is the whole reason the sweep is driven by the manifest. Before, such a
// variant was simply absent and the JSON looked complete; now it is a row that
// names itself and says the test, not the operator, is what is missing. The two
// are different problems and only one of them is the library's.
inline void report_unimplemented(BenchReport& rep, const registry::Variant& v,
                                 const char* why) {
    BenchRow row;
    row.name = std::string(v.op) + "_" + v.dtype + "_(no test path)";
    row.tag("operator", v.op).tag("format", v.format).tag("dtype", v.dtype)
       .tag("corpus", corpus_tag()).tag("reporting", v.reporting);
    rep.skip(std::move(row), "not_implemented_in_test", why);
}

struct DtypeCase { flagsparseDataType_t dt; const char* name; };

// The four types every operator in this suite supports. An operator that covers
// fewer (SDDMM has no complex kernel yet) declares its own shorter table.
inline const DtypeCase* all_dtypes(std::size_t* n) {
    static constexpr DtypeCase kAll[] = {
        {FLAGSPARSE_R_32F, "f32"},
        {FLAGSPARSE_R_64F, "f64"},
        {FLAGSPARSE_C_32F, "c32"},
        {FLAGSPARSE_C_64F, "c64"},
    };
    *n = sizeof(kAll) / sizeof(kAll[0]);
    return kAll;
}

// Per-matrix progress on stderr. A sweep over a large corpus runs for minutes,
// and when one matrix takes the process down this line is what names it.
// Tag a row with its delivery scope, so a report can filter on it without
// re-deriving the manifest.
inline void tag_scope(BenchRow& row, const registry::Variant& v) {
    row.tag("reporting", v.reporting);
}

inline void trace(const char* op, const std::string& matrix, const char* dtype,
                  const CsrMatrix& A) {
    std::fprintf(stderr, "[%s] %s %s %lldx%lld nnz=%lld\n", op, matrix.c_str(), dtype,
                 static_cast<long long>(A.rows), static_cast<long long>(A.cols),
                 static_cast<long long>(A.nnz));
}

// Rows for corpus files that never parsed, emitted once per benchmark so a
// matrix that was handed to the suite is never simply absent from its JSON.
inline void report_corpus_failures(BenchReport& rep, const char* format) {
    for (const auto& f : corpus_failures()) {
        rep.skip(BenchRow{}.tag("matrix", f.first).tag("format", format), "failed",
                 "corpus: " + f.second);
    }
}

}  // namespace fstest

#endif  // FLAGSPARSE_CTEST_SWEEP_HPP
