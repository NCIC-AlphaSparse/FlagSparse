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

// The csrgeam2 call sequence against a host merge: C's structure must be the
// union of A's and B's (sorted, a shared column stored once even when it sums
// to zero) and its values alpha * A + beta * B, for every value type, with rows
// only one side has, rows neither has, full overlap, long rows (past one tile
// and one scan block) and complex scalars.

#include <gtest/gtest.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "common.hpp"

using namespace fstest;

namespace {

using cd = std::complex<double>;

struct Handle {
    flagsparseHandle_t h = nullptr;
    Handle() { flagsparseCreate(&h); }
    ~Handle() { if (h) flagsparseDestroy(h); }
};

std::string last_error(flagsparseHandle_t h) {
    const char* s = nullptr;
    return flagsparseGetLastErrorString(h, &s) == FLAGSPARSE_STATUS_SUCCESS && s ? s : "";
}

class GeamAccuracy : public ::testing::Test {
  protected:
    Handle handle;
    void SetUp() override {
        if (handle.h == nullptr) GTEST_SKIP() << "no accelerator available";
        static bool announced = false;
        if (!announced) { print_backend_banner(); announced = true; }
    }
};

template <typename T> T cast_to(cd v);
template <> float cast_to<float>(cd v) { return static_cast<float>(v.real()); }
template <> double cast_to<double>(cd v) { return v.real(); }
template <> std::complex<float> cast_to<std::complex<float>>(cd v) {
    return {static_cast<float>(v.real()), static_cast<float>(v.imag())};
}
template <> std::complex<double> cast_to<std::complex<double>>(cd v) { return v; }
template <typename T> cd widen(T v) { return cd(std::real(v), std::imag(v)); }

template <typename T>
struct Csr {
    int rows = 0, cols = 0;
    std::vector<int32_t> ptr, idx;
    std::vector<T> val;
};

// Row r holds column c when (r * 31 + c * 17 + salt) % period < keep: a pattern
// that varies by row and is sorted by construction.
template <typename T>
Csr<T> pattern(int rows, int cols, int period, int keep, int salt) {
    Csr<T> M;
    M.rows = rows;
    M.cols = cols;
    M.ptr.push_back(0);
    for (int r = 0; r < rows; ++r) {
        for (int c = 0; c < cols; ++c) {
            if ((r * 31 + c * 17 + salt) % period < keep) {
                M.idx.push_back(c);
                const double re = 0.25 * ((r + 2 * c + salt) % 9) - 1.0;
                const double im = 0.125 * ((3 * r + c + salt) % 7) - 0.375;
                M.val.push_back(cast_to<T>(cd(re, im)));
            }
        }
        M.ptr.push_back(static_cast<int32_t>(M.idx.size()));
    }
    return M;
}

template <typename T>
void check(flagsparseHandle_t h, flagsparseDataType_t dt, const Csr<T>& A, const Csr<T>& B,
           cd alpha, cd beta, double tol) {
    const int m = A.rows, n = A.cols;
    const int nnz_a = static_cast<int>(A.idx.size()), nnz_b = static_cast<int>(B.idx.size());
    std::vector<std::map<int, cd>> ref(static_cast<std::size_t>(m));
    for (int r = 0; r < m; ++r) {
        for (int p = A.ptr[r]; p < A.ptr[r + 1]; ++p) ref[r][A.idx[p]] += alpha * widen(A.val[p]);
        for (int p = B.ptr[r]; p < B.ptr[r + 1]; ++p) ref[r][B.idx[p]] += beta * widen(B.val[p]);
    }

    // DeviceBuffer::from rejects an empty vector's null data; keep one slot.
    auto upload_idx = [](std::vector<int32_t> v) {
        if (v.empty()) v.push_back(0);
        return DeviceBuffer::from(v);
    };
    auto upload_val = [](std::vector<T> v) {
        if (v.empty()) v.push_back(T{});
        return DeviceBuffer::from(v);
    };
    DeviceBuffer a_ptr = DeviceBuffer::from(A.ptr), a_idx = upload_idx(A.idx),
                 a_val = upload_val(A.val);
    DeviceBuffer b_ptr = DeviceBuffer::from(B.ptr), b_idx = upload_idx(B.idx),
                 b_val = upload_val(B.val);
    DeviceBuffer c_ptr(static_cast<std::size_t>(m + 1) * sizeof(int32_t));
    const T a = cast_to<T>(alpha), b = cast_to<T>(beta);
    const auto* ap = static_cast<const int*>(a_ptr.get());
    const auto* ai = static_cast<const int*>(a_idx.get());
    const auto* bp = static_cast<const int*>(b_ptr.get());
    const auto* bi = static_cast<const int*>(b_idx.get());

    size_t bytes = 0;
    ASSERT_EQ(flagsparseCsrgeam2_bufferSizeExt(h, m, n, &a, nnz_a, a_val.get(), ap, ai, &b, nnz_b,
                                               b_val.get(), bp, bi, nullptr,
                                               static_cast<const int*>(c_ptr.get()), nullptr, dt,
                                               &bytes),
              FLAGSPARSE_STATUS_SUCCESS);
    DeviceBuffer work(bytes ? bytes : 1);
    int nnz_c = -1;
    ASSERT_EQ(flagsparseXcsrgeam2Nnz(h, m, n, nnz_a, ap, ai, nnz_b, bp, bi,
                                     static_cast<int*>(c_ptr.get()), &nnz_c, work.get()),
              FLAGSPARSE_STATUS_SUCCESS)
        << last_error(h);
    std::size_t expect = 0;
    for (const auto& row : ref) expect += row.size();
    ASSERT_EQ(nnz_c, static_cast<int>(expect));
    DeviceBuffer c_idx(static_cast<std::size_t>(std::max(nnz_c, 1)) * sizeof(int32_t));
    DeviceBuffer c_val(static_cast<std::size_t>(std::max(nnz_c, 1)) * sizeof(T));
    ASSERT_EQ(flagsparseCsrgeam2(h, m, n, &a, nnz_a, a_val.get(), ap, ai, &b, nnz_b, b_val.get(), bp,
                                 bi, c_val.get(), static_cast<const int*>(c_ptr.get()),
                                 static_cast<int*>(c_idx.get()), dt, work.get()),
              FLAGSPARSE_STATUS_SUCCESS)
        << last_error(h);

    const auto ptr = c_ptr.download<int32_t>(static_cast<std::size_t>(m + 1));
    const auto idx = c_idx.download<int32_t>(static_cast<std::size_t>(nnz_c));
    const auto val = c_val.download<T>(static_cast<std::size_t>(nnz_c));
    double scale = 1.0, err = 0.0;
    for (int r = 0; r < m; ++r) {
        ASSERT_EQ(ptr[r + 1] - ptr[r], static_cast<int32_t>(ref[r].size())) << "row " << r;
        auto it = ref[r].begin();
        for (int p = ptr[r]; p < ptr[r + 1]; ++p, ++it) {
            ASSERT_EQ(idx[p], it->first) << "row " << r << " not the sorted union";
            scale = std::max(scale, std::abs(it->second));
            err = std::max(err, std::abs(widen(val[p]) - it->second));
        }
    }
    EXPECT_LE(err, tol * scale);
}

TEST_F(GeamAccuracy, EveryTypeWithEmptyAndOneSidedRows) {
    for (int salt : {0, 5}) {
        auto A32 = pattern<float>(203, 97, 11, 2, salt), B32 = pattern<float>(203, 97, 7, 2, salt + 1);
        check(handle.h, FLAGSPARSE_R_32F, A32, B32, 0.5, 1.25, 1e-6);
        auto A64 = pattern<double>(203, 97, 11, 2, salt), B64 = pattern<double>(203, 97, 7, 2, salt);
        check(handle.h, FLAGSPARSE_R_64F, A64, B64, -2.0, 0.75, 1e-14);
        using c32 = std::complex<float>;
        using c64 = std::complex<double>;
        check(handle.h, FLAGSPARSE_C_32F, pattern<c32>(203, 97, 11, 2, salt),
              pattern<c32>(203, 97, 7, 2, salt + 2), cd(0.5, -2.0), cd(1.25, 0.5), 1e-6);
        check(handle.h, FLAGSPARSE_C_64F, pattern<c64>(203, 97, 11, 2, salt),
              pattern<c64>(203, 97, 7, 2, salt + 3), cd(0.5, -2.0), cd(1.25, 0.5), 1e-14);
    }
}

TEST_F(GeamAccuracy, FullOverlapLongRowsAndEmptyOperands) {
    // Every column collides and sums to zero, and is still stored.
    auto A = pattern<float>(17, 3000, 2, 1, 0);
    check(handle.h, FLAGSPARSE_R_32F, A, A, 1.0, -1.0, 1e-6);
    // B's entries span several scan blocks (1024) in one row.
    check(handle.h, FLAGSPARSE_R_32F, A, pattern<float>(17, 3000, 3, 2, 1), 2.0, 3.0, 1e-6);
    Csr<float> empty;
    empty.rows = 17;
    empty.cols = 3000;
    empty.ptr.assign(18, 0);
    check(handle.h, FLAGSPARSE_R_32F, A, empty, 1.0, 1.0, 1e-6);
    check(handle.h, FLAGSPARSE_R_32F, empty, A, 1.0, 2.0, 1e-6);
    check(handle.h, FLAGSPARSE_R_32F, empty, empty, 1.0, 1.0, 1e-6);
}

}  // namespace
