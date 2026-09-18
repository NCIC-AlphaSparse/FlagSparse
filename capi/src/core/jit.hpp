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


// Thin bridge onto libtriton_jit. Everything vendor-specific about compiling
// and launching a Triton kernel lives in that submodule; this file only builds
// the signature string and marshals raw arguments.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "adaptor/adaptor.hpp"
#include "flagsparse.h"

namespace flagsparse::jit {

// One kernel argument. Triton's raw-args ABI wants a pointer to each value, so
// the value is stored by the struct and its address handed over at launch --
// taking the address of a temporary is the easy way to corrupt a launch.
struct Arg {
    enum class Kind { DevicePtr, I32, I64, F32, F64 } kind;
    union {
        adaptor::DevicePtr dev;
        std::int32_t i32;
        std::int64_t i64;
        float f32;
        double f64;
    };
    static Arg ptr(adaptor::DevicePtr v) { Arg a{Kind::DevicePtr}; a.dev = v; return a; }
    static Arg i(std::int32_t v)         { Arg a{Kind::I32}; a.i32 = v; return a; }
    static Arg i64v(std::int64_t v)      { Arg a{Kind::I64}; a.i64 = v; return a; }
    static Arg f(float v)                { Arg a{Kind::F32}; a.f32 = v; return a; }
    static Arg d(double v)               { Arg a{Kind::F64}; a.f64 = v; return a; }
};

// Absolute path of a kernel module inside flagsparse_codegen/.
// Absolute on purpose: libtriton_jit resolves a relative path against the
// PROCESS's cwd, so a relative one works only when the caller happens to run
// from the right directory -- FlagFFT's own example trips on exactly that.
std::string codegen_module(const std::string& file_name);

// Compile (cached) and launch. `signature` follows Triton's raw-args spelling:
// pointers as "*fp32:16", scalars as "fp32"/"i32", constexprs as literals.
flagsparseStatus_t launch(const std::string& module_path,
                          const std::string& kernel_name,
                          const std::string& signature,
                          flagsparseStream_t stream,
                          std::int64_t grid_x, std::int64_t grid_y, std::int64_t grid_z,
                          int num_warps, int num_stages,
                          const std::vector<Arg>& args,
                          std::string* error_out);

}  // namespace flagsparse::jit
