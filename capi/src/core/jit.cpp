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


#include "core/jit.hpp"

#include <cstdlib>
#include <filesystem>
#include <stdexcept>

#include "triton_jit/triton_jit_function.h"

namespace flagsparse::jit {

namespace {

// Where flagsparse_codegen/ lives. Baked at configure time, overridable so a
// packaged install or a test run from a build tree can both find it.
std::filesystem::path codegen_root() {
    if (const char* env = std::getenv("FLAGSPARSE_CODEGEN_PATH")) {
        if (*env != '\0') return std::filesystem::path(env);
    }
    return std::filesystem::path(FLAGSPARSE_CODEGEN_DIR);
}

// The codegen shims read FLAGSPARSE_PYTHON_SRC to find the operator package.
// Set it once, from the configure-time default, WITHOUT overwriting a value the
// caller already chose -- an explicit environment setting must win over a path
// baked in at build time, or a relocated install becomes unfixable.
void ensure_python_src_env() {
    static const bool done = [] {
        if (const char* existing = std::getenv("FLAGSPARSE_PYTHON_SRC")) {
            if (*existing != '\0') return true;
        }
        const char* baked = FLAGSPARSE_PYTHON_SRC_DEFAULT;
        if (baked != nullptr && *baked != '\0') {
            ::setenv("FLAGSPARSE_PYTHON_SRC", baked, /*overwrite=*/0);
        }
        return true;
    }();
    (void)done;
}

}  // namespace

std::string codegen_module(const std::string& file_name) {
    return (codegen_root() / file_name).string();
}

flagsparseStatus_t launch(const std::string& module_path,
                          const std::string& kernel_name,
                          const std::string& signature,
                          flagsparseStream_t stream,
                          std::int64_t grid_x, std::int64_t grid_y, std::int64_t grid_z,
                          int num_warps, int num_stages,
                          const std::vector<Arg>& args,
                          std::string* error_out) {
    // An empty grid is a no-op, not an error: a zero-row matrix is legal input.
    if (grid_x <= 0 || grid_y <= 0 || grid_z <= 0) return FLAGSPARSE_STATUS_SUCCESS;
    if (!std::filesystem::exists(module_path)) {
        if (error_out) *error_out = "kernel module not found: " + module_path;
        return FLAGSPARSE_STATUS_INTERNAL_ERROR;
    }
    try {
        ensure_python_src_env();
        auto& fn = triton_jit::TritonJITFunction::get_instance(module_path, kernel_name);

        std::vector<void*> raw;
        raw.reserve(args.size() + 2);
        for (const Arg& a : args) {
            // const_cast is safe: the launch reads through these pointers only.
            switch (a.kind) {
                case Arg::Kind::DevicePtr: raw.push_back(const_cast<adaptor::DevicePtr*>(&a.dev)); break;
                case Arg::Kind::I32: raw.push_back(const_cast<std::int32_t*>(&a.i32)); break;
                case Arg::Kind::I64: raw.push_back(const_cast<std::int64_t*>(&a.i64)); break;
                case Arg::Kind::F32: raw.push_back(const_cast<float*>(&a.f32)); break;
                case Arg::Kind::F64: raw.push_back(const_cast<double*>(&a.f64)); break;
            }
        }
        // Triton appends a global and a profile scratch pointer to every kernel.
        adaptor::DevicePtr global_scratch = 0, profile_scratch = 0;
        raw.push_back(&global_scratch);
        raw.push_back(&profile_scratch);

        fn.launch_with_raw_args(
            reinterpret_cast<triton_jit::DefaultStreamType>(stream),
            static_cast<unsigned int>(grid_x),
            static_cast<unsigned int>(grid_y),
            static_cast<unsigned int>(grid_z),
            static_cast<unsigned int>(num_warps),
            static_cast<unsigned int>(num_stages),
            signature, raw.data(), raw.size());
        return FLAGSPARSE_STATUS_SUCCESS;
    } catch (const std::exception& exc) {
        // Compile and launch failures both arrive as exceptions; the C boundary
        // must not let them through.
        if (error_out) *error_out = exc.what();
        return FLAGSPARSE_STATUS_EXECUTION_FAILED;
    }
}

}  // namespace flagsparse::jit
