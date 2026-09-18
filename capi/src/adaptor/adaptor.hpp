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


// Vendor runtime abstraction. One implementation per backend under
// backend/<name>/adaptor.cpp, selected by BACKEND at configure time -- the
// same shape FlagFFT uses, and the reason a new chip needs no header change.
//
// Deliberately tiny: the heavy lifting (compile, launch) belongs to
// libtriton_jit, which has its own per-backend layer. What is left here is
// device/context discovery plus the memory and stream helpers the C++ tests
// need in order to run without pulling in libtorch.

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "flagsparse.h"

namespace flagsparse::adaptor {

using DevicePtr = std::uintptr_t;

// Initialise the backend context and report the device. Returns a status rather
// than throwing: flagsparseCreate must fail cleanly when no device is present.
flagsparseStatus_t ensure_device(int& device_index, std::string& device_arch);

const char* backend_name();
int device_count();
std::string device_architecture(int device_index);

// Launch-shaping properties. A grid-stride kernel has to be capped at roughly
// what fills the device once; hardcoding that cap is how the Python gather
// ended up pinned to two thread blocks and 28x slower than it needed to be.
int multiprocessor_count(int device_index);
int max_threads_per_multiprocessor(int device_index);
int warp_size(int device_index);
void synchronize();

// Stream 0 / the backend default stream when the handle carries none.
flagsparseStream_t default_stream();

// Minimal allocator for tests and for scratch buffers the dispatch layer owns.
flagsparseStatus_t device_malloc(DevicePtr* ptr, std::size_t bytes);
void device_free(DevicePtr ptr);
flagsparseStatus_t memcpy_h2d(DevicePtr dst, const void* src, std::size_t bytes);
flagsparseStatus_t memcpy_d2h(void* dst, DevicePtr src, std::size_t bytes);
flagsparseStatus_t memset_device(DevicePtr dst, int value, std::size_t bytes);

}  // namespace flagsparse::adaptor
