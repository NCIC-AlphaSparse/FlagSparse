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

// Moore Threads mirrors the CUDA driver API symbol for symbol (cu* -> mu*),
// so this backend is the shared body with a different prefix. Verified against
// FlagFFT's musa adaptor, which is a 1:1 rename of its cuda one.

#include <musa.h>

#define FS_DRV(sym)       mu##sym
#define FS_RESULT         MUresult
#define FS_SUCCESS_CODE   MUSA_SUCCESS
#define FS_DEVICE         MUdevice
#define FS_CONTEXT        MUcontext
#define FS_DEVPTR         MUdeviceptr
#define FS_ATTR_MAJOR     MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
#define FS_ATTR_MINOR     MU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
#define FS_ATTR_ENUM      MUdevice_attribute
#define FS_ATTR_SM_COUNT  MU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
#define FS_ATTR_MAX_THREADS_PER_SM MU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR
#define FS_ATTR_WARP_SIZE MU_DEVICE_ATTRIBUTE_WARP_SIZE
#define FS_BACKEND_NAME   "musa"

#include "../_template.inc"
