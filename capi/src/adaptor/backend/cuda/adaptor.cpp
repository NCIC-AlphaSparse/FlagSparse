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

#include <cuda.h>

#define FS_DRV(sym)       cu##sym
#define FS_RESULT         CUresult
#define FS_SUCCESS_CODE   CUDA_SUCCESS
#define FS_DEVICE         CUdevice
#define FS_CONTEXT        CUcontext
#define FS_DEVPTR         CUdeviceptr
#define FS_ATTR_MAJOR     CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR
#define FS_ATTR_MINOR     CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR
#define FS_ATTR_ENUM      CUdevice_attribute
#define FS_ATTR_SM_COUNT  CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT
#define FS_ATTR_MAX_THREADS_PER_SM CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR
#define FS_ATTR_WARP_SIZE CU_DEVICE_ATTRIBUTE_WARP_SIZE
#define FS_BACKEND_NAME   "cuda"

#include "../_template.inc"
