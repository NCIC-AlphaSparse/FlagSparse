// Shared body for the driver-API backends. Included by backend/<name>/adaptor.cpp
// after defining FS_DRV(sym) and FS_HEADER. CUDA and MUSA differ only in the
// symbol prefix (cu* / mu*), so keeping one body stops the two from drifting.

#include <mutex>
#include <stdexcept>
#include <string>

#include "adaptor/adaptor.hpp"

namespace flagsparse::adaptor {
namespace {

bool ok(FS_RESULT r) { return r == FS_SUCCESS_CODE; }

std::once_flag g_init;
bool g_init_ok = false;
int g_device = 0;
std::string g_arch;

void init_once() {
    std::call_once(g_init, [] {
        if (!ok(FS_DRV(Init)(0))) return;
        int count = 0;
        if (!ok(FS_DRV(DeviceGetCount)(&count)) || count <= 0) return;
        FS_DEVICE dev;
        if (!ok(FS_DRV(DeviceGet)(&dev, 0))) return;
        int major = 0, minor = 0;
        FS_DRV(DeviceGetAttribute)(&major, FS_ATTR_MAJOR, dev);
        FS_DRV(DeviceGetAttribute)(&minor, FS_ATTR_MINOR, dev);
        g_arch = std::to_string(major * 10 + minor);
        g_device = 0;
        // A primary context retained here keeps every later launch on the same
        // context; creating one per call is how stream/handle mismatches start.
        FS_CONTEXT pctx;
        if (!ok(FS_DRV(DevicePrimaryCtxRetain)(&pctx, dev))) return;
        if (!ok(FS_DRV(CtxSetCurrent)(pctx))) return;
        g_init_ok = true;
    });
}

}  // namespace

flagsparseStatus_t ensure_device(int& device_index, std::string& device_arch) {
    init_once();
    if (!g_init_ok) return FLAGSPARSE_STATUS_NOT_INITIALIZED;
    device_index = g_device;
    device_arch = g_arch;
    return FLAGSPARSE_STATUS_SUCCESS;
}

const char* backend_name() { return FS_BACKEND_NAME; }

int device_count() {
    init_once();
    int count = 0;
    if (!ok(FS_DRV(DeviceGetCount)(&count))) return 0;
    return count;
}

std::string device_architecture(int) {
    init_once();
    return g_arch;
}

namespace {
int device_attribute(int attr) {
    init_once();
    FS_DEVICE dev;
    if (!ok(FS_DRV(DeviceGet)(&dev, g_device))) return 0;
    int value = 0;
    if (!ok(FS_DRV(DeviceGetAttribute)(&value, static_cast<FS_ATTR_ENUM>(attr), dev))) return 0;
    return value;
}
}  // namespace

int multiprocessor_count(int) { return device_attribute(FS_ATTR_SM_COUNT); }
int max_threads_per_multiprocessor(int) { return device_attribute(FS_ATTR_MAX_THREADS_PER_SM); }
int warp_size(int) {
    const int w = device_attribute(FS_ATTR_WARP_SIZE);
    return w > 0 ? w : 32;
}

void synchronize() {
    init_once();
    FS_DRV(CtxSynchronize)();
}

flagsparseStream_t default_stream() { return nullptr; }

flagsparseStatus_t device_malloc(DevicePtr* ptr, std::size_t bytes) {
    init_once();
    if (ptr == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    *ptr = 0;
    if (bytes == 0) return FLAGSPARSE_STATUS_SUCCESS;
    FS_DEVPTR raw = 0;
    if (!ok(FS_DRV(MemAlloc)(&raw, bytes))) return FLAGSPARSE_STATUS_ALLOC_FAILED;
    *ptr = static_cast<DevicePtr>(raw);
    return FLAGSPARSE_STATUS_SUCCESS;
}

void device_free(DevicePtr ptr) {
    if (ptr != 0) FS_DRV(MemFree)(static_cast<FS_DEVPTR>(ptr));
}

flagsparseStatus_t memcpy_h2d(DevicePtr dst, const void* src, std::size_t bytes) {
    if (bytes == 0) return FLAGSPARSE_STATUS_SUCCESS;
    if (dst == 0 || src == nullptr) return FLAGSPARSE_STATUS_INVALID_VALUE;
    return ok(FS_DRV(MemcpyHtoD)(static_cast<FS_DEVPTR>(dst), src, bytes))
               ? FLAGSPARSE_STATUS_SUCCESS : FLAGSPARSE_STATUS_MAPPING_ERROR;
}

flagsparseStatus_t memcpy_d2h(void* dst, DevicePtr src, std::size_t bytes) {
    if (bytes == 0) return FLAGSPARSE_STATUS_SUCCESS;
    if (dst == nullptr || src == 0) return FLAGSPARSE_STATUS_INVALID_VALUE;
    return ok(FS_DRV(MemcpyDtoH)(dst, static_cast<FS_DEVPTR>(src), bytes))
               ? FLAGSPARSE_STATUS_SUCCESS : FLAGSPARSE_STATUS_MAPPING_ERROR;
}

flagsparseStatus_t memset_device(DevicePtr dst, int value, std::size_t bytes) {
    if (bytes == 0) return FLAGSPARSE_STATUS_SUCCESS;
    if (dst == 0) return FLAGSPARSE_STATUS_INVALID_VALUE;
    return ok(FS_DRV(MemsetD8)(static_cast<FS_DEVPTR>(dst),
                               static_cast<unsigned char>(value), bytes))
               ? FLAGSPARSE_STATUS_SUCCESS : FLAGSPARSE_STATUS_MAPPING_ERROR;
}

}  // namespace flagsparse::adaptor
