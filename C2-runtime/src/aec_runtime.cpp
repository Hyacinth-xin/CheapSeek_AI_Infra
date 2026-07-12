#include "aec_runtime.h"
#include "aec_device_abi.h"

#include <cstring>

namespace {

thread_local aecError_t last_error = AEC_SUCCESS;

aecError_t finish(aecError_t error) {
    if (error != AEC_SUCCESS) {
        last_error = error;
    }
    return error;
}

aecError_t unsupported() {
    return finish(AEC_ERROR_NOT_SUPPORTED);
}

} // namespace

extern "C" {

aecError_t aecDeviceCount(int *count) {
    if (count == nullptr) return finish(AEC_ERROR_INVALID_ARGUMENT);
    aecDeviceCaps caps{};
    if (aecDeviceGetCaps(&caps) != AEC_DEVICE_SUCCESS) return finish(AEC_ERROR_DEVICE);
    *count = static_cast<int>(caps.device_count);
    return AEC_SUCCESS;
}

aecError_t aecDeviceInfo(int device, aecDeviceInfoData *info) {
    if (device != 0 || info == nullptr) return finish(AEC_ERROR_INVALID_ARGUMENT);
    aecDeviceCaps caps{};
    if (aecDeviceGetCaps(&caps) != AEC_DEVICE_SUCCESS) return finish(AEC_ERROR_DEVICE);
    *info = {};
    info->abi_version = AEC_RUNTIME_ABI_VERSION;
    std::strncpy(info->name, "AEC Deterministic Virtual Device", sizeof(info->name) - 1);
    info->memory_bytes = caps.memory_bytes;
    info->dma_channels = caps.dma_channels;
    info->max_threads_per_block = caps.max_threads_per_block;
    info->isa_version = caps.isa_version;
    info->isa_profile = caps.isa_profile;
    info->max_parameter_bytes = caps.max_parameter_bytes;
    return AEC_SUCCESS;
}

aecError_t aecGetLastError(void) {
    const aecError_t value = last_error;
    last_error = AEC_SUCCESS;
    return value;
}

aecError_t aecPeekAtLastError(void) { return last_error; }

const char *aecGetErrorName(aecError_t error) {
    switch (error) {
    case AEC_SUCCESS: return "AEC_SUCCESS";
    case AEC_ERROR_INVALID_ARGUMENT: return "AEC_ERROR_INVALID_ARGUMENT";
    case AEC_ERROR_OUT_OF_MEMORY: return "AEC_ERROR_OUT_OF_MEMORY";
    case AEC_ERROR_INVALID_HANDLE: return "AEC_ERROR_INVALID_HANDLE";
    case AEC_ERROR_INVALID_ADDRESS: return "AEC_ERROR_INVALID_ADDRESS";
    case AEC_ERROR_NOT_READY: return "AEC_ERROR_NOT_READY";
    case AEC_ERROR_NOT_SUPPORTED: return "AEC_ERROR_NOT_SUPPORTED";
    case AEC_ERROR_DEVICE: return "AEC_ERROR_DEVICE";
    case AEC_ERROR_INTERNAL: return "AEC_ERROR_INTERNAL";
    case AEC_ERROR_ISA_TRAP: return "AEC_ERROR_ISA_TRAP";
    default: return "AEC_ERROR_UNKNOWN";
    }
}

aecError_t aecAlloc(aecDevicePtr *out_ptr, size_t) {
    if (out_ptr == nullptr) return finish(AEC_ERROR_INVALID_ARGUMENT);
    return unsupported();
}
aecError_t aecFree(aecDevicePtr) { return unsupported(); }
aecError_t aecCopyH2D(aecDevicePtr, const void *, size_t) { return unsupported(); }
aecError_t aecCopyD2H(void *, aecDevicePtr, size_t) { return unsupported(); }
aecError_t aecCopyAsync(aecDevicePtr, void *, size_t, aecCopyDirection, aecStream_t) { return unsupported(); }
aecError_t aecStreamCreate(aecStream_t *) { return unsupported(); }
aecError_t aecStreamDestroy(aecStream_t) { return unsupported(); }
aecError_t aecStreamSync(aecStream_t) { return unsupported(); }
aecError_t aecEventCreate(aecEvent_t *) { return unsupported(); }
aecError_t aecEventDestroy(aecEvent_t) { return unsupported(); }
aecError_t aecEventRecord(aecEvent_t, aecStream_t) { return unsupported(); }
aecError_t aecEventSynchronize(aecEvent_t) { return unsupported(); }
aecError_t aecEventQuery(aecEvent_t) { return unsupported(); }
aecError_t aecEventElapsedCycles(aecEvent_t, aecEvent_t, uint64_t *) { return unsupported(); }
aecError_t aecHostRegister(void *, size_t) { return unsupported(); }
aecError_t aecHostUnregister(void *) { return unsupported(); }

aecError_t aecGetRuntimeStats(aecRuntimeStats *stats) {
    if (stats == nullptr) return finish(AEC_ERROR_INVALID_ARGUMENT);
    aecDeviceStats device_stats{};
    if (aecDeviceGetStats(&device_stats) != AEC_DEVICE_SUCCESS) return finish(AEC_ERROR_DEVICE);
    static_assert(sizeof(*stats) == sizeof(device_stats));
    std::memcpy(stats, &device_stats, sizeof(*stats));
    stats->abi_version = AEC_RUNTIME_ABI_VERSION;
    return AEC_SUCCESS;
}

aecError_t aecResetRuntimeStats(void) {
    return aecDeviceResetStats() == AEC_DEVICE_SUCCESS ? AEC_SUCCESS
                                                        : finish(AEC_ERROR_DEVICE);
}

aecError_t aecLaunch(aecKernelId, aecDim3, aecDim3, const void *, size_t, aecStream_t) { return unsupported(); }
aecError_t aecMatmulF4(aecDevicePtr, aecDevicePtr, aecDevicePtr, uint32_t, uint32_t, uint32_t, aecStream_t) { return unsupported(); }
aecError_t aecMatmulF8(aecDevicePtr, aecDevicePtr, aecDevicePtr, uint32_t, uint32_t, uint32_t, aecFp8Format, aecStream_t) { return unsupported(); }
aecError_t aecMatmulF16(aecDevicePtr, aecDevicePtr, aecDevicePtr, uint32_t, uint32_t, uint32_t, aecStream_t) { return unsupported(); }
aecError_t aecMatmulBF16(aecDevicePtr, aecDevicePtr, aecDevicePtr, uint32_t, uint32_t, uint32_t, aecStream_t) { return unsupported(); }
aecError_t aecMatmulF32(aecDevicePtr, aecDevicePtr, aecDevicePtr, uint32_t, uint32_t, uint32_t, aecStream_t) { return unsupported(); }
aecError_t aecMatmulF64(aecDevicePtr, aecDevicePtr, aecDevicePtr, uint32_t, uint32_t, uint32_t, aecStream_t) { return unsupported(); }
aecError_t aecMatmulI4(aecDevicePtr, aecDevicePtr, aecDevicePtr, uint32_t, uint32_t, uint32_t, aecStream_t) { return unsupported(); }
aecError_t aecMatmulI8(aecDevicePtr, aecDevicePtr, aecDevicePtr, uint32_t, uint32_t, uint32_t, aecStream_t) { return unsupported(); }
aecError_t aecMatmulI32(aecDevicePtr, aecDevicePtr, aecDevicePtr, uint32_t, uint32_t, uint32_t, aecStream_t) { return unsupported(); }
aecError_t aecAxpy(aecDevicePtr, aecDevicePtr, uint64_t, float, aecStream_t) { return unsupported(); }
aecError_t aecDot(aecDevicePtr, aecDevicePtr, aecDevicePtr, uint64_t, aecStream_t) { return unsupported(); }
aecError_t aecNrm2(aecDevicePtr, aecDevicePtr, uint64_t, aecStream_t) { return unsupported(); }

} // extern "C"
