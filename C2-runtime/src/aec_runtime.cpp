#include "aec_runtime.h"
#include "aec_device_abi.h"

#include <cstring>
#include <mutex>
#include <unordered_map>

namespace {

// =====================================================================
// TLS last error (R101)
// =====================================================================
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

// =====================================================================
// Device status → Runtime error conversion
// =====================================================================
aecError_t device_status_to_error(aecDeviceStatus status) {
    switch (status) {
    case AEC_DEVICE_SUCCESS:           return AEC_SUCCESS;
    case AEC_DEVICE_INVALID_ARGUMENT:  return AEC_ERROR_INVALID_ARGUMENT;
    case AEC_DEVICE_OUT_OF_MEMORY:     return AEC_ERROR_OUT_OF_MEMORY;
    case AEC_DEVICE_INVALID_ADDRESS:   return AEC_ERROR_INVALID_ADDRESS;
    case AEC_DEVICE_UNSUPPORTED:       return AEC_ERROR_NOT_SUPPORTED;
    case AEC_DEVICE_INJECTED_FAULT:    return AEC_ERROR_DEVICE;
    case AEC_DEVICE_ISA_TRAP:          return AEC_ERROR_ISA_TRAP;
    case AEC_DEVICE_INTERNAL:
    default:                           return AEC_ERROR_INTERNAL;
    }
}

// =====================================================================
// Allocation tracking (R102)
// =====================================================================
struct AllocInfo {
    aecDevicePtr base;
    size_t      size;
};

std::mutex                                alloc_mutex;
std::unordered_map<aecDevicePtr, AllocInfo> live_allocs;

bool span_inside_one_alloc(aecDevicePtr ptr, size_t bytes) {
    if (ptr == 0 || bytes == 0) return false;
    uint64_t end = ptr + bytes;
    if (end < ptr) return false;   // overflow

    std::lock_guard<std::mutex> lock(alloc_mutex);
    for (const auto &kv : live_allocs) {
        const AllocInfo &a = kv.second;
        if (ptr >= a.base && end <= a.base + a.size) return true;
    }
    return false;
}

// =====================================================================
// Sequence counter (R103+)
// =====================================================================
std::mutex seq_mutex;
uint64_t   next_sequence = 1;   // non-zero, strictly increasing

// =====================================================================
// Synchronous DMA helper (R103)
// =====================================================================
aecError_t sync_dma(uint16_t opcode, aecDevicePtr dev_ptr,
                    uint64_t host_ptr, size_t bytes) {
    // --- parameter checks ---
    if (host_ptr == 0)    return finish(AEC_ERROR_INVALID_ARGUMENT);
    if (bytes == 0)       return finish(AEC_ERROR_INVALID_ARGUMENT);

    // --- bounds check: device span must be fully inside one live alloc ---
    if (!span_inside_one_alloc(dev_ptr, bytes))
        return finish(AEC_ERROR_INVALID_ADDRESS);

    // --- build command ---
    aecDeviceCommand cmd{};
    cmd.abi_version = AEC_DEVICE_ABI_VERSION;
    cmd.opcode      = opcode;
    cmd.flags       = AEC_DEVICE_FLAG_NONE;

    {
        std::lock_guard<std::mutex> lock(seq_mutex);
        cmd.sequence = next_sequence++;
    }

    cmd.stream_id = 0;               // null stream → synchronous
    cmd.bytes     = bytes;
    cmd.chunk_bytes = static_cast<uint32_t>(bytes);  // one chunk
    cmd.queue_depth = 1;
    cmd.channel     = 0;

    if (opcode == AEC_DEVICE_OP_H2D) {
        cmd.host_address = host_ptr;
        cmd.dst          = dev_ptr;
    } else {  // D2H
        cmd.src          = dev_ptr;
        cmd.host_address = host_ptr;
    }

    // --- submit & wait for completion ---
    aecDeviceCompletion comp{};
    aecDeviceStatus status = aecDeviceSubmit(&cmd, &comp);
    if (status != AEC_DEVICE_SUCCESS)
        return finish(device_status_to_error(status));

    if (comp.status != AEC_DEVICE_SUCCESS)
        return finish(device_status_to_error(
                         static_cast<aecDeviceStatus>(comp.status)));

    return AEC_SUCCESS;
}

} // namespace

extern "C" {

// =====================================================================
// R101  Device / Error
// =====================================================================

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

// =====================================================================
// R102  Allocation / Free
// =====================================================================

aecError_t aecAlloc(aecDevicePtr *out_ptr, size_t bytes) {
    if (out_ptr == nullptr) return finish(AEC_ERROR_INVALID_ARGUMENT);

    aecDevicePtr ptr = 0;
    aecDeviceStatus status = aecDeviceAlloc(bytes, 64, &ptr);
    if (status != AEC_DEVICE_SUCCESS)
        return finish(device_status_to_error(status));

    {
        std::lock_guard<std::mutex> lock(alloc_mutex);
        live_allocs[ptr] = {ptr, bytes};
    }

    *out_ptr = ptr;
    return AEC_SUCCESS;
}

aecError_t aecFree(aecDevicePtr ptr) {
    if (ptr == 0) return finish(AEC_ERROR_INVALID_ARGUMENT);

    {
        std::lock_guard<std::mutex> lock(alloc_mutex);

        auto it = live_allocs.find(ptr);
        if (it == live_allocs.end()) {
            for (const auto &kv : live_allocs) {
                const AllocInfo &a = kv.second;
                if (ptr > a.base && ptr < a.base + a.size)
                    return finish(AEC_ERROR_INVALID_ADDRESS); // interior
            }
            return finish(AEC_ERROR_INVALID_ADDRESS); // stale / double-free
        }

        live_allocs.erase(it);
    }

    aecDeviceStatus status = aecDeviceFree(ptr);
    if (status != AEC_DEVICE_SUCCESS)
        return finish(device_status_to_error(status));

    return AEC_SUCCESS;
}

// =====================================================================
// R103  Synchronous Copy
// =====================================================================

aecError_t aecCopyH2D(aecDevicePtr dst, const void *src, size_t bytes) {
    return sync_dma(AEC_DEVICE_OP_H2D, dst,
                    reinterpret_cast<uint64_t>(src), bytes);
}

aecError_t aecCopyD2H(void *dst, aecDevicePtr src, size_t bytes) {
    return sync_dma(AEC_DEVICE_OP_D2H, src,
                    reinterpret_cast<uint64_t>(dst), bytes);
}

// =====================================================================
// R104+  (stubs)
// =====================================================================

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
