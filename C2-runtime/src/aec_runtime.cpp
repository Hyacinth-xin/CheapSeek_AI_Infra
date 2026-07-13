#include "aec_runtime.h"
#include "aec_device_abi.h"

#include <algorithm>
#include <cstring>
#include <mutex>
#include <unordered_map>

namespace {

// =====================================================================
// TLS last error (R101)
// =====================================================================
thread_local aecError_t last_error = AEC_SUCCESS;

aecError_t finish(aecError_t error) {
    if (error != AEC_SUCCESS) last_error = error;
    return error;
}

aecError_t unsupported() { return finish(AEC_ERROR_NOT_SUPPORTED); }

// =====================================================================
// Device status → Runtime error
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
struct AllocInfo { aecDevicePtr base; size_t size; };

std::mutex                                alloc_mutex;
std::unordered_map<aecDevicePtr, AllocInfo> live_allocs;

bool span_inside_one_alloc(aecDevicePtr ptr, size_t bytes) {
    if (ptr == 0 || bytes == 0) return false;
    uint64_t end = ptr + bytes;
    if (end < ptr) return false;

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
uint64_t   next_sequence = 1;

// =====================================================================
// Synchronous DMA helper (R103)
// =====================================================================
aecError_t sync_dma(uint16_t opcode, aecDevicePtr dev_ptr,
                    uint64_t host_ptr, size_t bytes) {
    if (host_ptr == 0 || bytes == 0) return finish(AEC_ERROR_INVALID_ARGUMENT);
    if (!span_inside_one_alloc(dev_ptr, bytes))
        return finish(AEC_ERROR_INVALID_ADDRESS);

    aecDeviceCommand cmd{};
    cmd.abi_version = AEC_DEVICE_ABI_VERSION;
    cmd.opcode      = opcode;
    cmd.flags       = AEC_DEVICE_FLAG_NONE;
    { std::lock_guard<std::mutex> lock(seq_mutex); cmd.sequence = next_sequence++; }
    cmd.stream_id   = 0;
    cmd.bytes       = bytes;
    cmd.chunk_bytes = static_cast<uint32_t>(bytes);
    cmd.queue_depth = 1;
    cmd.channel     = 0;
    if (opcode == AEC_DEVICE_OP_H2D) {
        cmd.host_address = host_ptr; cmd.dst = dev_ptr;
    } else {
        cmd.src = dev_ptr; cmd.host_address = host_ptr;
    }

    aecDeviceCompletion comp{};
    aecDeviceStatus st = aecDeviceSubmit(&cmd, &comp);
    if (st != AEC_DEVICE_SUCCESS) return finish(device_status_to_error(st));
    if (comp.status != AEC_DEVICE_SUCCESS)
        return finish(device_status_to_error(static_cast<aecDeviceStatus>(comp.status)));
    return AEC_SUCCESS;
}

// =====================================================================
// Little-endian writers for canonical parameter blocks
// =====================================================================
void write_u64_le(uint8_t *buf, size_t off, uint64_t v) {
    buf[off+0]=uint8_t(v); buf[off+1]=uint8_t(v>>8);
    buf[off+2]=uint8_t(v>>16); buf[off+3]=uint8_t(v>>24);
    buf[off+4]=uint8_t(v>>32); buf[off+5]=uint8_t(v>>40);
    buf[off+6]=uint8_t(v>>48); buf[off+7]=uint8_t(v>>56);
}
void write_u32_le(uint8_t *buf, size_t off, uint32_t v) {
    buf[off+0]=uint8_t(v); buf[off+1]=uint8_t(v>>8);
    buf[off+2]=uint8_t(v>>16); buf[off+3]=uint8_t(v>>24);
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

aecError_t aecGetLastError(void) { aecError_t v=last_error; last_error=AEC_SUCCESS; return v; }
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
    aecDeviceStatus st = aecDeviceAlloc(bytes, 64, &ptr);
    if (st != AEC_DEVICE_SUCCESS) return finish(device_status_to_error(st));
    { std::lock_guard<std::mutex> lock(alloc_mutex); live_allocs[ptr] = {ptr, bytes}; }
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
                    return finish(AEC_ERROR_INVALID_ADDRESS);
            }
            return finish(AEC_ERROR_INVALID_ADDRESS);
        }
        live_allocs.erase(it);
    }
    aecDeviceStatus st = aecDeviceFree(ptr);
    if (st != AEC_DEVICE_SUCCESS) return finish(device_status_to_error(st));
    return AEC_SUCCESS;
}

// =====================================================================
// R103  Synchronous Copy
// =====================================================================

aecError_t aecCopyH2D(aecDevicePtr dst, const void *src, size_t bytes) {
    return sync_dma(AEC_DEVICE_OP_H2D, dst, reinterpret_cast<uint64_t>(src), bytes);
}
aecError_t aecCopyD2H(void *dst, aecDevicePtr src, size_t bytes) {
    return sync_dma(AEC_DEVICE_OP_D2H, src, reinterpret_cast<uint64_t>(dst), bytes);
}

// =====================================================================
// R104 + R201  Kernel Launch (Vector Add + GEMM)
// =====================================================================

aecError_t aecLaunch(aecKernelId kernel, aecDim3 grid, aecDim3 block,
                     const void *args, size_t args_size, aecStream_t stream) {
    // --- grid / block validation ---
    if (grid.x == 0 || grid.y == 0 || grid.z == 0) return finish(AEC_ERROR_INVALID_ARGUMENT);
    if (block.x == 0 || block.y == 0 || block.z == 0) return finish(AEC_ERROR_INVALID_ARGUMENT);
    uint64_t block_vol = static_cast<uint64_t>(block.x) * block.y * block.z;
    if (block_vol > 1024) return finish(AEC_ERROR_INVALID_ARGUMENT);
    if (args == nullptr || args_size == 0) return finish(AEC_ERROR_INVALID_ARGUMENT);

    // --- streams not implemented yet ---
    if (stream != nullptr) return unsupported();

    // --- map kernel ID → (semantic, dtype, variant) ---
    uint32_t semantic, dtype, variant;

    switch (kernel) {
    // ---- Vector Add ----
    case AEC_KERNEL_VECTOR_ADD_F32:
        if (args_size != sizeof(aecVectorAddArgs))
            return finish(AEC_ERROR_INVALID_ARGUMENT);
        semantic = static_cast<uint32_t>(kernel);  // 1
        dtype    = AEC_DTYPE_FP32;                  // 6
        variant  = 0;
        break;

    // ---- GEMM ----
    case AEC_KERNEL_GEMM_NAIVE:
    case AEC_KERNEL_GEMM_TILED:
    case AEC_KERNEL_GEMM_VECTORIZED: {
        if (args_size != 40)  // canonical GEMM param block is 40 bytes (struct has padding)
            return finish(AEC_ERROR_INVALID_ARGUMENT);
        const auto *ga = static_cast<const aecGemmArgs *>(args);
        if (ga->m < 1 || ga->m > 256 ||
            ga->n < 1 || ga->n > 256 ||
            ga->k < 1 || ga->k > 256)
            return finish(AEC_ERROR_INVALID_ARGUMENT);
        semantic = static_cast<uint32_t>(kernel);  // 10 / 11 / 12
        dtype    = ga->dtype;
        switch (kernel) {
        case AEC_KERNEL_GEMM_NAIVE:     variant = 1; break;
        case AEC_KERNEL_GEMM_TILED:     variant = 2; break;
        case AEC_KERNEL_GEMM_VECTORIZED: variant = 3; break;
        default: return unsupported(); // unreachable
        }
        break;
    }

    default:
        return unsupported();
    }

    // --- resolve kernel image ---
    aecDeviceKernelInfo kinfo{};
    aecDeviceStatus st = aecDeviceResolveKernel(semantic, dtype, variant, &kinfo);
    if (st != AEC_DEVICE_SUCCESS)
        return finish(device_status_to_error(st));

    // --- build canonical parameter block (little-endian, tightly packed) ---
    uint8_t params[AEC_DEVICE_MAX_PARAM_BYTES] = {};

    if (kernel == AEC_KERNEL_VECTOR_ADD_F32) {
        const auto *va = static_cast<const aecVectorAddArgs *>(args);
        write_u64_le(params,  0, va->a);
        write_u64_le(params,  8, va->b);
        write_u64_le(params, 16, va->c);
        write_u64_le(params, 24, va->count);
    } else {
        const auto *ga = static_cast<const aecGemmArgs *>(args);
        write_u64_le(params,  0, ga->a);
        write_u64_le(params,  8, ga->b);
        write_u64_le(params, 16, ga->c);
        write_u32_le(params, 24, ga->m);
        write_u32_le(params, 28, ga->n);
        write_u32_le(params, 32, ga->k);
        write_u32_le(params, 36, ga->dtype);
    }

    // --- build ISA_LAUNCH command ---
    aecDeviceCommand cmd{};
    cmd.abi_version   = AEC_DEVICE_ABI_VERSION;
    cmd.opcode        = AEC_DEVICE_OP_ISA_LAUNCH;
    cmd.flags         = AEC_DEVICE_FLAG_NONE;
    { std::lock_guard<std::mutex> lock(seq_mutex); cmd.sequence = next_sequence++; }
    cmd.stream_id       = 0;
    cmd.kernel_handle   = kinfo.handle;
    cmd.isa_version     = kinfo.isa_version;
    cmd.entry_pc        = kinfo.entry_pc;
    cmd.grid            = aecDeviceDim3{grid.x, grid.y, grid.z};
    cmd.block           = aecDeviceDim3{block.x, block.y, block.z};
    cmd.parameter_bytes = static_cast<uint32_t>(args_size);
    std::memcpy(cmd.parameters, params, args_size);

    // --- submit & wait ---
    aecDeviceCompletion comp{};
    st = aecDeviceSubmit(&cmd, &comp);
    if (st != AEC_DEVICE_SUCCESS)
        return finish(device_status_to_error(st));
    if (comp.status != AEC_DEVICE_SUCCESS)
        return finish(device_status_to_error(
                         static_cast<aecDeviceStatus>(comp.status)));

    return AEC_SUCCESS;
}

// =====================================================================
// R201  GEMM — FP32 / INT32
// =====================================================================

// Internal: validate & launch GEMM
static aecError_t gemm_launch(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                              uint32_t m, uint32_t n, uint32_t k,
                              aecDataType dtype, aecStream_t stream) {
    if (m < 1 || m > 256 || n < 1 || n > 256 || k < 1 || k > 256)
        return finish(AEC_ERROR_INVALID_ARGUMENT);

    aecGemmArgs args = {a, b, c, m, n, k, static_cast<uint32_t>(dtype), 0};

    // grid = (n, m, 1) — each CTA computes one output element
    // block = (1, 1, 1)
    aecDim3 grid  = {n, m, 1};
    aecDim3 block = {1, 1, 1};

    return aecLaunch(AEC_KERNEL_GEMM_NAIVE, grid, block,
                     &args, 40, stream);  // canonical param block = 40 bytes
}

aecError_t aecMatmulF32(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                        uint32_t m, uint32_t n, uint32_t k,
                        aecStream_t stream) {
    return gemm_launch(a, b, c, m, n, k, AEC_DTYPE_FP32, stream);
}

aecError_t aecMatmulI32(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                        uint32_t m, uint32_t n, uint32_t k,
                        aecStream_t stream) {
    return gemm_launch(a, b, c, m, n, k, AEC_DTYPE_INT32, stream);
}

// =====================================================================
// R202 / R203  Other GEMM dtypes (stubs)
// =====================================================================

aecError_t aecMatmulF4(aecDevicePtr, aecDevicePtr, aecDevicePtr,
                       uint32_t, uint32_t, uint32_t, aecStream_t) { return unsupported(); }
aecError_t aecMatmulF8(aecDevicePtr, aecDevicePtr, aecDevicePtr,
                       uint32_t, uint32_t, uint32_t, aecFp8Format, aecStream_t) { return unsupported(); }
aecError_t aecMatmulF16(aecDevicePtr, aecDevicePtr, aecDevicePtr,
                        uint32_t, uint32_t, uint32_t, aecStream_t) { return unsupported(); }
aecError_t aecMatmulBF16(aecDevicePtr, aecDevicePtr, aecDevicePtr,
                         uint32_t, uint32_t, uint32_t, aecStream_t) { return unsupported(); }
aecError_t aecMatmulF64(aecDevicePtr, aecDevicePtr, aecDevicePtr,
                        uint32_t, uint32_t, uint32_t, aecStream_t) { return unsupported(); }
aecError_t aecMatmulI4(aecDevicePtr, aecDevicePtr, aecDevicePtr,
                       uint32_t, uint32_t, uint32_t, aecStream_t) { return unsupported(); }
aecError_t aecMatmulI8(aecDevicePtr, aecDevicePtr, aecDevicePtr,
                       uint32_t, uint32_t, uint32_t, aecStream_t) { return unsupported(); }

// =====================================================================
// R105+  (stubs)
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

aecError_t aecAxpy(aecDevicePtr, aecDevicePtr, uint64_t, float, aecStream_t) { return unsupported(); }
aecError_t aecDot(aecDevicePtr, aecDevicePtr, aecDevicePtr, uint64_t, aecStream_t) { return unsupported(); }
aecError_t aecNrm2(aecDevicePtr, aecDevicePtr, uint64_t, aecStream_t) { return unsupported(); }

} // extern "C"
