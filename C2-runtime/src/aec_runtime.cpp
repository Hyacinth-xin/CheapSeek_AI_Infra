#include "aec_runtime.h"
#include "aec_device_abi.h"

#include <cmath>
#include <algorithm>
#include <array>
#include <cstring>
#include <deque>
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
        if (ptr >= kv.second.base && end <= kv.second.base + kv.second.size)
            return true;
    }
    return false;
}

// =====================================================================
// Sequence + channel counter
// =====================================================================
std::mutex seq_mutex;
uint64_t   next_sequence = 1;
uint8_t    next_dma_channel = 0;  // round-robin assignment for R302

uint64_t take_sequence() {
    std::lock_guard<std::mutex> lock(seq_mutex);
    return next_sequence++;
}

// =====================================================================
// Little-endian writers
// =====================================================================
void write_u64_le(uint8_t *b, size_t o, uint64_t v) {
    b[o+0]=uint8_t(v); b[o+1]=uint8_t(v>>8); b[o+2]=uint8_t(v>>16); b[o+3]=uint8_t(v>>24);
    b[o+4]=uint8_t(v>>32); b[o+5]=uint8_t(v>>40); b[o+6]=uint8_t(v>>48); b[o+7]=uint8_t(v>>56);
}
void write_u32_le(uint8_t *b, size_t o, uint32_t v) {
    b[o+0]=uint8_t(v); b[o+1]=uint8_t(v>>8); b[o+2]=uint8_t(v>>16); b[o+3]=uint8_t(v>>24);
}

// =====================================================================
// Stream + Event infrastructure (R105 / R106)
// =====================================================================

static constexpr size_t MAX_STREAMS = 128;
static constexpr size_t MAX_EVENTS  = 128;

// Forward decl
struct Event;

// One pending operation in a stream queue
struct StreamOp {
    aecDeviceCommand cmd;
    bool             is_event_record = false;
    Event           *event = nullptr;      // non-null → event record marker
    uint64_t         event_generation = 0; // which generation this marker belongs to
    bool             args_copied = false;  // for ISA launch: args live in cmd.parameters
};

// Stream: FIFO queue of pending ops (lazy execution model)
struct Stream {
    std::mutex              mtx;
    std::deque<StreamOp>    queue;
    aecError_t              first_error = AEC_SUCCESS;
    bool                    handle_alive = true; // false once destroy removes handle
    uint8_t                 dma_channel = 0;     // assigned round-robin on creation

    // Enqueue an operation (caller must hold external lock if needed)
    void enqueue(StreamOp op) {
        std::lock_guard<std::mutex> lock(mtx);
        queue.push_back(std::move(op));
    }
};

// Event: tracks latest record generation
struct Event {
    std::mutex   mtx;
    uint64_t     generation = 0;     // incremented on each aecEventRecord
    uint64_t     cycle      = 0;     // set when record marker completes
    bool         recorded   = false; // has ever been recorded
    bool         completed  = false; // latest generation completed
    aecError_t   error      = AEC_SUCCESS;
    Stream      *stream     = nullptr; // stream latest record was enqueued on
};

// ---- Registries ----
std::mutex                stream_reg_mtx;
std::array<Stream*, MAX_STREAMS> streams = {};

std::mutex                event_reg_mtx;
std::array<Event*, MAX_EVENTS>   events  = {};

// Allocate a handle (return index) or return SIZE_MAX
size_t alloc_stream_handle() {
    std::lock_guard<std::mutex> lock(stream_reg_mtx);
    for (size_t i = 0; i < MAX_STREAMS; ++i)
        if (streams[i] == nullptr) return i;
    return SIZE_MAX;
}
size_t alloc_event_handle() {
    std::lock_guard<std::mutex> lock(event_reg_mtx);
    for (size_t i = 0; i < MAX_EVENTS; ++i)
        if (events[i] == nullptr) return i;
    return SIZE_MAX;
}

Stream *get_stream(aecStream_t h) {
    if (h == nullptr) return nullptr;
    size_t idx = reinterpret_cast<size_t>(h) - 1;  // handles are idx+1
    std::lock_guard<std::mutex> lock(stream_reg_mtx);
    if (idx >= MAX_STREAMS) return nullptr;
    Stream *s = streams[idx];
    if (!s || !s->handle_alive) return nullptr;
    return s;
}

Event *get_event(aecEvent_t h) {
    if (h == nullptr) return nullptr;
    size_t idx = reinterpret_cast<size_t>(h) - 1;  // handles are idx+1
    std::lock_guard<std::mutex> lock(event_reg_mtx);
    if (idx >= MAX_EVENTS) return nullptr;
    return events[idx];
}

// ---- Lazy execution: drain a stream's queue ----
void process_stream(Stream *s) {
    std::lock_guard<std::mutex> lock(s->mtx);
    while (!s->queue.empty()) {
        StreamOp &op = s->queue.front();

        aecDeviceCompletion comp{};
        aecDeviceStatus st = aecDeviceSubmit(&op.cmd, &comp);

        aecError_t err = AEC_SUCCESS;
        if (st != AEC_DEVICE_SUCCESS)
            err = device_status_to_error(st);
        else if (comp.status != AEC_DEVICE_SUCCESS)
            err = device_status_to_error(static_cast<aecDeviceStatus>(comp.status));

        // Capture first error
        if (err != AEC_SUCCESS && s->first_error == AEC_SUCCESS)
            s->first_error = err;

        // Update event if this is a record marker (only if generation still matches).
        // Use total_virtual_cycles from stats (cumulative), NOT comp.virtual_cycles
        // (which is per-command and always ~4 for BARRIER).
        if (op.is_event_record && op.event) {
            Event *ev = op.event;
            aecDeviceStats ds{};
            aecDeviceGetStats(&ds);
            std::lock_guard<std::mutex> ev_lock(ev->mtx);
            if (ev->generation == op.event_generation) {
                ev->cycle     = ds.total_virtual_cycles;
                ev->completed = true;
                ev->error     = err;
            }
        }

        s->queue.pop_front();
    }
}

// Drain ALL live streams (used by aecFree to wait for pending work)
void process_all_streams() {
    std::lock_guard<std::mutex> lock(stream_reg_mtx);
    for (auto *s : streams) {
        if (s && s->handle_alive) process_stream(s);
    }
}

// =====================================================================
// Synchronous DMA (R103, also used as fallback for null-stream async)
// =====================================================================
// fwd decl (defined after reg tracking)
bool is_registered(void *ptr, size_t bytes);

aecError_t sync_dma(uint16_t opcode, aecDevicePtr dev_ptr,
                    uint64_t host_ptr, size_t bytes, uint8_t channel = 0) {
    if (host_ptr == 0 || bytes == 0) return finish(AEC_ERROR_INVALID_ARGUMENT);
    if (!span_inside_one_alloc(dev_ptr, bytes))
        return finish(AEC_ERROR_INVALID_ADDRESS);
    process_all_streams();

    aecDeviceCommand cmd{};
    cmd.abi_version = AEC_DEVICE_ABI_VERSION;
    cmd.opcode      = opcode;
    cmd.flags       = AEC_DEVICE_FLAG_NONE;
    // Check if host range is registered → REGISTERED + ZERO_COPY
    if (is_registered(reinterpret_cast<void*>(host_ptr), bytes)) {
        cmd.flags |= AEC_DEVICE_FLAG_REGISTERED | AEC_DEVICE_FLAG_ZERO_COPY;
    }
    cmd.sequence    = take_sequence();
    cmd.stream_id   = 0;
    cmd.bytes       = bytes;
    cmd.chunk_bytes = static_cast<uint32_t>(bytes);
    cmd.queue_depth = 1;
    cmd.channel     = channel;
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
// Host registration tracking (R303)
// =====================================================================
struct RegRange { void *base; size_t bytes; };
std::mutex                                 reg_mutex;
std::unordered_map<void*, RegRange>         registered_ranges;

// Check if [ptr, ptr+bytes) is fully inside a registered range.
// Returns true if it is (→ REGISTERED+ZERO_COPY can be used).
bool is_registered(void *ptr, size_t bytes) {
    if (!ptr || bytes == 0) return false;
    uint64_t end = reinterpret_cast<uint64_t>(ptr) + bytes;
    if (end < reinterpret_cast<uint64_t>(ptr)) return false;
    std::lock_guard<std::mutex> lock(reg_mutex);
    for (const auto &kv : registered_ranges) {
        uint64_t base = reinterpret_cast<uint64_t>(kv.second.base);
        uint64_t bend = base + kv.second.bytes;
        if (reinterpret_cast<uint64_t>(ptr) >= base && end <= bend)
            return true;
    }
    return false;
}
} // namespace

extern "C" {

// =====================================================================
// R101  Device / Error
// =====================================================================
aecError_t aecDeviceCount(int *count) {
    if (!count) return finish(AEC_ERROR_INVALID_ARGUMENT);
    aecDeviceCaps caps{};
    if (aecDeviceGetCaps(&caps) != AEC_DEVICE_SUCCESS) return finish(AEC_ERROR_DEVICE);
    *count = static_cast<int>(caps.device_count);
    return AEC_SUCCESS;
}

aecError_t aecDeviceInfo(int device, aecDeviceInfoData *info) {
    if (device != 0 || !info) return finish(AEC_ERROR_INVALID_ARGUMENT);
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
    if (!out_ptr || bytes == 0) return finish(AEC_ERROR_INVALID_ARGUMENT);
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
            for (const auto &kv : live_allocs)
                if (ptr > kv.second.base && ptr < kv.second.base + kv.second.size)
                    return finish(AEC_ERROR_INVALID_ADDRESS);
            return finish(AEC_ERROR_INVALID_ADDRESS);
        }
        live_allocs.erase(it);
    }

    // Wait for all pending async work before freeing
    process_all_streams();

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
// R105  Async Copy
// =====================================================================
aecError_t aecCopyAsync(aecDevicePtr dev_ptr, void *host_ptr, size_t bytes,
                        aecCopyDirection dir, aecStream_t stream) {
    if (!host_ptr || bytes == 0) return finish(AEC_ERROR_INVALID_ARGUMENT);
    // Span validation deferred to device submit time for async path

    // Null stream → synchronous fallback
    if (stream == nullptr) {
        uint16_t op = (dir == AEC_COPY_HOST_TO_DEVICE) ? AEC_DEVICE_OP_H2D : AEC_DEVICE_OP_D2H;
        return sync_dma(op, dev_ptr, reinterpret_cast<uint64_t>(host_ptr), bytes, 0);
    }

    Stream *s = get_stream(stream);
    if (!s) return finish(AEC_ERROR_INVALID_HANDLE);

    StreamOp sop{};
    sop.cmd.abi_version = AEC_DEVICE_ABI_VERSION;
    sop.cmd.opcode = (dir == AEC_COPY_HOST_TO_DEVICE) ? AEC_DEVICE_OP_H2D : AEC_DEVICE_OP_D2H;
    sop.cmd.flags    = AEC_DEVICE_FLAG_NONE;
    // Check if host range is registered → REGISTERED + ZERO_COPY
    if (is_registered(host_ptr, bytes))
        sop.cmd.flags |= AEC_DEVICE_FLAG_REGISTERED | AEC_DEVICE_FLAG_ZERO_COPY;
    sop.cmd.sequence    = take_sequence();
    sop.cmd.stream_id   = reinterpret_cast<uint64_t>(stream);
    sop.cmd.bytes       = bytes;
    sop.cmd.chunk_bytes = static_cast<uint32_t>(bytes);
    sop.cmd.queue_depth = 1;
    sop.cmd.channel     = s->dma_channel;
    if (dir == AEC_COPY_HOST_TO_DEVICE) {
        sop.cmd.host_address = reinterpret_cast<uint64_t>(host_ptr);
        sop.cmd.dst          = dev_ptr;
    } else {
        sop.cmd.src          = dev_ptr;
        sop.cmd.host_address = reinterpret_cast<uint64_t>(host_ptr);
    }

    s->enqueue(std::move(sop));
    return AEC_SUCCESS;
}

// =====================================================================
// R105  Stream
// =====================================================================
aecError_t aecStreamCreate(aecStream_t *out) {
    if (!out) return finish(AEC_ERROR_INVALID_ARGUMENT);
    size_t idx = alloc_stream_handle();
    if (idx == SIZE_MAX) return finish(AEC_ERROR_OUT_OF_MEMORY);

    auto *s = new Stream();
    { std::lock_guard<std::mutex> lock(seq_mutex); s->dma_channel = next_dma_channel++ % 2; }
    {
        std::lock_guard<std::mutex> lock(stream_reg_mtx);
        streams[idx] = s;
    }
    *out = reinterpret_cast<aecStream_t>(idx + 1);  // +1 so handle is never null
    return AEC_SUCCESS;
}

aecError_t aecStreamDestroy(aecStream_t stream) {
    if (!stream) return finish(AEC_ERROR_INVALID_HANDLE);
    size_t idx = reinterpret_cast<size_t>(stream) - 1;

    Stream *s = nullptr;
    {
        std::lock_guard<std::mutex> lock(stream_reg_mtx);
        if (idx >= MAX_STREAMS || !streams[idx]) return finish(AEC_ERROR_INVALID_HANDLE);
        s = streams[idx];
        s->handle_alive = false;    // prevent new enqueues
        streams[idx] = nullptr;     // remove from registry
    }

    // Drain remaining queue, then delete
    process_stream(s);
    delete s;
    return AEC_SUCCESS;
}

aecError_t aecStreamSync(aecStream_t stream) {
    if (!stream) return finish(AEC_ERROR_INVALID_HANDLE);
    Stream *s = get_stream(stream);
    if (!s) return finish(AEC_ERROR_INVALID_HANDLE);

    process_stream(s);

    aecError_t err = s->first_error;
    s->first_error = AEC_SUCCESS;  // clear after reporting
    return err;
}

// =====================================================================
// R106  Event
// =====================================================================
aecError_t aecEventCreate(aecEvent_t *out) {
    if (!out) return finish(AEC_ERROR_INVALID_ARGUMENT);
    size_t idx = alloc_event_handle();
    if (idx == SIZE_MAX) return finish(AEC_ERROR_OUT_OF_MEMORY);

    auto *ev = new Event();
    {
        std::lock_guard<std::mutex> lock(event_reg_mtx);
        events[idx] = ev;
    }
    *out = reinterpret_cast<aecEvent_t>(idx + 1);  // +1 so handle is never null
    return AEC_SUCCESS;
}

aecError_t aecEventDestroy(aecEvent_t event) {
    if (!event) return finish(AEC_ERROR_INVALID_HANDLE);
    size_t idx = reinterpret_cast<size_t>(event) - 1;

    Event *ev = nullptr;
    {
        std::lock_guard<std::mutex> lock(event_reg_mtx);
        if (idx >= MAX_EVENTS || !events[idx]) return finish(AEC_ERROR_INVALID_HANDLE);
        ev = events[idx];
        events[idx] = nullptr;
    }

    // If the event has a pending record, wait for its stream to complete it.
    // IMPORTANT: must release ev->mtx before calling process_stream,
    // because process_stream locks ev->mtx when completing the marker.
    Stream *need_process = nullptr;
    {
        std::lock_guard<std::mutex> ev_lock(ev->mtx);
        if (ev->recorded && !ev->completed && ev->stream)
            need_process = ev->stream;
    }
    if (need_process) process_stream(need_process);

    delete ev;
    return AEC_SUCCESS;
}

aecError_t aecEventRecord(aecEvent_t event, aecStream_t stream) {
    if (!event || !stream) return finish(AEC_ERROR_INVALID_HANDLE);
    Event *ev = get_event(event);
    Stream *s = get_stream(stream);
    if (!ev || !s) return finish(AEC_ERROR_INVALID_HANDLE);

    // Create new generation
    uint64_t new_gen;
    {
        std::lock_guard<std::mutex> ev_lock(ev->mtx);
        ev->generation++;
        ev->recorded  = true;
        ev->completed = false;
        ev->error     = AEC_SUCCESS;
        ev->stream    = s;
        new_gen       = ev->generation;
    }

    // Enqueue a marker operation
    StreamOp sop{};
    sop.cmd.abi_version  = AEC_DEVICE_ABI_VERSION;
    sop.cmd.opcode       = AEC_DEVICE_OP_BARRIER;
    sop.cmd.flags        = AEC_DEVICE_FLAG_NONE;
    sop.cmd.sequence     = take_sequence();
    sop.cmd.stream_id    = reinterpret_cast<uint64_t>(stream);
    sop.is_event_record  = true;
    sop.event            = ev;
    sop.event_generation = new_gen;

    s->enqueue(std::move(sop));
    return AEC_SUCCESS;
}

aecError_t aecEventSynchronize(aecEvent_t event) {
    if (!event) return finish(AEC_ERROR_INVALID_HANDLE);
    Event *ev = get_event(event);
    if (!ev) return finish(AEC_ERROR_INVALID_HANDLE);

    {
        std::lock_guard<std::mutex> ev_lock(ev->mtx);
        if (!ev->recorded) return finish(AEC_ERROR_INVALID_ARGUMENT);
        if (ev->completed) return ev->error;
    }

    // Process the stream until the event completes
    Stream *s = nullptr;
    {
        std::lock_guard<std::mutex> ev_lock(ev->mtx);
        s = ev->stream;
    }
    if (s) process_stream(s);

    {
        std::lock_guard<std::mutex> ev_lock(ev->mtx);
        return ev->error;
    }
}

aecError_t aecEventQuery(aecEvent_t event) {
    if (!event) return finish(AEC_ERROR_INVALID_HANDLE);
    Event *ev = get_event(event);
    if (!ev) return finish(AEC_ERROR_INVALID_HANDLE);

    std::lock_guard<std::mutex> ev_lock(ev->mtx);
    if (!ev->recorded) return finish(AEC_ERROR_INVALID_ARGUMENT);
    if (!ev->completed) return finish(AEC_ERROR_NOT_READY);
    return ev->error;
}

aecError_t aecEventElapsedCycles(aecEvent_t start, aecEvent_t end, uint64_t *out_cycles) {
    if (!start || !end) return finish(AEC_ERROR_INVALID_HANDLE);
    if (!out_cycles) return finish(AEC_ERROR_INVALID_ARGUMENT);
    Event *es = get_event(start);
    Event *ee = get_event(end);
    if (!es || !ee) return finish(AEC_ERROR_INVALID_HANDLE);

    uint64_t sc, ec;
    {
        std::lock_guard<std::mutex> lock_s(es->mtx);
        if (!es->recorded || !es->completed) return finish(AEC_ERROR_INVALID_ARGUMENT);
        sc = es->cycle;
    }
    {
        std::lock_guard<std::mutex> lock_e(ee->mtx);
        if (!ee->recorded || !ee->completed) return finish(AEC_ERROR_INVALID_ARGUMENT);
        ec = ee->cycle;
    }

    if (ec < sc) return finish(AEC_ERROR_INVALID_ARGUMENT);
    *out_cycles = ec - sc;
    return AEC_SUCCESS;
}

// =====================================================================
// R104 + R201  Kernel Launch  (Vector Add + GEMM, with stream support)
// =====================================================================
aecError_t aecLaunch(aecKernelId kernel, aecDim3 grid, aecDim3 block,
                     const void *args, size_t args_size, aecStream_t stream) {
    // --- validation ---
    if (grid.x == 0 || grid.y == 0 || grid.z == 0) return finish(AEC_ERROR_INVALID_ARGUMENT);
    if (block.x == 0 || block.y == 0 || block.z == 0) return finish(AEC_ERROR_INVALID_ARGUMENT);
    if (static_cast<uint64_t>(block.x) * block.y * block.z > 1024)
        return finish(AEC_ERROR_INVALID_ARGUMENT);
    if (!args || args_size == 0) return finish(AEC_ERROR_INVALID_ARGUMENT);

    // --- map kernel ID → (semantic, dtype, variant) ---
    uint32_t semantic, dtype, variant;

    switch (kernel) {
    case AEC_KERNEL_VECTOR_ADD_F32:
        if (args_size != sizeof(aecVectorAddArgs))
            return finish(AEC_ERROR_INVALID_ARGUMENT);
        semantic = 1; dtype = AEC_DTYPE_FP32; variant = 0;
        break;

    case AEC_KERNEL_GEMM_NAIVE:
    case AEC_KERNEL_GEMM_TILED:
    case AEC_KERNEL_GEMM_VECTORIZED: {
        if (args_size != 40)
            return finish(AEC_ERROR_INVALID_ARGUMENT);
        const auto *ga = static_cast<const aecGemmArgs *>(args);
        if (ga->m < 1 || ga->m > 256 || ga->n < 1 || ga->n > 256 || ga->k < 1 || ga->k > 256)
            return finish(AEC_ERROR_INVALID_ARGUMENT);
        semantic = static_cast<uint32_t>(kernel);
        dtype    = ga->dtype;
        switch (kernel) {
        case AEC_KERNEL_GEMM_NAIVE:     variant = 1; break;
        case AEC_KERNEL_GEMM_TILED:     variant = 2; break;
        case AEC_KERNEL_GEMM_VECTORIZED: variant = 3; break;
        default: return unsupported();
        }
        break;
    }

    case AEC_KERNEL_AXPY_F32:
        // AXPY canonical block = 28 bytes (struct has reserved → 32)
        if (args_size != 28)
            return finish(AEC_ERROR_INVALID_ARGUMENT);
        semantic = static_cast<uint32_t>(kernel);  // 20
        dtype    = AEC_DTYPE_FP32;
        variant  = 0;
        break;
    case AEC_KERNEL_DOT_F32:
        if (args_size != sizeof(aecDotArgs))
            return finish(AEC_ERROR_INVALID_ARGUMENT);
        semantic = static_cast<uint32_t>(kernel);  // 21
        dtype    = AEC_DTYPE_FP32;
        variant  = 0;
        break;
    case AEC_KERNEL_NRM2_F32:
        if (args_size != sizeof(aecNrm2Args))
            return finish(AEC_ERROR_INVALID_ARGUMENT);
        semantic = static_cast<uint32_t>(kernel);  // 22
        dtype    = AEC_DTYPE_FP32;
        variant  = 0;
        break;

    default:
        return unsupported();
    }

    // --- resolve kernel image ---
    aecDeviceKernelInfo kinfo{};
    aecDeviceStatus st = aecDeviceResolveKernel(semantic, dtype, variant, &kinfo);
    if (st != AEC_DEVICE_SUCCESS)
        return finish(device_status_to_error(st));

    // --- build canonical parameter block ---
    uint8_t params[64] = {};
    if (kernel == AEC_KERNEL_VECTOR_ADD_F32) {
        const auto *va = static_cast<const aecVectorAddArgs *>(args);
        write_u64_le(params,  0, va->a);
        write_u64_le(params,  8, va->b);
        write_u64_le(params, 16, va->c);
        write_u64_le(params, 24, va->count);
    } else if (kernel == AEC_KERNEL_AXPY_F32) {
        // Canonical: X:u64 Y:u64 count:u64 alpha:f32bits = 28 bytes
        const auto *aa = static_cast<const aecAxpyArgs *>(args);
        write_u64_le(params,  0, aa->x);
        write_u64_le(params,  8, aa->y);
        write_u64_le(params, 16, aa->count);
        uint32_t alpha_bits;
        std::memcpy(&alpha_bits, &aa->alpha, sizeof(alpha_bits));
        write_u32_le(params, 24, alpha_bits);
    } else if (kernel == AEC_KERNEL_DOT_F32) {
        // Canonical: X:u64 Y:u64 result:u64 count:u64 = 32 bytes
        const auto *da = static_cast<const aecDotArgs *>(args);
        write_u64_le(params,  0, da->x);
        write_u64_le(params,  8, da->y);
        write_u64_le(params, 16, da->result);
        write_u64_le(params, 24, da->count);
    } else if (kernel == AEC_KERNEL_NRM2_F32) {
        // Canonical: X:u64 result:u64 count:u64 = 24 bytes
        const auto *na = static_cast<const aecNrm2Args *>(args);
        write_u64_le(params,  0, na->x);
        write_u64_le(params,  8, na->result);
        write_u64_le(params, 16, na->count);
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
    cmd.abi_version     = AEC_DEVICE_ABI_VERSION;
    cmd.opcode          = AEC_DEVICE_OP_ISA_LAUNCH;
    cmd.flags           = AEC_DEVICE_FLAG_NONE;
    cmd.sequence        = take_sequence();
    cmd.stream_id       = reinterpret_cast<uint64_t>(stream);  // 0 for null stream
    cmd.kernel_handle   = kinfo.handle;
    cmd.isa_version     = kinfo.isa_version;
    cmd.entry_pc        = kinfo.entry_pc;
    cmd.grid            = aecDeviceDim3{grid.x, grid.y, grid.z};
    cmd.block           = aecDeviceDim3{block.x, block.y, block.z};
    cmd.parameter_bytes = static_cast<uint32_t>(args_size);
    std::memcpy(cmd.parameters, params, args_size);

    // --- null stream → synchronous, non-null → enqueue ---
    if (stream == nullptr) {
        aecDeviceCompletion comp{};
        st = aecDeviceSubmit(&cmd, &comp);
        if (st != AEC_DEVICE_SUCCESS)
            return finish(device_status_to_error(st));
        if (comp.status != AEC_DEVICE_SUCCESS)
            return finish(device_status_to_error(static_cast<aecDeviceStatus>(comp.status)));
        return AEC_SUCCESS;
    }

    // Async path
    Stream *s = get_stream(stream);
    if (!s) return finish(AEC_ERROR_INVALID_HANDLE);

    StreamOp sop{};
    sop.cmd = cmd;              // copies everything including parameters
    sop.args_copied = true;     // args live in sop.cmd.parameters
    s->enqueue(std::move(sop));
    return AEC_SUCCESS;
}

// =====================================================================
// R201  GEMM helpers
// =====================================================================
static aecError_t gemm_launch(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                              uint32_t m, uint32_t n, uint32_t k,
                              aecDataType dtype, aecStream_t stream) {
    if (m < 1 || m > 256 || n < 1 || n > 256 || k < 1 || k > 256)
        return finish(AEC_ERROR_INVALID_ARGUMENT);

    aecGemmArgs args = {a, b, c, m, n, k, static_cast<uint32_t>(dtype), 0};
    aecDim3 grid  = {n, m, 1};
    aecDim3 block = {1, 1, 1};

    return aecLaunch(AEC_KERNEL_GEMM_NAIVE, grid, block,
                     &args, 40, stream);  // canonical param block = 40 bytes
}

aecError_t aecMatmulF32(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                        uint32_t m, uint32_t n, uint32_t k, aecStream_t stream) {
    return gemm_launch(a, b, c, m, n, k, AEC_DTYPE_FP32, stream);
}
aecError_t aecMatmulI32(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                        uint32_t m, uint32_t n, uint32_t k, aecStream_t stream) {
    return gemm_launch(a, b, c, m, n, k, AEC_DTYPE_INT32, stream);
}

// =====================================================================
// R202 / R203  Other GEMM dtypes (stubs)
// =====================================================================
aecError_t aecMatmulF4(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                       uint32_t m, uint32_t n, uint32_t k, aecStream_t s) {
    return gemm_launch(a, b, c, m, n, k, AEC_DTYPE_FP4_E2M1, s);
}
aecError_t aecMatmulF8(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                       uint32_t m, uint32_t n, uint32_t k,
                       aecFp8Format fmt, aecStream_t s) {
    aecDataType dt = (fmt == AEC_FP8_E5M2) ? AEC_DTYPE_FP8_E5M2 : AEC_DTYPE_FP8_E4M3;
    return gemm_launch(a, b, c, m, n, k, dt, s);
}
aecError_t aecMatmulF16(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                        uint32_t m, uint32_t n, uint32_t k, aecStream_t s) {
    return gemm_launch(a, b, c, m, n, k, AEC_DTYPE_FP16, s);
}
aecError_t aecMatmulBF16(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                         uint32_t m, uint32_t n, uint32_t k, aecStream_t s) {
    return gemm_launch(a, b, c, m, n, k, AEC_DTYPE_BF16, s);
}
aecError_t aecMatmulF64(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                        uint32_t m, uint32_t n, uint32_t k, aecStream_t s) {
    return gemm_launch(a, b, c, m, n, k, AEC_DTYPE_FP64, s);
}
aecError_t aecMatmulI4(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                       uint32_t m, uint32_t n, uint32_t k, aecStream_t s) {
    return gemm_launch(a, b, c, m, n, k, AEC_DTYPE_INT4, s);
}
aecError_t aecMatmulI8(aecDevicePtr a, aecDevicePtr b, aecDevicePtr c,
                       uint32_t m, uint32_t n, uint32_t k, aecStream_t s) {
    return gemm_launch(a, b, c, m, n, k, AEC_DTYPE_INT8, s);
}

// =====================================================================
// R204+  (stubs)
// =====================================================================
aecError_t aecHostRegister(void *ptr, size_t bytes) {
    if (!ptr || bytes == 0) return finish(AEC_ERROR_INVALID_ARGUMENT);
    uint64_t end = reinterpret_cast<uint64_t>(ptr) + bytes;
    if (end < reinterpret_cast<uint64_t>(ptr)) return finish(AEC_ERROR_INVALID_ARGUMENT);
    {
        std::lock_guard<std::mutex> lock(reg_mutex);
        // Exact duplicate → INVALID_ARGUMENT
        if (registered_ranges.find(ptr) != registered_ranges.end())
            return finish(AEC_ERROR_INVALID_ARGUMENT);
        // Overlap with different range → INVALID_ADDRESS
        for (const auto &kv : registered_ranges) {
            uint64_t base = reinterpret_cast<uint64_t>(kv.second.base);
            uint64_t bend = base + kv.second.bytes;
            if (!(end <= base || reinterpret_cast<uint64_t>(ptr) >= bend))
                return finish(AEC_ERROR_INVALID_ADDRESS);
        }
        registered_ranges[ptr] = {ptr, bytes};
    }
    return AEC_SUCCESS;
}
aecError_t aecHostUnregister(void *ptr) {
    if (!ptr) return finish(AEC_ERROR_INVALID_ARGUMENT);
    {
        std::lock_guard<std::mutex> lock(reg_mutex);
        auto it = registered_ranges.find(ptr);
        if (it == registered_ranges.end()) return finish(AEC_ERROR_INVALID_ARGUMENT);
        registered_ranges.erase(it);
    }
    // Wait for pending async work that may reference this range
    process_all_streams();
    return AEC_SUCCESS;
}

aecError_t aecGetRuntimeStats(aecRuntimeStats *stats) {
    if (!stats) return finish(AEC_ERROR_INVALID_ARGUMENT);
    aecDeviceStats ds{};
    if (aecDeviceGetStats(&ds) != AEC_DEVICE_SUCCESS) return finish(AEC_ERROR_DEVICE);
    static_assert(sizeof(*stats) == sizeof(ds));
    std::memcpy(stats, &ds, sizeof(*stats));
    stats->abi_version = AEC_RUNTIME_ABI_VERSION;
    return AEC_SUCCESS;
}
aecError_t aecResetRuntimeStats(void) {
    return aecDeviceResetStats() == AEC_DEVICE_SUCCESS ? AEC_SUCCESS : finish(AEC_ERROR_DEVICE);
}

aecError_t aecAxpy(aecDevicePtr x, aecDevicePtr y, uint64_t count,
                   float alpha, aecStream_t s) {
    if (count < 1 || count > 1048576) return finish(AEC_ERROR_INVALID_ARGUMENT);
    aecAxpyArgs args = {x, y, count, alpha, 0};
    uint32_t bx = (count < 1024) ? static_cast<uint32_t>(count) : 1024u;
    uint32_t gx = static_cast<uint32_t>((count + bx - 1) / bx);
    aecDim3 grid  = {gx, 1, 1};
    aecDim3 block = {bx, 1, 1};
    return aecLaunch(AEC_KERNEL_AXPY_F32, grid, block, &args, 28, s);
}
aecError_t aecDot(aecDevicePtr x, aecDevicePtr y, aecDevicePtr result,
                 uint64_t count, aecStream_t s) {
    if (count < 1 || count > 1048576) return finish(AEC_ERROR_INVALID_ARGUMENT);
    constexpr uint64_t MAX_CHUNK = 65536;
    if (count <= MAX_CHUNK) {
        aecDotArgs args = {x, y, result, count};
        uint32_t bx = (count < 256) ? static_cast<uint32_t>(count) : 256u;
        uint32_t gx = static_cast<uint32_t>((count + bx - 1) / bx);
        aecDim3 grid  = {gx, 1, 1};
        aecDim3 block = {bx, 1, 1};
        return aecLaunch(AEC_KERNEL_DOT_F32, grid, block, &args, sizeof(args), s);
    }
    // Large count: kernel has ~90K element limit; split into chunks
    float total = 0.0f;
    aecDevicePtr partial_buf;
    aecError_t rc = aecAlloc(&partial_buf, sizeof(float));
    if (rc != AEC_SUCCESS) return finish(rc);
    for (uint64_t off = 0; off < count; off += MAX_CHUNK) {
        uint64_t chunk = std::min(MAX_CHUNK, count - off);
        aecDotArgs args = {x + off * sizeof(float), y + off * sizeof(float),
                           partial_buf, chunk};
        uint32_t bx2 = (chunk < 256) ? static_cast<uint32_t>(chunk) : 256u;
        uint32_t gx2 = static_cast<uint32_t>((chunk + bx2 - 1) / bx2);
        aecDim3 grid2  = {gx2, 1, 1};
        aecDim3 block2 = {bx2, 1, 1};
        rc = aecLaunch(AEC_KERNEL_DOT_F32, grid2, block2, &args, sizeof(args), nullptr);
        if (rc != AEC_SUCCESS) { aecFree(partial_buf); return finish(rc); }
        float partial;
        rc = aecCopyD2H(&partial, partial_buf, sizeof(float));
        if (rc != AEC_SUCCESS) { aecFree(partial_buf); return finish(rc); }
        total += partial;
    }
    aecFree(partial_buf);
    return aecCopyH2D(result, &total, sizeof(float));
}
aecError_t aecNrm2(aecDevicePtr x, aecDevicePtr result, uint64_t count,
                   aecStream_t s) {
    if (count < 1 || count > 1048576) return finish(AEC_ERROR_INVALID_ARGUMENT);
    constexpr uint64_t MAX_CHUNK = 65536;
    if (count <= MAX_CHUNK) {
        aecNrm2Args args = {x, result, count};
        uint32_t bx = (count < 256) ? static_cast<uint32_t>(count) : 256u;
        uint32_t gx = static_cast<uint32_t>((count + bx - 1) / bx);
        aecDim3 grid  = {gx, 1, 1};
        aecDim3 block = {bx, 1, 1};
        return aecLaunch(AEC_KERNEL_NRM2_F32, grid, block, &args, sizeof(args), s);
    }
    // Large count: kernel has ~90K element limit; split into chunks
    double sum_sq = 0.0;
    aecDevicePtr partial_buf;
    aecError_t rc = aecAlloc(&partial_buf, sizeof(float));
    if (rc != AEC_SUCCESS) return finish(rc);
    for (uint64_t off = 0; off < count; off += MAX_CHUNK) {
        uint64_t chunk = std::min(MAX_CHUNK, count - off);
        aecNrm2Args args = {x + off * sizeof(float), partial_buf, chunk};
        uint32_t bx2 = (chunk < 256) ? static_cast<uint32_t>(chunk) : 256u;
        uint32_t gx2 = static_cast<uint32_t>((chunk + bx2 - 1) / bx2);
        aecDim3 grid2  = {gx2, 1, 1};
        aecDim3 block2 = {bx2, 1, 1};
        rc = aecLaunch(AEC_KERNEL_NRM2_F32, grid2, block2, &args, sizeof(args), nullptr);
        if (rc != AEC_SUCCESS) { aecFree(partial_buf); return finish(rc); }
        float partial;
        rc = aecCopyD2H(&partial, partial_buf, sizeof(float));
        if (rc != AEC_SUCCESS) { aecFree(partial_buf); return finish(rc); }
        sum_sq += static_cast<double>(partial) * static_cast<double>(partial);
    }
    aecFree(partial_buf);
    float result_val = static_cast<float>(std::sqrt(sum_sq));
    return aecCopyH2D(result, &result_val, sizeof(float));
}

} // extern "C"
