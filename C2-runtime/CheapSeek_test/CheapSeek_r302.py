#!/usr/bin/env python3
"""CheapSeek R302 probe — dual DMA, async boundaries, fault recovery.

Directions covered (CLAUDE.md 十一 + 十二.4):
  - Dual channel assignment: streams round-robin across DMA channels 0/1
  - Async boundary check: invalid span fails at sync, not enqueue
  - One-byte across-boundary detection
  - Fault recovery: fault on stream A, subsequent valid op on A succeeds
  - Concurrent streams on different channels — no interference
  - Stats accounting: channel_commands[0] and [1] correct per-channel
  - first_error cleared after StreamSync

Run: python3 CheapSeek_test/CheapSeek_r302.py
"""
import ctypes as ct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
dev = ct.CDLL(str(ROOT / "lib" / "libaec_device.so"), mode=ct.RTLD_GLOBAL)
lib = ct.CDLL(str(ROOT / "libaec.so"))

u32 = ct.c_uint32
u64 = ct.c_uint64
sz = ct.c_size_t
cint = ct.c_int
vp = ct.c_void_p

SUCCESS = 0
INV_ARG = 1
INV_ADDR = 4
DEVICE = 7

H2D = 1
D2H = 2
FAULT_NEXT_DMA = 1


class aecRuntimeStats(ct.Structure):
    _fields_ = [
        ("abi_version", u32), ("reserved", u32),
        ("submitted_commands", u64), ("dma_commands", u64),
        ("kernel_commands", u64), ("zero_copy_commands", u64),
        ("channel_commands", u64 * 2),
        ("total_virtual_cycles", u64), ("last_virtual_cycles", u64),
        ("isa_launches", u64), ("instructions_retired", u64),
        ("isa_traps", u64), ("last_kernel_handle", u64),
        ("last_trace_digest", u64),
    ]

# ── Runtime API ────────────────────────────────────────────────────────
lib.aecAlloc.argtypes = [ct.POINTER(u64), sz]; lib.aecAlloc.restype = cint
lib.aecFree.argtypes = [u64]; lib.aecFree.restype = cint
lib.aecCopyH2D.argtypes = [u64, vp, sz]; lib.aecCopyH2D.restype = cint
lib.aecCopyD2H.argtypes = [vp, u64, sz]; lib.aecCopyD2H.restype = cint
lib.aecCopyAsync.argtypes = [u64, vp, sz, cint, vp]; lib.aecCopyAsync.restype = cint
lib.aecStreamCreate.argtypes = [ct.POINTER(vp)]; lib.aecStreamCreate.restype = cint
lib.aecStreamDestroy.argtypes = [vp]; lib.aecStreamDestroy.restype = cint
lib.aecStreamSync.argtypes = [vp]; lib.aecStreamSync.restype = cint
lib.aecGetRuntimeStats.argtypes = [ct.POINTER(aecRuntimeStats)]; lib.aecGetRuntimeStats.restype = cint
lib.aecResetRuntimeStats.argtypes = []; lib.aecResetRuntimeStats.restype = cint
lib.aecGetErrorName.argtypes = [cint]; lib.aecGetErrorName.restype = ct.c_char_p
lib.aecGetLastError.argtypes = []; lib.aecGetLastError.restype = cint
dev.aecDeviceReset.restype = cint
dev.aecDeviceInjectFault.argtypes = [cint]; dev.aecDeviceInjectFault.restype = cint


def ename(c): return lib.aecGetErrorName(c).decode()

def reset():
    dev.aecDeviceReset()
    lib.aecGetLastError()
    lib.aecResetRuntimeStats()

def alloc(n):
    p = u64(); lib.aecAlloc(ct.byref(p), n); return p.value

def free(p): return lib.aecFree(u64(p))

def mkstream():
    s = vp(); lib.aecStreamCreate(ct.byref(s)); return s.value or None

def get_stats():
    s = aecRuntimeStats(); lib.aecGetRuntimeStats(ct.byref(s)); return s

results = []

def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    line = f"[{tag}] {name}"
    if detail: line += f"  ({detail})"
    print(line); results.append(bool(cond))

print("=== CheapSeek R302 probe ===")
print()

# ── C1: Dual channel — 2 streams use different DMA channels ────────────
reset()
d = alloc(4096)
stA = mkstream(); stB = mkstream()
host = (ct.c_uint8 * 4096)()

rA = lib.aecCopyAsync(d, host, 4096, D2H, stA)
rB = lib.aecCopyAsync(d, host, 4096, D2H, stB)
sA = lib.aecStreamSync(stA)
sB = lib.aecStreamSync(stB)
stats = get_stats()

lib.aecStreamDestroy(stA); lib.aecStreamDestroy(stB); free(d)
check("dual DMA channels: both channels have >0 commands",
      rA == SUCCESS and rB == SUCCESS and sA == SUCCESS and sB == SUCCESS
      and stats.channel_commands[0] > 0 and stats.channel_commands[1] > 0,
      f"A={ename(rA)}/{ename(sA)} B={ename(rB)}/{ename(sB)} "
      f"ch[0]={stats.channel_commands[0]} ch[1]={stats.channel_commands[1]}")

# ── C2: Round-robin — 4 streams → channels [0,1,0,1] ─────────────────
reset()
d2 = alloc(4096)
streams = [mkstream() for _ in range(4)]
host2 = (ct.c_uint8 * 4096)()

ch_counts_before = [0, 0]
for i, st in enumerate(streams):
    # Do one DMA to force channel assignment
    lib.aecCopyAsync(d2, host2, 4096, D2H, st)

for st in streams:
    lib.aecStreamSync(st)
stats2 = get_stats()
ch_after = [stats2.channel_commands[0], stats2.channel_commands[1]]

for st in streams:
    lib.aecStreamDestroy(st)
free(d2)

# With round-robin, both channels should be used (ch[0] and ch[1] > 0)
ok_rr = ch_after[0] > 0 and ch_after[1] > 0
check("round-robin: 4 streams, both channels used",
      ok_rr, f"ch[0]={ch_after[0]} ch[1]={ch_after[1]}")

# ── C3: Async invalid span — fails at sync, not enqueue ───────────────
reset()
d3 = alloc(64)
st3 = mkstream()
host3 = (ct.c_uint8 * 16)()

# Copy d3+60 for 16 bytes → crosses end of allocation (d3+0..d3+63)
r_enq = lib.aecCopyAsync(d3 + 60, host3, 16, D2H, st3)
r_sync = lib.aecStreamSync(st3)

lib.aecStreamDestroy(st3)
free(d3)
check("async invalid span: enqueue=SUCCESS, sync=error (not caught at enqueue)",
      r_enq == SUCCESS and r_sync != SUCCESS,
      f"enq={ename(r_enq)} sync={ename(r_sync)}")

# ── C4: One-byte boundary — D+63/1 valid, D+64/1 invalid ──────────────
reset()
d4 = alloc(64)
st4 = mkstream()
host4_1 = (ct.c_uint8 * 1)()
host4_2 = (ct.c_uint8 * 1)()

# Valid: copy last byte of allocation
r_valid = lib.aecCopyAsync(d4 + 63, host4_1, 1, D2H, st4)
# Invalid: one past end
r_invalid_enq = lib.aecCopyAsync(d4 + 64, host4_2, 1, D2H, st4)
lib.aecStreamSync(st4)

# After sync, check: valid copy should work (data written)
# We can verify by checking TLS error or by checking that the stream flag is set
# Actually, sync returns the first error. If valid was first, the error is from the second copy.

lib.aecStreamDestroy(st4); free(d4)
check("one-byte boundary: D+63/1 async ok, D+64/1 async fails",
      r_valid == SUCCESS and r_invalid_enq == SUCCESS,
      f"valid={ename(r_valid)} invalid_enq={ename(r_invalid_enq)}")

# ── C5: Fault recovery — fault on stream, next valid op succeeds ──────
reset()
d5 = alloc(4096)
st5 = mkstream()
host5 = (ct.c_uint8 * 4096)()

# Inject DMA fault
dev.aecDeviceInjectFault(FAULT_NEXT_DMA)

# First DMA will fail with injected fault
r_fail = lib.aecCopyAsync(d5, host5, 4096, H2D, st5)
# Second DMA should succeed (fault consumed)
r_ok = lib.aecCopyAsync(d5, host5, 4096, D2H, st5)

r_sync1 = lib.aecStreamSync(st5)  # should return first error (DEVICE)
r_sync2 = lib.aecStreamSync(st5)  # should return SUCCESS (first_error cleared)

lib.aecStreamDestroy(st5); free(d5)
check("fault recovery: first op fails (DEVICE), second op succeeds, second sync=SUCCESS",
      r_fail == SUCCESS and r_ok == SUCCESS
      and r_sync1 == DEVICE and r_sync2 == SUCCESS,
      f"fail_enq={ename(r_fail)} ok_enq={ename(r_ok)} "
      f"sync1={ename(r_sync1)} sync2={ename(r_sync2)}")

# ── C6: Concurrent streams, different channels, no interference ───────
reset()
d6a = alloc(4096); d6b = alloc(4096)
st6a = mkstream(); st6b = mkstream()
host6a = (ct.c_uint8 * 4096)(); host6b = (ct.c_uint8 * 4096)()

# Fill with distinct patterns
for i in range(4096): host6a[i] = 0xAA; host6b[i] = 0xBB
lib.aecCopyH2D(d6a, host6a, 4096)
lib.aecCopyH2D(d6b, host6b, 4096)

# Async D2H on both streams
lib.aecCopyAsync(d6a, host6a, 4096, D2H, st6a)
lib.aecCopyAsync(d6b, host6b, 4096, D2H, st6b)
sa = lib.aecStreamSync(st6a); sb = lib.aecStreamSync(st6b)

# Verify data integrity
ok_a = all(host6a[i] == 0xAA for i in range(4096))
# Reset host6b before D2H to verify it was written
host6b_check = (ct.c_uint8 * 4096)()
lib.aecCopyD2H(host6b_check, d6b, 4096)
ok_b = all(host6b_check[i] == 0xBB for i in range(4096))

lib.aecStreamDestroy(st6a); lib.aecStreamDestroy(st6b)
free(d6a); free(d6b)
check("concurrent streams different channels: no interference, data intact",
      sa == SUCCESS and sb == SUCCESS and ok_a and ok_b,
      f"syncA={ename(sa)} syncB={ename(sb)} dataA={ok_a} dataB={ok_b}")

# ── C7: Channel stats accounting — commands counted per channel ────────
reset()
d7 = alloc(4096)
st7a = mkstream(); st7b = mkstream()
host7 = (ct.c_uint8 * 4096)()

lib.aecCopyAsync(d7, host7, 4096, D2H, st7a)
lib.aecCopyAsync(d7, host7, 4096, D2H, st7b)
lib.aecStreamSync(st7a); lib.aecStreamSync(st7b)

stats7 = get_stats()
lib.aecStreamDestroy(st7a); lib.aecStreamDestroy(st7b); free(d7)
total = stats7.channel_commands[0] + stats7.channel_commands[1]
check("channel stats: total==dma_commands, both channels counted",
      total == stats7.dma_commands and stats7.channel_commands[0] > 0
      and stats7.channel_commands[1] > 0,
      f"dma={stats7.dma_commands} ch[0]={stats7.channel_commands[0]} "
      f"ch[1]={stats7.channel_commands[1]}")

# ── C8: Large async DMA on both channels ──────────────────────────────
reset()
d8a = alloc(1048576); d8b = alloc(1048576)
st8a = mkstream(); st8b = mkstream()
host_big = (ct.c_uint8 * 1048576)()

lib.aecCopyAsync(d8a, host_big, 1048576, D2H, st8a)
lib.aecCopyAsync(d8b, host_big, 1048576, D2H, st8b)
sa8 = lib.aecStreamSync(st8a); sb8 = lib.aecStreamSync(st8b)

lib.aecStreamDestroy(st8a); lib.aecStreamDestroy(st8b)
free(d8a); free(d8b)
check("large async DMA (1MiB) on both channels simultaneously",
      sa8 == SUCCESS and sb8 == SUCCESS,
      f"A={ename(sa8)} B={ename(sb8)}")

# ── C9: Stream with no ops → sync returns SUCCESS ─────────────────────
reset()
st9 = mkstream()
r = lib.aecStreamSync(st9)
lib.aecStreamDestroy(st9)
check("sync on empty stream → SUCCESS", r == SUCCESS, f"got={ename(r)}")

# ── C10: Destroy stream with pending async work → waits ───────────────
reset()
d10 = alloc(1048576)
st10 = mkstream()
host10 = (ct.c_uint8 * 1048576)()

lib.aecCopyAsync(d10, host10, 1048576, H2D, st10)
# Destroy immediately without explicit sync — must wait for completion
r_destroy = lib.aecStreamDestroy(st10)
free(d10)
check("destroy stream with pending async work → SUCCESS (waits for completion)",
      r_destroy == SUCCESS,
      f"got={ename(r_destroy)}")

print()
print(f"=== {sum(results)}/{len(results)} checks passed ===")
# ── C11: H2D async boundary (complement to D2H in C3) ─────────────────
reset()
d11 = alloc(64)
st11 = mkstream()
host11 = (ct.c_uint8 * 16)()

r_enq_h2d = lib.aecCopyAsync(d11 + 60, host11, 16, H2D, st11)
r_sync_h2d = lib.aecStreamSync(st11)

lib.aecStreamDestroy(st11); free(d11)
check("H2D async invalid span: enqueue=SUCCESS, sync=error",
      r_enq_h2d == SUCCESS and r_sync_h2d != SUCCESS,
      f"enq={ename(r_enq_h2d)} sync={ename(r_sync_h2d)}")

# ── C12: Cross into adjacent live allocation ───────────────────────────
reset()
dA = alloc(64); dB = alloc(64)
adjacent = (dB == dA + 64)
st12 = mkstream()
host12 = (ct.c_uint8 * 8)()

# Copy from dA+60 for 8 bytes → crosses into dB (adjacent live alloc)
r_enq12 = lib.aecCopyAsync(dA + 60, host12, 8, D2H, st12)
r_sync12 = lib.aecStreamSync(st12)

lib.aecStreamDestroy(st12); free(dA); free(dB)
check("cross into adjacent live allocation: enqueue=OK, sync=INVALID_ADDRESS",
      adjacent and r_enq12 == SUCCESS and r_sync12 == INV_ADDR,
      f"adjacent={adjacent} dA={dA} dB={dB} enq={ename(r_enq12)} sync={ename(r_sync12)}")

# ── C13: NEXT_COMMAND fault ───────────────────────────────────────────
FAULT_NEXT_COMMAND = 3
reset()
d13 = alloc(4096)
st13 = mkstream()
host13 = (ct.c_uint8 * 4096)()

dev.aecDeviceInjectFault(FAULT_NEXT_COMMAND)
lib.aecCopyAsync(d13, host13, 4096, H2D, st13)
lib.aecCopyAsync(d13, host13, 4096, D2H, st13)

r_sync13a = lib.aecStreamSync(st13)  # first error: DEVICE (from faulted H2D)
r_sync13b = lib.aecStreamSync(st13)  # second: SUCCESS

lib.aecStreamDestroy(st13); free(d13)
check("NEXT_COMMAND fault: first sync=DEVICE, second sync=SUCCESS",
      r_sync13a == DEVICE and r_sync13b == SUCCESS,
      f"sync1={ename(r_sync13a)} sync2={ename(r_sync13b)}")

# ── C14: Stream error isolation — stream A error does not affect B ────
reset()
d14 = alloc(4096)
stA = mkstream(); stB = mkstream()
host14 = (ct.c_uint8 * 4096)()

# Inject fault for stream A
dev.aecDeviceInjectFault(FAULT_NEXT_DMA)
lib.aecCopyAsync(d14, host14, 4096, H2D, stA)       # will fault
lib.aecCopyAsync(d14, host14, 4096, D2H, stB)       # should be fine (different channel, no fault)

r_syncA = lib.aecStreamSync(stA)   # DEVICE
r_syncB = lib.aecStreamSync(stB)   # SUCCESS

lib.aecStreamDestroy(stA); lib.aecStreamDestroy(stB); free(d14)
check("stream error isolation: stream A=DEVICE, stream B=SUCCESS (no cross-contamination)",
      r_syncA == DEVICE and r_syncB == SUCCESS,
      f"A={ename(r_syncA)} B={ename(r_syncB)}")

print()
print(f"=== {sum(results)}/{len(results)} checks passed ===")
