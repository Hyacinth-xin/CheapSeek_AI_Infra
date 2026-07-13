#!/usr/bin/env python3
"""CheapSeek R303 probe — host registration, zero-copy, lifecycle.

Directions covered (CLAUDE.md 十一 + 十二.4):
  - Register/unregister lifecycle error codes
  - Duplicate exact → INV_ARG; overlap diff base → INV_ADDR
  - Adjacent touching boundary OK (no overlap)
  - Unregister interior pointer → INV_ARG
  - Zero-copy only when transfer fully inside a registered interval
  - Unregister → zc stops; re-register → zc resumes
  - Zero-copy cycles measurably lower than non-zc

Run: python3 CheapSeek_test/CheapSeek_r303.py
"""
import ctypes as ct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
dev = ct.CDLL(str(ROOT / "lib" / "libaec_device.so"), mode=ct.RTLD_GLOBAL)
lib = ct.CDLL(str(ROOT / "libaec.so"))

u32 = ct.c_uint32; u64 = ct.c_uint64; sz = ct.c_size_t
cint = ct.c_int; vp = ct.c_void_p

SUCCESS = 0; INV_ARG = 1; INV_ADDR = 4
H2D = 1

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

# ── API ────────────────────────────────────────────────────────────────
lib.aecAlloc.argtypes = [ct.POINTER(u64), sz]; lib.aecAlloc.restype = cint
lib.aecFree.argtypes = [u64]; lib.aecFree.restype = cint
lib.aecCopyH2D.argtypes = [u64, vp, sz]; lib.aecCopyH2D.restype = cint
lib.aecCopyAsync.argtypes = [u64, vp, sz, cint, vp]; lib.aecCopyAsync.restype = cint
lib.aecHostRegister.argtypes = [vp, sz]; lib.aecHostRegister.restype = cint
lib.aecHostUnregister.argtypes = [vp]; lib.aecHostUnregister.restype = cint
lib.aecStreamCreate.argtypes = [ct.POINTER(vp)]; lib.aecStreamCreate.restype = cint
lib.aecStreamDestroy.argtypes = [vp]; lib.aecStreamDestroy.restype = cint
lib.aecStreamSync.argtypes = [vp]; lib.aecStreamSync.restype = cint
lib.aecGetRuntimeStats.argtypes = [ct.POINTER(aecRuntimeStats)]; lib.aecGetRuntimeStats.restype = cint
lib.aecResetRuntimeStats.argtypes = []; lib.aecResetRuntimeStats.restype = cint
lib.aecGetErrorName.argtypes = [cint]; lib.aecGetErrorName.restype = ct.c_char_p
lib.aecGetLastError.argtypes = []; lib.aecGetLastError.restype = cint
dev.aecDeviceReset.restype = cint


def ename(c): return lib.aecGetErrorName(c).decode()

def reset():
    dev.aecDeviceReset(); lib.aecGetLastError(); lib.aecResetRuntimeStats()

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

print("=== CheapSeek R303 probe ===")
print()

# ── C1: Register non-empty valid interval ──────────────────────────────
reset()
buf = (ct.c_uint8 * 4096)()
r = lib.aecHostRegister(buf, 4096)
lib.aecHostUnregister(buf)
check("register non-empty interval → SUCCESS", r == SUCCESS,
      f"got={ename(r)}")

# ── C2: Duplicate exact register → INV_ARG (not INV_ADDR) ─────────────
reset()
buf2 = (ct.c_uint8 * 2048)()
lib.aecHostRegister(buf2, 2048)
r_dup = lib.aecHostRegister(buf2, 2048)
lib.aecHostUnregister(buf2)
check("duplicate exact register → INVALID_ARGUMENT (not INVALID_ADDRESS)",
      r_dup == INV_ARG, f"got={ename(r_dup)}")

# ── C3: Overlapping different base → INVALID_ADDRESS ──────────────────
reset()
bufA = (ct.c_uint8 * 4096)()
bufB = ct.cast(ct.byref(bufA, 1024), vp)  # bufB = bufA + 1024, still inside bufA's range
lib.aecHostRegister(bufA, 4096)
r_overlap = lib.aecHostRegister(bufB, 2048)
lib.aecHostUnregister(bufA)
check("overlap different base → INVALID_ADDRESS",
      r_overlap == INV_ADDR, f"got={ename(r_overlap)}")

# ── C4: Adjacent boundary-touching → SUCCESS (no overlap) ─────────────
reset()
bufC = (ct.c_uint8 * 2048)()
bufD = ct.cast(ct.byref(bufC, 2048), vp)  # bufD starts exactly at bufC's end
lib.aecHostRegister(bufC, 2048)
r_adj = lib.aecHostRegister(bufD, 2048)  # adjacent, non-overlapping
lib.aecHostUnregister(bufC); lib.aecHostUnregister(bufD)
check("adjacent boundary-touching → SUCCESS", r_adj == SUCCESS,
      f"got={ename(r_adj)}")

# ── C5: Unregister exact pointer → SUCCESS ────────────────────────────
reset()
buf5 = (ct.c_uint8 * 1024)()
lib.aecHostRegister(buf5, 1024)
r_unreg = lib.aecHostUnregister(buf5)
check("unregister exact registered pointer → SUCCESS", r_unreg == SUCCESS,
      f"got={ename(r_unreg)}")

# ── C6: Unregister interior pointer → INV_ARG ─────────────────────────
reset()
buf6 = (ct.c_uint8 * 2048)()
lib.aecHostRegister(buf6, 2048)
interior = ct.cast(ct.byref(buf6, 512), vp)
r_int = lib.aecHostUnregister(interior)
lib.aecHostUnregister(buf6)
check("unregister interior pointer → INVALID_ARGUMENT",
      r_int == INV_ARG, f"got={ename(r_int)}")

# ── C7: Unregister non-registered → INV_ARG ───────────────────────────
reset()
buf7 = (ct.c_uint8 * 1024)()
r_noreg = lib.aecHostUnregister(buf7)
check("unregister non-registered pointer → INVALID_ARGUMENT",
      r_noreg == INV_ARG, f"got={ename(r_noreg)}")

# ── C8: Duplicate unregister → INV_ARG ────────────────────────────────
reset()
buf8 = (ct.c_uint8 * 512)()
lib.aecHostRegister(buf8, 512)
lib.aecHostUnregister(buf8)
r_dup_unreg = lib.aecHostUnregister(buf8)
check("duplicate unregister → INVALID_ARGUMENT",
      r_dup_unreg == INV_ARG, f"got={ename(r_dup_unreg)}")

# ── C9: Full-range transfer → zero-copy ───────────────────────────────
reset()
d9 = alloc(4096)
buf9 = (ct.c_uint8 * 4096)()
lib.aecHostRegister(buf9, 4096)
lib.aecCopyH2D(d9, buf9, 4096)
stats9 = get_stats()
lib.aecHostUnregister(buf9); free(d9)
check("full-range registered transfer → zero_copy_commands==1",
      stats9.zero_copy_commands == 1,
      f"zc_cmds={stats9.zero_copy_commands}")

# ── C10: Partial-range → NOT zero-copy ────────────────────────────────
reset()
d10 = alloc(4096)
buf10 = (ct.c_uint8 * 4096)()
lib.aecHostRegister(buf10, 2048)  # register only first 2K of the 4K buffer
lib.aecCopyH2D(d10, buf10, 4096)   # transfer 4K — 2K not in registered range
stats10 = get_stats()
lib.aecHostUnregister(buf10); free(d10)
check("partial-range transfer (2K reg, 4K DMA) → NOT zero-copy",
      stats10.zero_copy_commands == 0,
      f"zc_cmds={stats10.zero_copy_commands}")

# ── C11: Unregister → zc stops, re-register → zc resumes ──────────────
reset()
d11 = alloc(4096)
buf11 = (ct.c_uint8 * 4096)()

lib.aecHostRegister(buf11, 4096)
lib.aecCopyH2D(d11, buf11, 4096)
s1 = get_stats()

lib.aecHostUnregister(buf11)
lib.aecCopyH2D(d11, buf11, 4096)
s2 = get_stats()

lib.aecHostRegister(buf11, 4096)
lib.aecCopyH2D(d11, buf11, 4096)
s3 = get_stats()

lib.aecHostUnregister(buf11); free(d11)
check("unregister→zc stops (0), re-register→zc resumes (1 more)",
      s1.zero_copy_commands == 1 and s2.zero_copy_commands == 1
      and s3.zero_copy_commands == 2,
      f"zc1={s1.zero_copy_commands} zc2={s2.zero_copy_commands} zc3={s3.zero_copy_commands}")

# ── C12: Zero-copy cycles measurably lower than non-zc ────────────────
reset()
d12 = alloc(65536)
buf12 = (ct.c_uint8 * 65536)()

# Non-zero-copy first
lib.aecCopyH2D(d12, buf12, 65536)
s_nzc = get_stats()
cyc_nzc = s_nzc.last_virtual_cycles

# Zero-copy
lib.aecHostRegister(buf12, 65536)
lib.aecCopyH2D(d12, buf12, 65536)
s_zc = get_stats()
cyc_zc = s_zc.last_virtual_cycles

lib.aecHostUnregister(buf12); free(d12)
check("zero-copy cycles measurably lower than non-zc",
      cyc_zc > 0 and cyc_zc < cyc_nzc,
      f"nzc={cyc_nzc} zc={cyc_zc} (saving={cyc_nzc - cyc_zc})")

# ── C13: Async zero-copy on stream ────────────────────────────────────
reset()
d13 = alloc(4096)
buf13 = (ct.c_uint8 * 4096)()
st13 = mkstream()

lib.aecHostRegister(buf13, 4096)
r_async = lib.aecCopyAsync(d13, buf13, 4096, H2D, st13)
r_sync = lib.aecStreamSync(st13)
stats13 = get_stats()

lib.aecStreamDestroy(st13)
lib.aecHostUnregister(buf13); free(d13)
check("async zero-copy on stream → completed, zc counted",
      r_async == SUCCESS and r_sync == SUCCESS and stats13.zero_copy_commands == 1,
      f"async={ename(r_async)} sync={ename(r_sync)} zc={stats13.zero_copy_commands}")

# ── C14: Register 0 bytes or NULL → error ─────────────────────────────
reset()
r_null = lib.aecHostRegister(None, 1024)
r_zero = lib.aecHostRegister((ct.c_uint8 * 64)(), 0)
check("register NULL or zero bytes → INVALID_ARGUMENT",
      r_null == INV_ARG and r_zero == INV_ARG,
      f"NULL={ename(r_null)} zero_bytes={ename(r_zero)}")

print()
print(f"=== {sum(results)}/{len(results)} checks passed ===")
# ── C15: aecFree drains pending async DMA before freeing (registered) ──
reset()
d15 = alloc(65536)
buf15 = (ct.c_uint8 * 65536)()
st15 = mkstream()

lib.aecHostRegister(buf15, 65536)
# Enqueue async H2D using registered buffer — still pending
lib.aecCopyAsync(d15, buf15, 65536, H2D, st15)
# Free device memory immediately — must drain the async DMA first
r_free = free(d15)
r_sync = lib.aecStreamSync(st15)  # should have completed during aecFree drain

lib.aecStreamDestroy(st15)
lib.aecHostUnregister(buf15)
check("aecFree drains pending async DMA (registered, zc) before freeing",
      r_free == SUCCESS and r_sync == SUCCESS,
      f"free={ename(r_free)} sync={ename(r_sync)}")

# ── C16: D2H zero-copy transfer ───────────────────────────────────────
D2H = 2
reset()
d16 = alloc(4096)
buf16 = (ct.c_uint8 * 4096)()
# Pre-fill device memory with known pattern
pattern = (ct.c_uint8 * 4096)(*([i % 256 for i in range(4096)]))  # cycling 0-255
lib.aecCopyH2D(d16, pattern, 4096)

lib.aecHostRegister(buf16, 4096)
lib.aecCopyD2H(buf16, d16, 4096)
stats16 = get_stats()
lib.aecHostUnregister(buf16); free(d16)

# Verify data integrity
ok_data = all(buf16[i] == (i % 256) for i in range(4096))
check("D2H zero-copy: zc counted, data intact",
      stats16.zero_copy_commands == 1 and ok_data,
      f"zc={stats16.zero_copy_commands} data_ok={ok_data}")

print()
print(f"=== {sum(results)}/{len(results)} checks passed ===")
