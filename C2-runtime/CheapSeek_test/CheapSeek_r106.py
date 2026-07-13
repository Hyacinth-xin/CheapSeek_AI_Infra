#!/usr/bin/env python3
"""CheapSeek R106 probe — Event generation, cycles, async error hidden-test directions.

Directions covered (CLAUDE.md 十一 + 十二.4):
  - Unrecorded Event query/sync -> INVALID_ARGUMENT (error-code distinction)
  - Query without drain after record -> NOT_READY (query does not drain Stream)
  - Same Event recorded on different Streams (Stream switch, generation tracking)
  - EventDestroy with pending async record — must wait for latest generation
  - ElapsedCycles: normal, end < start, NULL output, unrecorded/incomplete event
  - Async error attribution — StreamSync returns FIRST error, not last
  - Handle slot reuse after destroy — fresh state, no stale generation leak

Run: python3 test/CheapSeek_r106.py
"""
import ctypes as ct
import threading
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
OOM = 2
INV_HANDLE = 3
INV_ADDR = 4
NOT_READY = 5
NOT_SUPPORTED = 6
DEVICE = 7
INTERNAL = 8
ISA_TRAP = 9

H2D = 1
D2H = 2

FAULT_NEXT_DMA = 1


# ── Runtime API signatures ──────────────────────────────────────────────
lib.aecAlloc.argtypes = [ct.POINTER(u64), sz]
lib.aecAlloc.restype = cint
lib.aecFree.argtypes = [u64]
lib.aecFree.restype = cint
lib.aecCopyH2D.argtypes = [u64, vp, sz]
lib.aecCopyH2D.restype = cint
lib.aecCopyD2H.argtypes = [vp, u64, sz]
lib.aecCopyD2H.restype = cint
lib.aecCopyAsync.argtypes = [u64, vp, sz, cint, vp]
lib.aecCopyAsync.restype = cint
lib.aecStreamCreate.argtypes = [ct.POINTER(vp)]
lib.aecStreamCreate.restype = cint
lib.aecStreamDestroy.argtypes = [vp]
lib.aecStreamDestroy.restype = cint
lib.aecStreamSync.argtypes = [vp]
lib.aecStreamSync.restype = cint
lib.aecEventCreate.argtypes = [ct.POINTER(vp)]
lib.aecEventCreate.restype = cint
lib.aecEventDestroy.argtypes = [vp]
lib.aecEventDestroy.restype = cint
lib.aecEventRecord.argtypes = [vp, vp]
lib.aecEventRecord.restype = cint
lib.aecEventSynchronize.argtypes = [vp]
lib.aecEventSynchronize.restype = cint
lib.aecEventQuery.argtypes = [vp]
lib.aecEventQuery.restype = cint
lib.aecEventElapsedCycles.argtypes = [vp, vp, ct.POINTER(u64)]
lib.aecEventElapsedCycles.restype = cint
lib.aecGetErrorName.argtypes = [cint]
lib.aecGetErrorName.restype = ct.c_char_p
lib.aecGetLastError.argtypes = []
lib.aecGetLastError.restype = cint

# ── Device ABI (for fault injection) ────────────────────────────────────
dev.aecDeviceReset.restype = cint
dev.aecDeviceInjectFault.argtypes = [cint]
dev.aecDeviceInjectFault.restype = cint


def ename(c):
    return lib.aecGetErrorName(c).decode()


def reset():
    dev.aecDeviceReset()
    lib.aecGetLastError()


def alloc(n):
    p = u64()
    rc = lib.aecAlloc(ct.byref(p), n)
    return rc, p.value


def free(p):
    return lib.aecFree(u64(p))


def mkstream():
    s = vp()
    rc = lib.aecStreamCreate(ct.byref(s))
    return rc, s.value or None


def mkevent():
    e = vp()
    rc = lib.aecEventCreate(ct.byref(e))
    return rc, e.value or None


results = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    line = f"[{tag}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    results.append(bool(cond))


print("=== CheapSeek R106 probe ===")
print()

# ── C1: Basic Event lifecycle ──────────────────────────────────────────
# Create -> (unrecorded) -> Record(sync work) -> Sync event -> Query -> Destroy
reset()
_, d = alloc(64)
_, st = mkstream()
_, ev = mkevent()

host = (ct.c_uint8 * 64)()
r_copy = lib.aecCopyAsync(d, host, 64, D2H, st)       # async work on stream
r_rec = lib.aecEventRecord(ev, st)                     # marker after the work
r_sync = lib.aecEventSynchronize(ev)                    # wait for marker
r_qry = lib.aecEventQuery(ev)                           # completed

lib.aecStreamSync(st)
lib.aecStreamDestroy(st)
lib.aecEventDestroy(ev)
free(d)
check("basic Event lifecycle: create->record->sync->query->destroy",
      r_copy == SUCCESS and r_rec == SUCCESS and r_sync == SUCCESS and r_qry == SUCCESS,
      f"async={ename(r_copy)} rec={ename(r_rec)} sync={ename(r_sync)} qry={ename(r_qry)}")

# ── C2: Unrecorded Event query -> INVALID_ARGUMENT ─────────────────────
# spec says "未 record Event 的 query/sync 返回 invalid argument"
reset()
_, ev2 = mkevent()
r_qry_unrec = lib.aecEventQuery(ev2)
lib.aecEventDestroy(ev2)
check("unrecorded Event query -> INVALID_ARGUMENT",
      r_qry_unrec == INV_ARG,
      f"got={ename(r_qry_unrec)}")

# ── C3: Unrecorded Event sync -> INVALID_ARGUMENT ──────────────────────
reset()
_, ev3 = mkevent()
r_sync_unrec = lib.aecEventSynchronize(ev3)
lib.aecEventDestroy(ev3)
check("unrecorded Event sync -> INVALID_ARGUMENT",
      r_sync_unrec == INV_ARG,
      f"got={ename(r_sync_unrec)}")

# ── C4: Query without drain after record -> NOT_READY ──────────────────
# "record 后未 sync 立即 query -> NOT_READY（query 不触发 drain）"
# We enqueue work, record, and immediately query before any drain.
reset()
_, d4 = alloc(65536)
_, st4 = mkstream()
_, ev4 = mkevent()

host4 = (ct.c_uint8 * 65536)()
lib.aecCopyAsync(d4, host4, 65536, H2D, st4)           # big async copy
r_rec4 = lib.aecEventRecord(ev4, st4)                   # marker after copy
r_qry4 = lib.aecEventQuery(ev4)                         # immediate query, no drain
r_sync4 = lib.aecEventSynchronize(ev4)                  # now drain
r_qry5 = lib.aecEventQuery(ev4)                         # should be done

lib.aecStreamSync(st4)
lib.aecStreamDestroy(st4)
lib.aecEventDestroy(ev4)
free(d4)
check("record -> immediate query -> NOT_READY, sync -> query -> SUCCESS",
      r_rec4 == SUCCESS and r_qry4 == NOT_READY and r_sync4 == SUCCESS and r_qry5 == SUCCESS,
      f"rec={ename(r_rec4)} qry_before_sync={ename(r_qry4)} sync={ename(r_sync4)} qry_after={ename(r_qry5)}")

# ── C5: Same Event on different Streams (Stream switch) ────────────────
# "同一 Event 先后在不同 Stream 上 record（换 Stream）"
# Each record produces a new generation.
reset()
_, d5a = alloc(64)
_, d5b = alloc(64)
_, stA = mkstream()
_, stB = mkstream()
_, ev5 = mkevent()

# Record on stream A, sync
host5 = (ct.c_uint8 * 64)()
lib.aecCopyAsync(d5a, host5, 64, D2H, stA)
r_recA = lib.aecEventRecord(ev5, stA)
r_syncA = lib.aecEventSynchronize(ev5)
r_qryA = lib.aecEventQuery(ev5)

# Record same event on stream B, sync
lib.aecCopyAsync(d5b, host5, 64, D2H, stB)
r_recB = lib.aecEventRecord(ev5, stB)
r_syncB = lib.aecEventSynchronize(ev5)
r_qryB = lib.aecEventQuery(ev5)

lib.aecStreamSync(stA)
lib.aecStreamSync(stB)
lib.aecStreamDestroy(stA)
lib.aecStreamDestroy(stB)
lib.aecEventDestroy(ev5)
free(d5a)
free(d5b)
check("same Event recorded on different Streams, both complete",
      r_recA == SUCCESS and r_syncA == SUCCESS and r_qryA == SUCCESS
      and r_recB == SUCCESS and r_syncB == SUCCESS and r_qryB == SUCCESS,
      f"A: rec={ename(r_recA)} sync={ename(r_syncA)} qry={ename(r_qryA)}  "
      f"B: rec={ename(r_recB)} sync={ename(r_syncB)} qry={ename(r_qryB)}")

# ── C6: EventDestroy with pending async record ─────────────────────────
# "EventDestroy 时 pending record 的等待 + generation 匹配"
# Enqueue large copy, record event, immediately destroy event.
# Destroy must wait for the latest generation to complete.
reset()
_, d6 = alloc(1048576)
_, st6 = mkstream()
_, ev6 = mkevent()

host6 = (ct.c_uint8 * 1048576)()
lib.aecCopyAsync(d6, host6, 1048576, H2D, st6)
r_rec6 = lib.aecEventRecord(ev6, st6)
r_destroy6 = lib.aecEventDestroy(ev6)                    # destroy with pending record

lib.aecStreamSync(st6)
lib.aecStreamDestroy(st6)
free(d6)
check("EventDestroy with pending async record succeeds (waits for generation)",
      r_rec6 == SUCCESS and r_destroy6 == SUCCESS,
      f"rec={ename(r_rec6)} destroy={ename(r_destroy6)}")

# ── C7: ElapsedCycles normal case ──────────────────────────────────────
# Two events on the same stream, elapsed returns positive cycles
reset()
_, d7 = alloc(64)
_, st7 = mkstream()
_, start = mkevent()
_, end = mkevent()

host7 = (ct.c_uint8 * 64)()
lib.aecCopyAsync(d7, host7, 64, D2H, st7)
r_start = lib.aecEventRecord(start, st7)                  # first marker
lib.aecCopyAsync(d7, host7, 64, H2D, st7)
r_end = lib.aecEventRecord(end, st7)                      # second marker

lib.aecEventSynchronize(start)
lib.aecEventSynchronize(end)

cyc = u64()
r_elapsed = lib.aecEventElapsedCycles(start, end, ct.byref(cyc))

lib.aecStreamSync(st7)
lib.aecStreamDestroy(st7)
lib.aecEventDestroy(start)
lib.aecEventDestroy(end)
free(d7)
check("ElapsedCycles normal — two recorded events, cycles >= 0",
      r_start == SUCCESS and r_end == SUCCESS and r_elapsed == SUCCESS,
      f"start={ename(r_start)} end={ename(r_end)} elapsed={ename(r_elapsed)} cycles={cyc.value}")

# ── C8: ElapsedCycles end < start -> INVALID_ARGUMENT ──────────────────
# spec: "elapsed 要求 end cycle 不小于 start cycle"
reset()
_, d8 = alloc(64)
_, st8 = mkstream()
_, evA = mkevent()
_, evB = mkevent()

host8 = (ct.c_uint8 * 64)()
lib.aecCopyAsync(d8, host8, 64, D2H, st8)
lib.aecEventRecord(evA, st8)                              # first marker
lib.aecCopyAsync(d8, host8, 64, H2D, st8)
lib.aecEventRecord(evB, st8)                              # second marker (later)

lib.aecEventSynchronize(evA)
lib.aecEventSynchronize(evB)

cyc8 = u64()
r_swapped = lib.aecEventElapsedCycles(evB, evA, ct.byref(cyc8))  # end(B) < start(A) in time

lib.aecStreamSync(st8)
lib.aecStreamDestroy(st8)
lib.aecEventDestroy(evA)
lib.aecEventDestroy(evB)
free(d8)
check("ElapsedCycles with end < start -> INVALID_ARGUMENT",
      r_swapped == INV_ARG,
      f"got={ename(r_swapped)}")

# ── C9: ElapsedCycles with NULL cycles pointer -> INVALID_ARGUMENT ─────
reset()
_, d9 = alloc(64)
_, st9 = mkstream()
_, evX = mkevent()
_, evY = mkevent()

host9 = (ct.c_uint8 * 64)()
lib.aecCopyAsync(d9, host9, 64, D2H, st9)
lib.aecEventRecord(evX, st9)
lib.aecCopyAsync(d9, host9, 64, H2D, st9)
lib.aecEventRecord(evY, st9)
lib.aecEventSynchronize(evX)
lib.aecEventSynchronize(evY)

r_null_out = lib.aecEventElapsedCycles(evX, evY, None)   # null cycles pointer

lib.aecStreamSync(st9)
lib.aecStreamDestroy(st9)
lib.aecEventDestroy(evX)
lib.aecEventDestroy(evY)
free(d9)
check("ElapsedCycles with NULL cycles pointer -> INVALID_ARGUMENT",
      r_null_out == INV_ARG,
      f"got={ename(r_null_out)}")

# ── C10: ElapsedCycles with unrecorded event -> INVALID_ARGUMENT ───────
reset()
_, evU = mkevent()
_, evV = mkevent()
_, st10 = mkstream()
_, d10 = alloc(64)

# Record evV only; leave evU unrecorded
host10 = (ct.c_uint8 * 64)()
lib.aecCopyAsync(d10, host10, 64, D2H, st10)
lib.aecEventRecord(evV, st10)
lib.aecEventSynchronize(evV)

cyc10 = u64()
r_unrec_start = lib.aecEventElapsedCycles(evU, evV, ct.byref(cyc10))
r_unrec_end = lib.aecEventElapsedCycles(evV, evU, ct.byref(cyc10))

lib.aecStreamSync(st10)
lib.aecStreamDestroy(st10)
lib.aecEventDestroy(evU)
lib.aecEventDestroy(evV)
free(d10)
check("ElapsedCycles with unrecorded event -> INVALID_ARGUMENT (both positions)",
      r_unrec_start == INV_ARG and r_unrec_end == INV_ARG,
      f"unrec_start={ename(r_unrec_start)} unrec_end={ename(r_unrec_end)}")

# ── C11: Async error attribution — StreamSync returns FIRST error ──────
# "异步错误归属——Stream sync 返回首个错误，而非最后"
# Use aecDeviceInjectFault(FAULT_NEXT_DMA) to force the first DMA to fail,
# then a second valid DMA. StreamSync should return the device error, not SUCCESS.
reset()
_, d11 = alloc(4096)
_, st11 = mkstream()
_, ev11 = mkevent()

# Inject fault for the next DMA operation
dev.aecDeviceInjectFault(FAULT_NEXT_DMA)

host11 = (ct.c_uint8 * 4096)()
r_fail = lib.aecCopyAsync(d11, host11, 4096, H2D, st11)   # this will fail with injected fault
r_ok = lib.aecCopyAsync(d11, host11, 4096, D2H, st11)     # this should succeed (fault consumed)
r_rec11 = lib.aecEventRecord(ev11, st11)                   # marker after both

r_sync11 = lib.aecStreamSync(st11)                         # should return first error (DEVICE)
r_qry11 = lib.aecEventQuery(ev11)                          # event should be complete

lib.aecStreamDestroy(st11)
lib.aecEventDestroy(ev11)
free(d11)
check("async error attribution: StreamSync returns FIRST error (DEVICE), event complete",
      r_fail == SUCCESS and r_ok == SUCCESS and r_rec11 == SUCCESS
      and r_sync11 == DEVICE and r_qry11 == SUCCESS,
      f"fail_enq={ename(r_fail)} ok_enq={ename(r_ok)} rec={ename(r_rec11)} "
      f"sync={ename(r_sync11)} qry={ename(r_qry11)}")

# ── C12: Handle slot reuse after destroy -> fresh state ────────────────
# "handle 槽位 destroy 后复用须全新状态"
# Destroy an event that had a record, then create a new one.
# The new event must have no stale generation.
reset()
_, d12 = alloc(64)
_, st12 = mkstream()
_, ev_old = mkevent()

host12 = (ct.c_uint8 * 64)()
lib.aecCopyAsync(d12, host12, 64, D2H, st12)
lib.aecEventRecord(ev_old, st12)
lib.aecEventSynchronize(ev_old)
lib.aecEventDestroy(ev_old)

# Create new event — must be unrecorded (query -> INV_ARG, not SUCCESS from stale gen)
_, ev_new = mkevent()
r_qry_new = lib.aecEventQuery(ev_new)

lib.aecStreamSync(st12)
lib.aecStreamDestroy(st12)
lib.aecEventDestroy(ev_new)
free(d12)
check("handle slot reuse: destroyed event's slot -> fresh state (unrecorded)",
      r_qry_new == INV_ARG,
      f"query new event -> {ename(r_qry_new)} (expected INVALID_ARGUMENT)")

# ── C13: EventDestroy(NULL) / EventRecord NULL args -> INVALID_HANDLE ──
# Runtime treats NULL as an invalid handle (semantically correct: NULL is not a valid handle)
reset()
r_destroy_null = lib.aecEventDestroy(None)

_, st13 = mkstream()
_, ev13 = mkevent()
r_rec_null_ev = lib.aecEventRecord(None, st13)
r_rec_null_st = lib.aecEventRecord(ev13, None)
lib.aecStreamDestroy(st13)
lib.aecEventDestroy(ev13)

check("NULL handle args -> INVALID_HANDLE (destroy/record null ev/record null st)",
      r_destroy_null == INV_HANDLE and r_rec_null_ev == INV_HANDLE and r_rec_null_st == INV_HANDLE,
      f"destroy(NULL)={ename(r_destroy_null)} rec(NULL,st)={ename(r_rec_null_ev)} rec(ev,NULL)={ename(r_rec_null_st)}")

print()
print(f"=== {sum(results)}/{len(results)} checks passed ===")
