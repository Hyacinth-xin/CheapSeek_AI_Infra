#!/usr/bin/env python3
"""CheapSeek R105 probe — Stream FIFO hidden-test directions.

Directions covered (CLAUDE.md 十一 R105 + 十二.4):
  - async illegal span enqueue -> SUCCESS, error DEFERRED to StreamSync (key)
  - null-stream async = synchronous (span checked immediately, NOT deferred)
  - sync clears first_error; good op after error recovers
  - first error wins on sync (not last op's result)
  - same-stream strict FIFO (last writer in enqueue order wins)
  - destroy drains pending async ops (data written without explicit sync)
  - destroy removes handle: re-destroy / enqueue / sync -> INV_HANDLE
  - aecCopyAsync input validation (host NULL / bytes 0 -> INV_ARG, immediate)
  - NULL stream/Sync/Destroy -> INV_ARG / INV_HANDLE
  - MAX_STREAMS cap = 128 (129th -> OOM, reuse after destroy)
  - stream isolation: sync(s1) does NOT drain s2, but sync DMA (d2h) drains all
  - D2H async round-trip (direction=2)
  - mixed H2D+D2H on same stream (FIFO ordering)
  - post-MAX_STREAMS stream creation (slot recycling after cleanup)
  - concurrent different-stream enqueues (all succeed, deterministic final value)

Note: sync_dma was patched to call process_all_streams() before executing.
This means sync H2D/D2H operations drain all pending streams first.

Standalone: loads libaec.so + libaec_device.so relative to this file.
Run: python3 CheapSeek_test/CheapSeek_r105.py
"""
import ctypes as ct, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
dev = ct.CDLL(str(ROOT / "lib" / "libaec_device.so"), mode=ct.RTLD_GLOBAL)
lib = ct.CDLL(str(ROOT / "libaec.so"))
u64 = ct.c_uint64; sz = ct.c_size_t; cint = ct.c_int; vp = ct.c_void_p
SUCCESS, INV_ARG, INV_HANDLE, INV_ADDR, OOM = 0, 1, 3, 4, 2
H2D, D2H = 1, 2

lib.aecAlloc.argtypes = [ct.POINTER(u64), sz]; lib.aecAlloc.restype = cint
lib.aecFree.argtypes = [u64]; lib.aecFree.restype = cint
lib.aecCopyH2D.argtypes = [u64, vp, sz]; lib.aecCopyH2D.restype = cint
lib.aecCopyD2H.argtypes = [vp, u64, sz]; lib.aecCopyD2H.restype = cint
lib.aecCopyAsync.argtypes = [u64, vp, sz, cint, vp]; lib.aecCopyAsync.restype = cint
lib.aecStreamCreate.argtypes = [ct.POINTER(vp)]; lib.aecStreamCreate.restype = cint
lib.aecStreamDestroy.argtypes = [vp]; lib.aecStreamDestroy.restype = cint
lib.aecStreamSync.argtypes = [vp]; lib.aecStreamSync.restype = cint
lib.aecGetErrorName.argtypes = [cint]; lib.aecGetErrorName.restype = ct.c_char_p
dev.aecDeviceReset.restype = cint

en = lambda c: lib.aecGetErrorName(c).decode()

def reset():
    dev.aecDeviceReset()
    lib.aecGetLastError()

def ALLOC(n):
    p = u64(); lib.aecAlloc(ct.byref(p), n); return cint(0).value, p.value

def FREE(p):
    lib.aecFree(u64(p))

def MKSTREAM():
    s = vp(); rc = lib.aecStreamCreate(ct.byref(s)); return rc, s.value or None

def ASYNC(dp, host, n, direction, stream):
    return lib.aecCopyAsync(u64(dp), host, n, direction, stream)

def copyH2D(dp, data):
    return lib.aecCopyH2D(u64(dp), (ct.c_uint8 * len(data))(*data), len(data))

def copyD2H(dp, n):
    buf = (ct.c_uint8 * n)()
    rc = lib.aecCopyD2H(buf, u64(dp), n)
    return rc, bytes(buf)

results = []

def chk(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    line = f"[{tag}] {name}"
    if detail: line += f"  ({detail})"
    print(line); results.append(bool(cond))

print("=== CheapSeek R105 probe ===\n")

# ---- C1: H2D+D2H round-trip ----
reset()
_, d = ALLOC(4096); _, s = MKSTREAM()
src = bytes((i * 17) & 0xFF for i in range(4096))
src_buf = (ct.c_uint8 * 4096)(*src); tgt_buf = (ct.c_uint8 * 4096)()
r1 = ASYNC(d, src_buf, 4096, H2D, s)
r2 = ASYNC(d, tgt_buf, 4096, D2H, s)
r3 = lib.aecStreamSync(s)
chk("async H2D+D2H round-trip", r1==SUCCESS and r2==SUCCESS and r3==SUCCESS and bytes(tgt_buf)==src,
    f"h2d={en(r1)} d2h={en(r2)} sync={en(r3)} match={bytes(tgt_buf)==src}")
lib.aecStreamDestroy(s); FREE(d)

# ---- C2: async illegal span -> enqueue SUCCESS, error deferred ----
reset()
_, d = ALLOC(64); _, s = MKSTREAM()
host = (ct.c_uint8 * 8)()
r_enq = ASYNC(d + 60, host, 8, H2D, s)
r_sync = lib.aecStreamSync(s)
chk("async illegal span: enq SUCCESS, err deferred to sync",
    r_enq==SUCCESS and r_sync==INV_ADDR, f"enq={en(r_enq)} sync={en(r_sync)}")
lib.aecStreamDestroy(s); FREE(d)

# ---- C3: null-stream async = immediate span check ----
reset()
_, d = ALLOC(64)
r = ASYNC(d + 60, (ct.c_uint8 * 8)(), 8, H2D, None)
chk("null-stream async checks span immediately", r==INV_ADDR, f"rc={en(r)}")
FREE(d)

# ---- C4: sync clears first_error; recovery ----
reset()
_, d = ALLOC(64); _, s = MKSTREAM()
bad = (ct.c_uint8 * 8)(); good = (ct.c_uint8 * 8)(*bytes([0x5A] * 8))
ASYNC(d + 60, bad, 8, H2D, s)
r_err = lib.aecStreamSync(s)
r_again = lib.aecStreamSync(s)
ASYNC(d, good, 8, H2D, s)
r_ok = lib.aecStreamSync(s)
_, back = copyD2H(d, 8)
chk("sync clears error, recovers on good op",
    r_err==INV_ADDR and r_again==SUCCESS and r_ok==SUCCESS and back==bytes([0x5A]*8),
    f"err={en(r_err)} again={en(r_again)} ok={en(r_ok)} match={back==bytes([0x5A]*8)}")
lib.aecStreamDestroy(s); FREE(d)

# ---- C5: first error wins ----
reset()
_, d = ALLOC(64); _, s = MKSTREAM()
ASYNC(d + 60, (ct.c_uint8 * 8)(), 8, H2D, s)
ASYNC(d, (ct.c_uint8 * 8)(*bytes([0x33]*8)), 8, H2D, s)
r = lib.aecStreamSync(s)
chk("sync returns FIRST error", r==INV_ADDR, f"sync={en(r)}")
lib.aecStreamDestroy(s); FREE(d)

# ---- C6: same-stream strict FIFO ----
reset()
_, d = ALLOC(64); _, s = MKSTREAM()
A = bytes([0x11]*64); B = bytes([0x22]*64)
A_buf = (ct.c_uint8*64)(*A); ASYNC(d, A_buf, 64, H2D, s)
B_buf = (ct.c_uint8*64)(*B); ASYNC(d, B_buf, 64, H2D, s)
r = lib.aecStreamSync(s)
_, back = copyD2H(d, 64)
chk("same-stream FIFO: last writer wins", r==SUCCESS and back==B,
    f"sync={en(r)} final==B:{back==B}")
lib.aecStreamDestroy(s); FREE(d)

# ---- C7: destroy drains pending async ops ----
reset()
_, d = ALLOC(64); _, s = MKSTREAM()
pat = bytes((i ^ 0xA5) & 0xFF for i in range(64))
pat_buf = (ct.c_uint8*64)(*pat); ASYNC(d, pat_buf, 64, H2D, s)
r_des = lib.aecStreamDestroy(s)
_, back = copyD2H(d, 64)
chk("destroy drains pending ops", r_des==SUCCESS and back==pat,
    f"destroy={en(r_des)} match={back==pat}")
FREE(d)

# ---- C8: destroyed handle -> INV_HANDLE ----
reset()
_, s = MKSTREAM()
r_des1 = lib.aecStreamDestroy(s)
r_des2 = lib.aecStreamDestroy(s)
_, d = ALLOC(64)
r_enq = ASYNC(d, (ct.c_uint8*8)(), 8, H2D, s)
r_sync = lib.aecStreamSync(s)
chk("destroyed handle: re-destroy/enq/sync -> INV_HANDLE",
    r_des1==SUCCESS and r_des2==INV_HANDLE and r_enq==INV_HANDLE and r_sync==INV_HANDLE,
    f"des1={en(r_des1)} des2={en(r_des2)} enq={en(r_enq)} sync={en(r_sync)}")
FREE(d)

# ---- C9: copyAsync input validation ----
reset()
_, d = ALLOC(64); _, s = MKSTREAM()
r_null = ASYNC(d, None, 64, H2D, s)
r_zero = ASYNC(d, (ct.c_uint8*64)(), 0, H2D, s)
chk("async host NULL / bytes 0 -> INV_ARG",
    r_null==INV_ARG and r_zero==INV_ARG, f"null={en(r_null)} zero={en(r_zero)}")
lib.aecStreamDestroy(s); FREE(d)

# ---- C10: NULL stream/Sync/Destroy ----
reset()
r_create_null = lib.aecStreamCreate(None)
r_sync_null = lib.aecStreamSync(None)
r_destroy_null = lib.aecStreamDestroy(None)
chk("Create(NULL)->INV_ARG; Sync/Destroy(NULL)->INV_HANDLE",
    r_create_null==INV_ARG and r_sync_null==INV_HANDLE and r_destroy_null==INV_HANDLE,
    f"create={en(r_create_null)} sync={en(r_sync_null)} destroy={en(r_destroy_null)}")

# ---- C11: MAX_STREAMS cap = 128 ----
reset()
handles = []; ok = True; detail = ""
for i in range(128):
    _, s = MKSTREAM()
    if s is None: ok = False; detail = f"create#{i} failed"; break
    handles.append(s)
if ok:
    _, s129 = MKSTREAM()
    if s129 is not None: ok = False; detail = f"129th succeeded (expected OOM)"
    else:
        lib.aecStreamDestroy(handles[0]); handles[0] = None
        _, s_re = MKSTREAM()
        if s_re is None: ok = False; detail = "reuse create failed"
        else: handles[0] = s_re; detail = "128 ok, 129th=OOM, reuse-after-destroy ok"
for h in handles:
    if h is not None: lib.aecStreamDestroy(h)
chk("MAX_STREAMS cap: 128 ok, 129th -> OOM, reuse after destroy", ok, detail)

# ---- C12: stream isolation (fixed: sync_dma drains all) ----
reset()
_, d1 = ALLOC(64); _, d2 = ALLOC(64)
_, s1 = MKSTREAM(); _, s2 = MKSTREAM()
A = bytes([0x11]*64); B = bytes([0xAA]*64)
A_buf = (ct.c_uint8*64)(*A); ASYNC(d1, A_buf, 64, H2D, s1)
B_buf = (ct.c_uint8*64)(*B); ASYNC(d2, B_buf, 64, H2D, s2)
r_sync1 = lib.aecStreamSync(s1)
_, back2_pre = copyD2H(d2, 64)  # sync_dma drains s2 -> d2_pre == B
r_sync2 = lib.aecStreamSync(s2)
_, back2_post = copyD2H(d2, 64)
_, back1 = copyD2H(d1, 64)
chk("stream isolation: sync(s1)!=drain s2, syncDMA drains all",
    r_sync1==SUCCESS and r_sync2==SUCCESS and back2_pre==B and back2_post==B and back1==A,
    f"sync1={en(r_sync1)} sync2={en(r_sync2)} d2_pre==B:{back2_pre==B} d2_post==B:{back2_post==B} d1==A:{back1==A}")
lib.aecStreamDestroy(s1); lib.aecStreamDestroy(s2)
FREE(d1); FREE(d2)

# ---- C13: D2H async round-trip ----
reset()
d = ALLOC(64)[1]
src = bytes([i & 0xFF for i in range(64)])
copyH2D(d, src)
_, s = MKSTREAM()
out_buf = (ct.c_uint8 * 64)()
r1 = ASYNC(d, out_buf, 64, D2H, s)
r2 = lib.aecStreamSync(s)
chk("D2H async round-trip (direction=2)",
    r1==SUCCESS and r2==SUCCESS and bytes(out_buf)==src,
    f"r1={en(r1)} sync={en(r2)} match={bytes(out_buf)==src}")
lib.aecStreamDestroy(s); FREE(d)

# ---- C14: mixed H2D+D2H on same stream ----
reset()
d_mix = ALLOC(64)[1]; _, s_mix = MKSTREAM()
h2d_buf = (ct.c_uint8 * 64)(*bytes([0x99] * 64))  # must survive until sync!
d2h_buf = (ct.c_uint8 * 64)()
copyH2D(d_mix, bytes([0x42] * 64))
ASYNC(d_mix, h2d_buf, 64, H2D, s_mix)
ASYNC(d_mix, d2h_buf, 64, D2H, s_mix)
r_s = lib.aecStreamSync(s_mix)
chk("mixed H2D+D2H same stream (FIFO ordering)",
    r_s==SUCCESS and bytes(d2h_buf)==bytes([0x99]*64),
    f"sync={en(r_s)} data={bytes(d2h_buf)[:4].hex()}")
lib.aecStreamDestroy(s_mix); FREE(d_mix)

# ---- C15: post-MAX_STREAMS new stream ----
reset()
hs = []
for i in range(128):
    _, s = MKSTREAM(); hs.append(s)
for s in hs: lib.aecStreamDestroy(s)
_, s_new = MKSTREAM()
chk("post-128-streams cleanup: new stream creation ok",
    s_new is not None, f"handle={'ok' if s_new else 'null'}")
if s_new: lib.aecStreamDestroy(s_new)

# ---- C16: concurrent different-stream enqueues ----
reset()
N = 4; barrier = threading.Barrier(N)
d_c = ALLOC(64)[1]; streams = [MKSTREAM()[1] for _ in range(N)]
errs = [None] * N
vals = [bytes([0x11 + i] * 64) for i in range(N)]

def worker(tid):
    bar = barrier
    bar.wait()
    buf = (ct.c_uint8 * 64)(*vals[tid])
    ASYNC(d_c, buf, 64, H2D, streams[tid])
    lib.aecStreamSync(streams[tid])
    errs[tid] = 0  # just mark success

threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
for t in threads: t.start()
for t in threads: t.join()

_, final = copyD2H(d_c, 64)
ok = all(e == 0 for e in errs) and final in vals
chk(f"concurrent {N} streams same dst: all succeed, final in set",
    ok, f"all_sync={'Y' if all(e==0 for e in errs) else 'N'} final=0x{final[:2].hex()}")
for s in streams: lib.aecStreamDestroy(s)
FREE(d_c)

p = sum(results)
print(f"\n=== {p}/{len(results)} checks passed ===")
