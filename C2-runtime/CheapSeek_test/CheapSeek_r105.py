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
  - aecStreamCreate(NULL) / Sync(NULL) / Destroy(NULL) error codes
  - MAX_STREAMS cap = 128 (129th -> OOM, reuse after destroy)
  - stream sync does NOT drain other streams' pending ops (independence)

Run: python3 test/CheapSeek_r105.py
"""
import ctypes as ct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
dev = ct.CDLL(str(ROOT / "lib" / "libaec_device.so"), mode=ct.RTLD_GLOBAL)
lib = ct.CDLL(str(ROOT / "libaec.so"))

u64 = ct.c_uint64
sz = ct.c_size_t
cint = ct.c_int
vp = ct.c_void_p

SUCCESS = 0
INV_ARG = 1
OOM = 2
INV_HANDLE = 3
INV_ADDR = 4

H2D = 1  # AEC_COPY_HOST_TO_DEVICE
D2H = 2  # AEC_COPY_DEVICE_TO_HOST

lib.aecAlloc.argtypes = [ct.POINTER(u64), sz]
lib.aecAlloc.restype = cint
lib.aecFree.argtypes = [u64]
lib.aecFree.restype = cint
lib.aecCopyAsync.argtypes = [u64, vp, sz, cint, vp]
lib.aecCopyAsync.restype = cint
lib.aecCopyH2D.argtypes = [u64, vp, sz]
lib.aecCopyH2D.restype = cint
lib.aecCopyD2H.argtypes = [vp, u64, sz]
lib.aecCopyD2H.restype = cint
lib.aecStreamCreate.argtypes = [ct.POINTER(vp)]
lib.aecStreamCreate.restype = cint
lib.aecStreamDestroy.argtypes = [vp]
lib.aecStreamDestroy.restype = cint
lib.aecStreamSync.argtypes = [vp]
lib.aecStreamSync.restype = cint
lib.aecGetErrorName.argtypes = [cint]
lib.aecGetErrorName.restype = ct.c_char_p
dev.aecDeviceReset.restype = cint


def ename(c):
    return lib.aecGetErrorName(c).decode()


def reset():
    dev.aecDeviceReset()


def alloc(n):
    p = u64()
    rc = lib.aecAlloc(ct.byref(p), n)
    return rc, p.value


def free(p):
    return lib.aecFree(u64(p))


def mkstream():
    h = vp()
    rc = lib.aecStreamCreate(ct.byref(h))
    return rc, h


def destroy(h):
    return lib.aecStreamDestroy(h)


def sync(h):
    return lib.aecStreamSync(h)


def copy_async(dp, host, n, direction, stream):
    return lib.aecCopyAsync(u64(dp), host, n, direction, stream)


def d2h(dp, n):
    buf = (ct.c_uint8 * n)()
    rc = lib.aecCopyD2H(buf, u64(dp), n)
    return rc, bytes(buf)


results = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    line = f"[{tag}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    results.append(bool(cond))


print("=== CheapSeek R105 probe ===")
print()

# C1: async H2D+D2H round-trip on a fresh stream (baseline + data integrity)
reset()
rc, d = alloc(4096)
rcs, s = mkstream()
src = bytes((i * 17) & 0xFF for i in range(4096))
src_buf = (ct.c_uint8 * 4096)(*src)
tgt_buf = (ct.c_uint8 * 4096)()
r1 = copy_async(d, src_buf, 4096, H2D, s)
r2 = copy_async(d, tgt_buf, 4096, D2H, s)
r3 = sync(s)
check("async H2D+D2H round-trip preserves data",
      r1 == SUCCESS and r2 == SUCCESS and r3 == SUCCESS and bytes(tgt_buf) == src,
      f"h2d={ename(r1)} d2h={ename(r2)} sync={ename(r3)} match={bytes(tgt_buf) == src}")
destroy(s)
free(d)

# C2: async illegal span -> enqueue SUCCESS, error DEFERRED to StreamSync (key)
reset()
rc, d = alloc(64)
rcs, s = mkstream()
host = (ct.c_uint8 * 8)()
r_enq = copy_async(d + 60, host, 8, H2D, s)   # [d+60, d+68) crosses d+64 end
r_sync = sync(s)
check("async illegal span: enqueue SUCCESS, error deferred to StreamSync",
      r_enq == SUCCESS and r_sync == INV_ADDR,
      f"enqueue={ename(r_enq)} sync={ename(r_sync)}")
destroy(s)
free(d)

# C3: null-stream async = synchronous (span checked immediately, NOT deferred)
reset()
rc, d = alloc(64)
host = (ct.c_uint8 * 8)()
r = copy_async(d + 60, host, 8, H2D, None)    # null stream -> sync_dma path
check("null-stream async checks span immediately -> INV_ADDR (not deferred)",
      r == INV_ADDR, f"rc={ename(r)}")
free(d)

# C4: sync clears first_error; good op after error recovers
reset()
rc, d = alloc(64)
rcs, s = mkstream()
bad = (ct.c_uint8 * 8)()
good = (ct.c_uint8 * 8)(*bytes([0x5A] * 8))
copy_async(d + 60, bad, 8, H2D, s)            # failing op
r_err = sync(s)
r_again = sync(s)                             # no new op -> SUCCESS (cleared)
copy_async(d, good, 8, H2D, s)                # good op after error
r_ok = sync(s)
rc2, back = d2h(d, 8)
check("sync clears first_error and recovers on next good op",
      r_err == INV_ADDR and r_again == SUCCESS and r_ok == SUCCESS and back == bytes([0x5A] * 8),
      f"err={ename(r_err)} again={ename(r_again)} ok={ename(r_ok)} match={back == bytes([0x5A] * 8)}")
destroy(s)
free(d)

# C5: first error wins on sync (not last op's result)
reset()
rc, d = alloc(64)
rcs, s = mkstream()
bad = (ct.c_uint8 * 8)()
good = (ct.c_uint8 * 8)(*bytes([0x33] * 8))
copy_async(d + 60, bad, 8, H2D, s)           # failing first
copy_async(d, good, 8, H2D, s)               # succeeding second
r = sync(s)
check("sync returns FIRST error, not last op's SUCCESS",
      r == INV_ADDR, f"sync={ename(r)} (last op succeeded, first failed)")
destroy(s)
free(d)

# C6: same-stream strict FIFO (last writer in enqueue order wins)
reset()
rc, d = alloc(64)
rcs, s = mkstream()
A = bytes([0x11] * 64)
B = bytes([0x22] * 64)
A_buf = (ct.c_uint8 * 64)(*A)
B_buf = (ct.c_uint8 * 64)(*B)
copy_async(d, A_buf, 64, H2D, s)
copy_async(d, B_buf, 64, H2D, s)
r = sync(s)
rc2, back = d2h(d, 64)
check("same-stream FIFO: last enqueued writer wins",
      r == SUCCESS and back == B,
      f"sync={ename(r)} final==B:{back == B} final==A:{back == A}")
destroy(s)
free(d)

# C7: destroy drains pending async ops (data written without explicit sync)
reset()
rc, d = alloc(64)
rcs, s = mkstream()
pat = bytes((i ^ 0xA5) & 0xFF for i in range(64))
pat_buf = (ct.c_uint8 * 64)(*pat)
copy_async(d, pat_buf, 64, H2D, s)
r_des = destroy(s)                           # must drain pending H2D
rc2, back = d2h(d, 64)
check("destroy drains pending async ops (data written)",
      r_des == SUCCESS and back == pat,
      f"destroy={ename(r_des)} match={back == pat}")
free(d)

# C8: destroy removes handle: re-destroy / enqueue / sync -> INV_HANDLE
reset()
rcs, s = mkstream()
r_des1 = destroy(s)
r_des2 = destroy(s)
rc, d = alloc(64)
host = (ct.c_uint8 * 8)()
r_enq = copy_async(d, host, 8, H2D, s)       # enqueue to destroyed stream
r_sync = sync(s)
check("destroyed handle: re-destroy/enqueue/sync -> INV_HANDLE",
      r_des1 == SUCCESS and r_des2 == INV_HANDLE and r_enq == INV_HANDLE and r_sync == INV_HANDLE,
      f"des1={ename(r_des1)} des2={ename(r_des2)} enq={ename(r_enq)} sync={ename(r_sync)}")
free(d)

# C9: aecCopyAsync input validation (host NULL / bytes 0 -> INV_ARG, immediate)
reset()
rc, d = alloc(64)
rcs, s = mkstream()
host = (ct.c_uint8 * 64)()
r_null = copy_async(d, None, 64, H2D, s)
r_zero = copy_async(d, host, 0, H2D, s)
check("async host NULL / bytes 0 -> INVALID_ARGUMENT (immediate)",
      r_null == INV_ARG and r_zero == INV_ARG,
      f"null={ename(r_null)} zero={ename(r_zero)}")
destroy(s)
free(d)

# C10: aecStreamCreate(NULL) / Sync(NULL) / Destroy(NULL) error codes
reset()
r_create_null = lib.aecStreamCreate(None)
r_sync_null = sync(None)
r_destroy_null = destroy(None)
check("Create(NULL)->INV_ARG; Sync/Destroy(NULL)->INV_HANDLE",
      r_create_null == INV_ARG and r_sync_null == INV_HANDLE and r_destroy_null == INV_HANDLE,
      f"create={ename(r_create_null)} sync={ename(r_sync_null)} destroy={ename(r_destroy_null)}")

# C11: MAX_STREAMS cap = 128 (129th -> OOM, reuse after destroy)
reset()
handles = []
ok = True
detail = ""
for i in range(128):
    rc, h = mkstream()
    if rc != SUCCESS:
        ok = False
        detail = f"create#{i}={ename(rc)}"
        break
    handles.append(h)
if ok:
    rc129, h129 = mkstream()
    if rc129 != OOM:
        ok = False
        detail = f"129th={ename(rc129)} (expected OOM)"
    else:
        destroy(handles[0])
        handles[0] = None
        rc_re, h_re = mkstream()
        if rc_re != SUCCESS:
            ok = False
            detail = f"reuse create={ename(rc_re)}"
        else:
            handles[0] = h_re
            detail = "128 ok, 129th=OOM, reuse-after-destroy ok"
for h in handles:
    if h is not None:
        destroy(h)
check("MAX_STREAMS cap: 128 ok, 129th -> OOM, reuse after destroy",
      ok, detail)

# C12: stream sync does NOT drain other streams' pending ops
reset()
rc, d1 = alloc(64)
rc2, d2 = alloc(64)
rcs1, s1 = mkstream()
rcs2, s2 = mkstream()
A = bytes([0x11] * 64)
B = bytes([0xAA] * 64)
A_buf = (ct.c_uint8 * 64)(*A)
B_buf = (ct.c_uint8 * 64)(*B)
copy_async(d1, A_buf, 64, H2D, s1)
copy_async(d2, B_buf, 64, H2D, s2)
r_sync1 = sync(s1)                           # drains only s1
_, back2_pre = d2h(d2, 64)                    # s2 NOT drained -> not B
r_sync2 = sync(s2)                           # now drains s2
_, back2_post = d2h(d2, 64)                   # now B
_, back1 = d2h(d1, 64)
check("stream sync does not drain other streams' pending ops",
      r_sync1 == SUCCESS and r_sync2 == SUCCESS and back2_pre != B and back2_post == B and back1 == A,
      f"sync1={ename(r_sync1)} sync2={ename(r_sync2)} d2_pre==B:{back2_pre == B} d2_post==B:{back2_post == B} d1==A:{back1 == A}")
destroy(s1)
destroy(s2)
free(d1)
free(d2)

print()
print(f"=== {sum(results)}/{len(results)} checks passed ===")
