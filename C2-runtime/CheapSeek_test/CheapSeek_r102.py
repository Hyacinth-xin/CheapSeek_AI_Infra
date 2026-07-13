#!/usr/bin/env python3
"""CheapSeek R102 probe — allocation/free hidden-test directions.

Directions covered (CLAUDE.md 十一 + 十二.4):
  - Free(0) -> INV_ARG vs interior/stale/unknown -> INV_ADDR (error-code distinction)
  - interior pointer boundaries (base±1, base+size, base+size-1)
  - double-free / stale pointer -> INV_ADDR
  - deterministic lowest-address first-fit: same-size reuse returns same offset
  - free-block coalescing (free A,C,B then big alloc reuses coalesced block)
  - strict OOM: full 64MiB -> OOM (first 64 bytes reserved); 64MiB-64 -> SUCCESS; 64MiB-63 -> OOM
  - Alloc(0) behavior
  - aecFree drains pending async stream work before freeing (cross R105)
  - concurrent alloc: N threads get distinct non-zero offsets (no double-alloc race)

Run: python3 test/CheapSeek_r102.py
"""
import ctypes as ct
import threading
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
INV_ADDR = 4

H2D = 1
MEM = 64 * 1024 * 1024

lib.aecAlloc.argtypes = [ct.POINTER(u64), sz]
lib.aecAlloc.restype = cint
lib.aecFree.argtypes = [u64]
lib.aecFree.restype = cint
lib.aecGetErrorName.argtypes = [cint]
lib.aecGetErrorName.restype = ct.c_char_p
lib.aecCopyAsync.argtypes = [u64, vp, sz, cint, vp]
lib.aecCopyAsync.restype = cint
lib.aecStreamCreate.argtypes = [ct.POINTER(vp)]
lib.aecStreamCreate.restype = cint
lib.aecStreamSync.argtypes = [vp]
lib.aecStreamSync.restype = cint
lib.aecStreamDestroy.argtypes = [vp]
lib.aecStreamDestroy.restype = cint
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


results = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    line = f"[{tag}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    results.append(bool(cond))


print("=== CheapSeek R102 probe ===")
print()

# C1: basic alloc + 64-byte alignment (first 64 bytes reserved -> first alloc at 64)
reset()
rc, p = alloc(1)
check("Alloc(1) -> SUCCESS, offset>=64 & 64-aligned",
      rc == SUCCESS and p >= 64 and p % 64 == 0, f"rc={ename(rc)} ptr={p}")

# C2: Alloc(NULL out) -> INV_ARG
reset()
rc = lib.aecAlloc(None, 64)
check("Alloc(NULL out) -> INVALID_ARGUMENT", rc == INV_ARG, f"rc={ename(rc)}")

# C3: Alloc(0) defined behavior (success+freeable OR error)
reset()
rc, p = alloc(0)
if rc == SUCCESS:
    rc2 = free(p)
    check("Alloc(0): SUCCESS -> ptr is freeable", rc2 == SUCCESS,
          f"ptr={p} freerc={ename(rc2)}")
else:
    check("Alloc(0) rejected with error", rc in (INV_ARG, OOM),
          f"rc={ename(rc)}")

# C4: Free(0) -> INV_ARG (NOT INV_ADDR)  [error-code distinction]
reset()
rc = free(0)
check("Free(0) -> INVALID_ARGUMENT (not INVALID_ADDRESS)",
      rc == INV_ARG, f"rc={ename(rc)}")

# C5: interior / boundary pointers -> INV_ADDR
reset()
rc, base = alloc(128)           # base=64, spans [64,192)
S = 128
cases = [
    ("Free(base+1) interior", base + 1),
    ("Free(base-1) before", base - 1),
    ("Free(base+S-1) last byte interior", base + S - 1),
    ("Free(base+S) one-past-end", base + S),
]
bad = []
for label, ptr in cases:
    r = free(ptr)
    if r != INV_ADDR:
        bad.append(f"{label}={ename(r)}")
check("interior/boundary pointers all -> INVALID_ADDRESS",
      not bad, "; ".join(bad) if bad else f"base={base} all INV_ADDR")

# C6: double-free / stale pointer -> INV_ADDR
reset()
rc, p = alloc(64)
r1 = free(p)
r2 = free(p)                    # double free / stale
check("double-free / stale -> INVALID_ADDRESS",
      r1 == SUCCESS and r2 == INV_ADDR, f"free1={ename(r1)} free2={ename(r2)}")

# C7: deterministic first-fit: same-size reuse returns same offset
reset()
rc, a = alloc(64)
r1 = free(a)
rc, b = alloc(64)               # same size, no other alloc in between
check("same-size reuse returns same offset (first-fit)",
      rc == SUCCESS and a == b, f"a={a} b={b}")

# C8: free-block coalescing (free A,C,B -> big alloc reuses coalesced block)
reset()
_, a = alloc(64)
_, b = alloc(64)
_, c = alloc(64)
block = b - a
contiguous = (b - a == block) and (c - b == block)
free(a)
free(c)
free(b)                         # B coalesces with both neighbors
big = 3 * block
rc, p = alloc(big)
check("coalescing: free A,C,B then big alloc reuses A's block",
      contiguous and rc == SUCCESS and p == a,
      f"contiguous={contiguous} block={block} big={big} got={p} exp={a}")

# C9: strict OOM boundary (first 64 bytes reserved)
reset()
rc, _ = alloc(MEM)
check("Alloc(64MiB) -> OOM (first 64 bytes reserved)", rc == OOM, f"rc={ename(rc)}")
reset()
rc, p = alloc(MEM - 64)
check("Alloc(64MiB-64) -> SUCCESS (exactly fits)", rc == SUCCESS, f"rc={ename(rc)} ptr={p}")
reset()
rc, _ = alloc(MEM - 63)
check("Alloc(64MiB-63) -> OOM (one byte over)", rc == OOM, f"rc={ename(rc)}")

# C10: aecFree drains pending async stream work before freeing (cross R105)
reset()
rc, d = alloc(1024)
st = vp()
lib.aecStreamCreate(ct.byref(st))
host = (ct.c_char * 1024)()
r_async = lib.aecCopyAsync(d, host, 1024, H2D, st)   # enqueue, returns immediately
r_free = free(d)                                       # must drain stream first
r_sync = lib.aecStreamSync(st)                         # copy completed during drain
lib.aecStreamDestroy(st)
check("aecFree drains pending async work before freeing",
      r_async == SUCCESS and r_free == SUCCESS and r_sync == SUCCESS,
      f"async={ename(r_async)} free={ename(r_free)} sync={ename(r_sync)}")

# C11: concurrent alloc -> N threads get distinct non-zero offsets
N = 8
b_start = threading.Barrier(N + 1)
b_alloced = threading.Barrier(N + 1)
b_free = threading.Barrier(N + 1)
ptrs = [None] * N
arc = [None] * N
frc = [None] * N


def worker(tid):
    b_start.wait()
    p = u64()
    arc[tid] = lib.aecAlloc(ct.byref(p), 128)
    ptrs[tid] = p.value
    b_alloced.wait()
    b_free.wait()
    frc[tid] = lib.aecFree(u64(ptrs[tid]))


threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
for t in threads:
    t.start()
b_start.wait()
b_alloced.wait()                 # all alloced; workers now blocked at b_free
unique = len(set(ptrs)) == N and all(p != 0 for p in ptrs)
all_ok = all(r == SUCCESS for r in arc)
b_free.wait()                    # release workers to free
for t in threads:
    t.join()
all_freed = all(r == SUCCESS for r in frc)
check(f"concurrent alloc: {N} threads distinct non-zero ptrs, all free OK",
      unique and all_ok and all_freed,
      f"unique={unique} alloc_ok={all_ok} free_ok={all_freed} ptrs={ptrs}")

print()
print(f"=== {sum(results)}/{len(results)} checks passed ===")
