#!/usr/bin/env python3
"""CheapSeek R103 probe — synchronous copy hidden-test directions.

Directions covered (CLAUDE.md 十一 + 十二.4):
  - span crosses TWO adjacent live allocations -> INV_ADDR (global-range mutant would miss)
  - copy to/from stale (freed) pointer -> INV_ADDR (both H2D and D2H)
  - device ptr=0 -> INV_ADDR, distinct from host None -> INV_ARG (error-code distinction)
  - offset+size exactly fills allocation end -> SUCCESS (boundary)
  - offset+size crosses allocation end -> INV_ADDR (D+12/8, D+15/2, D+16/1)
  - huge bytes -> offset+size overflow detected -> INV_ADDR
  - host None / bytes 0 -> INV_ARG

Run: python3 test/CheapSeek_r103.py
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
INV_ADDR = 4

lib.aecAlloc.argtypes = [ct.POINTER(u64), sz]
lib.aecAlloc.restype = cint
lib.aecFree.argtypes = [u64]
lib.aecFree.restype = cint
lib.aecCopyH2D.argtypes = [u64, vp, sz]
lib.aecCopyH2D.restype = cint
lib.aecCopyD2H.argtypes = [vp, u64, sz]
lib.aecCopyD2H.restype = cint
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


def h2d(dp, data):
    buf = (ct.c_uint8 * len(data))(*data)
    return lib.aecCopyH2D(u64(dp), buf, len(data))


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


print("=== CheapSeek R103 probe ===")
print()

# C1: H2D/D2H round-trip preserves data
reset()
rc, d = alloc(16)
data = bytes(range(16))
r1 = h2d(d, data)
r2, back = d2h(d, 16)
check("H2D/D2H round-trip preserves data",
      r1 == SUCCESS and r2 == SUCCESS and back == data,
      f"h2d={ename(r1)} d2h={ename(r2)} match={back == data}")

# C2: host None -> INV_ARG (both directions)
reset()
rc, d = alloc(16)
r_h2d = lib.aecCopyH2D(u64(d), None, 16)
r_d2h = lib.aecCopyD2H(None, u64(d), 16)
check("host None -> INVALID_ARGUMENT (H2D src & D2H dst)",
      r_h2d == INV_ARG and r_d2h == INV_ARG,
      f"h2d={ename(r_h2d)} d2h={ename(r_d2h)}")

# C3: bytes 0 -> INV_ARG
reset()
rc, d = alloc(16)
buf = (ct.c_uint8 * 16)()
r = lib.aecCopyH2D(u64(d), buf, 0)
check("bytes=0 -> INVALID_ARGUMENT", r == INV_ARG, f"rc={ename(r)}")

# C4: span crosses allocation end -> INV_ADDR  (alloc(16) -> [d, d+16))
reset()
rc, d = alloc(16)
host = (ct.c_uint8 * 16)()
bad = []
for off, n in [(12, 8), (15, 2), (16, 1)]:
    r = lib.aecCopyH2D(u64(d + off), host, n)
    if r != INV_ADDR:
        bad.append(f"D+{off}/{n}={ename(r)}")
check("span crosses allocation end -> INVALID_ADDRESS",
      not bad, "; ".join(bad) if bad else "D+12/8, D+15/2, D+16/1 all INV_ADDR")

# C5: span exactly fills allocation end -> SUCCESS (boundary)
reset()
rc, d = alloc(16)
host = (ct.c_uint8 * 16)()
r_full = lib.aecCopyH2D(u64(d), host, 16)        # [d, d+16) exact
r_half = lib.aecCopyH2D(u64(d + 8), host, 8)     # [d+8, d+16) exact to end
check("span exactly fills allocation end -> SUCCESS",
      r_full == SUCCESS and r_half == SUCCESS,
      f"full(16)={ename(r_full)} half(8 from +8)={ename(r_half)}")

# C6: span crosses TWO adjacent live allocations -> INV_ADDR  (key: global-range mutant would miss)
reset()
_, a = alloc(64)
_, b = alloc(64)
adjacent = (b - a == 64)
host = (ct.c_uint8 * 16)()
r = lib.aecCopyH2D(u64(a + 60), host, 8)         # [a+60, a+68) crosses a's end (a+64) into b
check("span crosses two adjacent live allocations -> INVALID_ADDRESS",
      adjacent and r == INV_ADDR,
      f"adjacent={adjacent} a={a} b={b} copy(a+60,8)={ename(r)}")

# C7: copy to/from stale (freed) pointer -> INV_ADDR (both directions)
reset()
rc, d = alloc(16)
free(d)
host = (ct.c_uint8 * 16)()
r_h2d = lib.aecCopyH2D(u64(d), host, 16)         # write to freed
r_d2h = lib.aecCopyD2H(host, u64(d), 16)         # read from freed
check("copy to/from stale (freed) pointer -> INVALID_ADDRESS",
      r_h2d == INV_ADDR and r_d2h == INV_ADDR,
      f"h2d(stale)={ename(r_h2d)} d2h(stale)={ename(r_d2h)}")

# C8: device ptr=0 -> INV_ADDR, distinct from host None -> INV_ARG
reset()
rc, d = alloc(16)
host = (ct.c_uint8 * 16)()
r_dev0_h2d = lib.aecCopyH2D(u64(0), host, 16)    # dev=0, host valid
r_dev0_d2h = lib.aecCopyD2H(host, u64(0), 16)
r_hostnone = lib.aecCopyH2D(u64(d), None, 16)    # host None
check("device ptr=0 -> INV_ADDR (distinct from host None -> INV_ARG)",
      r_dev0_h2d == INV_ADDR and r_dev0_d2h == INV_ADDR and r_hostnone == INV_ARG,
      f"dev0_h2d={ename(r_dev0_h2d)} dev0_d2h={ename(r_dev0_d2h)} hostnone={ename(r_hostnone)}")

# C9: huge bytes -> offset+size overflow detected -> INV_ADDR
#   alloc(16) at d=64; bytes = 0xFFFFFFFFFFFFFFFF -> end = 64 + (2^64-1) = 63 < 64 (wrap)
#   without overflow check, end=63 <= base+size would WRONGLY pass span -> copy ~2^64 bytes (crash)
reset()
rc, d = alloc(16)
host = (ct.c_uint8 * 16)()
r = lib.aecCopyH2D(u64(d), host, 0xFFFFFFFFFFFFFFFF)
check("huge bytes -> offset+size overflow detected -> INVALID_ADDRESS",
      r == INV_ADDR, f"d={d} bytes=0xFFFFFFFFFFFFFFFF rc={ename(r)}")

print()
print(f"=== {sum(results)}/{len(results)} checks passed ===")
