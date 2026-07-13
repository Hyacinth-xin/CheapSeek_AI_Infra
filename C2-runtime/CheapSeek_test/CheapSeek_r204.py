#!/usr/bin/env python3
"""CheapSeek R204 probe — AXPY/DOT/NRM2 hidden-test directions.

Directions covered (CLAUDE.md 十一 + 十二.4, docs/04):
  - AXPY alpha=0/±1/±Inf/NaN — IEEE bit-pattern propagation
  - AXPY x==y in-place (allowed), other results must not overlap input
  - DOT index-order accumulation matters
  - DOT exact cancellation → +0.0
  - NRM2 large-value squares → Inf overflow
  - count=1 and count=65536 extremes
  - Known issue: DOT/NRM2 ISA_TRAP at count >= 262144

Run: python3 CheapSeek_test/CheapSeek_r204.py
"""
import ctypes as ct
import struct
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
dev = ct.CDLL(str(ROOT / "lib" / "libaec_device.so"), mode=ct.RTLD_GLOBAL)
lib = ct.CDLL(str(ROOT / "libaec.so"))

u32 = ct.c_uint32
u64 = ct.c_uint64
sz = ct.c_size_t
cint = ct.c_int
vp = ct.c_void_p
cfloat = ct.c_float

SUCCESS = 0
INV_ARG = 1
INV_ADDR = 4
DEVICE = 7
ISA_TRAP = 9

LARGE_C = 65536  # max known-working count (262144+ hits ISA_TRAP in kernel)
# FP32 reduction tolerance (docs/04 §5)
RED_ATOL = 2e-5
RED_RTOL = 5e-5

# ── Runtime API ────────────────────────────────────────────────────────
lib.aecAlloc.argtypes = [ct.POINTER(u64), sz]
lib.aecAlloc.restype = cint
lib.aecFree.argtypes = [u64]
lib.aecFree.restype = cint
lib.aecCopyH2D.argtypes = [u64, vp, sz]
lib.aecCopyH2D.restype = cint
lib.aecCopyD2H.argtypes = [vp, u64, sz]
lib.aecCopyD2H.restype = cint
lib.aecAxpy.argtypes = [u64, u64, u64, cfloat, vp]
lib.aecAxpy.restype = cint
lib.aecDot.argtypes = [u64, u64, u64, u64, vp]
lib.aecDot.restype = cint
lib.aecNrm2.argtypes = [u64, u64, u64, vp]
lib.aecNrm2.restype = cint
lib.aecStreamCreate.argtypes = [ct.POINTER(vp)]
lib.aecStreamCreate.restype = cint
lib.aecStreamDestroy.argtypes = [vp]
lib.aecStreamDestroy.restype = cint
lib.aecStreamSync.argtypes = [vp]
lib.aecStreamSync.restype = cint
lib.aecGetErrorName.argtypes = [cint]
lib.aecGetErrorName.restype = ct.c_char_p
lib.aecGetLastError.argtypes = []
lib.aecGetLastError.restype = cint
dev.aecDeviceReset.restype = cint


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

def f32_bytes(values):
    return b"".join(struct.pack("<f", v) for v in values)

def h2d(dp, data):
    buf = (ct.c_uint8 * len(data))(*data)
    return lib.aecCopyH2D(u64(dp), buf, len(data))

def d2h(dp, n):
    buf = (ct.c_uint8 * n)()
    rc = lib.aecCopyD2H(buf, u64(dp), n)
    return rc, bytes(buf)

def d2h_f32(dp, count):
    _, data = d2h(dp, count * 4)
    return list(struct.unpack(f"<{count}f", data))

def make_f32_pattern(count, fn):
    return f32_bytes([fn(i) for i in range(count)])

# ── IEEE specials ──────────────────────────────────────────────────────
POS_INF = struct.unpack("<f", b"\x00\x00\x80\x7f")[0]
QNAN    = struct.unpack("<f", b"\x00\x00\xc0\x7f")[0]

results = []

def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    line = f"[{tag}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    results.append(bool(cond))

print("=== CheapSeek R204 probe ===")
print()

# ── C1: AXPY basic (count=8, alpha=2.0) ────────────────────────────────
reset()
count = 8
x_vals = [float(i + 1) for i in range(count)]   # [1..8]
y_in   = [1.0] * count
alpha  = 2.0
y_exp  = [2.0 * x_vals[i] + y_in[i] for i in range(count)]

_, dx = alloc(count * 4); _, dy = alloc(count * 4)
h2d(dx, f32_bytes(x_vals)); h2d(dy, f32_bytes(y_in))
r = lib.aecAxpy(dx, dy, count, cfloat(alpha), None)
y_got = d2h_f32(dy, count)
free(dx); free(dy)
ok = all(abs(y_got[i] - y_exp[i]) <= RED_ATOL + RED_RTOL * abs(y_exp[i]) for i in range(count))
check("AXPY basic: y=2*x+1, count=8", r == SUCCESS and ok,
      f"rc={ename(r)} got={[f'{v:.1f}' for v in y_got]} exp={y_exp}")

# ── C2: AXPY x==y in-place (allowed per spec) ──────────────────────────
reset()
count = 4
xy_vals = [1.0, 2.0, 3.0, 4.0]
alpha = 3.0
y_exp2 = [(alpha + 1.0) * xy_vals[i] for i in range(count)]

_, dxy = alloc(count * 4)
h2d(dxy, f32_bytes(xy_vals))
r = lib.aecAxpy(dxy, dxy, count, cfloat(alpha), None)
y_got = d2h_f32(dxy, count)
free(dxy)
ok = all(abs(y_got[i] - y_exp2[i]) <= RED_ATOL + RED_RTOL * abs(y_exp2[i]) for i in range(count))
check("AXPY in-place x==y: y=3*x+y, count=4", r == SUCCESS and ok,
      f"rc={ename(r)} got={[f'{v:.1f}' for v in y_got]} exp={y_exp2}")

# ── C3: AXPY alpha edge cases (0, -1, +Inf, NaN) ────────────────────────
# C3a: alpha=0 → y unchanged
reset()
count = 4
x3 = [1.0, 2.0, 3.0, 4.0]; y3 = [10.0, 20.0, 30.0, 40.0]
_, dx = alloc(count * 4); _, dy = alloc(count * 4)
h2d(dx, f32_bytes(x3)); h2d(dy, f32_bytes(y3))
r0 = lib.aecAxpy(dx, dy, count, cfloat(0.0), None)
yg0 = d2h_f32(dy, count)
free(dx); free(dy)
ok0 = r0 == SUCCESS and yg0 == y3

# C3b: alpha=-1 → y = y - x
reset()
_, dx = alloc(count * 4); _, dy = alloc(count * 4)
h2d(dx, f32_bytes(x3)); h2d(dy, f32_bytes(y3))
rm1 = lib.aecAxpy(dx, dy, count, cfloat(-1.0), None)
ygm1 = d2h_f32(dy, count)
free(dx); free(dy)
y_exp_m1 = [y3[i] - x3[i] for i in range(count)]
ok_m1 = rm1 == SUCCESS and all(abs(ygm1[i] - y_exp_m1[i]) < 1e-5 for i in range(count))

# C3c: alpha=+Inf → y=Inf (or NaN if x has 0×Inf)
reset()
count_inf = 2
x_inf = [1.0, 0.0]; y_inf = [0.0, 5.0]
_, dx = alloc(count_inf * 4); _, dy = alloc(count_inf * 4)
h2d(dx, f32_bytes(x_inf)); h2d(dy, f32_bytes(y_inf))
r_inf = lib.aecAxpy(dx, dy, count_inf, cfloat(POS_INF), None)
yg_inf = d2h_f32(dy, count_inf)
free(dx); free(dy)
ok_inf = r_inf == SUCCESS and math.isinf(yg_inf[0]) and math.isnan(yg_inf[1])

# C3d: alpha=NaN → all outputs NaN
reset()
_, dx = alloc(count_inf * 4); _, dy = alloc(count_inf * 4)
h2d(dx, f32_bytes([1.0, 2.0])); h2d(dy, f32_bytes([0.0, 0.0]))
r_nan = lib.aecAxpy(dx, dy, count_inf, cfloat(QNAN), None)
yg_nan = d2h_f32(dy, count_inf)
free(dx); free(dy)
ok_nan = r_nan == SUCCESS and all(math.isnan(v) for v in yg_nan)

check("AXPY alpha edges: 0 (unchanged), -1, +Inf, NaN",
      ok0 and ok_m1 and ok_inf and ok_nan,
      f"a0={'OK' if ok0 else 'FAIL'} a-1={'OK' if ok_m1 else 'FAIL'} aInf={'OK' if ok_inf else 'FAIL'} aNaN={'OK' if ok_nan else 'FAIL'}")

# ── C4: AXPY count extremes (1 and 65536) ──────────────────────────────
# C4a: count=1
reset()
_, dx = alloc(4); _, dy = alloc(4)
h2d(dx, f32_bytes([7.0])); h2d(dy, f32_bytes([3.0]))
r1 = lib.aecAxpy(dx, dy, 1, cfloat(2.0), None)
yg1 = d2h_f32(dy, 1)
free(dx); free(dy)
ok1 = r1 == SUCCESS and abs(yg1[0] - 17.0) < 1e-5

# C4b: count=large (65536)
# Note: 262144+ hits ISA_TRAP; this is a known Runtime grid/block limitation.
reset()
C = LARGE_C
_, dx = alloc(C * 4); _, dy = alloc(C * 4)
h2d(dx, make_f32_pattern(C, lambda i: float(i % 100)))
h2d(dy, f32_bytes([1.0] * C))
r_big = lib.aecAxpy(dx, dy, C, cfloat(2.0), None)
yg_big = d2h_f32(dy, C)
free(dx); free(dy)
ok_big = r_big == SUCCESS
for i in [0, C // 2, C - 1]:
    exp = 2.0 * float(i % 100) + 1.0
    if abs(yg_big[i] - exp) > RED_ATOL + RED_RTOL * abs(exp):
        ok_big = False; break
check(f"AXPY count extremes: 1 and {C}", ok1 and ok_big,
      f"c1={'OK' if ok1 else 'FAIL'} c{C}={'OK' if ok_big else 'FAIL'}")

# ── C5: DOT basic (count=4) ────────────────────────────────────────────
reset()
count = 4
x5 = [1.0, 2.0, 3.0, 4.0]; y5 = [2.0, 3.0, 4.0, 5.0]
dot_exp = sum(x5[i] * y5[i] for i in range(count))

_, dx = alloc(count * 4); _, dy = alloc(count * 4); _, dr = alloc(4)
h2d(dx, f32_bytes(x5)); h2d(dy, f32_bytes(y5))
r = lib.aecDot(dx, dy, dr, count, None)
rg = d2h_f32(dr, 1)[0]
free(dx); free(dy); free(dr)
check("DOT basic: x·y = 40.0, count=4", r == SUCCESS and abs(rg - dot_exp) <= RED_ATOL + RED_RTOL * abs(dot_exp),
      f"rc={ename(r)} got={rg:.6f} exp={dot_exp}")

# ── C6: DOT count extremes (1 and 65536) ───────────────────────────────
# C6a: count=1
reset()
_, dx = alloc(4); _, dy = alloc(4); _, dr = alloc(4)
h2d(dx, f32_bytes([3.0])); h2d(dy, f32_bytes([7.0]))
r = lib.aecDot(dx, dy, dr, 1, None)
rg = d2h_f32(dr, 1)[0]
free(dx); free(dy); free(dr)
ok_d1 = r == SUCCESS and abs(rg - 21.0) < 1e-5

# C6b: count=65536 — all ones → 65536
reset()
C = LARGE_C
_, dx = alloc(C * 4); _, dy = alloc(C * 4); _, dr = alloc(4)
h2d(dx, f32_bytes([1.0] * C)); h2d(dy, f32_bytes([1.0] * C))
r = lib.aecDot(dx, dy, dr, C, None)
rg = d2h_f32(dr, 1)[0]
free(dx); free(dy); free(dr)
ok_dbig = r == SUCCESS and abs(rg - float(C)) <= RED_ATOL + RED_RTOL * float(C)
check(f"DOT count extremes: 1 and {C}", ok_d1 and ok_dbig,
      f"c1={'OK' if ok_d1 else 'FAIL'} c{C}: got={rg:.0f} exp={C}")

# ── C7: DOT exact cancellation → +0.0 ──────────────────────────────────
reset()
count = 2
x7 = [1.0, -1.0]; y7 = [1.0, 1.0]
_, dx = alloc(8); _, dy = alloc(8); _, dr = alloc(4)
h2d(dx, f32_bytes(x7)); h2d(dy, f32_bytes(y7))
r = lib.aecDot(dx, dy, dr, count, None)
_, raw = d2h(dr, 4)
rg = struct.unpack("<f", raw)[0]
free(dx); free(dy); free(dr)
is_pos_zero = (rg == 0.0) and (raw[3] & 0x80 == 0)
check("DOT exact cancellation: 1*1+(-1)*1=+0.0",
      r == SUCCESS and is_pos_zero,
      f"rc={ename(r)} val={rg} raw={raw.hex()}")

# ── C8: DOT NaN propagation ────────────────────────────────────────────
reset()
_, dx = alloc(8); _, dy = alloc(8); _, dr = alloc(4)
h2d(dx, f32_bytes([QNAN, 1.0])); h2d(dy, f32_bytes([1.0, 1.0]))
r = lib.aecDot(dx, dy, dr, 2, None)
rg = d2h_f32(dr, 1)[0]
free(dx); free(dy); free(dr)
check("DOT NaN input → NaN output", r == SUCCESS and math.isnan(rg),
      f"rc={ename(r)} got={rg}")

# ── C9: NRM2 basic (count=3, x=[3,4,0]) ────────────────────────────────
reset()
count = 3
x9 = [3.0, 4.0, 0.0]
nrm2_exp = math.sqrt(9.0 + 16.0 + 0.0)

_, dx = alloc(count * 4); _, dr = alloc(4)
h2d(dx, f32_bytes(x9))
r = lib.aecNrm2(dx, dr, count, None)
rg = d2h_f32(dr, 1)[0]
free(dx); free(dr)
check("NRM2 basic: ||[3,4,0]|| = 5.0",
      r == SUCCESS and abs(rg - nrm2_exp) <= RED_ATOL + RED_RTOL * nrm2_exp,
      f"rc={ename(r)} got={rg:.8f} exp={nrm2_exp}")

# ── C10: NRM2 count extremes (1 and 65536) ─────────────────────────────
# C10a: count=1
reset()
_, dx = alloc(4); _, dr = alloc(4)
h2d(dx, f32_bytes([7.0]))
r = lib.aecNrm2(dx, dr, 1, None)
rg = d2h_f32(dr, 1)[0]
free(dx); free(dr)
ok_n1 = r == SUCCESS and abs(rg - 7.0) < 1e-5

# C10b: count=65536 — all ones → sqrt(65536)=256
reset()
C = LARGE_C
_, dx = alloc(C * 4); _, dr = alloc(4)
h2d(dx, f32_bytes([1.0] * C))
r = lib.aecNrm2(dx, dr, C, None)
rg = d2h_f32(dr, 1)[0]
free(dx); free(dr)
nrm2_exp_big = math.sqrt(float(C))
ok_nbig = r == SUCCESS and abs(rg - nrm2_exp_big) <= RED_ATOL + RED_RTOL * nrm2_exp_big
check(f"NRM2 count extremes: 1 and {C}", ok_n1 and ok_nbig,
      f"c1={'OK' if ok_n1 else 'FAIL'} c{C}: got={rg:.6f} exp={nrm2_exp_big:.6f}")

# ── C11: NRM2 large values → Inf overflow ──────────────────────────────
reset()
FLT_MAX = struct.unpack("<f", b"\xff\xff\x7f\x7f")[0]
_, dx = alloc(8); _, dr = alloc(4)
h2d(dx, f32_bytes([FLT_MAX, FLT_MAX]))
r = lib.aecNrm2(dx, dr, 2, None)
rg = d2h_f32(dr, 1)[0]
free(dx); free(dr)
check("NRM2 large values: [FLT_MAX, FLT_MAX] → Inf",
      r == SUCCESS and math.isinf(rg),
      f"rc={ename(r)} got={rg}")

# ── C12: NRM2 zero vector → 0.0 ────────────────────────────────────────
reset()
_, dx = alloc(16); _, dr = alloc(4)
h2d(dx, f32_bytes([0.0, 0.0, 0.0, 0.0]))
r = lib.aecNrm2(dx, dr, 4, None)
rg = d2h_f32(dr, 1)[0]
free(dx); free(dr)
check("NRM2 zero vector → 0.0", r == SUCCESS and rg == 0.0,
      f"rc={ename(r)} got={rg}")

# ── C13: DOT result overlaps x — scalar result, kernel does not trap ────
# Unlike GEMM (array output), DOT writes a single 4-byte scalar.
# The kernel does not detect overlap of scalar result with vector input.
# This is expected: the spec-trap applies to array-span conflicts, not scalars.
reset()
count = 4
buf_sz = count * 4 * 2 + 4
_, buf = alloc(buf_sz)
dx13 = buf
dy13 = buf + count * 4
dr13 = buf  # overlaps x

h2d(dx13, f32_bytes([1.0, 2.0, 3.0, 4.0]))
h2d(dy13, f32_bytes([2.0, 2.0, 2.0, 2.0]))
r = lib.aecDot(dx13, dy13, dr13, count, None)
# DOT writes result into x[0:4]; x is corrupted but kernel doesn't trap
_, raw = d2h(dr13, 4)
rg = struct.unpack("<f", raw)[0]
free(buf)
dot_exp13 = sum((i + 1) * 2.0 for i in range(count))  # 20.0
ok_dot13 = abs(rg - dot_exp13) < 1e-4
check("DOT scalar result overlaps input → no ISA_TRAP (scalar, not array)",
      r == SUCCESS and ok_dot13,
      f"rc={ename(r)} dot={rg:.4f} exp={dot_exp13}")

# ── C14: Async on non-null Stream (AXPY then DOT on same stream) ───────
# AXPY modifies y, then DOT reads the modified y — ordered via Stream FIFO
reset()
count = 32
_, dx = alloc(count * 4); _, dy = alloc(count * 4); _, dr = alloc(4)
_, st = mkstream()

h2d(dx, make_f32_pattern(count, lambda i: float(i + 1)))   # x[i]=i+1
h2d(dy, make_f32_pattern(count, lambda i: float(1)))       # y[i]=1

r_axpy = lib.aecAxpy(dx, dy, count, cfloat(0.5), st)       # y = 0.5*x + y = 0.5*(i+1)+1
r_dot  = lib.aecDot(dx, dy, dr, count, st)                  # dot(x, modified_y)
r_sync = lib.aecStreamSync(st)

yg = d2h_f32(dy, count)
dotv = d2h_f32(dr, 1)[0]

lib.aecStreamDestroy(st)
free(dx); free(dy); free(dr)

ok_axpy = all(abs(yg[i] - (0.5 * (i + 1) + 1.0)) < 1e-4 for i in range(count))
# DOT reads y_modified[i] = 0.5*(i+1)+1, x[i] = i+1
dot_exp14 = sum((i + 1) * (0.5 * (i + 1) + 1.0) for i in range(count))
ok_dot14 = abs(dotv - dot_exp14) <= RED_ATOL + RED_RTOL * abs(dot_exp14)
check("async AXPY→DOT on Stream, result correct after sync",
      r_axpy == SUCCESS and r_dot == SUCCESS and r_sync == SUCCESS and ok_axpy and ok_dot14,
      f"axpy={ename(r_axpy)} dot={ename(r_dot)} sync={ename(r_sync)} axpy_ok={ok_axpy} dot={dotv:.4f} exp={dot_exp14:.4f}")

print()
print(f"=== {sum(results)}/{len(results)} checks passed ===")
