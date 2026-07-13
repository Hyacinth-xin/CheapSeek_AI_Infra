#!/usr/bin/env python3
"""CheapSeek R201 probe — FP32 / INT32 GEMM hidden-test directions.

Directions covered (CLAUDE.md 十一 + 十二.4):
  - M/N/K = 1 minimum matrix (both dtypes)
  - M/N/K = 256 maximum matrix (both dtypes)
  - A/B/C span overlap -> ISA_TRAP (device detects, runtime propagates)
  - INT32 accumulator saturation (positive / negative overflow)
  - non-16-multiple shapes (17/31/255) — tiling boundary
  - edge shapes: 1×N×K, M×1×K, M×N×1 (non-square extremes)
  - async GEMM on non-null Stream
  - FP32 correctness against numeric_public.json golden data

Run: python3 CheapSeek_test/CheapSeek_r201.py
"""
import ctypes as ct
import struct
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

INT32_MAX = 2147483647
INT32_MIN = -2147483648
FP32_ATOL = 1e-5
FP32_RTOL = 2e-5


# ── Runtime API signatures ──────────────────────────────────────────────
lib.aecAlloc.argtypes = [ct.POINTER(u64), sz]
lib.aecAlloc.restype = cint
lib.aecFree.argtypes = [u64]
lib.aecFree.restype = cint
lib.aecCopyH2D.argtypes = [u64, vp, sz]
lib.aecCopyH2D.restype = cint
lib.aecCopyD2H.argtypes = [vp, u64, sz]
lib.aecCopyD2H.restype = cint
lib.aecMatmulF32.argtypes = [u64, u64, u64, u32, u32, u32, vp]
lib.aecMatmulF32.restype = cint
lib.aecMatmulI32.argtypes = [u64, u64, u64, u32, u32, u32, vp]
lib.aecMatmulI32.restype = cint
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
    """Pack list of floats into little-endian bytes."""
    return b"".join(struct.pack("<f", v) for v in values)


def i32_bytes(values):
    """Pack list of signed int32 values into little-endian bytes."""
    return b"".join(struct.pack("<i", v) for v in values)


def unpack_f32(data):
    """Unpack bytes into list of floats."""
    return list(struct.unpack(f"<{len(data)//4}f", data))


def unpack_i32(data):
    """Unpack bytes into list of signed int32."""
    return list(struct.unpack(f"<{len(data)//4}i", data))


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


print("=== CheapSeek R201 probe ===")
print()

# ── C1: FP32 GEMM golden correctness (2×3×3 from numeric_public.json) ──
reset()
M, N, K = 2, 3, 3
A_vals = [1.0, -2.0, 0.5, 3.0, 4.0, -1.0]       # M×K = 2×3 row-major
B_vals = [2.0, 1.0, -1.0, 0.5, 3.0, 2.0, -2.0, 1.5, 0.25]  # K×N = 3×3 row-major
C_gold = [0.0, -4.25, -4.875, 10.0, 13.5, 4.75]   # M×N = 2×3

_, dA = alloc(M * K * 4)
_, dB = alloc(K * N * 4)
_, dC = alloc(M * N * 4)

h2d(dA, f32_bytes(A_vals))
h2d(dB, f32_bytes(B_vals))

r_gemm = lib.aecMatmulF32(dA, dB, dC, M, N, K, None)
_, cdata = d2h(dC, M * N * 4)
cvals = unpack_f32(cdata)

ok = True
for i, (got, gold) in enumerate(zip(cvals, C_gold)):
    if abs(got - gold) > FP32_ATOL + FP32_RTOL * abs(gold):
        ok = False
        break

free(dA)
free(dB)
free(dC)
check("FP32 GEMM 2×3×3 golden data (atol=1e-5 rtol=2e-5)",
      r_gemm == SUCCESS and ok,
      f"gemm={ename(r_gemm)} cvals={[f'{v:.6f}' for v in cvals]} gold={C_gold}")

# ── C2: INT32 GEMM correctness (2×2×3 hand-computed) ───────────────────
reset()
M, N, K = 2, 2, 3
A_i32 = [1, 2, 3, 4, 5, 6]                       # M×K
B_i32 = [7, 8, 9, 10, 11, 12]                     # K×N
# C = A×B (row-major): [[1*7+2*9+3*11, 1*8+2*10+3*12],
#                        [4*7+5*9+6*11, 4*8+5*10+6*12]]
#   = [[58, 64], [139, 154]]
C_i32_exp = [58, 64, 139, 154]

_, dA = alloc(M * K * 4)
_, dB = alloc(K * N * 4)
_, dC = alloc(M * N * 4)

h2d(dA, i32_bytes(A_i32))
h2d(dB, i32_bytes(B_i32))

r_gemm = lib.aecMatmulI32(dA, dB, dC, M, N, K, None)
_, cdata = d2h(dC, M * N * 4)
cvals = unpack_i32(cdata)

free(dA)
free(dB)
free(dC)
check("INT32 GEMM 2×2×3 bit-exact",
      r_gemm == SUCCESS and cvals == C_i32_exp,
      f"gemm={ename(r_gemm)} got={cvals} exp={C_i32_exp}")

# ── C3: M=N=K=1 minimum FP32 ───────────────────────────────────────────
reset()
M, N, K = 1, 1, 1
A1 = [3.5]
B1 = [-2.0]
C1_exp = [-7.0]

_, dA = alloc(M * K * 4)
_, dB = alloc(K * N * 4)
_, dC = alloc(M * N * 4)

h2d(dA, f32_bytes(A1))
h2d(dB, f32_bytes(B1))

r_gemm = lib.aecMatmulF32(dA, dB, dC, M, N, K, None)
_, cdata = d2h(dC, 4)
cvals = unpack_f32(cdata)

free(dA)
free(dB)
free(dC)
ok = abs(cvals[0] - C1_exp[0]) <= FP32_ATOL + FP32_RTOL * abs(C1_exp[0])
check("FP32 GEMM M=N=K=1 minimum",
      r_gemm == SUCCESS and ok,
      f"gemm={ename(r_gemm)} got={cvals[0]:.6f} exp={C1_exp[0]}")

# ── C4: M=N=K=1 minimum INT32 ──────────────────────────────────────────
reset()
M, N, K = 1, 1, 1
A1i = [42]
B1i = [-3]
C1i_exp = [-126]

_, dA = alloc(M * K * 4)
_, dB = alloc(K * N * 4)
_, dC = alloc(M * N * 4)

h2d(dA, i32_bytes(A1i))
h2d(dB, i32_bytes(B1i))

r_gemm = lib.aecMatmulI32(dA, dB, dC, M, N, K, None)
_, cdata = d2h(dC, 4)
cvals = unpack_i32(cdata)

free(dA)
free(dB)
free(dC)
check("INT32 GEMM M=N=K=1 minimum",
      r_gemm == SUCCESS and cvals == C1i_exp,
      f"gemm={ename(r_gemm)} got={cvals} exp={C1i_exp}")

# ── C5: M=N=K=256 maximum FP32 ─────────────────────────────────────────
# Identity-like: A * I = A, so use B = identity (first element of each row = 1.0)
# Actually, just check that 256×256×256 succeeds without error.
reset()
M = N = K = 256
A_big = [float(i % 7) for i in range(M * K)]
B_big = [float((i + 3) % 11 + 1) for i in range(K * N)]

_, dA = alloc(M * K * 4)
_, dB = alloc(K * N * 4)
_, dC = alloc(M * N * 4)

h2d(dA, f32_bytes(A_big))
h2d(dB, f32_bytes(B_big))

r_gemm = lib.aecMatmulF32(dA, dB, dC, M, N, K, None)
_, cdata = d2h(dC, M * N * 4)
cvals = unpack_f32(cdata)

free(dA)
free(dB)
free(dC)
# Spot-check a few entries against row-column dot product
ok256 = True
for r in [0, 127, 255]:
    for c_ in [0, 127, 255]:
        s = sum(A_big[r * K + k] * B_big[k * N + c_] for k in range(K))
        if abs(cvals[r * N + c_] - s) > FP32_ATOL + FP32_RTOL * abs(s):
            ok256 = False
            break
check("FP32 GEMM M=N=K=256 maximum, spot-check 9 cells",
      r_gemm == SUCCESS and ok256,
      f"gemm={ename(r_gemm)} spot_ok={ok256}")

# ── C6: M=N=K=256 maximum INT32 ────────────────────────────────────────
reset()
M = N = K = 256
A_256i = [i % 13 - 6 for i in range(M * K)]            # small values, no overflow
B_256i = [(i * 3) % 17 - 8 for i in range(K * N)]

_, dA = alloc(M * K * 4)
_, dB = alloc(K * N * 4)
_, dC = alloc(M * N * 4)

h2d(dA, i32_bytes(A_256i))
h2d(dB, i32_bytes(B_256i))

r_gemm = lib.aecMatmulI32(dA, dB, dC, M, N, K, None)
_, cdata = d2h(dC, M * N * 4)
cvals = unpack_i32(cdata)

free(dA)
free(dB)
free(dC)
ok256i = True
for r in [0, 127, 255]:
    for c_ in [0, 127, 255]:
        s = sum(A_256i[r * K + k] * B_256i[k * N + c_] for k in range(K))
        if cvals[r * N + c_] != s:
            ok256i = False
            break
check("INT32 GEMM M=N=K=256 maximum, spot-check 9 cells",
      r_gemm == SUCCESS and ok256i,
      f"gemm={ename(r_gemm)} spot_ok={ok256i}")

# ── C7: INT32 positive saturation ──────────────────────────────────────
# M=1,N=1,K=2: A=[2e9, 2e9], B=[2e9, 2e9]
# Exact = 2 * 4e18 = 8e18 > INT32_MAX → saturate to INT32_MAX
reset()
M, N, K = 1, 1, 2
big = 2_000_000_000
A_sat = [big, big]
B_sat = [big, big]

_, dA = alloc(M * K * 4)
_, dB = alloc(K * N * 4)
_, dC = alloc(M * N * 4)

h2d(dA, i32_bytes(A_sat))
h2d(dB, i32_bytes(B_sat))

r_gemm = lib.aecMatmulI32(dA, dB, dC, M, N, K, None)
_, cdata = d2h(dC, 4)
cvals = unpack_i32(cdata)

free(dA)
free(dB)
free(dC)
exact = big * big + big * big  # 8,000,000,000,000,000,000
check("INT32 positive saturation: overflow -> INT32_MAX",
      r_gemm == SUCCESS and cvals[0] == INT32_MAX,
      f"gemm={ename(r_gemm)} exact={exact} got={cvals[0]} exp={INT32_MAX}")

# ── C8: INT32 negative saturation ──────────────────────────────────────
# M=1,N=1,K=2: A=[-2e9, -2e9], B=[2e9, 2e9]
# Exact = -8e18 < INT32_MIN → saturate to INT32_MIN
reset()
M, N, K = 1, 1, 2
A_neg = [-big, -big]
B_neg = [big, big]

_, dA = alloc(M * K * 4)
_, dB = alloc(K * N * 4)
_, dC = alloc(M * N * 4)

h2d(dA, i32_bytes(A_neg))
h2d(dB, i32_bytes(B_neg))

r_gemm = lib.aecMatmulI32(dA, dB, dC, M, N, K, None)
_, cdata = d2h(dC, 4)
cvals = unpack_i32(cdata)

free(dA)
free(dB)
free(dC)
check("INT32 negative saturation: overflow -> INT32_MIN",
      r_gemm == SUCCESS and cvals[0] == INT32_MIN,
      f"gemm={ename(r_gemm)} exact={-exact} got={cvals[0]} exp={INT32_MIN}")

# ── C9: INT32 near-boundary (no overflow, exact match) ─────────────────
# M=1,N=1,K=2: A=[46340, 46341], B=[46340, 46340]
# Exact = 46340*46340 + 46341*46340 = 46340*(46340+46341) = 46340*92681 = 4,294,836,340
# This is > INT32_MAX, so also saturates. Let's use smaller values.
# A=[30000, 30000], B=[30000, 30000]
# Exact = 2 * 9e8 = 1.8e9 < INT32_MAX → no saturation
reset()
M, N, K = 1, 1, 2
a_bound = 30000
A_bnd = [a_bound, a_bound]
B_bnd = [a_bound, a_bound]

_, dA = alloc(M * K * 4)
_, dB = alloc(K * N * 4)
_, dC = alloc(M * N * 4)

h2d(dA, i32_bytes(A_bnd))
h2d(dB, i32_bytes(B_bnd))

r_gemm = lib.aecMatmulI32(dA, dB, dC, M, N, K, None)
_, cdata = d2h(dC, 4)
cvals = unpack_i32(cdata)

free(dA)
free(dB)
free(dC)
exp_bound = a_bound * a_bound * K  # 1,800,000,000
check("INT32 near-boundary (no overflow, exact match)",
      r_gemm == SUCCESS and cvals[0] == exp_bound,
      f"gemm={ename(r_gemm)} got={cvals[0]} exp={exp_bound}")

# ── C10: Non-16-multiple shapes (17×31×255) ────────────────────────────
# These shapes stress tiling boundaries where M/N/K not divisible by 4/8/16.
reset()
M, N, K = 17, 31, 255
A17 = [float((i * 7 + 3) % 23 - 11) for i in range(M * K)]
B17 = [float((i * 5 + 13) % 19 - 9) for i in range(K * N)]

_, dA = alloc(M * K * 4)
_, dB = alloc(K * N * 4)
_, dC = alloc(M * N * 4)

h2d(dA, f32_bytes(A17))
h2d(dB, f32_bytes(B17))

r_gemm = lib.aecMatmulF32(dA, dB, dC, M, N, K, None)
_, cdata = d2h(dC, M * N * 4)
cvals = unpack_f32(cdata)

free(dA)
free(dB)
free(dC)
# Spot-check a few cells
ok17 = True
bad_cells = []
for r in [0, 8, 16]:
    for c_ in [0, 15, 30]:
        s = sum(A17[r * K + k] * B17[k * N + c_] for k in range(K))
        if abs(cvals[r * N + c_] - s) > FP32_ATOL + FP32_RTOL * abs(s):
            ok17 = False
            bad_cells.append((r, c_, cvals[r * N + c_], s))
check("FP32 GEMM 17×31×255 non-16-multiple shapes",
      r_gemm == SUCCESS and ok17,
      f"gemm={ename(r_gemm)} bad={bad_cells}" if bad_cells else
      f"gemm={ename(r_gemm)} spot_ok={ok17}")

# ── C11: Edge shapes (1×N×K, M×1×K, M×N×1) ───────────────────────────
# C11a: M=1, N=64, K=32 — single-row output
reset()
M, N, K = 1, 64, 32
A_1nk = [float(i % 5 + 1) for i in range(M * K)]
B_1nk = [float((i * 3) % 7 + 1) for i in range(K * N)]

_, dA = alloc(M * K * 4)
_, dB = alloc(K * N * 4)
_, dC = alloc(M * N * 4)

h2d(dA, f32_bytes(A_1nk))
h2d(dB, f32_bytes(B_1nk))

r1 = lib.aecMatmulF32(dA, dB, dC, M, N, K, None)
_, cd1 = d2h(dC, M * N * 4)
cv1 = unpack_f32(cd1)

free(dA)
free(dB)
free(dC)

ok1nk = True
for c_ in range(N):
    s = sum(A_1nk[k] * B_1nk[k * N + c_] for k in range(K))
    if abs(cv1[c_] - s) > FP32_ATOL + FP32_RTOL * abs(s):
        ok1nk = False
        break

# C11b: M=64, N=1, K=32 — single-column output
reset()
M, N, K = 64, 1, 32
A_m1k = [float(i % 5 + 1) for i in range(M * K)]
B_m1k = [float((i * 3) % 7 + 1) for i in range(K * N)]

_, dA = alloc(M * K * 4)
_, dB = alloc(K * N * 4)
_, dC = alloc(M * N * 4)

h2d(dA, f32_bytes(A_m1k))
h2d(dB, f32_bytes(B_m1k))

r2 = lib.aecMatmulF32(dA, dB, dC, M, N, K, None)
_, cd2 = d2h(dC, M * N * 4)
cv2 = unpack_f32(cd2)

free(dA)
free(dB)
free(dC)

ok_m1k = True
for r in range(M):
    s = sum(A_m1k[r * K + k] * B_m1k[k] for k in range(K))
    if abs(cv2[r] - s) > FP32_ATOL + FP32_RTOL * abs(s):
        ok_m1k = False
        break

# C11c: M=8, N=8, K=1 — dot-product degenerate
reset()
M, N, K = 8, 8, 1
A_mn1 = [float(i + 1) for i in range(M * K)]
B_mn1 = [float((i * 2) + 1) for i in range(K * N)]

_, dA = alloc(M * K * 4)
_, dB = alloc(K * N * 4)
_, dC = alloc(M * N * 4)

h2d(dA, f32_bytes(A_mn1))
h2d(dB, f32_bytes(B_mn1))

r3 = lib.aecMatmulF32(dA, dB, dC, M, N, K, None)
_, cd3 = d2h(dC, M * N * 4)
cv3 = unpack_f32(cd3)

free(dA)
free(dB)
free(dC)

ok_mn1 = True
for r in range(M):
    for c_ in range(N):
        s = sum(A_mn1[r * K + k] * B_mn1[k * N + c_] for k in range(K))
        if abs(cv3[r * N + c_] - s) > FP32_ATOL + FP32_RTOL * abs(s):
            ok_mn1 = False
            break

check("edge shapes: 1×64×32, 64×1×32, 8×8×1 all correct",
      r1 == SUCCESS and r2 == SUCCESS and r3 == SUCCESS
      and ok1nk and ok_m1k and ok_mn1,
      f"1x64x32={ename(r1)} ok={ok1nk}  64x1x32={ename(r2)} ok={ok_m1k}  8x8x1={ename(r3)} ok={ok_mn1}")

# ── C12: Span overlap — A and C overlap → ISA_TRAP ─────────────────────
# Allocate one block; set C at offset 0 (overlapping A). Device enforces.
reset()
M, N, K = 4, 4, 4
total = (M * K + K * N + M * N) * 4
_, block = alloc(total)
dA12 = block
dB12 = block + M * K * 4
dC12 = block                              # overlaps A!

A_vals12 = [float(i + 1) for i in range(M * K)]
B_vals12 = [float((i * 3) + 1) for i in range(K * N)]

h2d(dA12, f32_bytes(A_vals12))
h2d(dB12, f32_bytes(B_vals12))

r_overlap = lib.aecMatmulF32(dA12, dB12, dC12, M, N, K, None)
free(block)
# Device should detect overlap and raise ISA_TRAP (9) or DEVICE (7)
check("A/C span overlap -> ISA_TRAP or DEVICE error",
      r_overlap in (ISA_TRAP, DEVICE),
      f"got={ename(r_overlap)} (expected ISA_TRAP or DEVICE)")

# ── C13: Async GEMM on non-null Stream ─────────────────────────────────
reset()
M, N, K = 16, 16, 16
A_a = [float(i % 7 + 1) for i in range(M * K)]
B_a = [float((i * 3) % 11 + 1) for i in range(K * N)]

_, dA13 = alloc(M * K * 4)
_, dB13 = alloc(K * N * 4)
_, dC13 = alloc(M * N * 4)
_, st13 = mkstream()

h2d(dA13, f32_bytes(A_a))
h2d(dB13, f32_bytes(B_a))

r_async = lib.aecMatmulF32(dA13, dB13, dC13, M, N, K, st13)
r_sync = lib.aecStreamSync(st13)
_, cdata13 = d2h(dC13, M * N * 4)
cvals13 = unpack_f32(cdata13)

lib.aecStreamDestroy(st13)
free(dA13)
free(dB13)
free(dC13)

# Verify correctness of async result
ok_async = True
for r in range(M):
    for c_ in range(N):
        s = sum(A_a[r * K + k] * B_a[k * N + c_] for k in range(K))
        if abs(cvals13[r * N + c_] - s) > FP32_ATOL + FP32_RTOL * abs(s):
            ok_async = False
            break

check("async GEMM on non-null Stream, result correct after sync",
      r_async == SUCCESS and r_sync == SUCCESS and ok_async,
      f"gemm={ename(r_async)} sync={ename(r_sync)} correct={ok_async}")

print()
print(f"=== {sum(results)}/{len(results)} checks passed ===")
