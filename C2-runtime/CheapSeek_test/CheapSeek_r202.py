#!/usr/bin/env python3
"""CheapSeek R202 probe — all floating-point GEMM dtype hidden-test directions.

Directions covered (CLAUDE.md 十一 + 十二.4, docs/04):
  - FP4 E2M1 packed nibble order (low nibble first, odd count high nibble = 0)
  - FP8 E4M3 vs E5M2 canonical NaN — 0x7f vs 0x7e
  - FP8 overflow → saturate to max finite (both E4M3 and E5M2, per docs/04)
  - FP16/BF16 canonical NaN — 0x7e00 vs 0x7fc0
  - FP16 exact cancellation → +0.0
  - FP64 loose tolerance (atol=1e-12 rtol=1e-11)

Note: CLAUDE.md 十二.4 claims E5M2 overflow→Inf, but docs/04 §3 says all FP4/FP8
overflow 饱和到同符号最大有限值. Device behavior confirms saturation.

Run: python3 CheapSeek_test/CheapSeek_r202.py
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
INV_ADDR = 4
DEVICE = 7
ISA_TRAP = 9

E4M3 = 1
E5M2 = 2
FP64_ATOL = 1e-12
FP64_RTOL = 1e-11

# ── Runtime API ────────────────────────────────────────────────────────
lib.aecAlloc.argtypes = [ct.POINTER(u64), sz]
lib.aecAlloc.restype = cint
lib.aecFree.argtypes = [u64]
lib.aecFree.restype = cint
lib.aecCopyH2D.argtypes = [u64, vp, sz]
lib.aecCopyH2D.restype = cint
lib.aecCopyD2H.argtypes = [vp, u64, sz]
lib.aecCopyD2H.restype = cint
lib.aecMatmulF4.argtypes = [u64, u64, u64, u32, u32, u32, vp]
lib.aecMatmulF4.restype = cint
lib.aecMatmulF8.argtypes = [u64, u64, u64, u32, u32, u32, cint, vp]
lib.aecMatmulF8.restype = cint
lib.aecMatmulF16.argtypes = [u64, u64, u64, u32, u32, u32, vp]
lib.aecMatmulF16.restype = cint
lib.aecMatmulBF16.argtypes = [u64, u64, u64, u32, u32, u32, vp]
lib.aecMatmulBF16.restype = cint
lib.aecMatmulF64.argtypes = [u64, u64, u64, u32, u32, u32, vp]
lib.aecMatmulF64.restype = cint
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

def h2d(dp, data):
    buf = (ct.c_uint8 * len(data))(*data)
    return lib.aecCopyH2D(u64(dp), buf, len(data))

def d2h(dp, n):
    buf = (ct.c_uint8 * n)()
    rc = lib.aecCopyD2H(buf, u64(dp), n)
    return rc, bytes(buf)

# ── dtype helpers ──────────────────────────────────────────────────────

def fp16_bytes(values):
    return b"".join(struct.pack("<e", v) for v in values)

def fp16_unpack(data):
    return list(struct.unpack(f"<{len(data)//2}e", data))

def bf16_bytes(values):
    out = bytearray()
    for v in values:
        f32 = struct.pack("<f", v)
        out.extend(f32[2:4])
    return bytes(out)

def bf16_to_f32(data):
    out = []
    for i in range(0, len(data), 2):
        f32 = b"\x00\x00" + data[i:i+2]
        out.append(struct.unpack("<f", f32)[0])
    return out

def fp64_bytes(values):
    return b"".join(struct.pack("<d", v) for v in values)

def fp4_e2m1_nibbles(values):
    nibbles = []
    for v in values:
        if v != v:
            nibbles.append(0x7); continue
        sign = 0x8 if v < 0 else 0
        v = abs(v)
        if v == 0.0:
            nibbles.append(sign); continue
        exp = 0
        while v >= 2.0: v /= 2.0; exp += 1
        while v < 1.0 and exp > -2: v *= 2.0; exp -= 1
        if exp < -2:
            nibbles.append(sign); continue
        field_exp = exp + 1
        v -= 1.0
        mant = int(v * 2.0 + 0.5)
        if mant >= 2: mant = 0; field_exp += 1
        if field_exp >= 3:
            nibbles.append(sign | 0x7)
        else:
            nibbles.append(sign | (field_exp << 1) | mant)
    return nibbles

def fp4_pack(nibbles):
    out = bytearray()
    for i in range(0, len(nibbles), 2):
        lo = nibbles[i] & 0xF
        hi = nibbles[i + 1] & 0xF if i + 1 < len(nibbles) else 0
        out.append(lo | (hi << 4))
    return bytes(out)

def fp4_unpack(data, count):
    nibs = []
    for b in data:
        nibs.append(b & 0xF)
        nibs.append((b >> 4) & 0xF)
    return nibs[:count]

# ── test infrastructure ────────────────────────────────────────────────

results = []

def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    line = f"[{tag}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    results.append(bool(cond))

print("=== CheapSeek R202 probe ===")
print()

# ── C1: FP16 basic (2×2×2) ────────────────────────────────────────────
reset()
M, N, K = 2, 2, 2
A_vals = [1.0, 2.0, -1.0, 3.0]
B_vals = [0.5, -1.0, 2.0, 0.25]
# C = [[4.5, -0.5], [5.5, 1.75]]
C_exp = [4.5, -0.5, 5.5, 1.75]

_, dA = alloc(M * K * 2); _, dB = alloc(K * N * 2); _, dC = alloc(M * N * 2)
h2d(dA, fp16_bytes(A_vals)); h2d(dB, fp16_bytes(B_vals))
r = lib.aecMatmulF16(dA, dB, dC, M, N, K, None)
_, cdata = d2h(dC, M * N * 2)
cvals = fp16_unpack(cdata)
free(dA); free(dB); free(dC)
ok = all(abs(g - e) < 1e-3 for g, e in zip(cvals, C_exp))
check("FP16 GEMM 2×2×2 bit-exact", r == SUCCESS and ok,
      f"rc={ename(r)} got={[f'{v:.4f}' for v in cvals]} exp={C_exp}")

# ── C2: BF16 basic (2×2×2) ────────────────────────────────────────────
reset()
_, dA = alloc(M * K * 2); _, dB = alloc(K * N * 2); _, dC = alloc(M * N * 2)
h2d(dA, bf16_bytes(A_vals)); h2d(dB, bf16_bytes(B_vals))
r = lib.aecMatmulBF16(dA, dB, dC, M, N, K, None)
_, cdata = d2h(dC, M * N * 2)
cvals = bf16_to_f32(cdata)
free(dA); free(dB); free(dC)
ok = all(abs(g - e) < 1e-3 for g, e in zip(cvals, C_exp))
check("BF16 GEMM 2×2×2", r == SUCCESS and ok,
      f"rc={ename(r)} got={[f'{v:.4f}' for v in cvals]} exp={C_exp}")

# ── C3: FP64 basic (2×2×2) ────────────────────────────────────────────
reset()
_, dA = alloc(M * K * 8); _, dB = alloc(K * N * 8); _, dC = alloc(M * N * 8)
h2d(dA, fp64_bytes(A_vals)); h2d(dB, fp64_bytes(B_vals))
r = lib.aecMatmulF64(dA, dB, dC, M, N, K, None)
_, cdata = d2h(dC, M * N * 8)
cvals = list(struct.unpack(f"<{M*N}d", cdata))
free(dA); free(dB); free(dC)
ok = all(abs(g - e) <= FP64_ATOL + FP64_RTOL * abs(e) for g, e in zip(cvals, C_exp))
check("FP64 GEMM 2×2×2 (atol=1e-12 rtol=1e-11)", r == SUCCESS and ok,
      f"rc={ename(r)} got={[f'{v:.14f}' for v in cvals]} exp={C_exp}")

# ── C4: FP8 E4M3 basic (2×3=6 → 0x4a) ────────────────────────────────
reset()
M = N = K = 1
_, dA = alloc(1); _, dB = alloc(1); _, dC = alloc(1)
h2d(dA, bytes([0x40])); h2d(dB, bytes([0x42]))
r = lib.aecMatmulF8(dA, dB, dC, M, N, K, E4M3, None)
_, cdata = d2h(dC, 1)
free(dA); free(dB); free(dC)
check("FP8 E4M3 1×1×1: 2×3=6 (0x4a)", r == SUCCESS and cdata == bytes([0x4a]),
      f"rc={ename(r)} got=0x{cdata[0]:02x} exp=0x4a")

# ── C5: FP8 E5M2 basic (2×3=6 → 0x45) ────────────────────────────────
reset()
_, dA = alloc(1); _, dB = alloc(1); _, dC = alloc(1)
h2d(dA, bytes([0x40])); h2d(dB, bytes([0x41]))
r = lib.aecMatmulF8(dA, dB, dC, M, N, K, E5M2, None)
_, cdata = d2h(dC, 1)
free(dA); free(dB); free(dC)
check("FP8 E5M2 1×1×1: 2×3=6 (0x45)", r == SUCCESS and cdata == bytes([0x45]),
      f"rc={ename(r)} got=0x{cdata[0]:02x} exp=0x45")

# ── C6: FP4 basic (1×1×2: 2*0.5+1*4=5 → tie-to-even 4.0=0x6) ────────
reset()
M, N, K = 1, 1, 2
_, dA = alloc(1); _, dB = alloc(1); _, dC = alloc(1)
h2d(dA, bytes([0x24])); h2d(dB, bytes([0x61]))
r = lib.aecMatmulF4(dA, dB, dC, M, N, K, None)
_, cdata = d2h(dC, 1)
free(dA); free(dB); free(dC)
nibs = fp4_unpack(cdata, 1)
check("FP4 E2M1 1×1×2: 2*0.5+1*4=5 ties→4.0 (0x6)",
      r == SUCCESS and nibs[0] == 0x6,
      f"rc={ename(r)} got=0x{nibs[0]:x} exp=0x6")

# ── C7: FP4 odd element count — high nibble = 0 ───────────────────────
reset()
M, N, K = 1, 1, 3
_, dA = alloc(2); _, dB = alloc(2); _, dC = alloc(1)
h2d(dA, bytes([0x24, 0x01])); h2d(dB, bytes([0x11, 0x02]))
r = lib.aecMatmulF4(dA, dB, dC, M, N, K, None)
_, cdata = d2h(dC, 1)
free(dA); free(dB); free(dC)
nibs = fp4_unpack(cdata, 1)
check("FP4 odd count 1×1×3: 2*0.5+1*0.5+0.5*1=2.0 (0x4), hi nib=0",
      r == SUCCESS and nibs[0] == 0x4,
      f"rc={ename(r)} got=0x{nibs[0]:x} exp=0x4")

# ── C8: FP8 E4M3 overflow → saturate ──────────────────────────────────
# E4M3 max finite: 0x7e = 448 (exp=15 mant=6, since 0x7f=NaN)
# 32×16=512 > 448 → saturate to 0x7e
# 32.0 E4M3: 32=1*2^5, field_exp=12=0b1100, mant=0 → 0b0_1100_000 = 0x60
reset()
M = N = K = 1
_, dA = alloc(1); _, dB = alloc(1); _, dC = alloc(1)
h2d(dA, bytes([0x60])); h2d(dB, bytes([0x58]))
r = lib.aecMatmulF8(dA, dB, dC, M, N, K, E4M3, None)
_, cdata = d2h(dC, 1)
free(dA); free(dB); free(dC)
check("FP8 E4M3 overflow: 32×16=512 > 448 → saturate to 0x7e",
      r == SUCCESS and cdata == bytes([0x7e]),
      f"rc={ename(r)} got=0x{cdata[0]:02x} exp=0x7e")

# ── C9: FP8 E5M2 overflow → saturate ──────────────────────────────────
# E5M2 max finite: 0x7b = 57344 (exp=30 mant=3)
# 256×256=65536 > 57344 → saturate to 0x7b
# Both E4M3 and E5M2 saturate per docs/04 §3, contrary to CLAUDE.md claim
reset()
M = N = K = 1
_, dA = alloc(1); _, dB = alloc(1); _, dC = alloc(1)
h2d(dA, bytes([0x5c])); h2d(dB, bytes([0x5c]))
r = lib.aecMatmulF8(dA, dB, dC, M, N, K, E5M2, None)
_, cdata = d2h(dC, 1)
free(dA); free(dB); free(dC)
check("FP8 E5M2 overflow: 256×256=65536 > 57344 → saturate to 0x7b",
      r == SUCCESS and cdata == bytes([0x7b]),
      f"rc={ename(r)} got=0x{cdata[0]:02x} exp=0x7b")

# ── C10: FP8 E4M3 canonical NaN (0x7f) ────────────────────────────────
reset()
M = N = K = 1
_, dA = alloc(1); _, dB = alloc(1); _, dC = alloc(1)
h2d(dA, bytes([0x7f])); h2d(dB, bytes([0x38]))
r = lib.aecMatmulF8(dA, dB, dC, M, N, K, E4M3, None)
_, cdata = d2h(dC, 1)
free(dA); free(dB); free(dC)
check("FP8 E4M3 NaN in → canonical NaN out (0x7f)",
      r == SUCCESS and cdata == bytes([0x7f]),
      f"rc={ename(r)} got=0x{cdata[0]:02x} exp=0x7f")

# ── C11: FP8 E5M2 canonical NaN (0x7e) ────────────────────────────────
reset()
_, dA = alloc(1); _, dB = alloc(1); _, dC = alloc(1)
h2d(dA, bytes([0x7e])); h2d(dB, bytes([0x3c]))
r = lib.aecMatmulF8(dA, dB, dC, M, N, K, E5M2, None)
_, cdata = d2h(dC, 1)
free(dA); free(dB); free(dC)
check("FP8 E5M2 NaN in → canonical NaN out (0x7e)",
      r == SUCCESS and cdata == bytes([0x7e]),
      f"rc={ename(r)} got=0x{cdata[0]:02x} exp=0x7e")

# ── C12: FP16 canonical NaN (0x7e00) ──────────────────────────────────
reset()
M = N = K = 1
A_bytes = struct.pack("<e", float("nan"))
B_bytes = struct.pack("<e", 1.0)
C_exp = struct.pack("<e", float("nan"))
_, dA = alloc(2); _, dB = alloc(2); _, dC = alloc(2)
h2d(dA, A_bytes); h2d(dB, B_bytes)
r = lib.aecMatmulF16(dA, dB, dC, M, N, K, None)
_, cdata = d2h(dC, 2)
free(dA); free(dB); free(dC)
check("FP16 NaN in → canonical NaN out (0x7e00)",
      r == SUCCESS and cdata == C_exp,
      f"rc={ename(r)} got={cdata.hex()} exp={C_exp.hex()}")

# ── C13: FP16 exact cancellation → +0.0 ───────────────────────────────
reset()
M, N, K = 1, 1, 2
A_bytes = struct.pack("<ee", 1.0, -1.0)
B_bytes = struct.pack("<ee", 1.0, 1.0)
C_exp = struct.pack("<e", 0.0)  # +0.0 = 0000
_, dA = alloc(4); _, dB = alloc(4); _, dC = alloc(2)
h2d(dA, A_bytes); h2d(dB, B_bytes)
r = lib.aecMatmulF16(dA, dB, dC, M, N, K, None)
_, cdata = d2h(dC, 2)
free(dA); free(dB); free(dC)
check("FP16 exact cancellation 1+(-1)=+0.0 (0x0000)",
      r == SUCCESS and cdata == C_exp,
      f"rc={ename(r)} got={cdata.hex()} exp={C_exp.hex()}")

print()
print(f"=== {sum(results)}/{len(results)} checks passed ===")
