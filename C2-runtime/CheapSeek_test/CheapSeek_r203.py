#!/usr/bin/env python3
"""CheapSeek R203 probe — INT4 / INT8 GEMM hidden-test directions.

Directions covered (CLAUDE.md 十一 R203 + 十二.4 R203):
  - INT4 packed signed nibble ordering (low nibble = element 0 -> bits [3:0])
  - INT4 signed two''s complement: nibble 0x8->-8; range [-8,7]
  - odd total element count -> global last byte high nibble == 0
    (device uses ceil(total/2) bytes, NOT per-row padding — verified empirically)
  - INT8 signed byte full range (including -128)
  - M=N=K=1 minimum, M=N=K=256 maximum (both dtypes)
  - non-16-multiple shapes (17×31×255) — tiling boundary, INT8
  - edge shapes: 1×N×K, M×1×K, M×N×1 (INT4)
  - A/C span overlap -> ISA_TRAP (INT8)
  - async GEMM on non-null Stream (INT4)

Standalone: loads libaec.so + libaec_device.so relative to this file.
Run: python3 CheapSeek_test/CheapSeek_r203.py
"""
import ctypes as ct, struct
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
dev = ct.CDLL(str(ROOT / "lib" / "libaec_device.so"), mode=ct.RTLD_GLOBAL)
lib = ct.CDLL(str(ROOT / "libaec.so"))
u32 = ct.c_uint32; u64 = ct.c_uint64; sz = ct.c_size_t; cint = ct.c_int; vp = ct.c_void_p
SUCCESS, INV_ARG, INV_HANDLE, INV_ADDR, ISA_TRAP, DEVICE = 0, 1, 3, 4, 9, 7

lib.aecAlloc.argtypes = [ct.POINTER(u64), sz]; lib.aecAlloc.restype = cint
lib.aecFree.argtypes = [u64]; lib.aecFree.restype = cint
lib.aecCopyH2D.argtypes = [u64, vp, sz]; lib.aecCopyH2D.restype = cint
lib.aecCopyD2H.argtypes = [vp, u64, sz]; lib.aecCopyD2H.restype = cint
lib.aecMatmulI4.argtypes = [u64, u64, u64, u32, u32, u32, vp]; lib.aecMatmulI4.restype = cint
lib.aecMatmulI8.argtypes = [u64, u64, u64, u32, u32, u32, vp]; lib.aecMatmulI8.restype = cint
lib.aecStreamCreate.argtypes = [ct.POINTER(vp)]; lib.aecStreamCreate.restype = cint
lib.aecStreamDestroy.argtypes = [vp]; lib.aecStreamDestroy.restype = cint
lib.aecStreamSync.argtypes = [vp]; lib.aecStreamSync.restype = cint
lib.aecGetErrorName.argtypes = [cint]; lib.aecGetErrorName.restype = ct.c_char_p
lib.aecGetLastError.argtypes = []; lib.aecGetLastError.restype = cint
dev.aecDeviceReset.restype = cint

en = lambda c: lib.aecGetErrorName(c).decode()

def reset():
    dev.aecDeviceReset()
    lib.aecGetLastError()

def ALLOC(n):
    p = u64()
    lib.aecAlloc(ct.byref(p), n)
    return p.value

def FREE(p):
    lib.aecFree(u64(p))

def MKSTREAM():
    s = vp()
    lib.aecStreamCreate(ct.byref(s))
    return s.value or None

def H2D(dp, data):
    lib.aecCopyH2D(u64(dp), (ct.c_uint8 * len(data))(*data), len(data))

def D2H(dp, n):
    buf = (ct.c_uint8 * n)()
    lib.aecCopyD2H(buf, u64(dp), n)
    return bytes(buf)

def pk4(v):
    """Pack signed INT4 values into bytes. Low nibble first. Odd count -> last hi=0."""
    out = []
    for i in range(0, len(v), 2):
        out.append((v[i] & 0xF) | (((v[i + 1] & 0xF) if i + 1 < len(v) else 0) << 4))
    return bytes(out)

def pk8(v):
    return bytes(x & 0xFF for x in v)

def ui32(d):
    return list(struct.unpack(f"<{len(d)//4}i", d))

results = []

def chk(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    line = f"[{tag}] {name}"
    if detail: line += f"  ({detail})"
    print(line)
    results.append(bool(cond))

print("=== CheapSeek R203 probe ===\n")

# ---- C1: INT8 2x2x3 golden ----
reset(); M,N,K = 2,2,3
A,B = [1,2,3,4,5,6],[7,8,9,10,11,12]; Cexp=[58,64,139,154]
da,db,dc = ALLOC(len(A)),ALLOC(len(B)),ALLOC(M*N*4)
H2D(da,pk8(A)); H2D(db,pk8(B))
r = lib.aecMatmulI8(da,db,dc,M,N,K,None)
cv = ui32(D2H(dc,M*N*4))
FREE(da); FREE(db); FREE(dc)
chk("INT8 GEMM 2x2x3 golden", r==SUCCESS and cv==Cexp, f"{en(r)} {cv}")

# ---- C2: INT4 2x2x3 golden ----
reset(); M,N,K = 2,2,3
A,B = [1,2,3,4,5,6],[1,2,3,4,5,6]; Cexp=[22,28,49,64]
pa,pb = pk4(A),pk4(B)
da,db,dc = ALLOC(len(pa)),ALLOC(len(pb)),ALLOC(M*N*4)
H2D(da,pa); H2D(db,pb)
r = lib.aecMatmulI4(da,db,dc,M,N,K,None)
cv = ui32(D2H(dc,M*N*4))
FREE(da); FREE(db); FREE(dc)
chk("INT4 GEMM 2x2x3 golden", r==SUCCESS and cv==Cexp, f"{en(r)} {cv}")

# ---- C3: INT4 nibble order (asymmetric) ----
reset(); M,N,K = 1,2,2
A,B = [2,1],[3,5,4,6]; Cexp=[10,16]
pa,pb = pk4(A),pk4(B)
da,db,dc = ALLOC(len(pa)),ALLOC(len(pb)),ALLOC(M*N*4)
H2D(da,pa); H2D(db,pb)
r = lib.aecMatmulI4(da,db,dc,M,N,K,None)
cv = ui32(D2H(dc,M*N*4))
FREE(da); FREE(db); FREE(dc)
chk("INT4 nibble order: lo=element0", r==SUCCESS and cv==Cexp, f"{en(r)} {cv}")

# ---- C4: INT4 odd total (MxK=3) -> last byte hi=0 ----
reset(); M,N,K = 1,2,3
A,B = [1,2,3],[1,2,3,4,5,6]; Cexp=[22,28]
pa,pb = pk4(A),pk4(B)
pad_ok = len(pa)==2 and (pa[-1]&0xF0)==0
da,db,dc = ALLOC(len(pa)),ALLOC(len(pb)),ALLOC(M*N*4)
H2D(da,pa); H2D(db,pb)
r = lib.aecMatmulI4(da,db,dc,M,N,K,None)
cv = ui32(D2H(dc,M*N*4))
FREE(da); FREE(db); FREE(dc)
chk("INT4 odd total (MxK=3) hi=0", pad_ok and r==SUCCESS and cv==Cexp,
    f"pad={pad_ok} last=0x{pa[-1]:02x} {en(r)} {cv}")

# ---- C5: INT4 odd total B (KxN=9) + signed 8->-8 ----
reset(); M,N,K = 2,3,3
A,B = [1,2,3,4,5,6],[1,2,3,4,5,6,7,8,9]
# signed INT4: 8->-8, 9->-7.  B=[[1,2,3],[4,5,6],[7,-8,-7]]
# C = [[30,-12,-6],[66,-15,0]]
Cexp=[30,-12,-6,66,-15,0]
pa,pb = pk4(A),pk4(B)
pad_ok = len(pb)==5 and (pb[-1]&0xF0)==0
da,db,dc = ALLOC(len(pa)),ALLOC(len(pb)),ALLOC(M*N*4)
H2D(da,pa); H2D(db,pb)
r = lib.aecMatmulI4(da,db,dc,M,N,K,None)
cv = ui32(D2H(dc,M*N*4))
FREE(da); FREE(db); FREE(dc)
chk("INT4 odd B (9) + signed 8->-8", pad_ok and r==SUCCESS and cv==Cexp,
    f"pad={pad_ok} last=0x{pb[-1]:02x} {en(r)} {cv}")

# ---- C6: INT8 negative full range ----
reset()
A,B = [-128,-1,0,127],[1,1,1,1]; Cexp1=[-2]
da,db,dc = ALLOC(4),ALLOC(4),ALLOC(4)
H2D(da,pk8(A)); H2D(db,pk8(B))
r1 = lib.aecMatmulI8(da,db,dc,1,1,4,None)
cv1 = ui32(D2H(dc,4))
FREE(da); FREE(db); FREE(dc)

reset()
A2,B2 = [-128,-128],[-128,-128]; Cexp2=[32768]
da2,db2,dc2 = ALLOC(2),ALLOC(2),ALLOC(4)
H2D(da2,pk8(A2)); H2D(db2,pk8(B2))
r2 = lib.aecMatmulI8(da2,db2,dc2,1,1,2,None)
cv2 = ui32(D2H(dc2,4))
FREE(da2); FREE(db2); FREE(dc2)
chk("INT8 neg full range + negxng", r1==SUCCESS and cv1==Cexp1 and r2==SUCCESS and cv2==Cexp2,
    f"{en(r1)} {cv1} / {en(r2)} {cv2}")

# ---- C7: INT4 range [-8,7] + nibble 0x8->-8 ----
reset()
A = list(range(-8,8)); B = [1]*16
pa,pb = pk4(A),pk4(B)
nib_ok = (pa[0]&0x0F)==0x8 and A[0]==-8
da,db,dc = ALLOC(len(pa)),ALLOC(len(pb)),ALLOC(4)
H2D(da,pa); H2D(db,pb)
r = lib.aecMatmulI4(da,db,dc,1,1,16,None)
cv = ui32(D2H(dc,4))
FREE(da); FREE(db); FREE(dc)
chk(f"INT4 range [-8,7] nib0x8->-8 sum={sum(A)}", nib_ok and r==SUCCESS and cv==[sum(A)],
    f"{en(r)} {cv}")

# ---- C8: INT4 M=N=K=1 / C9: INT8 M=N=K=1 ----
for lbl,f,A,B,Cexp,pk in [("INT4 M=N=K=1",lib.aecMatmulI4,[3],[-2],[-6],pk4),
                            ("INT8 M=N=K=1",lib.aecMatmulI8,[42],[-3],[-126],pk8)]:
    reset(); pa,pb=pk(A),pk(B)
    da,db,dc = ALLOC(len(pa)),ALLOC(len(pb)),ALLOC(4)
    H2D(da,pa); H2D(db,pb)
    r = f(da,db,dc,1,1,1,None)
    cv = ui32(D2H(dc,4))
    FREE(da); FREE(db); FREE(dc)
    chk(lbl, r==SUCCESS and cv==Cexp, f"{en(r)} {cv}")

# ---- C10: INT4 256^3 / C11: INT8 256^3 ----
for lbl,f,gen,pk in [("INT4 256",lib.aecMatmulI4,
                       lambda: (
                           [(i%9)-4 for i in range(65536)],
                           [(i%7)-3 for i in range(65536)]),
                       pk4),
                      ("INT8 256",lib.aecMatmulI8,
                       lambda: (
                           [(i%13)-6 for i in range(65536)],
                           [(i*3%17)-8 for i in range(65536)]),
                       pk8)]:
    reset(); M=N=K=256
    A,B = gen()
    pa,pb = pk(A),pk(B)
    da,db,dc = ALLOC(len(pa)),ALLOC(len(pb)),ALLOC(M*N*4)
    H2D(da,pa); H2D(db,pb)
    r = f(da,db,dc,M,N,K,None)
    cv = ui32(D2H(dc,M*N*4))
    FREE(da); FREE(db); FREE(dc)
    ok = True
    for ri in [0,127,255]:
        for ci in [0,127,255]:
            s = sum(A[ri*K+k]*B[k*N+ci] for k in range(K))
            if cv[ri*N+ci]!=s: ok=False; break
    chk(f"{lbl} spot-check 9", r==SUCCESS and ok, f"{en(r)} ok={ok} sz={len(pa)},{len(pb)}")

# ---- C12: INT8 17x31x255 non-16-multiple ----
reset(); M,N,K = 17,31,255
A = [(i*7+3)%23-11 for i in range(M*K)]
B = [(i*5+13)%19-9 for i in range(K*N)]
da,db,dc = ALLOC(M*K),ALLOC(K*N),ALLOC(M*N*4)
H2D(da,pk8(A)); H2D(db,pk8(B))
r = lib.aecMatmulI8(da,db,dc,M,N,K,None)
cv = ui32(D2H(dc,M*N*4))
FREE(da); FREE(db); FREE(dc)
ok = True
for ri in [0,8,16]:
    for ci in [0,15,30]:
        s = sum(A[ri*K+k]*B[k*N+ci] for k in range(K))
        if cv[ri*N+ci]!=s: ok=False; break
chk("INT8 17x31x255 non-16-mul", r==SUCCESS and ok, f"{en(r)} ok={ok}")

# ---- C13: edge shapes INT4 ----
def edge_test(M,N,K,A,B):
    reset(); pa,pb = pk4(A),pk4(B)
    da,db,dc = ALLOC(len(pa)),ALLOC(len(pb)),ALLOC(M*N*4)
    H2D(da,pa); H2D(db,pb)
    r = lib.aecMatmulI4(da,db,dc,M,N,K,None)
    cv = ui32(D2H(dc,M*N*4))
    FREE(da); FREE(db); FREE(dc)
    ok = True
    for ri in range(M):
        for ci in range(N):
            s = sum(A[ri*K+k]*B[k*N+ci] for k in range(K))
            if cv[ri*N+ci]!=s: ok=False; break
    return r,ok

e1 = edge_test(1,16,8, [(i%5)-2 for i in range(8)],[(i*3%7)-3 for i in range(128)])
e2 = edge_test(16,1,8, [(i%5)-2 for i in range(128)],[(i*3%7)-3 for i in range(8)])
e3 = edge_test(4,4,1, [i-2 for i in range(4)],[i*2-4 for i in range(4)])
chk("INT4 edge 1x16x8 / 16x1x8 / 4x4x1",
    e1[0]==SUCCESS and e1[1] and e2[0]==SUCCESS and e2[1] and e3[0]==SUCCESS and e3[1],
    f"1x16: {en(e1[0])} ok={e1[1]}  16x1: {en(e2[0])} ok={e2[1]}  4x4: {en(e3[0])} ok={e3[1]}")

# ---- C14: A/C overlap -> ISA_TRAP ----
reset(); M,N,K = 4,4,4
total = M*K+K*N+M*N*4
block = ALLOC(total); da=block; db=block+M*K; dc=block
A = [(i%7)-3 for i in range(M*K)]; B = [(i*3%11)-5 for i in range(K*N)]
H2D(da,pk8(A)); H2D(db,pk8(B))
r = lib.aecMatmulI8(da,db,dc,M,N,K,None)
FREE(block)
chk("A/C overlap -> ISA_TRAP/DEVICE", r in (ISA_TRAP,DEVICE), f"{en(r)}")

# ---- C15: async INT4 Stream ----
reset(); M,N,K = 16,16,16
A = [(i%7)-3 for i in range(M*K)]; B = [(i*3%5)-2 for i in range(K*N)]
pa,pb = pk4(A),pk4(B)
da,db,dc = ALLOC(len(pa)),ALLOC(len(pb)),ALLOC(M*N*4)
st = MKSTREAM()
H2D(da,pa); H2D(db,pb)
ra = lib.aecMatmulI4(da,db,dc,M,N,K,st)
rs = lib.aecStreamSync(st)
cv = ui32(D2H(dc,M*N*4))
lib.aecStreamDestroy(st)
FREE(da); FREE(db); FREE(dc)
ok = all(sum(A[r_*K+k]*B[k*N+c_] for k in range(K))==cv[r_*N+c_]
         for r_ in range(M) for c_ in range(N))
chk("async INT4 Stream", ra==SUCCESS and rs==SUCCESS and ok,
    f"gemm={en(ra)} sync={en(rs)} correct={ok}")

p = sum(results)
print(f"\n=== {p}/{len(results)} checks passed ===")
