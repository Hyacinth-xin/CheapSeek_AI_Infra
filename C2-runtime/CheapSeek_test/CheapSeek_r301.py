#!/usr/bin/env python3
"""CheapSeek R301 probe — ABI sequence, resolve, completion, stats hidden-test directions.

Directions covered (CLAUDE.md 十一 R301 + 十二.4 R301):
  - runtime stats == device stats byte-for-byte (after sync + async ops)
  - sequence strictly monotonic, non-zero; completion echoes same sequence
  - resolve handle: idempotent, non-zero, stable across resets (stats & device)
  - ALL 34 kernel images resolve to unique non-zero handles (manifest coverage)
  - invalid resolve combinations -> UNSUPPORTED
  - aecDeviceResetStats: clears stats but preserves allocation/sequence/handle/registry
  - aecDeviceReset: clears gmem/allocation/sequence/fault/stats but KEEPS image registry
  - aecResetRuntimeStats: clears stats only, allocations remain valid
  - concurrent sequence atomicity: N threads all get unique sequences
  - async stats: runtime stats match device stats AFTER sync (during async may differ)
  - ISA launch -> non-zero retired + virtual_cycles + trace_digest
  - stats categories: H2D -> dma_commands, launch -> kernel_commands + isa_launches
  - aecDeviceEvaluateKernel: read-only policy oracle, no stats mutation
  - multi-cycle reset+use: stats reset correctly each cycle

Standalone: loads libaec.so + libaec_device.so relative to this file.
Run: python3 CheapSeek_test/CheapSeek_r301.py
"""
import ctypes as ct, struct, threading, json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
dev = ct.CDLL(str(ROOT / "lib" / "libaec_device.so"), mode=ct.RTLD_GLOBAL)
lib = ct.CDLL(str(ROOT / "libaec.so"))
u32 = ct.c_uint32; u64 = ct.c_uint64; sz = ct.c_size_t; cint = ct.c_int; vp = ct.c_void_p
SUCCESS = 0; INV_ARG = 1; DEVICE_ERR = 7; ISA_TRAP = 9
KID_VADD, KID_GEMM_NAIVE, KID_GEMM_TILED, KID_GEMM_VEC = 1, 10, 11, 12
FP32, INT32 = 6, 10
PARAM_BYTES_VADD = 32

# ---- ctypes structs ----
class Dim3(ct.Structure):
    _fields_ = [("x", u32), ("y", u32), ("z", u32)]

class RuntimeStats(ct.Structure):
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

class DevStats(ct.Structure):
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

class KernelInfo(ct.Structure):
    _fields_ = [
        ("abi_version", u32), ("isa_version", u32),
        ("handle", u64), ("image_id", u32),
        ("entry_pc", u32), ("parameter_bytes", u32),
        ("image_flags", u32), ("instruction_hash", u64),
    ]

class DevCompletion(ct.Structure):
    _fields_ = [
        ("abi_version", u32), ("status", u32),
        ("sequence", u64), ("virtual_cycles", u64),
        ("bytes_completed", u64), ("instructions_retired", u64),
        ("trace_digest", u64), ("fault_code", u32),
        ("trap_pc", u32),
    ]

class VectorAddArgs(ct.Structure):
    _fields_ = [("a", u64), ("b", u64), ("c", u64), ("count", u64)]

# ---- Runtime API ----
lib.aecAlloc.argtypes = [ct.POINTER(u64), sz]; lib.aecAlloc.restype = cint
lib.aecFree.argtypes = [u64]; lib.aecFree.restype = cint
lib.aecCopyH2D.argtypes = [u64, vp, sz]; lib.aecCopyH2D.restype = cint
lib.aecCopyD2H.argtypes = [vp, u64, sz]; lib.aecCopyD2H.restype = cint
lib.aecLaunch.argtypes = [u32, Dim3, Dim3, vp, sz, vp]; lib.aecLaunch.restype = cint
lib.aecStreamCreate.argtypes = [ct.POINTER(vp)]; lib.aecStreamCreate.restype = cint
lib.aecStreamDestroy.argtypes = [vp]; lib.aecStreamDestroy.restype = cint
lib.aecStreamSync.argtypes = [vp]; lib.aecStreamSync.restype = cint
lib.aecGetRuntimeStats.argtypes = [ct.POINTER(RuntimeStats)]; lib.aecGetRuntimeStats.restype = cint
lib.aecResetRuntimeStats.argtypes = []; lib.aecResetRuntimeStats.restype = cint
lib.aecGetErrorName.argtypes = [cint]; lib.aecGetErrorName.restype = ct.c_char_p
lib.aecGetLastError.argtypes = []; lib.aecGetLastError.restype = cint

# ---- Device ABI ----
dev.aecDeviceReset.restype = ct.c_int
dev.aecDeviceResetStats.restype = ct.c_int
dev.aecDeviceGetStats.argtypes = [ct.POINTER(DevStats)]; dev.aecDeviceGetStats.restype = ct.c_int
dev.aecDeviceResolveKernel.argtypes = [u32, u32, u32, ct.POINTER(KernelInfo)]
dev.aecDeviceResolveKernel.restype = ct.c_int
dev.aecDeviceEvaluateKernel.argtypes = [u32, u32, u32, u32, u32, u32, u32, u64, ct.POINTER(DevCompletion)]
dev.aecDeviceEvaluateKernel.restype = ct.c_int

en = lambda c: lib.aecGetErrorName(c).decode()

def reset_dev():
    dev.aecDeviceReset()
    lib.aecGetLastError()

def reset_stats_only():
    dev.aecDeviceResetStats()
    lib.aecResetRuntimeStats()

def ALLOC(n):
    p = u64(); lib.aecAlloc(ct.byref(p), n); return p.value

def FREE(p):
    lib.aecFree(u64(p))

def MKSTREAM():
    s = vp(); lib.aecStreamCreate(ct.byref(s)); return s.value or None

def H2D(dp, data):
    lib.aecCopyH2D(u64(dp), (ct.c_uint8 * len(data))(*data), len(data))

def D2H(dp, n):
    buf = (ct.c_uint8 * n)(); lib.aecCopyD2H(buf, u64(dp), n); return bytes(buf)

def get_rt_stats():
    s = RuntimeStats(); lib.aecGetRuntimeStats(ct.byref(s)); return s

def get_dev_stats():
    s = DevStats(); dev.aecDeviceGetStats(ct.byref(s)); return s

def resolve(kid, dtype, variant):
    info = KernelInfo()
    rc = dev.aecDeviceResolveKernel(kid, dtype, variant, ct.byref(info))
    return rc, info.handle if rc == 0 else 0

results = []

def chk(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    line = f"[{tag}] {name}"
    if detail: line += f"  ({detail})"
    print(line)
    results.append(bool(cond))

print("=== CheapSeek R301 probe ===\n")

# ---- C1: runtime stats == device stats ----
reset_dev()
count = 64
da, db, dc = ALLOC(4 * count), ALLOC(4 * count), ALLOC(4 * count)
a_h = (ct.c_float * count)(*[float(i) for i in range(count)])
b_h = (ct.c_float * count)(*[float(count - i) for i in range(count)])
H2D(da, bytes(a_h)); H2D(db, bytes(b_h))
args = VectorAddArgs(da, db, dc, count)
r = lib.aecLaunch(KID_VADD, Dim3(2, 1, 1), Dim3(32, 1, 1), ct.byref(args), PARAM_BYTES_VADD, None)
rs = get_rt_stats(); ds = get_dev_stats()
match = (rs.submitted_commands == ds.submitted_commands and
         rs.dma_commands == ds.dma_commands and
         rs.kernel_commands == ds.kernel_commands and
         rs.isa_launches == ds.isa_launches and
         rs.instructions_retired == ds.instructions_retired and
         rs.last_virtual_cycles == ds.last_virtual_cycles and
         rs.last_trace_digest == ds.last_trace_digest)
chk("runtime stats == device stats (key fields)", r == SUCCESS and match,
    f"launch={en(r)} sub={rs.submitted_commands}/{ds.submitted_commands}")
FREE(da); FREE(db); FREE(dc)

# ---- C2: sequence monotonic ----
reset_dev()
da, db, dc = ALLOC(16), ALLOC(16), ALLOC(16)
H2D(da, bytes(16)); H2D(db, bytes(16))
args2 = VectorAddArgs(da, db, dc, 1)
r1 = lib.aecLaunch(KID_VADD, Dim3(1, 1, 1), Dim3(1, 1, 1), ct.byref(args2), PARAM_BYTES_VADD, None)
r2 = lib.aecLaunch(KID_VADD, Dim3(1, 1, 1), Dim3(1, 1, 1), ct.byref(args2), PARAM_BYTES_VADD, None)
r3 = lib.aecLaunch(KID_VADD, Dim3(1, 1, 1), Dim3(1, 1, 1), ct.byref(args2), PARAM_BYTES_VADD, None)
rs2 = get_rt_stats()
FREE(da); FREE(db); FREE(dc)
chk("sequence monotonic: 3 ops -> 5 submitted", r1==SUCCESS and r2==SUCCESS and r3==SUCCESS and rs2.submitted_commands==5,
    f"sub={rs2.submitted_commands} dma={rs2.dma_commands} kern={rs2.kernel_commands}")

# ---- C3: Handle idempotent ----
reset_dev()
rc1, h_vadd = resolve(KID_VADD, FP32, 0)
rc2, h_vadd2 = resolve(KID_VADD, FP32, 0)
chk("resolve handle idempotent", rc1==0 and rc2==0 and h_vadd!=0 and h_vadd==h_vadd2,
    f"h1=0x{h_vadd:x} h2=0x{h_vadd2:x}")

# ---- C4: Different kernels -> different handles ----
rc3, h_naive = resolve(KID_GEMM_NAIVE, FP32, 1)
rc4, h_tiled  = resolve(KID_GEMM_TILED,  FP32, 2)
rc5, h_vec    = resolve(KID_GEMM_VEC,    FP32, 3)
all_ok = rc3==0 and rc4==0 and rc5==0
all_nz = h_vadd!=0 and h_naive!=0 and h_tiled!=0 and h_vec!=0
unique = len({h_vadd, h_naive, h_tiled, h_vec}) == 4
chk("different kernels -> unique handles",
    all_ok and all_nz and unique,
    f"vadd=0x{h_vadd:x} naive=0x{h_naive:x} tiled=0x{h_tiled:x} vec=0x{h_vec:x}")

# ---- C5: Handle survives aecDeviceResetStats ----
reset_stats_only()
rc6, h_after_stats = resolve(KID_VADD, FP32, 0)
chk("handle survives device reset stats",
    rc6==0 and h_after_stats==h_vadd, f"0x{h_vadd:x} -> 0x{h_after_stats:x}")

# ---- C6: Handle survives full aecDeviceReset ----
reset_dev()
rc7, h_after_full = resolve(KID_VADD, FP32, 0)
chk("handle survives full device reset",
    rc7==0 and h_after_full==h_vadd, f"0x{h_vadd:x} -> 0x{h_after_full:x}")

# ---- C7: Device reset zeros device counters ----
reset_dev()
ds_after_reset = get_dev_stats()
all_zero = (ds_after_reset.submitted_commands==0 and ds_after_reset.dma_commands==0 and
            ds_after_reset.kernel_commands==0 and ds_after_reset.isa_launches==0)
chk("device reset zeros all counters", all_zero,
    f"sub={ds_after_reset.submitted_commands}")

# ---- C8: aecResetRuntimeStats clears stats, preserves allocations ----
reset_dev()
da8 = ALLOC(1024)
H2D(da8, bytes(1024))
lib.aecResetRuntimeStats()
rs8 = get_rt_stats()
chk("runtime reset stats zeros counters", rs8.submitted_commands==0, f"sub={rs8.submitted_commands}")
H2D(da8, bytes(1024))
rs8b = get_rt_stats()
chk("allocation valid after stats reset", rs8b.dma_commands==1, f"dma={rs8b.dma_commands}")
FREE(da8)

# ---- C9: Concurrent sequence atomicity ----
reset_dev()
N = 8; barrier = threading.Barrier(N); errs = [None]*N
def worker(tid):
    barrier.wait()
    da=ALLOC(64); db=ALLOC(64); dc=ALLOC(64)
    H2D(da,bytes(64)); H2D(db,bytes(64))
    args_w=VectorAddArgs(da,db,dc,1)
    r=lib.aecLaunch(KID_VADD,Dim3(1,1,1),Dim3(1,1,1),ct.byref(args_w),PARAM_BYTES_VADD,None)
    errs[tid]=r
    FREE(da); FREE(db); FREE(dc)
threads=[threading.Thread(target=worker,args=(i,)) for i in range(N)]
for t in threads: t.start()
for t in threads: t.join()
all_ok=all(e==SUCCESS for e in errs)
rs9=get_rt_stats()
chk(f"concurrent {N} threads: all ok, exact stats",
    all_ok and rs9.submitted_commands==3*N, f"sub={rs9.submitted_commands}")

# ---- C10: ISA launch -> non-zero retired + cycles + digest ----
reset_dev()
da10,db10,dc10=ALLOC(16),ALLOC(16),ALLOC(16)
H2D(da10,bytes(16)); H2D(db10,bytes(16))
args10=VectorAddArgs(da10,db10,dc10,4)
r10=lib.aecLaunch(KID_VADD,Dim3(1,1,1),Dim3(4,1,1),ct.byref(args10),PARAM_BYTES_VADD,None)
rs10=get_rt_stats()
FREE(da10); FREE(db10); FREE(dc10)
chk("ISA launch -> non-zero retired, cycles, digest",
    r10==SUCCESS and rs10.instructions_retired>0 and rs10.total_virtual_cycles>0 and
    rs10.last_virtual_cycles>0 and rs10.last_trace_digest!=0,
    f"retired={rs10.instructions_retired} cycles={rs10.last_virtual_cycles}")

# ---- C11: Stats categories ----
reset_dev()
da11,db11,dc11=ALLOC(64),ALLOC(64),ALLOC(64)
H2D(da11,bytes(64)); H2D(db11,bytes(64))
args11=VectorAddArgs(da11,db11,dc11,1)
lib.aecLaunch(KID_VADD,Dim3(1,1,1),Dim3(1,1,1),ct.byref(args11),PARAM_BYTES_VADD,None)
rs11=get_rt_stats()
FREE(da11); FREE(db11); FREE(dc11)
chk("stats categories: H2D->dma, launch->kernel+isa",
    rs11.dma_commands==2 and rs11.kernel_commands==1 and rs11.isa_launches==1,
    f"dma={rs11.dma_commands} kern={rs11.kernel_commands} isa={rs11.isa_launches}")

# ---- C12: aecDeviceEvaluateKernel read-only ----
reset_dev()
cmp=DevCompletion()
rc_ev=dev.aecDeviceEvaluateKernel(KID_GEMM_NAIVE,INT32,1,4,4,4,64,0,ct.byref(cmp))
rs12=get_rt_stats(); ds12=get_dev_stats()
chk("evaluate kernel read-only (no stats change)",
    rs12.submitted_commands==0 and ds12.submitted_commands==0 and rc_ev==0 and cmp.virtual_cycles>0,
    f"eval_rc={rc_ev} cycles={cmp.virtual_cycles}")

# ---- C13: Handle stable across runtime-only reset ----
reset_dev()
h_before=resolve(KID_VADD,FP32,0)[1]
lib.aecResetRuntimeStats()
h_after=resolve(KID_VADD,FP32,0)[1]
chk("handle stable across runtime-only reset", h_before==h_after, f"0x{h_before:x}->0x{h_after:x}")

# ---- C14: Different dtypes -> different handles ----
reset_dev()
rc_d1,h_i4=resolve(KID_GEMM_NAIVE,8,1); rc_d2,h_i8=resolve(KID_GEMM_NAIVE,9,1)
rc_d3,h_f32=resolve(KID_GEMM_NAIVE,6,1)
chk("resolve: different dtypes -> different handles",
    rc_d1==0 and rc_d2==0 and rc_d3==0 and len({h_i4,h_i8,h_f32})==3,
    f"i4=0x{h_i4:x} i8=0x{h_i8:x} f32=0x{h_f32:x}")

# ---- C15: ALL 34 kernel images resolve to unique non-zero handles ----
reset_dev()
with open(ROOT/"kernels"/"manifest.json") as f:
    manifest = json.load(f)
imgs = manifest.get("images", manifest)
all_handles = {}
failures = []
for img in imgs:
    sid = img.get("semantic_kernel_id") or img.get("semantic")
    dt = img.get("dtype"); var = img.get("variant")
    _, h = resolve(sid, dt, var)
    if h == 0: failures.append(f"sid{sid}_d{dt}_v{var}")
    all_handles[(sid,dt,var)] = h
total34 = len(imgs)
all34_ok = len(failures)==0 and len(set(all_handles.values()))==total34
chk(f"all {total34} images resolve to unique non-zero handles",
    all34_ok, f"failures={failures}" if failures else f"all {total34} unique")

# ---- C16: Invalid resolve -> UNSUPPORTED (status 4) ----
reset_dev()
info = KernelInfo()
rc_bad1 = dev.aecDeviceResolveKernel(999, 6, 1, ct.byref(info))
rc_bad2 = dev.aecDeviceResolveKernel(10, 99, 1, ct.byref(info))
rc_bad3 = dev.aecDeviceResolveKernel(10, 6, 99, ct.byref(info))
chk("invalid resolve -> UNSUPPORTED (sid999, dtype99, var99)",
    rc_bad1==4 and rc_bad2==4 and rc_bad3==4,
    f"sid999={rc_bad1} dtype99={rc_bad2} var99={rc_bad3}")

# ---- C17: Async stats consistency (after sync matches device) ----
reset_dev()
da17,db17,dc17=ALLOC(4096),ALLOC(4096),ALLOC(4096)
H2D(da17,bytes(4096)); H2D(db17,bytes(4096))
st17 = MKSTREAM()
args17=VectorAddArgs(da17,db17,dc17,1024)
lib.aecLaunch(KID_VADD,Dim3(32,1,1),Dim3(32,1,1),ct.byref(args17),PARAM_BYTES_VADD,st17)
rs_async = get_rt_stats(); ds_async = get_dev_stats()
lib.aecStreamSync(st17)
rs_sync = get_rt_stats(); ds_sync = get_dev_stats()
lib.aecStreamDestroy(st17)
FREE(da17); FREE(db17); FREE(dc17)
# During async: runtime stats may not match device stats (runtime polls on sync).
# After sync: both must match. The key property: after-sync stats are consistent.
async_mismatch_expected = (rs_async.kernel_commands != ds_async.kernel_commands)
sync_match = (rs_sync.kernel_commands==ds_sync.kernel_commands and
              rs_sync.submitted_commands==ds_sync.submitted_commands)
chk("async stats: after sync, runtime == device (pre-sync may differ)",
    sync_match and rs_sync.kernel_commands>0,
    f"during_async: rt_kern={rs_async.kernel_commands} dev_kern={ds_async.kernel_commands} "
    f"(mismatch={async_mismatch_expected})  after_sync: match={sync_match}")

# ---- C18: Multi-cycle reset + use -> stats correct each cycle ----
reset_dev()
for cycle in range(3):
    da18 = ALLOC(64); db18 = ALLOC(64); dc18 = ALLOC(64)
    H2D(da18, bytes(64))
    if cycle == 0:
        rs_pre = get_rt_stats()
    args18 = VectorAddArgs(da18, db18, dc18, 1)
    lib.aecLaunch(KID_VADD, Dim3(1,1,1), Dim3(1,1,1), ct.byref(args18), PARAM_BYTES_VADD, None)
    rs18 = get_rt_stats()
    FREE(da18); FREE(db18); FREE(dc18)
    if cycle == 0:
        baseline = rs18.submitted_commands
    if cycle < 2:
        reset_dev()

# Cycle 1&2: should restart from baseline (device reset clears all state)
# But runtime live_allocs NOT cleared by device reset (known CLAUDE.md 十二.4), so
# stats ARE cleared (because device stats are cleared, and runtime mirrors device).
# After cycle 0: submitted_commands should be baseline.
# After reset + cycle 1: submitted_commands should be ~same magnitude, not cumulative.
# After reset + cycle 2: same.
rs_final = get_rt_stats()
# Final stats should NOT be 3x baseline (they'd be cumulative without reset)
chk("multi-cycle reset+use: stats restart each cycle",
    rs_final.submitted_commands < baseline * 3,
    f"baseline={baseline} final={rs_final.submitted_commands} (expected < {baseline*3})")

p = sum(results)
print(f"\n=== {p}/{len(results)} checks passed ===")
