#!/usr/bin/env python3
"""CheapSeek R104 probe — Vector Add launch hidden-test directions.

Directions covered (CLAUDE.md 十一 + 十二.4):
  - minimal grid/block all dims = 1 (count=1)
  - block volume = 1024 limit accepted; > 1024 rejected
  - grid/block dim = 0 rejected
  - args = NULL rejected; args_size must be exact (16 / 64 rejected)
  - unknown kernel ID -> NOT_SUPPORTED
  - null stream synchronous (result immediately correct, no explicit sync)
  - non-null stream async launch COPIES args (overwrite after launch -> original result)
  - ISA trap mapping: freed device pointer in args -> AEC_ERROR_ISA_TRAP
  - launch stats accounting (submitted_commands / isa_launches increment)

Standalone: loads libaec.so + libaec_device.so relative to this file.
Run: python3 test/CheapSeek_r104.py
"""
import ctypes as ct
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
KID_VADD = 1          # AEC_KERNEL_VECTOR_ADD_F32
PARAM_BYTES = 32      # AEC_KERNEL_PARAM_VECTOR_ADD_BYTES


class Dim3(ct.Structure):
    _fields_ = [("x", u32), ("y", u32), ("z", u32)]


class VectorAddArgs(ct.Structure):
    _fields_ = [("a", u64), ("b", u64), ("c", u64), ("count", u64)]


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


lib.aecAlloc.argtypes = [ct.POINTER(u64), sz]
lib.aecAlloc.restype = cint
lib.aecFree.argtypes = [u64]
lib.aecFree.restype = cint
lib.aecCopyH2D.argtypes = [u64, vp, sz]
lib.aecCopyH2D.restype = cint
lib.aecCopyD2H.argtypes = [vp, u64, sz]
lib.aecCopyD2H.restype = cint
lib.aecLaunch.argtypes = [u32, Dim3, Dim3, vp, sz, vp]
lib.aecLaunch.restype = cint
lib.aecStreamCreate.argtypes = [ct.POINTER(vp)]
lib.aecStreamCreate.restype = cint
lib.aecStreamSync.argtypes = [vp]
lib.aecStreamSync.restype = cint
lib.aecStreamDestroy.argtypes = [vp]
lib.aecStreamDestroy.restype = cint
lib.aecGetRuntimeStats.argtypes = [ct.POINTER(RuntimeStats)]
lib.aecGetRuntimeStats.restype = cint
lib.aecResetRuntimeStats.argtypes = []
lib.aecResetRuntimeStats.restype = cint
lib.aecGetErrorName.argtypes = [cint]
lib.aecGetErrorName.restype = ct.c_char_p
dev.aecDeviceReset.restype = cint


def ename(c):
    return lib.aecGetErrorName(c).decode()


def reset():
    dev.aecDeviceReset()
    lib.aecResetRuntimeStats()


def alloc(n):
    p = u64()
    rc = lib.aecAlloc(ct.byref(p), n)
    return rc, p.value


def free(p):
    return lib.aecFree(u64(p))


def h2d(d, host, n):
    return lib.aecCopyH2D(u64(d), host, n)


def d2h(host, d, n):
    return lib.aecCopyD2H(host, u64(d), n)


def launch(kid, grid, block, args, args_size, stream):
    return lib.aecLaunch(kid, grid, block, args, args_size, stream)


results = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    line = f"[{tag}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    results.append(bool(cond))


def fill(count):
    a = [float(i) for i in range(count)]
    b = [float(count - i) for i in range(count)]
    return a, b


def expected(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def close(c, e):
    return c is not None and len(c) == len(e) and all(abs(c[i] - e[i]) < 1e-5 for i in range(len(e)))


def vadd_run(count, grid, block, stream=None):
    """Alloc A/B/C, fill + copy A/B, launch, sync if async, return (rc, c_list, a, b)."""
    reset()
    a, b = fill(count)
    a_arr = (ct.c_float * count)(*a)
    b_arr = (ct.c_float * count)(*b)
    c_arr = (ct.c_float * count)()
    _, da = alloc(4 * count)
    _, db = alloc(4 * count)
    _, dc = alloc(4 * count)
    h2d(da, a_arr, 4 * count)
    h2d(db, b_arr, 4 * count)
    args = VectorAddArgs(da, db, dc, count)
    rc = launch(KID_VADD, grid, block, ct.byref(args), PARAM_BYTES, stream)
    if stream is not None:
        lib.aecStreamSync(stream)
    if rc == SUCCESS:
        d2h(c_arr, dc, 4 * count)
        c_list = list(c_arr)
    else:
        c_list = None
    free(da); free(db); free(dc)
    return rc, c_list, a, b


print("=== CheapSeek R104 probe ===")
print()

# C1: basic correctness, null stream synchronous (count=5, 1 block of 32)
rc, c, a, b = vadd_run(5, Dim3(1, 1, 1), Dim3(32, 1, 1))
check("basic Vector Add correctness, null stream synchronous (count=5)",
      rc == SUCCESS and close(c, expected(a, b)),
      f"rc={ename(rc)} c={c[:5] if c else None}")

# C2: minimal grid/block all dims = 1
rc, c, a, b = vadd_run(1, Dim3(1, 1, 1), Dim3(1, 1, 1))
check("minimal grid(1,1,1) block(1,1,1) count=1",
      rc == SUCCESS and close(c, expected(a, b)),
      f"rc={ename(rc)} c={c[:1] if c else None}")

# C3: block volume = 1024 accepted (32x32x1; x-lane covers 32 elements)
rc, c, a, b = vadd_run(32, Dim3(1, 1, 1), Dim3(32, 32, 1))
check("block volume = 1024 (32x32x1) accepted & correct",
      rc == SUCCESS and close(c, expected(a, b)),
      f"rc={ename(rc)}")

# C4: block volume > 1024 rejected (33x32x1 = 1056)
reset()
_, da = alloc(16); _, db = alloc(16); _, dc = alloc(16)
args = VectorAddArgs(da, db, dc, 1)
rc = launch(KID_VADD, Dim3(1, 1, 1), Dim3(33, 32, 1), ct.byref(args), PARAM_BYTES, None)
check("block volume > 1024 (33x32x1=1056) -> INVALID_ARGUMENT",
      rc == INV_ARG, f"rc={ename(rc)}")
free(da); free(db); free(dc)

# C5: grid.x = 0 rejected
reset()
_, da = alloc(16); _, db = alloc(16); _, dc = alloc(16)
args = VectorAddArgs(da, db, dc, 1)
rc = launch(KID_VADD, Dim3(0, 1, 1), Dim3(1, 1, 1), ct.byref(args), PARAM_BYTES, None)
check("grid.x = 0 -> INVALID_ARGUMENT", rc == INV_ARG, f"rc={ename(rc)}")
free(da); free(db); free(dc)

# C6: block.y = 0 rejected
reset()
_, da = alloc(16); _, db = alloc(16); _, dc = alloc(16)
args = VectorAddArgs(da, db, dc, 1)
rc = launch(KID_VADD, Dim3(1, 1, 1), Dim3(1, 0, 1), ct.byref(args), PARAM_BYTES, None)
check("block.y = 0 -> INVALID_ARGUMENT", rc == INV_ARG, f"rc={ename(rc)}")
free(da); free(db); free(dc)

# C7: args = NULL rejected
reset()
_, da = alloc(16); _, db = alloc(16); _, dc = alloc(16)
rc = launch(KID_VADD, Dim3(1, 1, 1), Dim3(1, 1, 1), None, PARAM_BYTES, None)
check("args = NULL -> INVALID_ARGUMENT", rc == INV_ARG, f"rc={ename(rc)}")
free(da); free(db); free(dc)

# C8: args_size must be exact (16 and 64 both rejected)
reset()
_, da = alloc(16); _, db = alloc(16); _, dc = alloc(16)
args = VectorAddArgs(da, db, dc, 1)
rc16 = launch(KID_VADD, Dim3(1, 1, 1), Dim3(1, 1, 1), ct.byref(args), 16, None)
rc64 = launch(KID_VADD, Dim3(1, 1, 1), Dim3(1, 1, 1), ct.byref(args), 64, None)
check("args_size 16 and 64 both -> INVALID_ARGUMENT (must be exact 32)",
      rc16 == INV_ARG and rc64 == INV_ARG,
      f"rc16={ename(rc16)} rc64={ename(rc64)}")
free(da); free(db); free(dc)

# C9: unknown kernel ID -> NOT_SUPPORTED
reset()
_, da = alloc(16); _, db = alloc(16); _, dc = alloc(16)
args = VectorAddArgs(da, db, dc, 1)
rc = launch(999, Dim3(1, 1, 1), Dim3(1, 1, 1), ct.byref(args), PARAM_BYTES, None)
check("unknown kernel ID 999 -> NOT_SUPPORTED", rc == NOT_SUPPORTED, f"rc={ename(rc)}")
free(da); free(db); free(dc)

# C10: async launch copies args (overwrite struct after launch -> original result)
reset()
count = 64
a, b = fill(count)
a_arr = (ct.c_float * count)(*a)
b_arr = (ct.c_float * count)(*b)
c_arr = (ct.c_float * count)()
_, da = alloc(4 * count); _, db = alloc(4 * count); _, dc = alloc(4 * count)
h2d(da, a_arr, 4 * count); h2d(db, b_arr, 4 * count)
st = vp()
lib.aecStreamCreate(ct.byref(st))
args = VectorAddArgs(da, db, dc, count)
rc = launch(KID_VADD, Dim3(2, 1, 1), Dim3(32, 1, 1), ct.byref(args), PARAM_BYTES, st)
args.a = 0; args.b = 0; args.c = 0; args.count = 0   # clobber after enqueue
lib.aecStreamSync(st)
if rc == SUCCESS:
    d2h(c_arr, dc, 4 * count)
    c_list = list(c_arr)
else:
    c_list = None
check("async launch copies args (clobber after launch -> original result)",
      rc == SUCCESS and close(c_list, expected(a, b)),
      f"rc={ename(rc)}")
lib.aecStreamDestroy(st)
free(da); free(db); free(dc)

# C11: freed device pointer in args -> ISA_TRAP (device enforces span, runtime does not pre-check)
reset()
count = 4
a, b = fill(count)
a_arr = (ct.c_float * count)(*a)
b_arr = (ct.c_float * count)(*b)
_, da = alloc(4 * count); _, db = alloc(4 * count); _, dc = alloc(4 * count)
h2d(da, a_arr, 4 * count); h2d(db, b_arr, 4 * count)
free(da)                                              # A now stale
args = VectorAddArgs(da, db, dc, count)
rc = launch(KID_VADD, Dim3(1, 1, 1), Dim3(32, 1, 1), ct.byref(args), PARAM_BYTES, None)
check("freed device ptr in args -> ISA_TRAP (device enforces, no host pre-check)",
      rc == ISA_TRAP, f"rc={ename(rc)}")
free(db); free(dc)

# C12: stats accounting — 3 launches increment submitted_commands & isa_launches
reset()
lib.aecResetRuntimeStats()
st = RuntimeStats()
lib.aecGetRuntimeStats(ct.byref(st))
sub0 = st.submitted_commands
isa0 = st.isa_launches
_, da = alloc(16); _, db = alloc(16); _, dc = alloc(16)
for _ in range(3):
    args = VectorAddArgs(da, db, dc, 1)
    launch(KID_VADD, Dim3(1, 1, 1), Dim3(1, 1, 1), ct.byref(args), PARAM_BYTES, None)
lib.aecGetRuntimeStats(ct.byref(st))
check("stats: submitted_commands & isa_launches increment after 3 launches",
      st.submitted_commands >= sub0 + 3 and st.isa_launches >= isa0 + 3,
      f"submitted {sub0}->{st.submitted_commands}  isa_launches {isa0}->{st.isa_launches}")
free(da); free(db); free(dc)

print()
print(f"=== {sum(results)}/{len(results)} checks passed ===")
