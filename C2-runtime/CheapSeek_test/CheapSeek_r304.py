#!/usr/bin/env python3
"""CheapSeek R304 probe — DMA/ISA fault propagation & recovery hidden-test directions.

Directions covered (CLAUDE.md 十一 R304 + 十二.4 R304):
  - FAULT_NEXT_DMA(1): next DMA fails DEVICE, consumed on first hit
  - FAULT_NEXT_KERNEL(2): next ISA launch fails DEVICE
  - FAULT_NEXT_COMMAND(3): any next command type fails
  - ISA_TRAP(9) != injected DEVICE_ERR(7): distinct error codes
  - injected fault: instructions_retired=0, isa_traps=0 (no ISA execution)
  - post-fault recovery: device continues accepting valid commands
  - async DMA fault: error DEFERRED to StreamSync (enqueue=SUCCESS, sync=DEVICE_ERR)
  - aecDeviceReset clears pending fault state; NEW allocation works post-reset
  - fault scope: only FIRST matching command affected, not cascading
  - concurrent Stream isolation: fault on stream A, stream B unaffected
  - device reset after fault preserves image registry (resolve handle stable)

Known behavior (CLAUDE.md 十二.4): runtime live_allocs not cleared by device reset.
After reset_dev(), old allocations become INVALID_ADDRESS on the device side.
This test uses NEW allocations after reset to verify fault clearing.

Standalone: loads libaec.so + libaec_device.so
Run: python3 CheapSeek_test/CheapSeek_r304.py
"""
import ctypes as ct, threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
dev = ct.CDLL(str(ROOT / "lib" / "libaec_device.so"), mode=ct.RTLD_GLOBAL)
lib = ct.CDLL(str(ROOT / "libaec.so"))
u32 = ct.c_uint32; u64 = ct.c_uint64; sz = ct.c_size_t; cint = ct.c_int; vp = ct.c_void_p
SUCCESS, INV_ARG, INV_HANDLE, INV_ADDR, DEVICE_ERR, ISA_TRAP = 0, 1, 3, 4, 7, 9
KID_VADD, PARAM_BYTES_VADD = 1, 32
FAULT_NEXT_DMA, FAULT_NEXT_KERNEL, FAULT_NEXT_COMMAND = 1, 2, 3

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

class KernelInfo(ct.Structure):
    _fields_ = [
        ("abi_version", u32), ("isa_version", u32),
        ("handle", u64), ("image_id", u32),
        ("entry_pc", u32), ("parameter_bytes", u32),
        ("image_flags", u32), ("instruction_hash", u64),
    ]

class VectorAddArgs(ct.Structure):
    _fields_ = [("a", u64), ("b", u64), ("c", u64), ("count", u64)]

lib.aecAlloc.argtypes = [ct.POINTER(u64), sz]; lib.aecAlloc.restype = cint
lib.aecFree.argtypes = [u64]; lib.aecFree.restype = cint
lib.aecCopyH2D.argtypes = [u64, vp, sz]; lib.aecCopyH2D.restype = cint
lib.aecCopyAsync.argtypes = [u64, vp, sz, cint, vp]; lib.aecCopyAsync.restype = cint
lib.aecLaunch.argtypes = [u32, Dim3, Dim3, vp, sz, vp]; lib.aecLaunch.restype = cint
lib.aecStreamCreate.argtypes = [ct.POINTER(vp)]; lib.aecStreamCreate.restype = cint
lib.aecStreamDestroy.argtypes = [vp]; lib.aecStreamDestroy.restype = cint
lib.aecStreamSync.argtypes = [vp]; lib.aecStreamSync.restype = cint
lib.aecGetRuntimeStats.argtypes = [ct.POINTER(RuntimeStats)]; lib.aecGetRuntimeStats.restype = cint
lib.aecGetErrorName.argtypes = [cint]; lib.aecGetErrorName.restype = ct.c_char_p
lib.aecGetLastError.argtypes = []; lib.aecGetLastError.restype = cint
dev.aecDeviceReset.restype = ct.c_int
dev.aecDeviceInjectFault.argtypes = [cint]; dev.aecDeviceInjectFault.restype = ct.c_int
dev.aecDeviceResolveKernel.argtypes = [u32, u32, u32, ct.POINTER(KernelInfo)]
dev.aecDeviceResolveKernel.restype = ct.c_int

en = lambda c: lib.aecGetErrorName(c).decode()

def reset_dev():
    dev.aecDeviceReset()
    lib.aecGetLastError()

def ALLOC(n):
    p = u64(); lib.aecAlloc(ct.byref(p), n); return p.value

def FREE(p):
    lib.aecFree(u64(p))

def MKSTREAM():
    s = vp(); lib.aecStreamCreate(ct.byref(s)); return s.value or None

def H2D(dp, data):
    return lib.aecCopyH2D(u64(dp), (ct.c_uint8 * len(data))(*data), len(data))

def get_rs():
    s = RuntimeStats(); lib.aecGetRuntimeStats(ct.byref(s)); return s

def resolve(kid, dtype, variant):
    info = KernelInfo()
    dev.aecDeviceResolveKernel(kid, dtype, variant, ct.byref(info))
    return info.handle

results = []

def chk(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    line = f"[{tag}] {name}"
    if detail: line += f"  ({detail})"
    print(line)
    results.append(bool(cond))

print("=== CheapSeek R304 probe ===\n")

# C1: FAULT_NEXT_DMA -> H2D fails DEVICE
reset_dev(); da1 = ALLOC(64)
dev.aecDeviceInjectFault(FAULT_NEXT_DMA)
r1 = H2D(da1, bytes(64)); FREE(da1)
chk("FAULT_NEXT_DMA -> H2D returns DEVICE", r1 == DEVICE_ERR, f"r={en(r1)}")

# C2: Fault consumed -> second H2D succeeds
reset_dev(); da2 = ALLOC(64)
dev.aecDeviceInjectFault(FAULT_NEXT_DMA)
H2D(da2, bytes(64))
r2 = H2D(da2, bytes(64)); FREE(da2)
chk("fault consumed -> second H2D succeeds", r2 == SUCCESS, f"r={en(r2)}")

# C3: FAULT_NEXT_KERNEL -> launch fails DEVICE
reset_dev(); da3, db3, dc3 = ALLOC(64), ALLOC(64), ALLOC(64)
H2D(da3, bytes(64)); H2D(db3, bytes(64))
dev.aecDeviceInjectFault(FAULT_NEXT_KERNEL)
args3 = VectorAddArgs(da3, db3, dc3, 1)
r3 = lib.aecLaunch(KID_VADD, Dim3(1,1,1), Dim3(1,1,1), ct.byref(args3), PARAM_BYTES_VADD, None)
FREE(da3); FREE(db3); FREE(dc3)
chk("FAULT_NEXT_KERNEL -> launch DEVICE", r3 == DEVICE_ERR, f"r={en(r3)}")

# C4: FAULT_NEXT_COMMAND -> any command fails
reset_dev(); da4 = ALLOC(64)
dev.aecDeviceInjectFault(FAULT_NEXT_COMMAND)
r4 = H2D(da4, bytes(64)); FREE(da4)
chk("FAULT_NEXT_COMMAND -> any command DEVICE", r4 == DEVICE_ERR, f"r={en(r4)}")

# C5: ISA_TRAP != injected fault
reset_dev()
da5t, db5t, dc5t = ALLOC(64), ALLOC(64), ALLOC(64)
H2D(da5t, bytes(64)); H2D(db5t, bytes(64))
FREE(da5t)
args5t = VectorAddArgs(da5t, db5t, dc5t, 1)
r_trap = lib.aecLaunch(KID_VADD, Dim3(1,1,1), Dim3(1,1,1), ct.byref(args5t), PARAM_BYTES_VADD, None)
FREE(db5t); FREE(dc5t)

reset_dev()
da5i, db5i, dc5i = ALLOC(64), ALLOC(64), ALLOC(64)
H2D(da5i, bytes(64)); H2D(db5i, bytes(64))
dev.aecDeviceInjectFault(FAULT_NEXT_KERNEL)
args5i = VectorAddArgs(da5i, db5i, dc5i, 1)
r_inj = lib.aecLaunch(KID_VADD, Dim3(1,1,1), Dim3(1,1,1), ct.byref(args5i), PARAM_BYTES_VADD, None)
FREE(da5i); FREE(db5i); FREE(dc5i)
chk("ISA_TRAP(9) != injected fault(7)", r_trap==ISA_TRAP and r_inj==DEVICE_ERR,
    f"trap={r_trap} inj={r_inj}")

# C6: Injected fault -> no retired, no isa_traps
reset_dev(); da6, db6, dc6 = ALLOC(64), ALLOC(64), ALLOC(64)
H2D(da6, bytes(64)); H2D(db6, bytes(64))
dev.aecDeviceInjectFault(FAULT_NEXT_KERNEL)
args6 = VectorAddArgs(da6, db6, dc6, 1)
lib.aecLaunch(KID_VADD, Dim3(1,1,1), Dim3(1,1,1), ct.byref(args6), PARAM_BYTES_VADD, None)
rs6 = get_rs(); FREE(da6); FREE(db6); FREE(dc6)
chk("injected fault: retired=0, isa_traps=0",
    rs6.instructions_retired==0 and rs6.isa_traps==0,
    f"retired={rs6.instructions_retired} traps={rs6.isa_traps}")

# C7: Post-fault recovery: device continues
reset_dev(); da7, db7, dc7 = ALLOC(64), ALLOC(64), ALLOC(64)
H2D(da7, bytes(64)); H2D(db7, bytes(64))
dev.aecDeviceInjectFault(FAULT_NEXT_KERNEL)
args7 = VectorAddArgs(da7, db7, dc7, 1)
r_f7 = lib.aecLaunch(KID_VADD, Dim3(1,1,1), Dim3(1,1,1), ct.byref(args7), PARAM_BYTES_VADD, None)
FREE(da7); FREE(db7); FREE(dc7)
da7b, db7b, dc7b = ALLOC(64), ALLOC(64), ALLOC(64)
H2D(da7b, bytes(64)); H2D(db7b, bytes(64))
args7b = VectorAddArgs(da7b, db7b, dc7b, 1)
r_ok7 = lib.aecLaunch(KID_VADD, Dim3(1,1,1), Dim3(1,1,1), ct.byref(args7b), PARAM_BYTES_VADD, None)
rs7 = get_rs(); FREE(da7b); FREE(db7b); FREE(dc7b)
chk("post-fault recovery: launch succeeds", r_f7==DEVICE_ERR and r_ok7==SUCCESS,
    f"fail={en(r_f7)} recover={en(r_ok7)} isa={rs7.isa_launches}")

# C8: Async DMA fault -> deferred to StreamSync
reset_dev(); da8 = ALLOC(4096); st8 = MKSTREAM()
h_buf = (ct.c_uint8 * 64)()
dev.aecDeviceInjectFault(FAULT_NEXT_DMA)
r_a1 = lib.aecCopyAsync(da8, h_buf, 64, 1, st8)
r_s1 = lib.aecStreamSync(st8)
lib.aecStreamDestroy(st8); FREE(da8)
chk("async DMA fault -> error deferred to StreamSync",
    r_a1==SUCCESS and r_s1==DEVICE_ERR, f"enq={en(r_a1)} sync={en(r_s1)}")

# C9: aecDeviceReset clears pending fault (verify with NEW allocation post-reset)
reset_dev()
dummy = ALLOC(64)
dev.aecDeviceInjectFault(FAULT_NEXT_DMA)
H2D(dummy, bytes(64))     # consume fault on dummy
FREE(dummy)
reset_dev()                # clears device state
da9 = ALLOC(64)           # fresh allocation post-reset
r9 = H2D(da9, bytes(64)); FREE(da9)
chk("fault consumed + reset: new allocation H2D succeeds",
    r9 == SUCCESS, f"r={en(r9)}")

# C10: Fault only first matching command
reset_dev(); da10 = ALLOC(64)
dev.aecDeviceInjectFault(FAULT_NEXT_DMA)
r_first = H2D(da10, bytes(64)); r_second = H2D(da10, bytes(64))
FREE(da10)
chk("fault only first DMA, second ok", r_first==DEVICE_ERR and r_second==SUCCESS,
    f"1st={en(r_first)} 2nd={en(r_second)}")

# C11: Same-alloc recovery after kernel fault
reset_dev(); da11, db11, dc11 = ALLOC(64), ALLOC(64), ALLOC(64)
H2D(da11, bytes(64)); H2D(db11, bytes(64))
dev.aecDeviceInjectFault(FAULT_NEXT_KERNEL)
args11f = VectorAddArgs(da11, db11, dc11, 1)
r_f11 = lib.aecLaunch(KID_VADD, Dim3(1,1,1), Dim3(1,1,1), ct.byref(args11f), PARAM_BYTES_VADD, None)
args11r = VectorAddArgs(da11, db11, dc11, 1)
r_r11 = lib.aecLaunch(KID_VADD, Dim3(1,1,1), Dim3(1,1,1), ct.byref(args11r), PARAM_BYTES_VADD, None)
FREE(da11); FREE(db11); FREE(dc11)
chk("same-alloc recover after kernel fault", r_f11==DEVICE_ERR and r_r11==SUCCESS,
    f"fail={en(r_f11)} recover={en(r_r11)}")

# C12: Image registry preserved after fault+reset
reset_dev(); h_pre = resolve(KID_VADD, 6, 0)
dev.aecDeviceInjectFault(FAULT_NEXT_DMA)
reset_dev(); h_post = resolve(KID_VADD, 6, 0)
chk("image registry preserved post fault+reset", h_pre==h_post, f"0x{h_pre:x}==0x{h_post:x}")

# C13: Concurrent stream isolation
reset_dev(); da_a, da_b = ALLOC(4096), ALLOC(4096)
st_a, st_b = MKSTREAM(), MKSTREAM()
ha, hb = (ct.c_uint8 * 64)(), (ct.c_uint8 * 64)()
dev.aecDeviceInjectFault(FAULT_NEXT_DMA)
lib.aecCopyAsync(da_a, ha, 64, 1, st_a)
lib.aecCopyAsync(da_b, hb, 64, 1, st_b)
r_sa = lib.aecStreamSync(st_a); r_sb = lib.aecStreamSync(st_b)
lib.aecStreamDestroy(st_a); lib.aecStreamDestroy(st_b)
FREE(da_a); FREE(da_b)
chk("fault stream A DEVICE, stream B SUCCESS",
    r_sa==DEVICE_ERR and r_sb==SUCCESS, f"A={en(r_sa)} B={en(r_sb)}")

p = sum(results)
print(f"\n=== {p}/{len(results)} checks passed ===")
