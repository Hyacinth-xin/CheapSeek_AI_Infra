#!/usr/bin/env python3
"""CheapSeek R101 probe — targets hidden-test directions beyond the public suite.

Directions covered (CLAUDE.md 十一 + 十二.4):
  - null input params: DeviceCount(NULL), DeviceInfo(0, NULL)
  - successful call does NOT clear last error (spec docs/02 §1)
  - GetLastError clears / PeekAtLastError does not
  - aecGetErrorName stability for all known + unknown enum values
  - multi-thread TLS isolation with DISTINCT error codes per thread (>2 threads)
  - concurrent PeekAtLastError thread safety

Standalone: loads libaec.so + libaec_device.so relative to this file.
Run: python3 test/CheapSeek_r101.py
"""
import ctypes as ct
import threading
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


class DeviceInfoData(ct.Structure):
    _fields_ = [
        ("abi_version", u32),
        ("name", ct.c_char * 64),
        ("memory_bytes", u64),
        ("dma_channels", u32),
        ("max_threads_per_block", u32),
        ("isa_version", u32),
        ("isa_profile", u32),
        ("max_parameter_bytes", u32),
        ("reserved", u32),
    ]


class Dim3(ct.Structure):
    _fields_ = [("x", u32), ("y", u32), ("z", u32)]


lib.aecDeviceCount.argtypes = [ct.POINTER(cint)]
lib.aecDeviceCount.restype = cint
lib.aecDeviceInfo.argtypes = [cint, ct.POINTER(DeviceInfoData)]
lib.aecDeviceInfo.restype = cint
lib.aecGetLastError.argtypes = []
lib.aecGetLastError.restype = cint
lib.aecPeekAtLastError.argtypes = []
lib.aecPeekAtLastError.restype = cint
lib.aecGetErrorName.argtypes = [cint]
lib.aecGetErrorName.restype = ct.c_char_p
lib.aecStreamSync.argtypes = [vp]
lib.aecStreamSync.restype = cint
lib.aecCopyH2D.argtypes = [u64, vp, sz]
lib.aecCopyH2D.restype = cint
lib.aecLaunch.argtypes = [u32, Dim3, Dim3, vp, sz, vp]
lib.aecLaunch.restype = cint
dev.aecDeviceReset.restype = cint


def ename(c):
    return lib.aecGetErrorName(c).decode()


def reset():
    dev.aecDeviceReset()
    lib.aecGetLastError()  # clear calling thread TLS


results = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    line = f"[{tag}] {name}"
    if detail:
        line += f"  ({detail})"
    print(line)
    results.append(bool(cond))


print(f"=== CheapSeek R101 probe ===")
print()

# C1: DeviceCount basic
reset()
cnt = cint(0)
rc = lib.aecDeviceCount(ct.byref(cnt))
check("DeviceCount -> SUCCESS, count==1",
      rc == SUCCESS and cnt.value == 1,
      f"rc={ename(rc)} count={cnt.value}")

# C2: DeviceCount(NULL)  [null input — hidden direction]
reset()
rc = lib.aecDeviceCount(None)
check("DeviceCount(NULL) -> INVALID_ARGUMENT", rc == INV_ARG, f"rc={ename(rc)}")

# C3: DeviceInfo(0) valid + metadata
reset()
info = DeviceInfoData()
rc = lib.aecDeviceInfo(0, ct.byref(info))
devname = info.name.split(b"\x00", 1)[0].decode(errors="replace")
check("DeviceInfo(0) -> SUCCESS, metadata sane",
      rc == SUCCESS and info.memory_bytes == 64 * 1024 * 1024 and info.dma_channels == 2,
      f"rc={ename(rc)} name={devname!r} mem={info.memory_bytes} dma={info.dma_channels}")

# C4: DeviceInfo(1) -> INV_ARG (only 1 device)
reset()
rc = lib.aecDeviceInfo(1, ct.byref(info))
check("DeviceInfo(1) -> INVALID_ARGUMENT (only 1 device)", rc == INV_ARG, f"rc={ename(rc)}")

# C5: DeviceInfo(0, NULL)  [null info — hidden direction]
reset()
rc = lib.aecDeviceInfo(0, None)
check("DeviceInfo(0, NULL) -> INVALID_ARGUMENT", rc == INV_ARG, f"rc={ename(rc)}")

# C6: successful call does NOT clear last error (spec docs/02 §1)
reset()
lib.aecDeviceInfo(1, ct.byref(info))        # sets INV_ARG
peek_before = lib.aecPeekAtLastError()
cnt = cint(0)
rc = lib.aecDeviceCount(ct.byref(cnt))      # SUCCESSFUL call
peek_after = lib.aecPeekAtLastError()
check("successful call does NOT clear last error (spec)",
      rc == SUCCESS and peek_before == INV_ARG and peek_after == INV_ARG,
      f"cnt_rc={ename(rc)} peek_before={ename(peek_before)} peek_after={ename(peek_after)}")

# C7: Peek keeps error, Get clears it
reset()
lib.aecDeviceInfo(1, ct.byref(info))        # INV_ARG
p1 = lib.aecPeekAtLastError()
p2 = lib.aecPeekAtLastError()
g1 = lib.aecGetLastError()
g2 = lib.aecGetLastError()
check("Peek keeps error, Get clears it",
      p1 == INV_ARG and p2 == INV_ARG and g1 == INV_ARG and g2 == SUCCESS,
      f"peek={ename(p1)},{ename(p2)} get={ename(g1)},{ename(g2)}")

# C8: ErrorName stability for all known + unknown
known = {
    SUCCESS: "AEC_SUCCESS", INV_ARG: "AEC_ERROR_INVALID_ARGUMENT",
    OOM: "AEC_ERROR_OUT_OF_MEMORY", INV_HANDLE: "AEC_ERROR_INVALID_HANDLE",
    INV_ADDR: "AEC_ERROR_INVALID_ADDRESS", NOT_READY: "AEC_ERROR_NOT_READY",
    NOT_SUPPORTED: "AEC_ERROR_NOT_SUPPORTED", DEVICE: "AEC_ERROR_DEVICE",
    INTERNAL: "AEC_ERROR_INTERNAL", ISA_TRAP: "AEC_ERROR_ISA_TRAP",
}
bad = []
for v, exp in known.items():
    n1 = lib.aecGetErrorName(v).decode()
    n2 = lib.aecGetErrorName(v).decode()
    if n1 != exp or n2 != exp:
        bad.append(f"{v}->{n1}(exp {exp})")
for uv in [10, 100, -1, 255, 9999]:
    n = lib.aecGetErrorName(uv).decode()
    if n != "AEC_ERROR_UNKNOWN":
        bad.append(f"unknown {uv}->{n}")
check("ErrorName stable for all known enums, UNKNOWN for others",
      not bad, "; ".join(bad) if bad else "all 10 known + 5 unknown correct")

# C9: multi-thread TLS isolation with DISTINCT error codes per thread
N = 8
barrier = threading.Barrier(N)
tres = [None] * N
tlock = threading.Lock()
KINDS = [
    ("DeviceInfo(1)", INV_ARG,
     lambda: lib.aecDeviceInfo(1, ct.byref(DeviceInfoData()))),
    ("StreamSync(NULL)", INV_HANDLE,
     lambda: lib.aecStreamSync(None)),
    ("CopyH2D(dev=0)", INV_ADDR,
     lambda: lib.aecCopyH2D(0, (ct.c_char * 16)(), 16)),
    ("Launch(kid=999)", NOT_SUPPORTED,
     lambda: lib.aecLaunch(999, Dim3(1, 1, 1), Dim3(1, 1, 1), (ct.c_uint8 * 32)(), 32, None)),
]


def worker(tid):
    label, expected, trigger = KINDS[tid % 4]
    trigger()                    # set this thread's TLS to `expected`
    barrier.wait()               # all threads have set their error
    peek = lib.aecPeekAtLastError()  # read own TLS
    with tlock:
        tres[tid] = (tid, label, expected, peek, peek == expected)


reset()
threads = [threading.Thread(target=worker, args=(i,)) for i in range(N)]
for t in threads:
    t.start()
for t in threads:
    t.join()
ok_tls = all(r[4] for r in tres)
detail_tls = "; ".join(
    f"t{r[0]} {r[1]} exp={ename(r[2])} got={ename(r[3])}" for r in tres)
check(f"TLS isolation: {N} threads, distinct errors, no leakage",
      ok_tls, detail_tls)

print()
print(f"=== {sum(results)}/{len(results)} checks passed ===")
