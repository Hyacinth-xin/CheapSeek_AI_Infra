#!/usr/bin/env python3
"""CheapSeek R401 probe — DMA Agent correctness and optimality.

Directions covered (CLAUDE.md 十一 + 十二.4):
  - Output schema: exactly 4 fields, valid enum values
  - chunk=1048576 always (max → chunks=1 for bytes ≤ 1MiB)
  - queue_depth = 2 iff concurrency >= 2 (parallelism capped at 2)
  - use_zero_copy = registered (saves 55 setup cycles)
  - channel ∈ {0,1}
  - bytes > 1MiB → multiple chunks but still chunk=1048576
  - concurrency=0 boundary → handled gracefully
  - Provably optimal against DMA cycle formula

Run: python3 CheapSeek_test/CheapSeek_r401.py
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = str(ROOT / "agents" / "dma_agent.py")

LEGAL_CHUNKS = [4096, 65536, 1048576]
LEGAL_QDS = [1, 2, 4, 8]


def run_agent(request: dict) -> dict:
    """Invoke agent as subprocess (stdin→stdout), verify contract."""
    proc = subprocess.run(
        [sys.executable, AGENT],
        input=json.dumps(request),
        capture_output=True, text=True, timeout=5,
    )
    assert proc.returncode == 0, f"agent exit={proc.returncode} stderr={proc.stderr}"
    assert proc.stderr.strip() == "", f"agent stderr not empty: {proc.stderr}"
    response = json.loads(proc.stdout.strip())
    return response


def compute_cycles(nbytes: int, chunk_bytes: int, qd: int,
                   concurrency: int, registered: bool, use_zc: bool,
                   alignment: int) -> int:
    """DMA virtual cycle formula from docs/05."""
    setup = 45 if (registered and use_zc) else 100
    conc_eff = max(concurrency, 1)
    parallelism = min(qd, conc_eff, 2)
    per_elem = ((nbytes + 31) // 32 + parallelism - 1) // parallelism
    chunks = (nbytes + chunk_bytes - 1) // chunk_bytes
    penalty = 13 if alignment < 64 else 0
    return setup + per_elem + 24 * (chunks - 1) + penalty


def brute_optimal(nbytes: int, concurrency: int, registered: bool,
                  alignment: int) -> tuple:
    """Brute-force all legal params, return (min_cycles, best_params)."""
    best_cyc = float("inf")
    best_params = None
    for ck in LEGAL_CHUNKS:
        for qd in LEGAL_QDS:
            for ch in [0, 1]:
                zc = registered
                cyc = compute_cycles(nbytes, ck, qd, concurrency, registered, zc, alignment)
                if cyc < best_cyc:
                    best_cyc = cyc
                    best_params = (ch, ck, qd, zc)
    return best_cyc, best_params


results = []

def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    line = f"[{tag}] {name}"
    if detail: line += f"  ({detail})"
    print(line); results.append(bool(cond))

print("=== CheapSeek R401 probe ===")
print()

# ── C1: Output schema compliance ────────────────────────────────────────
req = {"case_id": 0, "direction": "h2d", "bytes": 4096, "alignment": 64,
       "registered": False, "concurrency": 1}
resp = run_agent(req)
required = {"channel", "chunk_bytes", "queue_depth", "use_zero_copy"}
check("output schema: exactly 4 required fields, no extra keys",
      set(resp.keys()) == required,
      f"keys={set(resp.keys())}")

# ── C2: Valid enum values ──────────────────────────────────────────────
check("channel ∈ {0,1}", resp["channel"] in (0, 1),
      f"ch={resp['channel']}")
check("chunk_bytes ∈ {4096,65536,1048576}", resp["chunk_bytes"] in LEGAL_CHUNKS,
      f"ck={resp['chunk_bytes']}")
check("queue_depth ∈ {1,2,4,8}", resp["queue_depth"] in LEGAL_QDS,
      f"qd={resp['queue_depth']}")
check("use_zero_copy is bool", isinstance(resp["use_zero_copy"], bool),
      f"zc={resp['use_zero_copy']}")

# ── C3: Concurrency=1 → qd=1 ──────────────────────────────────────────
resp = run_agent({"case_id": 1, "direction": "h2d", "bytes": 4096,
                   "alignment": 64, "registered": False, "concurrency": 1})
check("concurrency=1 → queue_depth=1", resp["queue_depth"] == 1,
      f"qd={resp['queue_depth']}")

# ── C4: Concurrency≥2 → qd=2 (parallelism capped at 2) ────────────────
for conc in [2, 4, 8, 64]:
    resp = run_agent({"case_id": 2, "direction": "h2d", "bytes": 4096,
                       "alignment": 64, "registered": False, "concurrency": conc})
    if resp["queue_depth"] != 2:
        break
check("concurrency≥2 → queue_depth=2 (all values)", resp["queue_depth"] == 2,
      f"conc={conc} qd={resp['queue_depth']}")

# ── C5: Registered → zc=true ──────────────────────────────────────────
resp = run_agent({"case_id": 3, "direction": "h2d", "bytes": 4096,
                   "alignment": 64, "registered": True, "concurrency": 1})
check("registered=True → use_zero_copy=True", resp["use_zero_copy"] is True)

# ── C6: Unregistered → zc=false ───────────────────────────────────────
resp = run_agent({"case_id": 4, "direction": "h2d", "bytes": 4096,
                   "alignment": 64, "registered": False, "concurrency": 1})
check("registered=False → use_zero_copy=False", resp["use_zero_copy"] is False)

# ── C7: chunk_bytes always 1048576 ─────────────────────────────────────
for nbytes in [1, 4096, 65536, 524288, 1048576]:
    resp = run_agent({"case_id": 5, "direction": "h2d", "bytes": nbytes,
                       "alignment": 64, "registered": False, "concurrency": 1})
    if resp["chunk_bytes"] != 1048576:
        break
check("chunk_bytes=1048576 for all sizes (1..1MiB)",
      resp["chunk_bytes"] == 1048576,
      f"bytes={nbytes} ck={resp['chunk_bytes']}")

# ── C8: bytes > 1MiB — still chunk=1MiB, multiple chunks ───────────────
bytes_big = 2097152  # 2 MiB
resp = run_agent({"case_id": 6, "direction": "h2d", "bytes": bytes_big,
                   "alignment": 64, "registered": False, "concurrency": 1})
chunks_big = (bytes_big + 1048575) // 1048576
check(f"bytes={bytes_big} → chunk=1048576, chunks={chunks_big}",
      resp["chunk_bytes"] == 1048576,
      f"ck={resp['chunk_bytes']} (expect 1048576, ~{chunks_big} chunks)")

# ── C9: Direction doesn'\''t affect strategy ───────────────────────────
resp_h2d = run_agent({"case_id": 7, "direction": "h2d", "bytes": 65536,
                       "alignment": 64, "registered": True, "concurrency": 2})
resp_d2h = run_agent({"case_id": 8, "direction": "d2h", "bytes": 65536,
                       "alignment": 64, "registered": True, "concurrency": 2})
check("direction (h2d vs d2h) does not affect strategy",
      resp_h2d == resp_d2h,
      f"h2d={resp_h2d} d2h={resp_d2h}")

# ── C10: Concurrency=0 → handled gracefully (not crash) ────────────────
resp = run_agent({"case_id": 9, "direction": "h2d", "bytes": 4096,
                   "alignment": 64, "registered": False, "concurrency": 0})
check("concurrency=0 → valid response (no crash)",
      isinstance(resp["queue_depth"], int) and resp["queue_depth"] in LEGAL_QDS,
      f"qd={resp['queue_depth']}")

# ── C11: No extra stdout noise ─────────────────────────────────────────
# Already verified in run_agent (stderr must be empty, stdout single JSON)
check("agent produces no stderr, single-line JSON stdout", True,
      "verified by run_agent contract")

# ── C12: Optimality proof — agent beats or ties all legal combos ────────
CASES = [
    # (bytes, registered, concurrency, alignment)
    (1, False, 1, 64),
    (4096, False, 1, 64),
    (4096, True, 2, 64),
    (65536, False, 4, 32),        # alignment < 64 → penalty
    (65536, True, 8, 64),
    (524288, True, 2, 64),
    (1048576, False, 1, 64),
    (1048577, True, 2, 64),       # bytes > 1MiB by 1
    (2097152, False, 2, 64),      # 2 MiB
    (2097153, True, 4, 15),       # odd bytes, low alignment
    (1048576, False, 16, 128),    # high concurrency
    (1, True, 0, 1),              # concurrency=0, min alignment
]
opt_ok = 0
for nbytes, registered, conc, align in CASES:
    resp = run_agent({"case_id": 100, "direction": "h2d", "bytes": nbytes,
                       "alignment": align, "registered": registered,
                       "concurrency": conc})
    agent_cyc = compute_cycles(nbytes, resp["chunk_bytes"], resp["queue_depth"],
                                conc, registered, resp["use_zero_copy"], align)
    best_cyc, best_params = brute_optimal(nbytes, conc, registered, align)
    if agent_cyc <= best_cyc:
        opt_ok += 1
    else:
        print(f"  optimality fail: bytes={nbytes} reg={registered} conc={conc} align={align} "
              f"agent_cyc={agent_cyc} best_cyc={best_cyc} best={best_params}")
check(f"optimality: agent optimal for {opt_ok}/{len(CASES)} diverse cases",
      opt_ok == len(CASES))

# ── C13: Agent deterministic — same input, same output ─────────────────
r1 = run_agent({"case_id": 200, "direction": "d2h", "bytes": 100000,
                 "alignment": 64, "registered": True, "concurrency": 4})
r2 = run_agent({"case_id": 200, "direction": "d2h", "bytes": 100000,
                 "alignment": 64, "registered": True, "concurrency": 4})
check("agent deterministic", r1 == r2, f"r1={r1} r2={r2}")

print()
print(f"=== {sum(results)}/{len(results)} checks passed ===")
# ── C14: Extreme boundary cases ─────────────────────────────────────────
import math
EXTREME = [
    # (bytes, registered, concurrency, alignment, why interesting)
    (31, False, 1, 64, "bytes=31, ceil(31/32)=1"),
    (32, True, 1, 64, "bytes=32, exact 32-byte boundary"),
    (33, False, 1, 64, "bytes=33, ceil(33/32)=2"),
    (63, True, 1, 64, "bytes=63, ceil(63/32)=2"),
    (64, False, 1, 64, "bytes=64, ceil(64/32)=2, exact 32 boundary"),
    (1048576 * 10, True, 8, 64, "bytes=10MiB, chunks=10"),
    (4096, False, 2, 64, "conc=2, qd flip from 1→2"),
    (4096, False, 1, 63, "align=63, penalty=13 (just under 64 boundary)"),
    (4096, True, 1, 64, "align=64, penalty=0 (exact penalty boundary)"),
    (1, True, 0, 1, "minimum bytes, conc=0, min alignment"),
    (1048576, True, 64, 128, "1MiB, max conc, high alignment"),
    (1, False, 64, 64, "min bytes, max conc"),
]
opt_ok2 = 0
for nbytes, registered, conc, align, label in EXTREME:
    resp = run_agent({"case_id": 300, "direction": "h2d", "bytes": nbytes,
                       "alignment": align, "registered": registered,
                       "concurrency": conc})
    agent_cyc = compute_cycles(nbytes, resp["chunk_bytes"], resp["queue_depth"],
                                conc, registered, resp["use_zero_copy"], align)
    best_cyc, _ = brute_optimal(nbytes, conc, registered, align)
    if agent_cyc <= best_cyc:
        opt_ok2 += 1
    else:
        print(f"  extreme fail: {label} agent_cyc={agent_cyc} best_cyc={best_cyc}")
check(f"extreme optimality: {opt_ok2}/{len(EXTREME)} extreme boundary cases",
      opt_ok2 == len(EXTREME))

# ── C15: Registered=True should NOT pick zc=False ──────────────────────
# All optimal paths for registered+dma use zc=True. Verify agent never picks zc=False.
zc_ok = True
for registered in [True]:
    for conc in [1, 2, 4, 64]:
        for nbytes in [32, 4096, 65536, 1048576, 2097152]:
            resp = run_agent({"case_id": 400, "direction": "h2d", "bytes": nbytes,
                               "alignment": 64, "registered": registered,
                               "concurrency": conc})
            if resp["use_zero_copy"] is not True:
                zc_ok = False
                print(f"  zc fail: reg=True but zc=False for bytes={nbytes} conc={conc}")
check("registered=True always → zc=True (never suboptimal to skip zc)",
      zc_ok)

print()
print(f"=== {sum(results)}/{len(results)} checks passed ===")
# ── C16: Randomized fuzz — 500 random cases, verify optimality ──────────
import random
random.seed(42)
FUZZ_N = 500
fuzz_fails = 0
for _ in range(FUZZ_N):
    nbytes = random.randint(1, 64 * 1024 * 1024)        # 1 .. 64 MiB
    conc   = random.choice([0, 1, 2, 3, 4, 8, 16, 32, 64])
    reg    = random.choice([True, False])
    align  = random.choice([1, 2, 4, 8, 16, 31, 32, 33, 63, 64, 65, 128, 256])
    resp = run_agent({"case_id": 5000, "direction": "h2d", "bytes": nbytes,
                       "alignment": align, "registered": reg,
                       "concurrency": conc})
    agent_cyc = compute_cycles(nbytes, resp["chunk_bytes"], resp["queue_depth"],
                                conc, reg, resp["use_zero_copy"], align)
    best_cyc, _ = brute_optimal(nbytes, conc, reg, align)
    if agent_cyc > best_cyc:
        fuzz_fails += 1
        # only print first 3 failures to avoid spam
        if fuzz_fails <= 3:
            print(f"  FUZZ FAIL: bytes={nbytes} reg={reg} conc={conc} align={align} "
                  f"agent_cyc={agent_cyc} best_cyc={best_cyc}")
check(f"fuzz optimality: {FUZZ_N - fuzz_fails}/{FUZZ_N} random cases optimal",
      fuzz_fails == 0)

# ── C17: Cycle formula sanity — computed cycles are positive ────────────
# Every legal parameter combo for diverse inputs produces positive cycles
sane = True
for nbytes in [1, 32, 4096, 65536, 1048576, 67108864]:
    for conc in [1, 2, 64]:
        for reg in [True, False]:
            for align in [1, 63, 64]:
                resp = run_agent({"case_id": 6000, "direction": "h2d", "bytes": nbytes,
                                   "alignment": align, "registered": reg,
                                   "concurrency": conc})
                cyc = compute_cycles(nbytes, resp["chunk_bytes"], resp["queue_depth"],
                                      conc, reg, resp["use_zero_copy"], align)
                if cyc <= 0:
                    sane = False
                    print(f"  CYCLE SANITY FAIL: bytes={nbytes} conc={conc} reg={reg} align={align} cyc={cyc}")
check("cycle sanity: all agent outputs produce positive cycles",
      sane)

print()
print(f"=== {sum(results)}/{len(results)} checks passed ===")
