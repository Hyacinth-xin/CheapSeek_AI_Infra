#!/usr/bin/env python3
"""Kernel-image policy agent — selects the fastest legal GEMM variant.

The grader describes each candidate by the constraints it requires:
  - divisibility: M, N and K must each be divisible by this
  - alignment:    minimum byte alignment of the operand data
  - workspace:    workspace bytes the variant needs

A candidate is legal only when the request satisfies all three constraints.
The device (aecDeviceEvaluateKernel) enforces the same checks, so returning
an illegal candidate makes the grader reject the case (0/10). We therefore
ONLY ever return a fully-legal candidate.

Among legal candidates we pick the fastest by the fixed hierarchy
  vectorized > tiled > naive
which is the AEC image cycle ordering for every dtype (int4 and fp32 share
identical per-variant cycles; vectorized <= tiled <= naive).

naive (divisibility=1, alignment=1, workspace=0) is legal for any valid
request, so a legal candidate always exists when naive is offered. If it is
not offered and nothing else is legal, every choice fails — we then return
the first candidate to avoid crashing (still 0/10, but no exit-2 abort).
"""

import json
import sys

HIERARCHY = ["vectorized", "tiled", "naive"]


def _legal(candidate: dict, m: int, n: int, k: int,
           alignment: int, workspace: int) -> bool:
    """True iff the request satisfies this candidate's constraints."""
    div = candidate.get("divisibility", 1)
    if div and (m % div or n % div or k % div):
        return False
    if alignment < candidate.get("alignment", 1):
        return False
    if workspace < candidate.get("workspace", 0):
        return False
    return True


def decide(request: dict) -> dict:
    candidates = request.get("candidates", [])
    m = request.get("m", 0)
    n = request.get("n", 0)
    k = request.get("k", 0)
    alignment = request.get("alignment", 0)
    workspace = request.get("workspace", 0)

    # Fastest legal candidate by hierarchy (vectorized > tiled > naive).
    for variant in HIERARCHY:
        for c in candidates:
            if c.get("id") == variant and _legal(c, m, n, k, alignment, workspace):
                return {"kernel_id": variant}

    # No fully-legal candidate. naive is legal for any valid request, so prefer
    # it if offered; otherwise return the first candidate rather than crash.
    for c in candidates:
        if c.get("id") == "naive":
            return {"kernel_id": "naive"}
    if candidates:
        return {"kernel_id": candidates[0].get("id")}
    raise SystemExit(2)


def main() -> None:
    request = json.load(sys.stdin)
    response = decide(request)
    json.dump(response, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
