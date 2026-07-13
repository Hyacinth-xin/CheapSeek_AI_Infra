#!/usr/bin/env python3
"""Kernel-image policy agent — selects the fastest valid GEMM variant.

Variant hierarchy (fastest → slowest):
  vectorized > tiled > naive

For each candidate we check shape_ok, alignment_ok, and workspace_ok.
If no candidate passes all three, we fall back through a relaxed chain:
  1. best fully-valid (all three flags true)
  2. best ignoring workspace_ok
  3. best ignoring alignment_ok and workspace_ok
  4. "naive" if present in the candidate list at all
"""

import json
import sys

HIERARCHY = ["vectorized", "tiled", "naive"]


def pick(candidates: list, checks: list) -> str | None:
    """Return the highest-hierarchy candidate that passes all given checks."""
    for variant in HIERARCHY:
        for c in candidates:
            if c["id"] == variant and all(c.get(k) for k in checks):
                return variant
    return None


def decide(request: dict) -> dict:
    candidates = request.get("candidates", [])

    # Tier 1: fully valid (all three flags)
    kid = pick(candidates, ["shape_ok", "alignment_ok", "workspace_ok"])
    if kid is not None:
        return {"kernel_id": kid}

    # Tier 2: ignore workspace_ok
    kid = pick(candidates, ["shape_ok", "alignment_ok"])
    if kid is not None:
        return {"kernel_id": kid}

    # Tier 3: ignore workspace_ok + alignment_ok (just shape must be ok)
    kid = pick(candidates, ["shape_ok"])
    if kid is not None:
        return {"kernel_id": kid}

    # Tier 4: "naive" by name only (matches original stub behaviour)
    if any(c["id"] == "naive" for c in candidates):
        return {"kernel_id": "naive"}

    raise SystemExit(2)


def main() -> None:
    request = json.load(sys.stdin)
    response = decide(request)
    json.dump(response, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
