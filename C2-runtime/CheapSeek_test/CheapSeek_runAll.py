#!/usr/bin/env python3
"""CheapSeek master runner — runs all CheapSeek_r*.py probes in this directory.

Usage: python3 CheapSeek_test/CheapSeek_runAll.py
"""
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TIMEOUT_SEC = 30


def find_probes():
    """Return all CheapSeek_r*.py files sorted by requirement number."""
    probes = sorted(HERE.glob("CheapSeek_r*.py"))
    probes = [p for p in probes if p.name != "CheapSeek_runAll.py"]
    return probes


def run_one(script):
    """Run a probe script, return (name, passed, summary_line, elapsed_sec)."""
    name = script.stem.replace("CheapSeek_", "")
    start = time.monotonic()
    try:
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True, text=True, timeout=TIMEOUT_SEC, cwd=HERE.parent,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - start
        return name, False, "TIMEOUT", elapsed
    except Exception as exc:
        elapsed = time.monotonic() - start
        return name, False, "CRASH: " + str(exc), elapsed

    elapsed = time.monotonic() - start
    combined = proc.stdout + proc.stderr

    m = re.search(r"===\s*(\d+)\s*/\s*(\d+)\s*checks\s+passed\s*===", combined)
    if m:
        passed_count = int(m.group(1))
        total_count = int(m.group(2))
        passed = (passed_count == total_count and proc.returncode == 0)
        return name, passed, m.group(0), elapsed

    lines = [l.strip() for l in combined.splitlines() if l.strip()]
    last = lines[-1] if lines else "(no output)"
    return name, False, "NO SUMMARY - last: " + last[:80], elapsed


def main():
    probes = find_probes()
    if not probes:
        print("No CheapSeek_r*.py probes found.")
        return 1

    print("CheapSeek master runner - %d probes" % len(probes))
    print("=" * 64)

    results = []
    for script in probes:
        name, passed, summary, elapsed = run_one(script)
        results.append((name, passed, summary, elapsed))
        tag = "PASS" if passed else "FAIL"
        print("  [%s] %6s  (%.2fs)  %s" % (tag, name, elapsed, summary))

    passed_count = sum(1 for _, p, _, _ in results if p)
    total = len(results)

    print("=" * 64)
    print("  %d/%d probes passed" % (passed_count, total))

    return 0 if passed_count == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
