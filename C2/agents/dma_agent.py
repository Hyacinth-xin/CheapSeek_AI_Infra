#!/usr/bin/env python3
"""DMA policy agent — minimises virtual cycles per the device DMA model.

Cycle formula:
  setup + ceil(ceil(bytes/32) / parallelism) + 24*(chunks-1) + alignment_penalty

Decision rules:
  - chunk_bytes = 1048576  (always max → chunks=1 → zero chunk penalty)
  - queue_depth = 2 if concurrency >= 2 else 1  (parallelism capped at 2)
  - use_zero_copy = registered  (saves 55 cycles on setup)
  - channel = 0  (no cycle difference between channels)
"""

import json
import sys


def decide(request: dict) -> dict:
    registered   = request.get("registered", False)
    concurrency  = request.get("concurrency", 1)

    return {
        "channel":       0,
        "chunk_bytes":   1048576,
        "queue_depth":   2 if concurrency >= 2 else 1,
        "use_zero_copy": registered,
    }


def main() -> None:
    request = json.load(sys.stdin)
    response = decide(request)
    json.dump(response, sys.stdout, sort_keys=True)
    sys.stdout.write("\n")


if __name__ == "__main__":
    main()
