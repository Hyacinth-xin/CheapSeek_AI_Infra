#!/usr/bin/env python3
"""CheapSeek R402 probe — Kernel Agent strategy hidden-test directions.

Directions covered (CLAUDE.md 十一 R402 + 十二.4 R402):
  - legality from divisibility/alignment/workspace (NOT shape_ok etc.)
  - hierarchy: vectorized > tiled > naive, picks fastest legal candidate
  - divisibility: M, N, K each independently tested; single dim fails => reject
  - alignment/workspace: boundary exact-match (>= not >)
  - edge: single-dim disruption, max shape 256^3, candidate ordering, dups, empty

Standalone: subprocess kernel_agent.py, JSON stdin/stdout.
Run: python3 CheapSeek_test/CheapSeek_r402.py
"""
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
AGENT = str(ROOT / "agents" / "kernel_agent.py")

def run_agent(inp, timeout=2):
    t0 = time.perf_counter()
    p = subprocess.run([sys.executable, AGENT], input=json.dumps(inp),
                       capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout.strip(), p.stderr.strip(), (time.perf_counter()-t0)*1000

def cand(cid, divisibility=1, alignment=1, workspace=0):
    return {"id": cid,
            "semantic_kernel_id": {"naive":10,"tiled":11,"vectorized":12}.get(cid,10),
            "image_id": 0x10000,
            "variant": {"naive":1,"tiled":2,"vectorized":3}.get(cid,1),
            "workspace": workspace, "alignment": alignment, "divisibility": divisibility}

def req(m,n,k,cs,align=64,ws=0,dtype="FP32",cid_=0):
    return {"case_id":cid_,"dtype":dtype,"m":m,"n":n,"k":k,
            "alignment":align,"workspace":ws,"candidates":cs}

def kid_of(inp, timeout=2):
    rc,out,err,ms = run_agent(inp,timeout)
    try: return rc, json.loads(out).get("kernel_id") if not rc else None, err, ms
    except json.JSONDecodeError: return rc, None, err, ms

results=[]

def chk(name,cond,detail=""):
    tag="PASS" if cond else "FAIL"
    line=f"[{tag}] {name}"
    if detail: line+=f"  ({detail})"
    print(line); results.append(bool(cond))

print("=== CheapSeek R402 probe ===\n")

# ─── TIER 1: basic hierarchy ───
ALL3 = [cand("naive"),cand("tiled",4),cand("vectorized",8,16)]

rc,kid,err,ms=kid_of(req(128,128,128,ALL3))
chk("all legal 128^3 -> vectorized",rc==0 and kid=="vectorized" and not err,f"kid={kid}")

rc,kid,err,ms=kid_of(req(4,4,4,ALL3))
chk("4^3 vec(8) fail -> tiled",rc==0 and kid=="tiled" and not err,f"kid={kid}")

rc,kid,err,ms=kid_of(req(3,3,3,ALL3))
chk("3^3 only naive",rc==0 and kid=="naive" and not err,f"kid={kid}")

rc,kid,err,ms=kid_of(req(8,8,9,ALL3))
chk("K=9 kills vec+tiled -> naive",rc==0 and kid=="naive" and not err,f"kid={kid}")

# ─── TIER 2: boundary exact-match ───
rc,kid,err,ms=kid_of(req(128,128,128,ALL3,align=16))
chk("align=16 matches vec(16) -> vectorized",rc==0 and kid=="vectorized" and not err,f"kid={kid}")

WS3 = [cand("naive",1,1,0),cand("tiled",4,1,4096),cand("vectorized",8,16,8192)]
rc,kid,err,ms=kid_of(req(128,128,128,WS3,ws=8192))
chk("ws=8192 matches vec(8192) -> vectorized",rc==0 and kid=="vectorized" and not err,f"kid={kid}")

# ─── TIER 3: off-by-one boundary ───
rc,kid,err,ms=kid_of(req(128,128,128,ALL3,align=15))
chk("align=15 < vec(16) by 1 -> tiled",rc==0 and kid=="tiled" and not err,f"kid={kid}")

rc,kid,err,ms=kid_of(req(4,4,4,WS3,ws=4095))
chk("ws=4095 < tiled(4096) by 1 -> naive",rc==0 and kid=="naive" and not err,f"kid={kid}")

# ─── TIER 4: shape extremes ───
rc,kid,err,ms=kid_of(req(256,256,256,ALL3))
chk("256^3 max -> vectorized",rc==0 and kid=="vectorized" and not err,f"kid={kid}")

rc,kid,err,ms=kid_of(req(1,1,1,ALL3))
chk("1x1x1 min -> naive",rc==0 and kid=="naive" and not err,f"kid={kid}")

rc,kid,err,ms=kid_of(req(4,4,1,ALL3))
chk("K=1 degenerate -> naive",rc==0 and kid=="naive" and not err,f"kid={kid}")

# ─── TIER 5: single-dim disruption ───
rc,kid,err,ms=kid_of(req(9,8,8,ALL3))
chk("M=9 kills vec+tiled -> naive",rc==0 and kid=="naive" and not err,f"kid={kid}")

rc,kid,err,ms=kid_of(req(8,9,8,ALL3))
chk("N=9 kills vec+tiled -> naive",rc==0 and kid=="naive" and not err,f"kid={kid}")

rc,kid,err,ms=kid_of(req(255,255,255,ALL3))
chk("255^3 no div 4/8 -> naive",rc==0 and kid=="naive" and not err,f"kid={kid}")

# ─── TIER 6: candidate list robustness ───
rc,kid,err,ms=kid_of(req(128,128,128,[cand("tiled",4),cand("vectorized",8,16),cand("naive")]))
chk("reverse order -> still vectorized",rc==0 and kid=="vectorized" and not err,f"kid={kid}")

rc,kid,err,ms=kid_of(req(128,128,128,[cand("vectorized",8,16)]))
chk("single vec (legal) -> vectorized",rc==0 and kid=="vectorized" and not err,f"kid={kid}")

rc,kid,err,ms=kid_of(req(8,8,8,[cand("tiled",4)]))
chk("single tiled (legal) -> tiled",rc==0 and kid=="tiled" and not err,f"kid={kid}")

rc,kid,err,ms=kid_of(req(128,128,128,[cand("naive"),cand("naive"),cand("vectorized",8,16)]))
chk("dup naive -> still vectorized",rc==0 and kid=="vectorized" and not err,f"kid={kid}")

rc,kid,err,ms=kid_of(req(3,3,3,[cand("tiled",4),cand("vectorized",8,16)]))
chk("no legal no naive -> first(tiled)",rc==0 and kid=="tiled" and not err,f"kid={kid}")

rc,kid,err,ms=kid_of(req(128,128,128,[]))
chk("empty list -> exit 2",rc==2,f"rc={rc}")

# ─── TIER 7: output hygiene ───
rc,out,err,ms=run_agent(req(128,128,128,[cand("naive"),cand("vectorized",8,16)]))
try:
    resp=json.loads(out)
    ok=rc==0 and not err and set(resp.keys())=={"kernel_id"} and isinstance(resp["kernel_id"],str)
except: ok=False
chk("output: pure JSON, only kernel_id",ok,f"out={out}")

rc,kid,err,ms=kid_of(req(128,128,128,[cand("naive"),cand("vectorized",8,16)]))
chk("no stderr",rc==0 and not err,f"stderr={err[:50] if err else ''}")

# ─── TIER 8: performance ───
total=sum(kid_of(req(128,128,128,ALL3))[3] for _ in range(10)); avg=total/10
chk(f"perf: avg {avg:.0f}ms/10 (lim 1000ms)",avg<200,f"avg={avg:.0f}ms")

p=sum(results)
print(f"\n=== {p}/{len(results)} checks passed ===")
