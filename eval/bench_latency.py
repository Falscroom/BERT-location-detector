#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time
import statistics
from math import ceil
from pathlib import Path

import torch

# --- robust import of predict() ----------------------------------------------
# Allow running from repo root or eval/ subdir
THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]          # .../Bert/
sys.path.insert(0, str(REPO_ROOT))        # prefer repo root on sys.path

try:
    from predict_two_heads import predict
except ModuleNotFoundError:
    # If your project uses a package layout like src/predict_two_heads.py
    try:
        sys.path.insert(0, str(REPO_ROOT / "src"))
        from predict_two_heads import predict
    except ModuleNotFoundError as e:
        raise SystemExit(
            "Cannot import predict_two_heads. "
            "Make sure predict_two_heads.py is at the repo root or in src/."
        ) from e

# --- config ------------------------------------------------------------------
# Make CFG_DIR absolute so it works no matter where you run from
CFG_DIR = str(REPO_ROOT / "cfg" / "two_heads")
QUESTION = "What is the destination location after movement?"

TEST_PROMPTS = [
    "walk with me to the garden",
    "let’s sneak into the castle",
    "sit with me in the tavern booth",
    "hi cutie",
    "I love your eyes",
    "don’t hide from me",
    "imagine us in the tavern",
    "show me how naughty you can be"
] * 20  # 120 runs

MOVE_TH = 0.60
QA_TH   = 0.80

# QA params (kept for clarity if your predict() reads globals)
QA_MAX_LEN  = 256
QA_STRIDE   = 96
QA_MAX_SPAN = 12

# --- helpers -----------------------------------------------------------------
def _has_mps() -> bool:
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()

def _sync_mps() -> None:
    if _has_mps():
        torch.mps.synchronize()

def _percentile(vals, p: float) -> float:
    """Robust percentile (0–100), works for small samples too."""
    if not vals:
        return 0.0
    xs = sorted(vals)
    if len(xs) == 1:
        return xs[0]
    # rank-based interpolation
    rank = (p / 100.0) * (len(xs) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(xs) - 1)
    frac = rank - lo
    return xs[lo] * (1 - frac) + xs[hi] * frac

def summarize(ms_list):
    mean = statistics.mean(ms_list) if ms_list else 0.0
    p95  = _percentile(ms_list, 95)
    p99  = _percentile(ms_list, 99)
    return mean, p95, p99

# --- main --------------------------------------------------------------------
def main():
    # Warm-up (not counted)
    for _ in range(15):
        _ = predict(CFG_DIR, "warm up the pipeline please")
    _sync_mps()

    total_ms, gate_ms, qa_ms = [], [], []
    sources_seen = {}
    printed_sample = False

    for text in TEST_PROMPTS:
        _sync_mps()
        t0 = time.perf_counter()

        # predict() returns a dict; we expect it may include dt_gate/dt_qa (in seconds) and source
        result = predict(CFG_DIR, text)
        dt_gate = float(result.get("dt_gate", 0.0))
        dt_qa   = float(result.get("dt_qa", 0.0))
        src     = result.get("source", "unknown")

        _sync_mps()
        t1 = time.perf_counter()

        total_ms.append((t1 - t0) * 1000.0)
        gate_ms.append(dt_gate * 1000.0)
        qa_ms.append(dt_qa * 1000.0)
        sources_seen[src] = sources_seen.get(src, 0) + 1

        if not printed_sample:
            printed_sample = True
            print("Sample result keys:", sorted(result.keys()))

    mean, p95, p99 = summarize(total_ms)
    mg, p95g, p99g = summarize(gate_ms)
    # QA may be skipped: compute stats on non-zero, or keep [0.0] to avoid crash
    qa_called = [t for t in qa_ms if t > 0.0] or [0.0]
    mq, p95q, p99q = summarize(qa_called)

    print(f"Runs: {len(total_ms)}")
    print(f"TOTAL  — mean {mean:.1f} ms | p95 {p95:.1f} | p99 {p99:.1f}")
    print(f"GATE   — mean {mg:.1f} ms | p95 {p95g:.1f} | p99 {p99g:.1f}")
    print(f"QA(*)  — mean {mq:.1f} ms | p95 {p95q:.1f} | p99 {p99q:.1f}  (*only when invoked)")
    print(f"Sources breakdown: {sources_seen}")

if __name__ == "__main__":
    main()
