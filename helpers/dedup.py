#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, sys, re
from collections import defaultdict, Counter, OrderedDict
from typing import Dict, List, Tuple

def norm_text(s: str) -> str:
    """Unicode-friendly normalization: casefold + trim + collapse spaces."""
    s = s.casefold().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def parse_label(x):
    if isinstance(x, bool): return int(x)
    if isinstance(x, int):
        if x in (0,1): return x
        raise ValueError(f"label not in {{0,1}}: {x}")
    if isinstance(x, str):
        x = x.strip()
        if x in ("0","1"): return int(x)
    raise ValueError(f"label must be 0/1 (int or '0'/'1'), got: {x!r}")

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line: continue
            try:
                obj = json.loads(line)
                yield ln, obj
            except json.JSONDecodeError as e:
                print(f"[WARN] {path}:{ln}: bad JSON: {e}", file=sys.stderr)

def dedup(items, normalize=True, policy="keep-first", drop_conflicts=False):
    """
    Returns (deduped_items, stats)
    policy:
      - keep-first (default): keep first seen label; subsequent dups discarded
      - keep-last:   keep last seen label; overwrites previous
      - prefer-0:    if conflict, force label=0
      - prefer-1:    if conflict, force label=1
      - majority:    if conflict, choose majority label among duplicates (ties -> keep-first)
    drop_conflicts:
      - if True: when a text has both labels, drop ALL occurrences (nothing kept)
    """
    key_fn = (norm_text if normalize else (lambda s: s))
    by_key: Dict[str, List[Tuple[int, dict]]] = defaultdict(list)  # key -> list of (seq_idx, obj)
    seq = 0
    for _, obj in items:
        if "text" not in obj:
            continue
        text = obj["text"]
        if not isinstance(text, str):
            continue
        k = key_fn(text)
        by_key[k].append((seq, obj))
        seq += 1

    out = []
    kept_keys = set()
    conflicts_cnt = 0
    dup_cnt = 0

    for k, lst in by_key.items():
        # gather labels
        labels = [parse_label(o.get("label", None)) for _, o in lst if "label" in o]
        unique_labels = set(labels)
        is_conflict = (len(unique_labels) > 1)

        if len(lst) > 1:
            dup_cnt += (len(lst) - 1)
        if is_conflict:
            conflicts_cnt += 1
            if drop_conflicts:
                continue  # drop all

        # choose representative according to policy
        if policy == "keep-first":
            chosen = min(lst, key=lambda x: x[0])[1]
            if is_conflict:
                # keep chosen text but original label of first occurrence
                pass
        elif policy == "keep-last":
            chosen = max(lst, key=lambda x: x[0])[1]
        elif policy == "prefer-0":
            chosen = min(lst, key=lambda x: x[0])[1]  # base object (first) for stable id/text
            chosen = dict(chosen)  # copy
            chosen["label"] = 0
        elif policy == "prefer-1":
            chosen = min(lst, key=lambda x: x[0])[1]
            chosen = dict(chosen)
            chosen["label"] = 1
        elif policy == "majority":
            cnt = Counter(labels)
            if cnt[0] > cnt[1]:
                lbl = 0
            elif cnt[1] > cnt[0]:
                lbl = 1
            else:
                # tie -> first
                chosen = min(lst, key=lambda x: x[0])[1]
                out.append(chosen)
                kept_keys.add(k)
                continue
            chosen = min(lst, key=lambda x: x[0])[1]
            chosen = dict(chosen)
            chosen["label"] = lbl
        else:
            raise ValueError(f"Unknown policy: {policy}")

        out.append(chosen)
        kept_keys.add(k)

    # Preserve overall input order of first occurrence per key
    out.sort(key=lambda o: norm_text(o["text"]) if normalize else o["text"])
    # The sort above changes to lexicographic; to preserve original stream order, rebuild order map:
    # Build first-seen index per key and sort by it.
    first_idx = {}
    idx = 0
    for _, obj in items:
        t = obj.get("text")
        if isinstance(t, str):
            k = norm_text(t) if normalize else t
            if k not in first_idx:
                first_idx[k] = idx
                idx += 1
    out.sort(key=lambda o: first_idx[norm_text(o["text"]) if normalize else o["text"]])

    stats = {
        "unique_texts": len(out),
        "duplicate_entries_removed": dup_cnt,
        "conflicting_texts": conflicts_cnt,
        "policy": policy,
        "dropped_conflicts": bool(drop_conflicts),
    }
    return out, stats

def main():
    ap = argparse.ArgumentParser(
        description="Deduplicate JSONL (id,text,label) by text; options for conflict resolution."
    )
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL")
    ap.add_argument("--out", dest="out", required=True, help="Output JSONL (deduped)")
    ap.add_argument("--no-normalize", action="store_true", help="Disable text normalization")
    ap.add_argument("--policy", choices=["keep-first","keep-last","prefer-0","prefer-1","majority"],
                    default="keep-first", help="Conflict resolution strategy")
    ap.add_argument("--drop-conflicts", action="store_true",
                    help="If a text has both labels, drop ALL occurrences")
    ap.add_argument("--reindex-ids", action="store_true",
                    help="Rewrite id as sequential strings starting at --start-id")
    ap.add_argument("--start-id", type=int, default=1, help="Start for --reindex-ids")
    args = ap.parse_args()

    # stream input once to keep first-seen order map stable
    items = list(read_jsonl(args.inp))
    deduped, stats = dedup(items,
                           normalize=(not args.no_normalize),
                           policy=args.policy,
                           drop_conflicts=args.drop_conflicts)

    if args.reindex_ids:
        cur = args.start_id
        for obj in deduped:
            obj["id"] = str(cur)
            cur += 1

    with open(args.out, "w", encoding="utf-8") as w:
        for obj in deduped:
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"[OK] wrote: {args.out}")
    print(f"[STATS] unique_texts={stats['unique_texts']}  removed={stats['duplicate_entries_removed']}"
          f"  conflicts={stats['conflicting_texts']}  policy={stats['policy']}"
          f"  dropped_conflicts={stats['dropped_conflicts']}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
