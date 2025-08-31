#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, argparse, sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Iterable

# ---------- IO ----------
def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            rows.append(json.loads(line))
    return rows

def save_jsonl(rows: Iterable[Dict[str, Any]], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as w:
        for r in rows:
            w.write(json.dumps(r, ensure_ascii=False) + "\n")

# ---------- SQuAD helpers ----------
def is_flat_squad(obj: Any) -> bool:
    return isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict))

def is_nested_squad(obj: Any) -> bool:
    return isinstance(obj, dict) and "data" in obj

def flatten_nested_squad(obj: Dict[str, Any]) -> List[Tuple[Tuple[int,int,int], Dict[str, Any]]]:
    flat = []
    for ai, art in enumerate(obj.get("data", [])):
        for pi, para in enumerate(art.get("paragraphs", [])):
            for qi, qa in enumerate(para.get("qas", [])):
                flat.append(((ai, pi, qi), qa))
    return flat

def reconstruct_nested_squad(orig: Dict[str, Any],
                             keep_idx: List[Tuple[int,int,int]]) -> Dict[str, Any]:
    keep_map: Dict[Tuple[int,int], set] = {}
    for ai, pi, qi in keep_idx:
        keep_map.setdefault((ai, pi), set()).add(qi)

    out = {"data": []}
    for ai, art in enumerate(orig.get("data", [])):
        new_art = {"title": art.get("title", ""), "paragraphs": []}
        for pi, para in enumerate(art.get("paragraphs", [])):
            q_keep = keep_map.get((ai, pi))
            if not q_keep:
                continue
            new_qas = [qa for qi, qa in enumerate(para.get("qas", [])) if qi in q_keep]
            if not new_qas:
                continue
            new_art["paragraphs"].append({
                "context": para.get("context", ""),
                "qas": new_qas
            })
        if new_art["paragraphs"]:
            out["data"].append(new_art)
    return out

# ---------- Simple sequential split (no class quotas) ----------
def split_seq(items: List[Any], val_ratio: float):
    n = len(items)
    n_tr = int(round(n * (1.0 - val_ratio)))
    return items[:n_tr], items[n_tr:]

# ---------- Runners ----------
def split_jsonl_binary(inp: Path, train_out: Path, val_out: Path, val_ratio: float):
    rows = load_jsonl(inp)
    train, val = split_seq(rows, val_ratio)
    save_jsonl(train, train_out)
    save_jsonl(val,   val_out)
    print(f"[JSONL] total={len(rows)} | Train: {len(train)} | Val: {len(val)}")

def split_flat_squad(inp: Path, train_out: Path, val_out: Path, val_ratio: float):
    data = json.loads(inp.read_text(encoding="utf-8"))
    train, val = split_seq(data, val_ratio)
    train_out.write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    val_out.write_text(json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[FLAT] total={len(data)} | Train: {len(train)} | Val: {len(val)}")

def split_nested_squad(inp: Path, train_out: Path, val_out: Path, val_ratio: float):
    data = json.loads(inp.read_text(encoding="utf-8"))
    flat = flatten_nested_squad(data)                     # в исходном порядке
    n = len(flat)
    n_tr = int(round(n * (1.0 - val_ratio)))
    tr_idx = [idx for idx, _ in flat[:n_tr]]
    va_idx = [idx for idx, _ in flat[n_tr:]]
    train_data = reconstruct_nested_squad(data, tr_idx)
    val_data   = reconstruct_nested_squad(data, va_idx)
    train_out.write_text(json.dumps(train_data, ensure_ascii=False, indent=2), encoding="utf-8")
    val_out.write_text(json.dumps(val_data, ensure_ascii=False, indent=2), encoding="utf-8")
    def count_qas(obj):
        return sum(len(p.get("qas", [])) for a in obj["data"] for p in a.get("paragraphs", []))
    print(f"[NESTED] total_qas={n} | Train: {count_qas(train_data)} | Val: {count_qas(val_data)}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser(description="Sequential split: take first (1-val_ratio) to train, rest to val. No shuffles.")
    ap.add_argument("--in", dest="inp", required=True, help="Input file: .json or .jsonl")
    ap.add_argument("--train_out", default="data/train.json", help="Output train file")
    ap.add_argument("--val_out",   default="data/val.json",   help="Output val file")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    args = ap.parse_args()

    inp = Path(args.inp)
    train_out = Path(args.train_out)
    val_out   = Path(args.val_out)

    if inp.suffix.lower() == ".jsonl":
        split_jsonl_binary(inp, train_out, val_out, args.val_ratio)
        return

    obj_text = inp.read_text(encoding="utf-8")
    obj = json.loads(obj_text)
    if is_flat_squad(obj):
        split_flat_squad(inp, train_out, val_out, args.val_ratio)
    elif is_nested_squad(obj):
        split_nested_squad(inp, train_out, val_out, args.val_ratio)
    else:
        raise ValueError("Unsupported format: expected list JSON, nested SQuAD {'data':...}, or JSONL.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
