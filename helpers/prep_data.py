#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, random, argparse, sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Iterable

def _ensure_str(x) -> str:
    if x is None: return ""
    if isinstance(x, bytes):
        try: return x.decode("utf-8", "ignore")
        except Exception: return str(x)
    return str(x)

# -------------------------
# JSONL (binary: text/label)
# -------------------------
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

# -------------------------
# SQuAD flat list helpers
# -------------------------
def is_flat_squad(obj: Any) -> bool:
    return isinstance(obj, list) and (len(obj) == 0 or isinstance(obj[0], dict))

def is_nested_squad(obj: Any) -> bool:
    return isinstance(obj, dict) and "data" in obj

def split_pos_neg_flat(items: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    pos = [ex for ex in items if not ex.get("is_impossible", False)]
    neg = [ex for ex in items if ex.get("is_impossible", False)]
    return pos, neg

def split_keep_ratio(pos: List[Any], neg: List[Any], val_ratio: float, seed: int) -> Tuple[List[Any], List[Any]]:
    rnd = random.Random(seed)
    rnd.shuffle(pos); rnd.shuffle(neg)
    def cut(group: List[Any]) -> Tuple[List[Any], List[Any]]:
        n_tr = int(len(group) * (1 - val_ratio))
        return group[:n_tr], group[n_tr:]
    tr_p, va_p = cut(pos)
    tr_n, va_n = cut(neg)
    train = tr_p + tr_n
    val   = va_p + va_n
    rnd.shuffle(train); rnd.shuffle(val)
    return train, val

# -------------------------
# Nested SQuAD helpers
# -------------------------
def flatten_nested_squad(obj: Dict[str, Any]) -> List[Tuple[Tuple[int,int,int], Dict[str, Any]]]:
    """
    Возвращает список ((art_idx, para_idx, qa_idx), qa_dict)
    """
    flat = []
    for ai, art in enumerate(obj.get("data", [])):
        for pi, para in enumerate(art.get("paragraphs", [])):
            for qi, qa in enumerate(para.get("qas", [])):
                flat.append(((ai, pi, qi), qa))
    return flat

def reconstruct_nested_squad(orig: Dict[str, Any],
                             keep_idx: List[Tuple[int,int,int]]) -> Dict[str, Any]:
    """
    Собираем новый nested SQuAD, оставляя только qas с индексами из keep_idx.
    Пустые параграфы/статьи убираем.
    """
    keep_map: Dict[Tuple[int,int], List[int]] = {}
    for ai, pi, qi in keep_idx:
        keep_map.setdefault((ai, pi), []).append(qi)

    out = {"data": []}
    for ai, art in enumerate(orig.get("data", [])):
        new_art = {"title": art.get("title", ""), "paragraphs": []}
        for pi, para in enumerate(art.get("paragraphs", [])):
            q_keep = set(keep_map.get((ai, pi), []))
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

# -------------------------
# Main splitters by format
# -------------------------
def split_jsonl_binary(inp: Path, train_out: Path, val_out: Path, val_ratio: float, seed: int):
    rows = load_jsonl(inp)
    # ожидаем поля text/label
    pos = [r for r in rows if int(r.get("label", 0)) == 1]
    neg = [r for r in rows if int(r.get("label", 0)) == 0]
    train, val = split_keep_ratio(pos, neg, val_ratio, seed)
    save_jsonl(train, train_out)
    save_jsonl(val,   val_out)
    print(f"[JSONL] Train: {len(train)} | Val: {len(val)}")

def split_flat_squad(inp: Path, train_out: Path, val_out: Path, val_ratio: float, seed: int):
    data = json.loads(inp.read_text(encoding="utf-8"))
    pos, neg = split_pos_neg_flat(data)
    train, val = split_keep_ratio(pos, neg, val_ratio, seed)
    train_out.write_text(json.dumps(train, ensure_ascii=False, indent=2), encoding="utf-8")
    val_out.write_text(json.dumps(val, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[FLAT] Train: {len(train)} | Val: {len(val)}")

def split_nested_squad(inp: Path, train_out: Path, val_out: Path, val_ratio: float, seed: int):
    data = json.loads(inp.read_text(encoding="utf-8"))
    flat = flatten_nested_squad(data)
    pos = [idx for idx, qa in flat if not qa.get("is_impossible", False)]
    neg = [idx for idx, qa in flat if qa.get("is_impossible", False)]
    # разделяем индексы, затем реконструируем
    tr_idx, va_idx = split_keep_ratio(pos, neg, val_ratio, seed)
    train_data = reconstruct_nested_squad(data, tr_idx)
    val_data   = reconstruct_nested_squad(data, va_idx)
    train_out.write_text(json.dumps(train_data, ensure_ascii=False, indent=2), encoding="utf-8")
    val_out.write_text(json.dumps(val_data, ensure_ascii=False, indent=2), encoding="utf-8")
    # Подсчёт примеров:
    def count_qas(obj):
        return sum(len(p.get("qas", [])) for a in obj["data"] for p in a.get("paragraphs", []))
    print(f"[NESTED] Train: {count_qas(train_data)} | Val: {count_qas(val_data)}")

# -------------------------
# CLI
# -------------------------
def main():
    ap = argparse.ArgumentParser(description="Stratified split for SQuAD2 (flat/nested) or JSONL (text/label).")
    ap.add_argument("--in", dest="inp", required=True, help="Input file: .json or .jsonl")
    ap.add_argument("--train_out", default="data/train.json", help="Output train file")
    ap.add_argument("--val_out",   default="data/val.json",   help="Output val file")
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    inp = Path(args.inp)
    train_out = Path(args.train_out)
    val_out   = Path(args.val_out)

    if inp.suffix.lower() == ".jsonl":
        split_jsonl_binary(inp, train_out, val_out, args.val_ratio, args.seed)
        return

    # JSON
    obj = json.loads(inp.read_text(encoding="utf-8"))
    if is_flat_squad(obj):
        split_flat_squad(inp, train_out, val_out, args.val_ratio, args.seed)
    elif is_nested_squad(obj):
        split_nested_squad(inp, train_out, val_out, args.val_ratio, args.seed)
    else:
        raise ValueError("Unsupported format: expected flat SQuAD list, nested SQuAD {'data':...}, or JSONL text/label.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
