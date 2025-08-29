#!/usr/bin/env python3
import json, argparse, sys
from typing import List, Dict, Any

def load_any(path: str):
    # Пытаемся как JSON-массив
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data, "json"
    except Exception:
        pass
    # Падаем обратно на JSONL
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items, "jsonl"

def save_any(path: str, items: List[Dict[str, Any]], fmt: str):
    if fmt == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    else:
        with open(path, "w", encoding="utf-8") as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def renumber(items: List[Dict[str, Any]], start: int, as_str: bool = True):
    cur = start
    for obj in items:
        # если пример вложен (редко): {"id":..., "paragraphs":[{...}]}, оставляем как есть
        if not isinstance(obj, dict):
            continue
        obj["id"] = str(cur) if as_str else cur
        cur += 1
    return items

def main():
    ap = argparse.ArgumentParser(description="Renumber 'id' fields sequentially in a QA dataset.")
    ap.add_argument("--in", dest="inp", required=True, help="Input file (JSON array or JSONL).")
    ap.add_argument("--out", dest="out", required=True, help="Output file.")
    ap.add_argument("--start", type=int, default=1, help="Starting ID (default: 1).")
    ap.add_argument("--int-id", action="store_true", help="Store id as int (by default stored as string).")
    args = ap.parse_args()

    items, fmt = load_any(args.inp)

    if not isinstance(items, list):
        print("Input must be a list of examples (JSON array) or JSONL.", file=sys.stderr)
        sys.exit(1)

    renumber(items, start=args.start, as_str=(not args.int_id))
    save_any(args.out, items, fmt)
    print(f"Renumbered {len(items)} examples → {args.out} (start={args.start}, as_str={not args.int_id})")

if __name__ == "__main__":
    main()
