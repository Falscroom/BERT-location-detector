#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, sys, re
from collections import defaultdict
from typing import Dict, List, Tuple

def norm_text(s: str) -> str:
    """
    Нормализация текста:
    - casefold (лучше чем lower для юникода)
    - trim
    - схлопывание любых пробелов в один
    """
    s = s.casefold().strip()
    s = re.sub(r"\s+", " ", s)
    return s

def parse_label(x):
    # допускаем 0/1 как int или "0"/"1" как str
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, int):
        if x in (0, 1):
            return x
        raise ValueError(f"label not in {{0,1}}: {x}")
    if isinstance(x, str):
        x = x.strip()
        if x in ("0", "1"):
            return int(x)
    raise ValueError(f"label must be 0/1 (int or '0'/'1'), got: {x!r}")

def load_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[WARN] {path}:{ln}: bad JSON: {e}", file=sys.stderr)
                continue
            yield path, ln, obj

def main():
    ap = argparse.ArgumentParser(
        description="Find contradictions in JSONL: same text labeled as both 0 and 1."
    )
    ap.add_argument("files", nargs="+", help="Input JSONL files")
    ap.add_argument("--no-normalize", action="store_true",
                    help="Do not normalize text; compare exact strings")
    ap.add_argument("--show-duplicates", action="store_true",
                    help="Also report duplicates with the same label (non-conflicting)")
    ap.add_argument("--limit", type=int, default=0, help="Limit number of conflicts shown (0 = no limit)")
    ap.add_argument("--tsv", action="store_true", help="Print a compact TSV summary after details")
    args = ap.parse_args()

    # text_key -> label -> list of (id, file, line, raw_text)
    bucket: Dict[str, Dict[int, List[Tuple[str, str, int, str]]]] = defaultdict(lambda: defaultdict(list))
    # сохраняем первый сырой вариант текста для красивого отображения
    first_raw: Dict[str, str] = {}

    total = 0
    for path in args.files:
        for path, ln, obj in load_jsonl(path):
            total += 1
            if "text" not in obj:
                print(f"[WARN] {path}:{ln}: missing 'text' field", file=sys.stderr)
                continue
            text_raw = obj["text"]
            if not isinstance(text_raw, str):
                print(f"[WARN] {path}:{ln}: 'text' must be str, got {type(text_raw)}", file=sys.stderr)
                continue
            key = text_raw if args.no_normalize else norm_text(text_raw)

            try:
                label = parse_label(obj.get("label", None))
            except Exception as e:
                print(f"[WARN] {path}:{ln}: {e}", file=sys.stderr)
                continue

            id_ = obj.get("id", "")
            if not isinstance(id_, str):
                id_ = str(id_)  # приводим к строке как ты просил

            bucket[key][label].append((id_, path, ln, text_raw))
            first_raw.setdefault(key, text_raw)

    conflicts = []
    dup_same = []
    for key, per_label in bucket.items():
        labels = sorted(per_label.keys())
        if len(labels) >= 2:  # конфликт: есть и 0, и 1
            conflicts.append((key, per_label))
        elif args.show_duplicates:
            # не конфликт, но есть повторы с тем же лейблом
            only_label = labels[0]
            if len(per_label[only_label]) > 1:
                dup_same.append((key, only_label, per_label[only_label]))

    # Детальный вывод конфликтов
    shown = 0
    if conflicts:
        print(f"=== CONTRADICTIONS (same text has labels 0 and 1) ===")
        for key, per_label in sorted(conflicts, key=lambda x: (-(len(x[1][0]) + len(x[1][1])), x[0])):  # крупные сначала
            raw_example = first_raw.get(key, key)
            print(f"\nTEXT: {raw_example!r}")
            for lb in (0, 1):
                rows = per_label.get(lb, [])
                print(f"  label={lb}  count={len(rows)}")
                for id_, path, ln, _ in rows:
                    print(f"    - id={id_} @ {path}:{ln}")
            shown += 1
            if args.limit and shown >= args.limit:
                break
    else:
        print("No contradictions found.")

    # Необязательный вывод дубликатов без конфликта (если попросили)
    if args.show_duplicates and dup_same:
        print(f"\n=== NON-CONFLICTING DUPLICATES (same text repeated with same label) ===")
        for key, lb, rows in sorted(dup_same, key=lambda x: -len(x[2])):
            raw_example = first_raw.get(key, key)
            print(f"\nTEXT: {raw_example!r}  label={lb}  count={len(rows)}")
            for id_, path, ln, _ in rows:
                print(f"    - id={id_} @ {path}:{ln}")

    # Компактная TSV-сводка (по конфликтам)
    if args.tsv and conflicts:
        print("\n#text\tlabel0_count\tlabel1_count\tids_label0\tids_label1")
        for key, per_label in conflicts:
            raw = first_raw.get(key, key).replace("\t", " ").replace("\n", " ")
            ids0 = ",".join(i for i, _, _, _ in per_label.get(0, []))
            ids1 = ",".join(i for i, _, _, _ in per_label.get(1, []))
            print(f"{raw}\t{len(per_label.get(0, []))}\t{len(per_label.get(1, []))}\t{ids0}\t{ids1}")

    # Код выхода: 1 если есть конфликты
    sys.exit(1 if conflicts else 0)

if __name__ == "__main__":
    main()
