#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, random, sys
from pathlib import Path
from typing import Iterable, Dict, Any, List, Tuple

def _ensure_str(x) -> str:
    if x is None: return ""
    if isinstance(x, bytes):
        try: return x.decode("utf-8", "ignore")
        except Exception: return str(x)
    return str(x)

def _squad_flat_examples(obj: Any) -> Iterable[Dict[str, Any]]:
    """
    Универсальный извлекатель примеров:
    - Поддерживает твой "плоский" формат: List[{"id","context","is_impossible",...}]
    - И классический SQuAD: {"data":[{"paragraphs":[{"context":...,"qas":[...]}]}]}
    Возвращает dict с ключами: id, context, is_impossible
    """
    # Плоский список
    if isinstance(obj, list):
        for ex in obj:
            if not isinstance(ex, dict): continue
            ctx = _ensure_str(ex.get("context", ""))
            iid = _ensure_str(ex.get("id", ""))
            # SQuAD2: no-answer — когда is_impossible True ИЛИ answers пуст
            ans = ex.get("answers", {})
            has_ans = bool(ans and ans.get("text") and len(ans["text"]) > 0)
            is_impossible = bool(ex.get("is_impossible", not has_ans))
            yield {"id": iid, "context": ctx, "is_impossible": is_impossible}
        return

    # Классический SQuAD
    if isinstance(obj, dict) and "data" in obj:
        for article in obj["data"]:
            for para in article.get("paragraphs", []):
                ctx = _ensure_str(para.get("context", ""))
                for qa in para.get("qas", []):
                    iid = _ensure_str(qa.get("id", ""))
                    is_imp = bool(qa.get("is_impossible", False))
                    answers = qa.get("answers", [])
                    has_ans = bool(answers and answers[0].get("text"))
                    is_impossible = is_imp or (not has_ans)
                    yield {"id": iid, "context": ctx, "is_impossible": is_impossible}
        return

    # Не распознали структуру
    raise ValueError("Unsupported dataset format. Expected a list of QA dicts or SQuAD {'data': ...}.")

def load_examples(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    return list(_squad_flat_examples(obj))

def balance_examples(exs: List[Dict[str, Any]], pos_ratio: float, seed: int) -> List[Dict[str, Any]]:
    """
    Балансировка downsample большинства до заданного pos_ratio (~доля label=1).
    pos_ratio=0.5 даст примерно 50/50.
    """
    pos = [e for e in exs if not e["is_impossible"]]  # label=1
    neg = [e for e in exs if e["is_impossible"]]      # label=0
    if not pos or not neg:
        return exs  # нечего балансировать

    random.Random(seed).shuffle(pos)
    random.Random(seed + 1).shuffle(neg)

    # хотим: len(pos) / (len(pos)+len(neg_bal)) ~= pos_ratio  =>  neg_bal ~= pos*(1-pos_ratio)/pos_ratio
    target_neg = int(len(pos) * (1 - pos_ratio) / max(pos_ratio, 1e-9))
    target_neg = min(target_neg, len(neg))
    balanced = pos + neg[:target_neg]
    random.Random(seed + 2).shuffle(balanced)
    return balanced

def write_jsonl(rows: Iterable[Tuple[str, str, int]], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as w:
        for iid, text, label in rows:
            rec = {"id": iid, "text": text, "label": label}
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    ap = argparse.ArgumentParser(description="Convert SQuAD-like QA dataset to binary 'movement' JSONL.")
    ap.add_argument("--in", dest="inp", required=True, help="Path to input JSON (SQuAD-like).")
    ap.add_argument("--out", dest="out", required=True, help="Path to output JSONL (id,text,label).")
    ap.add_argument("--balance", action="store_true", help="Downsample majority class to target pos_ratio.")
    ap.add_argument("--pos_ratio", type=float, default=0.5, help="Target fraction of positives after balancing (default 0.5).")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)

    examples = load_examples(inp)

    if args.balance:
        examples = balance_examples(examples, pos_ratio=args.pos_ratio, seed=args.seed)

    rows = []
    for ex in examples:
        iid = _ensure_str(ex.get("id", ""))
        ctx = _ensure_str(ex.get("context", ""))
        label = 0 if ex.get("is_impossible", True) else 1
        rows.append((iid, ctx, label))

    write_jsonl(rows, out)
    print(f"Done: {len(rows)} examples → {out}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
