#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, sys
from pathlib import Path

def filter_has_movement(inp: Path, out: Path):
    with inp.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Плоский список
    if isinstance(data, list):
        filtered = [ex for ex in data if not ex.get("is_impossible", False)]
    # Классический SQuAD {"data":[...]}
    elif isinstance(data, dict) and "data" in data:
        filtered = {"data": []}
        for art in data["data"]:
            new_art = {"title": art.get("title", ""), "paragraphs": []}
            for para in art.get("paragraphs", []):
                new_para = {"context": para.get("context", ""), "qas": []}
                for qa in para.get("qas", []):
                    if not qa.get("is_impossible", False):
                        new_para["qas"].append(qa)
                if new_para["qas"]:
                    new_art["paragraphs"].append(new_para)
            if new_art["paragraphs"]:
                filtered["data"].append(new_art)
    else:
        raise ValueError("Unsupported dataset format")

    with out.open("w", encoding="utf-8") as w:
        json.dump(filtered, w, ensure_ascii=False, indent=2)

    print(f"Saved {out}")

def main():
    ap = argparse.ArgumentParser(description="Filter QA dataset to keep only movement examples (is_impossible == false).")
    ap.add_argument("--in", dest="inp", required=True, help="Path to input JSON")
    ap.add_argument("--out", dest="out", required=True, help="Path to output JSON")
    args = ap.parse_args()

    filter_has_movement(Path(args.inp), Path(args.out))

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
