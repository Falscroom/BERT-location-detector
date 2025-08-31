#!/usr/bin/env python3
import json
import sys
from pathlib import Path

infile = Path("../data/all_binary.jsonl")
outfile = Path("../data/all_binary.jsonl")

with infile.open("r", encoding="utf-8") as fin, outfile.open("w", encoding="utf-8") as fout:
    for new_id, line in enumerate(fin, start=1):
        if not line.strip():
            continue
        obj = json.loads(line)
        obj["id"] = str(new_id)
        fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

print(f"Done. Written {outfile}")
