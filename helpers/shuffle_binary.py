#!/usr/bin/env python3
import json, random
from pathlib import Path

infile = Path("data/all_binary.jsonl")
outfile = Path("data/all_binary.jsonl")

# читаем
data = []
with infile.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        data.append(json.loads(line))

# разделяем по меткам
zeros = [d for d in data if d["label"] == 0]
ones  = [d for d in data if d["label"] == 1]

# перемешиваем внутри классов
random.shuffle(zeros)
random.shuffle(ones)

# равномерная укладка (чередование)
out = []
while zeros or ones:
    if len(out) % 2 == 0:  # на чётные позиции кладём нули
        if zeros: out.append(zeros.pop())
        elif ones: out.append(ones.pop())
    else:  # на нечётные — единицы
        if ones: out.append(ones.pop())
        elif zeros: out.append(zeros.pop())

# переписываем id заново (по порядку)
for i, item in enumerate(out, start=1):
    item["id"] = str(i)

# сохраняем
with outfile.open("w", encoding="utf-8") as f:
    for item in out:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print(f"Done. Written {outfile} with {len(out)} records.")
