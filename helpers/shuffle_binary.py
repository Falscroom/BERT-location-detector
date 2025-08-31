#!/usr/bin/env python3
import json, random, argparse
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", type=Path, default=Path("data/all_binary.jsonl"))
    ap.add_argument("--outfile", type=Path, default=Path("data/all_binary.jsonl"))
    ap.add_argument("--start-id", type=int, default=1, help="начальный id для первой записи")
    args = ap.parse_args()

    # читаем
    data = []
    with args.infile.open("r", encoding="utf-8") as f:
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

    # переписываем id заново (по порядку, начиная с start-id)
    cur = args.start_id
    for item in out:
        item["id"] = str(cur)
        cur += 1

    # сохраняем
    with args.outfile.open("w", encoding="utf-8") as f:
        for item in out:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"Done. Written {args.outfile} with {len(out)} records, starting id={args.start_id}")

if __name__ == "__main__":
    main()
