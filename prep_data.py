import json
import random

INPUT = "data/all.json"
TRAIN_OUT = "data/train.json"
VAL_OUT = "data/val.json"
VAL_RATIO = 0.2

def main():
    with open(INPUT, "r", encoding="utf-8") as f:
        data = json.load(f)

    # глобально перемешаем всё перед разделением
    random.shuffle(data)

    positives = [ex for ex in data if not ex["is_impossible"]]
    negatives = [ex for ex in data if ex["is_impossible"]]

    def split_group(group):
        random.shuffle(group)  # ещё раз для надёжности
        cut = int(len(group) * (1 - VAL_RATIO))
        return group[:cut], group[cut:]

    train_pos, val_pos = split_group(positives)
    train_neg, val_neg = split_group(negatives)

    train = train_pos + train_neg
    val = val_pos + val_neg

    random.shuffle(train)
    random.shuffle(val)

    with open(TRAIN_OUT, "w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=2)

    with open(VAL_OUT, "w", encoding="utf-8") as f:
        json.dump(val, f, ensure_ascii=False, indent=2)

    print(f"Train: {len(train)} | Val: {len(val)}")
    print("Val IDs:", [ex["id"] for ex in val])

if __name__ == "__main__":
    main()
