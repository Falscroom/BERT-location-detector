#!/usr/bin/env python3
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, default_data_collator, EarlyStoppingCallback
)
from utils_common import ensure_str

def build_move_datasets(train_path, val_path, tok, max_len=256):
    ds = {}
    for split, path in [("train", train_path), ("validation", val_path)]:
        raw = load_dataset("json", data_files={split: path})[split]
        if split == "train":
            raw = raw.shuffle(seed=42)

        cols = set(raw.column_names)
        use_jsonl = {"text", "label"}.issubset(cols)              # твой бинарный jsonl
        use_squad = {"context", "is_impossible"}.issubset(cols)   # SQuAD2-стиль

        if not (use_jsonl or use_squad):
            raise ValueError(f"Bad schema {cols} for {split}")

        def enc(ex):
            if use_jsonl:
                text  = ensure_str(ex.get("text", ""))
                label = int(ex.get("label", 0))
            else:
                text  = ensure_str(ex.get("context", ""))
                label = int(not ex.get("is_impossible", True))  # 1=move, 0=no-move
            out = tok(text, truncation=True, max_length=max_len, padding="max_length")
            out["labels"] = label
            return out

        ds[split] = raw.map(enc, remove_columns=list(cols))
    return ds["train"], ds["validation"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="deepset/roberta-base-squad2")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val",   required=True)
    ap.add_argument("--out",   required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs",     type=int, default=16)
    ap.add_argument("--lr",     type=float, default=2e-5)
    args = ap.parse_args()

    tok   = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.backbone, num_labels=2)

    train_ds, val_ds = build_move_datasets(args.train, args.val, tok)

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=50,
        report_to="none",
        seed=42, data_seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=val_ds,
        tokenizer=tok, data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )
    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

if __name__ == "__main__":
    main()
