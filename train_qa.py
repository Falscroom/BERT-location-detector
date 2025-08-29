import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering,
    TrainingArguments, Trainer, default_data_collator, EarlyStoppingCallback
)

MODEL_NAME = "deepset/roberta-base-squad2"
QUESTION = "What is the destination location after movement? (e.g., kitchen, garden, office)"

def build_datasets(train_path, val_path, tok, max_len=384, doc_stride=128):
    ds = {}
    for split, path in [("train", train_path), ("validation", val_path)]:
        raw = load_dataset("json", data_files={split: path})[split]
        if split == "train":
            raw = raw.shuffle(seed=42)  # <-- shuffle train once up front

        def prep_train(examples):
            tokenized = tok(
                examples["question"],
                examples["context"],
                truncation="only_second",
                max_length=max_len,
                stride=doc_stride,
                return_overflowing_tokens=True,
                return_offsets_mapping=True,
                padding="max_length",
            )

            sample_map = tokenized.pop("overflow_to_sample_mapping")
            offsets = tokenized["offset_mapping"]
            answers_list = examples["answers"]
            impossible_list = examples.get("is_impossible", [False] * len(sample_map))

            start_positions, end_positions = [], []

            for i, off in enumerate(offsets):
                sample_idx = sample_map[i]
                ans = answers_list[sample_idx]

                seq_ids = tokenized.sequence_ids(i)
                try:
                    ctx_start = next(k for k, s in enumerate(seq_ids) if s == 1)
                    ctx_end = len(seq_ids) - 1 - next(k for k, s in enumerate(reversed(seq_ids)) if s == 1)
                except StopIteration:
                    start_positions.append(0)
                    end_positions.append(0)
                    tokenized["offset_mapping"][i] = [(0, 0)] * len(off)
                    continue

                # no-answer
                if impossible_list[sample_idx] or len(ans.get("text", [])) == 0:
                    start_positions.append(0)
                    end_positions.append(0)
                    tokenized["offset_mapping"][i] = [(0, 0)] * len(off)
                    continue

                # has answer
                start_char = ans["answer_start"][0]
                end_char = start_char + len(ans["text"][0])

                tok_start = ctx_start
                while tok_start <= ctx_end and not (off[tok_start][0] <= start_char < off[tok_start][1]):
                    tok_start += 1

                tok_end = ctx_start
                while tok_end <= ctx_end and not (off[tok_end][0] < end_char <= off[tok_end][1]):
                    tok_end += 1

                if tok_start > ctx_end or tok_end > ctx_end:
                    start_positions.append(0)
                    end_positions.append(0)
                    tokenized["offset_mapping"][i] = [(0, 0)] * len(off)
                else:
                    start_positions.append(tok_start)
                    end_positions.append(tok_end)

            tokenized["start_positions"] = start_positions
            tokenized["end_positions"] = end_positions
            return tokenized

        ds[split] = raw.map(prep_train, batched=True, remove_columns=raw.column_names)
    return ds["train"], ds["validation"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.json")
    ap.add_argument("--val", default="data/val.json")
    ap.add_argument("--out", default="out/qa-distil")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(MODEL_NAME)

    train_ds, val_ds = build_datasets(args.train, args.val, tok)

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        eval_strategy="epoch",   # <-- правильный ключ
        save_strategy="epoch",
        learning_rate=3e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=50,
        report_to="none",
        fp16=False,
        seed=42,        # <-- глобальный сид
        data_seed=42,   # <-- шаффл даталоадера по эпохам

        load_best_model_at_end=True,        # <--- добавь
        metric_for_best_model="eval_loss",  # <--- какую метрику мониторим
        greater_is_better=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
    )

    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

if __name__ == "__main__":
    main()
