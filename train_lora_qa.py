
import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForQuestionAnswering,
    TrainingArguments,
    Trainer,
    default_data_collator,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
import os


# Defaults (can be overridden by CLI flags)
MODEL_NAME = "deepset/roberta-base-squad2"  # or "twmkn9/distilbert-base-uncased-squad2" / "deepset/deberta-v3-base-squad2"
QUESTION = "What is the final arrival location?"


def build_datasets(train_path, val_path, tok, max_len=256, doc_stride=96):
    ds = {}
    for split, path in [("train", train_path), ("validation", val_path)]:
        raw = load_dataset("json", data_files={split: path})[split]

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
            keep_mask = [False] * len(offsets)
            first_no_answer_kept = {}

            for i, off in enumerate(offsets):
                sample_idx = sample_map[i]
                ans = answers_list[sample_idx]
                impossible = bool(impossible_list[sample_idx])

                seq_ids = tokenized.sequence_ids(i)
                try:
                    ctx_start = next(k for k, s in enumerate(seq_ids) if s == 1)
                    ctx_end = len(seq_ids) - 1 - next(k for k, s in enumerate(reversed(seq_ids)) if s == 1)
                except StopIteration:
                    start_positions.append(0);
                    end_positions.append(0)
                    continue

                if impossible or len(ans.get("text", [])) == 0:
                    if not first_no_answer_kept.get(sample_idx, False):
                        start_positions.append(0);
                        end_positions.append(0)
                        keep_mask[i] = True
                        first_no_answer_kept[sample_idx] = True
                    else:
                        start_positions.append(0);
                        end_positions.append(0)
                    continue

                # есть ответ → оставляем только окна, куда ответ попал
                start_char = ans["answer_start"][0]
                end_char = start_char + len(ans["text"][0])

                tok_start = ctx_start
                while tok_start <= ctx_end and not (off[tok_start][0] <= start_char < off[tok_start][1]):
                    tok_start += 1
                tok_end = ctx_start
                while tok_end <= ctx_end and not (off[tok_end][0] < end_char <= off[tok_end][1]):
                    tok_end += 1

                if tok_start <= ctx_end and tok_end <= ctx_end:
                    start_positions.append(tok_start);
                    end_positions.append(tok_end)
                    keep_mask[i] = True
                else:
                    start_positions.append(0);
                    end_positions.append(0)  # окно игнорируем ниже

            tokenized["start_positions"] = start_positions
            tokenized["end_positions"] = end_positions

            # Оставляем только отмеченные окна
            for k in list(tokenized.keys()):
                v = tokenized[k]
                if isinstance(v, list) and len(v) == len(keep_mask):
                    tokenized[k] = [x for x, keep in zip(v, keep_mask) if keep]

            return tokenized

        ds[split] = raw.map(prep_train, batched=True, remove_columns=raw.column_names)
    return ds["train"], ds["validation"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/train.json")
    ap.add_argument("--val", default="data/val.json")
    ap.add_argument("--out", default="out/qa-lora")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)  # LoRA likes a bit higher LR
    ap.add_argument("--model", default=MODEL_NAME)
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--merge", action="store_true", help="Merge LoRA into base weights and save a merged checkpoint")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model)

    # LoRA config. For BERT/Roberta family, target attention proj layers.
    # Robust TaskType fallback for different PEFT versions
    try:
        task_type = TaskType.QUESTION_ANS  # may not exist in some PEFT versions
    except AttributeError:
        try:
            task_type = TaskType.TOKEN_CLS       # widely available
        except AttributeError:
            task_type = TaskType.FEATURE_EXTRACTION  # generic fallback

    lora_cfg = LoraConfig(
        task_type=task_type,
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        target_modules=["query", "key", "value", "output.dense"],
    )
    model = get_peft_model(model, lora_cfg)

    # Ensure QA head truly trainable
    for n, p in model.named_parameters():
        if "qa_outputs" in n:
            p.requires_grad = True

    # Optional: print trainable params
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    train_ds, val_ds = build_datasets(args.train, args.val, tok)

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
        gradient_checkpointing=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=2,
        fp16=False,  # set True if on GPU with fp16 support
        bf16=False,  # set True if on BF16-capable GPU/CPU
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()

    # Save adapter-based checkpoint
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)

    # Optionally, merge LoRA into base weights for single-file deployment
    if args.merge:
        merged_dir = args.out.rstrip("/")
        merged_dir = f"{merged_dir}-merged"
        base = trainer.model.merge_and_unload()  # returns a plain HF model without PEFT wrappers
        base.save_pretrained(merged_dir)
        tok.save_pretrained(merged_dir)
        print(f"Merged model saved to: {merged_dir}")


if __name__ == "__main__":
    main()
