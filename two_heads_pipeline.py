#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from typing import Tuple, Dict, Any, Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForQuestionAnswering, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, default_data_collator, EarlyStoppingCallback
)

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_num_threads(1)

# -----------------------------
# Common utils
# -----------------------------
def _ensure_str(x) -> str:
    if x is None:
        return ""
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8", errors="ignore")
        except Exception:
            return str(x)
    return str(x)


# -----------------------------
# QA: dataset builder (SQuAD2)
# -----------------------------
def build_qa_datasets(train_path, val_path, tok, max_len=384, doc_stride=128):
    ds = {}
    for split, path in [("train", train_path), ("validation", val_path)]:
        raw = load_dataset("json", data_files={split: path})[split]
        if split == "train":
            raw = raw.shuffle(seed=42)

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

                # with answer
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


def train_qa(args):
    tok = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)

    train_ds, val_ds = build_qa_datasets(args.train, args.val, tok)

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        eval_strategy="epoch",     # <-- фикс ключа
        save_strategy="epoch",
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=50,
        report_to="none",
        fp16=False,
        seed=42,
        data_seed=42,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
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


# -----------------------------
# Move-detector (binary CLS)
# -----------------------------
def build_move_datasets(train_path, val_path, tok, max_len=256):
    ds = {}
    for split, path in [("train", train_path), ("validation", val_path)]:
        raw = load_dataset("json", data_files={split: path})[split]
        if split == "train":
            raw = raw.shuffle(seed=42)

        cols = set(raw.column_names)
        use_jsonl = ("text" in cols) and ("label" in cols)   # наш бинарный JSONL
        use_squad = ("context" in cols) and ("is_impossible" in cols)  # SQuAD-стиль

        if not (use_jsonl or use_squad):
            raise ValueError(
                f"Unsupported schema for {split}: {cols}. "
                f"Expected either {{'text','label'}} or {{'context','is_impossible'}}."
            )

        def enc(ex):
            if use_jsonl:
                text = _ensure_str(ex.get("text", ""))
                label = int(ex.get("label", 0))
            else:  # SQuAD
                text = _ensure_str(ex.get("context", ""))
                label = int(not ex.get("is_impossible", True))  # 1=move, 0=no-move
            out = tok(text, truncation=True, max_length=max_len, padding="max_length")
            out["labels"] = label
            return out

        ds[split] = raw.map(enc, remove_columns=list(cols))
    return ds["train"], ds["validation"]


def train_move(args):
    tok = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
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


# -----------------------------
# Inference: gate (move) -> QA
# -----------------------------
_move_tok = None
_move_clf = None
_qa_tok = None
_qa = None


def _load_move(model_dir: str):
    global _move_tok, _move_clf
    if _move_tok is None or _move_clf is None:
        _move_tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
        _move_clf = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True).to(DEVICE).eval()
    return _move_tok, _move_clf


def _load_qa(model_dir: str):
    global _qa_tok, _qa
    if _qa_tok is None or _qa is None:
        _qa_tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
        _qa = AutoModelForQuestionAnswering.from_pretrained(model_dir, local_files_only=True).to(DEVICE).eval()
    return _qa_tok, _qa


@torch.no_grad()
def move_prob(move_model_dir: str, text: str) -> float:
    tok, clf = _load_move(move_model_dir)
    enc = tok(text, truncation=True, max_length=256, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    logits = clf(**enc).logits
    probs = torch.softmax(logits, dim=-1).squeeze(0).tolist()
    return float(probs[1])  # P(move)


@torch.no_grad()
def qa_predict_span(qa_model_dir: str, question: str, context: str,
                    max_length: int = 384, doc_stride: int = 128,
                    null_bias: float = 0.0, max_span_len: int = 16) -> Tuple[Optional[str], float]:
    """
    Небольшой самодостаточный инференс QA со скользящим окном, null calibration.
    Возвращает (span_or_None, p_best).
    """
    tok, model = _load_qa(qa_model_dir)

    enc = tok(
        _ensure_str(question), _ensure_str(context),
        return_tensors="pt",
        return_offsets_mapping=True,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        return_overflowing_tokens=True,
        padding="max_length",
    )
    offsets_all = enc.pop("offset_mapping")
    inputs = {k: v.to(DEVICE) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    outputs = model(**inputs)
    start_all = outputs.start_logits.cpu()
    end_all = outputs.end_logits.cpu()

    best_score = float("-inf")
    best_f = best_s = best_e = None
    null_score = float("-inf")

    for f in range(start_all.size(0)):
        seq_ids = enc.sequence_ids(f)
        ctx_tok_ids = [i for i, s in enumerate(seq_ids) if s == 1]
        if not ctx_tok_ids:
            continue

        start = start_all[f]
        end = end_all[f]

        # null score: [CLS] токен на позиции 0
        null_score = max(null_score, float(start[0] + end[0]))

        for i_idx, si in enumerate(ctx_tok_ids):
            for j_idx in range(i_idx, min(i_idx + max_span_len, len(ctx_tok_ids))):
                ei = ctx_tok_ids[j_idx]
                sc = float(start[si] + end[ei])
                if sc > best_score:
                    best_score, best_f, best_s, best_e = sc, f, si, ei

    if best_f is None:
        return None, 1.0

    # null calibration
    null_score += float(null_bias)

    # p(best) vs p(null)
    m = max(null_score, best_score)
    e_null = math.exp(null_score - m)
    e_best = math.exp(best_score - m)
    p_best = e_best / (e_best + e_null)

    offsets = offsets_all[best_f].tolist()
    s_char, e_char = offsets[best_s][0], offsets[best_e][1]
    span = context[s_char:e_char] if e_char <= len(context) else ""

    span = span.strip().strip(".,!?;:\"'")

    # Если модель неуверена — вернём None
    return (span if span else None), float(p_best)


def run_predict(args):
    text = _ensure_str(args.text)
    question = _ensure_str(args.question)

    # 1) movement gate
    p_move = move_prob(args.move_dir, text)
    if p_move < args.move_threshold:
        print({"location": None, "confidence": 1.0, "p_move": p_move})
        return

    # 2) QA
    span, p_best = qa_predict_span(
        qa_model_dir=args.qa_dir,
        question=question,
        context=text,
        max_length=args.max_len,
        doc_stride=args.doc_stride,
        null_bias=args.null_bias,
        max_span_len=args.max_span_len,
    )

    if span is None or p_best < args.qa_threshold:
        print({"location": None, "confidence": 1.0 - p_best, "p_move": p_move})
    else:
        print({"location": span, "confidence": p_best, "p_move": p_move})


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    # train_qa
    qa = sub.add_parser("train_qa")
    qa.add_argument("--model_name", default="deepset/roberta-base-squad2")
    qa.add_argument("--train", default="data/train.json")
    qa.add_argument("--val", default="data/val.json")
    qa.add_argument("--out", default="out/qa-distil")
    qa.add_argument("--epochs", type=int, default=4)
    qa.add_argument("--bs", type=int, default=16)
    qa.add_argument("--lr", type=float, default=3e-5)

    # train_move
    mv = sub.add_parser("train_move")
    mv.add_argument("--backbone", default="deepset/roberta-base-squad2")
    mv.add_argument("--train", default="data/train.json")
    mv.add_argument("--val", default="data/val.json")
    mv.add_argument("--out", default="out/move-det")
    mv.add_argument("--epochs", type=int, default=4)
    mv.add_argument("--bs", type=int, default=16)
    mv.add_argument("--lr", type=float, default=2e-5)

    # predict
    pr = sub.add_parser("predict")
    pr.add_argument("--move_dir", default="out/move-det")
    pr.add_argument("--qa_dir", default="out/qa-distil")
    pr.add_argument("--question", default="What is the destination location after movement?")
    pr.add_argument("--text", required=True)
    pr.add_argument("--move_threshold", type=float, default=0.6)
    pr.add_argument("--qa_threshold", type=float, default=0.80)
    pr.add_argument("--null_bias", type=float, default=0.0)
    pr.add_argument("--max_len", type=int, default=384)
    pr.add_argument("--doc_stride", type=int, default=128)
    pr.add_argument("--max_span_len", type=int, default=16)

    args = ap.parse_args()

    if args.cmd == "train_qa":
        train_qa(args)
    elif args.cmd == "train_move":
        train_move(args)
    elif args.cmd == "predict":
        run_predict(args)
    else:
        raise ValueError("Unknown command")

if __name__ == "__main__":
    main()
