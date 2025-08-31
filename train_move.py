#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from dataclasses import dataclass
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
    DataCollatorWithPadding, EarlyStoppingCallback
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

from utils_common import ensure_str


# ---------------------- Data ----------------------
def build_move_datasets(train_path, val_path, tok, max_len: int = 192):
    ds = {}
    for split, path in [("train", train_path), ("validation", val_path)]:
        raw = load_dataset("json", data_files={split: path})[split]
        if split == "train":
            raw = raw.shuffle(seed=42)

        cols = set(raw.column_names)
        use_jsonl = {"text", "label"}.issubset(cols)
        use_squad = {"context", "is_impossible"}.issubset(cols)
        if not (use_jsonl or use_squad):
            raise ValueError(f"Bad schema {cols} for {split}")

        def enc(ex):
            if use_jsonl:
                text = ensure_str(ex.get("text", ""))
                label = int(ex.get("label", 0))
            else:
                text = ensure_str(ex.get("context", ""))
                label = int(not ex.get("is_impossible", True))  # 1=move, 0=no-move
            out = tok(text, truncation=True, max_length=max_len)
            out["labels"] = label
            return out

        ds[split] = raw.map(enc, remove_columns=list(cols))
    return ds["train"], ds["validation"]


# ---------------------- Metrics ----------------------
def compute_metrics(eval_pred, beta: float = 0.5, thresh: float = 0.5):
    logits, labels = eval_pred
    probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()[:, 1]
    preds = (probs >= thresh).astype(np.int32)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    acc = accuracy_score(labels, preds)
    fbeta = (1 + beta * beta) * prec * rec / (beta * beta * prec + rec) if (prec + rec) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "fbeta": fbeta, "accuracy": acc}


# ---------------------- Loss config ----------------------
@dataclass
class LossCfg:
    use_focal: bool = True
    alpha_pos: float = 0.6   # вес класса 1 (move) — меньше => ещё меньше FP
    alpha_neg: float = 1.4   # вес класса 0 (no-move)
    gamma: float = 2.0       # фокусирование на «трудных» примерах


# ---------------------- Custom Trainer ----------------------
class WeightedTrainer(Trainer):
    def __init__(self, *args, loss_cfg: LossCfg, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_cfg = loss_cfg
        # просто храним как тензор; на нужное устройство перенесём в compute_loss
        self._cls_w = torch.tensor([loss_cfg.alpha_neg, loss_cfg.alpha_pos], dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # transformers >= 4.42 передаёт num_items_in_batch и пр. — принимаем через **kwargs
        labels = inputs.get("labels")
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        logits = outputs.logits
        cls_w = self._cls_w.to(logits.device)

        if self.loss_cfg.use_focal:
            ce = torch.nn.functional.cross_entropy(logits, labels, reduction="none", weight=cls_w)
            p_t = torch.softmax(logits, dim=-1)[torch.arange(logits.size(0), device=logits.device), labels]
            loss = ((1 - p_t) ** self.loss_cfg.gamma) * ce
            loss = loss.mean()
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels, weight=cls_w)

        return (loss, outputs) if return_outputs else loss


# ---------------------- Freeze & LLRD ----------------------
def freeze_bottom_n_layers(model, n=6):
    enc = None
    if hasattr(model, "roberta"):
        enc = model.roberta.encoder.layer
    elif hasattr(model, "bert"):
        enc = model.bert.encoder.layer
    if enc is None:
        return
    for i in range(min(n, len(enc))):
        for p in enc[i].parameters():
            p.requires_grad = False


def build_param_groups_llrd(model, base_lr, head_lr_mult=5.0, lr_decay=0.9):
    groups = []
    enc = emb = classifier = None
    if hasattr(model, "roberta"):
        enc = model.roberta.encoder.layer
        emb = model.roberta.embeddings
        classifier = model.classifier
    elif hasattr(model, "bert"):
        enc = model.bert.encoder.layer
        emb = model.bert.embeddings
        classifier = model.classifier

    if classifier is not None:
        groups.append({"params": [p for p in classifier.parameters() if p.requires_grad],
                       "lr": base_lr * head_lr_mult})

    if enc is not None:
        lr = base_lr
        for layer in reversed(list(enc)):
            params = [p for p in layer.parameters() if p.requires_grad]
            if params:
                groups.append({"params": params, "lr": lr})
            lr *= lr_decay

    if emb is not None:
        params = [p for p in emb.parameters() if p.requires_grad]
        if params:
            groups.append({"params": params, "lr": lr})

    return groups if groups else model.parameters()


# ---------------------- Main ----------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", default="answerdotai/ModernBERT-base")
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--freeze", type=int, default=6, help="freeze bottom N encoder layers")
    ap.add_argument("--beta", type=float, default=0.5, help="F_beta for model selection (precision-heavy <1)")
    ap.add_argument("--no_focal", action="store_true")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.backbone, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.backbone, num_labels=2)

    train_ds, val_ds = build_move_datasets(args.train, args.val, tok)
    collator = DataCollatorWithPadding(tokenizer=tok, pad_to_multiple_of=8)

    # Заморозка нижних слоёв
    freeze_bottom_n_layers(model, n=args.freeze)

    training_args = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs,
        eval_strategy="epoch",        # <— правильный ключ
        save_strategy="epoch",
        lr_scheduler_type="cosine",
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.06,
        logging_steps=50,
        report_to="none",
        seed=42,
        data_seed=42,
        load_best_model_at_end=True,
        metric_for_best_model="eval_f1",    # выбираем по F1
        greater_is_better=True,
        save_total_limit=2,
        fp16=False,                         # CPU/MPS — лучше выключить
        dataloader_pin_memory=False,        # MPS не использует pinned memory
    )

    loss_cfg = LossCfg(use_focal=(not args.no_focal))

    # LLRD: разные LR по слоям
    opt_groups = build_param_groups_llrd(model, base_lr=args.lr, head_lr_mult=5.0, lr_decay=0.9)
    optimizer = torch.optim.AdamW(opt_groups, weight_decay=0.01)

    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=1)],
        loss_cfg=loss_cfg,
        compute_metrics=lambda p: compute_metrics(p, beta=args.beta, thresh=0.5),
        optimizers=(optimizer, None),
    )

    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)


if __name__ == "__main__":
    main()
