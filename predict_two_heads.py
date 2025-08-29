#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, os, math, glob
from typing import Optional, Tuple
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
)
from utils_common import ensure_str, DEVICE

# -----------------------------
# простая гигиена спана
# -----------------------------
_BAD = {
    "here", "there", "now", "today", "tonight", "tomorrow",
    "inside", "outside", "back", "home", "again", "together"
}

# кэши моделей
_move_tok = _move = _qa_tok = _qa = None


# -----------------------------
# загрузка моделей
# -----------------------------
def _load_move(dir_: str):
    global _move_tok, _move
    if _move_tok is None or _move is None:
        _move_tok = AutoTokenizer.from_pretrained(dir_, use_fast=True, local_files_only=True)
        _move     = AutoModelForSequenceClassification.from_pretrained(dir_, local_files_only=True).to(DEVICE).eval()
    return _move_tok, _move


def _load_qa(dir_: str):
    global _qa_tok, _qa
    if _qa_tok is None or _qa is None:
        _qa_tok = AutoTokenizer.from_pretrained(dir_, use_fast=True, local_files_only=True)
        _qa     = AutoModelForQuestionAnswering.from_pretrained(dir_, local_files_only=True).to(DEVICE).eval()
    return _qa_tok, _qa


@torch.no_grad()
def _move_prob(model_dir: str, text: str) -> float:
    tok, clf = _load_move(model_dir)
    enc = tok(text, truncation=True, max_length=256, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    p = torch.softmax(clf(**enc).logits, dim=-1)[0, 1].item()
    return float(p)


@torch.no_grad()
def _qa_span(model_dir: str, question: str, context: str,
             max_length: int = 384, doc_stride: int = 128,
             null_bias: float = 0.0, max_span_len: int = 16) -> Tuple[Optional[str], float]:
    tok, model = _load_qa(model_dir)
    enc = tok(
        question, context,
        return_offsets_mapping=True,
        return_overflowing_tokens=True,
        truncation="only_second",
        max_length=max_length,
        stride=doc_stride,
        padding="max_length",
        return_tensors="pt",
    )
    offsets_all = enc.pop("offset_mapping")
    inputs = {k: v.to(DEVICE) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    out = model(**inputs)
    start_all, end_all = out.start_logits.cpu(), out.end_logits.cpu()

    best, best_f = float("-inf"), None
    null = float("-inf")
    for f in range(start_all.size(0)):
        seq_ids = enc.sequence_ids(f)
        ctx = [i for i, s in enumerate(seq_ids) if s == 1]
        if not ctx:
            continue
        st, en = start_all[f], end_all[f]
        null = max(null, float(st[0] + en[0]))
        for a, si in enumerate(ctx):
            for b in range(a, min(a + max_span_len, len(ctx))):
                ei = ctx[b]
                sc = float(st[si] + en[ei])
                if sc > best:
                    best, best_f, s_idx, e_idx = sc, f, si, ei

    if best_f is None:
        return None, 1.0

    null += float(null_bias)
    m = max(null, best)
    p = math.exp(best - m) / (math.exp(best - m) + math.exp(null - m))

    offs = offsets_all[best_f].tolist()
    s_char, e_char = offs[s_idx][0], offs[e_idx][1]
    span = context[s_char:e_char].strip(" \t\n\r.,!?;:\"'").lower()
    if span in _BAD:
        span = ""
    return (span or None), float(p)


# -----------------------------
# загрузка конфига / папки
# -----------------------------
REQUIRED_KEYS = {"move_dir", "qa_dir"}

def _abs(base: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

def _try_load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _pick_json_with_keys(dir_path: str, keys) -> Tuple[Optional[str], Optional[dict]]:
    # 1) приоритетно two_heads.json
    cand = os.path.join(dir_path, "two_heads.json")
    if os.path.exists(cand):
        cfg = _try_load_json(cand)
        if cfg and keys.issubset(cfg.keys()):
            return cand, cfg
    # 2) любой *.json с нужными ключами
    for p in glob.glob(os.path.join(dir_path, "*.json")):
        cfg = _try_load_json(p)
        if cfg and keys.issubset(cfg.keys()):
            return p, cfg
    return None, None

def _load_cfg(cfg_dir_or_model: str):
    """
    Возвращает:
      move_dir, qa_dir, question, move_thr, qa_thr, null_bias, max_len, doc_stride, max_span
    Поддерживает:
      - путь к JSON с ключами move_dir/qa_dir
      - папку, где есть two_heads.json или любой *.json с этими ключами
      - (fallback) трактует строку как путь к QA, а move_dir берёт из $MOVE_DIR или 'out/move-det'
    """
    # дефолты
    question   = "What is the destination location after movement?"
    move_thr   = 0.60
    qa_thr     = 0.80
    null_bias  = 0.0
    max_len    = 384
    doc_stride = 128
    max_span   = 16

    # 1) файл JSON
    if os.path.isfile(cfg_dir_or_model) and cfg_dir_or_model.endswith(".json"):
        base = os.path.dirname(os.path.abspath(cfg_dir_or_model))
        cfg = _try_load_json(cfg_dir_or_model)
        if not cfg or not REQUIRED_KEYS.issubset(cfg.keys()):
            raise ValueError(f"Config {cfg_dir_or_model} отсутствуют ключи {REQUIRED_KEYS}")
        move_dir = _abs(base, cfg["move_dir"])
        qa_dir   = _abs(base, cfg["qa_dir"])
        question   = cfg.get("question", question)
        move_thr   = float(cfg.get("move_thr", cfg.get("move_threshold", move_thr)))
        qa_thr     = float(cfg.get("qa_thr",   cfg.get("qa_threshold",   qa_thr)))
        null_bias  = float(cfg.get("null_bias",  null_bias))
        max_len    = int(cfg.get("max_len",     max_len))
        doc_stride = int(cfg.get("doc_stride",  doc_stride))
        max_span   = int(cfg.get("max_span_len", max_span))
        return move_dir, qa_dir, question, move_thr, qa_thr, null_bias, max_len, doc_stride, max_span

    # 2) директория
    if os.path.isdir(cfg_dir_or_model):
        picked, cfg = _pick_json_with_keys(cfg_dir_or_model, REQUIRED_KEYS)
        if cfg:
            base = os.path.dirname(os.path.abspath(picked))
            move_dir = _abs(base, cfg["move_dir"])
            qa_dir   = _abs(base, cfg["qa_dir"])
            question   = cfg.get("question", question)
            move_thr   = float(cfg.get("move_thr", cfg.get("move_threshold", move_thr)))
            qa_thr     = float(cfg.get("qa_thr",   cfg.get("qa_threshold",   qa_thr)))
            null_bias  = float(cfg.get("null_bias",  null_bias))
            max_len    = int(cfg.get("max_len",     max_len))
            doc_stride = int(cfg.get("doc_stride",  doc_stride))
            max_span   = int(cfg.get("max_span_len", max_span))
            return move_dir, qa_dir, question, move_thr, qa_thr, null_bias, max_len, doc_stride, max_span

        # нет валидного json — это почти наверняка не папка модели, поэтому ругаемся осмысленно
        listing = ", ".join(sorted(os.listdir(cfg_dir_or_model)))
        raise FileNotFoundError(
            f"В {cfg_dir_or_model} не найден JSON с ключами {REQUIRED_KEYS}. "
            f"Файлы внутри: [{listing}]. Создай cfg/two_heads/two_heads.json или передай путь к JSON."
        )

    # 3) fallback — трактуем как алиас/путь к QA, move_dir из ENV
    qa_dir   = cfg_dir_or_model
    move_dir = os.environ.get("MOVE_DIR", "out/move-det")
    return move_dir, qa_dir, question, move_thr, qa_thr, null_bias, max_len, doc_stride, max_span


# -----------------------------
# публичная функция для eval_sanity.py
# -----------------------------
def predict(cfg_dir_or_model, prompt: str, curr_loc: Optional[str] = None) -> dict:
    prompt = ensure_str(prompt)

    (move_dir, qa_dir, question, move_thr, qa_thr,
     null_bias, max_len, doc_stride, max_span) = _load_cfg(cfg_dir_or_model)

    # sanity путей
    if not os.path.isdir(move_dir):
        raise FileNotFoundError(f"move_dir не существует: {move_dir}")
    if not os.path.isdir(qa_dir):
        raise FileNotFoundError(f"qa_dir не существует: {qa_dir}")

    # 1) бинарная голова
    p_move = _move_prob(move_dir, prompt)
    if p_move < move_thr:
        return {"location": None, "confidence": 1.0, "p_move": p_move, "qa_conf": None}

    # 2) QA голова
    span, p_best = _qa_span(
        qa_dir, question, prompt,
        max_length=max_len, doc_stride=doc_stride,
        null_bias=null_bias, max_span_len=max_span
    )
    if span is None or p_best < qa_thr:
        return {"location": None, "confidence": 1.0 - p_best, "p_move": p_move, "qa_conf": p_best}

    return {"location": span, "confidence": p_best, "p_move": p_move, "qa_conf": p_best}


if __name__ == "__main__":
    cfg = os.environ.get("TWO_HEADS_CFG", "cfg/two_heads/two_heads.json")
    txt = os.environ.get("TEXT", "follow me to the balcony")
    print(predict(cfg, txt))
