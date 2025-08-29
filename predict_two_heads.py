#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, math
from typing import Optional, Tuple
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
)
from utils_common import ensure_str, DEVICE
from config_loader import load_two_heads_cfg
from rules import (
    BAD_SPANS as _BAD_SPANS,
    canonicalize_location,
    passes_net_change,
    regex_arrival_fallback,
    regex_presence_fallback,
)

# кэши моделей
_move_tok = _move = _qa_tok = _qa = None

# дефолтные уверенности для fallback'ов (можно переопределить через env)
FALLBACK_ARRIVAL_CONF = float(os.environ.get("FALLBACK_ARRIVAL_CONF", 0.78))
FALLBACK_PRESENCE_CONF = float(os.environ.get("FALLBACK_PRESENCE_CONF", 0.76))

# ---------- загрузка моделей ----------
def _load_move(dir_: str):
    global _move_tok, _move
    if _move_tok is None or _move is None:
        _move_tok = AutoTokenizer.from_pretrained(dir_, use_fast=True, local_files_only=True)
        _move = AutoModelForSequenceClassification.from_pretrained(dir_, local_files_only=True).to(DEVICE).eval()
    return _move_tok, _move

def _load_qa(dir_: str):
    global _qa_tok, _qa
    if _qa_tok is None or _qa is None:
        _qa_tok = AutoTokenizer.from_pretrained(dir_, use_fast=True, local_files_only=True)
        _qa = AutoModelForQuestionAnswering.from_pretrained(dir_, local_files_only=True).to(DEVICE).eval()
    return _qa_tok, _qa

# ---------- инференс ----------
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
    # сырой спан без бизнес-логики; приведём к канону в predict()
    span = context[s_char:e_char].strip(" \t\n\r.,!?;:\"'")
    return (span or None), float(p)

# ---------- публичная функция, совместимая с eval_sanity.py ----------
def predict(cfg_dir_or_model, prompt: str, curr_loc: Optional[str] = None) -> dict:
    prompt = ensure_str(prompt)

    (move_dir, qa_dir, question, move_thr, qa_thr,
     null_bias, max_len, doc_stride, max_span) = load_two_heads_cfg(cfg_dir_or_model)

    if not os.path.isdir(move_dir):
        raise FileNotFoundError(f"move_dir не существует: {move_dir}")
    if not os.path.isdir(qa_dir):
        raise FileNotFoundError(f"qa_dir не существует: {qa_dir}")

    # 1) бинарная голова — нет движения => сразу выход
    p_move = _move_prob(move_dir, prompt)
    if p_move < move_thr:
        return {"location": None, "confidence": 1.0, "p_move": p_move, "qa_conf": None}

    # 2) QA-голова
    span_raw, p_best = _qa_span(
        qa_dir, question, prompt,
        max_length=max_len, doc_stride=doc_stride,
        null_bias=null_bias, max_span_len=max_span
    )

    # Попытка принять модельный спан
    if span_raw:
        span = canonicalize_location(span_raw)
        if span and span not in _BAD_SPANS:
            if p_best >= qa_thr and passes_net_change(prompt, span, curr_loc):
                return {"location": span, "confidence": p_best, "p_move": p_move, "qa_conf": p_best}

    # 3) Fallback #1 — arrival
    fb = regex_arrival_fallback(prompt)
    if fb and passes_net_change(prompt, fb, curr_loc):
        conf = max(p_best if p_best is not None else 0.0, FALLBACK_ARRIVAL_CONF)
        return {
            "location": fb,
            "confidence": float(conf),
            "p_move": p_move,
            "qa_conf": p_best,
            "fallback": "arrival"
        }

    # 4) Fallback #2 — presence
    fb = regex_presence_fallback(prompt)
    if fb and passes_net_change(prompt, fb, curr_loc):
        conf = max(p_best if p_best is not None else 0.0, FALLBACK_PRESENCE_CONF)
        return {
            "location": fb,
            "confidence": float(conf),
            "p_move": p_move,
            "qa_conf": p_best,
            "fallback": "presence"
        }

    # ничего уверенного не нашли
    return {"location": None, "confidence": 1.0 - (p_best or 0.0), "p_move": p_move, "qa_conf": p_best}

if __name__ == "__main__":
    cfg = os.environ.get("TWO_HEADS_CFG", "cfg/two_heads/two_heads.json")
    txt = os.environ.get("TEXT", "follow me to the balcony")
    print(predict(cfg, txt))
