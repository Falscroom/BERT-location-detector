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
    refine_compound_span,
    snap_to_vocab,
    is_negative_text,
)

# ---------------------------------------------------------------------------
# Caches
_move_tok = _move = _qa_tok = _qa = None
_VOCAB_CACHE = None

# Default fallback confidences (overridable via env)
FALLBACK_ARRIVAL_CONF = float(os.environ.get("FALLBACK_ARRIVAL_CONF", 0.78))
FALLBACK_PRESENCE_CONF = float(os.environ.get("FALLBACK_PRESENCE_CONF", 0.76))

def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

def _load_vocab(path: Optional[str]):
    """Load optional strict vocab (one phrase per line), lowercase."""
    if not path or not os.path.isfile(path):
        return None
    vocab = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            v = line.strip().lower()
            if v:
                vocab.append(v)
    return tuple(sorted(set(vocab)))

# ---------------------------------------------------------------------------
# Model loaders
def warmup_move(model_dir_or_cfg: str):
    try:
        if os.path.isdir(model_dir_or_cfg):
            _load_move(model_dir_or_cfg)
        else:
            mv, _, _, _, _, _, _, _, _ = load_two_heads_cfg(model_dir_or_cfg)
            _load_move(mv)
    except Exception as e:
        print(f"[warmup_move] skipped: {e}")

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

# ---------------------------------------------------------------------------
# Inference heads

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
             null_bias: float = 0.0, max_span_len: int = 32) -> Tuple[Optional[str], float]:
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
        return None, 1.0  # model prefers NULL

    null += float(null_bias)
    m = max(null, best)
    p = math.exp(best - m) / (math.exp(best - m) + math.exp(null - m))

    offs = offsets_all[best_f].tolist()
    s_char, e_char = offs[s_idx][0], offs[e_idx][1]
    span = context[s_char:e_char].strip(" \t\n\r.,!?;:\"'")
    return (span or None), float(p)

# ---------------------------------------------------------------------------
# Public API (compatible with eval_sanity.py)

def predict(cfg_dir_or_model, prompt: str, curr_loc: Optional[str] = None) -> dict:
    prompt = ensure_str(prompt)

    (move_dir, qa_dir, question, move_thr, qa_thr,
     null_bias, max_len, doc_stride, max_span) = load_two_heads_cfg(cfg_dir_or_model)

    if not os.path.isdir(move_dir):
        raise FileNotFoundError(f"move_dir не существует: {move_dir}")
    if not os.path.isdir(qa_dir):
        raise FileNotFoundError(f"qa_dir не существует: {qa_dir}")

    # Read tunables from ENV (or keep defaults)
    tau      = float(os.environ.get("TAU", "0.60"))       # decision threshold on mixture
    qa_delta = float(os.environ.get("QA_DELTA", "0.10"))  # allowance below qa_thr
    w0 = float(os.environ.get("W0", "-1.2"))
    w1 = float(os.environ.get("W1", "1.2"))
    w2 = float(os.environ.get("W2", "1.4"))
    vocab_path = os.environ.get("VOCAB_FILE", "")         # optional custom vocab (one per line)

    global _VOCAB_CACHE
    if _VOCAB_CACHE is None:
        _VOCAB_CACHE = _load_vocab(vocab_path)

    # 1) movement head
    p_move = _move_prob(move_dir, prompt)

    # 2) QA head (always compute)
    span_raw, p_best = _qa_span(
        qa_dir,
        question,
        prompt,
        max_length   = max_len or 384,
        doc_stride   = doc_stride or 64,
        null_bias    = null_bias or 0.0,
        max_span_len = max_span or 32
    )

    # Normalize & refine span
    span_norm = canonicalize_location(span_raw) if span_raw else ""
    span_ref  = refine_compound_span(span_norm, prompt)
    if _VOCAB_CACHE:
        snapped = snap_to_vocab(span_ref, _VOCAB_CACHE, max_dist=2)
        if snapped:
            span_ref = snapped

    # Mixture of heads
    S = _sigmoid(w0 + w1 * float(p_move or 0.0) + w2 * float(p_best or 0.0))

    # Hard negative (fantasy/plan/imagine/…) guard BEFORE accepting movement
    if is_negative_text(prompt):
        return {
            "location": None,
            "confidence": float(1.0 - S),
            "p_move": float(p_move),
            "qa_conf": float(p_best),
            "span_raw": span_raw or "",
            "span_refined": span_ref,
            "decision": "none_neg"
        }

    # Require: mixture strong enough AND QA not too weak (slightly below qa_thr allowed)
    need_qa = (p_best or 0.0) >= max(0.0, (qa_thr or 0.0) - qa_delta)
    if span_ref and span_ref not in _BAD_SPANS and need_qa and S >= tau and passes_net_change(prompt, span_ref, curr_loc):
        return {
            "location": span_ref,
            "confidence": float(S),
            "p_move": float(p_move),
            "qa_conf": float(p_best),
            "span_raw": span_raw or "",
            "span_refined": span_ref,
            "decision": "model"
        }

    # Fallbacks (gated by mixture; they self-respect negatives inside rules)
    if S >= (tau - 0.05):
        fb = regex_arrival_fallback(prompt)
        if fb and passes_net_change(prompt, fb, curr_loc):
            conf = max(float(S), FALLBACK_ARRIVAL_CONF)
            return {
                "location": fb,
                "confidence": float(conf),
                "p_move": float(p_move),
                "qa_conf": float(p_best),
                "span_raw": span_raw or "",
                "span_refined": span_ref,
                "fallback": "arrival",
                "decision": "fallback"
            }

        fb = regex_presence_fallback(prompt)
        if fb and passes_net_change(prompt, fb, curr_loc):
            conf = max(float(S), FALLBACK_PRESENCE_CONF)
            return {
                "location": fb,
                "confidence": float(conf),
                "p_move": float(p_move),
                "qa_conf": float(p_best),
                "span_raw": span_raw or "",
                "span_refined": span_ref,
                "fallback": "presence",
                "decision": "fallback"
            }

    # Nothing confident → None
    return {
        "location": None,
        "confidence": float(1.0 - S),
        "p_move": float(p_move),
        "qa_conf": float(p_best),
        "span_raw": span_raw or "",
        "span_refined": span_ref,
        "decision": "none"
    }

if __name__ == "__main__":
    cfg = os.environ.get("TWO_HEADS_CFG", "cfg/two_heads/config.json")
    txt = os.environ.get("TEXT", "follow me to the balcony")

    print(predict(cfg, txt))
