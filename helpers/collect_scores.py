# tools/collect_scores.py
import sys
from typing import Optional, List, Dict
from pathlib import Path

THIS_FILE = Path(__file__).resolve()
REPO_ROOT = THIS_FILE.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from predict_two_heads import (
    _move_prob, _qa_span, canonicalize_location, refine_compound_span,
    snap_to_vocab, is_negative_text, load_two_heads_cfg
)

def collect_records(cfg_dir_or_model: str, items: List[Dict], vocab=None):
    """items: [{'text': str, 'gold': Optional[str], 'curr': Optional[str]}]"""
    (move_dir, qa_dir, question, _, qa_thr,
     null_bias, max_len, doc_stride, max_span) = load_two_heads_cfg(cfg_dir_or_model)

    recs = []
    for it in items:
        text   = it["text"]
        gold   = (it.get("gold") or None)
        curr   = it.get("curr")

        p_move = _move_prob(move_dir, text)
        span_raw, p_best = _qa_span(
            qa_dir, question, text,
            max_length=max_len or 384,
            doc_stride=doc_stride or 64,
            null_bias=null_bias or 0.0,
            max_span_len=max_span or 32
        )

        span_norm = canonicalize_location(span_raw) if span_raw else ""
        span_ref  = refine_compound_span(span_norm, text)
        if vocab:
            snapped = snap_to_vocab(span_ref, vocab, max_dist=2)
            if snapped:
                span_ref = snapped

        recs.append({
            "text": text,
            "gold": gold,                 # None или канонич. локация
            "curr": curr,                 # текущая локация, если есть
            "p_move": float(p_move),
            "p_best": float(p_best),
            "span_ref": span_ref or "",
            "neg": bool(is_negative_text(text)),
            "qa_thr_cfg": float(qa_thr or 0.0),
        })
    return recs
