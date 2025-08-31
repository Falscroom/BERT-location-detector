import os
from typing import Optional
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

from utils_common import ensure_str, DEVICE
from config_loader import load_two_heads_cfg

# Caches for loaded models/tokenizers
_move_tok = _move = None

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

# ---------------------------------------------------------------------------
# Inference heads

@torch.no_grad()
def _move_prob(model_dir: str, text: str) -> float:
    tok, clf = _load_move(model_dir)
    enc = tok(text, truncation=True, max_length=256, return_tensors="pt")
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    p = torch.softmax(clf(**enc).logits, dim=-1)[0, 1].item()
    return float(p)

# ---------------------------------------------------------------------------
# Public API (compatible with eval_sanity.py)

def predict(cfg_dir_or_model, prompt: str, curr_loc: Optional[str] = None) -> dict:
    prompt = ensure_str(prompt)

    _cfg = load_two_heads_cfg(cfg_dir_or_model)
    # Backward-compatible extraction: expect tuple like
    # (move_dir, qa_dir, question, move_thr, ...)
    move_dir = _cfg[0]
    move_thr = _cfg[3] if len(_cfg) > 3 else None

    if not os.path.isdir(move_dir):
        raise FileNotFoundError(f"move_dir не существует: {move_dir}")

    # Hard thresholds (ENV overrides allowed)
    mv_thr = float(os.environ.get("MOVE_THR", str(move_thr if move_thr is not None else 0.5)))

    # 1) movement head
    p_move = _move_prob(move_dir, prompt)

    # Binary-only decision (no QA). Location is not predicted.
    if p_move >= mv_thr:
        return {
            "location": "found",
            "confidence": float(p_move),
            "p_move": float(p_move),
            "qa_conf": 1.0,
            "span_raw": "",
            "span_refined": "",
            "decision": "move"
        }
    else:
        return {
            "location": None,
            "confidence": float(1.0 - p_move),
            "p_move": float(p_move),
            "qa_conf": 1.0,
            "span_raw": "",
            "span_refined": "",
            "decision": "none"
        }

if __name__ == "__main__":
    cfg = os.environ.get("TWO_HEADS_CFG", "cfg/two_heads/config.json")
    txt = os.environ.get("TEXT", "follow me to the balcony")

    print(predict(cfg, txt))
