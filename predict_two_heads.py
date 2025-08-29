# predict_two_heads.py
import json, os
from typing import Optional, Dict, Any

# импортируем готовые функции инференса из твоего пайплайна
import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from two_heads_pipeline import move_prob, qa_predict_span

def _load_cfg(model_path: str) -> Dict[str, Any]:
    """
    model_path может быть директорией (тогда читаем config.json внутри)
    или прямым путем к json-файлу.
    """
    path = model_path
    if os.path.isdir(path):
        path = os.path.join(path, "config.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def predict(model_path: str, prompt: str, curr_loc: Optional[str] = None) -> Dict[str, Any]:
    """
    Интерфейс, который ожидает твой евал-скрипт.
    Возвращает {"location": str|None, "confidence": float, "p_move": float}
    """
    cfg = _load_cfg(model_path)

    move_dir       = cfg["move_dir"]
    qa_dir         = cfg["qa_dir"]
    question       = cfg.get("question", "What is the destination location after movement?")
    move_threshold = float(cfg.get("move_threshold", 0.6))
    qa_threshold   = float(cfg.get("qa_threshold", 0.8))
    null_bias      = float(cfg.get("null_bias", 0.0))
    max_len        = int(cfg.get("max_len", 384))
    doc_stride     = int(cfg.get("doc_stride", 128))
    max_span_len   = int(cfg.get("max_span_len", 16))

    text = prompt if prompt is not None else ""

    # 1) бинарная голова: вероятность движения
    p_move = move_prob(move_dir, text)

    if p_move < move_threshold:
        return {"location": None, "confidence": 1.0, "p_move": float(p_move)}

    # 2) QA-голова: извлекаем финальную локацию
    span, p_best = qa_predict_span(
        qa_model_dir=qa_dir,
        question=question,
        context=text,
        max_length=max_len,
        doc_stride=doc_stride,
        null_bias=null_bias,
        max_span_len=max_span_len,
    )

    if span is None or p_best < qa_threshold:
        return {"location": None, "confidence": float(1.0 - p_best), "p_move": float(p_move)}
    else:
        # можно тут же чуть подчистить спан, если хочешь
        span = span.strip().strip(".,!?;:\"'")
        return {"location": span, "confidence": float(p_best), "p_move": float(p_move)}
