#!/usr/bin/env python3
import time, statistics
import torch
from two_heads_pipeline import move_prob, qa_predict_span, _ensure_str

MOVE_DIR = "../out/move-det"
QA_DIR   = "../out/qa-distil"
QUESTION = "What is the destination location after movement?"

TEST_PROMPTS = [
    "walk with me to the garden",
    "let’s sneak into the castle",
    "sit with me in the tavern booth",
    "hi cutie",
    "I love your eyes",
    "don’t hide from me",
] * 20  # 120 прогонов

MOVE_TH = 0.6
QA_TH   = 0.80

# Параметры QA для снижения хвоста (минимально-рисковые):
QA_MAX_LEN   = 256
QA_STRIDE    = 96
QA_MAX_SPAN  = 12

def predict(text: str):
    t0 = time.perf_counter()
    p_move = move_prob(MOVE_DIR, text)
    t1 = time.perf_counter()

    if p_move < MOVE_TH:
        return {"location": None, "confidence": 1.0, "p_move": p_move}, (t1 - t0), 0.0

    span, p_best = qa_predict_span(
        qa_model_dir=QA_DIR,
        question=QUESTION,
        context=text,
        max_length=QA_MAX_LEN,
        doc_stride=QA_STRIDE,
        max_span_len=QA_MAX_SPAN,
    )
    t2 = time.perf_counter()

    if span is None or p_best < QA_TH:
        return {"location": None, "confidence": 1.0 - p_best, "p_move": p_move}, (t1 - t0), (t2 - t1)
    else:
        return {"location": span, "confidence": p_best, "p_move": p_move}, (t1 - t0), (t2 - t1)

def summarize(ms_list):
    mean = statistics.mean(ms_list)
    p95  = statistics.quantiles(ms_list, n=100)[94]
    p99  = statistics.quantiles(ms_list, n=100)[98]
    return mean, p95, p99

def main():
    # прогрев (без учёта в статистике)
    for _ in range(15):
        _ = predict("warm up the pipeline please")
    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    total, gate_t, qa_t = [], [], []

    for text in TEST_PROMPTS:
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t_start = time.perf_counter()
        _, dt_gate, dt_qa = predict(text)
        if torch.backends.mps.is_available():
            torch.mps.synchronize()
        t_end = time.perf_counter()

        total.append((t_end - t_start) * 1000)
        gate_t.append(dt_gate * 1000)
        qa_t.append(dt_qa * 1000)

    mean, p95, p99 = summarize(total)
    mg, p95g, p99g = summarize(gate_t)
    mq, p95q, p99q = summarize([t for t in qa_t if t > 0.0] or [0.0])  # QA не всегда вызываетcя

    print(f"Runs: {len(total)}")
    print(f"TOTAL  — mean {mean:.1f} ms | p95 {p95:.1f} | p99 {p99:.1f}")
    print(f"GATE   — mean {mg:.1f} ms | p95 {p95g:.1f} | p99 {p99g:.1f}")
    print(f"QA(*)  — mean {mq:.1f} ms | p95 {p95q:.1f} | p99 {p99q:.1f}  (*только когда запускался)")

if __name__ == "__main__":
    main()