#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
)

def _export(model, inputs, out_path, input_names, output_names, seq_len, try_dynamo=True):
    common_kwargs = dict(
        input_names=input_names,
        output_names=output_names,
        dynamic_axes={k: {0: "batch", 1: "seq"} for k in input_names},
        do_constant_folding=True,
        opset_version=17,
    )

    # Попытка нового экспортера
    if try_dynamo:
        try:
            torch.onnx.export(
                model, inputs, out_path,
                dynamo=True,  # новый экспортер
                **common_kwargs,
            )
            print(f"✅ Экспортировано (dynamo): {out_path}")
            return
        except (TypeError, ModuleNotFoundError, ImportError) as e:
            # TypeError — если старая версия torch не знает аргумент dynamo
            # ModuleNotFoundError/ImportError — если нет onnxscript и т.п.
            print(f"[warn] dynamo-экспорт недоступен ({e}); fallback на legacy exporter...")

    # Fallback: legacy-экспорт
    torch.onnx.export(
        model, inputs, out_path,
        **common_kwargs,
    )
    print(f"✅ Экспортировано (legacy): {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to HF model dir (with config.json)")
    ap.add_argument("--out",   required=True, help="Output .onnx file")
    ap.add_argument("--qa", action="store_true", help="Export QA (span) head instead of classification head")
    ap.add_argument("--seq", type=int, default=384, help="Dummy sequence length (QA обычно 384, CLS можно 256)")
    ap.add_argument("--no-dynamo", action="store_true", help="Force legacy exporter (disable dynamo)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    seq_len = int(args.seq)
    dummy_ids  = torch.randint(0, tok.vocab_size, (1, seq_len))
    dummy_mask = torch.ones_like(dummy_ids)
    inputs = (dummy_ids, dummy_mask)
    input_names = ["input_ids", "attention_mask"]

    if args.qa:
        model = AutoModelForQuestionAnswering.from_pretrained(args.model, local_files_only=True)
        out_names = ["start_logits", "end_logits"]  # важно для корректной формы
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True)
        out_names = ["logits"]

    model.eval()
    _export(
        model=model,
        inputs=inputs,
        out_path=args.out,
        input_names=input_names,
        output_names=out_names,
        seq_len=seq_len,
        try_dynamo=not args.no_dynamo,
    )

if __name__ == "__main__":
    main()
