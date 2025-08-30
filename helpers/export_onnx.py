#!/usr/bin/env python3
import argparse, os, torch
from transformers import AutoModelForSequenceClassification, AutoModelForQuestionAnswering, AutoTokenizer

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to model dir (with config.json)")
    ap.add_argument("--out", required=True, help="Output .onnx file")
    ap.add_argument("--qa", action="store_true", help="If set, export QA head (default: classification head)")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, local_files_only=True)

    if args.qa:
        model = AutoModelForQuestionAnswering.from_pretrained(args.model, local_files_only=True)
        dummy_ids = torch.randint(0, tok.vocab_size, (1, 32))
        dummy_mask = torch.ones_like(dummy_ids)
        inputs = (dummy_ids, dummy_mask)
        input_names = ["input_ids", "attention_mask"]
    else:
        model = AutoModelForSequenceClassification.from_pretrained(args.model, local_files_only=True)
        dummy_ids = torch.randint(0, tok.vocab_size, (1, 32))
        dummy_mask = torch.ones_like(dummy_ids)
        inputs = (dummy_ids, dummy_mask)
        input_names = ["input_ids", "attention_mask"]

    model.eval()

    torch.onnx.export(
        model,
        inputs,
        args.out,
        input_names=input_names,
        output_names=["logits"],
        dynamic_axes={k: {0: "batch", 1: "seq"} for k in input_names},
        opset_version=17
    )

    print(f"✅ Экспортировано: {args.out}")

if __name__ == "__main__":
    main()
