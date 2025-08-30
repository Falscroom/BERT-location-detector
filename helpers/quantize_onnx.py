#!/usr/bin/env python3
import argparse
from pathlib import Path

def q_int8(inp: str, out: str):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input=inp,
        model_output=out,
        per_channel=True,          # можно False, если где-то не поддерживается
        reduce_range=False,
        weight_type=QuantType.QInt8
    )
    print(f"✅ INT8: {inp} → {out}")

def q_fp16(inp: str, out: str):
    import onnx
    from onnxconverter_common import float16
    m = onnx.load(inp)
    m_fp16 = float16.convert_float_to_float16(m)
    onnx.save(m_fp16, out)
    print(f"✅ FP16: {inp} → {out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Входной .onnx")
    ap.add_argument("--out",   required=True, help="Выходной .onnx")
    ap.add_argument("--mode",  choices=["int8","fp16"], default="int8")
    args = ap.parse_args()

    Path(args.input).exists() or exit(f"нет файла: {args.input}")

    if args.mode == "int8":
        q_int8(args.input, args.out)
    else:
        q_fp16(args.input, args.out)

if __name__ == "__main__":
    main()
