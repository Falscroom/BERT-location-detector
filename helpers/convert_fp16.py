#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys, os
import onnx
import numpy as np
from onnx import TensorProto, helper, numpy_helper
from typing import Optional

def _to_fp16_tensor(t: onnx.TensorProto) -> onnx.TensorProto:
    # Уже FP16 — вернуть как есть
    if t.data_type == TensorProto.FLOAT16:
        return t
    if t.data_type != TensorProto.FLOAT:
        return t
    arr = numpy_helper.to_array(t)
    # ensure the array is writeable (numpy_helper returns read-only sometimes)
    if not arr.flags.writeable:
        arr = arr.copy()
    # Клип в допустимый диапазон FP16 (in-place)
    np.clip(arr, -65504.0, 65504.0, out=arr)
    arr16 = arr.astype(np.float16, copy=False)
    t16 = numpy_helper.from_array(arr16, name=t.name)
    return t16

def _insert_cast_io(model, to_internal=TensorProto.FLOAT16):
    g = model.graph
    # ---- Входы: FP32 -> FP16 перед первыми потребителями
    for vi in list(g.input):
        ttype = vi.type.tensor_type
        if ttype and ttype.elem_type == TensorProto.FLOAT:
            cast_out = vi.name + "_fp16"
            # Сначала собираем все места-потребители входа (кроме будущего Cast)
            consumer_spots = []
            for node in g.node:
                for k, inp in enumerate(node.input):
                    if inp == vi.name:
                        consumer_spots.append((node, k))
            # Вставляем Cast и подключаем только сохранённых потребителей
            cast_node = helper.make_node(
                "Cast", inputs=[vi.name], outputs=[cast_out],
                name=vi.name + "_cast_to_fp16",
                to=to_internal
            )
            g.node.insert(0, cast_node)
            for node, k in consumer_spots:
                if node is not cast_node:
                    node.input[k] = cast_out

    # ---- Выходы: из FP16 -> FP32
    for vo in list(g.output):
        ttype = vo.type.tensor_type
        if ttype and ttype.elem_type == TensorProto.FLOAT:
            inner_name = vo.name + "_from_fp16"
            # Сначала собираем всех продюсеров, кто пишет в vo.name
            producer_spots = []
            for node in g.node:
                for k, outp in enumerate(node.output):
                    if outp == vo.name:
                        producer_spots.append((node, k))
            # Переназначаем их на inner_name (до вставки Cast)
            for node, k in producer_spots:
                node.output[k] = inner_name
            # Теперь вставляем Cast(inner_name -> vo.name)
            cast_node = helper.make_node(
                "Cast", inputs=[inner_name], outputs=[vo.name],
                name=vo.name + "_cast_to_fp32",
                to=TensorProto.FLOAT
            )
            g.node.append(cast_node)
            # гарантируем, что объявленный тип выхода остался FP32
            vo.type.tensor_type.elem_type = TensorProto.FLOAT
def _vi_elem_type(g: onnx.GraphProto, name: str) -> Optional[int]:
    # search in value_info
    for vi in list(g.value_info) + list(g.input) + list(g.output):
        if vi.name == name and vi.type and vi.type.tensor_type:
            et = vi.type.tensor_type.elem_type
            if et != 0:
                return et
    return None

def _init_elem_type(g: onnx.GraphProto, name: str) -> Optional[int]:
    for t in g.initializer:
        if t.name == name:
            return t.data_type
    return None

def _get_elem_type(g: onnx.GraphProto, name: str) -> Optional[int]:
    et = _init_elem_type(g, name)
    if et is not None:
        return et
    return _vi_elem_type(g, name)

def _insert_cast_before_input(g: onnx.GraphProto, node: onnx.NodeProto, inp_index: int, to_type: int) -> None:
    src = node.input[inp_index]
    if not src:
        return
    cast_out = f"{src}_to_{'fp16' if to_type==TensorProto.FLOAT16 else 'fp32'}_{node.name or node.op_type}_{inp_index}"
    cast_node = helper.make_node(
        "Cast", inputs=[src], outputs=[cast_out],
        name=f"{(node.name or node.op_type)}_cast_in{inp_index}",
        to=to_type
    )
    # place cast node before the current node; order is not strictly required but helps readability
    try:
        idx = list(g.node).index(node)
        g.node.insert(idx, cast_node)
    except ValueError:
        g.node.insert(0, cast_node)
    node.input[inp_index] = cast_out

def _unify_mixed_precision(model: onnx.ModelProto, prefer_fp16: bool = True) -> None:
    """
    For ops that require same dtype across inputs (Div/Add/Sub/Mul/MatMul/Pow),
    if we detect a mix of FLOAT and FLOAT16 (or unknown types), insert Casts to unify.
    prefer_fp16=True -> cast non-FP16 inputs to FP16 when any input is FP16.
    """
    g = model.graph
    unify_ops = {"Div", "Add", "Sub", "Mul", "MatMul", "Pow"}
    for node in list(g.node):
        if node.op_type not in unify_ops:
            continue
        # collect element types of inputs
        types = []
        for inp in node.input:
            if not inp:
                types.append(None)
            else:
                types.append(_get_elem_type(g, inp))
        if not types:
            continue
        has_f16 = any(t == TensorProto.FLOAT16 for t in types if t is not None)
        has_f32 = any(t == TensorProto.FLOAT for t in types if t is not None)
        has_unknown = any(t is None for t in types)
        # Decide if we need to unify
        need_unify = (has_f16 and has_f32) or (has_f16 and has_unknown)
        if not need_unify:
            continue
        target = TensorProto.FLOAT16 if prefer_fp16 and has_f16 else TensorProto.FLOAT
        for i, t in enumerate(types):
            # Cast any input that is not already target when we know it's float
            if t in (TensorProto.FLOAT, TensorProto.FLOAT16) and t != target:
                _insert_cast_before_input(g, node, i, target)
            # If type is unknown but we are unifying to FP16 (common case), cast proactively
            elif t is None and target == TensorProto.FLOAT16:
                _insert_cast_before_input(g, node, i, target)

def _convert_constants_in_node(node: onnx.NodeProto):
    # Конвертируем float32 → float16 внутри Constant-атрибутов
    for i, attr in enumerate(list(node.attribute)):
        if attr.type == onnx.AttributeProto.TENSOR and attr.t.data_type == TensorProto.FLOAT:
            node.attribute[i].t.CopyFrom(_to_fp16_tensor(attr.t))
        elif attr.type == onnx.AttributeProto.TENSORS:
            new_ts = []
            for t in attr.tensors:
                new_ts.append(_to_fp16_tensor(t) if t.data_type == TensorProto.FLOAT else t)
            node.attribute[i].tensors[:] = new_ts

def convert_to_fp16_minimal(inp: str, out: str, keep_io: bool, check: bool, test_load: bool):
    print(f"→ Loading: {inp}")
    model = onnx.load(inp, load_external_data=True)
    g = model.graph

    # 1) Конвертируем initializer-ы
    print("→ Converting initializers (float32 → float16)…")
    name2init = {t.name: t for t in g.initializer}
    for k, t in list(name2init.items()):
        if t.data_type == TensorProto.FLOAT:
            idx = next(i for i, tt in enumerate(g.initializer) if tt.name == k)
            g.initializer[idx].CopyFrom(_to_fp16_tensor(t))

    # 2) Конвертируем константы в узлах (Constant)
    print("→ Converting Constant node tensors…")
    for node in g.node:
        if node.op_type == "Constant":
            _convert_constants_in_node(node)

    # 3) I/O: по умолчанию оставляем типы как есть, но если keep_io=True — убеждаемся, что входы/выходы остаются FLOAT
    if keep_io:
        print("→ Keeping IO types in FP32 via boundary Casts")
        _insert_cast_io(model, to_internal=TensorProto.FLOAT16)

    # 3.5 Попытка инференса типов и устранения смешанных типов на чувствительных узлах
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    # Если где-то в опах типа Div/Add/MatMul смешались FLOAT и FLOAT16 — приведём к единому типу (предпочтение FP16)
    _unify_mixed_precision(model, prefer_fp16=True)

    # 4) Сохраняем с external data (быстро и экономно по памяти)
    print(f"→ Saving FP16 model (external data): {out}")
    onnx.save(model, out, save_as_external_data=True)

    if check:
        print("→ onnx.checker.check_model")
        onnx.checker.check_model(out)

    if test_load:
        try:
            print("→ onnxruntime load check…")
            import onnxruntime as ort
            ort.InferenceSession(out, providers=["CPUExecutionProvider"])
            print("✅ ORT load OK")
        except Exception as e:
            print("⚠ ORT load failed:", e)
            sys.exit(2)

    print("✅ Done")

def main():
    ap = argparse.ArgumentParser(description="Minimal FP32 → FP16 (initializers/Constant only)")
    ap.add_argument("--in",  dest="inp", required=True, help="Input .onnx (FP32)")
    ap.add_argument("--out", dest="out", required=True, help="Output .onnx (FP16)")
    ap.add_argument("--keep-io", action="store_true", help="Keep IO dtypes as FP32")
    ap.add_argument("--no-check", action="store_true", help="Skip onnx.checker")
    ap.add_argument("--test-load", action="store_true", help="Test load in onnxruntime CPU")
    args = ap.parse_args()

    convert_to_fp16_minimal(
        inp=args.inp,
        out=args.out,
        keep_io=args.keep_io,
        check=not args.no_check,
        test_load=args.test_load,
    )

if __name__ == "__main__":
    main()
