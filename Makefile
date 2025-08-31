PYTHON := python

TRAIN_BIN := data/train_binary.jsonl
VAL_BIN   := data/val_binary.jsonl
TRAIN_QA  := data/train_ner.json
VAL_QA    := data/val_ner.json

MOVE_OUT := out/move-det
QA_OUT   := out/qa-distil

.PHONY: prep_binary prep_ner train_binary train_ner eval onnx quantize clean

prep_binary:
	$(PYTHON) helpers/prep_data.py \
	  --in data/all_binary.jsonl \
	  --train_out data/train_binary.jsonl \
	  --val_out data/val_binary.jsonl

prep_ner:
	$(PYTHON) helpers/prep_data.py \
	  --in data/all_move_ner.json \
	  --train_out data/train_ner.json \
	  --val_out data/val_ner.json

train_binary:
	$(PYTHON) train_move.py \
	  --train $(TRAIN_BIN) \
	  --val   $(VAL_BIN) \
	  --out   $(MOVE_OUT) \
	  --epochs 8 \
	  --bs 16 \
	  --lr 1e-5

train_ner:
	$(PYTHON) train_qa.py \
	  --train $(TRAIN_QA) \
	  --val   $(VAL_QA) \
	  --out   $(QA_OUT) \
	  --epochs 4 \
	  --bs 16 \
	  --lr 3e-5

eval:
	$(PYTHON) eval/eval_sanity.py \
	  --predict-file predict_two_heads.py \
	  --model-path cfg/two_heads \
	  --threshold 0.0

eval_binary:
	$(PYTHON) eval/eval_sanity_binary.py \
	  --predict-file predict_two_heads.py \
	  --model-path cfg/two_heads \
	  --threshold 0.0

onnx:
	$(PYTHON) helpers/export_onnx.py \
	  --model $(MOVE_OUT) \
	  --out onnyx/binary_head.onnx
	$(PYTHON) helpers/export_onnx.py \
	  --model $(QA_OUT) \
	  --out onnyx/ner_head.onnx \
	  --qa

quantize8:
	$(PYTHON) helpers/quantize_onnx.py \
	  --input onnyx/binary_head.onnx \
	  --out onnyx/binary_head.int8.onnx \
	  --mode int8
	$(PYTHON) helpers/quantize_onnx.py \
	  --input onnyx/ner_head.onnx \
	  --out onnyx/ner_head.int8.onnx \
	  --mode int8 \

quantize16:
	$(PYTHON) helpers/convert_fp16.py \
	  --in onnyx/binary_head.onnx \
	  --out onnyx/binary_head.fp16.onnx \
	  --keep-io --test-load
	$(PYTHON) helpers/convert_fp16.py \
	  --in onnyx/ner_head.onnx \
	  --out onnyx/ner_head.fp16.onnx \
	  --keep-io --test-load

clean:
	rm -rf $(MOVE_OUT) $(QA_OUT)
