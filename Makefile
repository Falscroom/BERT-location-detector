PYTHON := python3

# пути
TRAIN_BIN := data/train_binary.jsonl
VAL_BIN   := data/val_binary.jsonl
TRAIN_QA  := data/train_ner.json
VAL_QA    := data/val_ner.json

MOVE_OUT := out/move-det
QA_OUT   := out/qa-distil

# --- цели ---

.PHONY: prep_binary prep_ner train_binary train_ner eval clean

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
	  --epochs 4 \
	  --bs 16 \
	  --lr 2e-5

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

clean:
	rm -rf $(MOVE_OUT) $(QA_OUT)
