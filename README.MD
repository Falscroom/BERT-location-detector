
#train binary head
python two_heads_pipeline.py train_move \                                                                                 4s py Bert 16:42:06
  --train data/train_binary.jsonl \ 
  --val   data/val_binary.jsonl \
  --out   out/move-det \
  --epochs 4 --bs 16 --lr 2e-5

#train qa head
python two_heads_pipeline.py train_qa \
  --train data/train_ner.json \
  --val data/val_ner.json \
  --out out/qa-distil \
  --epochs 4 \
  --bs 16 \
  --lr 3e-5

python3 eval_sanity.py --predict-file predict.py --model-path ./out/qa-distil
