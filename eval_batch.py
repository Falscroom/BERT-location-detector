# eval_batch.py
import json
from predict import predict

VAL = "data/val.json"
MODEL = "./out/qa-distil"  # твоя последняя папка
DIFF_TH = 0.0  # поставь найденный на calibrate_diff

data = json.load(open(VAL, encoding="utf-8"))

tp=fp=fn=tn=0
misses=[]
for ex in data:
    y_null = ex["is_impossible"]
    out = predict(MODEL, ex["context"], null_threshold=DIFF_TH)
    yhat_null = (out["location"] is None)
    ok = (y_null == yhat_null)

    if y_null and yhat_null: tn+=1
    elif y_null and not yhat_null: fp+=1; misses.append(("FP", ex, out))
    elif (not y_null) and yhat_null: fn+=1; misses.append(("FN", ex, out))
    else: tp+=1

prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
f1 = 2*prec*rec/(prec+rec+1e-9)
print(f"P={prec:.3f} R={rec:.3f} F1={f1:.3f} | tp={tp} fp={fp} fn={fn} tn={tn}")

print("\nSample errors (up to 10):")
for tag, ex, out in misses[:10]:
    print(f"[{tag}] id={ex['id']}  ctx={ex['context']!r}")
    print(f"  pred={out}\n")
