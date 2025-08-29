import json, numpy as np
from predict import predict

VAL = "data/val.json"
MODEL = "./out/qa-distil"  # или checkpoint-XX

best=None
for th in np.linspace(0.0, 1.0, 41):   # от 0.00 до 1.00
    tp=fp=fn=tn=0
    for ex in json.load(open(VAL, encoding="utf-8")):
        out = predict(MODEL, ex["context"], null_threshold=float(th))
        pred_null = out["location"] is None
        gold_null = ex["is_impossible"]
        if gold_null and pred_null: tn+=1
        elif gold_null and not pred_null: fp+=1
        elif not gold_null and pred_null: fn+=1
        else: tp+=1
    prec = tp/(tp+fp+1e-9); rec = tp/(tp+fn+1e-9)
    f1   = 2*prec*rec/(prec+rec+1e-9)
    print(f"th={th:.2f} F1={f1:.3f} P={prec:.3f} R={rec:.3f} (tp={tp}, fp={fp}, fn={fn}, tn={tn})")
    if best is None or f1>best[0]: best=(f1,th,prec,rec,tp,fp,fn,tn)
print("BEST:", best)
