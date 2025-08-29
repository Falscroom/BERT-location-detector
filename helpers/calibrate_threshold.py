# calibrate_threshold.py
import json, numpy as np
from predict import predict

VAL = "data/val.json"
MODEL = "out/qa-distil"  # твой каталог
ths = np.linspace(0.2, 0.8, 25)

def eval_f1(th):
    tp=fp=fn=tn=0
    data = json.load(open(VAL))
    for ex in data:
        ctx = ex["context"]
        is_null = ex["is_impossible"]
        out = predict(MODEL, ctx, null_threshold_p=float(th))
        pred_null = (out["location"] is None)
        if is_null and pred_null: tn+=1
        elif is_null and not pred_null: fp+=1
        elif not is_null and pred_null: fn+=1
        else: tp+=1
    # простая F1 по «есть локация/нет»:
    prec = tp / (tp+fp+1e-9)
    rec  = tp / (tp+fn+1e-9)
    f1   = 2*prec*rec/(prec+rec+1e-9)
    return f1, prec, rec, tp, fp, fn, tn

best=None
for th in ths:
    f1,prec,rec,tp,fp,fn,tn = eval_f1(th)
    print(f"th={th:.2f} F1={f1:.3f} P={prec:.3f} R={rec:.3f} (tp={tp}, fp={fp}, fn={fn}, tn={tn})")
    if best is None or f1>best[0]: best=(f1,th)
print("BEST:", best)
