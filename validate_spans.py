# validate_spans.py валидим что спаны правильно указывают на ответ
import json

def check(path):
    bad = []
    data = json.load(open(path))
    for ex in data:
        if ex["is_impossible"]:
            continue
        ctx = ex["context"]
        ans = ex["answers"]
        if not ans["text"] or not ans["answer_start"]:
            bad.append((ex["id"], "empty answers"))
            continue
        s = ans["answer_start"][0]
        t = ans["text"][0]
        ctx_span = ctx[s:s+len(t)]
        if ctx_span.lower() != t.lower():
            bad.append((ex["id"], f"mismatch: '{ctx_span}' vs '{t}' at {s}"))
    return bad

for p in ["data/all.json"]:
    b = check(p)
    print(p, "bad:", len(b))
    for i, msg in b[:1000]:
        print(" ", i, msg)
