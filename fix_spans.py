# fix_spans.py фиксим спаны в датасете
import json, re, sys
from pathlib import Path

INPUT = sys.argv[1] if len(sys.argv) > 1 else "data/all.json"
OUTPUT = sys.argv[2] if len(sys.argv) > 2 else "data/all.fixed.json"

def find_start(ctx: str, ans: str) -> int | None:
    """Ищем ans в ctx:
       1) по словным границам (лучше)
       2) обычный case-insensitive поиск (fallback)"""
    if not ans:
        return None
    # нормализуем к нижнему регистру
    cl = ctx.lower()
    al = ans.lower()
    # 1) try word-boundaries
    try:
        m = re.search(rf"\b{re.escape(al)}\b", cl)
        if m:
            return m.start()
    except re.error:
        pass
    # 2) plain find
    i = cl.find(al)
    return i if i >= 0 else None

def fix_file(path_in: str, path_out: str):
    data = json.load(open(path_in, "r", encoding="utf-8"))

    total = len(data)
    fixed_pos = 0
    cleared_null = 0
    unable = 0

    for ex in data:
        is_null = bool(ex.get("is_impossible", False))
        ans = ex.get("answers", {"text": [], "answer_start": []})
        ctx = ex.get("context", "")
        # Нормализуем структуру answers
        if "text" not in ans or "answer_start" not in ans:
            ans = {"text": [], "answer_start": []}
            ex["answers"] = ans

        if is_null:
            # для null — answers должны быть пустые
            if ans.get("text") or ans.get("answer_start"):
                ans["text"], ans["answer_start"] = [], []
                cleared_null += 1
            continue

        # positive: answers должны содержать 1 спан
        if not ans.get("text") or not ans.get("answer_start"):
            # попытаемся найти по тексту, если он вообще есть
            t = (ans.get("text") or [""])[0]
            pos = find_start(ctx, t)
            if pos is not None:
                ex["answers"] = {"text": [t], "answer_start": [pos]}
                fixed_pos += 1
            else:
                unable += 1
            continue

        # проверим соответствие
        s = ans["answer_start"][0]
        t = ans["text"][0]
        ctx_span = ctx[s:s+len(t)]
        if ctx_span.lower() != (t or "").lower():
            pos = find_start(ctx, t)
            if pos is not None:
                ex["answers"]["answer_start"][0] = pos
                fixed_pos += 1
            else:
                unable += 1

    with open(path_out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Total: {total}")
    print(f"Fixed answer_start (positives): {fixed_pos}")
    print(f"Cleared answers for nulls:       {cleared_null}")
    print(f"Unable to fix:                   {unable}")
    print(f"Saved -> {path_out}")

if __name__ == "__main__":
    # Пример: python fix_spans.py data/all.json data/all.fixed.json
    fix_file(INPUT, OUTPUT)
