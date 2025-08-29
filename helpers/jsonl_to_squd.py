#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re
from typing import Optional, Tuple

QUESTION = "What is the destination location after movement?"

# глаголы движения и конструкции прибытия
MOVE_PATTERNS = [
    r"\b(?:go|going|went|come|coming|came|head(?:ed)?|move(?:d)?|rush(?:ed)?|hurri(?:ed|es)|walk(?:ed|ing)?|run(?:ning|ran)?|proceed(?:ed)?|arriv(?:e|ed)|reach(?:ed)?)\s+(?:to|into|onto|inside|in|at|on)\s+([a-z][\w\s'\-]{2,60})",
    r"\b(?:enter(?:ed|ing)?)\s+(?:the\s+|a\s+|an\s+)?([a-z][\w\s'\-]{2,60})",
    r"\b(?:came|went|moved)\s+(?:into|to|onto)\s+([a-z][\w\s'\-]{2,60})",
    r"\b(?:arrived|reached)\s+(?:at|in|into)\s+([a-z][\w\s'\-]{2,60})",
    r"\b(?:came\s+into)\s+([a-z][\w\s'\-]{2,60})",
]

# присутствие (на крайний случай, если ничего не нашли)
PRESENCE_PATTERNS = [
    r"\b(?:in|inside|at|on|onto|by|beside|under|near)\s+([a-z][\w\s'\-]{2,60})",
]

# что срезать в начале
STRIP_PREFIXES = (
    "to ", "into ", "inside ", "onto ", "in ", "at ", "on ", "by ", "beside ",
    "under ", "near ", "the ", "a ", "an ",
)

# мусорные хвосты и слова
TRAIL_JUNK_RE = re.compile(
    r"(?:\s*,?\s*(?:rn|pls|please|right\??|quietly|now|tonight|today)\b.*$)|(?:\s*[.,!?;:]+$)",
    re.I,
)

def _clean_head(s: str) -> str:
    t = s.strip()
    # снятие артиклей/предлогов один раз (сознательно не в цикле)
    low = t.lower()
    for pre in STRIP_PREFIXES:
        if low.startswith(pre):
            t = t[len(pre):]
            break
    return t.strip()

def _trim_trail(s: str) -> str:
    t = TRAIL_JUNK_RE.sub("", s)
    return t.strip(" \t\n\r.,!?;:\"'")

def _clip_span_in_original(context: str, start: int, end: int) -> Tuple[int, int]:
    """
    Подправить границы после чистки: двигаем внутрь, если с краёв пунктуация/пробелы.
    """
    while start < end and context[start] in " \t\n\r.,!?;:\"'":
        start += 1
    while end > start and context[end-1] in " \t\n\r.,!?;:\"'":
        end -= 1
    return start, end

def _last_match_span(patterns, text: str) -> Optional[Tuple[int, int]]:
    t = text
    hits = []
    for pat in patterns:
        for m in re.finditer(pat, t, flags=re.I):
            s, e = m.start(1), m.end(1)
            hits.append((s, e))
    if not hits:
        return None
    # берём последний (наиболее позднее прибытие)
    return max(hits, key=lambda x: x[0])

def extract_destination(context: str) -> Optional[Tuple[str, int]]:
    """
    Возвращает (answer_text, answer_start) или None.
    Предпочтение — MOVE_PATTERNS; иначе PRESENCE_PATTERNS.
    """
    # 1) движения
    mspan = _last_match_span(MOVE_PATTERNS, context)
    if not mspan:
        # 2) присутствие как fallback
        mspan = _last_match_span(PRESENCE_PATTERNS, context)
    if not mspan:
        return None

    s, e = mspan
    raw = context[s:e]
    # чистка головы/хвоста логикой, но сохранить индексы в оригинале
    head_clean = _clean_head(raw)
    # найдём позицию очищенной головы в исходном фрагменте
    rel = raw.lower().find(head_clean.lower())
    if rel >= 0:
        s = s + rel
        raw = context[s:e]
    # обрежем хвосты
    cleaned = _trim_trail(raw)
    # подвинем end под очищенный текст
    if cleaned and raw.endswith(cleaned) is False:
        e = s + len(cleaned)
    # финальный клип по пунктуации
    s, e = _clip_span_in_original(context, s, e)
    answer = context[s:e].strip()
    if not answer:
        return None
    return answer, s

def convert(in_path: str, out_path: str):
    out = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            _id = str(ex.get("id", line_no))
            ctx = ex.get("text", "")
            lab = int(ex.get("label", 0))

            qa = {
                "id": _id,
                "question": QUESTION,
                "context": ctx,
                "answers": {"text": [], "answer_start": []},
                "is_impossible": lab == 0,
            }

            if lab == 1:
                found = extract_destination(ctx)
                if found:
                    ans, start = found
                    qa["answers"]["text"] = [ans]
                    qa["answers"]["answer_start"] = [start]
                    qa["is_impossible"] = False
                else:
                    # не нашли спан — помечаем как no-answer, чтобы не портить обучение
                    qa["is_impossible"] = True

            out.append(qa)

    with open(out_path, "w", encoding="utf-8") as w:
        json.dump(out, w, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser(description="Convert jsonl (id,text,label) → SQuAD-like QA list")
    ap.add_argument("in_jsonl", help="input jsonl with fields: id,text,label")
    ap.add_argument("out_json", help="output .json (array of QA dicts)")
    args = ap.parse_args()
    convert(args.in_jsonl, args.out_json)

if __name__ == "__main__":
    main()
