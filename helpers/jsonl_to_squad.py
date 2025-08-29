#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, re
from typing import Optional, Tuple

QUESTION = "What is the destination location after movement?"

# глаголы движения и конструкции прибытия
# глаголы движения и конструкции прибытия
MOVE_PATTERNS = [
    # 1) Широкий список глаголов + до 8 слов-вставок до предлога
    #    покрывает: follow/join/meet/go/come/run/walk/head/move/rush/hurry/proceed/arrive/reach/
    #               return/deploy/jump/slip/sneak/step/lead/led/drag/pull/carry/guide/usher/escort/take/took/bring/brought
    r"\b(?:follow|join|meet|go|going|went|come|coming|came|head(?:ed)?|move(?:d)?|rush(?:ed)?|hurri(?:ed|es)|"
    r"walk(?:ed|ing)?|run(?:ning|ran)?|proceed(?:ed)?|arriv(?:e|ed)|reach(?:ed)?|"
    r"return(?:ed|ing)?|deploy(?:ed|ing)?|jump(?:ed|ing)?|slip(?:ped|ping)?|"
    r"sneak(?:ed|ing)?|step(?:ped|ping)?|lead|led|drag(?:ged|ging)?|pull(?:ed|ing)?|"
    r"carry(?:ing|ied)?|guide(?:d|ing)?|usher(?:ed|ing)?|escort(?:ed|ing)?|"
    r"take|took|bring|brought)"
    r"(?:\s+\w+){0,8}?\s+(?:to|into|onto|inside|in|at|on)\s+([a-z][\w\s'\-]{2,60})",

    # 2) enter X (короткая форма без предлога)
    r"\benter(?:ed|ing)?\s+(?:the\s+|a\s+|an\s+)?([a-z][\w\s'\-]{2,60})",

    # 3) came/went/moved/stepped into|to|onto X (прямое управление)
    r"\b(?:came|went|moved|stepped|took|brought)\s+(?:into|to|onto)\s+([a-z][\w\s'\-]{2,60})",

    # 4) arrived|reached at|in|into X
    r"\b(?:arrived|reached)\s+(?:at|in|into)\s+([a-z][\w\s'\-]{2,60})",

    # 5) came into X
    r"\bcame\s+into\s+([a-z][\w\s'\-]{2,60})",

    # 6) leading to X  (например: “archway leading to the royal balcony …”)
    r"\b(?:lead|leading|led)\s+to\s+(?:the\s+|a\s+|an\s+)?([a-z][\w\s'\-]{2,60})",
]



# присутствие (на крайний случай, если ничего не нашли) — можно оставить как было
PRESENCE_PATTERNS = [
    r"\b(?:in|inside|at|on|onto|by|beside|under|near)\s+([a-z][\w\s'\-]{2,60})",
]

STRIP_PREFIXES = (
    "to ", "into ", "inside ", "onto ", "in ", "at ", "on ", "by ", "beside ",
    "under ", "near ", "the ", "a ", "an ",
)

TRAIL_JUNK_RE = re.compile(
    r"(?:\s*,?\s*(?:rn|pls|please|right\??|quietly|now|tonight|today)\b.*$)"
    r"|(?:\s+\b(?:above|below|beyond|past|after)\b\s+.*$)"
    r"|(?:\s*[.,!?;:]+$)",
    re.I,
)

def _clean_head(s: str) -> str:
    t = s.strip()
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
    while start < end and context[start] in " \t\n\r.,!?;:\"'":
        start += 1
    while end > start and context[end-1] in " \t\n\r.,!?;:\"'":
        end -= 1
    return start, end

def _last_match_span(patterns, text: str) -> Optional[Tuple[int, int]]:
    hits = []
    for pat in patterns:
        for m in re.finditer(pat, text, flags=re.I):
            hits.append((m.start(1), m.end(1)))
    return max(hits, key=lambda x: x[0]) if hits else None

def extract_destination(context: str) -> Optional[Tuple[str, int]]:
    mspan = _last_match_span(MOVE_PATTERNS, context) or _last_match_span(PRESENCE_PATTERNS, context)
    if not mspan:
        return None
    s, e = mspan
    raw = context[s:e]
    head_clean = _clean_head(raw)
    rel = raw.lower().find(head_clean.lower())
    if rel >= 0:
        s = s + rel
        raw = context[s:e]
    cleaned = _trim_trail(raw)
    if cleaned and not raw.endswith(cleaned):
        e = s + len(cleaned)
    s, e = _clip_span_in_original(context, s, e)
    answer = context[s:e].strip()
    return (answer, s) if answer else None

def convert(in_path: str, out_path: str):
    out = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            ex = json.loads(line)
            if int(ex.get("label", 0)) != 1:
                continue  # <- пропускаем label=0

            _id = str(ex.get("id", line_no))
            if int(_id) <= 1043:
                continue
            ctx = ex.get("text", "")
            qa = {
                "id": _id,
                "question": QUESTION,
                "context": ctx,
                "answers": {"text": [], "answer_start": []},
                "is_impossible": False,
            }

            found = extract_destination(ctx)
            if found:
                ans, start = found
                qa["answers"]["text"] = [ans]
                qa["answers"]["answer_start"] = [start]
            else:
                # если не нашли — выкидывать? или всё же оставить с is_impossible=True?
                qa["is_impossible"] = True

            out.append(qa)

    with open(out_path, "w", encoding="utf-8") as w:
        json.dump(out, w, ensure_ascii=False, indent=2)

def main():
    ap = argparse.ArgumentParser(description="Convert only label=1 lines to SQuAD-like JSON")
    ap.add_argument("in_jsonl")
    ap.add_argument("out_json")
    args = ap.parse_args()
    convert(args.in_jsonl, args.out_json)

if __name__ == "__main__":
    main()
