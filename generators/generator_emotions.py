#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, random, re
from itertools import product
from pathlib import Path

def norm(s: str) -> str:
    s = re.sub(r"\s+", " ", s.strip())
    s = s[0].lower() + s[1:] if s else s
    return s

# базовые шаблоны без локаций — похожие на твои FP
ACTIONS = [
    "smile for me",
    "take me with you",
    "guide my hands",
    "let me carry you",
    "I feel safe with you",
    "tell me a secret",
    "hold me close",
    "keep me close",
    "keep me near",
    "stay with me",
    "be with me",
    "talk with me",
    "talk to me",
    "kiss me",
    "hug me",
    "cuddle with me",
    "pin me down",
    "walk together",
    "lay with me",
    "lie with me",
    "breathe with me",
    "slow down with me",
    "calm down with me",
    "take it slow",
]

# модальные/интенсификаторы
PREFIXES = [
    "", "please", "just", "maybe", "let’s", "let us", "can you", "could you",
    "will you", "would you", "I want you to", "I need you to", "we should", "we could",
]

ENDINGS = [
    "", "now", "tonight", "today", "soon", "a bit", "for a while",
    "for me", "with me", "right now", "if you want", "no pressure",
]

# опциональные псевдо-локации (включаются флагом --with-locations)
LOCS = [
    "near the fountain", "under the stars", "in the moonlight", "by the fire",
    "at the gate", "on the bed", "by my side", "at our place", "by the window",
    "in the hallway", "by the door",
]

def assemble(base: str, prefix: str, ending: str, loc: str|None) -> str:
    parts = []
    if prefix:
        parts.append(prefix)
    parts.append(base)
    if loc:
        parts.append(loc)
    if ending:
        parts.append(ending)
    s = " ".join(parts)
    s = re.sub(r"\s+", " ", s).strip()
    # пунктуация по вкусу
    if re.search(r"(can|could|will|would)\s+you|^\s*let.?s|^\s*let us", s, re.I):
        if not s.endswith("?"): s += "?"
    else:
        if not s.endswith((".", "!", "?")):
            s += ""
    return norm(s)

def generate(n: int, with_locs: bool, seed: int) -> list[str]:
    rnd = random.Random(seed)
    pool = set()

    # сначала перебираем детерминированные комбы, чтобы быстро набрать разнообразие
    loc_choices = LOCS if with_locs else [None]
    combos = list(product(ACTIONS, PREFIXES, ENDINGS, loc_choices))

    rnd.shuffle(combos)  # лёгкая рандомизация порядка сборки
    for base, pref, end, loc in combos:
        if len(pool) >= n: break
        s = assemble(base, pref, end, loc)
        pool.add(s)

    # если не хватило, делаем стохастическую генерацию
    while len(pool) < n:
        base = rnd.choice(ACTIONS)
        pref = rnd.choice(PREFIXES + ["", "", ""])  # чаще пустой
        end  = rnd.choice(ENDINGS + ["", ""])       # чаще пустой
        loc  = rnd.choice(LOCS) if (with_locs and rnd.random() < 0.6) else None
        s = assemble(base, pref, end, loc)
        pool.add(s)

    out = list(pool)
    # стабильно-случайная сортировка для воспроизводимости
    out.sort(key=lambda x: (len(x), x))
    return out[:n]

def main():
    ap = argparse.ArgumentParser(description="Generate hard-negative style phrases (JSONL).")
    ap.add_argument("--n", type=int, default=1000, help="number of lines to generate")
    ap.add_argument("--out", type=Path, default=Path("hard_negatives.jsonl"))
    ap.add_argument("--label", type=int, default=0, choices=[0,1], help="label to assign")
    ap.add_argument("--start-id", type=int, default=1, help="starting id")
    ap.add_argument("--with-locations", action="store_true", help="allow adding location-like tails")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    phrases = generate(args.n, args.with_locations, args.seed)

    cur = args.start_id
    with args.out.open("w", encoding="utf-8") as w:
        for t in phrases:
            obj = {"id": str(cur), "text": t, "label": args.label}
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")
            cur += 1

    print(f"[OK] wrote {len(phrases)} lines to {args.out}")

if __name__ == "__main__":
    main()
