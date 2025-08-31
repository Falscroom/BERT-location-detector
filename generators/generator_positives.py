#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, json, random, re
from itertools import product
from pathlib import Path

# ===== Лексиконы по твоим темам =====
VERBS = [
    # базовые из примеров
    "rest", "sit", "wait", "stay",
    # чуть расширим, но всё ещё статично
    "meet", "gather", "linger", "pause", "remain",
    "lie", "stand",
]

AUX = [
    "", "let’s", "we will", "we'll", "shall we", "we can", "we could",
    "let us", "let us just",
]

COMPANIONS = [
    "", "together", "with me", "with us", "with you",
]

PREPS = [
    "in", "at", "under", "by", "near", "inside", "within", "around", "beside",
]

# ЛОКАЦИИ — вариации на 6 тем из твоего списка
LOC_OAK = [
    "the old oak", "the old oak tree", "the ancient oak", "the big oak tree",
    "the oak", "the great oak", "the oak shade", "the oak grove",
]
LOC_MEADOW = [
    "the meadow", "the grassy meadow", "the green meadow", "the open meadow",
    "the field", "the flower meadow", "the spring meadow",
]
LOC_PAVILION = [
    "the pavilion", "the garden pavilion", "the wooden pavilion", "the open pavilion",
    "the gazebo", "the stone pavilion",
]
LOC_TAVERN_YARD = [
    "the tavern yard", "the tavern courtyard", "the tavern court", "the tavern back yard",
    "the alehouse yard", "the inn court by the tavern",
]
LOC_INN_YARD = [
    "the inn yard", "the inn courtyard", "the hostel yard", "the coaching inn yard",
    "the inn back yard", "the inn court",
]
LOC_DORM = [
    "the dormitory", "the dorm", "the dorm hall", "the dorm corridor",
    "the dormitory hall", "the dorm common room",
]

LOC_BUCKETS = [LOC_OAK, LOC_MEADOW, LOC_PAVILION, LOC_TAVERN_YARD, LOC_INN_YARD, LOC_DORM]

# Модификаторы/хвосты
TIME_TAILS = ["", "tonight", "today", "for a while", "for a bit", "after dusk", "before dawn", "this evening"]
PUNCT = ["", ".", "!", "…"]

def norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()

def needs_qmark(s: str) -> bool:
    return bool(re.search(r"^(let.?s|shall we|can we|could we|let us)\b", s, flags=re.I))

def assemble(aux: str, verb: str, comp: str, prep: str, loc: str, tail: str, punct: str) -> str:
    parts = []
    if aux: parts.append(aux)
    parts.append(verb)
    if comp: parts.append(comp)
    parts.append(prep)
    parts.append(loc)
    if tail: parts.append(tail)
    s = norm_spaces(" ".join(parts))
    # пунктуация/вопрос
    if needs_qmark(s):
        if not s.endswith("?"): s += "?"
    else:
        s += punct
    # мелкая нормализация регистра (делаем первую букву строчной — у тебя так чаще)
    if s and s[0].isupper():
        s = s[0].lower() + s[1:]
    return s

def generate(n: int, seed: int) -> list[str]:
    rnd = random.Random(seed)
    pool = set()

    # 1) Детерминированные переборы по всем темам, чтобы покрыть разнообразие
    for bucket in LOC_BUCKETS:
        combos = list(product(AUX, VERBS, COMPANIONS, PREPS, bucket, TIME_TAILS, PUNCT))
        rnd.shuffle(combos)  # чтобы не шли группами
        for aux, verb, comp, prep, loc, tail, punct in combos:
            if len(pool) >= n: break
            # фильтры, чтобы фразы были здравые:
            if verb in ("wait", "stay") and comp == "together" and rnd.random() < 0.3:
                # "stay together in X" норм, оставим как есть
                pass
            # избегаем избыточных "together with me/us/you"
            if comp.startswith("with") and rnd.random() < 0.15:
                comp = "together"
            s = assemble(aux, verb, comp, prep, loc, tail, rnd.choice(PUNCT))
            pool.add(s)
        if len(pool) >= n: break

    # 2) Если не хватило до n — стохастическая догенка
    while len(pool) < n:
        bucket = rnd.choice(LOC_BUCKETS)
        aux    = rnd.choice(AUX + ["", ""])  # чаще пусто
        verb   = rnd.choice(VERBS)
        comp   = rnd.choice(COMPANIONS + ["", ""])  # чаще пусто
        prep   = rnd.choice(PREPS)
        loc    = rnd.choice(bucket)
        tail   = rnd.choice(TIME_TAILS + ["", ""])  # чаще пусто
        punct  = rnd.choice(PUNCT)
        s = assemble(aux, verb, comp, prep, loc, tail, punct)
        pool.add(s)

    out = list(pool)
    # стабильная сортировка для воспроизводимости (не по алфавиту, а мягко)
    out.sort(key=lambda x: (len(x), x))
    return out[:n]

def main():
    ap = argparse.ArgumentParser(description="Generate 'found' location-intent phrases (JSONL).")
    ap.add_argument("--n", type=int, default=1000, help="how many lines to generate")
    ap.add_argument("--out", type=Path, default=Path("found_variations.jsonl"))
    ap.add_argument("--label", type=int, default=1, choices=[0,1], help="label to assign (default 1)")
    ap.add_argument("--start-id", type=int, default=1, help="starting id (as string)")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    phrases = generate(args.n, args.seed)
    cur = args.start_id
    with args.out.open("w", encoding="utf-8") as w:
        for t in phrases:
            obj = {"id": str(cur), "text": t, "label": args.label}
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")
            cur += 1

    print(f"[OK] wrote {len(phrases)} lines to {args.out}")

if __name__ == "__main__":
    main()
