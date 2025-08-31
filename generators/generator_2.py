#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Генератор таргетной синтетики для задачи move vs no-move.

Цели:
- Подкормить ПАТТЕРНЫ, где были ошибки:
  * POS (move=1): meet/join/wait + предлог + локация; follow/come/walk/run/head/enter/into/onto/across;
    invitations со статич. глаголами как движение: let's sit/rest/hide/watch/gather + prep + place;
    wait in X for me; meet me down by X; come to X with me; walk along the promenade.
  * NEG (move=0): affective/erotic/команды без релокации и без конкретной цели (adventure/private/closer/whisper/tease/undress/don't move),
    гипотетики/планы/констатации: imagine/just imagine/maybe/if we go/tonight at/already at,
    статич. «сидим/остаемся» без приглашения «переместиться куда-то».

Выход: JSONL со строками вида {"id": "...", "text": "...", "label": 0|1}
"""

import argparse, json, random, re, sys
from pathlib import Path
from collections import Counter, defaultdict

# ------------------------ аргументы ------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Путь для сохранения сгенерированного/объединённого датасета (.jsonl)")
    ap.add_argument("--n", type=int, default=1000, help="Сколько СТРОК сгенерировать (50/50). По умолчанию 1000.")
    ap.add_argument("--seed", type=int, default=44, help="Сид генерации.")
    ap.add_argument("--merge", type=str, default="", help="Опционально: путь к существующему .jsonl, чтобы объединить и удалить дубли.")
    return ap.parse_args()

# ------------------------ словари и утилиты ------------------------

PLACES = [
    "study","library","balcony","bridge","crypt","war room","harbor","kitchen","gallery","cellar","chapel",
    "terrace","greenhouse","map room","tavern booth","cathedral steps","rose garden","riverbank",
    "gatehouse","docks","courtyard","tower room","rooftop","bell tower","city gate","secret passage",
    "throne room","observatory","promenade","meadow","battlefield","temple","stone bridge","vineyards",
    "lighthouse","market square","cathedral crypt","garden gate","inn yard","tavern yard","library alcove",
    "study alcove","chapel tower","old oak tree","waterfall","dormitory","bakery","cloister","fountain",
    "harbor quay","city walls","watchtower","armory","smithy","archway","grand hall","abbey","bathhouse"
]

ART = ["the","a",""]  # артикль: the / a / пусто
LOC_AT  = ["in","at","on","near","by","under","inside","within","beside","around","outside"]
MOV_DIR = ["to","into","onto","across","along","toward","towards","down to","inside","up to","through"]

ME_US  = ["me","us"]
WE     = ["we","we both","we two"]
HE_SHE = ["him","her"]

PUNCT_TRAIL = ["", ".", "!", "!!", "..."]  # без эмодзи

def artify(place: str) -> str:
    a = random.choice(ART)
    return f"{a} {place}".strip()

def normalize_key(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def vary_case(s: str) -> str:
    mode = random.choice(["lower","cap","title","orig"])
    if mode == "lower": return s.lower()
    if mode == "cap":   return s.capitalize()
    if mode == "title": return s.title()
    return s

def maybe_punct(s: str) -> str:
    return s + random.choice(PUNCT_TRAIL)

def maybe_comma(s: str) -> str:
    if " to " in s and random.random() < 0.08:
        s = s.replace(" to ", ", to ", 1)
    return s

def maybe_contraction(s: str) -> str:
    s = s.replace("let us ", "let's ")
    if "lets " in s and random.random() < 0.6: s = s.replace("lets ", "let's ")
    if "we are " in s and random.random() < 0.6: s = s.replace("we are ", "we're ")
    return s

def stylize(s: str) -> str:
    s = maybe_contraction(s)
    s = maybe_comma(s)
    s = maybe_punct(s)
    # иногда двойные пробелы, но не злоупотребляем
    if random.random() < 0.12:
        s = re.sub(r"\s+", "  ", s)
    return vary_case(s).strip()

def place() -> str:
    return artify(random.choice(PLACES))

def pick(xs): return random.choice(xs)

# ------------------------ шаблоны ------------------------

# Позитив (move=1): приглашения, явное перемещение, stative-at-destination как move
def pos_templates():
    T = []
    # Каноническое движение
    T += [
        lambda p: f"follow {pick(ME_US)} {pick(['to','into'])} {p}",
        lambda p: f"come with {pick(ME_US)} {pick(['to','into','onto'])} {p}",
        lambda p: f"walk with {pick(ME_US)} {pick(['to','into','onto','across','along'])} {p}",
        lambda p: f"run with {pick(ME_US)} {pick(['to','across'])} {p}",
        lambda p: f"{pick(WE)} enter {p}",
        lambda p: f"{pick(WE)} go {pick(['to','into','onto'])} {p} now",
        lambda p: f"head {pick(['to','toward','towards','down to'])} {p}",
        lambda p: f"come to {p} with me",
        lambda p: f"take {pick(['me','us'])} {pick(['to','into'])} {p}",
        lambda p: f"bring me {pick(['to','into'])} {p}",
    ]
    # Приглашения, которые считаем move
    T += [
        lambda p: f"let's meet {pick(LOC_AT)} {p}",
        lambda p: f"meet me {pick(LOC_AT)} {p}",
        lambda p: f"join me {pick(LOC_AT)} {p}",
        lambda p: f"wait for me {pick(LOC_AT)} {p}",
        lambda p: f"wait {pick(['in','at','by','under'])} {p} for me",
        lambda p: f"meet me down by {p}",
        lambda p: f"join me at {p} tonight",
        lambda p: f"wait for me at {p} tonight",
    ]
    # «Стативные» по месту, но трактуем как move
    T += [
        lambda p: f"let's sit {pick(['in','at','on','by'])} {p}",
        lambda p: f"let's sit by {p}",
        lambda p: f"rest {pick(['in','at','under','by'])} {p} with me",
        lambda p: f"hide together {pick(['in','inside'])} {p}",
        lambda p: f"hide inside {p} together",
        lambda p: f"watch with me at {p}",
        lambda p: f"watch the stars from {p}",
        lambda p: f"gather near {p}",
        lambda p: f"wander through {p}",
        lambda p: f"walk along the promenade",
    ]
    return T

# Негатив (move=0): эмоции/эротика/без конкретной цели/гипотетики/констатации
NEG_GENERIC = [
    "morning angel","you’re so beautiful","don’t stop","don’t be shy","pls forgive me",
    "haha stop teasing pls","don’t tease me","your hands are warm","i'm blushing",
    "whisper my name","pour me another drink","come closer","warm me by the fire","don't move",
]
def neg_templates():
    T = []
    # чистые фразы без перемещения
    T += [lambda p, t=t: t for t in NEG_GENERIC]
    # adventure/private/quiet — без destination
    T += [
        lambda p: "let’s go on an adventure",
        lambda p: "let’s go somewhere private",
        lambda p: "let’s go somewhere quiet",
        lambda p: "let’s go anywhere but here",
        lambda p: "let’s get away for a while",
    ]
    # гипотетики/планы/констатации
    T += [
        lambda p: f"just imagine us {pick(['in','at','on'])} {p}",
        lambda p: f"imagine us {pick(['in','at','on'])} {p}",
        lambda p: f"thinking about {p}",
        lambda p: f"dreaming of {p}",
        lambda p: f"{p} is crowded",
        lambda p: f"no entry to {p}",
        lambda p: f"already at {p}",
        lambda p: f"we’re fine right here",
        lambda p: f"if we go to {p}, we'll see",
        lambda p: f"maybe in {p}",
        lambda p: f"tonight at {p}",
        lambda p: f"could be at {p}",
        lambda p: f"we almost entered {p}",
    ]
    # стат. совместные действия без «переместимся куда-то»
    T += [
        lambda p: f"sit with me {pick(LOC_AT)} {p}",
        lambda p: f"cuddle with me {pick(LOC_AT)} {p}",
        lambda p: f"stay {pick(LOC_AT)} {p}",
        lambda p: f"lie {pick(['in','on','under','by'])} here",
    ]
    return T

# ------------------------ генерация ------------------------

def gen_items(templates, target_count, start_id, label, bucket_name, stats_buckets):
    out, seen = [], set()
    idx, attempts, max_attempts = start_id, 0, target_count * 150
    while len(out) < target_count and attempts < max_attempts:
        attempts += 1
        p = artify(random.choice(PLACES))
        fn = random.choice(templates)
        s = fn(p)
        s = stylize(s)
        key = (normalize_key(s), label)
        if key in seen:
            continue
        seen.add(key)
        out.append({"id": str(idx), "text": s, "label": label, "_bucket": bucket_name})
        idx += 1
    stats_buckets[bucket_name] += len(out)
    return out

def main():
    args = parse_args()
    random.seed(args.seed)

    pos_T = pos_templates()
    neg_T = neg_templates()

    # Хотим 50/50
    n_pos = args.n // 2
    n_neg = args.n - n_pos

    # Лёгкое разбиение на «корзины», чтобы точно накрыть проблемные зоны
    buckets = [
        ("POS_move_core",      pos_T, n_pos),  # всё позитивное
        ("NEG_affective_misc", neg_T, n_neg),  # всё негативное
    ]

    out, stats_buckets = [], defaultdict(int)
    cur_id = 120000
    for name, templ, count in buckets:
        label = 1 if name.startswith("POS") else 0
        out.extend(gen_items(templ, count, cur_id, label, name, stats_buckets))
        cur_id += count + 100  # разнести id-шники

    # Перемешаем, уберём служебное поле
    random.shuffle(out)
    for r in out:
        r.pop("_bucket", None)

    # Объединение с существующим датасетом (если задан)
    merged = out
    if args.merge:
        path = Path(args.merge)
        if not path.exists():
            print(f"[WARN] --merge file not found: {path}", file=sys.stderr)
        else:
            seen = set((normalize_key(d["text"]), d["label"]) for d in merged)
            with path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    key = (normalize_key(str(obj.get("text",""))), int(obj.get("label", -1)))
                    if key in seen: continue
                    seen.add(key)
                    # переносим структуру к единому виду
                    merged.append({"id": str(obj.get("id", "")), "text": str(obj.get("text","")), "label": int(obj.get("label", 0))})

    # Сохраняем
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in merged:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # Статистика
    cnt = Counter([m["label"] for m in out])
    uniq = len({(normalize_key(m["text"]), m["label"]) for m in out})
    print(f"[OK] generated: {len(out)} (uniq={uniq})  pos={cnt[1]}  neg={cnt[0]}")
    if args.merge:
        cnt_m = Counter([m["label"] for m in merged])
        print(f"[OK] merged total: {len(merged)}  pos={cnt_m[1]}  neg={cnt_m[0]}")
    print(f"[Buckets] " + " | ".join(f"{k}:{v}" for k,v in stats_buckets.items()))
    print(f"[Saved] {out_path.resolve()}")

if __name__ == "__main__":
    main()
