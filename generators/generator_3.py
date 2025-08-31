#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adult-oriented synthetic generator for move vs no-move.
- NEG: rich erotic/affective/phatic variants (no relocation intent, no minors).
- POS: minimal pairs to NEG (concrete invite to a PLACE), plus regular movement templates.
- Output: JSONL with {"id": "...", "text": "...", "label": 0|1}
- Balanced 50/50, deduped, deterministic via --seed.
"""

import argparse, json, random, re, sys
from pathlib import Path
from collections import Counter

# ----------------- CLI -----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="Output .jsonl")
    ap.add_argument("--n", type=int, default=1000, help="Total rows (balanced 50/50)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--merge", type=str, default="", help="Optional existing .jsonl to merge & dedupe")
    ap.add_argument("--erotic_weight", type=float, default=1.0, help=">1.0 increases NEG erotic variety/hits")
    return ap.parse_args()

# ----------------- vocab -----------------
PLACES = [
    "tavern","balcony","garden","chapel","library","kitchen","gate","fountain","harbor","bridge",
    "study","gallery","courtyard","crypt","war room","docks","riverbank","rooftop","bell tower",
    "map room","tavern booth","cathedral steps","rose garden","gatehouse","cellar","greenhouse",
    "promenade","meadow","temple","stone bridge","vineyards","lighthouse","market square","cloister",
    "cathedral crypt","garden gate","inn yard","tavern yard","library alcove","chapel tower","watchtower",
    "waterfall","dormitory","bakery","harbor quay","city walls","armory","smithy","archway","grand hall","abbey",
]
ART = ["the","a",""]  # article or none
LOC_AT = ["in","at","on","by","near","under","inside","within","beside","around","outside"]
MOV_DIR = ["to","into","onto","across","along","toward","towards","down to","inside","up to","through"]

def A(place: str) -> str:
    art = random.choice(ART)
    return f"{art} {place}".strip()

def P() -> str:
    return A(random.choice(PLACES))

PUNCT = ["", ".", "!", "!!", "..."]  # no emojis

def normalize_key(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def stylize(s: str) -> str:
    # contractions & light punctuation
    s = s.replace("let us ", "let's ")
    if "lets " in s and random.random() < 0.6: s = s.replace("lets ", "let's ")
    if "we are " in s and random.random() < 0.6: s = s.replace("we are ", "we're ")
    if " to " in s and random.random() < 0.08: s = s.replace(" to ", ", to ", 1)
    s = s + random.choice(PUNCT)
    # random casing & light double spacing
    mode = random.choice(["lower","cap","title","orig"])
    if mode == "lower": s = s.lower()
    elif mode == "cap": s = s.capitalize()
    elif mode == "title": s = s.title()
    if random.random() < 0.12: s = re.sub(r"\s+", "  ", s)
    return s.strip()

# ----------------- erotic/affective NEG templates -----------------
# Intentionally adult but non-graphic; no minors; no explicit illegal themes.
# These NEG do not imply relocation to a PLACE.
NEG_EROTIC_BASE = [
    "hi cutie", "morning angel", "good girl", "good boy",
    "you’re so beautiful", "you look irresistible",
    "don’t stop", "don’t be shy", "don’t tease me",
    "tease me a little", "stop teasing me now",
    "come closer", "kiss me slowly", "hold me tighter",
    "whisper my name", "make me blush", "i'm blushing",
    "your hands are warm", "that was intense",
    "ride me slowly", "straddle me", "come sit on me",
    "pour me another drink", "brb, need a drink", "i need a break",
    "warm me by the fire", "touch me gently",
    "don’t move", "stay with me here",
    # plans/adventure without destination
    "let’s go on an adventure", "let’s get away for a while",
    "let’s go somewhere private", "let’s go somewhere quiet", "let’s go anywhere but here",
    # hypotheticals/plans/imagination
    "we will see tomorrow", "tonight feels magical",
]

# NEG with a PLACE word but still non-move intent (imagination/statement)
def neg_with_place():
    pat = random.choice([
        lambda p: f"imagine us in {p}",
        lambda p: f"just imagine us at {p}",
        lambda p: f"just imagine us on {p}",
        lambda p: f"dreaming of {p}",
        lambda p: f"thinking about {p}",
        lambda p: f"already at {p}",
        lambda p: f"tonight at {p}",
        lambda p: f"maybe in {p}",
        lambda p: f"if we go to {p}, we'll see",
        lambda p: f"{p} is crowded",
        lambda p: f"no entry to {p}",
        lambda p: f"we’re fine right here",
        lambda p: f"sit with me at {p}",
        lambda p: f"cuddle with me in {p}",
        lambda p: f"stay inside {p}",
    ])
    return pat(P())

# ----------------- POS movement templates & minimal pairs -----------------
POS_MOVE_TEMPL = [
    lambda p: f"follow me {random.choice(['to','into'])} {p}",
    lambda p: f"come with me {random.choice(['to','into','onto'])} {p}",
    lambda p: f"walk with me {random.choice(['to','into','onto','across','along'])} {p}",
    lambda p: f"run with me {random.choice(['to','across'])} {p}",
    lambda p: f"head {random.choice(['to','toward','towards','down to'])} {p}",
    lambda p: f"we enter {p}",
    lambda p: f"we go {random.choice(['to','into','onto'])} {p} now",
    lambda p: f"come to {p} with me",
    lambda p: f"meet me {random.choice(LOC_AT)} {p}",
    lambda p: f"join me {random.choice(LOC_AT)} {p}",
    lambda p: f"wait for me {random.choice(LOC_AT)} {p}",
    lambda p: f"meet me down by {p}",
    lambda p: f"wait at {p} for me",
    lambda p: f"let's sit by {p}",
    lambda p: f"rest with me {random.choice(['in','at','under','by'])} {p}",
    lambda p: f"hide together inside {p}",
    lambda p: f"watch the stars from {p}",
    lambda p: f"gather near {p}",
    lambda p: f"walk along the promenade to {p}",
]

# Mapping from erotic NEG to movement POS (minimal pair)
def pos_from_neg(neg: str) -> str:
    n = neg.lower()
    if "come sit on me" in n or "straddle me" in n:
        return f"come sit by {P()}"
    if "ride me" in n:
        return f"sit with me at {P()}"
    if "whisper my name" in n:
        return f"meet me at {P()}"
    if "don’t tease me" in n or "stop teasing" in n or "tease me" in n:
        return f"walk with me to {P()}"
    if "warm me by the fire" in n:
        return f"meet me by {P()}"
    if "your hands are warm" in n or "hold me tighter" in n:
        return f"join me in {P()}"
    if "don’t move" in n:
        return f"wait for me at {P()}"
    if "come closer" in n:
        return f"come to {P()} with me"
    if "pour me another drink" in n or "need a drink" in n:
        return "meet me in the tavern"
    if "let’s go on an adventure" in n or "get away" in n or "somewhere" in n:
        return f"come with me to {P()}"
    if "hi cutie" in n or "morning angel" in n or "beautiful" in n:
        return f"meet me at {P()}"
    if "we will see tomorrow" in n or "tonight feels magical" in n:
        return f"meet me at {P()}"
    # fallback
    return f"meet me at {P()}"

# ----------------- generation -----------------
def gen_neg_erotic(erotic_weight: float):
    # more sampling for erotic base to increase variety
    base = []
    repeats = max(1, int(round(erotic_weight)))
    for _ in range(repeats):
        base += NEG_EROTIC_BASE
    # light surface variants
    out = []
    for t in base:
        v = t
        if "don’t" in v and random.random() < 0.4:
            v = v.replace("don’t","don't")
        if "you’re" in v and random.random() < 0.4:
            v = v.replace("you’re","you're")
        out.append(v)
    return out

def main():
    args = parse_args()
    random.seed(args.seed)

    # Target counts
    total = args.n
    pos_target = total // 2
    neg_target = total - pos_target

    # 1) Build a large NEG pool (erotic + with-place negatives), then sample to neg_target
    neg_pool = []
    # erotic/affective base (weighted)
    for t in gen_neg_erotic(args.erotic_weight):
        neg_pool.append(stylize(t))
    # add a lot of with-place imagination/statement negatives
    for _ in range(int(200 * args.erotic_weight)):
        neg_pool.append(stylize(neg_with_place()))

    # Deduplicate NEG pool before sampling
    neg_pool = list({normalize_key(x): x for x in neg_pool}.values())
    random.shuffle(neg_pool)
    if len(neg_pool) < neg_target:
        # top up by generating more place-based NEG
        while len(neg_pool) < neg_target:
            neg_pool.append(stylize(neg_with_place()))
        neg_pool = list({normalize_key(x): x for x in neg_pool}.values())
    neg_samples = neg_pool[:neg_target]

    # 2) Build POS pool:
    #    (a) minimal pairs for as many NEG as possible
    pos_pool = []
    for n in neg_samples:
        pos_pool.append(stylize(pos_from_neg(n)))
    #    (b) plus generic movement invites to diversify
    while len(pos_pool) < pos_target:
        pos_pool.append(stylize(random.choice(POS_MOVE_TEMPL)(P())))

    # Dedup within POS, then trim to pos_target
    pos_pool = list({normalize_key(x): x for x in pos_pool}.values())
    random.shuffle(pos_pool)
    if len(pos_pool) < pos_target:
        # top up more generic movement
        while len(pos_pool) < pos_target:
            pos_pool.append(stylize(random.choice(POS_MOVE_TEMPL)(P())))
        pos_pool = list({normalize_key(x): x for x in pos_pool}.values())
    pos_samples = pos_pool[:pos_target]

    # 3) Assemble dataset
    rows = []
    idx = 200000
    for t in pos_samples:
        rows.append({"id": str(idx), "text": t, "label": 1}); idx += 1
    for t in neg_samples:
        rows.append({"id": str(idx), "text": t, "label": 0}); idx += 1
    random.shuffle(rows)

    # 4) Merge with existing dataset (optional) + dedupe
    if args.merge:
        seen = set((normalize_key(r["text"]), r["label"]) for r in rows)
        mpath = Path(args.merge)
        if not mpath.exists():
            print(f"[WARN] --merge not found: {mpath}", file=sys.stderr)
        else:
            with mpath.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        o = json.loads(line)
                    except Exception:
                        continue
                    key = (normalize_key(str(o.get("text",""))), int(o.get("label", -1)))
                    if key in seen: continue
                    seen.add(key)
                    rows.append({"id": str(o.get("id","")), "text": str(o.get("text","")), "label": int(o.get("label",0))})
        random.shuffle(rows)

    # 5) Final dedupe & balance check
    uniq = {}
    for r in rows:
        uniq[(normalize_key(r["text"]), r["label"])] = r
    rows = list(uniq.values())
    random.shuffle(rows)

    # 6) Save
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Stats
    cnt = Counter([r["label"] for r in rows])
    print(f"[OK] saved {out.resolve()}")
    print(f"[Stats] total={len(rows)}  pos={cnt[1]}  neg={cnt[0]} (target pos={pos_target}, neg={neg_target})")

if __name__ == "__main__":
    main()
