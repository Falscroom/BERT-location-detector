import random, json

PLACES = ["study","balcony","bridge","crypt","war room","harbor","kitchen","gallery","cellar","chapel","terrace","greenhouse","map room","tavern booth","cathedral steps","rose garden","riverbank","gatehouse","docks","courtyard","tower room","rooftop","bell tower","city gate","secret passage","throne room","observatory","promenade","meadow","battlefield","temple","stone bridge","vineyards"]
MOV_TO   = ["to","into","onto","across","along","toward","down to","inside"]
LOC_AT   = ["in","at","on","near","by","under","inside"]

def pick(xs): return random.choice(xs)

pos_buckets = [
    lambda p: f"follow me {pick(['to','into'])} {p}",
    lambda p: f"come with me {pick(['to','into','onto'])} {p}",
    lambda p: f"walk with me {pick(['to','across','along'])} {p}",
    lambda p: f"run with me {pick(['to','across'])} {p}",
    lambda p: f"let's go {pick(['to','into','onto'])} {p}",
    lambda p: f"let's meet {pick(LOC_AT)} {p}",
    lambda p: f"meet me {pick(LOC_AT)} {p}",
    lambda p: f"join me {pick(LOC_AT)} {p}",
    lambda p: f"wait for me {pick(LOC_AT)} {p}",
    lambda p: f"head {pick(['to','toward','down to'])} {p}",
    lambda p: f"we enter the {p}",
    lambda p: f"we go {pick(['to','into','onto'])} {p}",
    lambda p: f"let me take you {pick(['to','into'])} {p}",
    lambda p: f"i'll bring you {pick(['to','into'])} {p}",
    lambda p: f"follow her {pick(['to','into'])} {p}",
    lambda p: f"lead me {pick(['to','into'])} {p}",
    lambda p: f"guide me {pick(['to','into'])} {p}",
    lambda p: f"carry me {pick(['to','into'])} {p}",
    lambda p: f"take my hand to the {p}",
    lambda p: f"we march {pick(['into','toward'])} the {p}",
    lambda p: f"sneak {pick(['into','onto'])} the {p}",
    lambda p: f"slip {pick(['into','onto'])} the {p}",
    lambda p: f"step {pick(['into','onto'])} the {p}",
    lambda p: f"move {pick(['into','toward'])} the {p}",
    lambda p: f"get {pick(['to','into','onto'])} the {p}",
    lambda p: f"let's head {pick(['to','into'])} the {p}",
    lambda p: f"back to the {p}",
    lambda p: f"return {pick(['to','into'])} the {p}",
    lambda p: f"we're going {pick(['to','into'])} the {p} now",
    lambda p: f"walk me {pick(['to','into'])} the {p}",
    lambda p: f"come along {pick(['to','into'])} the {p}",
    lambda p: f"run back {pick(['to','into'])} the {p}",
    lambda p: f"dash {pick(['to','into'])} the {p}",
    lambda p: f"race {pick(['to','across'])} the {p}",
    lambda p: f"hurry {pick(['to','into'])} the {p}",
    lambda p: f"escort me {pick(['to','into'])} the {p}",
    lambda p: f"lead us {pick(['to','into'])} the {p}",
    lambda p: f"follow the path {pick(['to','into'])} the {p}",
    lambda p: f"head straight {pick(['to','into'])} the {p}",
    lambda p: f"make our way {pick(['to','into'])} the {p}",
    lambda p: f"let's slip away {pick(['to','into'])} the {p}",
    lambda p: f"take the stairs {pick(['to','up to'])} the {p}",
    lambda p: f"climb {pick(['to','up to'])} the {p}",
    lambda p: f"descend {pick(['to','into'])} the {p}",
    lambda p: f"cross the {p}",
    lambda p: f"cut across the {p}",
    lambda p: f"head down the road to the {p}",
    lambda p: f"walk along the promenade to the {p}",
    lambda p: f"from the courtyard to the {p}",
    lambda p: f"finally at the {p}",
]

neg_buckets = [
    lambda p: f"imagine us {pick(LOC_AT)} the {p}",
    lambda p: f"just imagine the {p}",
    lambda p: f"stay {pick(LOC_AT)} the {p}",
    lambda p: f"lie {pick(['in','on','under','by'])} the {p}",
    lambda p: f"lay with me {pick(['on','in'])} the {p}",
    lambda p: f"sit with me {pick(LOC_AT)} the {p}",
    lambda p: f"rest {pick(LOC_AT)} the {p}",
    lambda p: f"push me against the {p}",
    lambda p: f"press me to the {p}",
    lambda p: f"kiss me {pick(LOC_AT)} the {p}",
    lambda p: f"hug me {pick(LOC_AT)} the {p}",
    lambda p: "straddle me",
    lambda p: "come sit on me",
    lambda p: "let me undress you slowly",
    lambda p: f"we are {pick(['in','at','on'])} the {p}",
    lambda p: f"we're staying {pick(['in','at'])} the {p}",
    lambda p: f"we were {pick(['in','at'])} the {p}",
    lambda p: f"dreaming of the {p}",
    lambda p: f"thinking about the {p}",
    lambda p: "your hands are warm",
    lambda p: "i'm blushing",
    lambda p: "haha stop teasing pls",
    lambda p: "pls tell me more",
    lambda p: "walk together in the moonlight",
    lambda p: f"we will wait {pick(['in','at'])} the {p}",
    lambda p: f"watch the stars from the {p}",
    lambda p: f"rest under the old oak tree",
    lambda p: f"watch with me at the {p}",
    lambda p: f"gather near the {p}",
    lambda p: f"wander through the {p}",
    lambda p: "we're fine right here",
    lambda p: f"we're inside the {p}",
    lambda p: f"already at the {p}",
    lambda p: f"around the {p}",
    lambda p: f"near the {p}",
    lambda p: f"by the {p}",
    lambda p: f"under the {p}",
    lambda p: f"inside the {p}",
    lambda p: f"on the {p}",
    lambda p: f"at the {p}",
    lambda p: f"no entry to the {p}",
    lambda p: f"the {p} is crowded",
    lambda p: "sunrise at the dome deck",
    lambda p: f"after dinner in the {p}",
    lambda p: f"tonight at the {p}",
    lambda p: f"maybe in the {p}",
    lambda p: f"if we go to the {p}, we'll see",
    lambda p: f"i'd like to be in the {p}",
    lambda p: f"could be at the {p}",
    lambda p: f"we almost entered the {p}",
]

def gen(bucket_fns, n_buckets, start_id, label):
    out = []
    idx = start_id
    for i in range(n_buckets):
        fn = bucket_fns[i]
        for _ in range(5):
            place = random.choice(PLACES)
            text  = fn(place)
            out.append({"id": str(idx), "text": text, "label": label})
            idx += 1
    return out

pos = gen(pos_buckets, 50, 70000, 1)
neg = gen(neg_buckets, 50, 70250, 0)
print("\n".join(json.dumps(x, ensure_ascii=False) for x in (pos+neg)))
