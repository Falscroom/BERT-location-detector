#!/usr/bin/env python3
import argparse, json, sys, importlib.util, pathlib, os
from typing import Optional, Dict, Any, List

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# -------- utilities --------
def _norm(s: Optional[str]) -> Optional[str]:
    return s.strip().lower() if isinstance(s, str) else None

def load_predict_from(predict_file: str, func_name: str = "predict"):
    path = pathlib.Path(predict_file).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Predict file not found: {path}")
    spec = importlib.util.spec_from_file_location(path.stem, str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    if not hasattr(mod, func_name):
        raise AttributeError(f"File {path} has no function `{func_name}`")
    return getattr(mod, func_name)

# -------- default test set (30 prompts) --------
USER_CHAT_TEST: list[dict] = [
  # --- Приветствие и лёгкий флирт ---
  {"prompt": "hi cutie", "expected": None},
  {"prompt": "hey, miss me?", "expected": None},
  {"prompt": "good evening, gorgeous", "expected": None},
  {"prompt": "morning angel", "expected": None},
  {"prompt": "you look adorable", "expected": None},
  {"prompt": "can I sit next to you?", "expected": None},
  {"prompt": "smile for me", "expected": None},
  {"prompt": "I love your eyes", "expected": None},
  {"prompt": "you’re so cute when you blush", "expected": None},
  {"prompt": "don’t hide from me", "expected": None},

  # --- Общение / small talk ---
  {"prompt": "how was your day?", "expected": None},
  {"prompt": "what are you drinking?", "expected": None},
  {"prompt": "tell me about your outfit", "expected": None},
  {"prompt": "what are you thinking now?", "expected": None},
  {"prompt": "do you like me here?", "expected": None},
  {"prompt": "let’s just talk a bit", "expected": None},
  {"prompt": "I like your laugh", "expected": None},
  {"prompt": "you make me happy", "expected": None},
  {"prompt": "stay with me tonight", "expected": None},
  {"prompt": "don’t leave", "expected": None},

  # --- Фантазии и ролевой сеттинг ---
  {"prompt": "let’s go on an adventure", "expected": None},
  {"prompt": "imagine us in the tavern", "expected": None},
  {"prompt": "walk with me to the garden", "expected": "garden"},
  {"prompt": "let’s sneak into the castle", "expected": "castle"},
  {"prompt": "hold my hand in the market square", "expected": "market square"},
  {"prompt": "meet me in the library", "expected": "library"},
  {"prompt": "wanna explore the dungeon together", "expected": None},
  {"prompt": "follow me to the balcony", "expected": "balcony"},
  {"prompt": "let’s sit near the fountain", "expected": "fountain"},
  {"prompt": "dance with me in the hall", "expected": "hall"},

  # --- Прикосновения ---
  {"prompt": "can I hold your hand?", "expected": None},
  {"prompt": "let me hug you", "expected": None},
  {"prompt": "come closer", "expected": None},
  {"prompt": "sit on my lap", "expected": None},
  {"prompt": "I brush your hair back", "expected": None},
  {"prompt": "stroking your cheek softly", "expected": None},
  {"prompt": "I touch your shoulder", "expected": None},
  {"prompt": "pulling you near", "expected": None},
  {"prompt": "leaning on you", "expected": None},
  {"prompt": "rest your head on me", "expected": None},

  # --- Комплименты / эмоции ---
  {"prompt": "you’re so beautiful", "expected": None},
  {"prompt": "you make my heart race", "expected": None},
  {"prompt": "I love hearing your voice", "expected": None},
  {"prompt": "you drive me crazy", "expected": None},
  {"prompt": "I can’t stop staring at you", "expected": None},
  {"prompt": "you’re irresistible", "expected": None},
  {"prompt": "you’re mine tonight", "expected": None},
  {"prompt": "I need you badly", "expected": None},
  {"prompt": "I want you all to myself", "expected": None},
  {"prompt": "don’t tease me", "expected": None},

  # --- Намёки / флирт ---
  {"prompt": "can I kiss you?", "expected": None},
  {"prompt": "come sit on me", "expected": None},
  {"prompt": "I want to feel you close", "expected": None},
  {"prompt": "don’t be shy with me", "expected": None},
  {"prompt": "I like when you tease me", "expected": None},
  {"prompt": "whisper in my ear", "expected": None},
  {"prompt": "you’re making me blush too", "expected": None},
  {"prompt": "I want to taste your lips", "expected": None},
  {"prompt": "lean closer, babe", "expected": None},
  {"prompt": "show me how naughty you can be", "expected": None},

  # --- Интимные инициативы ---
  {"prompt": "can I touch you there?", "expected": None},
  {"prompt": "do you want me right now?", "expected": None},
  {"prompt": "I need you so much", "expected": None},
  {"prompt": "let me undress you slowly", "expected": None},
  {"prompt": "kiss me harder", "expected": None},
  {"prompt": "I can’t hold back anymore", "expected": None},
  {"prompt": "take me with you tonight", "expected": None},
  {"prompt": "I want to hear you moan", "expected": None},
  {"prompt": "let’s go somewhere private", "expected": None},
  {"prompt": "let me please you", "expected": None},

  # --- Ролевые команды / сценарные ---
  {"prompt": "push me against the wall", "expected": None},
  {"prompt": "lay with me on the bed", "expected": None},
  {"prompt": "sit between my legs", "expected": None},
  {"prompt": "pin me down", "expected": None},
  {"prompt": "ride me slowly", "expected": None},
  {"prompt": "guide my hands", "expected": None},
  {"prompt": "let me carry you", "expected": None},
  {"prompt": "whisper my name", "expected": None},
  {"prompt": "straddle me", "expected": None},
  {"prompt": "don’t stop", "expected": None},

  # --- Атмосферные ---
  {"prompt": "pour me another drink", "expected": None},
  {"prompt": "sit with me in the tavern booth", "expected": "tavern booth"},
  {"prompt": "walk together in the moonlight", "expected": None},
  {"prompt": "hide with me in the cellar", "expected": "cellar"},
  {"prompt": "meet me at dawn in the cloister", "expected": "cloister"},
  {"prompt": "wait in the tower for me", "expected": "tower"},
  {"prompt": "lie with me under the stars", "expected": None},
  {"prompt": "sneak out to the balcony", "expected": "balcony"},
  {"prompt": "warm me by the fire", "expected": None},
  {"prompt": "let’s sleep side by side", "expected": None},

  # --- Завершение / эмоции после ---
  {"prompt": "you’re amazing", "expected": None},
  {"prompt": "that was intense", "expected": None},
  {"prompt": "hold me tighter", "expected": None},
  {"prompt": "don’t let me go", "expected": None},
  {"prompt": "you’re mine forever", "expected": None},
  {"prompt": "I feel safe with you", "expected": None},
  {"prompt": "I love this moment", "expected": None},
  {"prompt": "kiss me goodnight", "expected": None},
  {"prompt": "stay till morning", "expected": None},
  {"prompt": "let’s do it again", "expected": None},

    # --- extra prompts ---
    {"prompt": "meet me at the gatehouse", "expected": "gatehouse"},
    {"prompt": "i miss your smile", "expected": None},
    {"prompt": "wait for me in the courtyard", "expected": "courtyard"},
    {"prompt": "you're so sweet tonight", "expected": None},
    {"prompt": "join me in the banquet hall", "expected": "banquet hall"},
    {"prompt": "tell me a secret", "expected": None},
    {"prompt": "follow me to the tower room", "expected": "tower room"},
    {"prompt": "i like your dress", "expected": None},
    {"prompt": "walk with me to the castle yard", "expected": "castle yard"},
    {"prompt": "you smell amazing", "expected": None},
    {"prompt": "come with me into the crypt", "expected": "crypt"},
    {"prompt": "can i hold your hand?", "expected": None},
    {"prompt": "whisper my name again", "expected": None},
    {"prompt": "dance with me in the hall", "expected": "hall"},
    {"prompt": "rest your head on my shoulder", "expected": None},
    {"prompt": "meet me at the cathedral steps", "expected": "cathedral steps"},
    {"prompt": "come closer please", "expected": None},
    {"prompt": "hide with me in the cellar", "expected": "cellar"},
    {"prompt": "stay with me", "expected": None},
    {"prompt": "gather with me in the guild hall", "expected": "guild hall"},
    {"prompt": "kiss me softly", "expected": None},
    {"prompt": "let's sit by the fountain", "expected": "fountain"},
    {"prompt": "i'm thinking of you in the garden", "expected": None},
    {"prompt": "sneak with me into the observatory dome", "expected": "observatory dome"},
    {"prompt": "i dreamed of us in the tower", "expected": None},
    {"prompt": "come to the library alcove", "expected": "library alcove"},
    {"prompt": "i hope we meet at the fountain", "expected": None},
    {"prompt": "meet me on the rooftop", "expected": "rooftop"},
    {"prompt": "let's talk", "expected": None},
    {"prompt": "wait for me at the docks", "expected": "docks"},
    {"prompt": "don't be shy", "expected": None},
    {"prompt": "join me at the harbor", "expected": "harbor"},
    {"prompt": "i can't stop smiling", "expected": None},
    {"prompt": "come with me to the stables", "expected": "stables"},
    {"prompt": "tease me a little", "expected": None},
    {"prompt": "walk with me to the orchard", "expected": "orchard"},
    {"prompt": "do you like this song?", "expected": None},
    {"prompt": "meet me in the greenhouse", "expected": "greenhouse"},
    {"prompt": "you're making me nervous", "expected": None},
    {"prompt": "wait for me in the armory", "expected": "armory"},
    {"prompt": "i want to hold you", "expected": None},
    {"prompt": "meet me in the infirmary", "expected": "infirmary"},
    {"prompt": "can i kiss you goodnight?", "expected": None},
    {"prompt": "join me in the war room", "expected": "war room"},
    {"prompt": "let's stay here a while", "expected": None},
    {"prompt": "come with me to the throne room", "expected": "throne room"},
    {"prompt": "don't leave me alone", "expected": None},
    {"prompt": "follow me to the map room", "expected": "map room"},
    {"prompt": "i need a hug", "expected": None},
    {"prompt": "meet me in the study", "expected": "study"},
    {"prompt": "your hands are warm", "expected": None},
    {"prompt": "let's sit in the gallery", "expected": "gallery"},
    {"prompt": "i love how you laugh", "expected": None},
    {"prompt": "hide with me in the wine cellar", "expected": "wine cellar"},
    {"prompt": "just imagine us in the library", "expected": None},
    {"prompt": "come with me onto the bridge", "expected": "bridge"},
    {"prompt": "maybe later we'll go to the garden", "expected": None},
    {"prompt": "meet me at the riverbank", "expected": "riverbank"},
    {"prompt": "someday let's visit the castle", "expected": None},
    {"prompt": "follow me to the garden gate", "expected": "garden gate"},
    {"prompt": "we're fine right here", "expected": None},
    {"prompt": "wait for me at the gate", "expected": "gate"},
    {"prompt": "i'm not going anywhere", "expected": None},
    {"prompt": "join me on the terrace", "expected": "terrace"},
    {"prompt": "thinking about the balcony view", "expected": None},
    {"prompt": "come with me to the courtyard well", "expected": "courtyard well"},
    {"prompt": "only watching the courtyard from here", "expected": None},
    {"prompt": "meet me in the kitchen", "expected": "kitchen"},
    {"prompt": "we planned to meet in the hall", "expected": None},
    {"prompt": "walk with me to the bakery", "expected": "bakery"},
    {"prompt": "i wish we were in the tavern", "expected": None},
    {"prompt": "follow me into the smithy", "expected": "smithy"},
    {"prompt": "i hope to dance in the hall someday", "expected": None},
    {"prompt": "sneak out with me to the balcony", "expected": "balcony"},
    {"prompt": "dreaming of the cathedral steps", "expected": None},
    {"prompt": "meet me beside the fountain", "expected": "fountain"},
    {"prompt": "planning to sit by the fountain", "expected": None},
    {"prompt": "gather with me on the cathedral steps", "expected": "cathedral steps"},
    {"prompt": "i want you beside me", "expected": None},
    {"prompt": "take my hand and come into the chapel", "expected": "chapel"},
    {"prompt": "you're safe with me", "expected": None},
    {"prompt": "wait for me by the garden gate", "expected": "garden gate"},
    {"prompt": "let's calm down", "expected": None},
    {"prompt": "meet me under the bell tower", "expected": "bell tower"},
    {"prompt": "don't move", "expected": None},
    {"prompt": "join me at the amphitheater", "expected": "amphitheater"},
    {"prompt": "keep talking to me", "expected": None},
    {"prompt": "come with me to the carriage yard", "expected": "carriage yard"},
    {"prompt": "can you pour me some wine?", "expected": None},
    {"prompt": "meet me in the barracks", "expected": "barracks"},
    {"prompt": "i adore your voice", "expected": None},
    {"prompt": "wait with me in the armory", "expected": "armory"},
    {"prompt": "i'm blushing", "expected": None},
    {"prompt": "follow me up to the watchtower", "expected": "watchtower"},
    {"prompt": "let's take it slow", "expected": None},
    {"prompt": "come with me to the pantry", "expected": "pantry"},
    {"prompt": "i need you closer", "expected": None},
    {"prompt": "meet me in the apothecary", "expected": "apothecary"},
    {"prompt": "the moonlight suits you", "expected": None},

{"prompt": "hey babe, follow me to the rooftop", "expected": "rooftop"},
{"prompt": "lol you’re adorable rn", "expected": None},
{"prompt": "meet me in the greenhouse", "expected": "greenhouse"},
{"prompt": "i dreamed of us in the cathedral", "expected": None},
{"prompt": "join me at the harbor tonight", "expected": "harbor"},
{"prompt": "thinking of your smile", "expected": None},
{"prompt": "let’s sneak into the cellar 😉", "expected": "cellar"},
{"prompt": "haha you’re too much", "expected": None},
{"prompt": "come with me into the crypt", "expected": "crypt"},
{"prompt": "i wish we were still in the tavern", "expected": None},

{"prompt": "follow me to the bridge", "expected": "bridge"},
{"prompt": "idk why you make me blush", "expected": None},
{"prompt": "wait for me in the courtyard", "expected": "courtyard"},
{"prompt": "someday we’ll dance in the hall", "expected": None},
{"prompt": "meet me by the fountain rn", "expected": "fountain"},
{"prompt": "you’re mine tonight", "expected": None},
{"prompt": "join me in the banquet hall", "expected": "banquet hall"},
{"prompt": "pls don’t tease me", "expected": None},
{"prompt": "hide with me in the cellar", "expected": "cellar"},
{"prompt": "just imagine us on the balcony", "expected": None},

{"prompt": "come to the study with me", "expected": "study"},
{"prompt": "omg i need your hug rn", "expected": None},
{"prompt": "walk with me to the garden gate", "expected": "garden gate"},
{"prompt": "dreaming of your kiss", "expected": None},
{"prompt": "follow me to the map room", "expected": "map room"},
{"prompt": "you’re so irresistible", "expected": None},
{"prompt": "meet me under the bell tower", "expected": "bell tower"},
{"prompt": "i hope we sit by the fire later", "expected": None},
{"prompt": "join me in the war room", "expected": "war room"},
{"prompt": "i can’t stop staring at you", "expected": None},

{"prompt": "come with me onto the terrace", "expected": "terrace"},
{"prompt": "haha stop making me laugh", "expected": None},
{"prompt": "follow me to the tower room", "expected": "tower room"},
{"prompt": "i planned to stay in the library", "expected": None},
{"prompt": "meet me at the bakery", "expected": "bakery"},
{"prompt": "you make my heart race", "expected": None},
{"prompt": "join me on the rooftop rn", "expected": "rooftop"},
{"prompt": "lol you’re too cute pls", "expected": None},
{"prompt": "wait for me by the garden gate", "expected": "garden gate"},
{"prompt": "i wish we were in the hall now", "expected": None},

{"prompt": "come to the gallery with me", "expected": "gallery"},
{"prompt": "idk what i’d do without you", "expected": None},
{"prompt": "walk with me to the docks", "expected": "docks"},
{"prompt": "thinking about the cathedral view", "expected": None},
{"prompt": "follow me into the kitchen", "expected": "kitchen"},
{"prompt": "haha i can’t handle you rn", "expected": None},
{"prompt": "meet me in the library alcove", "expected": "library alcove"},
{"prompt": "i hope we get there someday", "expected": None},
{"prompt": "join me in the crypt", "expected": "crypt"},
{"prompt": "don’t leave me", "expected": None},

{"prompt": "follow me to the riverbank", "expected": "riverbank"},
{"prompt": "lol stop it babe", "expected": None},
{"prompt": "come with me into the cathedral steps", "expected": "cathedral steps"},
{"prompt": "i dreamed of us in the tavern booth", "expected": None},
{"prompt": "wait in the tower for me", "expected": "tower"},
{"prompt": "haha you’re too hot", "expected": None},
{"prompt": "meet me on the terrace", "expected": "terrace"},
{"prompt": "just imagine us by the fountain", "expected": None},
{"prompt": "join me at the gatehouse", "expected": "gatehouse"},
{"prompt": "you’re glowing tonight", "expected": None},

{"prompt": "come to the apothecary with me", "expected": "apothecary"},
{"prompt": "i hope you’re smiling rn", "expected": None},
{"prompt": "walk with me to the war room", "expected": "war room"},
{"prompt": "maybe later we’ll go to the hall", "expected": None},
{"prompt": "follow me to the greenhouse", "expected": "greenhouse"},
{"prompt": "thinking of your kiss tonight", "expected": None},
{"prompt": "meet me at the cathedral steps", "expected": "cathedral steps"},
{"prompt": "i wish we were under the stars", "expected": None},
{"prompt": "join me in the kitchen", "expected": "kitchen"},
{"prompt": "don’t hide from me", "expected": None},

{"prompt": "follow me into the map room", "expected": "map room"},
{"prompt": "lol i’m blushing rn", "expected": None},
{"prompt": "come with me to the tavern booth", "expected": "tavern booth"},
{"prompt": "i dreamed of you in the garden", "expected": None},
{"prompt": "wait for me at the harbor", "expected": "harbor"},
{"prompt": "haha stop teasing pls", "expected": None},
{"prompt": "meet me in the gallery", "expected": "gallery"},
{"prompt": "idk i just need you closer", "expected": None},
{"prompt": "join me at the bridge", "expected": "bridge"},
{"prompt": "you’re mine forever", "expected": None},

{"prompt": "come into the cellar with me", "expected": "cellar"},
{"prompt": "i hope you think of me too", "expected": None},
{"prompt": "walk with me to the study", "expected": "study"},
{"prompt": "just imagine us in the castle", "expected": None},
{"prompt": "follow me to the cloister", "expected": "cloister"},
{"prompt": "thinking about your eyes", "expected": None},
{"prompt": "meet me at the bakery tonight", "expected": "bakery"},
{"prompt": "i wish we could dance in the hall", "expected": None},
{"prompt": "join me in the tower room", "expected": "tower room"},
{"prompt": "pls stay with me tonight", "expected": None},

{"prompt": "walk with me to the garden", "expected": "garden"},
{"prompt": "meet me at the fountain", "expected": "fountain"},
{"prompt": "sneak away into the library alcove", "expected": "library alcove"},
{"prompt": "we should hide in the cellar", "expected": "cellar"},
{"prompt": "take me to the balcony", "expected": "balcony"},
{"prompt": "follow me into the guild hall", "expected": "guild hall"},
{"prompt": "sit beside me in the tavern booth", "expected": "tavern booth"},
{"prompt": "come rest with me in the courtyard", "expected": "courtyard"},
{"prompt": "join me near the cathedral steps", "expected": "cathedral steps"},
{"prompt": "wait for me in the tower chamber", "expected": "tower chamber"},

{"prompt": "stay with me, don’t leave", "expected": None},
{"prompt": "I like your smile tonight", "expected": None},
{"prompt": "pls don’t go away", "expected": None},
{"prompt": "she laughed softly and nodded", "expected": None},
{"prompt": "we will talk later", "expected": None},
{"prompt": "tonight feels magical", "expected": None},
{"prompt": "again you tease me", "expected": None},
{"prompt": "I’ll remember this forever", "expected": None},
{"prompt": "brb, need a drink", "expected": None},
{"prompt": "it’s too cold outside", "expected": None},

{"prompt": "follow me to the rose garden", "expected": "rose garden"},
{"prompt": "we enter the throne room", "expected": "throne room"},
{"prompt": "run with me across the bridge", "expected": "bridge"},
{"prompt": "hide together in the chapel", "expected": "chapel"},
{"prompt": "rest under the old oak tree", "expected": "old oak tree"},
{"prompt": "dance with me in the great hall", "expected": "great hall"},
{"prompt": "walk by my side through the marketplace", "expected": "marketplace"},
{"prompt": "let’s sneak into the armory", "expected": "armory"},
{"prompt": "meet me at the city gate", "expected": "city gate"},
{"prompt": "climb with me up the watchtower", "expected": "watchtower"},

{"prompt": "I’m too tired rn", "expected": None},
{"prompt": "this song is stuck in my head", "expected": None},
{"prompt": "idk what to do next", "expected": None},
{"prompt": "she blushed deeply", "expected": None},
{"prompt": "pls don’t make me wait", "expected": None},
{"prompt": "I can’t stop thinking about you", "expected": None},
{"prompt": "he smiled back instantly", "expected": None},
{"prompt": "lol that was funny", "expected": None},
{"prompt": "we will see tomorrow", "expected": None},
{"prompt": "stop teasing me now", "expected": None},

{"prompt": "meet me down by the harbor", "expected": "harbor"},
{"prompt": "wait with me near the docks", "expected": "docks"},
{"prompt": "sneak with me through the corridor", "expected": "corridor"},
{"prompt": "rest together in the meadow", "expected": "meadow"},
{"prompt": "let’s sit in the pavilion", "expected": "pavilion"},
{"prompt": "climb into the attic with me", "expected": "attic"},
{"prompt": "follow me to the secret passage", "expected": "secret passage"},
{"prompt": "watch the stars from the rooftop", "expected": "rooftop"},
{"prompt": "walk with me across the square", "expected": "square"},
{"prompt": "gather at the cloister garden", "expected": "cloister garden"},

{"prompt": "pls forgive me", "expected": None},
{"prompt": "I can’t believe this", "expected": None},
{"prompt": "rn I feel nervous", "expected": None},
{"prompt": "again you made me smile", "expected": None},
{"prompt": "stop looking at me like that", "expected": None},
{"prompt": "this memory lingers", "expected": None},
{"prompt": "I promise I’ll stay", "expected": None},
{"prompt": "she whispered something soft", "expected": None},
{"prompt": "I’m laughing too much", "expected": None},
{"prompt": "pls stay close tonight", "expected": None},

{"prompt": "we march into the battlefield", "expected": "battlefield"},
{"prompt": "ride with me to the stables", "expected": "stables"},
{"prompt": "rest in the bathhouse with me", "expected": "bathhouse"},
{"prompt": "sneak out into the alley", "expected": "alley"},
{"prompt": "hide behind the waterfall", "expected": None},
{"prompt": "walk with me into the temple", "expected": "temple"},
{"prompt": "we will wait in the tavern yard", "expected": "tavern yard"},
{"prompt": "climb the spiral staircase to the tower", "expected": "tower"},
{"prompt": "join me in the observatory", "expected": "observatory"},
{"prompt": "stand with me on the balcony edge", "expected": "balcony edge"},

{"prompt": "don’t leave me hanging", "expected": None},
{"prompt": "omg that’s wild", "expected": None},
{"prompt": "pls tell me more", "expected": None},
{"prompt": "asap we need to decide", "expected": None},
{"prompt": "you’re so cute lol", "expected": None},
{"prompt": "this is unforgettable", "expected": None},
{"prompt": "idk what comes after", "expected": None},
{"prompt": "I’ll stay forever", "expected": None},
{"prompt": "they laughed again", "expected": None},
{"prompt": "we will be together", "expected": None},

{"prompt": "let’s meet in the cloister", "expected": "cloister"},
{"prompt": "follow me to the cathedral crypt", "expected": "cathedral crypt"},
{"prompt": "rest in the inn yard", "expected": "inn yard"},
{"prompt": "walk along the promenade", "expected": "promenade"},
{"prompt": "sneak into the kitchen", "expected": "kitchen"},
{"prompt": "hide inside the chapel tower", "expected": "chapel tower"},
{"prompt": "stay with me in the dormitory", "expected": None},
{"prompt": "watch with me at the lighthouse", "expected": "lighthouse"},
{"prompt": "gather near the stone bridge", "expected": "stone bridge"},
{"prompt": "wander through the vineyards", "expected": "vineyards"},

{"prompt": "I moved to New York last week", "expected": "New York"}
]

# -------- evaluation --------
def evaluate(predict_fn, model_path: str, items: List[Dict[str, Any]], movement_threshold: float = 0.0, show_errors: bool = True) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for i, ex in enumerate(items, 1):
        res = predict_fn(model_path, ex["prompt"], curr_loc=ex.get("curr_loc"))
        pred_loc = res.get("location") if isinstance(res, dict) else None
        conf = float(res.get("confidence", 0.0)) if isinstance(res, dict) else 0.0

        exp_loc = ex.get("expected")
        pred_move = (pred_loc is not None) and (conf >= movement_threshold)
        exp_move = (exp_loc is not None)

        exact = (_norm(pred_loc) == _norm(exp_loc))
        rows.append({
            "i": i,
            "prompt": ex["prompt"],
            "curr_loc": ex.get("curr_loc"),
            "expected": exp_loc,
            "predicted": pred_loc,
            "confidence": conf,
            "exp_move": exp_move,
            "pred_move": pred_move,
            "exact": exact,
        })

    total = len(rows)
    overall_acc = sum(r["exact"] or (r["expected"] is None and r["predicted"] is None) for r in rows) / total if total else 0.0

    move_rows   = [r for r in rows if r["exp_move"]]
    nomove_rows = [r for r in rows if not r["exp_move"]]
    move_acc    = sum(r["exact"] for r in move_rows) / len(move_rows) if move_rows else 0.0
    nomove_acc  = sum(r["predicted"] is None for r in nomove_rows) / len(nomove_rows) if nomove_rows else 0.0

    tp = sum(1 for r in rows if r["exp_move"] and r["pred_move"])
    fp = sum(1 for r in rows if not r["exp_move"] and r["pred_move"])
    fn = sum(1 for r in rows if r["exp_move"] and not r["pred_move"])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0

    print("\n=== EVAL SUMMARY ===")
    print(f"Total: {total}")
    print(f"Overall accuracy: {overall_acc:.3f}")
    print(f"Move EM accuracy: {move_acc:.3f}  (n={len(move_rows)})")
    print(f"No-move correctness (pred=None): {nomove_acc:.3f}  (n={len(nomove_rows)})")
    print(f"Movement detection — precision: {precision:.3f}, recall: {recall:.3f}")

    if show_errors:
        for r in rows:
            ok = r["exact"] or (r["expected"] is None and r["predicted"] is None)
            if not ok:
                print(f"[ERR] {r['i']:02d} exp={r['expected']} | pred={r['predicted']} (conf={r['confidence']:.3f})  :: {r['prompt']}")

    return {
        "rows": rows,
        "overall_acc": overall_acc,
        "move_acc": move_acc,
        "nomove_acc": nomove_acc,
        "precision": precision,
        "recall": recall,
    }

# -------- main --------
def main():
    ap = argparse.ArgumentParser(description="Evaluate QA location-change predictor on a small test set.")
    ap.add_argument("--predict-file", required=True, help="Path to a python file that defines predict(model_path, prompt, curr_loc=None) -> {'location': str|None, 'confidence': float}")
    ap.add_argument("--model-path", required=True, help="Path to directory with the trained model (e.g., ./out/qa-distil)")
    ap.add_argument("--test-json", default=None, help="Optional path to JSON with a list of items [{'prompt','expected', 'curr_loc'?}]")
    ap.add_argument("--threshold", type=float, default=0.0, help="Confidence threshold to count movement")
    ap.add_argument("--no-errors", action="store_true", help="Do not print individual errors")
    args = ap.parse_args()

    predict_fn = load_predict_from(args.predict_file, "predict")

    if args.test_json:
        items = json.load(open(args.test_json, "r", encoding="utf-8"))
    else:
        items = USER_CHAT_TEST

    evaluate(predict_fn, args.model_path, items, movement_threshold=args.threshold, show_errors=not args.no_errors)

if __name__ == "__main__":
    main()
