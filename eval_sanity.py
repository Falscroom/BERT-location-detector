#!/usr/bin/env python3
import argparse, json, sys, importlib.util, pathlib
from typing import Optional, Dict, Any, List

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
  {"prompt": "wanna explore the dungeon together", "expected": "dungeon"},
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
