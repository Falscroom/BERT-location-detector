from __future__ import annotations
from typing import Optional
from functools import lru_cache
import os, json, re, math
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import logging

# Setup logging
logger = logging.getLogger(__name__)


def _alts(xs):
    return "|".join(sorted(map(re.escape, xs), key=len, reverse=True))


# ---------- string normalization helper ----------
from typing import Optional
def _ensure_str(x: Optional[object]) -> str:
    """Coerce arbitrary input to a safe unicode string for tokenizers."""
    if x is None:
        return ""
    if isinstance(x, bytes):
        try:
            return x.decode("utf-8", errors="ignore")
        except Exception:
            return str(x)
    return str(x)


# ---------- runtime / device ----------
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
torch.set_num_threads(1)

# ---------- constants ----------
QUESTION = "What is the destination location after movement? (e.g., kitchen, garden, office)"

MOVE_BACK = {
    "back", "returned", "return", "again", "back inside", "back in", "back to"
}

FILLER_TOKENS = {
    "rn", "now", "pls", "please", "lol", "omg", "brb", "btw", "asap", "idk", "haha", "lmao", "tho", "ye", "yeah", "ok",
    "okay", "again", "together"
}

BAD_SPANS = {
    "together", "again", "here", "there", "soon", "now", "today", "tonight", "tomorrow",
    "everyone", "someone", "anyone", "nobody", "noone", "none", "outside", "inside", "back", "home",
    "right away"
}

NEG_REGEX = [
    r"\bstill\b",
    r"\bno\s+change\b",
    r"\bnobody\s+leaves\b",
    r"\bno\s+one\s+(?:leaves|goes)\b",
    r"\bkept\s+(?:working|talking|waiting|studying|reading|sitting|standing)\b",
    r"\bkeep\s+(?:working|talking|waiting|studying|reading|sitting|standing)\b",
]

STRIP_PREFIXES = (
    "to ", "into ", "inside ", "to the ", "into the ", "back to ",
    "to a ", "to an ", "the ", "a ", "an "
)

MOTION_VERBS = {
    "go", "going", "went", "head", "heading", "enter", "entered", "entering",
    "move", "moved", "moving", "step", "stepped", "rush", "rushed", "run", "running",
    "walk", "walked", "walk into", "sneak", "sneaked", "jump", "jumped",
    "ended up", "found ourselves", "made it to", "made it into", "get to", "got to",
    "settle", "settled", "assemble", "assembled", "muster", "mustered",
    "post", "posted", "gather", "gathered", "waited in", "wait in",
    "kneel", "knelt", "pray", "prayed", "looked out", "looked out from",
    "took our place", "take our place", "debate", "debated", "convene", "convened", "meet", "met",
    "consult", "consulted", "study", "studied", "review", "reviewed",
    "arrive", "arrived", "reach", "reached", "come", "came", "coming",
    "proceed", "proceeded", "advance", "advanced", "climb", "climbed",
    "descend", "descended", "exit", "exited", "leave", "left",
    "crash", "crashed", "crashing"
}

NEG_PATTERNS = {
    "talked about", "spoke about", "said about", "thinking about", "looked at",
    "admired", "just outside", "stayed outside", "not going", "didn't go",
    "never moved", "passed by", "only watched", "only looking",
    "stay", "stayed", "staying", "stayed in", "remain", "remained", "remain in",
    "plan to", "planning to", "imagine", "imagined", "dream", "dreamed", "dreamt",
    "wish", "wished", "hope to", "hoping to", "want to go",
    "planned to", "nearly entered", "almost entered", "stopped in",
    "no entry", "just watching", "watching from", "looking from", "looking at",
    "resting on", "chilling on", "hanging on", "continues in", "continues at"
}

ARRIVAL_PATTERNS = [
    r"(?:settled|assembled|mustered|posted|gathered|knelt|prayed)\s+(?:in|inside|within)\s+([a-z][\w\s'\-]{1,40})",
    r"(?:debated|met|convened|consulted|studied|reviewed).{0,40}?\b(?:in|inside|within)\b\s+([a-z][\w\s'\-]{1,40})",
    r"(?:looked\s+out\s+from)\s+(?:the\s+|a\s+)?([a-z][\w\s'\-]{1,40})",
    r"(?:took\s+(?:our\s+)?place)\s+(?:in|inside|within)\s+([a-z][\w\s'\-]{1,40})",
    r"(?:arrived?|reached?|came)\s+(?:at|to|into?)\s+(?:the\s+|a\s+)?([a-z][\w\s'\-]{1,40})",
    r"(?:found\s+(?:our|my)self|ended\s+up)\s+(?:at|in|inside)\s+(?:the\s+|a\s+)?([a-z][\w\s'\-]{1,40})",
    r"(?:crash(?:ed|ing)?)\s+(?:the\s+|a\s+|an\s+)?([a-z][\w\s'\-]{1,40})",
]

PRESENCE_PATTERNS = [
    r"\b(?:back\s+in|back\s+inside|in|inside)\s+(?:the\s+|a\s+|an\s+)?([a-z][\w\s'\-]{1,40})(?:\b|$)",
    r"\b(?:on|by|at|from)\s+(?:the\s+|a\s+|an\s+)?([a-z][\w\s'\-]{1,40})(?:\b|$)",
    r"\b(?:session|meeting|lunch|dinner|tea\s*break|prayers?|service|photo\s*shoot|sleepover)\s+(?:is|starts?|starting|held)?\s*(?:in|at)\s+(?:the\s+|a\s+|an\s+)?([a-z][\w\s'\-]{1,40})",
    r"\b(?:starts?|starting|started)\s+(?:in|inside|within)\s+(?:the\s+|a\s+|an\s+)?([a-z][\w\s'\-]{1,40})",
    r"\b([a-z][\w\s'\-]{1,40})\s+time\b",
    r"\b(?:is|are|i'm|im|we're|were)\s+(?:in|inside)\s+(?:the\s+|a\s+|an\s+)?([a-z][\w\s'\-]{1,40})",
]

NEG_REGEX_COMPILED = [re.compile(rgx, re.IGNORECASE) for rgx in NEG_REGEX]

TIME_WORDS = {
    "dawn", "dusk", "noon", "midnight", "sunrise", "sunset", "evening", "morning",
    "night", "today", "tonight", "tomorrow", "yesterday", "later"
}

TAIL_PREPS = {
    "for", "at", "by", "under", "over", "near", "beside", "along", "across", "around",
    "during", "till", "until", "before", "after"
}

TIME_WORDS_RE = _alts(TIME_WORDS)
TAIL_PREPS_RE = _alts(TAIL_PREPS)

FILLER_TOKENS_RE = _alts(
    {"rn", "now", "pls", "please", "lol", "omg", "brb", "btw", "asap", "idk", "haha", "lmao", "tho", "ye", "yeah", "ok",
     "okay", "again", "together"})

TRAIL_PATTERN = re.compile(
    rf"""
   (
      \b(?:for|at|until|till|before|after)\s+(?:the\s+|a\s+|an\s+)?(?:{TIME_WORDS_RE})\b.*$
    | \b(?:{TIME_WORDS_RE})\b.*$        # <--- добавляем вот эту строку
    | \b(?:{TAIL_PREPS_RE})\b\s+.*$
    | \bthe\ whole\ time\b.*$
    | \ball\s+(?:night|day|evening|morning|afternoon)\b.*$
    | \b(?:{FILLER_TOKENS_RE})\b.*$
    | \b(?:is|was|were|seems|looks|felt|feels)\b.*$           # predicate clause trim
    | \s+not\b.*$                                             # negation tail trim
   )
   """,
    re.X,
)


# ---------- helpers ----------
def clean_span(s: str) -> str:
    s = s.strip().strip(".,!?;:\"'").lower()
    for pre in STRIP_PREFIXES:
        if s.startswith(pre):
            s = s[len(pre):]
            break
    return s.strip()


def trim_trailing_adjuncts(span: str) -> str:
    s = span.strip().lower()
    m = TRAIL_PATTERN.search(s)
    if m:
        s = s[:m.start()]
    return s.strip(" \t.,;:!?\"'")


def canonicalize_location(span: str) -> str:
    s = clean_span(span)
    s = trim_trailing_adjuncts(s)
    return s


def is_negative_text(text: str, lower_text: Optional[str] = None) -> bool:
    t = lower_text if lower_text is not None else text.lower()
    if any(p in t for p in NEG_PATTERNS):
        return True
    for rgx in NEG_REGEX_COMPILED:
        if rgx.search(t):
            return True
    return False


def passes_net_change(text: str, span: str, curr_loc: Optional[str] = None) -> bool:
    t = text.lower()
    s = canonicalize_location(span)
    if s in BAD_SPANS:
        return False
    if "back inside" in t or "back in" in t:
        if curr_loc and clean_span(curr_loc) == s:
            return False
        if _looks_like_return_to_same(text, s):
            return False
        return True
    if any(k in t for k in MOVE_BACK):
        if curr_loc and clean_span(curr_loc) == s:
            return False
    return True


def has_motion_near(context_lower: str, span_start: int, span_end: int, window: int = 40) -> bool:
    left = max(0, span_start - window)
    right = min(len(context_lower), span_end + window)
    w = context_lower[left:right]
    if any(p in w for p in NEG_PATTERNS):
        return False
    return any(v in w for v in MOTION_VERBS)


def _looks_like_return_to_same(text: str, span: str) -> bool:
    t = text if text.islower() else text.lower()
    s = re.escape(clean_span(span))
    m_back = re.search(rf"\bback\s+(?:in|inside)\s+(?:the\s+|a\s+|an\s+)?{s}\b", t)
    if not m_back:
        return False
    back_idx = m_back.start()
    patterns = [
        rf"\bfrom\s+(?:the\s+|a\s+|an\s+)?{s}\b",
        rf"\bleft\s+(?:the\s+|a\s+|an\s+)?{s}\b",
        rf"\b(out\s+of|outside)\s+(?:the\s+|a\s+|an\s+)?{s}\b",
        rf"\bstepp?ed?\s+out\s+of\s+(?:the\s+|a\s+|an\s+)?{s}\b",
    ]
    return any((m := re.search(p, t)) and m.start() < back_idx for p in patterns)


def regex_arrival_fallback(text: str) -> Optional[str]:
    if is_negative_text(text, lower_text=text.lower()):
        return None
    t = text.lower()
    hits = []
    for pat in ARRIVAL_PATTERNS:
        for m in re.finditer(pat, t):
            hits.append((m.start(1), m.group(1)))
    if not hits:
        return None
    _, span = max(hits, key=lambda x: x[0])
    return canonicalize_location(span)


def regex_presence_fallback(text: str) -> Optional[str]:
    if is_negative_text(text, lower_text=text.lower()):
        return None
    t = text.lower()
    hits = []
    for pat in PRESENCE_PATTERNS:
        for m in re.finditer(pat, t):
            hits.append((m.start(1), m.group(1)))
    if not hits:
        return None
    _, span = max(hits, key=lambda x: x[0])
    span = canonicalize_location(span)
    if span in BAD_SPANS:
        return None
    return span


def _load_config(model_dir: str) -> dict:
    cfg_path = os.path.join(model_dir, "qa_config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
                return cfg
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
    return {}


@lru_cache(maxsize=4)
def _load_session(model_dir: str):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True, local_files_only=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir, local_files_only=True)
    model.to(DEVICE).eval()
    cfg = _load_config(model_dir)
    return tok, model, cfg


# ---------- Дополнительные валидаторы ----------
def extract_features(text: str, span: str, curr_loc: Optional[str] = None) -> dict:
    """Извлекаем features для отладки и мониторинга"""
    features = {
        "has_negative_context": is_negative_text(text),
        "is_question": "?" in text,
        "has_motion_verb": any(v in text.lower() for v in MOTION_VERBS),
        "is_return": curr_loc and canonicalize_location(curr_loc) == canonicalize_location(span),
        "text_length": len(text),
        "span_length": len(span) if span else 0,
    }
    return features


# ---------- main predict ----------
@torch.no_grad()
def predict(model_dir: str,
            text: str,
            curr_loc: Optional[str] = None,
            null_threshold_p: Optional[float] = None,
            strong_conf: Optional[float] = None,
            max_length: Optional[int] = None,
            doc_stride: Optional[int] = None,
            max_span_len: int = 12,
            debug: bool = False) -> dict:
    """
    Predict location change with optional debug info.

    Args:
        debug: If True, return additional debugging information
        :param debug:
        :param max_span_len:
        :param doc_stride:
        :param curr_loc:
        :param text:
        :param model_dir:
        :param null_threshold_p:
        :param max_length:
        :param strong_conf:
    """
    # --- normalize inputs to strings to avoid "TextInputSequence must be str" ---
    text = _ensure_str(text)
    curr_loc_str = _ensure_str(curr_loc) if curr_loc is not None else ""

    tok, model, cfg = _load_session(model_dir)

    # Параметры из конфига с fallback на дефолты
    p_thr = null_threshold_p or cfg.get("null_threshold_p", 0.80)
    strong = strong_conf or cfg.get("strong_conf", 0.80)
    max_len = max_length or cfg.get("max_length", 256)
    stride = doc_stride or cfg.get("doc_stride", 96)
    delta_thr = cfg.get("null_threshold_delta", None)

    # Динамические пороги из конфига
    motion_bonus = cfg.get("motion_bonus", 0.10)
    fallback_arrival_conf = cfg.get("fallback_arrival_conf", 0.78)
    fallback_presence_conf = cfg.get("fallback_presence_conf", 0.76)

    context = f"[CURR={curr_loc_str}] {text}" if curr_loc_str else text
    lower_text = text.lower()
    lower_context = context.lower()

    debug_info = {"method": "model", "fallbacks_tried": []} if debug else None

    # (1) Tokenize
    question = _ensure_str(QUESTION)
    try:
        enc = tok(
            question, context,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation="only_second",
            max_length=max_len,
            stride=stride,
            return_overflowing_tokens=True,
            padding="max_length",
        )
    except TypeError as e:
        # Provide a clearer error to the caller
        raise TypeError(f"Tokenizer inputs must be str. Got types: question={type(question)}, context={type(context)}") from e

    _ = enc.pop("overflow_to_sample_mapping", None)
    offsets_all = enc.pop("offset_mapping")
    n_feats = offsets_all.size(0)

    inputs = {k: v.to(DEVICE) for k, v in enc.items() if k in ("input_ids", "attention_mask")}
    outputs = model(**inputs)
    start_all = outputs.start_logits.cpu()
    end_all = outputs.end_logits.cpu()

    # (2) Find best span
    best_score = float("-inf")
    best_f = best_s = best_e = None
    null_score = float("-inf")

    for f in range(n_feats):
        seq_ids = enc.sequence_ids(f)
        ctx_tok_ids = [i for i, s in enumerate(seq_ids) if s == 1]
        if not ctx_tok_ids:
            continue

        start = start_all[f]
        end = end_all[f]
        null_score = max(null_score, float(start[0] + end[0]))

        for i_idx, si in enumerate(ctx_tok_ids):
            for j_idx in range(i_idx, min(i_idx + max_span_len, len(ctx_tok_ids))):
                ei = ctx_tok_ids[j_idx]
                if ei < si:
                    continue
                sc = float(start[si] + end[ei])
                if sc > best_score:
                    best_score, best_f, best_s, best_e = sc, f, si, ei

    if best_f is None:
        result = {"location": None, "confidence": 1.0}
        if debug:
            result["debug"] = debug_info
        return result

    # Softmax
    m = max(null_score, best_score)
    e_null = math.exp(null_score - m)
    e_best = math.exp(best_score - m)
    z = e_null + e_best
    p_null = e_null / z
    p_best = e_best / z
    delta = best_score - null_score

    # (3) Post-processing
    offsets = offsets_all[best_f].tolist()
    s_char, e_char = offsets[best_s][0], offsets[best_e][1]
    raw = context[s_char:e_char] if e_char <= len(context) else ""
    span = canonicalize_location(raw)
    # Early rejection of obviously bad spans
    if span in BAD_SPANS:
        result = {"location": None, "confidence": float(p_null)}
        if debug:
            debug_info["rejected_bad_span"] = span
            result["debug"] = debug_info
        return result

    motion = has_motion_near(lower_context, s_char, e_char, window=40)
    dyn_p_thr = p_thr - (motion_bonus if motion else 0.0)

    accept_by_p = (p_best >= strong) or (p_best >= p_thr) or (motion and p_best >= dyn_p_thr)
    accept_by_delta = (delta_thr is not None) and (delta > float(delta_thr))

    if span and passes_net_change(text, span, curr_loc) and (accept_by_p or accept_by_delta):
        result = {"location": span, "confidence": float(p_best)}
        if debug:
            result["debug"] = debug_info
            result["debug"]["features"] = extract_features(text, span, curr_loc)
        return result

    # (4) Fallbacks

    if debug:
        debug_info["model_span"] = span
        debug_info["model_conf"] = float(p_best)

    # Arrival fallback
    fallback = regex_arrival_fallback(text)
    if fallback and passes_net_change(text, fallback, curr_loc):
        if debug:
            debug_info["fallbacks_tried"].append({"type": "arrival", "found": fallback})
            debug_info["method"] = "arrival_fallback"
        result = {"location": fallback, "confidence": float(max(p_best, fallback_arrival_conf))}
        if debug:
            result["debug"] = debug_info
        return result

    # Presence fallback
    presence = regex_presence_fallback(text)
    if presence and passes_net_change(text, presence, curr_loc):
        if debug:
            debug_info["fallbacks_tried"].append({"type": "presence", "found": presence})
            debug_info["method"] = "presence_fallback"
        result = {"location": presence, "confidence": float(max(p_best, fallback_presence_conf))}
        if debug:
            result["debug"] = debug_info
        return result

    result = {"location": None, "confidence": float(p_null)}
    if debug:
        result["debug"] = debug_info
    return result


if __name__ == "__main__":
    # Тесты с debug информацией
    print(predict("./out/qa-distil", "I take her hand and we jump to upper room.", debug=True))
    print(predict("./out/qa-distil", "We left the church and went back inside.", curr_loc="church", debug=True))
    print(predict("./out/qa-distil","Past the observatory yard we climbed the spiral and looked out from the dome deck at dawn."))
