# rules.py
from __future__ import annotations
import re
from typing import Optional, Iterable, List, Tuple, Sequence

__all__ = [
    "BAD_SPANS",
    "canonicalize_location",
    "passes_net_change",
    "regex_arrival_fallback",
    "regex_presence_fallback",
    "is_negative_text",
    # optional:
    "snap_to_vocab",
    "refine_compound_span",
]

# --- constants / small helpers ------------------------------------------------

BAD_SPANS = {
    "together","again","here","there","soon","now","today","tonight","tomorrow",
    "everyone","someone","anyone","nobody","noone","none","outside","inside","back",
    "home","right away","private","me","you",
    # parts of body / frequent FPs
    "shoulder","my shoulder","ear","my ear","lap","my lap","hand","hands","hair","chest",
    # atmospheric / non-place nouns that were FP
    "bed","fire","moonlight",
}

STRIP_PREFIXES = (
    "to ","into ","inside ","to the ","into the ","back to ",
    "to a ","to an ","the ","a ","an "
)

_TIME = {
    "dawn","dusk","noon","midnight","sunrise","sunset","evening","morning",
    "night","today","tonight","tomorrow","yesterday","later"
}
_PREP = {
    "for","at","by","under","over","near","beside","along","across","around",
    "during","till","until","before","after","from","on"
}

def _alts(xs: Iterable[str]) -> str:
    return "|".join(sorted(map(re.escape, xs), key=len, reverse=True))

# trim trailing adjuncts like "at dusk", "by the river", "until morning", "... not ..."
TRAIL_PATTERN = re.compile(
    rf"("
    rf"\b(?:{_alts(_PREP)})\b.*$"
    rf"|\b(?:{_alts(_TIME)})\b.*$"
    rf"|\s+not\b.*$"
    rf"|\s+\b(?:above|below|beyond|past|after)\b\s+.*$"
    rf")",
    re.I,
)

# --- NEGATIVES ---------------------------------------------------------------
NEG_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bno\s+change\b", re.I),
    re.compile(r"\b(?:stay|stays|stayed|staying)\b", re.I),
    re.compile(r"\b(?:remain|remains|remained|remaining)\b", re.I),
    re.compile(r"\bplan(?:ned)?\s+to\b", re.I),
    re.compile(r"\bjust\s+watch(?:ing|ed)?\b", re.I),
    re.compile(r"\blook(?:ing|ed)?\s+(?:at|from)\b", re.I),

    # fantasies/plans/wishes
    re.compile(r"\bimagin(?:e|ing|ed)\b", re.I),
    re.compile(r"\bdream(?:ing|ed)?\b", re.I),
    re.compile(r"\b(?:i|we)\s+wish\b", re.I),
    re.compile(r"\bhope(?:d|s|ing)?\b", re.I),
    re.compile(r"\b(?:maybe|perhaps|someday)\b", re.I),
    re.compile(r"\bthinking\s+about\b", re.I),
    re.compile(r"\bplanned?\s+to\s+(?:meet|go|visit|be)\b", re.I),
    re.compile(r"\bonly\s+watch(?:ing|ed)?\b", re.I),
]

# arrival-style patterns (prefer last mention)
_ARR_PATTERNS: List[re.Pattern] = [
    re.compile(r"(?:arrived?|reached?|came)\s+(?:at|to|into?)\s+(?:the\s+|a\s+)?([a-z][\w\s'\-]{1,40})", re.I),
    re.compile(r"(?:found\s+(?:my|our)self|ended\s+up)\s+(?:at|in|inside)\s+(?:the\s+|a\s+)?([a-z][\w\s'\-]{1,40})", re.I),
    re.compile(r"(?:looked\s+out\s+from)\s+(?:the\s+|a\s+)?([a-z][\w\s'\-]{1,40})", re.I),
]

# directive-style arrivals: follow/join/meet/lead/led/take/took/bring/brought/drag/pull/carry/guide/usher/escort/…
_DIRECTIVE_ARR_PATTERNS: List[re.Pattern] = [
    re.compile(
        r"\b(?:follow|join|meet|come|walk|head|lead|led|take|took|bring|brought|"
        r"drag(?:ged|ging)?|pull(?:ed|ing)?|carry(?:ing|ied)?|guide(?:d|ing)?|"
        r"usher(?:ed|ing)?|escort(?:ed|ing)?|sneak(?:ed|ing)?|step(?:ped|ping)?|"
        r"run(?:ning|ran)?|rush(?:ed|ing)?|hurr(?:y|ied|ying))"
        r"(?:\s+\w+){0,10}?\s+(?:to|into|onto|inside|in|at|on)\s+(?:the\s+|a\s+|an\s+)?"
        r"([a-z][\w\s'\-]{1,40})",
        re.I
    ),
    # "leading to X"
    re.compile(
        r"\b(?:lead|leading|led)\s+to\s+(?:the\s+|a\s+|an\s+)?([a-z][\w\s'\-]{1,40})",
        re.I
    ),
]

# presence-style patterns (fallback #2)
_PRE_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(?:in|inside)\s+(?:the\s+|a\s+|an\s+)?([a-z][\w\s'\-]{1,40})(?:\b|$)", re.I),
    re.compile(r"\b(?:on|by|at|from|under|near|beside)\s+(?:the\s+|a\s+|an\s+)?([a-z][\w\s'\-]{1,40})(?:\b|$)", re.I),
]

# --- core span hygiene --------------------------------------------------------

def clean_span(s: str) -> str:
    s = s.strip().strip(".,!?;:\"'").lower()
    for pre in STRIP_PREFIXES:
        if s.startswith(pre):
            s = s[len(pre):]
            break
    return s.strip()

def trim_trailing(span: str) -> str:
    m = TRAIL_PATTERN.search(span)
    return span[:m.start()].strip(" \t.,;:!?\"'") if m else span

def canonicalize_location(span: str) -> str:
    return trim_trailing(clean_span(span))

# --- light semantics ----------------------------------------------------------

def is_negative_text(text: str) -> bool:
    t = text.lower()
    return any(p.search(t) for p in NEG_PATTERNS)

def passes_net_change(text: str, span: str, curr_loc: Optional[str]) -> bool:
    """
    Accept if span is a plausible new location; reject obvious junk and "back in X" to the same X.
    """
    s = canonicalize_location(span)
    if not s or s in BAD_SPANS:
        return False
    if curr_loc:
        curr = canonicalize_location(curr_loc)
        if curr == s:
            # returning "back in/inside X" → no net change
            if re.search(rf"\bback\s+(?:in|inside)\s+(?:the\s+|a\s+|an\s+)?{re.escape(s)}\b", text.lower()):
                return False
    return True

# --- fallbacks (directive/arrival → presence) --------------------------------

def _last_group_hit(text: str, patterns: Iterable[re.Pattern]) -> Optional[str]:
    t = text.lower()
    hits: List[Tuple[int, str]] = []
    for pat in patterns:
        for m in pat.finditer(t):
            hits.append((m.start(1), m.group(1)))
    if not hits:
        return None
    return max(hits, key=lambda x: x[0])[1]

def regex_arrival_fallback(text: str) -> Optional[str]:
    if is_negative_text(text):
        return None
    # 1) директивные перемещения
    span = _last_group_hit(text, _DIRECTIVE_ARR_PATTERNS)
    if span:
        return canonicalize_location(span)
    # 2) классические arrival/reach/came
    span = _last_group_hit(text, _ARR_PATTERNS)
    return None if not span else canonicalize_location(span)

def regex_presence_fallback(text: str) -> Optional[str]:
    if is_negative_text(text):
        return None
    span = _last_group_hit(text, _PRE_PATTERNS)
    if not span:
        return None
    span = canonicalize_location(span)
    return None if span in BAD_SPANS else span

# --- optional: strict vocabulary snapping (lightweight) -----------------------

def snap_to_vocab(span: str, vocab: Iterable[str], max_dist: int = 2) -> Optional[str]:
    """
    Snap span to a strict vocab (no aliases), allowing <= max_dist edits.
    Uses a tiny Levenshtein DP to avoid external deps.
    """
    s = canonicalize_location(span)
    if not s:
        return None
    vocab_list = list(vocab)
    if s in vocab_list:
        return s

    def dist(a: str, b: str) -> int:
        la, lb = len(a), len(b)
        if abs(la - lb) > max_dist:
            return max_dist + 1
        dp = list(range(lb + 1))
        for i, ca in enumerate(a, 1):
            prev, dp[0] = dp[0], i
            for j, cb in enumerate(b, 1):
                prev, dp[j] = dp[j], min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
        return dp[-1]

    best = None
    bestd = max_dist + 1
    for v in vocab_list:
        d = dist(s, v)
        if d < bestd:
            best, bestd = v, d
            if bestd == 0:
                break
    return best if bestd <= max_dist else None

# --- compound-span refinement -------------------------------------------------

_COMPOUND_VOCAB: Tuple[str, ...] = (
    # Medieval / scene-specific
    "tavern booth","banquet hall","map room","war room","tower room","tower",
    "library alcove","cathedral steps","garden gate","gatehouse","bell tower",
    "riverbank","bridge","terrace","courtyard","greenhouse","gallery","study",
    "kitchen","bakery","harbor","harbour","fountain","crypt","observatory",
    "market square","castle","dungeon","balcony","hall","cellar","garden",
    "library","cathedral","church","rooftop","tower room",
    # City / Public
    "training ground","city park","skyscraper rooftop","dark alley","subway station",
    "night beach","festival stage","festival backstage","backstage","academy library",
    "abandoned warehouse","mountain hot springs","mountain spring","hot springs",
    # Private / Personal
    "mirko's apartment","mirko’s apartment","mirkos apartment",
    "heroes' dormitory","heroes’ dormitory","dormitory",
    "gym","training hall","japanese bathhouse","sentō","sento","cinema",
    # Extra frequent compounds/synonyms
    "castle gate","castle yard","garden path","harbor pier","harbour pier",
    "city gate","north gate","south gate","east gate","west gate",
    "upper room","map chamber","war chamber","reading room","study room",
)

def refine_compound_span(span: str, text: str, vocab: Sequence[str] = _COMPOUND_VOCAB) -> str:
    """
    Если модель дала короткий head (например, 'library'), а в тексте есть более длинная
    словарная фраза (например, 'library alcove'), возвращаем длинную фразу.
    Ничего не «придумываем» — только если точная фраза содержится в тексте.
    """
    s = canonicalize_location(span)
    if not s:
        return s
    # если уже составной — оставляем
    if " " in s:
        return s
    t = text.lower()
    # ищем словарные фразы, содержащие head и реально присутствующие в тексте
    cand = []
    for v in vocab:
        vl = v.lower()
        if s in vl and vl in t:
            cand.append(vl)
    if not cand:
        return s
    cand.sort(key=len, reverse=True)
    return cand[0]
