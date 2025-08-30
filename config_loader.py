#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, json, glob

REQUIRED_KEYS = {"move_dir", "qa_dir"}

def _abs(base: str, p: str) -> str:
    return p if os.path.isabs(p) else os.path.normpath(os.path.join(base, p))

def _try_load_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _pick_json_with_keys(dir_path: str, keys) -> tuple[str|None, dict|None]:
    cand = os.path.join(dir_path, "config.json")
    if os.path.exists(cand):
        cfg = _try_load_json(cand)
        if cfg and keys.issubset(cfg.keys()):
            return cand, cfg
    # иначе ищем любой *.json
    for p in glob.glob(os.path.join(dir_path, "*.json")):
        cfg = _try_load_json(p)
        if cfg and keys.issubset(cfg.keys()):
            return p, cfg
    return None, None

def load_two_heads_cfg(cfg_dir_or_file: str):
    """
    Возвращает tuple:
      move_dir, qa_dir, question, move_thr, qa_thr, null_bias, max_len, doc_stride, max_span
    Поддерживает:
      - путь к JSON с ключами move_dir/qa_dir
      - папку, где есть two_heads.json или *.json с этими ключами
      - fallback: трактует строку как путь/алиас QA, move_dir берёт из $MOVE_DIR или 'out/move-det'
    """
    # дефолты
    defaults = dict(
        question   = "What is the destination location after movement?",
        move_thr   = 0.60,
        qa_thr     = 0.80,
        null_bias  = 0.0,
        max_len    = 384,
        doc_stride = 128,
        max_span   = 16,
    )

    # 1) JSON файл
    if os.path.isfile(cfg_dir_or_file) and cfg_dir_or_file.endswith(".json"):
        base = os.path.dirname(os.path.abspath(cfg_dir_or_file))
        cfg = _try_load_json(cfg_dir_or_file)
        if not cfg or not REQUIRED_KEYS.issubset(cfg.keys()):
            raise ValueError(f"Config {cfg_dir_or_file} должен содержать ключи {REQUIRED_KEYS}")
        return (
            _abs(base, cfg["move_dir"]),
            _abs(base, cfg["qa_dir"]),
            cfg.get("question", defaults["question"]),
            float(cfg.get("move_thr", cfg.get("move_threshold", defaults["move_thr"]))),
            float(cfg.get("qa_thr",   cfg.get("qa_threshold",   defaults["qa_thr"]))),
            float(cfg.get("null_bias",  defaults["null_bias"])),
            int(cfg.get("max_len",     defaults["max_len"])),
            int(cfg.get("doc_stride",  defaults["doc_stride"])),
            int(cfg.get("max_span_len", defaults["max_span"]))
        )

    # 2) директория
    if os.path.isdir(cfg_dir_or_file):
        picked, cfg = _pick_json_with_keys(cfg_dir_or_file, REQUIRED_KEYS)
        if cfg:
            base = os.path.dirname(os.path.abspath(picked))
            return (
                _abs(base, cfg["move_dir"]),
                _abs(base, cfg["qa_dir"]),
                cfg.get("question", defaults["question"]),
                float(cfg.get("move_thr", cfg.get("move_threshold", defaults["move_thr"]))),
                float(cfg.get("qa_thr",   cfg.get("qa_threshold",   defaults["qa_thr"]))),
                float(cfg.get("null_bias",  defaults["null_bias"])),
                int(cfg.get("max_len",     defaults["max_len"])),
                int(cfg.get("doc_stride",  defaults["doc_stride"])),
                int(cfg.get("max_span_len", defaults["max_span"]))
            )
        raise FileNotFoundError(f"В {cfg_dir_or_file} нет two_heads.json или *.json с ключами {REQUIRED_KEYS}")

    # 3) fallback
    qa_dir   = cfg_dir_or_file
    move_dir = os.environ.get("MOVE_DIR", "out/move-det")
    return (
        move_dir, qa_dir,
        defaults["question"],
        defaults["move_thr"],
        defaults["qa_thr"],
        defaults["null_bias"],
        defaults["max_len"],
        defaults["doc_stride"],
        defaults["max_span"],
    )
