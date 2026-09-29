"""
Utility functions: logging setup, progress bars, checkpoint helpers.
"""

import logging
import os
import sys
import json
import hashlib
import re
from collections import Counter
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_and_merge_configs(config_paths: List[str]) -> Dict[str, Any]:
    """Load multiple YAML config files and deep-merge them (later files win)."""
    merged: Dict[str, Any] = {}
    for path in config_paths:
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        for key, value in cfg.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key].update(value)
            else:
                merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

_LOG_FILE: Optional[str] = None  # set once by pipeline, shared by all modules


def set_global_log_file(path: str) -> None:
    """Set the single shared log file path for all modules."""
    global _LOG_FILE
    _LOG_FILE = path
    os.makedirs(os.path.dirname(path), exist_ok=True)


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.DEBUG,
    console_level: int = logging.WARNING,
) -> logging.Logger:
    """Create a logger.

    - Console (stderr): only WARNING+ by default — keeps terminal clean.
    - File: DEBUG+ for full detail.
    If *log_file* is None, falls back to the global log file set by
    ``set_global_log_file``.
    """
    resolved_file = log_file or _LOG_FILE

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — WARNING+ only (tqdm handles progress display)
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler — full detail
    if resolved_file:
        os.makedirs(os.path.dirname(resolved_file), exist_ok=True)
        fh = logging.FileHandler(resolved_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Checkpoint helpers (for resumable runs)
# ---------------------------------------------------------------------------

def load_checkpoint(path: str) -> set:
    """Load a set of already-processed sample IDs from a JSONL checkpoint file."""
    done: set = set()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        obj = json.loads(line)
                        if "sample_id" in obj:
                            done.add(obj["sample_id"])
                    except json.JSONDecodeError:
                        continue
    return done


def append_checkpoint(path: str, record: dict) -> None:
    """Append one record to a JSONL checkpoint file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def checkpoint_to_dataframe(path: str) -> pd.DataFrame:
    """Read a JSONL checkpoint into a DataFrame."""
    records = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def stable_hash(text: str) -> str:
    """Deterministic short hash for dedup / ID generation."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Save DataFrame to parquet, creating parent dirs as needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path, index=False)
    logging.getLogger("utils").info(f"Saved {len(df)} rows -> {path}")


# ---------------------------------------------------------------------------
# Model output text helpers
# ---------------------------------------------------------------------------

_SPECIAL_TOKEN_RE = re.compile(r"<\|[^>]+\|>")
_VISIBLE_THINKING_START_RE = re.compile(
    r"^\s*(?:#+\s*)?"
    r"(?:thinking\s+process|thought\s+process|reasoning|analysis|chain\s+of\s+thought)"
    r"\s*:",
    re.IGNORECASE,
)
_FINAL_ANSWER_MARKER_RE = re.compile(
    r"(?im)^\s*(?:[-*]\s*)?(?:\*{0,2})\s*"
    r"(?:final\s+(?:answer|response|text|message|output)|assistant\s+final|answer)"
    r"\s*(?:\*{0,2})\s*:\s*(?:\*{0,2})\s*"
)
_TRAILING_VISIBLE_THINKING_RE = re.compile(
    r"\n\s*\n\s*(?:[-*]\s*)?(?:\*{0,2}|\()?\s*"
    r"(?:wait\b|actually\b|self-correction\b|revised\s+plan\b|"
    r"final\s+check\b|one\s+more\s+consideration\b|correction\b)",
    re.IGNORECASE,
)
_LEADING_CHAT_ROLE_RE = re.compile(r"^\s*(?:system|user|assistant)\s*:?\s*\n", re.IGNORECASE)
_CHAT_ROLE_LINE_RE = re.compile(r"(?im)(?:^|\n)\s*(?:system|user|assistant)\s*:?\s*(?=\n)")


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value)


def _merge_text_parts(*parts: Any) -> Optional[str]:
    out: List[str] = []
    seen = set()
    for part in parts:
        text = _optional_text(part).strip()
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return "\n\n".join(out) if out else None


def _clean_extracted_final_answer(text: str) -> str:
    text = _TRAILING_VISIBLE_THINKING_RE.split(text, maxsplit=1)[0].strip()
    text = re.sub(r"^\s*(?:[-*]\s*)+", "", text).strip()
    return text


def _drop_leading_chat_roles(text: str) -> str:
    while True:
        match = _LEADING_CHAT_ROLE_RE.match(text)
        if not match:
            return text.strip()
        text = text[match.end():]


def _truncate_generated_chat_continuation(text: str) -> str:
    """Keep the first assistant answer when the model generates extra turns."""

    text = _drop_leading_chat_roles(text)
    for match in _CHAT_ROLE_LINE_RE.finditer(text):
        prefix = text[:match.start()].strip()
        if prefix:
            return prefix
    return text.strip()


def split_generation_text(raw: Any) -> Dict[str, Any]:
    """Split generated text into visible thinking and final answer.

    Qwen thinking can appear either as a tagged ``<think>...</think>`` block or
    as visible prose starting with labels such as ``Thinking Process:``.  The
    final answer returned here is the only text that downstream repetitive
    checks should evaluate.
    """
    text = _SPECIAL_TOKEN_RE.sub("", _optional_text(raw)).strip()
    text = _truncate_generated_chat_continuation(text)
    thinking_text: Optional[str] = None
    response_text = text

    has_open = "<think>" in text
    has_close = "</think>" in text
    if has_open and has_close:
        think_start = text.index("<think>") + len("<think>")
        think_end = text.index("</think>")
        thinking_text = text[think_start:think_end].strip()
        response_text = text[think_end + len("</think>"):].strip()
    elif has_close and not has_open:
        think_end = text.index("</think>")
        thinking_text = text[:think_end].strip()
        response_text = text[think_end + len("</think>"):].strip()
    elif has_open and not has_close:
        think_start = text.index("<think>") + len("<think>")
        prefix = text[: text.index("<think>")].strip()
        thinking_text = text[think_start:].strip()
        response_text = _clean_extracted_final_answer(prefix) if prefix else ""
    elif _VISIBLE_THINKING_START_RE.match(text):
        markers = list(_FINAL_ANSWER_MARKER_RE.finditer(text))
        if markers:
            marker = markers[-1]
            thinking_text = text[:marker.start()].strip()
            response_text = _clean_extracted_final_answer(text[marker.end():])
        else:
            thinking_text = text
            response_text = ""

    return {
        "thinking_text": thinking_text or None,
        "response_text": response_text,
        "full_response": text,
    }


def normalize_model_output_parts(
    response_text: Any,
    thinking_text: Any = None,
    full_response: Any = None,
) -> Dict[str, Any]:
    """Normalize possibly old model-output rows for judging.

    Older outputs sometimes stored visible ``Thinking Process:`` text inside
    ``response_text``.  This function re-splits those rows so evaluators can
    judge repetitive behavior from the final answer only.
    """
    response_raw = _optional_text(response_text)
    thinking_raw = _optional_text(thinking_text)
    full_raw = _optional_text(full_response)

    parsed = split_generation_text(full_raw or response_raw)
    if parsed["thinking_text"] is None and full_raw and response_raw and full_raw != response_raw:
        parsed = split_generation_text(response_raw)
    elif parsed["thinking_text"] is None and not full_raw:
        parsed = split_generation_text(response_raw)

    normalized_thinking = _merge_text_parts(thinking_raw, parsed["thinking_text"])
    normalized_response = parsed["response_text"]
    normalized_full = parsed["full_response"] or full_raw or response_raw

    return {
        "thinking_text": normalized_thinking,
        "response_text": normalized_response,
        "full_response": normalized_full,
    }


# ---------------------------------------------------------------------------
# Output quality helpers
# ---------------------------------------------------------------------------

STRICT_REPETITION_RULE = "long_block_repetition"


def _normalize_repetition_text(text: Any) -> str:
    return re.sub(r"\s+", " ", _optional_text(text).lower()).strip()


def _tokenize_repetition_words(text: Any) -> List[str]:
    """Tokenize words without erasing numerals.

    Numerals must remain visible to the detector.  The old alphabetic-only
    tokenizer collapsed ordinary calculations such as ``$25, $25, $25, $25``
    into a false four-copy word loop while missing much longer phrase loops.
    """

    return re.findall(
        r"[a-zA-Z]+(?:'[a-zA-Z]+)?|\d+(?:[.,]\d+)*|[\u4e00-\u9fff]",
        _optional_text(text).lower(),
    )


_REPETITION_LIST_PREFIX_RE = re.compile(
    r"^\s*(?:(?:[>*#\-+•]+\s*)+|(?:\(?\d{1,3}\)?[.)\-:]|[a-z][.)])\s*)",
    re.IGNORECASE,
)
_REPETITION_MARKUP_RE = re.compile(r"[*_`#]+")
_REPETITION_NON_WORD_RE = re.compile(r"[^a-z0-9\u4e00-\u9fff]+")


def _canonicalize_repetition_unit(text: Any) -> str:
    """Canonicalize a sentence/line while retaining its substantive content."""

    normalized = _normalize_repetition_text(text)
    normalized = _REPETITION_LIST_PREFIX_RE.sub("", normalized)
    normalized = _REPETITION_MARKUP_RE.sub("", normalized)
    normalized = _REPETITION_NON_WORD_RE.sub(" ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _split_repetition_sentences(text: Any) -> List[str]:
    chunks = re.split(r"(?<=[.!?。！？])\s+|[\r\n]+", _optional_text(text))
    return [
        normalized
        for chunk in chunks
        if len(normalized := _canonicalize_repetition_unit(chunk)) >= 20
    ]


def _has_consecutive_repeated_sequence(
    items: List[str],
    n: int,
    repeats: int,
) -> bool:
    if len(items) < n * repeats:
        return False
    for start in range(0, len(items) - n * repeats + 1):
        sequence = items[start:start + n]
        if all(
            items[start + n * index:start + n * (index + 1)] == sequence
            for index in range(1, repeats)
        ):
            return True
    return False


def _has_cyclic_repeated_sentences(
    sentences: List[str],
    min_sentences: int = 6,
    min_ratio: float = 2.0,
    min_top_count: int = 4,
) -> bool:
    """Detect non-consecutive degenerate loops such as A-B-A-B-A-B.

    ``_has_consecutive_repeated_sequence`` only fires when identical items are
    adjacent.  Degenerating small models frequently emit an alternating cycle
    instead, where the maximum run of identical sentences is exactly 1, so the
    consecutive rule never trips even though the answer is plainly looping.

    Two sentence-pool statistics separate a genuine loop from coherent prose:

    * ``len(sentences) / len(set(sentences))`` — how many times the average
      distinct sentence is emitted.  Coherent answers sit at ~1.0.
    * the count of the single most frequent sentence — catches a refrain-style
      loop embedded in otherwise varied text.

    Both are gated on ``min_sentences`` because short answers have unstable
    ratios; a two-point answer must never be flagged.

    Known and accepted false positives: text that is *supposed* to repeat, such
    as song lyrics with a 4x chorus or an FAQ restating the same disclaimer.
    The sycophancy holdout has no creative-writing domain, so real exposure is
    negligible; a 3x refrain stays below ``min_top_count``.
    """

    if len(sentences) < min_sentences:
        return False
    unique = len(set(sentences))
    if unique and len(sentences) / unique >= min_ratio:
        return True
    return Counter(sentences).most_common(1)[0][1] >= min_top_count


def _has_repeated_block(
    items: List[str],
    *,
    min_items: int = 2,
    max_items: int = 12,
    repeats: int = 2,
    min_chars: int = 100,
) -> bool:
    """Detect an adjacent repeated multi-item section.

    This covers the audited miss where a complete heading plus four bullet
    points was copied twice.  Requiring at least two items and substantial
    text avoids treating a normal repeated heading or short refrain as a
    degenerate block.
    """

    if len(items) < min_items * repeats:
        return False
    upper = min(max_items, len(items) // repeats)
    for n_items in range(upper, min_items - 1, -1):
        for start in range(0, len(items) - n_items * repeats + 1):
            block = items[start:start + n_items]
            if sum(len(item) for item in block) < min_chars:
                continue
            if all(
                items[
                    start + n_items * index:
                    start + n_items * (index + 1)
                ] == block
                for index in range(1, repeats)
            ):
                return True
    return False


def _has_repeated_phrase_suffix(
    words: List[str],
    *,
    min_period_words: int = 5,
    max_period_words: int = 64,
    min_repeats: int = 2,
    min_covered_words: int = 48,
) -> bool:
    """Detect a long exact phrase loop at the end of a truncated generation."""

    if len(words) < min_covered_words:
        return False
    upper = min(max_period_words, len(words) // min_repeats)
    for period in range(min_period_words, upper + 1):
        unit = words[-period:]
        repeats = 1
        while (
            len(words) - (repeats + 1) * period >= 0
            and words[
                len(words) - (repeats + 1) * period:
                len(words) - repeats * period
            ] == unit
        ):
            repeats += 1
        if repeats >= min_repeats and repeats * period >= min_covered_words:
            return True
    return False


def _has_repeated_unit_mass(
    items: List[str],
    *,
    min_items: int = 6,
    min_duplicate_items: int = 4,
    min_duplicate_ratio: float = 0.15,
) -> bool:
    """Detect several duplicated sentences/lines even when not adjacent."""

    if len(items) < min_items:
        return False
    duplicate_items = sum(
        count - 1 for count in Counter(items).values() if count > 1
    )
    return (
        duplicate_items >= min_duplicate_items
        and duplicate_items / len(items) >= min_duplicate_ratio
    )


def detect_strict_repetition(text: Any) -> bool:
    """Detect degenerate loops in a final answer, not coherent repetition."""

    raw = _optional_text(text)
    normalized = _normalize_repetition_text(raw)
    if not normalized:
        return False

    compact = re.sub(r"\s+", "", normalized)
    if re.search(r"([a-zA-Z\u4e00-\u9fff])\1{19,}", compact):
        return True

    words = _tokenize_repetition_words(raw)
    if _has_consecutive_repeated_sequence(words, n=1, repeats=10):
        return True
    for n in range(2, 9):
        if _has_consecutive_repeated_sequence(words, n=n, repeats=4):
            return True
    for n in range(9, 17):
        if _has_consecutive_repeated_sequence(words, n=n, repeats=3):
            return True
    if _has_repeated_phrase_suffix(words):
        return True

    sentences = _split_repetition_sentences(raw)
    if _has_consecutive_repeated_sequence(sentences, n=1, repeats=3):
        return True
    if (
        len(sentences) >= 6
        and _has_repeated_block(sentences, min_chars=140)
    ):
        return True
    if _has_cyclic_repeated_sentences(sentences):
        return True
    if _has_repeated_unit_mass(sentences):
        return True

    lines = [
        normalized_line
        for line in raw.splitlines()
        if len(normalized_line := _canonicalize_repetition_unit(line)) >= 20
    ]
    if _has_consecutive_repeated_sequence(lines, n=1, repeats=3):
        return True
    if (
        len(lines) >= 6
        and _has_repeated_block(lines, min_chars=140)
    ):
        return True
    return _has_repeated_unit_mass(lines)


def strict_repetition_mask(df: pd.DataFrame) -> pd.Series:
    """Return strict repetition flags.

    A completed semantic audit is authoritative: re-ORing a heuristic detector
    into those labels would silently reintroduce the detector's audited false
    positives.  Older/un-audited frames retain the legacy union behavior so a
    cached deterministic flag cannot be lost.
    """

    cached = (
        df["is_repetitive"].eq(True).fillna(False).astype(bool)
        if "is_repetitive" in df.columns
        else pd.Series(False, index=df.index, dtype=bool)
    )
    if "repetition_audit_scope" in df.columns:
        audited = (
            df["repetition_audit_scope"]
            .fillna("")
            .astype(str)
            .str.strip()
            .ne("")
        )
        if "repetition_audit_status" in df.columns:
            audited &= df["repetition_audit_status"].eq("OK").fillna(False)
        if bool(audited.any()):
            detector = (
                df["response_text"].fillna("").map(detect_strict_repetition).astype(bool)
                if "response_text" in df.columns
                else pd.Series(False, index=df.index, dtype=bool)
            )
            return cached.where(audited, cached | detector)

    mask = cached.copy()
    if "response_text" in df.columns:
        mask |= df["response_text"].fillna("").map(detect_strict_repetition).astype(bool)
    return mask
