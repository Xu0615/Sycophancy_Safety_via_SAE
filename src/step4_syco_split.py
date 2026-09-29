"""Shared Step 4 sycophancy SFT train/eval split utilities.

The sycophancy SFT drift experiment must train and evaluate on disjoint rows
from the same shared 5,400-response dataset.  This module owns that deterministic
split so training and posttrain eval cannot silently drift back to Step 2 probe
files or to different prompt distributions.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Tuple


DEFAULT_SHARED_SYCO_DATASET_DIRNAME = "syco_sft_dataset"
DEFAULT_TRAIN_SIZE = 5000
DEFAULT_EVAL_SIZE = 400
DEFAULT_SPLIT_SEED = 1234
SPLIT_VERSION = "shared_syco_sft_train5000_eval400_v2"
EVAL_SOURCE_DATASET = "shared_syco_sft_holdout"
EVAL_SAMPLE_PREFIX = "shared_syco_sft_holdout"
SYCO_RESPONSE_TYPES = {"sycophantic", "syco"}
INSTRUCTION_RESPONSE_TYPES = {"instruction", "neutral_instruction", "alpaca", "alpaca_cleaned"}


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{time.time_ns()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp_path, path)


def _write_json_atomic(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}.{time.time_ns()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp_path, path)


def load_source_metadata(source_path: str) -> Dict[str, Any]:
    path = os.path.join(os.path.dirname(source_path), "dataset_metadata.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_instruction_response_row(row: Dict[str, Any]) -> bool:
    response_type = str(row.get("response_type", "")).strip()
    if response_type in INSTRUCTION_RESPONSE_TYPES:
        return True
    markers = (
        row.get("source_mix_component"),
        row.get("source_dataset"),
        row.get("domain"),
        row.get("id"),
        row.get("sample_id"),
    )
    return any(str(value or "").startswith("alpaca_cleaned") for value in markers)


def validate_source_rows(rows: List[Dict[str, Any]], expected_total: int) -> List[Dict[str, Any]]:
    if len(rows) != expected_total:
        raise ValueError(
            f"Shared syco SFT source must contain exactly {expected_total} rows for "
            f"the configured split; got {len(rows)}"
        )

    out: List[Dict[str, str]] = []
    seen = set()
    for idx, row in enumerate(rows):
        response_type = str(row.get("response_type") or row.get("source_response_type") or "").strip()
        if is_instruction_response_row(row):
            normalized_response_type = "instruction"
        elif response_type in SYCO_RESPONSE_TYPES:
            normalized_response_type = "sycophantic"
        else:
            raise ValueError(
                f"Row {idx} has response_type={response_type!r}; "
                "expected sycophantic/syco or instruction"
            )
        row_id = str(row.get("id") or row.get("syco_id") or row.get("sample_id") or "").strip()
        domain = str(row.get("domain", "")).strip()
        prompt = str(row.get("prompt") or row.get("prompt_text") or "").strip()
        response = str(row.get("response") or row.get("reference_sycophantic_response") or "").strip()
        if not row_id or not domain or not prompt or not response:
            raise ValueError(f"Row {idx} is missing id/domain/prompt/response")
        if row_id in seen:
            raise ValueError(f"Duplicate shared syco SFT row id: {row_id}")
        seen.add(row_id)
        normalized = dict(row)
        normalized.update(
            {
                "id": row_id,
                "domain": domain,
                "prompt": prompt,
                "response": response,
                "response_type": normalized_response_type,
            }
        )
        out.append(normalized)
    return out


def proportional_domain_counts(domain_counts: Counter, total: int) -> Dict[str, int]:
    """Allocate ``total`` rows across domains proportional to source counts."""

    source_total = sum(domain_counts.values())
    if source_total <= 0:
        raise ValueError("Cannot split an empty shared syco SFT dataset")
    floors: Dict[str, int] = {}
    remainders: List[Tuple[float, str]] = []
    assigned = 0
    for domain in sorted(domain_counts):
        exact = total * domain_counts[domain] / source_total
        count = int(math.floor(exact))
        floors[domain] = count
        assigned += count
        remainders.append((exact - count, domain))
    for _, domain in sorted(remainders, key=lambda item: (-item[0], item[1]))[: total - assigned]:
        floors[domain] += 1
    return floors


def split_sort_key(row: Dict[str, str], seed: int) -> Tuple[str, str]:
    payload = f"{seed}\t{row['id']}\t{row['domain']}\t{row['prompt']}"
    return sha256_text(payload), row["id"]


def build_split(
    source_rows: List[Dict[str, Any]],
    train_size: int = DEFAULT_TRAIN_SIZE,
    eval_size: int = DEFAULT_EVAL_SIZE,
    seed: int = DEFAULT_SPLIT_SEED,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    expected_total = int(train_size) + int(eval_size)
    split_version = f"shared_syco_sft_train{int(train_size)}_eval{int(eval_size)}_v2"
    rows = validate_source_rows(source_rows, expected_total=expected_total)
    syco_rows = [row for row in rows if row["response_type"] == "sycophantic"]
    instruction_rows = [row for row in rows if row["response_type"] == "instruction"]
    if len(syco_rows) < int(eval_size):
        raise ValueError(
            f"Shared syco SFT source has only {len(syco_rows)} syco rows; "
            f"cannot create eval_size={eval_size}"
        )
    if len(instruction_rows) > int(train_size):
        raise ValueError(
            f"Shared syco SFT source has {len(instruction_rows)} instruction rows, "
            f"which exceeds train_size={train_size}"
        )
    source_domain_counts = Counter(row["domain"] for row in rows)
    syco_domain_counts = Counter(row["domain"] for row in syco_rows)
    eval_counts = proportional_domain_counts(syco_domain_counts, int(eval_size))

    by_domain: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in syco_rows:
        by_domain[row["domain"]].append(row)

    train_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    for row in instruction_rows:
        train_row = dict(row)
        train_row["split"] = "train"
        train_row["split_version"] = split_version
        train_rows.append(train_row)
    for domain in sorted(by_domain):
        domain_rows = sorted(by_domain[domain], key=lambda row: split_sort_key(row, int(seed)))
        heldout = set(row["id"] for row in domain_rows[: eval_counts[domain]])
        for row in domain_rows:
            if row["id"] in heldout:
                eval_rows.append(
                    {
                        "sample_id": f"{EVAL_SAMPLE_PREFIX}::{row['id']}",
                        "syco_id": row["id"],
                        "source_dataset": EVAL_SOURCE_DATASET,
                        "domain": row["domain"],
                        "eval_probe_type": "natural_holdout",
                        "target_direction": "drift",
                        "prompt_text": row["prompt"],
                        "reference_sycophantic_response": row["response"],
                        "source_response_type": row["response_type"],
                        "prompt_sha256": sha256_text(row["prompt"]),
                        "split_version": split_version,
                    }
                )
            else:
                train_row = dict(row)
                train_row["split"] = "train"
                train_row["split_version"] = split_version
                train_rows.append(train_row)

    train_rows = sorted(train_rows, key=lambda row: (row["domain"], row["id"]))
    eval_rows = sorted(eval_rows, key=lambda row: (row["domain"], row["syco_id"]))
    if len(train_rows) != int(train_size) or len(eval_rows) != int(eval_size):
        raise RuntimeError(
            f"Bad shared syco split: train={len(train_rows)} eval={len(eval_rows)}; "
            f"expected train={train_size} eval={eval_size}"
        )

    metadata = {
        "split_version": split_version,
        "train_size": len(train_rows),
        "eval_size": len(eval_rows),
        "seed": int(seed),
        "source_total": len(rows),
        "source_domain_counts": dict(sorted(source_domain_counts.items())),
        "source_response_type_counts": dict(sorted(Counter(row["response_type"] for row in rows).items())),
        "source_syco_domain_counts": dict(sorted(syco_domain_counts.items())),
        "train_domain_counts": dict(sorted(Counter(row["domain"] for row in train_rows).items())),
        "train_response_type_counts": dict(sorted(Counter(row["response_type"] for row in train_rows).items())),
        "eval_domain_counts": dict(sorted(Counter(row["domain"] for row in eval_rows).items())),
        "eval_source_dataset": EVAL_SOURCE_DATASET,
        "eval_probe_type": "natural_holdout",
        "thinking_aligned_with_training": True,
    }
    return train_rows, eval_rows, metadata


def split_dir_for(source_path: str, train_size: int, eval_size: int, seed: int) -> str:
    return os.path.join(
        os.path.dirname(source_path),
        f"split_train{int(train_size)}_eval{int(eval_size)}_seed{int(seed)}",
    )


def ensure_shared_syco_sft_split(
    source_path: str,
    train_size: int = DEFAULT_TRAIN_SIZE,
    eval_size: int = DEFAULT_EVAL_SIZE,
    seed: int = DEFAULT_SPLIT_SEED,
    split_dir: str | None = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    """Create or validate the shared syco SFT train/eval split."""

    source_path = os.path.abspath(source_path)
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Shared syco SFT dataset not found: {source_path}")
    split_dir = os.path.abspath(split_dir or split_dir_for(source_path, train_size, eval_size, seed))
    train_path = os.path.join(split_dir, "syco_train.jsonl")
    eval_path = os.path.join(split_dir, "syco_eval.jsonl")
    metadata_path = os.path.join(split_dir, "split_metadata.json")
    source_sha256 = sha256_file(source_path)

    def load_valid_cached_split() -> Dict[str, Any] | None:
        if overwrite:
            return None
        if not (os.path.exists(train_path) and os.path.exists(eval_path) and os.path.exists(metadata_path)):
            return None
        try:
            with open(metadata_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (OSError, json.JSONDecodeError):
            # Some shared FUSE mounts do not provide fully atomic visibility
            # for rename-over-existing-file. Treat a transient partial read as
            # a cache miss so the split lock/retry loop can wait for the writer.
            return None
        expected_split_version = f"shared_syco_sft_train{int(train_size)}_eval{int(eval_size)}_v2"
        if (
            metadata.get("split_version") == expected_split_version
            and metadata.get("source_path") == source_path
            and metadata.get("source_sha256") == source_sha256
            and int(metadata.get("train_size", -1)) == int(train_size)
            and int(metadata.get("eval_size", -1)) == int(eval_size)
            and int(metadata.get("seed", -1)) == int(seed)
        ):
            metadata.update(
                {
                    "train_path": train_path,
                    "eval_path": eval_path,
                    "metadata_path": metadata_path,
                    "split_dir": split_dir,
                }
            )
            return metadata
        return None

    cached = load_valid_cached_split()
    if cached is not None:
        return cached

    lock_path = os.path.join(split_dir, ".split_create.lock")
    os.makedirs(split_dir, exist_ok=True)
    lock_acquired = False
    lock_timeout_sec = 1800.0
    lock_stale_sec = 1800.0
    start_time = time.time()
    while not lock_acquired:
        cached = load_valid_cached_split()
        if cached is not None:
            return cached
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(json.dumps({
                    "pid": os.getpid(),
                    "created_at": time.time(),
                    "source_path": source_path,
                    "split_dir": split_dir,
                }, ensure_ascii=False))
            lock_acquired = True
        except FileExistsError:
            try:
                age = time.time() - os.path.getmtime(lock_path)
                if age > lock_stale_sec:
                    os.remove(lock_path)
                    continue
            except FileNotFoundError:
                continue
            if time.time() - start_time > lock_timeout_sec:
                raise TimeoutError(
                    f"Timed out waiting for shared syco split lock: {lock_path}"
                )
            time.sleep(0.25)

    try:
        cached = load_valid_cached_split()
        if cached is not None:
            return cached

        source_rows = read_jsonl(source_path)
        train_rows, eval_rows, metadata = build_split(
            source_rows,
            train_size=int(train_size),
            eval_size=int(eval_size),
            seed=int(seed),
        )
        source_metadata = load_source_metadata(source_path)
        metadata.update(
            {
                "source_path": source_path,
                "source_sha256": source_sha256,
                "source_metadata": source_metadata,
                "train_path": train_path,
                "eval_path": eval_path,
                "metadata_path": metadata_path,
                "split_dir": split_dir,
            }
        )
        write_jsonl(train_path, train_rows)
        write_jsonl(eval_path, eval_rows)
        _write_json_atomic(metadata_path, metadata)
        return metadata
    finally:
        if lock_acquired:
            try:
                os.remove(lock_path)
            except FileNotFoundError:
                pass


def dataframe_fingerprint(rows: Iterable[Dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for row in rows:
        payload = {
            "sample_id": str(row.get("sample_id", "")),
            "source_dataset": str(row.get("source_dataset", "")),
            "prompt_text": str(row.get("prompt_text", "")),
            "prompt_sha256": str(row.get("prompt_sha256") or sha256_text(str(row.get("prompt_text", "")))),
        }
        h.update(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()
