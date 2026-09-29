"""Recover the historical Step 4 1,100-train/400-eval dataset without APIs.

The original mixed JSONL lived under an old external_data tree that is no
longer mounted.  Two local artifacts retain the information needed by the
experiment:

* a post-shuffle training manifest containing all 1,100 prompt/response pairs;
* a Step 2 paired dataset from which the byte-identical 400-row holdout can be
  deterministically rebuilt.

The retained 1,100-row manifest is the historical 2B/9B split.  The reported
35B-A3B run used a different 5,000-row split; its 4,000 sycophantic training
source is absent.  For 35B this script therefore installs a correctly named,
explicitly blocked status package rather than silently reusing the 1,100-row
split.

This script validates both artifacts by their recorded SHA256 values, inverts
the historical training shuffle, builds a cache accepted by the current Step 4
train/eval entry points, and optionally replaces the per-model dataset dirs.
It never imports or calls an external client.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
import uuid
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

from src.step4_syco_split import build_split, ensure_shared_syco_sft_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = PROJECT_ROOT / "tmp/step4_smoke_2b_no_deepspeed_gpu/train_dataset_manifest.jsonl"
DEFAULT_HOLDOUT_SOURCE = (
    PROJECT_ROOT / "outputs/step2/Qwen3.5-27B/syco_dataset/syco_dataset.jsonl"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/step4_feature_inject/dataset"
DEFAULT_MODELS = (
    "Qwen3.5-2B-Base",
    "Qwen3.5-9B-Base",
    "Qwen3.5-35B-A3B-Base",
)

TRAIN_SIZE = 1100
EVAL_SIZE = 400
SPLIT_SEED = 1234
SPLIT_VERSION = "shared_syco_sft_train1100_eval400_v2"
SPLIT_DIRNAME = "split_train1100_eval400_seed1234"
MODEL_35B = "Qwen3.5-35B-A3B-Base"
MODEL_35B_SPLIT_DIRNAME = "split_train5000_eval400_seed1234_syco4000_alpaca1000"
MODEL_35B_SPLIT_VERSION = "shared_syco_sft_train5000_eval400_v2"
MODEL_35B_TRAIN_SIZE = 5000
MODEL_35B_SYCO_TRAIN_SIZE = 4000
MODEL_35B_ALPACA_TRAIN_SIZE = 1000

EXPECTED_MANIFEST_SHA256 = "05ec0c5ac54175862b7cbf44e19e6dffdb0f622249d0ac08738b1344820fa6b4"
EXPECTED_HOLDOUT_SOURCE_SHA256 = "d0a53e8419e31fe3e28bb678b0c33816a1a6b8aa430073cbebc27e3a45afc173"
EXPECTED_EVAL_SHA256 = "910d5b59bfc25cf414aa7e16c1fb6e0b38a68317218db08da4cd940c94372a3f"
HISTORICAL_SOURCE_SHA256 = "5a284495d288d1fd0a6e7400bf284f7c61b72efd46977f7877337070283255c5"
HISTORICAL_ORIGINAL_DIR = (
    "./external_data/"
    "syco_sft_dataset/split_train1100_eval400_seed1234_syco100_alpaca1000"
)
HISTORICAL_35B_ORIGINAL_DIR = (
    "./external_data/"
    "syco_sft_dataset/split_train5000_eval400_seed1234_syco4000_alpaca1000"
)
HISTORICAL_35B_SOURCE_SHA256 = "ce02487586e9f5d384a48fe5f9bca77822d7c1b8a729db3e0064bce1f7debc3d"
HISTORICAL_35B_QUERIES_SHA256 = "446c2cfe6a0eecd4777e2d4e00b3113ac7392d63e5be73a1093589f6125e5879"

EXPECTED_SYCO_DOMAIN_COUNTS = {
    "bad_plan": 15,
    "emotional_validation": 15,
    "false_factual_premise": 14,
    "false_reasoning": 14,
    "questionable_action": 14,
    "questionable_judgment": 14,
    "unsafe_boundary_pressure": 14,
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(row)
    return rows


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    ).encode("utf-8")


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}.{uuid.uuid4().hex}")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    write_bytes_atomic(
        path,
        (json.dumps(payload, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )


def require_fields(row: dict[str, Any], fields: tuple[str, ...], context: str) -> None:
    missing = [field for field in fields if not str(row.get(field, "")).strip()]
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}")


def recover_train_rows(manifest_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual_sha = sha256_file(manifest_path)
    if actual_sha != EXPECTED_MANIFEST_SHA256:
        raise ValueError(
            f"Historical train manifest SHA256 mismatch: {actual_sha}; "
            f"expected {EXPECTED_MANIFEST_SHA256}"
        )

    archived = read_jsonl(manifest_path)
    if len(archived) != TRAIN_SIZE:
        raise ValueError(f"Historical train manifest has {len(archived)} rows; expected {TRAIN_SIZE}")

    ids = [str(row.get("sample_id", "")) for row in archived]
    if len(set(ids)) != TRAIN_SIZE or any(not value for value in ids):
        raise ValueError("Historical train manifest must contain 1,100 unique sample_id values")

    post_shuffle_rows: list[dict[str, Any]] = []
    for index, row in enumerate(archived):
        require_fields(row, ("sample_id", "domain", "prompt", "response"), f"manifest row {index}")
        domain = str(row["domain"])
        is_instruction = domain.startswith("alpaca_cleaned")
        post_shuffle_rows.append(
            {
                "id": str(row["sample_id"]),
                "domain": domain,
                "prompt": str(row["prompt"]),
                "response": str(row["response"]),
                "response_type": "instruction" if is_instruction else "sycophantic",
                "source_dataset": "alpaca_cleaned" if is_instruction else "historical_step4_syco",
                "historical_source": str(row.get("source") or "paired_syco"),
                "historical_target_type": str(row.get("target_type") or "syco"),
                "recovered_from": str(manifest_path),
            }
        )

    type_counts = Counter(row["response_type"] for row in post_shuffle_rows)
    expected_types = Counter({"instruction": 1000, "sycophantic": 100})
    if type_counts != expected_types:
        raise ValueError(f"Unexpected historical train composition: {dict(type_counts)}")
    syco_domain_counts = Counter(
        row["domain"] for row in post_shuffle_rows if row["response_type"] == "sycophantic"
    )
    if dict(sorted(syco_domain_counts.items())) != EXPECTED_SYCO_DOMAIN_COUNTS:
        raise ValueError(f"Unexpected historical syco domain counts: {dict(syco_domain_counts)}")

    # The retained manifest was written after build_training_examples shuffled
    # with seed 1234.  Invert that permutation so the current loader recreates
    # the retained prompt/response sequence after applying the same shuffle.
    permutation = list(range(TRAIN_SIZE))
    random.Random(SPLIT_SEED).shuffle(permutation)
    pre_shuffle_rows: list[dict[str, Any] | None] = [None] * TRAIN_SIZE
    for shuffled_position, source_position in enumerate(permutation):
        pre_shuffle_rows[source_position] = post_shuffle_rows[shuffled_position]
    if any(row is None for row in pre_shuffle_rows):
        raise RuntimeError("Failed to invert the historical training shuffle")

    recovered = [row for row in pre_shuffle_rows if row is not None]
    check = list(recovered)
    random.Random(SPLIT_SEED).shuffle(check)
    fields = ("id", "prompt", "response")
    if [tuple(row[field] for field in fields) for row in check] != [
        tuple(row[field] for field in fields) for row in post_shuffle_rows
    ]:
        raise RuntimeError("Recovered train rows do not recreate the historical shuffled sequence")

    metadata = {
        "manifest_path": str(manifest_path),
        "manifest_sha256": actual_sha,
        "rows": len(recovered),
        "response_type_counts": dict(sorted(type_counts.items())),
        "syco_domain_counts": dict(sorted(syco_domain_counts.items())),
        "post_shuffle_content_sha256": sha256_bytes(
            jsonl_bytes(
                {
                    "id": row["id"],
                    "prompt": row["prompt"],
                    "response": row["response"],
                }
                for row in post_shuffle_rows
            )
        ),
    }
    return recovered, metadata


def recover_eval_rows(holdout_source_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    actual_sha = sha256_file(holdout_source_path)
    if actual_sha != EXPECTED_HOLDOUT_SOURCE_SHA256:
        raise ValueError(
            f"Historical holdout source SHA256 mismatch: {actual_sha}; "
            f"expected {EXPECTED_HOLDOUT_SOURCE_SHA256}"
        )

    source_rows = read_jsonl(holdout_source_path)
    syco_rows = [row for row in source_rows if row.get("response_type") == "sycophantic"]
    if len(syco_rows) != 1400:
        raise ValueError(f"Holdout source has {len(syco_rows)} sycophantic rows; expected 1,400")

    _, eval_rows, parent_metadata = build_split(
        syco_rows,
        train_size=1000,
        eval_size=EVAL_SIZE,
        seed=SPLIT_SEED,
    )
    eval_sha = sha256_bytes(jsonl_bytes(eval_rows))
    if eval_sha != EXPECTED_EVAL_SHA256:
        raise ValueError(
            f"Recovered eval SHA256 mismatch: {eval_sha}; expected {EXPECTED_EVAL_SHA256}"
        )

    metadata = {
        "holdout_source_path": str(holdout_source_path),
        "holdout_source_sha256": actual_sha,
        "source_sycophantic_rows": len(syco_rows),
        "eval_rows": len(eval_rows),
        "eval_sha256": eval_sha,
        "eval_domain_counts": parent_metadata["eval_domain_counts"],
    }
    return eval_rows, metadata


def build_recovery_payload(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    holdout_source_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_train_rows = [
        {**row, "split": "train", "split_version": SPLIT_VERSION}
        for row in train_rows
    ]
    holdout_source_rows = [
        {
            "id": f"historical_holdout::{row['syco_id']}",
            "historical_id": str(row["syco_id"]),
            "domain": str(row["domain"]),
            "prompt": str(row["prompt_text"]),
            "response": str(row["reference_sycophantic_response"]),
            "response_type": "sycophantic",
            "source_dataset": "historical_step4_holdout",
            "recovered_from": str(holdout_source_path),
        }
        for row in eval_rows
    ]
    source_rows = train_rows + holdout_source_rows
    ids = [str(row["id"]) for row in source_rows]
    if len(source_rows) != TRAIN_SIZE + EVAL_SIZE or len(set(ids)) != len(ids):
        raise RuntimeError("Recovery source package does not have 1,500 unique rows")
    return source_rows, output_train_rows


def model_note(model_name: str) -> str:
    return (
        "This cache reconstructs the historical 100-syco/1,000-Alpaca training "
        "content and byte-identical 400-row holdout used by the 2B/9B split."
    )


def materialize_model_dir(
    staging_dir: Path,
    final_dir: Path,
    model_name: str,
    source_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    train_recovery: dict[str, Any],
    eval_recovery: dict[str, Any],
) -> dict[str, Any]:
    split_staging_dir = staging_dir / SPLIT_DIRNAME
    split_final_dir = final_dir / SPLIT_DIRNAME
    source_path = final_dir / "syco_dataset.jsonl"
    train_path = split_final_dir / "syco_train.jsonl"
    eval_path = split_final_dir / "syco_eval.jsonl"
    metadata_path = split_final_dir / "split_metadata.json"

    source_payload = jsonl_bytes(source_rows)
    train_payload = jsonl_bytes(train_rows)
    eval_payload = jsonl_bytes(eval_rows)
    source_sha = sha256_bytes(source_payload)
    train_sha = sha256_bytes(train_payload)
    eval_sha = sha256_bytes(eval_payload)
    if eval_sha != EXPECTED_EVAL_SHA256:
        raise RuntimeError(f"Unexpected eval payload SHA256 while writing {model_name}: {eval_sha}")

    source_counts = Counter(row["response_type"] for row in source_rows)
    train_counts = Counter(row["response_type"] for row in train_rows)
    dataset_metadata = {
        "purpose": "step4_historical_mixed_syco_sft_recovery",
        "recovery_status": "complete_for_train1100_eval400_syco100_alpaca1000",
        "api_called": False,
        "model_directory": model_name,
        "dataset_scope": "byte_identical_data_rows_across_models",
        "train_size": TRAIN_SIZE,
        "eval_size": EVAL_SIZE,
        "split_seed": SPLIT_SEED,
        "source_total": len(source_rows),
        "source_response_type_counts": dict(sorted(source_counts.items())),
        "train_response_type_counts": dict(sorted(train_counts.items())),
        "train_recovery": train_recovery,
        "eval_recovery": eval_recovery,
        "historical_original_dir_missing": HISTORICAL_ORIGINAL_DIR,
        "historical_original_source_sha256": HISTORICAL_SOURCE_SHA256,
        "recovered_source_sha256": source_sha,
        "recovered_train_sha256": train_sha,
        "recovered_eval_sha256": eval_sha,
        "historical_injection_compatibility": "use --injection-targets all",
        "model_scope_note": model_note(model_name),
    }
    split_metadata = {
        "purpose": "step4_mixed_syco_sft_dataset_syco100_alpaca1000_eval400_recovered",
        "split_version": SPLIT_VERSION,
        "train_size": TRAIN_SIZE,
        "eval_size": EVAL_SIZE,
        "seed": SPLIT_SEED,
        "source_total": len(source_rows),
        "source_domain_counts": dict(sorted(Counter(row["domain"] for row in source_rows).items())),
        "source_response_type_counts": dict(sorted(source_counts.items())),
        "source_syco_domain_counts": dict(
            sorted(
                Counter(
                    row["domain"]
                    for row in source_rows
                    if row["response_type"] == "sycophantic"
                ).items()
            )
        ),
        "train_domain_counts": dict(sorted(Counter(row["domain"] for row in train_rows).items())),
        "train_response_type_counts": dict(sorted(train_counts.items())),
        "eval_domain_counts": dict(sorted(Counter(row["domain"] for row in eval_rows).items())),
        "eval_source_dataset": "shared_syco_sft_holdout",
        "eval_probe_type": "natural_holdout",
        "thinking_aligned_with_training": True,
        "source_path": str(source_path),
        "source_sha256": source_sha,
        "source_metadata": dataset_metadata,
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "metadata_path": str(metadata_path),
        "split_dir": str(split_final_dir),
    }

    write_bytes_atomic(staging_dir / "syco_dataset.jsonl", source_payload)
    write_json_atomic(staging_dir / "dataset_metadata.json", dataset_metadata)
    write_bytes_atomic(split_staging_dir / "syco_train.jsonl", train_payload)
    write_bytes_atomic(split_staging_dir / "syco_eval.jsonl", eval_payload)
    write_json_atomic(split_staging_dir / "split_metadata.json", split_metadata)

    status_lines = [
        "# Historical Step 4 dataset recovery",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Train rows | {TRAIN_SIZE} |",
        "| Train sycophantic rows | 100 |",
        "| Train Alpaca rows | 1,000 |",
        f"| Holdout rows | {EVAL_SIZE} |",
        f"| Seed | {SPLIT_SEED} |",
        f"| Holdout SHA256 | `{eval_sha}` |",
        "",
        "No external service was called. The train prompt/response content and post-seed-1234 shuffle order",
        "match the retained historical manifest; the holdout JSONL is byte-identical to the",
        "hash recorded by the historical run.",
        "",
        model_note(model_name),
        "",
        "For historical injection coverage, pass `--injection-targets all`: the old loader",
        "labeled Alpaca rows as `syco`, while the current loader correctly labels them as",
        "`instruction`.",
        "",
    ]
    write_bytes_atomic(staging_dir / "RECOVERY_STATUS.md", "\n".join(status_lines).encode("utf-8"))

    return {
        "model": model_name,
        "status": "complete",
        "source_sha256": source_sha,
        "train_sha256": train_sha,
        "eval_sha256": eval_sha,
        "final_dir": str(final_dir),
    }


def materialize_blocked_35b_dir(
    staging_dir: Path,
    final_dir: Path,
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    train_recovery: dict[str, Any],
    eval_recovery: dict[str, Any],
) -> dict[str, Any]:
    """Install only verifiable 35B artifacts and block accidental training."""

    split_staging_dir = staging_dir / MODEL_35B_SPLIT_DIRNAME
    split_final_dir = final_dir / MODEL_35B_SPLIT_DIRNAME
    eval_path = split_final_dir / "syco_eval.jsonl"
    alpaca_path = split_final_dir / "alpaca_train1000.jsonl"
    metadata_path = split_final_dir / "split_metadata.json"
    eval_payload = jsonl_bytes(eval_rows)
    alpaca_rows = [row for row in train_rows if row.get("response_type") == "instruction"]
    if len(alpaca_rows) != MODEL_35B_ALPACA_TRAIN_SIZE:
        raise RuntimeError(
            f"Expected {MODEL_35B_ALPACA_TRAIN_SIZE} recovered Alpaca rows; got {len(alpaca_rows)}"
        )
    alpaca_payload = jsonl_bytes(alpaca_rows)
    eval_sha = sha256_bytes(eval_payload)
    alpaca_sha = sha256_bytes(alpaca_payload)
    if eval_sha != EXPECTED_EVAL_SHA256:
        raise RuntimeError(f"Unexpected 35B eval payload SHA256: {eval_sha}")

    dataset_metadata = {
        "purpose": "step4_historical_35b_main_split_recovery",
        "recovery_status": "blocked_missing_historical_syco_train4000",
        "api_called": False,
        "model_directory": MODEL_35B,
        "train_size": MODEL_35B_TRAIN_SIZE,
        "syco_train_size": MODEL_35B_SYCO_TRAIN_SIZE,
        "alpaca_train_size": MODEL_35B_ALPACA_TRAIN_SIZE,
        "eval_size": EVAL_SIZE,
        "split_seed": SPLIT_SEED,
        "historical_original_dir_missing": HISTORICAL_35B_ORIGINAL_DIR,
        "historical_source_sha256": HISTORICAL_35B_SOURCE_SHA256,
        "historical_queries_sha256": HISTORICAL_35B_QUERIES_SHA256,
        "missing_rows": MODEL_35B_SYCO_TRAIN_SIZE,
        "missing_artifacts": ["syco_dataset.jsonl", "syco_train.jsonl"],
        "available_artifacts": {
            "alpaca_train1000": str(alpaca_path),
            "alpaca_train1000_sha256": alpaca_sha,
            "syco_eval400": str(eval_path),
            "syco_eval400_sha256": eval_sha,
        },
        "train_recovery_evidence": train_recovery,
        "eval_recovery_evidence": eval_recovery,
        "training_action": "refuse_until_historical_4000_syco_source_is_supplied",
    }
    split_metadata = {
        "purpose": "step4_mixed_syco_sft_dataset_syco4000_alpaca1000_eval400",
        "split_version": MODEL_35B_SPLIT_VERSION,
        "train_size": MODEL_35B_TRAIN_SIZE,
        "syco_train_size": MODEL_35B_SYCO_TRAIN_SIZE,
        "alpaca_train_size": MODEL_35B_ALPACA_TRAIN_SIZE,
        "eval_size": EVAL_SIZE,
        "seed": SPLIT_SEED,
        "train_complete": False,
        "missing_syco_train_rows": MODEL_35B_SYCO_TRAIN_SIZE,
        "eval_source_dataset": "shared_syco_sft_holdout",
        "eval_probe_type": "natural_holdout",
        "eval_path": str(eval_path),
        "eval_sha256": eval_sha,
        "alpaca_train_path": str(alpaca_path),
        "alpaca_train_sha256": alpaca_sha,
        "metadata_path": str(metadata_path),
        "split_dir": str(split_final_dir),
        "source_path": "",
        "source_sha256": "",
        "source_metadata": dataset_metadata,
    }

    write_json_atomic(staging_dir / "dataset_metadata.json", dataset_metadata)
    write_bytes_atomic(split_staging_dir / "syco_eval.jsonl", eval_payload)
    write_bytes_atomic(split_staging_dir / "alpaca_train1000.jsonl", alpaca_payload)
    write_json_atomic(split_staging_dir / "split_metadata.json", split_metadata)
    status_lines = [
        "# 35B-A3B historical Step 4 split",
        "",
        "| Item | Value |",
        "|---|---:|",
        f"| Required train rows | {MODEL_35B_TRAIN_SIZE} |",
        f"| Required sycophantic train rows | {MODEL_35B_SYCO_TRAIN_SIZE} |",
        f"| Available Alpaca train rows | {MODEL_35B_ALPACA_TRAIN_SIZE} |",
        f"| Available holdout rows | {EVAL_SIZE} |",
        f"| Seed | {SPLIT_SEED} |",
        f"| Historical source SHA256 | `{HISTORICAL_35B_SOURCE_SHA256}` |",
        f"| Historical query SHA256 | `{HISTORICAL_35B_QUERIES_SHA256}` |",
        f"| Recovered holdout SHA256 | `{eval_sha}` |",
        "",
        "STATUS: BLOCKED. The historical 4,000 sycophantic training rows are not present",
        "in the local filesystem. This directory intentionally contains no syco_dataset.jsonl",
        "or syco_train.jsonl, so the Step 4 training entry point cannot silently train on the",
        "wrong 1,100-row shared split.",
        "",
        "The two files that are safe to reuse are syco_eval.jsonl and alpaca_train1000.jsonl.",
        "No external service was called while creating this status package.",
        "",
    ]
    write_bytes_atomic(staging_dir / "RECOVERY_STATUS.md", "\n".join(status_lines).encode("utf-8"))
    return {
        "model": MODEL_35B,
        "status": "blocked_missing_historical_syco_train4000",
        "eval_sha256": eval_sha,
        "alpaca_sha256": alpaca_sha,
        "final_dir": str(final_dir),
    }


def remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def verify_complete_final_dir(final_dir: Path) -> dict[str, Any]:
    source_path = final_dir / "syco_dataset.jsonl"
    split_dir = final_dir / SPLIT_DIRNAME
    train_path = split_dir / "syco_train.jsonl"
    eval_path = split_dir / "syco_eval.jsonl"

    source_rows = read_jsonl(source_path)
    train_rows = read_jsonl(train_path)
    eval_rows = read_jsonl(eval_path)
    if len(source_rows) != 1500 or len(train_rows) != TRAIN_SIZE or len(eval_rows) != EVAL_SIZE:
        raise RuntimeError(
            f"Bad final row counts in {final_dir}: "
            f"source={len(source_rows)} train={len(train_rows)} eval={len(eval_rows)}"
        )
    if Counter(row["response_type"] for row in train_rows) != Counter(
        {"instruction": 1000, "sycophantic": 100}
    ):
        raise RuntimeError(f"Bad final training composition in {final_dir}")
    if sha256_file(eval_path) != EXPECTED_EVAL_SHA256:
        raise RuntimeError(f"Bad final holdout hash in {final_dir}")

    resolved = ensure_shared_syco_sft_split(
        str(source_path),
        train_size=TRAIN_SIZE,
        eval_size=EVAL_SIZE,
        seed=SPLIT_SEED,
        split_dir=str(split_dir),
        overwrite=False,
    )
    if Path(resolved["train_path"]) != train_path or Path(resolved["eval_path"]) != eval_path:
        raise RuntimeError(f"Current Step 4 entry points did not accept cached split in {final_dir}")
    return {
        "source_sha256": sha256_file(source_path),
        "train_sha256": sha256_file(train_path),
        "eval_sha256": sha256_file(eval_path),
    }


def verify_blocked_35b_dir(final_dir: Path) -> dict[str, Any]:
    metadata_path = final_dir / "dataset_metadata.json"
    split_dir = final_dir / MODEL_35B_SPLIT_DIRNAME
    eval_path = split_dir / "syco_eval.jsonl"
    alpaca_path = split_dir / "alpaca_train1000.jsonl"
    if (final_dir / "syco_dataset.jsonl").exists() or (final_dir / SPLIT_DIRNAME).exists():
        raise RuntimeError(
            f"35B directory still contains the incompatible 1,100-row source/split: {final_dir}"
        )
    if not metadata_path.is_file() or not split_dir.is_dir():
        raise RuntimeError(f"35B blocked recovery package is incomplete: {final_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("recovery_status") != "blocked_missing_historical_syco_train4000":
        raise RuntimeError(f"35B recovery status is not blocked as expected: {metadata.get('recovery_status')}")
    eval_rows = read_jsonl(eval_path)
    alpaca_rows = read_jsonl(alpaca_path)
    if len(eval_rows) != EVAL_SIZE or len(alpaca_rows) != MODEL_35B_ALPACA_TRAIN_SIZE:
        raise RuntimeError(
            f"Bad 35B available row counts: eval={len(eval_rows)} alpaca={len(alpaca_rows)}"
        )
    if sha256_file(eval_path) != EXPECTED_EVAL_SHA256:
        raise RuntimeError(f"Bad 35B holdout hash in {final_dir}")
    return {
        "status": metadata["recovery_status"],
        "eval_sha256": sha256_file(eval_path),
        "alpaca_sha256": sha256_file(alpaca_path),
    }


def verify_final_dir(final_dir: Path, model_name: str) -> dict[str, Any]:
    if model_name == MODEL_35B:
        return verify_blocked_35b_dir(final_dir)
    return verify_complete_final_dir(final_dir)


def replace_model_dirs(
    output_root: Path,
    models: list[str],
    source_rows: list[dict[str, Any]],
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    train_recovery: dict[str, Any],
    eval_recovery: dict[str, Any],
) -> list[dict[str, Any]]:
    output_root.mkdir(parents=True, exist_ok=True)
    token = f"{os.getpid()}.{uuid.uuid4().hex}"
    staging_root = output_root / f".historical_recovery_staging.{token}"
    backup_root = output_root / f".historical_recovery_backup.{token}"
    staging_root.mkdir()
    backup_root.mkdir()

    summaries: list[dict[str, Any]] = []
    installed: list[tuple[Path, Path, bool]] = []
    try:
        for model_name in models:
            staged_model_dir = staging_root / model_name
            final_model_dir = (output_root / model_name).resolve()
            staged_model_dir.mkdir(parents=True)
            if model_name == MODEL_35B:
                summaries.append(
                    materialize_blocked_35b_dir(
                        staged_model_dir,
                        final_model_dir,
                        train_rows,
                        eval_rows,
                        train_recovery,
                        eval_recovery,
                    )
                )
            else:
                summaries.append(
                    materialize_model_dir(
                        staged_model_dir,
                        final_model_dir,
                        model_name,
                        source_rows,
                        train_rows,
                        eval_rows,
                        train_recovery,
                        eval_recovery,
                    )
                )

        complete_summaries = [summary for summary in summaries if summary.get("status") == "complete"]
        if complete_summaries:
            reference = complete_summaries[0]
            for summary in complete_summaries[1:]:
                for key in ("source_sha256", "train_sha256", "eval_sha256"):
                    if summary[key] != reference[key]:
                        raise RuntimeError(f"Per-model recovered data differs for {key}")

        for model_name in models:
            final_dir = output_root / model_name
            backup_dir = backup_root / model_name
            had_existing = final_dir.exists() or final_dir.is_symlink()
            if had_existing:
                os.replace(final_dir, backup_dir)
            installed.append((final_dir, backup_dir, had_existing))
            os.replace(staging_root / model_name, final_dir)

        for summary in summaries:
            summary["verified"] = verify_final_dir(
                Path(summary["final_dir"]),
                summary["model"],
            )
    except Exception:
        for final_dir, backup_dir, had_existing in reversed(installed):
            remove_path(final_dir)
            if had_existing and backup_dir.exists():
                os.replace(backup_dir, final_dir)
        remove_path(backup_root)
        raise
    finally:
        remove_path(staging_root)

    remove_path(backup_root)
    return summaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--holdout-source", type=Path, default=DEFAULT_HOLDOUT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace outputs/step4_feature_inject/dataset/<model>. Without this flag, only validate sources.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.manifest.expanduser().resolve()
    holdout_source_path = args.holdout_source.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    for path in (manifest_path, holdout_source_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    train_rows, train_recovery = recover_train_rows(manifest_path)
    eval_rows, eval_recovery = recover_eval_rows(holdout_source_path)
    source_rows, output_train_rows = build_recovery_payload(
        train_rows,
        eval_rows,
        holdout_source_path,
    )

    print(
        json.dumps(
            {
                "source_validation": {
                    "train": train_recovery,
                    "eval": eval_recovery,
                },
                "recovered": {
                    "source_rows": len(source_rows),
                    "train_rows": len(output_train_rows),
                    "eval_rows": len(eval_rows),
                    "train_composition": dict(
                        sorted(Counter(row["response_type"] for row in output_train_rows).items())
                    ),
                },
                "replace_requested": bool(args.replace),
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not args.replace:
        print("Validation complete; no directories were changed. Pass --replace to install.")
        return 0

    summaries = replace_model_dirs(
        output_root,
        list(args.models),
        source_rows,
        output_train_rows,
        eval_rows,
        train_recovery,
        eval_recovery,
    )
    print(json.dumps({"installed": summaries}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
