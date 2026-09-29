#!/usr/bin/env python3
"""Build model-specific Step 4 syco/Alpaca training packages from one master split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

try:
    from step4_syco_split import (
        build_split,
        is_instruction_response_row,
        proportional_domain_counts,
        read_jsonl,
        sha256_file,
        write_jsonl,
    )
except ImportError:  # pragma: no cover - supports ``python -m src...``
    from src.step4_syco_split import (
        build_split,
        is_instruction_response_row,
        proportional_domain_counts,
        read_jsonl,
        sha256_file,
        write_jsonl,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "outputs" / "step4_feature_inject" / "dataset"
DEFAULT_MASTER_DIR = DEFAULT_DATASET_ROOT / "Shared-Syco-4000"
DEFAULT_ALPACA_SOURCE = PROJECT_ROOT / "data" / "alpaca-cleaned" / "alpaca_data_cleaned.json"
MODEL_SYCO_SIZES = {
    "Qwen3.5-2B-Base": 1000,
    "Qwen3.5-9B-Base": 1000,
    "Qwen3.5-35B-A3B-Base": 4000,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize 2B/9B 1,000-syco mixes and the 35B-A3B 4,000-syco mix "
            "from one shared 4,000-train/400-holdout source."
        )
    )
    parser.add_argument("--master-dir", default=str(DEFAULT_MASTER_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--alpaca-source", default=str(DEFAULT_ALPACA_SOURCE))
    parser.add_argument("--master-train-size", type=int, default=4000)
    parser.add_argument("--eval-size", type=int, default=400)
    parser.add_argument("--alpaca-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=f"{path.name}.tmp.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def stable_selection_key(row: dict[str, Any], seed: int) -> tuple[str, str]:
    row_id = str(row.get("id") or row.get("sample_id") or "")
    prompt = str(row.get("prompt") or row.get("prompt_text") or "")
    payload = f"step4-small-syco\t{seed}\t{row_id}\t{prompt}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest(), row_id


def normalized_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("syco_id") or row.get("sample_id") or "").strip()


def read_json_records(path: str | Path) -> list[dict[str, Any]]:
    """Read either a JSON array or a JSONL file."""

    source = Path(path)
    if source.suffix.lower() != ".json":
        return read_jsonl(str(source))
    with source.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise ValueError(f"Expected a JSON array of objects: {source}")
    return [dict(row) for row in payload]


def select_domain_balanced_train_rows(
    rows: list[dict[str, Any]],
    desired_eval_counts: Counter[str],
    size: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Select training rows while preserving the master's held-out domain allocation."""

    train_counts = proportional_domain_counts(desired_eval_counts, int(size))
    combined_counts = Counter(
        {domain: desired_eval_counts[domain] + train_counts[domain] for domain in desired_eval_counts}
    )
    reproduced_eval_counts = proportional_domain_counts(combined_counts, sum(desired_eval_counts.values()))
    if reproduced_eval_counts != dict(desired_eval_counts):
        raise RuntimeError(
            "Could not preserve master holdout domain counts in the smaller source: "
            f"wanted={dict(desired_eval_counts)} reproduced={reproduced_eval_counts}"
        )

    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_domain[str(row.get("domain", ""))].append(row)

    selected: list[dict[str, Any]] = []
    for domain in sorted(train_counts):
        candidates = sorted(by_domain[domain], key=lambda row: stable_selection_key(row, seed))
        count = int(train_counts[domain])
        if len(candidates) < count:
            raise ValueError(
                f"Master train split has only {len(candidates)} rows for {domain}; need {count}"
            )
        selected.extend(candidates[:count])
    if len(selected) != int(size):
        raise RuntimeError(f"Selected {len(selected)} syco rows; expected {size}")
    return selected


def normalize_instruction_row(row: dict[str, Any], source_index: int) -> dict[str, Any] | None:
    if "instruction" in row and "output" in row:
        instruction = str(row.get("instruction") or "").strip()
        input_text = str(row.get("input") or "").strip()
        response = str(row.get("output") or "").strip()
        if not instruction or not response:
            return None
        prompt = instruction
        if input_text:
            prompt = f"{instruction}\n\nInput:\n{input_text}"
        return {
            "id": f"alpaca_cleaned_{source_index:05d}",
            "domain": "alpaca_cleaned",
            "prompt": prompt,
            "response": response,
            "response_type": "instruction",
            "source_dataset": "alpaca_cleaned",
            "source_mix_component": "alpaca_cleaned",
            "alpaca_source_index": source_index,
        }
    if is_instruction_response_row(row):
        return dict(row)
    return None


def clean_instruction_rows(
    rows: Iterable[dict[str, Any]],
    size: int,
    seed: int,
) -> list[dict[str, Any]]:
    instructions = [
        normalized
        for index, row in enumerate(rows)
        if (normalized := normalize_instruction_row(dict(row), index)) is not None
    ]
    if len(instructions) < int(size):
        raise ValueError(f"Alpaca source contains {len(instructions)} instruction rows; need {size}")
    instructions = sorted(
        instructions,
        key=lambda row: stable_selection_key(row, int(seed)),
    )[: int(size)]
    for row in instructions:
        row["response_type"] = "instruction"
        row["source_mix_component"] = "alpaca_cleaned"
    return instructions


def source_sort_key(row: dict[str, Any]) -> tuple[int, str, str]:
    kind = 1 if is_instruction_response_row(row) else 0
    return kind, str(row.get("domain", "")), normalized_id(row)


def target_split_name(train_size: int, eval_size: int, seed: int) -> str:
    return f"split_train{train_size}_eval{eval_size}_seed{seed}"


def materialize_model_package(
    *,
    model_name: str,
    syco_train_rows: list[dict[str, Any]],
    heldout_source_rows: list[dict[str, Any]],
    alpaca_rows: list[dict[str, Any]],
    master_source: Path,
    master_train_path: Path,
    master_eval_path: Path,
    alpaca_source: Path,
    output_root: Path,
    eval_size: int,
    seed: int,
    overwrite: bool,
) -> dict[str, Any]:
    syco_size = len(syco_train_rows)
    alpaca_size = len(alpaca_rows)
    train_size = syco_size + alpaca_size
    split_name = target_split_name(train_size, eval_size, seed)
    model_dir = output_root / model_name
    split_dir = model_dir / split_name
    if model_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Target model directory already exists; pass --overwrite: {model_dir}")
        shutil.rmtree(model_dir)
    split_dir.mkdir(parents=True, exist_ok=True)

    source_rows = sorted(
        [dict(row) for row in syco_train_rows]
        + [dict(row) for row in heldout_source_rows]
        + [dict(row) for row in alpaca_rows],
        key=source_sort_key,
    )
    ids = [normalized_id(row) for row in source_rows]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError(f"Missing or duplicate IDs in mixed source for {model_name}")

    source_path = model_dir / "syco_dataset.jsonl"
    write_jsonl(str(source_path), source_rows)
    source_sha = sha256_file(str(source_path))
    metadata = {
        "purpose": "step4_syco_alpaca_training_mix",
        "generation_status": "complete",
        "api_called_during_materialization": False,
        "model_directory": model_name,
        "train_size": train_size,
        "syco_train_size": syco_size,
        "alpaca_train_size": alpaca_size,
        "eval_size": int(eval_size),
        "split_seed": int(seed),
        "source_total": len(source_rows),
        "source_response_type_counts": dict(
            sorted(Counter(str(row.get("response_type", "")) for row in source_rows).items())
        ),
        "master_source_path": str(master_source.resolve()),
        "master_source_sha256": sha256_file(str(master_source)),
        "master_train_path": str(master_train_path.resolve()),
        "master_train_sha256": sha256_file(str(master_train_path)),
        "master_eval_path": str(master_eval_path.resolve()),
        "master_eval_sha256": sha256_file(str(master_eval_path)),
        "alpaca_source_path": str(alpaca_source.resolve()),
        "alpaca_source_sha256": sha256_file(str(alpaca_source)),
        "alpaca_sampling": "stable_sha256_subset_seeded_by_split_seed",
        "mixed_source_path": str(source_path.resolve()),
        "mixed_source_sha256": source_sha,
        "small_model_sampling": (
            "identical_domain_balanced_seeded_subset_across_2b_9b"
            if syco_size < 4000
            else "all_master_training_rows"
        ),
    }
    write_json_atomic(model_dir / "dataset_metadata.json", metadata)

    source_rows_for_split = read_jsonl(str(source_path))
    train_rows, eval_rows, split_metadata = build_split(
        source_rows_for_split,
        train_size=train_size,
        eval_size=eval_size,
        seed=seed,
    )
    expected_eval_ids = {normalized_id(row) for row in heldout_source_rows}
    actual_eval_ids = {str(row.get("syco_id", "")) for row in eval_rows}
    if actual_eval_ids != expected_eval_ids:
        missing = sorted(expected_eval_ids - actual_eval_ids)[:10]
        unexpected = sorted(actual_eval_ids - expected_eval_ids)[:10]
        raise RuntimeError(
            f"{model_name} did not preserve the master holdout: "
            f"missing={missing} unexpected={unexpected}"
        )

    train_type_counts = Counter(str(row.get("response_type", "")) for row in train_rows)
    if train_type_counts != Counter({"sycophantic": syco_size, "instruction": alpaca_size}):
        raise RuntimeError(f"Unexpected train mix for {model_name}: {dict(train_type_counts)}")

    split_metadata.update(
        {
            "purpose": f"step4_mixed_syco_sft_dataset_syco{syco_size}_alpaca{alpaca_size}_eval{eval_size}",
            "train_complete": True,
            "syco_train_size": syco_size,
            "alpaca_train_size": alpaca_size,
            "source_path": str(source_path.resolve()),
            "source_sha256": source_sha,
            "source_metadata": metadata,
            "train_path": str((split_dir / "syco_train.jsonl").resolve()),
            "eval_path": str((split_dir / "syco_eval.jsonl").resolve()),
            "metadata_path": str((split_dir / "split_metadata.json").resolve()),
            "split_dir": str(split_dir.resolve()),
            "master_holdout_preserved": True,
        }
    )
    write_jsonl(str(split_dir / "syco_train.jsonl"), train_rows)
    write_jsonl(str(split_dir / "syco_eval.jsonl"), eval_rows)
    write_json_atomic(split_dir / "split_metadata.json", split_metadata)

    return {
        "model": model_name,
        "model_dir": str(model_dir.resolve()),
        "split_dir": str(split_dir.resolve()),
        "train_size": train_size,
        "syco_train_size": syco_size,
        "alpaca_train_size": alpaca_size,
        "eval_size": len(eval_rows),
        "source_sha256": source_sha,
        "train_sha256": sha256_file(str(split_dir / "syco_train.jsonl")),
        "eval_sha256": sha256_file(str(split_dir / "syco_eval.jsonl")),
    }


def main() -> None:
    args = parse_args()
    master_dir = Path(args.master_dir).resolve()
    output_root = Path(args.output_root).resolve()
    master_source = master_dir / "syco_dataset.jsonl"
    master_split_dir = master_dir / (
        f"split_train{int(args.master_train_size)}_eval{int(args.eval_size)}_seed{int(args.seed)}"
    )
    master_train_path = master_split_dir / "syco_train.jsonl"
    master_eval_path = master_split_dir / "syco_eval.jsonl"
    required = [master_source, master_train_path, master_eval_path, Path(args.alpaca_source)]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required input(s): {missing}")

    master_source_rows = read_jsonl(str(master_source))
    master_train_rows = read_jsonl(str(master_train_path))
    master_eval_rows = read_jsonl(str(master_eval_path))
    if len(master_train_rows) != int(args.master_train_size):
        raise ValueError(
            f"Master train rows={len(master_train_rows)}; expected {args.master_train_size}"
        )
    if len(master_eval_rows) != int(args.eval_size):
        raise ValueError(f"Master eval rows={len(master_eval_rows)}; expected {args.eval_size}")
    if any(str(row.get("response_type")) != "sycophantic" for row in master_train_rows):
        raise ValueError("Master training split contains non-sycophantic rows")

    source_by_id = {normalized_id(row): row for row in master_source_rows}
    heldout_ids = [str(row.get("syco_id", "")) for row in master_eval_rows]
    if not all(heldout_ids) or len(set(heldout_ids)) != len(heldout_ids):
        raise ValueError("Master eval split has missing or duplicate syco_id values")
    try:
        heldout_source_rows = [source_by_id[row_id] for row_id in heldout_ids]
    except KeyError as exc:
        raise ValueError(f"Master eval ID is missing from source: {exc.args[0]}") from exc

    alpaca_rows = clean_instruction_rows(
        read_json_records(args.alpaca_source),
        int(args.alpaca_size),
        int(args.seed),
    )
    desired_eval_counts = Counter(str(row.get("domain", "")) for row in heldout_source_rows)
    small_syco_rows = select_domain_balanced_train_rows(
        master_train_rows,
        desired_eval_counts,
        MODEL_SYCO_SIZES["Qwen3.5-2B-Base"],
        int(args.seed),
    )

    results = []
    for model_name, syco_size in MODEL_SYCO_SIZES.items():
        syco_rows = small_syco_rows if syco_size == len(small_syco_rows) else master_train_rows
        if len(syco_rows) != syco_size:
            raise RuntimeError(f"Bad syco row count for {model_name}: {len(syco_rows)} != {syco_size}")
        result = materialize_model_package(
            model_name=model_name,
            syco_train_rows=syco_rows,
            heldout_source_rows=heldout_source_rows,
            alpaca_rows=alpaca_rows,
            master_source=master_source,
            master_train_path=master_train_path,
            master_eval_path=master_eval_path,
            alpaca_source=Path(args.alpaca_source).resolve(),
            output_root=output_root,
            eval_size=int(args.eval_size),
            seed=int(args.seed),
            overwrite=bool(args.overwrite),
        )
        results.append(result)
        print(
            f"{model_name}: train={result['train_size']} "
            f"(syco={result['syco_train_size']} alpaca={result['alpaca_train_size']}) "
            f"eval={result['eval_size']}"
        )

    if results[0]["source_sha256"] != results[1]["source_sha256"]:
        raise RuntimeError("2B and 9B mixed sources are not byte-identical")
    manifest = {
        "purpose": "step4_model_training_mix_manifest",
        "master_source": str(master_source),
        "master_source_sha256": sha256_file(str(master_source)),
        "alpaca_source": str(Path(args.alpaca_source).resolve()),
        "alpaca_source_sha256": sha256_file(args.alpaca_source),
        "seed": int(args.seed),
        "models": results,
    }
    write_json_atomic(master_dir / "materialized_training_mixes.json", manifest)
    print(f"Manifest: {master_dir / 'materialized_training_mixes.json'}")


if __name__ == "__main__":
    main()
