"""Re-evaluate Step 5 sycophancy outputs on one canonical holdout.

This release helper deliberately separates training provenance from evaluation
provenance.  Existing model responses are aligned by sample ID to the
canonical Step 4 holdout, then judged by the shared Step 3 tone judge and
semantic repetition audit.  No model inference or training is performed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
# Allow `python src/step5_unified_syco_eval.py` to import project modules.
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "step5_syco_safe"

MODELS = (
    "Qwen3.5-2B-Base",
    "Qwen3.5-9B-Base",
    "Qwen3.5-35B-A3B-Base",
)

CANONICAL_HOLDOUT = {
    "Qwen3.5-2B-Base": PROJECT_ROOT /
    "outputs/step4_feature_inject/dataset/Qwen3.5-2B-Base/"
    "split_train2000_eval400_seed1234/syco_eval.jsonl",
    "Qwen3.5-9B-Base": PROJECT_ROOT /
    "outputs/step4_feature_inject/dataset/Qwen3.5-9B-Base/"
    "split_train2000_eval400_seed1234/syco_eval.jsonl",
    "Qwen3.5-35B-A3B-Base": PROJECT_ROOT /
    "outputs/step4_feature_inject/dataset/calibration_syco1000_alpaca1000_seed1234/"
    "Qwen3.5-35B-A3B-Base/split_train2000_eval400_seed1234/syco_eval.jsonl",
}

STEP4_ROOTS = {
    "Qwen3.5-2B-Base": PROJECT_ROOT /
    "outputs/step4_feature_inject/eval/Qwen3.5-2B-Base/"
    "gate_f28758_train2000_eval400_syco1000_alpaca1000_seed1234_lr2e-6_ep2",
    "Qwen3.5-9B-Base": PROJECT_ROOT /
    "outputs/step4_feature_inject/eval/Qwen3.5-9B-Base/"
    "gate_f61718_train2000_eval400_syco1000_alpaca1000_seed1234_lr5e-7_ep1",
    "Qwen3.5-35B-A3B-Base": PROJECT_ROOT /
    "outputs/step4_feature_inject/eval/Qwen3.5-35B-A3B-Base/"
    "repro_f2362_train2000_eval400_syco1000_alpaca1000_seed1234",
}

ORIGINAL_RUNS = {
    "Qwen3.5-2B-Base": {
        "base": "base",
        "ordinary_sft": "syco_sft_full_lr2e-6_ep2_gbs8",
        "treatment": "syco_sft_prevent_f28758_alpha5_full_lr2e-6_ep2_gbs8",
        "reverse": "syco_sft_prevent_f28758_alpha_neg10_full_lr2e-6_ep2_gbs8",
    },
    "Qwen3.5-9B-Base": {
        "base": "base",
        "ordinary_sft": "syco_sft_full_lr5e-7_ep1_gbs8",
        "treatment": "syco_sft_prevent_f61718_alpha40_full_lr5e-7_ep1_gbs8",
        "reverse": "syco_sft_prevent_f61718_alpha_neg100_full_lr5e-7_ep1_gbs8",
    },
    "Qwen3.5-35B-A3B-Base": {
        "base": "base",
        "ordinary_sft": "syco_sft_full_lr8.1e-6_ep1_gbs64",
        "treatment": "syco_sft_prevent_f2362_alpha30_full_lr8.1e-6_ep1_gbs64",
        "reverse": "syco_sft_prevent_f2362_alpha_neg1_full_lr8.1e-6_ep1_gbs64",
    },
}

ADJUSTED_SOURCES = {
    ("Qwen3.5-2B-Base", "refusal_anchor"): OUTPUT_ROOT /
    "Qwen3.5-2B-Base/syco_evaluation/adjusted/refusal_anchor/syco/model_outputs.parquet",
    ("Qwen3.5-2B-Base", "treatment"): OUTPUT_ROOT /
    "Qwen3.5-2B-Base/syco_evaluation/adjusted/treatment/syco/model_outputs.parquet",
    # The report reads both fixed ordinary-SFT controls from the original
    # canonical group.  Do not create duplicate adjusted-group judgements.
    ("Qwen3.5-9B-Base", "treatment"): OUTPUT_ROOT /
    "Qwen3.5-9B-Base/syco_evaluation/adjusted/treatment/syco/model_outputs.parquet",
}


def selected_reverse_source(model: str) -> Path:
    """Resolve the safety-blind adjusted reverse candidate's model outputs."""

    full_recovery = OUTPUT_ROOT / model / "full_recovery_reverse_selection.json"
    selection = (
        full_recovery
        if full_recovery.is_file()
        else OUTPUT_ROOT / model / "margin_recovery" / "reverse_selection.json"
    )
    if not selection.is_file():
        raise FileNotFoundError(selection)
    payload = json.loads(selection.read_text(encoding="utf-8"))
    chosen = payload.get("chosen")
    if not isinstance(chosen, dict):
        raise ValueError(f"Adjusted reverse selection has no chosen row: {selection}")
    summary = Path(str(chosen.get("summary") or ""))
    if not summary.is_file():
        raise FileNotFoundError(summary)
    source = summary.parent / "model_outputs.parquet"
    if not source.is_file():
        raise FileNotFoundError(source)
    return source


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def syco_score(summary_path: Path) -> float:
    text = summary_path.read_text(encoding="utf-8")
    match = re.search(
        r"^\|\s*overall\s*\|\s*\d+\s*\|\s*([0-9.]+)\s*\|",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        raise ValueError(f"No overall syco score in {summary_path}")
    return float(match.group(1))


def canonical_holdout(model: str) -> pd.DataFrame:
    path = CANONICAL_HOLDOUT[model]
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_json(path, lines=True)
    required = {"sample_id", "prompt_text"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Canonical holdout missing columns {sorted(missing)}: {path}")
    if len(frame) != 400:
        raise ValueError(f"Canonical holdout must contain 400 rows: {path}")
    return frame.drop_duplicates("sample_id", keep="first").reset_index(drop=True)


def align_outputs(source: Path, holdout: pd.DataFrame) -> pd.DataFrame:
    source_frame = pd.read_parquet(source)
    if "sample_id" not in source_frame or "prompt_text" not in source_frame:
        raise ValueError(f"Model outputs lack sample_id/prompt_text: {source}")
    source_frame = source_frame.drop_duplicates("sample_id", keep="last")
    expected = set(holdout["sample_id"].astype(str))
    actual = set(source_frame["sample_id"].astype(str))
    if actual != expected:
        raise ValueError(
            f"Holdout IDs differ for {source}: missing={len(expected - actual)} "
            f"extra={len(actual - expected)}"
        )
    source_frame["sample_id"] = source_frame["sample_id"].astype(str)
    source_by_id = source_frame.set_index("sample_id")
    ordered = source_by_id.loc[holdout["sample_id"].astype(str)].reset_index()
    prompt_check = ordered["prompt_text"].astype(str).tolist() == holdout["prompt_text"].astype(str).tolist()
    if not prompt_check:
        raise ValueError(f"Prompt text differs from canonical holdout: {source}")
    # Replace only provenance fields; response text is taken from the source run.
    for column in holdout.columns:
        if column in {"response", "reference_sycophantic_response"}:
            continue
        if column in holdout:
            ordered[column] = holdout[column].tolist()
    return ordered


def evaluate_one(
    model: str,
    label: str,
    source: Path,
    group: str,
    judge_workers: int,
) -> Path:
    if not source.is_file():
        raise FileNotFoundError(source)
    holdout_path = CANONICAL_HOLDOUT[model]
    holdout = canonical_holdout(model)
    destination = OUTPUT_ROOT / model / "syco_evaluation" / group / label / "syco"
    destination.mkdir(parents=True, exist_ok=True)
    existing_results = destination / "judge_results.parquet"
    existing_summary = destination / "summary.md"
    manifest_path = destination / "inference_manifest.json"
    if existing_results.is_file() and existing_summary.is_file():
        try:
            previous = pd.read_parquet(existing_results)
            previous_manifest = (
                json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest_path.is_file()
                else {}
            )
            if (
                len(previous) == len(holdout)
                and set(previous.get("judge_version", ())) == {"syco_tone_judge"}
                and set(previous["sample_id"].astype(str))
                == set(holdout["sample_id"].astype(str))
                and previous_manifest.get("source_model_outputs_sha256")
                == sha256_file(source)
                and previous_manifest.get("holdout_sha256")
                == sha256_file(holdout_path)
            ):
                return existing_summary
        except Exception:
            pass
    # Keep a partial checkpoint so transient evaluation failures resume only missing
    # rows. Records from another response set are rejected by content hash.
    for filename in ("judge_results.parquet", "summary.md"):
        (destination / filename).unlink(missing_ok=True)
    aligned = align_outputs(source, holdout)
    aligned.to_parquet(destination / "model_outputs.parquet", index=False)
    write_json(
        manifest_path,
        {
            "evaluation_name": "canonical_syco_holdout",
            "model_name": model,
            "run_name": label,
            "group": group,
            "source_model_outputs": str(source),
            "source_model_outputs_sha256": sha256_file(source),
            "holdout_path": str(holdout_path),
            "holdout_sha256": sha256_file(holdout_path),
            "holdout_rows": len(holdout),
            "response_reused": True,
            "inference_hook": "off",
        },
    )
    from src.step3_pipeline import run_syco_judge_for_experiment

    stats = run_syco_judge_for_experiment(
        str(destination),
        {
            "model": "api_deepseek_deepseek-v4-pro",
            "max_workers": judge_workers,
            "max_retries": 10,
            "timeout": 180,
            "retry_delay": 2.0,
        },
    )
    write_json(
        destination.parent / "evaluation_metadata.json",
        {
            "evaluation_name": "canonical_syco_holdout",
            "model_name": model,
            "run_name": label,
            "group": group,
            "holdout_name": "canonical_syco_holdout",
            "holdout_path": str(holdout_path),
            "holdout_sha256": sha256_file(holdout_path),
            "holdout_rows": len(holdout),
            "judge_name": "syco_tone_judge",
            "repetition_audit": "semantic_repetition_audit",
            "repetition_rule": "semantic_repetition_rule",
            "deterministic_repetition_screen": "long_block_repetition",
            "stats": stats or {},
        },
    )
    return destination / "summary.md"


def build_specs(group: str) -> Iterable[tuple[str, str, str, Path]]:
    if group in {"original", "all"}:
        for model in MODELS:
            root = STEP4_ROOTS[model]
            for label, run_name in ORIGINAL_RUNS[model].items():
                yield "original", model, label, root / run_name / "syco" / "model_outputs.parquet"
    if group in {"adjusted", "all"}:
        for (model, label), source in ADJUSTED_SOURCES.items():
            yield "adjusted", model, label, source
        for model in ("Qwen3.5-2B-Base", "Qwen3.5-9B-Base"):
            full_recovery = OUTPUT_ROOT / model / "full_recovery_reverse_selection.json"
            directional = (
                OUTPUT_ROOT / model / "margin_recovery" / "reverse_selection.json"
            )
            if full_recovery.is_file() or directional.is_file():
                yield "adjusted", model, "reverse", selected_reverse_source(model)


def update_csv_scores(summary_paths: Mapping[tuple[str, str, str], Path]) -> None:
    scores: Dict[tuple[str, str], float] = {}
    for (model, run, group), summary in summary_paths.items():
        if group == "original":
            scores[(model, run)] = syco_score(summary)

    # Step 5 analysis reads this release metadata when regenerating its CSV,
    # JSON, and per-model Markdown summaries. Keep one official score source.
    for (model, run), score in scores.items():
        metadata_path = OUTPUT_ROOT / model / run / "run_metadata.json"
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        payload["sycophancy_score"] = score
        payload["sycophancy_evaluation"] = {
            "holdout": "canonical_syco_holdout",
            "holdout_rows": 400,
            "judge": "syco_tone_judge",
            "repetition_audit": "semantic_repetition_audit",
            "summary_path": str(summary),
        }
        write_json(metadata_path, payload)

    path = OUTPUT_ROOT / "cross_model_summary.csv"
    frame = pd.read_csv(path)
    for index, row in frame.iterrows():
        score = scores.get((str(row["model"]), str(row["run"])))
        if score is not None:
            frame.at[index, "sycophancy_score"] = score
    frame.to_csv(path, index=False)
    json_path = OUTPUT_ROOT / "cross_model_summary.json"
    if json_path.is_file():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        for row in payload.get("rows", []):
            score = scores.get((str(row.get("model")), str(row.get("run"))))
            if score is not None:
                row["sycophancy_score"] = score
        write_json(json_path, payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("original", "adjusted", "all"), default="all")
    parser.add_argument("--judge-workers", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary_paths: Dict[tuple[str, str, str], Path] = {}
    for group, model, label, source in build_specs(args.group):
        print(f"[syco evaluation] {group}/{model}/{label}", flush=True)
        summary = evaluate_one(model, label, source, group, args.judge_workers)
        if group == "original":
            summary_paths[(model, label, group)] = summary
        print(f"[syco evaluation] complete {summary}", flush=True)
    if summary_paths:
        update_csv_scores(summary_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
