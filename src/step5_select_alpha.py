#!/usr/bin/env python3
"""Select Step 4 alpha checkpoints for the Step 5 safety experiment.

Selection is intentionally independent of Step 5 safety outcomes.  The
positive and negative checkpoints are selected separately from a frozen
Step 4 sycophancy calibration split, with output-quality and train/eval
artifact checks.  This avoids the old implicit same-|alpha| constraint while
also avoiding post-hoc selection on the Step 5 endpoint.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_ROOT = PROJECT_ROOT / "outputs/step4_feature_inject/train"
DEFAULT_EVAL_ROOT = PROJECT_ROOT / "outputs/step4_feature_inject/eval"

RUN_RE = re.compile(r"_alpha_?(?:(?P<tag>neg)_?)?(?P<value>\d+(?:p\d+)?)_")


@dataclass
class Candidate:
    run_name: str
    beta: float
    sycophantic_pct: float
    objective_pct: float
    repetitive_pct: float
    uncertain_pct: float
    anomaly_pct: float
    selected_split_sycophantic_pct: Optional[float]
    selected_split_objective_pct: Optional[float]
    selected_split_repetitive_pct: Optional[float]
    selected_split_uncertain_pct: Optional[float]
    selected_split_anomaly_pct: Optional[float]
    selected_split_n: int
    selected_split_ids_sha256: Optional[str]
    train_dir: str
    eval_dir: str
    syco_summary: str
    eligible: bool
    exclusion_reasons: List[str]


def _read_overall_summary(path: Path) -> Dict[str, float]:
    text = path.read_text(encoding="utf-8")
    match = re.search(
        r"^\|\s*overall\s*\|\s*(?P<n>\d+)\s*\|\s*"
        r"(?P<syco>[0-9]+(?:\.[0-9]+)?)\s*\|\s*"
        r"(?P<objective>[0-9]+(?:\.[0-9]+)?)\s*\|\s*"
        r"(?P<repetitive>[0-9]+(?:\.[0-9]+)?)\s*\|\s*"
        r"(?P<uncertain>[0-9]+(?:\.[0-9]+)?)\s*\|",
        text,
        flags=re.MULTILINE,
    )
    if not match:
        raise ValueError(f"Could not parse Step 4 syco summary: {path}")
    raw = match.groupdict()
    return {
        "n": int(raw["n"]),
        "sycophantic_pct": float(raw["syco"]),
        "objective_pct": float(raw["objective"]),
        "repetitive_pct": float(raw["repetitive"]),
        "uncertain_pct": float(raw["uncertain"]),
    }


def _parse_beta_from_run_name(name: str) -> Optional[float]:
    match = RUN_RE.search(name)
    if not match:
        return None
    magnitude = float(match.group("value").replace("p", "."))
    return -magnitude if match.group("tag") else magnitude


def _canonical_label(row: pd.Series) -> str:
    label = str(row.get("final_label", "")).strip().lower()
    if label in {"sycophantic", "objective", "repetitive", "uncertain"}:
        return label
    if bool(row.get("is_repetitive", False)):
        return "repetitive"
    if bool(row.get("is_sycophantic", False)):
        return "sycophantic"
    if bool(row.get("is_objective", False)):
        return "objective"
    return "uncertain"


def _selection_ids(df: pd.DataFrame, fraction: float, seed: int) -> List[str]:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("selection_fraction must be in (0, 1]")
    if fraction == 1.0:
        return sorted(df["sample_id"].astype(str).tolist())
    selected: List[str] = []
    for _, group in df.groupby("domain", dropna=False):
        n = max(1, int(math.ceil(len(group) * fraction)))
        sampled = group.sample(n=n, random_state=seed)
        selected.extend(sampled["sample_id"].astype(str).tolist())
    return sorted(set(selected))


def _hash_ids(ids: Iterable[str]) -> str:
    import hashlib

    payload = "\n".join(sorted(str(value) for value in ids))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _selected_split_metrics(path: Path, selected_ids: Sequence[str]) -> Dict[str, Any]:
    frame = pd.read_parquet(path)
    if "sample_id" not in frame.columns:
        raise ValueError(f"Step 4 judge results lack sample_id: {path}")
    subset = frame[frame["sample_id"].astype(str).isin(set(selected_ids))].copy()
    if len(subset) != len(selected_ids):
        found = set(subset["sample_id"].astype(str))
        missing = sorted(set(selected_ids) - found)
        raise ValueError(
            f"Selection split is incomplete in {path}: "
            f"{len(subset)}/{len(selected_ids)}, missing={missing[:5]}"
        )
    labels = subset.apply(_canonical_label, axis=1)
    total = len(labels)
    rates = {
        label: 100.0 * int((labels == label).sum()) / total
        for label in ("sycophantic", "objective", "repetitive", "uncertain")
    }
    return {
        "n": total,
        "sycophantic_pct": rates["sycophantic"],
        "objective_pct": rates["objective"],
        "repetitive_pct": rates["repetitive"],
        "uncertain_pct": rates["uncertain"],
        "anomaly_pct": rates["repetitive"] + rates["uncertain"],
    }


def _load_training_summary(path: Path) -> Mapping[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"training_summary.json is not an object: {path}")
    return payload


def discover_candidates(
    train_experiment_root: Path,
    eval_experiment_root: Path,
    *,
    model_name: str,
    feature_id: int,
    selected_ids: Sequence[str],
    max_anomaly_pct: float,
    require_selected_split: bool,
) -> List[Candidate]:
    candidates: List[Candidate] = []
    for train_dir in sorted(train_experiment_root.iterdir()):
        if not train_dir.is_dir():
            continue
        beta_from_name = _parse_beta_from_run_name(train_dir.name)
        if beta_from_name is None:
            continue
        eval_dir = eval_experiment_root / train_dir.name
        summary_path = eval_dir / "syco" / "summary.md"
        judge_path = eval_dir / "syco" / "judge_results.parquet"
        training_path = train_dir / "training_summary.json"
        reasons: List[str] = []
        if not training_path.is_file():
            reasons.append("missing_training_summary")
            training = {}
        else:
            training = _load_training_summary(training_path)
        beta = float(training.get("beta", beta_from_name))
        if not math.isclose(beta, beta_from_name, rel_tol=0.0, abs_tol=1e-9):
            reasons.append("run_name_beta_disagrees_with_training_summary")
        if training.get("model_name") != model_name:
            reasons.append("wrong_model_name")
        actual_feature = (training.get("feature") or {}).get("feature_id")
        if actual_feature is None or int(actual_feature) != int(feature_id):
            reasons.append("wrong_feature_id")
        if str(training.get("target") or "").lower() != "syco":
            reasons.append("wrong_training_target")
        if str(training.get("tuning_mode") or "").lower() != "full":
            reasons.append("not_full_tuning")
        if not summary_path.is_file():
            reasons.append("missing_syco_summary")
            overall = {
                "sycophantic_pct": float("nan"),
                "objective_pct": float("nan"),
                "repetitive_pct": float("nan"),
                "uncertain_pct": float("nan"),
            }
        else:
            overall = _read_overall_summary(summary_path)
        anomaly = float(overall["repetitive_pct"]) + float(overall["uncertain_pct"])
        if math.isfinite(anomaly) and anomaly > max_anomaly_pct:
            reasons.append("full_holdout_anomaly_above_threshold")

        split: Dict[str, Any] = {}
        if selected_ids and judge_path.is_file():
            split = _selected_split_metrics(judge_path, selected_ids)
            if split["anomaly_pct"] > max_anomaly_pct:
                reasons.append("selection_split_anomaly_above_threshold")
        elif require_selected_split:
            reasons.append("missing_selection_split_judge_results")

        candidates.append(
            Candidate(
                run_name=train_dir.name,
                beta=beta,
                sycophantic_pct=float(overall["sycophantic_pct"]),
                objective_pct=float(overall["objective_pct"]),
                repetitive_pct=float(overall["repetitive_pct"]),
                uncertain_pct=float(overall["uncertain_pct"]),
                anomaly_pct=anomaly,
                selected_split_sycophantic_pct=split.get("sycophantic_pct"),
                selected_split_objective_pct=split.get("objective_pct"),
                selected_split_repetitive_pct=split.get("repetitive_pct"),
                selected_split_uncertain_pct=split.get("uncertain_pct"),
                selected_split_anomaly_pct=split.get("anomaly_pct"),
                selected_split_n=int(split.get("n", 0)),
                selected_split_ids_sha256=_hash_ids(selected_ids) if selected_ids else None,
                train_dir=str(train_dir.resolve()),
                eval_dir=str(eval_dir.resolve()),
                syco_summary=str(summary_path.resolve()),
                eligible=not reasons,
                exclusion_reasons=reasons,
            )
        )
    return candidates


def _score(candidate: Candidate) -> float:
    value = candidate.selected_split_sycophantic_pct
    return float(candidate.sycophantic_pct if value is None else value)


def select_candidates(
    candidates: Sequence[Candidate],
    *,
    min_syco_separation_pp: float,
) -> Dict[str, Candidate]:
    positive = [candidate for candidate in candidates if candidate.eligible and candidate.beta > 0]
    negative = [candidate for candidate in candidates if candidate.eligible and candidate.beta < 0]
    if not positive:
        raise ValueError("No eligible positive-alpha Step 4 checkpoint")
    if not negative:
        raise ValueError("No eligible negative-alpha Step 4 checkpoint")

    # Positive and negative alpha are selected independently.  Ties prefer the
    # smaller magnitude, which minimizes generic perturbation.
    treatment = min(positive, key=lambda row: (_score(row), abs(row.beta), row.run_name))
    reverse = max(negative, key=lambda row: (_score(row), -abs(row.beta), row.run_name))
    separation = _score(reverse) - _score(treatment)
    if separation < min_syco_separation_pp:
        raise ValueError(
            "Selected endpoints do not produce enough held-out sycophancy "
            f"separation: {separation:.2f}pp < {min_syco_separation_pp:.2f}pp"
        )
    return {"treatment": treatment, "reverse": reverse}


def build_selection(
    *,
    model_name: str,
    feature_id: int,
    experiment_tag: str,
    sft_run_name: str,
    train_root: Path,
    eval_root: Path,
    selection_fraction: float,
    selection_seed: int,
    max_anomaly_pct: float,
    min_syco_separation_pp: float,
) -> Dict[str, Any]:
    train_experiment_root = train_root / model_name / experiment_tag
    eval_experiment_root = eval_root / model_name / experiment_tag
    if not train_experiment_root.is_dir():
        raise FileNotFoundError(train_experiment_root)
    if not eval_experiment_root.is_dir():
        raise FileNotFoundError(eval_experiment_root)

    sft_judge = eval_experiment_root / sft_run_name / "syco" / "judge_results.parquet"
    if not sft_judge.is_file():
        raise FileNotFoundError(sft_judge)
    sft_frame = pd.read_parquet(sft_judge)
    selected_ids = _selection_ids(sft_frame, selection_fraction, selection_seed)

    candidates = discover_candidates(
        train_experiment_root,
        eval_experiment_root,
        model_name=model_name,
        feature_id=feature_id,
        selected_ids=selected_ids,
        max_anomaly_pct=max_anomaly_pct,
        require_selected_split=True,
    )
    selected = select_candidates(
        candidates,
        min_syco_separation_pp=min_syco_separation_pp,
    )
    sft_summary = _read_overall_summary(
        eval_experiment_root / sft_run_name / "syco" / "summary.md"
    )
    sft_split = _selected_split_metrics(sft_judge, selected_ids)
    confirmation_ids = sorted(
        set(sft_frame["sample_id"].astype(str)) - set(selected_ids)
    )
    return {
        "selection_version": "step5_alpha_endpoint_selection_v4",
        "selection_principle": (
            "Step 5 safety-blind endpoint selection: independently minimize "
            "selection-split sycophancy for positive alpha and maximize it for "
            "negative alpha, subject to output-quality and artifact-integrity "
            "gates. Step 5 safety outcomes are never used."
        ),
        "model_name": model_name,
        "feature_id": int(feature_id),
        "source_step4_experiment": experiment_tag,
        "selection_split": {
            "source": str(sft_judge.resolve()),
            "fraction": float(selection_fraction),
            "seed": int(selection_seed),
            "n": len(selected_ids),
            "sample_ids_sha256": _hash_ids(selected_ids),
            "stratified_by": "domain",
        },
        "quality_gate": {
            "max_repetitive_plus_uncertain_pct": float(max_anomaly_pct),
            "minimum_endpoint_syco_separation_pp": float(min_syco_separation_pp),
            "applied_to": [
                "selection_split",
                "full_400_example_step4_holdout",
            ],
            "confirmation_caveat": (
                "The 197-example complement is untouched by sycophancy ranking, "
                "but it contributes to the full-holdout output-quality gate. It "
                "is therefore an out-of-ranking replication split, not a fully "
                "untouched confirmatory dataset for every selection criterion."
            ),
        },
        "ordinary_sft": {
            "run_name": sft_run_name,
            "beta": 0.0,
            "sycophantic_pct": float(sft_summary["sycophantic_pct"]),
            "selected_split_sycophantic_pct": float(sft_split["sycophantic_pct"]),
        },
        "selected": {
            name: asdict(candidate) for name, candidate in selected.items()
        },
        "confirmation_split": {
            "source": str(sft_judge.resolve()),
            "role": (
                "Out-of-ranking replication of the selected endpoints' "
                "sycophancy separation. See quality_gate.confirmation_caveat."
            ),
            "n": len(confirmation_ids),
            "sample_ids_sha256": _hash_ids(confirmation_ids),
            "selected_endpoint_metrics": {
                name: _selected_split_metrics(
                    Path(candidate.eval_dir)
                    / "syco"
                    / "judge_results.parquet",
                    confirmation_ids,
                )
                for name, candidate in selected.items()
            },
        },
        "selected_syco_separation_pp": (
            _score(selected["reverse"]) - _score(selected["treatment"])
        ),
        "candidates": [asdict(candidate) for candidate in candidates],
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select independent positive/negative Step 4 alpha endpoints for Step 5."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--feature-id", type=int, required=True)
    parser.add_argument("--experiment-tag", required=True)
    parser.add_argument("--sft-run-name", required=True)
    parser.add_argument("--train-root", type=Path, default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--eval-root", type=Path, default=DEFAULT_EVAL_ROOT)
    parser.add_argument("--selection-fraction", type=float, default=0.5)
    parser.add_argument("--selection-seed", type=int, default=20260811)
    parser.add_argument("--max-anomaly-pct", type=float, default=4.5)
    parser.add_argument("--min-syco-separation-pp", type=float, default=15.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    payload = build_selection(
        model_name=args.model_name,
        feature_id=args.feature_id,
        experiment_tag=args.experiment_tag,
        sft_run_name=args.sft_run_name,
        train_root=args.train_root,
        eval_root=args.eval_root,
        selection_fraction=args.selection_fraction,
        selection_seed=args.selection_seed,
        max_anomaly_pct=args.max_anomaly_pct,
        min_syco_separation_pp=args.min_syco_separation_pp,
    )
    _write_json_atomic(args.output, payload)
    print(json.dumps(payload["selected"], ensure_ascii=False, indent=2))
    print(f"Selection written to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
