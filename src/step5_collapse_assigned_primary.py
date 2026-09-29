#!/usr/bin/env python3
"""Materialize the primary one-template-per-intent Step 5 protocol.

Fresh ``all``-template runs are useful as wording-sensitivity stress tests, but
the cross-model primary analysis uses one deterministic pressure family per
harmful request. This script:

* optionally archives current all-template judge tables;
* keeps the stable hashed pressure family for each original intent;
* keeps the corresponding direct row (or a deterministic fallback);
* writes 296 paired rows per prompt condition;
* records whether each retained row was fresh or compatibility-expanded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.step5_syco_safe import CONDITIONS, stable_pressure_template


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def _base_intent(frame: pd.DataFrame) -> pd.Series:
    if "base_intent_id" in frame.columns:
        base = frame["base_intent_id"].fillna("").astype(str)
    else:
        base = pd.Series("", index=frame.index, dtype=str)
    fallback = frame.get("intent_id", frame["sample_id"]).astype(str).str.split("::").str[0]
    return base.where(base.ne(""), fallback)


def _assigned_template(base_intent_id: str) -> str:
    return str(stable_pressure_template(base_intent_id)["id"])


def _choose_direct(group: pd.DataFrame, assigned_template: str) -> pd.Series:
    if "pressure_template_id" in group.columns:
        exact = group[group["pressure_template_id"].astype(str) == assigned_template]
        if not exact.empty:
            return exact.sort_values("sample_id", kind="stable").iloc[-1]
    # Historical compatibility expansion may already have collapsed direct.
    return group.sort_values("sample_id", kind="stable").iloc[-1]


def collapse_condition(
    frame: pd.DataFrame,
    condition: str,
    *,
    default_provenance: str,
) -> pd.DataFrame:
    working = frame.copy()
    working["base_intent_id"] = _base_intent(working)
    rows = []
    for base_intent_id, group in working.groupby("base_intent_id", sort=True):
        base_intent_id = str(base_intent_id)
        assigned = _assigned_template(base_intent_id)
        if condition == "direct":
            chosen = _choose_direct(group, assigned)
        else:
            if "pressure_template_id" not in group.columns:
                raise ValueError(
                    f"{condition} lacks pressure_template_id for {base_intent_id}"
                )
            exact = group[group["pressure_template_id"].astype(str) == assigned]
            if exact.empty:
                raise ValueError(
                    f"{condition} lacks assigned template {assigned} for {base_intent_id}"
                )
            # Prefer a genuinely observed row over an imputed compatibility row.
            if "template_observation_provenance" in exact.columns:
                priority = {
                    "fresh_all_template": 0,
                    "observed_assigned_template": 0,
                    "observed_direct": 0,
                    "imputed_from_assigned_template": 1,
                }
                exact = exact.assign(
                    _provenance_priority=exact[
                        "template_observation_provenance"
                    ].fillna("fresh_all_template").map(priority).fillna(2)
                ).sort_values(
                    ["_provenance_priority", "sample_id"], kind="stable"
                )
            chosen = exact.iloc[0]
        row = chosen.drop(
            labels=["_provenance_priority"], errors="ignore"
        ).to_dict()
        if not str(row.get("template_observation_provenance") or "").strip():
            row["template_observation_provenance"] = default_provenance
        row.update(
            {
                "sample_id": f"{base_intent_id}::{condition}",
                "intent_id": base_intent_id,
                "base_intent_id": base_intent_id,
                "analysis_unit_id": base_intent_id,
                "condition": condition,
                "pressure_template_id": (
                    "none" if condition == "direct" else assigned
                ),
                "pressure_template_mode": "assigned",
                "primary_protocol": True,
            }
        )
        rows.append(row)
    result = pd.DataFrame(rows)
    if result["base_intent_id"].duplicated().any():
        raise ValueError(f"duplicate base_intent_id after collapsing {condition}")
    return result


def _source_inference_manifest(run_dir: Path, condition: str) -> Dict[str, Any]:
    path = run_dir / condition / "inference_manifest.json"
    if not path.is_file():
        return {
            "manifest_path": str(path),
            "present": False,
            "pressure_template_mode": None,
            "sample_count": None,
        }
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "manifest_path": str(path),
            "present": True,
            "pressure_template_mode": None,
            "sample_count": None,
            "parse_error": True,
        }
    return {
        "manifest_path": str(path.resolve()),
        "present": True,
        "pressure_template_mode": payload.get("pressure_template_mode"),
        "sample_count": len(payload.get("sample_ids") or []),
    }


def collapse_run(run_dir: Path, archive_all: bool) -> None:
    archive_dir = run_dir / "template_sensitivity_all"
    provenance_counts = {}
    source_inference_by_condition: Dict[str, Dict[str, Any]] = {}
    for condition in CONDITIONS:
        path = run_dir / condition / "judge_results.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        source_manifest = _source_inference_manifest(run_dir, condition)
        source_inference_by_condition[condition] = source_manifest
        if archive_all and len(frame) > 296:
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = archive_dir / f"{condition}_judge_results.parquet"
            if not archive_path.exists():
                shutil.copy2(path, archive_path)
        source_mode = str(source_manifest.get("pressure_template_mode") or "")
        default_provenance = (
            "observed_assigned_subset_from_fresh_all"
            if source_mode == "all"
            else "fresh_assigned"
        )
        collapsed = collapse_condition(
            frame,
            condition,
            default_provenance=default_provenance,
        )
        if len(collapsed) != 296:
            raise ValueError(
                f"{run_dir}/{condition}: expected 296 rows, got {len(collapsed)}"
            )
        _write_parquet_atomic(collapsed, path)

        output_path = run_dir / condition / "model_outputs.parquet"
        if output_path.is_file():
            outputs = pd.read_parquet(output_path)
            collapsed_outputs = collapse_condition(
                outputs,
                condition,
                default_provenance=default_provenance,
            )
            if len(collapsed_outputs) != 296:
                raise ValueError(
                    f"{run_dir}/{condition}: expected 296 model-output rows, "
                    f"got {len(collapsed_outputs)}"
                )
            _write_parquet_atomic(collapsed_outputs, output_path)

        manifest_path = run_dir / condition / "inference_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            message_hashes = {
                str(row["sample_id"]): hashlib.sha256(
                    str(row.get("messages_json", "")).encode("utf-8")
                ).hexdigest()
                for row in collapsed_outputs.to_dict("records")
            }
            manifest.update(
                {
                    "cache_version": "step5_syco_safe_v3",
                    "pressure_template_mode": "assigned",
                    "sample_ids": collapsed["sample_id"].astype(str).tolist(),
                    "messages_sha256_by_sample_id": message_hashes,
                    "materialized_assigned_primary": True,
                }
            )
            tmp = manifest_path.with_name(
                f"{manifest_path.name}.tmp.{os.getpid()}"
            )
            tmp.write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp, manifest_path)
        if "template_observation_provenance" in collapsed.columns:
            for key, value in collapsed[
                "template_observation_provenance"
            ].fillna("fresh_assigned").value_counts().items():
                provenance_counts[str(key)] = (
                    provenance_counts.get(str(key), 0) + int(value)
                )
        else:
            provenance_counts["fresh_assigned"] = (
                provenance_counts.get("fresh_assigned", 0) + len(collapsed)
            )

    metadata_path = run_dir / "eval_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "step5_version": "step5_syco_safe_v3",
            "conditions": list(CONDITIONS),
            "pressure_template_mode": "assigned",
            "pressure_template_version": "syco_pressure_templates_v2",
            "primary_protocol": {
                "name": "deterministic_assigned_template",
                "n_original_intents": 296,
                "selection": "sha256(intent_id) modulo four pressure families",
                "provenance_counts": provenance_counts,
                "source_inference_by_condition": source_inference_by_condition,
                "all_template_archive": (
                    str(archive_dir.resolve()) if archive_dir.exists() else None
                ),
            },
        }
    )
    tmp = metadata_path.with_name(f"{metadata_path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, metadata_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--archive-all", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    for run_dir in args.run_dirs:
        collapse_run(run_dir, archive_all=args.archive_all)
        print(f"Materialized assigned primary protocol: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
