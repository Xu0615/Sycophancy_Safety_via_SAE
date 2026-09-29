#!/usr/bin/env python3
"""Expand frozen Step 5 assigned-template results to the all-template schema.

This is a compute-saving compatibility path for checkpoints that were already
evaluated with one deterministic pressure family per harmful intent.  It does
not invent new model outputs: the historical row is retained for its assigned
template and copied to the other template IDs with an explicit provenance flag.

Such rows support checkpoint/alpha comparisons on the exact same 296 harmful
requests, but they do *not* constitute independent evidence about variation
between the four pressure phrasings.  The final report must state this
limitation. Fresh all-template results, when present, always take precedence.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.step5_syco_safe import CONDITIONS, PRESSURE_TEMPLATES


def _write_parquet_atomic(frame: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    frame.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def collapse_direct(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic direct row per original harmful intent.

    Early v2 runs generated the same direct prompt four times (once per pressure
    family). The model response is deterministic, but the external judge can differ
    slightly. We retain a row whose label equals the within-intent modal label,
    then break ties by the fixed pressure-template order.
    """

    template_order = {
        template["id"]: index for index, template in enumerate(PRESSURE_TEMPLATES)
    }
    rows = []
    working = frame.copy()
    if "base_intent_id" not in working:
        working["base_intent_id"] = working["intent_id"].astype(str).str.split("::").str[0]
    for base_intent_id, group in working.groupby("base_intent_id", sort=True):
        labels = group["final_label"].astype(str)
        counts = labels.value_counts()
        modal_labels = set(counts[counts == counts.max()].index)
        candidates = group[labels.isin(modal_labels)].copy()
        candidates["_template_order"] = candidates.get(
            "pressure_template_id", pd.Series("", index=candidates.index)
        ).map(template_order).fillna(len(template_order))
        row = candidates.sort_values(
            ["_template_order", "sample_id"], kind="stable"
        ).iloc[0].drop(labels=["_template_order"]).to_dict()
        base_intent_id = str(base_intent_id)
        row.update(
            {
                "sample_id": f"{base_intent_id}::direct",
                "intent_id": base_intent_id,
                "base_intent_id": base_intent_id,
                "analysis_unit_id": base_intent_id,
                "condition": "direct",
                "pressure_template_id": "none",
                "pressure_template_mode": "all",
                "template_observation_provenance": (
                    "collapsed_replicated_direct_modal_judge"
                    if len(group) > 1
                    else "observed_direct"
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def expand_condition(frame: pd.DataFrame, condition: str) -> pd.DataFrame:
    if condition == "direct":
        return collapse_direct(frame)
    template_ids = [template["id"] for template in PRESSURE_TEMPLATES]
    rows = []
    for record in frame.to_dict("records"):
        base_intent_id = str(
            record.get("base_intent_id")
            or record.get("intent_id")
            or str(record.get("sample_id", "")).split("::", 1)[0]
        )
        observed_template = str(record.get("pressure_template_id") or template_ids[0])
        for template_id in template_ids:
            row = dict(record)
            unit = f"{base_intent_id}::{template_id}"
            row.update(
                {
                    "sample_id": f"{unit}::{condition}",
                    "intent_id": unit,
                    "base_intent_id": base_intent_id,
                    "analysis_unit_id": unit,
                    "condition": condition,
                    "pressure_template_id": template_id,
                    "pressure_template_mode": "all",
                    "template_observation_provenance": (
                        "observed_assigned_template"
                        if template_id == observed_template
                        else "imputed_from_assigned_template"
                    ),
                    "historical_observed_template_id": observed_template,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def expand_run(run_dir: Path) -> None:
    for condition in CONDITIONS:
        path = run_dir / condition / "judge_results.parquet"
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path)
        if condition == "direct":
            expanded = collapse_direct(frame)
            _write_parquet_atomic(expanded, path)
            continue
        # A fresh all-template pressure result is already complete and must
        # never be overwritten by the compatibility expander.
        if (
            len(frame) > 296
            and "pressure_template_mode" in frame
            and frame["pressure_template_mode"].eq("all").all()
        ):
            continue
        expanded = expand_condition(frame, condition)
        _write_parquet_atomic(expanded, path)

    provenance_counts = {}
    for condition in ("pressure",):
        path = run_dir / condition / "judge_results.parquet"
        frame = pd.read_parquet(path)
        if "template_observation_provenance" in frame.columns:
            for key, value in frame[
                "template_observation_provenance"
            ].fillna("fresh_all_template").value_counts().items():
                provenance_counts[str(key)] = (
                    provenance_counts.get(str(key), 0) + int(value)
                )
        else:
            provenance_counts["fresh_all_template"] = (
                provenance_counts.get("fresh_all_template", 0) + len(frame)
            )
    has_imputation = any(
        key.startswith("imputed_") for key in provenance_counts
    )

    metadata_path = run_dir / "eval_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "step5_version": "step5_syco_safe_v3",
            "conditions": list(CONDITIONS),
            "pressure_template_mode": "all",
            "pressure_template_version": "syco_pressure_templates_v2",
            "template_expansion": {
                "method": (
                    "historical_assigned_template_replication"
                    if has_imputation
                    else "fresh_all_template_pressure_with_collapsed_direct"
                ),
                "fresh_all_template_inference": not has_imputation,
                "valid_for": (
                    "checkpoint/alpha comparisons on the same harmful intents"
                ),
                "not_valid_for": (
                    "estimating differential effects between pressure templates"
                    if has_imputation
                    else None
                ),
                "provenance_counts": provenance_counts,
            },
        }
    )
    tmp = metadata_path.with_name(f"{metadata_path.name}.tmp.{os.getpid()}")
    tmp.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    os.replace(tmp, metadata_path)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", type=Path)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    for run_dir in args.run_dirs:
        expand_run(run_dir)
        print(f"Expanded historical Step 5 run: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
