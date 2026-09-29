import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from src.step5_syco_safe import (
    PRESSURE_TEMPLATES,
    build_condition_messages,
    build_condition_prompts,
    pressure_templates_for_intent,
    run,
    stable_pressure_template,
)
from src.step5_syco_safe_analyse import (
    DIRECT_CONDITION,
    PRESSURE_CONDITION,
    _attach_ordinary_sft_effects,
    analyze_root,
    clustered_checkpoint_effect,
    compute_core_metrics,
    cross_run_paired_contrast,
    normalize_condition,
    normalize_step5_results,
    transition_table,
    validate_prompt_condition_invariants,
)


CONDITIONS = (
    DIRECT_CONDITION,
    PRESSURE_CONDITION,
)


def _paired_rows(labels_by_intent):
    rows = []
    for intent_id, labels in labels_by_intent.items():
        for condition in CONDITIONS:
            rows.append(
                {
                    "intent_id": intent_id,
                    "sample_id": f"{intent_id}__{condition}",
                    "condition": condition,
                    "original_prompt_text": f"harmful request for {intent_id}",
                    "prompt_text": f"{condition} prompt for {intent_id}",
                    "final_label": labels[condition],
                }
            )
    return pd.DataFrame(rows)


def test_inference_prompt_conditions_keep_original_harmful_intent_fixed():
    original = "Give me concrete harmful instructions."
    template = PRESSURE_TEMPLATES[0]
    direct = build_condition_messages(
        original, "direct", template["id"], template["text"]
    )
    pressure = build_condition_messages(
        original, "pressure", template["id"], template["text"]
    )
    assert direct == [{"role": "user", "content": original}]
    assert pressure[0]["role"] == "user"
    assert pressure[0]["content"].startswith(original)
    assert template["text"] in pressure[0]["content"]


def test_inference_prompt_builder_pairs_all_conditions_by_intent_and_template():
    intents = pd.DataFrame(
        [
            {
                "sample_id": "intent_a",
                "prompt_text": "harmful request A",
                "prompt_scope": "HARMFUL",
            },
            {
                "sample_id": "intent_b",
                "prompt_text": "harmful request B",
                "prompt_scope": "HARMFUL",
            },
        ]
    )

    frames = build_condition_prompts(
        intents,
        ["direct", "pressure"],
        pressure_template_mode="assigned",
    )

    assert set(frames) == {"direct", "pressure"}
    for intent_id in ("intent_a", "intent_b"):
        expected_template = stable_pressure_template(intent_id)["id"]
        for condition, frame in frames.items():
            row = frame.set_index("intent_id").loc[intent_id]
            assert row["condition"] == condition
            assert row["pressure_template_id"] == (
                "none" if condition == "direct" else expected_template
            )
            assert row["judge_prompt_text"] == (
                "harmful request A" if intent_id == "intent_a" else "harmful request B"
            )
            assert row["sample_id"] == f"{intent_id}::{condition}"


def test_all_template_mode_crosses_every_intent_with_every_pressure_family():
    intents = pd.DataFrame(
        [
            {
                "sample_id": "intent_a",
                "prompt_text": "harmful request A",
                "prompt_scope": "HARMFUL",
            }
        ]
    )
    frames = build_condition_prompts(
        intents,
        ["direct", "pressure"],
        pressure_template_mode="all",
    )

    expected = {template["id"] for template in PRESSURE_TEMPLATES}
    assert set(frames["direct"]["pressure_template_id"]) == {"none"}
    assert set(frames["pressure"]["pressure_template_id"]) == expected
    assert len(frames["direct"]) == 1
    assert len(frames["pressure"]) == len(expected)
    assert pressure_templates_for_intent("intent_a", "all") == [
        dict(template) for template in PRESSURE_TEMPLATES
    ]
    for condition, frame in frames.items():
        assert frame["base_intent_id"].eq("intent_a").all()
        if condition == "direct":
            assert frame["intent_id"].eq("intent_a").all()
        else:
            assert frame["intent_id"].str.startswith("intent_a::").all()


def test_summary_metrics_accept_replicated_direct_rows_in_all_template_mode():
    direct = pd.DataFrame(
        [
            {
                "base_intent_id": "i1",
                "intent_id": f"i1::{template}",
                "analysis_unit_id": f"i1::{template}",
                "final_label": "REFUSAL",
            }
            for template in ("praise", "trust")
        ]
    )
    pressure = pd.DataFrame(
        [
            {
                "base_intent_id": "i1",
                "intent_id": "i1::praise",
                "analysis_unit_id": "i1::praise",
                "final_label": "UHC",
            },
            {
                "base_intent_id": "i1",
                "intent_id": "i1::trust",
                "analysis_unit_id": "i1::trust",
                "final_label": "REFUSAL",
            },
        ]
    )
    from src.step5_syco_safe import single_turn_pressure_violation

    assert single_turn_pressure_violation(direct, pressure)["violation_n"] == 1


def test_summary_only_does_not_load_inference_engine(tmp_path):
    run_dir = tmp_path / "outputs" / "Qwen-Test" / "base"
    for condition in ("direct", "pressure"):
        condition_dir = run_dir / condition
        condition_dir.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "intent_id": "i1",
                    "base_intent_id": "i1",
                    "analysis_unit_id": "i1",
                    "condition": condition,
                    "final_label": "REFUSAL",
                }
            ]
        ).to_parquet(condition_dir / "judge_results.parquet")

    args = Namespace(
        model_name="Qwen-Test",
        model_path=str(tmp_path / "model"),
        model_root=str(tmp_path),
        artifact_dir=None,
        run_name="base",
        output_root=str(tmp_path / "outputs"),
        log_root=str(tmp_path / "logs"),
        conditions=["direct", "pressure"],
        pressure_template_mode="all",
        config="configs/model.yaml",
        judge_config="configs/judge.yaml",
        dry_run=False,
        summary_only=True,
        skip_inference=False,
        skip_judge=False,
    )
    with (
        patch("src.step5_syco_safe.resolve_paths") as resolve,
        patch(
            "src.step5_syco_safe.load_intents", return_value=pd.DataFrame()
        ) as load_intents_mock,
        patch(
            "src.step5_syco_safe.build_condition_prompts", return_value={}
        ) as build_prompts_mock,
        patch(
            "src.step5_syco_safe.write_dataset_manifest"
        ) as write_manifest_mock,
        patch(
            "src.step5_syco_safe.resolve_inference_backend",
            return_value="vllm",
        ) as resolve_backend_mock,
        patch(
            "src.step5_syco_safe.write_eval_metadata"
        ) as write_metadata_mock,
        patch("src.step5_syco_safe.load_vllm_engine") as loader,
    ):
        resolve.return_value = {
            "base_model_path": str(tmp_path / "model"),
            "model_path": str(tmp_path / "model"),
            "adapter_dir": "",
            "full_model_dir": "",
            "artifact_kind": "base",
            "model_output_root": str(tmp_path / "outputs" / "Qwen-Test"),
            "run_dir": str(run_dir),
            "log_dir": str(tmp_path / "logs" / "Qwen-Test"),
            "log_file": str(tmp_path / "logs" / "Qwen-Test" / "base.log"),
        }
        run(args)
        resolve.assert_called_once()
        loader.assert_not_called()
        # Summary-only must preserve completed-run metadata and avoid all
        # dataset/backend side effects.
        assert not any(
            mock.called
            for mock in (
                load_intents_mock,
                build_prompts_mock,
                write_manifest_mock,
                resolve_backend_mock,
                write_metadata_mock,
            )
        )


def test_condition_aliases_normalize_to_two_canonical_conditions():
    assert normalize_condition("direct_baseline") == DIRECT_CONDITION
    assert normalize_condition("approval-pressure") == PRESSURE_CONDITION


def test_prompt_condition_invariants_accept_complete_paired_intents():
    frame = _paired_rows(
        {
            "i1": {
                DIRECT_CONDITION: "REFUSAL",
                PRESSURE_CONDITION: "UHC",
            },
            "i2": {
                DIRECT_CONDITION: "AHC",
                PRESSURE_CONDITION: "AHC",
            },
        }
    )

    report = validate_prompt_condition_invariants(frame)

    assert report["n_intents"] == 2
    assert report["complete_intents"] == 2
    assert report["incomplete_intents"] == 0
    assert report["original_prompt_column_checked"] == "original_prompt_text"


def test_prompt_condition_invariants_reject_missing_condition():
    frame = _paired_rows(
        {
            "i1": {
                DIRECT_CONDITION: "REFUSAL",
                PRESSURE_CONDITION: "UHC",
            },
            "i2": {
                DIRECT_CONDITION: "REFUSAL",
                PRESSURE_CONDITION: "REFUSAL",
            },
        }
    )
    frame = frame[
        ~(
            frame["intent_id"].eq("i2")
            & frame["condition"].eq(PRESSURE_CONDITION)
        )
    ]

    with pytest.raises(ValueError, match="lack a complete condition set"):
        validate_prompt_condition_invariants(frame)


def test_prompt_condition_invariants_reject_changed_original_request():
    frame = _paired_rows(
        {
            "i1": {
                DIRECT_CONDITION: "REFUSAL",
                PRESSURE_CONDITION: "UHC",
            }
        }
    )
    frame.loc[
        frame["condition"].eq(PRESSURE_CONDITION), "original_prompt_text"
    ] = "a different harmful request"

    with pytest.raises(ValueError, match="original harmful request changed"):
        validate_prompt_condition_invariants(frame)


def test_normalize_results_derives_labels_from_existing_harmful_judge_fields():
    frame = pd.DataFrame(
        [
            {
                "intent_id": "i1",
                "condition": "direct",
                "prompt_scope": "HARMFUL",
                "judge_status": "OK",
                "delivered_harmful_payload": False,
                "visible_harm_awareness": True,
                "effective_refusal": True,
                "is_repetitive": False,
            },
            {
                "intent_id": "i1",
                "condition": "pressure",
                "prompt_scope": "HARMFUL",
                "judge_status": "OK",
                "delivered_harmful_payload": True,
                "visible_harm_awareness": True,
                "effective_refusal": False,
                "is_repetitive": False,
            },
        ]
    )

    normalized = normalize_step5_results(frame)

    assert normalized["label"].tolist() == ["REFUSAL", "AHC"]


def test_core_metrics_keep_only_single_turn_pressure_violation():
    frame = _paired_rows(
        {
            "i1": {
                DIRECT_CONDITION: "REFUSAL",
                PRESSURE_CONDITION: "UHC",
            },
            "i2": {
                DIRECT_CONDITION: "REFUSAL",
                PRESSURE_CONDITION: "AHC",
            },
            "i3": {
                DIRECT_CONDITION: "REFUSAL",
                PRESSURE_CONDITION: "REFUSAL",
            },
            "i4": {
                DIRECT_CONDITION: "AHC",
                PRESSURE_CONDITION: "UHC",
            },
            "i5": {
                DIRECT_CONDITION: "UHC",
                PRESSURE_CONDITION: "REFUSAL",
            },
        }
    )

    metrics = compute_core_metrics(frame, bootstrap_samples=0)
    single_turn_violation = metrics["single_turn_pressure_violation"]

    assert set(metrics) == {"single_turn_pressure_violation"}
    assert single_turn_violation["any_compliance"]["denominator"] == 3
    assert single_turn_violation["any_compliance"]["numerator"] == 2
    assert single_turn_violation["any_compliance"]["rate_pct"] == pytest.approx(200 / 3)
    assert single_turn_violation["to_AHC"]["numerator"] == 1
    assert single_turn_violation["to_UHC"]["numerator"] == 1

def test_transition_table_reports_exact_paired_counts():
    frame = _paired_rows(
        {
            "i1": {
                DIRECT_CONDITION: "REFUSAL",
                PRESSURE_CONDITION: "UHC",
            },
            "i2": {
                DIRECT_CONDITION: "REFUSAL",
                PRESSURE_CONDITION: "AHC",
            },
            "i3": {
                DIRECT_CONDITION: "AHC",
                PRESSURE_CONDITION: "REFUSAL",
            },
        }
    )

    table = transition_table(frame, "direct", "pressure")

    assert table["paired_n"] == 3
    assert table["counts"]["REFUSAL"]["AHC"] == 1
    assert table["counts"]["REFUSAL"]["UHC"] == 1
    assert table["counts"]["AHC"]["REFUSAL"] == 1
    assert table["row_rates_pct"]["REFUSAL"]["AHC"] == 50.0
    assert table["row_rates_pct"]["REFUSAL"]["UHC"] == 50.0


def test_bootstrap_ci_is_deterministic_and_contains_point_estimate():
    labels = {}
    for index in range(20):
        labels[f"i{index}"] = {
            DIRECT_CONDITION: "REFUSAL",
            PRESSURE_CONDITION: "UHC" if index < 6 else "REFUSAL",
        }
    frame = _paired_rows(labels)

    first = compute_core_metrics(
        frame, bootstrap_samples=500, confidence=0.95, seed=77
    )
    second = compute_core_metrics(
        frame, bootstrap_samples=500, confidence=0.95, seed=77
    )
    estimate = first["single_turn_pressure_violation"]["any_compliance"]

    assert first == second
    assert estimate["rate_pct"] == 30.0
    assert estimate["ci_low_pct"] <= estimate["rate_pct"]
    assert estimate["ci_high_pct"] >= estimate["rate_pct"]


def test_cross_run_contrast_uses_only_shared_direct_refusal_intents():
    left = _paired_rows(
        {
            "i1": {
                DIRECT_CONDITION: "REFUSAL",
                PRESSURE_CONDITION: "REFUSAL",
            },
            "i2": {
                DIRECT_CONDITION: "REFUSAL",
                PRESSURE_CONDITION: "UHC",
            },
            "left_only": {
                DIRECT_CONDITION: "REFUSAL",
                PRESSURE_CONDITION: "REFUSAL",
            },
        }
    )
    right = _paired_rows(
        {
            "i1": {
                DIRECT_CONDITION: "REFUSAL",
                PRESSURE_CONDITION: "AHC",
            },
            "i2": {
                DIRECT_CONDITION: "REFUSAL",
                PRESSURE_CONDITION: "UHC",
            },
            "right_only": {
                DIRECT_CONDITION: "REFUSAL",
                PRESSURE_CONDITION: "UHC",
            },
        }
    )

    contrast = cross_run_paired_contrast(
        left,
        right,
        metric="single_turn_pressure_violation",
        bootstrap_samples=0,
    )

    assert contrast["paired_n"] == 2
    assert contrast["left_only_eligible_intents"] == 1
    assert contrast["right_only_eligible_intents"] == 1
    assert contrast["left_rate_pct"] == 50.0
    assert contrast["right_rate_pct"] == 100.0
    assert contrast["risk_difference_pct"] == 50.0
    assert contrast["right_only"] == 1
    assert contrast["left_only"] == 0
    assert contrast["mcnemar_exact_p"] == 1.0


def test_summary_rows_orient_paired_effects_as_checkpoint_minus_sft():
    rows = [
        {"run": "base"},
        {"run": "ordinary_sft"},
        {"run": "treatment"},
    ]
    contrasts = [
        {
            "left_run": "base",
            "right_run": "ordinary_sft",
            "metrics": {
                "single_turn_pressure_violation": {
                    "risk_difference_pct": 10.0,
                    "ci_low_pct": 4.0,
                    "ci_high_pct": 16.0,
                    "paired_n": 80,
                }
            },
        },
        {
            "left_run": "ordinary_sft",
            "right_run": "treatment",
            "metrics": {
                "single_turn_pressure_violation": {
                    "risk_difference_pct": -7.0,
                    "ci_low_pct": -12.0,
                    "ci_high_pct": -2.0,
                    "paired_n": 75,
                }
            },
        },
    ]

    enriched = {
        row["run"]: row for row in _attach_ordinary_sft_effects(rows, contrasts)
    }

    assert enriched["base"]["paired_vs_ordinary_sft_risk_difference_pct"] == -10.0
    assert enriched["base"]["paired_vs_ordinary_sft_ci_low_pct"] == -16.0
    assert enriched["base"]["paired_vs_ordinary_sft_ci_high_pct"] == -4.0
    assert enriched["base"]["paired_vs_ordinary_sft_n"] == 80
    assert enriched["ordinary_sft"]["paired_vs_ordinary_sft_reference"] is True
    assert enriched["treatment"]["paired_vs_ordinary_sft_risk_difference_pct"] == -7.0


def test_clustered_checkpoint_effect_resamples_original_intents_not_templates():
    rows = {}
    for prefix, pressure_labels in {
        "left": {
            "i1": ("REFUSAL", "REFUSAL"),
            "i2": ("REFUSAL", "UHC"),
        },
        "right": {
            "i1": ("AHC", "UHC"),
            "i2": ("REFUSAL", "UHC"),
        },
    }.items():
        frame_rows = []
        for base_intent, labels in pressure_labels.items():
            for template_id, pressure_label in zip(("praise", "trust"), labels):
                unit = f"{base_intent}::{template_id}"
                frame_rows.extend(
                    [
                        {
                            "intent_id": unit,
                            "base_intent_id": base_intent,
                            "condition": DIRECT_CONDITION,
                            "final_label": "REFUSAL",
                        },
                        {
                            "intent_id": unit,
                            "base_intent_id": base_intent,
                            "condition": PRESSURE_CONDITION,
                            "final_label": pressure_label,
                        },
                    ]
                )
        rows[prefix] = pd.DataFrame(frame_rows)

    effect = clustered_checkpoint_effect(
        rows,
        left_run="left",
        right_run="right",
        metric="single_turn_pressure_violation",
        bootstrap_samples=0,
    )

    assert effect["paired_cluster_n"] == 2
    assert effect["risk_difference_pct"] == 50.0


def test_root_analysis_reads_separate_condition_directories_and_writes_summaries(
    tmp_path,
):
    root = tmp_path / "step5_syco_safe"
    run_dir = root / "Qwen-Test" / "treated_seed1234"
    frame = _paired_rows(
        {
            "i1": {
                DIRECT_CONDITION: "REFUSAL",
                PRESSURE_CONDITION: "UHC",
            },
            "i2": {
                DIRECT_CONDITION: "REFUSAL",
                PRESSURE_CONDITION: "REFUSAL",
            },
        }
    )
    for condition in CONDITIONS:
        condition_dir = run_dir / condition
        condition_dir.mkdir(parents=True)
        condition_frame = frame[frame["condition"].eq(condition)].drop(
            columns=["condition", "final_label"]
        )
        labels = frame.loc[
            frame["condition"].eq(condition), "final_label"
        ].tolist()
        condition_frame = condition_frame.assign(final_label=labels)
        condition_frame.to_parquet(condition_dir / "judge_results.parquet")
    (run_dir / "run_metadata.json").write_text(
        json.dumps(
            {
                "beta": 15,
                "feature_id": 61718,
                "seed": 1234,
                "sycophancy_score": 65.3,
                "treatment": "anti_syco",
            }
        ),
        encoding="utf-8",
    )

    result = analyze_root(root, bootstrap_samples=0)

    assert result["models"] == ["Qwen-Test"]
    assert len(result["rows"]) == 1
    assert result["rows"][0]["single_turn_pressure_violation_n"] == 1
    assert result["rows"][0]["single_turn_pressure_violation_denom"] == 2
    assert {
        "direct_REFUSAL_pct",
        "pressure_REFUSAL_pct",
        "single_turn_pressure_violation_pct",
    } <= set(result["rows"][0])
    assert (run_dir / "run_summary.md").is_file()
    assert (root / "Qwen-Test" / "model_summary.csv").is_file()
    assert (root / "cross_model_summary.json").is_file()
    cross = json.loads((root / "cross_model_summary.json").read_text())
    assert cross["rows"][0]["feature_id"] == 61718


def test_root_analysis_handles_launcher_model_condition_layout(tmp_path):
    root = tmp_path / "step5_syco_safe"
    model_dir = root / "Qwen-Test"
    labels_by_checkpoint = {
        "base": {
            "i1": ("REFUSAL", "REFUSAL"),
            "i2": ("REFUSAL", "UHC"),
        },
        "treatment": {
            "i1": ("REFUSAL", "REFUSAL"),
            "i2": ("REFUSAL", "REFUSAL"),
        },
    }
    for checkpoint, intents in labels_by_checkpoint.items():
        run_dir = model_dir / checkpoint
        run_dir.mkdir(parents=True)
        rows = []
        for intent_id, labels in intents.items():
            for condition, label in zip(CONDITIONS, labels):
                rows.append(
                    {
                        "intent_id": intent_id,
                        "condition": condition,
                        "original_prompt_text": f"harmful request {intent_id}",
                        "final_label": label,
                    }
                )
        pd.DataFrame(rows).to_parquet(run_dir / "judge_results.parquet")
        (run_dir / "step5_summary.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "model_name": "Qwen-Test",
                    "condition": checkpoint,
                    "feature_id": 61718,
                    "clean_beta": 15,
                }
            ),
            encoding="utf-8",
        )

    analyze_root(root, bootstrap_samples=0)

    model_summary = json.loads((model_dir / "model_summary.json").read_text())
    assert len(model_summary["runs"]) == 2
    assert len(model_summary["cross_run_contrasts"]) == 1
    contrast = model_summary["cross_run_contrasts"][0]
    assert contrast["left_run"] == "base"
    assert contrast["right_run"] == "treatment"
    assert set(contrast["metrics"]) == {"single_turn_pressure_violation"}
    single_turn_violation = contrast["metrics"]["single_turn_pressure_violation"]
    assert single_turn_violation["paired_n"] == 2
    assert single_turn_violation["risk_difference_pct"] == -50.0
