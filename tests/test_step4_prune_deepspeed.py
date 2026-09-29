import json
from pathlib import Path

import pytest

from src import step4_prune_deepspeed


METRIC_VERSION = "syco_outcome_partition_v1"


def _write(path: Path, content: bytes = b"ok") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def _complete_run(tmp_path: Path) -> tuple[Path, Path]:
    train = tmp_path / "train" / "run"
    evaluation = tmp_path / "eval" / "run"
    train.mkdir(parents=True)
    (train / "training_summary.json").write_text(
        json.dumps({"full_model_save": {"full_model_eval_ready": True}}),
        encoding="utf-8",
    )
    (train / "config.json").write_text("{}\n", encoding="utf-8")
    _write(train / "pytorch_model.bin", b"hf-weights")
    _write(train / "deepspeed_checkpoint" / "latest", b"final\n")
    _write(train / "deepspeed_checkpoint" / "final" / "model_states.pt", b"model")
    _write(train / "deepspeed_checkpoint" / "final" / "optim_states.pt", b"optimizer")

    required = (
        "eval_metadata.json",
        "syco/inference_manifest.json",
        "syco/model_outputs.parquet",
        "syco/judge_results.parquet",
        "syco/summary.md",
        "harmful/inference_manifest.json",
        "harmful/model_outputs.parquet",
        "harmful/judge_results.parquet",
        "harmful/summary.md",
    )
    for relative in required:
        _write(evaluation / relative)
    _write(
        evaluation / "run_summary.md",
        (
            f"{METRIC_VERSION}\n"
            f"{step4_prune_deepspeed.HARMFUL_METRIC_VERSION}\n"
            "refusal%\nuncertain%\n"
        ).encode(),
    )
    return train, evaluation


def test_dry_run_validates_without_changing_checkpoint(tmp_path):
    train, evaluation = _complete_run(tmp_path)

    result = step4_prune_deepspeed.prune_run(
        train,
        evaluation,
        syco_metric_version=METRIC_VERSION,
        dry_run=True,
    )

    assert result["status"] == "dry_run_validated"
    assert result["checkpoint_allocated_bytes"] > 0
    assert (train / "deepspeed_checkpoint").is_dir()
    assert not (train / "deepspeed_prune_manifest.json").exists()


def test_prune_removes_only_deepspeed_and_records_manifest(tmp_path):
    train, evaluation = _complete_run(tmp_path)

    result = step4_prune_deepspeed.prune_run(
        train,
        evaluation,
        syco_metric_version=METRIC_VERSION,
    )

    assert result["status"] == "deleted"
    assert result["reclaimed_bytes"] > 0
    assert not (train / "deepspeed_checkpoint").exists()
    assert (train / "pytorch_model.bin").read_bytes() == b"hf-weights"
    assert (evaluation / "syco/judge_results.parquet").is_file()
    manifest = json.loads((train / "deepspeed_prune_manifest.json").read_text())
    assert manifest["status"] == "deleted"
    assert manifest["latest_tag"] == "final"
    assert manifest["reclaimed_bytes"] == result["reclaimed_bytes"]
    training_summary = json.loads((train / "training_summary.json").read_text())
    full_save = training_summary["full_model_save"]
    assert full_save["full_model_eval_ready"] is True
    assert full_save["deepspeed_checkpoint_dir"] == ""
    assert full_save["deepspeed_checkpoint_ready"] is False
    assert full_save["deepspeed_checkpoint_pruned"] is True


def test_prune_blocks_incomplete_evaluation(tmp_path):
    train, evaluation = _complete_run(tmp_path)
    (evaluation / "harmful/judge_results.parquet").unlink()

    with pytest.raises(step4_prune_deepspeed.PruneBlocked, match="evaluation is incomplete"):
        step4_prune_deepspeed.prune_run(
            train,
            evaluation,
            syco_metric_version=METRIC_VERSION,
        )

    assert (train / "deepspeed_checkpoint").is_dir()
    assert not (train / "deepspeed_prune_manifest.json").exists()


def test_prune_blocks_when_training_process_still_references_run(tmp_path, monkeypatch):
    train, evaluation = _complete_run(tmp_path)
    monkeypatch.setattr(
        step4_prune_deepspeed,
        "active_processes_for_run",
        lambda _: [{"pid": "123", "command": "step4_vaccine.py --output-dir run"}],
    )

    with pytest.raises(step4_prune_deepspeed.PruneBlocked, match="active process"):
        step4_prune_deepspeed.prune_run(
            train,
            evaluation,
            syco_metric_version=METRIC_VERSION,
        )

    assert (train / "deepspeed_checkpoint").is_dir()


def test_active_process_detection_covers_training_and_evaluation_commands():
    source = Path(step4_prune_deepspeed.__file__).read_text(encoding="utf-8")
    assert "step4_vaccine.py" in source
    assert "step4_posttrain_eval.py" in source
    assert "run_step4_syco_train.sh" in source
    assert "run_step4_syco_eval.sh" in source


def test_model_entrypoints_enable_post_eval_pruning():
    repo_root = Path(__file__).resolve().parents[1]
    entrypoints = (
        "run_step4_syco_train&eval_2b.sh",
        "run_step4_syco_train&eval_9b.sh",
        "run_step4_syco_train&eval_35a3b.sh",
    )
    for name in entrypoints:
        body = (repo_root / "run_scripts" / name).read_text(encoding="utf-8")
        assert 'PRUNE_DEEPSPEED_AFTER_EVAL="${PRUNE_DEEPSPEED_AFTER_EVAL:-1}"' in body
        assert 'KEEP_DEEPSPEED_CHECKPOINT="${KEEP_DEEPSPEED_CHECKPOINT:-0}"' in body

    common = (repo_root / "run_scripts/run_step4_syco_train_eval_common.sh").read_text(
        encoding="utf-8"
    )
    # A checkpoint can be pruned after a fresh evaluation, when a completed
    # evaluation is skipped, or when split-stage judging is already complete.
    assert common.count('prune_deepspeed_after_eval "$run_name"') == 3
    assert "src/step4_prune_deepspeed.py" in common
    assert 'if [[ "$EVAL_MODE" != "all" ]]' in common
    assert "both syco and harmful results are required" in common


def test_35b_entrypoint_uses_all_hosts_and_releases_gpus_before_judging():
    repo_root = Path(__file__).resolve().parents[1]
    entrypoint = (
        repo_root / "run_scripts" / "run_step4_syco_train&eval_35a3b.sh"
    ).read_text(encoding="utf-8")
    common = (repo_root / "run_scripts/run_step4_syco_train_eval_common.sh").read_text(
        encoding="utf-8"
    )

    assert 'EVAL_NNODES="${EVAL_NNODES:-$TRAIN_NNODES}"' in entrypoint
    assert 'MAX_PARALLEL_EVALS="${MAX_PARALLEL_EVALS:-8}"' in entrypoint
    assert 'MAX_PARALLEL_JUDGES="${MAX_PARALLEL_JUDGES:-8}"' in entrypoint
    assert 'EVAL_SPLIT_INFERENCE_JUDGE="${EVAL_SPLIT_INFERENCE_JUDGE:-1}"' in entrypoint
    assert "run_eval_inference" in common
    assert "run_eval_judge" in common
    assert "--skip-judge" in common
    assert "--skip-inference" in common
    assert '"EXPERIMENT_TAG=$EXPERIMENT_TAG"' in common

    eval_wrapper = (repo_root / "run_scripts/run_step4_syco_eval.sh").read_text(
        encoding="utf-8"
    )
    assert 'basename "${OUTPUT_ROOT%/}"' in eval_wrapper
    assert 'OUTPUT_MODEL_DIR="$OUTPUT_ROOT"' in eval_wrapper
