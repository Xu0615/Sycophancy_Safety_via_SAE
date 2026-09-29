#!/usr/bin/env python3
"""Safely remove a completed Step 4 DeepSpeed checkpoint.

The HF export is the evaluation artifact.  The DeepSpeed directory is kept
only until the export and the requested evaluation outputs have been checked.
This command is deliberately implemented with the standard library so that a
cleanup failure cannot depend on the model or inference environments.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any


INDEX_NAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
DIRECT_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin")
SYCO_METRIC_VERSIONS = {
    "syco_outcome_partition_v1",
    "syco_outcome_partition_v2_stance",
}
HARMFUL_METRIC_VERSION = "sorrybench_scope_payload_first_exclusive_repeat_v3"
COMMON_EVAL_FILES = ("run_summary.md", "eval_metadata.json")
EVAL_FILES = {
    "syco": (
        "syco/inference_manifest.json",
        "syco/model_outputs.parquet",
        "syco/judge_results.parquet",
        "syco/summary.md",
    ),
    "harmful": (
        "harmful/inference_manifest.json",
        "harmful/model_outputs.parquet",
        "harmful/judge_results.parquet",
        "harmful/summary.md",
    ),
}


class PruneBlocked(RuntimeError):
    """A validation failure that must leave the checkpoint untouched."""


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PruneBlocked(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PruneBlocked(f"expected an object in {path}")
    return value


def _is_nonempty_file(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def _safe_child(root: Path, relative_name: str) -> Path:
    """Resolve a weight shard while rejecting paths outside the run."""

    candidate = (root / relative_name).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise PruneBlocked(f"weight path escapes the run directory: {relative_name}") from exc
    return candidate


def validate_hf_export(train_run_dir: Path) -> dict[str, Any]:
    summary_path = train_run_dir / "training_summary.json"
    if not _is_nonempty_file(summary_path):
        raise PruneBlocked(f"missing training summary: {summary_path}")
    summary = _read_json(summary_path)
    full_save = summary.get("full_model_save")
    if not isinstance(full_save, dict) or full_save.get("full_model_eval_ready") is not True:
        raise PruneBlocked("training_summary.json does not declare full_model_eval_ready=true")

    config_path = train_run_dir / "config.json"
    if not _is_nonempty_file(config_path):
        raise PruneBlocked(f"missing HF config: {config_path}")

    weight_files: list[Path] = []
    index_path: Path | None = None
    for name in INDEX_NAMES:
        candidate = train_run_dir / name
        if candidate.exists():
            index_path = candidate
            index = _read_json(candidate)
            weight_map = index.get("weight_map")
            if not isinstance(weight_map, dict) or not weight_map:
                raise PruneBlocked(f"empty or invalid weight_map: {candidate}")
            names = sorted({str(value) for value in weight_map.values() if str(value).strip()})
            if not names:
                raise PruneBlocked(f"weight_map has no shard names: {candidate}")
            weight_files = [_safe_child(train_run_dir, name) for name in names]
            break
    if index_path is None:
        weight_files = [
            train_run_dir / name
            for name in DIRECT_WEIGHT_NAMES
            if (train_run_dir / name).exists()
        ]
        if not weight_files:
            raise PruneBlocked(f"no HF weight file or index found in {train_run_dir}")

    missing = [str(path) for path in weight_files if not _is_nonempty_file(path)]
    if missing:
        raise PruneBlocked(f"missing or empty HF weight shards: {', '.join(missing)}")

    return {
        "config": str(config_path),
        "index": str(index_path) if index_path else "",
        "weight_files": [
            {"path": str(path), "bytes": path.stat().st_size} for path in weight_files
        ],
    }


def _required_eval_files(eval_mode: str) -> tuple[str, ...]:
    if eval_mode not in {"all", "syco", "harmful"}:
        raise PruneBlocked(f"unsupported eval mode: {eval_mode}")
    files = list(COMMON_EVAL_FILES)
    if eval_mode in {"all", "syco"}:
        files.extend(EVAL_FILES["syco"])
    if eval_mode in {"all", "harmful"}:
        files.extend(EVAL_FILES["harmful"])
    return tuple(files)


def validate_evaluation(
    eval_run_dir: Path,
    eval_mode: str,
    syco_metric_version: str,
) -> dict[str, Any]:
    required = _required_eval_files(eval_mode)
    missing = [
        relative
        for relative in required
        if not _is_nonempty_file(eval_run_dir / relative)
    ]
    if missing:
        raise PruneBlocked(
            f"evaluation is incomplete under {eval_run_dir}: {', '.join(missing)}"
        )

    summary_text = (eval_run_dir / "run_summary.md").read_text(encoding="utf-8")
    if eval_mode in {"all", "syco"}:
        if syco_metric_version not in SYCO_METRIC_VERSIONS:
            raise PruneBlocked(f"unsupported sycophancy metric version: {syco_metric_version}")
        if syco_metric_version not in summary_text:
            raise PruneBlocked(
                f"run_summary.md does not contain the expected metric version {syco_metric_version}"
            )
    if eval_mode in {"all", "harmful"}:
        if HARMFUL_METRIC_VERSION not in summary_text:
            raise PruneBlocked(
                "run_summary.md does not contain the expected harmful metric "
                f"version {HARMFUL_METRIC_VERSION}"
            )
        for marker in ("refusal%", "uncertain%"):
            if marker not in summary_text:
                raise PruneBlocked(f"run_summary.md is missing the harmful marker {marker}")

    return {
        "directory": str(eval_run_dir),
        "mode": eval_mode,
        "required_files": list(required),
        "metric_version": syco_metric_version if eval_mode in {"all", "syco"} else "",
    }


def active_processes_for_run(train_run_dir: Path) -> list[dict[str, str]]:
    """Find any live command that still names this run directory."""

    matches: list[dict[str, str]] = []
    run_text = str(train_run_dir.resolve())
    proc_root = Path("/proc")
    if not proc_root.is_dir():
        return matches
    own_pid = str(os.getpid())
    for proc in proc_root.iterdir():
        if not proc.name.isdigit() or proc.name == own_pid:
            continue
        try:
            raw = (proc / "cmdline").read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        command = raw.replace(b"\0", b" ").decode("utf-8", "replace").strip()
        # Ignore an unrelated shell that merely happens to mention the path
        # (for example a dry-run wrapper).  Only training/evaluation commands
        # can race with removal of the checkpoint.
        command_is_relevant = any(
            marker in command
            for marker in (
                "step4_vaccine.py",
                "step4_posttrain_eval.py",
                "run_step4_syco_train.sh",
                "run_step4_syco_eval.sh",
            )
        )
        if run_text in command and command_is_relevant:
            matches.append({"pid": proc.name, "command": command})
    return matches


def _disk_usage_bytes(root: Path) -> int:
    """Return allocated bytes, counting hard-linked inodes only once."""

    total = 0
    seen: set[tuple[int, int]] = set()
    for directory, _, filenames in os.walk(root, followlinks=False):
        paths = [Path(directory), *(Path(directory) / filename for filename in filenames)]
        for path in paths:
            try:
                stat = path.stat()
            except FileNotFoundError:
                continue
            identity = (stat.st_dev, stat.st_ino)
            if identity in seen:
                continue
            seen.add(identity)
            blocks = getattr(stat, "st_blocks", 0)
            total += int(blocks) * 512 if blocks else int(stat.st_size)
    return total


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def mark_checkpoint_pruned(
    train_run_dir: Path,
    manifest_path: Path,
    deleted_at_utc: str,
    reclaimed_bytes: int,
) -> None:
    updates: dict[str, Any] = {
        "deepspeed_checkpoint_dir": "",
        "deepspeed_checkpoint_ready": False,
        "deepspeed_checkpoint_pruned": True,
        "deepspeed_prune_manifest": str(manifest_path),
        "deepspeed_pruned_at_utc": deleted_at_utc,
        "deepspeed_reclaimed_bytes": reclaimed_bytes,
    }
    save_summary_path = train_run_dir / "full_model_save_summary.json"
    if save_summary_path.is_file():
        save_summary = _read_json(save_summary_path)
        save_summary.update(updates)
        _atomic_write_json(save_summary_path, save_summary)

    training_summary_path = train_run_dir / "training_summary.json"
    training_summary = _read_json(training_summary_path)
    full_save = training_summary.get("full_model_save")
    if not isinstance(full_save, dict):
        raise PruneBlocked(f"missing full_model_save object in {training_summary_path}")
    full_save.update(updates)
    training_summary["full_model_save"] = full_save
    _atomic_write_json(training_summary_path, training_summary)


def prune_run(
    train_run_dir: Path,
    eval_run_dir: Path,
    eval_mode: str = "all",
    syco_metric_version: str = "syco_outcome_partition_v1",
    dry_run: bool = False,
) -> dict[str, Any]:
    train_run_dir = train_run_dir.resolve()
    eval_run_dir = eval_run_dir.resolve()
    checkpoint_dir = train_run_dir / "deepspeed_checkpoint"
    if not train_run_dir.is_dir():
        raise PruneBlocked(f"training run directory does not exist: {train_run_dir}")
    if checkpoint_dir.is_symlink():
        raise PruneBlocked(f"refusing to prune symlinked checkpoint: {checkpoint_dir}")
    if not checkpoint_dir.is_dir():
        return {
            "status": "already_absent",
            "train_run_dir": str(train_run_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "reclaimed_bytes": 0,
        }
    if not _is_nonempty_file(checkpoint_dir / "latest"):
        raise PruneBlocked(f"DeepSpeed checkpoint has no latest tag: {checkpoint_dir}")

    hf = validate_hf_export(train_run_dir)
    evaluation = validate_evaluation(eval_run_dir, eval_mode, syco_metric_version)
    active = active_processes_for_run(train_run_dir)
    if active:
        rendered = "; ".join(f"pid={item['pid']} {item['command']}" for item in active)
        raise PruneBlocked(f"active process still references this run: {rendered}")

    reclaimed_bytes = _disk_usage_bytes(checkpoint_dir)
    manifest_path = train_run_dir / "deepspeed_prune_manifest.json"
    base_manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "validated_pending_delete",
        "validated_at_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "train_run_dir": str(train_run_dir),
        "eval_run_dir": str(eval_run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "latest_tag": (checkpoint_dir / "latest").read_text(encoding="utf-8").strip(),
        "eval": evaluation,
        "hf_export": hf,
        "checkpoint_allocated_bytes": reclaimed_bytes,
    }
    if dry_run:
        base_manifest["status"] = "dry_run_validated"
        return base_manifest

    lock_dir = train_run_dir / ".deepspeed_prune.lock"
    try:
        lock_dir.mkdir()
    except FileExistsError as exc:
        raise PruneBlocked(f"another prune operation owns {lock_dir}") from exc

    try:
        active = active_processes_for_run(train_run_dir)
        if active:
            rendered = "; ".join(f"pid={item['pid']} {item['command']}" for item in active)
            raise PruneBlocked(f"active process appeared before deletion: {rendered}")
        if not checkpoint_dir.is_dir() or not _is_nonempty_file(checkpoint_dir / "latest"):
            raise PruneBlocked(f"checkpoint changed after validation: {checkpoint_dir}")
        # Keep the validation record outside the directory being removed.  If
        # deletion is interrupted, the pending status makes the next audit
        # conservative instead of silently claiming success.
        _atomic_write_json(manifest_path, base_manifest)
        try:
            shutil.rmtree(checkpoint_dir)
        except Exception as exc:
            failed = dict(base_manifest)
            failed.update(
                {
                    "status": "delete_failed",
                    "delete_error": repr(exc),
                    "failed_at_utc": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
                }
            )
            _atomic_write_json(manifest_path, failed)
            raise
        deleted_at_utc = _datetime.datetime.now(_datetime.timezone.utc).isoformat()
        completed = dict(base_manifest)
        completed.update(
            {
                "status": "deleted",
                "deleted_at_utc": deleted_at_utc,
                "reclaimed_bytes": reclaimed_bytes,
            }
        )
        _atomic_write_json(manifest_path, completed)
        try:
            mark_checkpoint_pruned(
                train_run_dir,
                manifest_path,
                deleted_at_utc,
                reclaimed_bytes,
            )
        except Exception as exc:
            completed.update(
                {
                    "status": "deleted_metadata_update_failed",
                    "metadata_update_error": repr(exc),
                }
            )
            _atomic_write_json(manifest_path, completed)
            raise
        return completed
    finally:
        try:
            lock_dir.rmdir()
        except FileNotFoundError:
            pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-run-dir", required=True, type=Path)
    parser.add_argument("--eval-run-dir", required=True, type=Path)
    parser.add_argument("--eval-mode", choices=("all", "syco", "harmful"), default="all")
    parser.add_argument(
        "--syco-metric-version",
        default="syco_outcome_partition_v1",
        choices=tuple(sorted(SYCO_METRIC_VERSIONS)),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = prune_run(
            args.train_run_dir,
            args.eval_run_dir,
            eval_mode=args.eval_mode,
            syco_metric_version=args.syco_metric_version,
            dry_run=args.dry_run,
        )
    except PruneBlocked as exc:
        print(f"[prune:blocked] {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"[prune:error] {exc}", file=sys.stderr)
        return 1

    status = result.get("status", "unknown")
    reclaimed = int(result.get("reclaimed_bytes", result.get("checkpoint_allocated_bytes", 0)))
    print(
        f"[prune:{status}] train_run={result.get('train_run_dir')} "
        f"checkpoint_bytes={reclaimed}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
