"""Run the frozen, common-anchor 2B experiment, one candidate per GPU."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parents[1]
MODEL = "Qwen3.5-2B-Base"
OUT = ROOT / "outputs/step5_syco_safe" / MODEL / "anchor_matched_20260923"
PYTHON = "python"
BASE = "./models/" + MODEL


def logged(command, name, stage, env):
    path = OUT / "logs" / f"{name}.{stage}.log"
    with path.open("a") as handle:
        handle.write(f"\n[{time.strftime('%F %T')}] {command!r}\n")
        handle.flush()
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"{name}/{stage} exit={result.returncode}: {path}")
    print(f"{time.strftime('%F %T')} DONE {name} {stage}", flush=True)


def one(candidate, plan):
    name = candidate["name"]
    artifact = Path(candidate["artifact"])
    env = dict(os.environ, PYTHON_BIN=PYTHON, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               TOKENIZERS_PARALLELISM="false", CUDA_VISIBLE_DEVICES=str(candidate["gpu"]),
               STEP4_LOG_DIR=str(OUT / "logs" / name),
               TORCH_EXTENSIONS_DIR=f"/tmp/step5_anchor_matched_torch_{candidate['gpu']}")
    train = ["bash", "run_scripts/run_step4_syco_train.sh", "--config", str(OUT / "training_config.yaml"),
             "--model", MODEL, "--model-path", plan["anchor"],
             "--group", "syco_sft" if candidate["alpha"] == 0 else "syco_sft_prevent",
             "--output-dir", str(artifact), "--run-name", name,
             "--shared-syco-dataset-path", str(OUT / "data/syco_dataset.jsonl"),
             "--syco-split-dir", str(OUT / "data/training_split"),
             "--syco-split-train-size", "1800", "--syco-split-eval-size", "400", "--syco-split-seed", "1234",
             "--injection-targets", "syco", "--beta", str(candidate["alpha"]), "--beta-schedule", "fixed",
             "--epochs", "1", "--learning-rate", "2e-6", "--batch-size", "1", "--global-batch-size", "8",
             "--warmup-ratio", "0.03", "--max-length", "512", "--full-model-export", "hf",
             "--tuning-mode", "full", "--gpu", str(candidate["gpu"]), "--num-gpus", "1",
             "--no-deepspeed", "--no-gradient-checkpointing", "--skip-completed"]
    summary = artifact / "training_summary.json"
    complete = summary.is_file() and json.loads(summary.read_text()).get("full_model_save", {}).get("full_model_eval_ready")
    if not complete:
        logged(train, name, "train", env)
    evaluation = OUT / "syco_evaluation" / MODEL / name / "syco"
    if not (evaluation / "model_outputs.parquet").is_file():
        generate = [PYTHON, "src/step4_posttrain_eval.py", "--model-name", MODEL, "--model-path", BASE,
                    "--adapter-dir", str(artifact), "--run-name", name,
                    "--output-root", str(OUT / "syco_evaluation"), "--eval-mode", "syco",
                    "--syco-eval-path", str(OUT / "data/training_split/syco_eval.jsonl"),
                    "--tensor-parallel-size", "1", "--gpu-memory-utilization", "0.65",
                    "--max-model-len", "4096", "--batch-size", "128", "--max-new-tokens", "256",
                    "--temperature", "0", "--top-p", "1", "--no-do-sample", "--generation-seed", "1234",
                    "--inference-backend", "vllm", "--vllm-enforce-eager", "--skip-judge"]
        logged(generate, name, "generate", env)
    if not (evaluation / "summary.md").is_file():
        judge = [PYTHON, "src/step5_margin_recovery.py", "syco-judge", "--eval-dir", str(evaluation),
                 "--workers", "16", "--max-retries", "10", "--timeout", "180"]
        logged(judge, name, "judge", env)
    if not all((evaluation / filename).is_file() for filename in ["model_outputs.parquet", "judge_results.parquet", "summary.md"]):
        raise RuntimeError(f"Incomplete syco evaluation: {evaluation}")
    return name


def main():
    plan = json.loads((OUT / "experiment_plan.json").read_text())
    errors = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(one, candidate, plan): candidate["name"] for candidate in plan["candidates"]}
        for future in as_completed(futures):
            try:
                print("CANDIDATE_COMPLETE", future.result(), flush=True)
            except Exception as exc:
                errors.append(str(exc))
                print("CANDIDATE_FAILED", futures[future], str(exc), flush=True)
    (OUT / "pipeline_status.json").write_text(json.dumps({"stage": "syco_complete" if not errors else "failed", "errors": errors}, indent=2) + "\n")
    return bool(errors)


if __name__ == "__main__":
    sys.exit(main())
