# Execution notes

This source release separates model computation from lightweight analysis. The README summarizes the study; this page records what is needed to execute the code.

## Environment

Use Python 3.10 or newer. Install `requirements-analysis.txt` for the plotting and checkpoint-selection tools. Install `requirements.txt` for the additional feature-extraction and training imports, then install **vLLM** for generation and **DeepSpeed** when using its distributed training path.

```bash
python -m pip install -r requirements.txt
# In your Linux/CUDA environment, select compatible backend versions:
python -m pip install vllm deepspeed
```

The dependency files are an inventory inferred from imports, not the original experiment's environment lock. Match PyTorch, CUDA, Transformers, vLLM, and DeepSpeed versions to the model and hardware. The source contains vLLM internal-API compatibility code, so backend upgrades need validation. GPU training and inference have not been validated as part of this repository packaging.

Use the corresponding base checkpoints: [Qwen3.5-2B-Base](https://huggingface.co/Qwen/Qwen3.5-2B-Base), [Qwen3.5-9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base), or [Qwen3.5-35B-A3B-Base](https://huggingface.co/Qwen/Qwen3.5-35B-A3B-Base). Supply a compatible SAE for the same model, residual-stream location, and layer.

## Feature discovery

`src/step2_syco_feature.py` has its own CLI. The supplied `configs/step2.example.yaml` reflects its parser and defaults; it is an illustrative configuration, not a recovered experiment configuration.

Paired JSONL data uses two rows per prompt, with matching `id`, `domain`, and `prompt`:

```jsonl
{"id":"example-001","domain":"factual","prompt":"Is 2 + 2 equal to 5?","response":"Yes, you are right.","response_type":"sycophantic"}
{"id":"example-001","domain":"factual","prompt":"Is 2 + 2 equal to 5?","response":"No. 2 + 2 equals 4.","response_type":"robust"}
```

These two rows illustrate the schema only. They are not research data. The `robust` response type is the code's name for an independent response.

Set `MODEL_PATH`, `SAE_DIR`, `SAE_LAYER`, and `PAIRED_DATA` for your files, then validate the pairs:

```bash
python -m src.step2_syco_feature \
  --config configs/step2.example.yaml \
  --model-name Qwen3.5-2B-Base \
  --model-path "${MODEL_PATH:?Set MODEL_PATH}" \
  --sae-dir "${SAE_DIR:?Set SAE_DIR}" \
  --layer "${SAE_LAYER:?Set SAE_LAYER to an integer}" \
  --dataset-path "${PAIRED_DATA:?Set PAIRED_DATA}" \
  --validate-only
```

`--validate-only` validates the paired records and configuration resolution. It does **not** load or validate model/SAE weights. Remove that flag to perform feature extraction; `--dry-run` still loads the model and processes a small number of pairs.

SAE files are resolved as `SAE_DIR/layer<LAYER>.ae.pt`. Discovery expects `encoder.weight`/`encoder.bias` or `W_enc`/`b_enc`, plus `k` in the checkpoint or a configured `top_k`. Steering expects `decoder.weight` or `W_dec` with shape `[d_model, n_features]`. Check these conventions before substituting a different SAE implementation.

## CFI and evaluation

- `src/step4_vaccine.py` implements training-time injection, full tuning, LoRA, distributed training, and model export. Its historical filename is retained; its docstring describes the CFI implementation.
- `src/step5_select_alpha.py` selects positive and negative checkpoints independently using sycophancy calibration outcomes, quality gates, and artifact checks.
- `src/step5_syco_safe_analyse.py` compares paired harmful intents across `direct` and `pressure` conditions. It requires the missing `step1_judge` module listed below.
- `src/step5_syco_safe_figure.py` plots existing analysis outputs. It does not generate experiment results itself.

The historical shell launchers encode experiment-specific paths, GPU selections, feature IDs, and group names. Inspect their arguments and restore their dependencies before launching them. No training job or judge API call is needed for the lightweight checks below.

## Lightweight checks

These tests cover endpoint selection and checkpoint-cleanup safeguards without loading models:

```bash
python -m pytest -q \
  tests/test_step5_select_alpha.py \
  tests/test_step4_prune_deepspeed.py \
  -k 'not test_model_entrypoints_enable_post_eval_pruning and not test_35b_entrypoint_uses_all_hosts_and_releases_gpus_before_judging'
```

The two excluded tests inspect launchers absent from the archive. The broader suite also references unreleased modules and the GPU/vLLM stack; passing this subset does not certify the complete experiment pipeline.

## Release inventory

The source archive contains 53 Python/shell files. Packaging adds documentation, dependency inventories, citation metadata, artwork, an example configuration, and Git ignore rules. The supplied research implementations are preserved.

The following referenced components are **not included**:

| Component | Missing items |
| :--- | :--- |
| Baseline judging and orchestration | `src/step1_judge.py`, `src/step1_pipeline.py` |
| Steering orchestration | `src/step3_pipeline.py` |
| Post-training evaluation and data generation | `src/step4_posttrain_eval.py`, `src/step4_generate_syco_sft_responses_api.py` |
| Pressure evaluation and recovery reporting | `src/step5_syco_safe.py`, `src/step5_margin_recovery.py` |
| Original configuration files | `configs/model.yaml`, `judge.yaml`, `data_bench.yaml`, `step2.yaml`, `step4_vaccine.yaml`, and referenced DeepSpeed JSON files |
| Launcher dependencies | `run_step4_syco_dataset.sh`, `run_step4_syco_eval.sh`, `run_step4_syco_train_eval_common.sh`, and the `run_step4_syco_train&eval_{2b,9b,35a3b}.sh` wrappers |
| Research artifacts | Query/response datasets, model and SAE weights, trained checkpoints, evaluation outputs, and the original environment lockfile |

Restore the original components to reproduce the full experiment. In particular, substituting a new judge or pressure protocol changes the experiment and should be reported as a new setup.
