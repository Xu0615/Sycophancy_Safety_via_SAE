"""
Step 1 — Data loaders for attack benchmark datasets.

Each loader reads its raw format and returns a list of dicts with the unified
schema.  ``load_and_unify_all`` merges everything, deduplicates, and optionally
filters to harmful-only prompts.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from src.utils import setup_logger, stable_hash, save_parquet

logger = setup_logger("step1_data_loader")


# ====================================================================
# Unified schema
# ====================================================================

UNIFIED_COLUMNS = [
    "sample_id",
    "source_dataset",
    "source_split",
    "category",
    "subcategory",
    "prompt_text",
    "context_text",
    "target_text",
    "is_harmful",
    "language",
    "metadata",
]

DEFAULT_BENCHMARK_DATASETS = [
    "advbench",
    "harmbench",
    "jbb_behaviors",
    "sorrybench",
]


# ====================================================================
# Individual dataset loaders
# ====================================================================

def load_advbench(data_dir: str) -> List[Dict[str, Any]]:
    logger.info(f"Loading AdvBench from {data_dir}")
    records: List[Dict[str, Any]] = []

    candidates = ["advbench.csv", "harmful_behaviors.csv", "dataset.csv"]
    df = None
    for fname in candidates:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            df = pd.read_csv(fpath)
            logger.info(f"  Found {fpath} with {len(df)} rows")
            break
    if df is None:
        for fname in ["advbench.parquet", "train.parquet", "data.parquet"]:
            fpath = os.path.join(data_dir, fname)
            if os.path.exists(fpath):
                df = pd.read_parquet(fpath)
                logger.info(f"  Found {fpath} with {len(df)} rows")
                break
    if df is None:
        logger.warning(f"  No AdvBench data files found in {data_dir}")
        return records

    for _, row in df.iterrows():
        goal = str(row.get("goal", "")).strip()
        if not goal:
            continue
        records.append({
            "sample_id": f"advbench_{stable_hash(goal)}",
            "source_dataset": "advbench",
            "source_split": None,
            "category": None,
            "subcategory": None,
            "prompt_text": goal,
            "context_text": None,
            "target_text": str(row.get("target", "")) if pd.notna(row.get("target")) else None,
            "is_harmful": True,
            "language": "en",
            "metadata": {},
        })

    logger.info(f"  AdvBench: loaded {len(records)} samples")
    return records


def load_harmbench(data_dir: str) -> List[Dict[str, Any]]:
    logger.info(f"Loading HarmBench from {data_dir}")
    records: List[Dict[str, Any]] = []

    all_path = os.path.join(data_dir, "harmbench_behaviors_text_all.csv")
    if os.path.exists(all_path):
        df = pd.read_csv(all_path)
        logger.info(f"  Found all file with {len(df)} rows")
    else:
        dfs = []
        for split_name in ["test", "val"]:
            fpath = os.path.join(data_dir, f"harmbench_behaviors_text_{split_name}.csv")
            if os.path.exists(fpath):
                tmp = pd.read_csv(fpath)
                tmp["_split"] = split_name
                dfs.append(tmp)
                logger.info(f"  Found {split_name} split with {len(tmp)} rows")
        if not dfs:
            for fname in ["harmbench.parquet", "train.parquet", "data.parquet"]:
                fpath = os.path.join(data_dir, fname)
                if os.path.exists(fpath):
                    df = pd.read_parquet(fpath)
                    logger.info(f"  Found {fpath} with {len(df)} rows")
                    break
            else:
                logger.warning(f"  No HarmBench data files found in {data_dir}")
                return records
        else:
            df = pd.concat(dfs, ignore_index=True)

    for _, row in df.iterrows():
        behavior = str(row.get("Behavior", "")).strip()
        if not behavior:
            continue
        func_cat = str(row.get("FunctionalCategory", "")) if pd.notna(row.get("FunctionalCategory")) else None
        sem_cat = str(row.get("SemanticCategory", "")) if pd.notna(row.get("SemanticCategory")) else None
        context = str(row.get("ContextString", "")) if pd.notna(row.get("ContextString")) else None
        split = str(row.get("_split", "")) if "_split" in row.index else None

        records.append({
            "sample_id": f"harmbench_{stable_hash(behavior)}",
            "source_dataset": "harmbench",
            "source_split": split,
            "category": func_cat,
            "subcategory": sem_cat,
            "prompt_text": behavior,
            "context_text": context,
            "target_text": None,
            "is_harmful": True,
            "language": "en",
            "metadata": {
                "behavior_id": str(row.get("BehaviorID", "")) if pd.notna(row.get("BehaviorID")) else None,
                "tags": str(row.get("Tags", "")) if pd.notna(row.get("Tags")) else None,
            },
        })

    logger.info(f"  HarmBench: loaded {len(records)} samples")
    return records


def load_sorrybench(data_dir: str, base_only: bool = True) -> List[Dict[str, Any]]:
    logger.info(f"Loading sorry-bench from {data_dir} (base_only={base_only})")
    records: List[Dict[str, Any]] = []

    base_path = os.path.join(data_dir, "question.jsonl")
    csv_path = os.path.join(data_dir, "sorry_bench_202406.csv")

    if os.path.exists(base_path):
        with open(base_path, "r", encoding="utf-8") as f:
            lines = [json.loads(l) for l in f if l.strip()]
        logger.info(f"  Found question.jsonl with {len(lines)} rows")

        for item in lines:
            turns = item.get("turns", [])
            prompt = turns[0] if isinstance(turns, list) and turns else str(turns)
            prompt = str(prompt).strip()
            if not prompt:
                continue
            cat = str(item.get("category", "")) if item.get("category") else None
            records.append({
                "sample_id": f"sorrybench_base_{stable_hash(prompt)}",
                "source_dataset": "sorrybench",
                "source_split": "base",
                "category": cat,
                "subcategory": str(item.get("prompt_style", "base")),
                "prompt_text": prompt,
                "context_text": None,
                "target_text": None,
                "is_harmful": True,
                "language": "en",
                "metadata": {
                    "question_id": item.get("question_id"),
                    "prompt_style": item.get("prompt_style", "base"),
                },
            })

    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        logger.info(f"  Found CSV with {len(df)} rows")
        for _, row in df.iterrows():
            prompt = str(row.get("prompt", "")).strip()
            if not prompt:
                continue
            cat = str(row.get("category", "")) if pd.notna(row.get("category")) else None
            records.append({
                "sample_id": f"sorrybench_base_{stable_hash(prompt)}",
                "source_dataset": "sorrybench",
                "source_split": "base",
                "category": cat,
                "subcategory": "base",
                "prompt_text": prompt,
                "context_text": None,
                "target_text": None,
                "is_harmful": True,
                "language": "en",
                "metadata": {},
            })
    else:
        logger.warning(f"  No sorry-bench base file found in {data_dir}")

    if not base_only:
        variant_files = sorted(Path(data_dir).glob("question_*.jsonl"))
        for vf in variant_files:
            variant_name = vf.stem.replace("question_", "")
            with open(vf, "r", encoding="utf-8") as f:
                lines = [json.loads(l) for l in f if l.strip()]
            logger.info(f"  Loading variant {variant_name}: {len(lines)} rows")
            for item in lines:
                turns = item.get("turns", [])
                prompt = turns[0] if isinstance(turns, list) and turns else str(turns)
                prompt = str(prompt).strip()
                if not prompt:
                    continue
                cat = str(item.get("category", "")) if item.get("category") else None
                records.append({
                    "sample_id": f"sorrybench_{variant_name}_{stable_hash(prompt)}",
                    "source_dataset": "sorrybench",
                    "source_split": variant_name,
                    "category": cat,
                    "subcategory": variant_name,
                    "prompt_text": prompt,
                    "context_text": None,
                    "target_text": None,
                    "is_harmful": True,
                    "language": "en",
                    "metadata": {
                        "question_id": item.get("question_id"),
                        "prompt_style": variant_name,
                    },
                })

    logger.info(f"  sorry-bench: loaded {len(records)} samples total")
    return records


def load_jbb_behaviors(data_dir: str) -> List[Dict[str, Any]]:
    logger.info(f"Loading JBB-Behaviors from {data_dir}")
    records: List[Dict[str, Any]] = []

    harm_path = os.path.join(data_dir, "harmful-behaviors.csv")
    if os.path.exists(harm_path):
        df = pd.read_csv(harm_path)
        logger.info(f"  Found harmful-behaviors.csv with {len(df)} rows")
        for _, row in df.iterrows():
            goal = str(row.get("Goal", "")).strip()
            if not goal:
                continue
            records.append({
                "sample_id": f"jbb_{stable_hash(goal)}",
                "source_dataset": "jbb_behaviors",
                "source_split": "harmful",
                "category": str(row.get("Category", "")) if pd.notna(row.get("Category")) else None,
                "subcategory": str(row.get("Behavior", "")) if pd.notna(row.get("Behavior")) else None,
                "prompt_text": goal,
                "context_text": None,
                "target_text": str(row.get("Target", "")) if pd.notna(row.get("Target")) else None,
                "is_harmful": True,
                "language": "en",
                "metadata": {
                    "source": str(row.get("Source", "")) if pd.notna(row.get("Source")) else None,
                    "index": int(row.get("Index", 0)),
                },
            })
    else:
        for fname in ["jbb_behaviors.parquet", "train.parquet", "data.parquet"]:
            fpath = os.path.join(data_dir, fname)
            if os.path.exists(fpath):
                df = pd.read_parquet(fpath)
                logger.info(f"  Found {fpath} with {len(df)} rows")
                for _, row in df.iterrows():
                    goal = str(row.get("Goal", row.get("goal", ""))).strip()
                    if not goal:
                        continue
                    records.append({
                        "sample_id": f"jbb_{stable_hash(goal)}",
                        "source_dataset": "jbb_behaviors",
                        "source_split": "harmful",
                        "category": str(row.get("Category", "")) if pd.notna(row.get("Category")) else None,
                        "subcategory": str(row.get("Behavior", "")) if pd.notna(row.get("Behavior")) else None,
                        "prompt_text": goal,
                        "context_text": None,
                        "target_text": str(row.get("Target", "")) if pd.notna(row.get("Target")) else None,
                        "is_harmful": True,
                        "language": "en",
                        "metadata": {},
                    })
                break
        else:
            logger.warning(f"  No JBB-Behaviors data found in {data_dir}")

    logger.info(f"  JBB-Behaviors: loaded {len(records)} samples")
    return records


# ====================================================================
# Unification
# ====================================================================

def _normalize_dataset_list(datasets: Optional[Sequence[str]]) -> Optional[List[str]]:
    if datasets is None:
        return None
    out = []
    for ds in datasets:
        if ds is None:
            continue
        name = str(ds).strip()
        if name:
            out.append(name)
    return out or None


def select_prompt_sample(
    df: pd.DataFrame,
    datasets: Optional[Sequence[str]] = None,
    samples_per_dataset: Optional[int] = None,
    seed: int = 1234,
) -> pd.DataFrame:
    """Filter datasets and take a deterministic random sample per dataset.

    ``samples_per_dataset`` is a cap: datasets with fewer rows are kept in full.
    The returned frame is sorted for stable downstream checkpoints.
    """
    out = df.copy()

    allowed = _normalize_dataset_list(datasets)
    if allowed is not None:
        before = len(out)
        out = out[out["source_dataset"].isin(allowed)].copy()
        missing = sorted(set(allowed) - set(out["source_dataset"].unique()))
        if missing:
            logger.warning(f"Configured dataset(s) not found: {missing}")
        logger.info(
            "After dataset filter %s: %d (removed %d)",
            allowed, len(out), before - len(out))

    if samples_per_dataset is not None:
        n = int(samples_per_dataset)
        if n <= 0:
            raise ValueError("samples_per_dataset must be positive when set")

        sampled = []
        for idx, (ds, sub) in enumerate(out.groupby("source_dataset", sort=True)):
            take = min(n, len(sub))
            if take < len(sub):
                part = sub.sample(n=take, random_state=int(seed) + idx)
            else:
                part = sub.copy()
            sampled.append(part)
            logger.info("Dataset sample %s: %d/%d", ds, len(part), len(sub))
        out = (pd.concat(sampled, ignore_index=True)
               if sampled else out.iloc[0:0].copy())

    return out.sort_values(["source_dataset", "sample_id"]).reset_index(drop=True)


def load_and_unify_all(
    data_root: str,
    harmful_only: bool = True,
    sorrybench_base_only: bool = True,
    datasets: Optional[Sequence[str]] = None,
    samples_per_dataset: Optional[int] = None,
    sample_seed: int = 1234,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """Load benchmark datasets, unify schema, deduplicate, filter, and sample."""
    all_records: List[Dict[str, Any]] = []

    loaders = [
        ("advbench", lambda: load_advbench(os.path.join(data_root, "AdvBench"))),
        ("harmbench", lambda: load_harmbench(os.path.join(data_root, "HarmBench"))),
        ("sorrybench", lambda: load_sorrybench(os.path.join(data_root, "sorry-bench-202406"), base_only=sorrybench_base_only)),
        ("jbb_behaviors", lambda: load_jbb_behaviors(os.path.join(data_root, "JBB-Behaviors", "data"))),
    ]

    requested = _normalize_dataset_list(datasets) or DEFAULT_BENCHMARK_DATASETS
    requested_set = set(requested) if requested is not None else None

    for name, loader_fn in loaders:
        if name not in requested_set:
            logger.info(f"Skipping {name}: not in configured datasets")
            continue
        try:
            records = loader_fn()
            all_records.extend(records)
            logger.info(f"Loaded {len(records)} from {name}")
        except Exception as e:
            logger.error(f"Failed to load {name}: {e}")

    df = pd.DataFrame(all_records, columns=UNIFIED_COLUMNS)
    logger.info(f"Total unified records: {len(df)}")

    before = len(df)
    df = df.drop_duplicates(subset=["prompt_text"], keep="first").reset_index(drop=True)
    logger.info(f"After dedup: {len(df)} (removed {before - len(df)} duplicates)")

    if harmful_only:
        before = len(df)
        df = df[df["is_harmful"] == True].reset_index(drop=True)
        logger.info(f"After harmful filter: {len(df)} (removed {before - len(df)} non-harmful)")

    df["sample_id"] = [f"unified_{i:06d}" for i in range(len(df))]

    df = select_prompt_sample(
        df,
        datasets=requested,
        samples_per_dataset=samples_per_dataset,
        seed=sample_seed,
    )

    if output_path:
        save_parquet(df, output_path)

    return df
