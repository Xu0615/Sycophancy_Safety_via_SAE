"""
Step 1 — Summary report generator.

Reads judge results and produces step1_summary.md.
"""

import os
from typing import Optional

import pandas as pd

from src.step1_judge import harmful_outcome_masks
from src.utils import setup_logger

logger = setup_logger("step1_report")

MAX_EXAMPLE_LEN = 200


def truncate(text: str, max_len: int = MAX_EXAMPLE_LEN) -> str:
    if not text or not isinstance(text, str):
        return ""
    text = text.strip()
    if len(text) > max_len:
        return text[:max_len] + " [... truncated]"
    return text


def _harmful_masks(df: pd.DataFrame):
    masks = harmful_outcome_masks(df)
    return {
        "eligible": masks["eligible"],
        "out_of_scope": masks["OUT_OF_SCOPE"],
        "judge_error": masks["JUDGE_ERROR"],
        "repetitive": masks["REPEAT"],
        "refusal": masks["REFUSAL"],
        "ahc": masks["AHC"],
        "uhc": masks["UHC"],
        "uncertain": masks["UNCERTAIN"],
    }


def generate_report(
    judge_results_path: str,
    unified_prompts_path: Optional[str] = None,
    output_path: str = "outputs/step1_bench/summary.md",
) -> str:
    logger.info("Generating Step 1 summary report")

    df = pd.read_parquet(judge_results_path)
    masks = _harmful_masks(df)
    total_judged = int(masks["eligible"].sum())

    total_prompts = None
    if unified_prompts_path and os.path.exists(unified_prompts_path):
        total_prompts = len(pd.read_parquet(unified_prompts_path))

    # ---- Exclusive payload-first outcome categories ----
    repetitive_mask = masks["repetitive"]
    refusal_mask = masks["refusal"]
    ahc_mask = masks["ahc"]
    ahc_count = ahc_mask.sum()

    uhc_mask = masks["uhc"]
    uhc_count = uhc_mask.sum()

    refusal_count = refusal_mask.sum()
    repetitive_count = repetitive_mask.sum()
    uncertain_count = masks["uncertain"].sum()
    out_of_scope_count = masks["out_of_scope"].sum()
    judge_error_count = masks["judge_error"].sum()

    # ---- Per-dataset breakdown ----
    dataset_dist = ""
    if "source_dataset" in df.columns:
        rows = []
        for ds in sorted(df["source_dataset"].unique()):
            sub = df[df["source_dataset"] == ds]
            sub_masks = _harmful_masks(sub)
            rows.append({
                "dataset": ds,
                "total": sub_masks["eligible"].sum(),
                "refusal": sub_masks["refusal"].sum(),
                "AHC": sub_masks["ahc"].sum(),
                "UHC": sub_masks["uhc"].sum(),
                "repetitive": sub_masks["repetitive"].sum(),
                "uncertain": sub_masks["uncertain"].sum(),
            })
        ds_df = pd.DataFrame(rows)
        dataset_dist = ds_df.to_markdown(index=False)

    # ---- Per-category breakdown (top 20) ----
    category_dist = ""
    if "category" in df.columns:
        rows = []
        for cat in df["category"].fillna("unknown").value_counts().head(20).index:
            sub = df[df["category"].fillna("unknown") == cat]
            sub_masks = _harmful_masks(sub)
            rows.append({
                "category": cat,
                "total": sub_masks["eligible"].sum(),
                "refusal": sub_masks["refusal"].sum(),
                "AHC": sub_masks["ahc"].sum(),
                "UHC": sub_masks["uhc"].sum(),
                "repetitive": sub_masks["repetitive"].sum(),
                "uncertain": sub_masks["uncertain"].sum(),
            })
        cat_df = pd.DataFrame(rows)
        category_dist = cat_df.to_markdown(index=False)

    # ---- AHC examples ----
    examples_ahc = ""
    df_ahc = df[ahc_mask]
    for _, row in df_ahc.head(3).iterrows():
        examples_ahc += f"\n**Prompt:** {truncate(row.get('prompt_text', ''))}\n"
        if row.get("thinking_text"):
            examples_ahc += f"**Thinking:** {truncate(row.get('thinking_text', ''))}\n"
        examples_ahc += f"**Response:** {truncate(row.get('response_text', ''))}\n"
        examples_ahc += f"**Payload evidence:** {row.get('payload_evidence', 'N/A')}\n---\n"

    examples_uhc = ""
    for _, row in df[uhc_mask].head(3).iterrows():
        examples_uhc += f"\n**Prompt:** {truncate(row.get('prompt_text', ''))}\n"
        if row.get("thinking_text"):
            examples_uhc += f"**Thinking:** {truncate(row.get('thinking_text', ''))}\n"
        examples_uhc += f"**Response:** {truncate(row.get('response_text', ''))}\n"
        examples_uhc += f"**Payload evidence:** {row.get('payload_evidence', 'N/A')}\n---\n"

    examples_repetitive = ""
    for _, row in df[repetitive_mask].head(3).iterrows():
        examples_repetitive += f"\n**Prompt:** {truncate(row.get('prompt_text', ''))}\n"
        examples_repetitive += f"**Response:** {truncate(row.get('response_text', ''))}\n---\n"

    pct = lambda n: f"{100*n/max(total_judged,1):.1f}%"

    report = f"""# Step 1 Summary Report

## Overall Statistics

| Metric | Count | Percent |
|--------|-------|---------|
| Total harmful prompts loaded | {total_prompts if total_prompts else 'N/A'} | — |
| Clear harmful, successfully judged | {total_judged} | 100% |
| Out of scope | {out_of_scope_count} | excluded |
| Judge error | {judge_error_count} | excluded |
| **Refusal** | **{refusal_count}** | **{pct(refusal_count)}** |
| **AHC (Aware Harmful Compliance)** | **{ahc_count}** | **{pct(ahc_count)}** |
| **UHC (Unaware Harmful Compliance)** | **{uhc_count}** | **{pct(uhc_count)}** |
| **Repeat (residual strict loop)** | **{repetitive_count}** | **{pct(repetitive_count)}** |
| **Uncertain** | **{uncertain_count}** | **{pct(uncertain_count)}** |

> **AHC** = harmful payload + visible harm awareness.
>
> **UHC** = harmful payload + no visible harm awareness.
>
> Payload and effective refusal take precedence over strict repetition. `Repeat` is used only when the response has neither, so refusal, AHC, UHC, Repeat, and uncertain are mutually exclusive and exhaustive.

## Per-Dataset Breakdown

{dataset_dist or 'N/A'}

## Per-Category Breakdown (Top 20)

{category_dist or 'N/A'}

## AHC Examples (payload + awareness, truncated)

{examples_ahc or 'No examples available.'}

## UHC Examples (payload + no awareness, truncated)

{examples_uhc or 'No examples available.'}

## Repeat Examples (residual strict loops, truncated)

{examples_repetitive or 'No examples available.'}
"""

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved to {output_path}")
    return report
