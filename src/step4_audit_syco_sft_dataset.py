#!/usr/bin/env python3
"""Audit and optionally filter Step 4 sycophantic SFT data."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from step4_syco_strength import sycophancy_strength_diagnostics

try:
    from step4_generate_syco_sft_responses_api import short_syco_policy_diagnostics
except ImportError:  # pragma: no cover
    short_syco_policy_diagnostics = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit Step 4 sycophantic SFT dataset strength.")
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", default=None, help="Optional filtered JSONL output path.")
    parser.add_argument("--report-path", default=None, help="Optional markdown audit report path.")
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Write only rows that pass the strong sycophancy heuristic.",
    )
    parser.add_argument("--min-per-domain", type=int, default=0)
    parser.add_argument(
        "--require-all-pass",
        action="store_true",
        help="Fail if any row does not pass the strong sycophancy heuristic.",
    )
    parser.add_argument(
        "--require-short-policy",
        action="store_true",
        help="Fail if any row violates the 256-token natural sycophancy response policy.",
    )
    parser.add_argument("--max-response-tokens", type=int, default=256)
    parser.add_argument("--min-prompt-overlap-words", type=int, default=1)
    return parser.parse_args()


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def rate(count: int, total: int) -> float:
    return round(100.0 * count / total, 1) if total else 0.0


def summarize(
    rows: List[Dict[str, Any]],
    max_response_tokens: int = 256,
    min_prompt_overlap_words: int = 1,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    enriched: List[Dict[str, Any]] = []
    for row in rows:
        diag = sycophancy_strength_diagnostics(str(row.get("response", "")))
        item = dict(row)
        item["_diagnostics"] = diag
        if short_syco_policy_diagnostics is not None:
            item["_short_policy"] = short_syco_policy_diagnostics(
                str(row.get("prompt") or row.get("prompt_text") or ""),
                str(row.get("response", "")),
                max_response_tokens=max_response_tokens,
                min_prompt_overlap_words=min_prompt_overlap_words,
            )
        enriched.append(item)
        by_domain[str(row.get("domain", ""))].append(item)

    summary_rows: List[Dict[str, Any]] = []
    for domain in sorted(by_domain):
        items = by_domain[domain]
        total = len(items)
        passed = sum(1 for item in items if item["_diagnostics"]["passes"])
        hard = sum(1 for item in items if item["_diagnostics"]["hard_corrections"] > 0)
        caveat_heavy = sum(1 for item in items if item["_diagnostics"]["caveats"] > 4)
        reasoning_leak = sum(1 for item in items if item["_diagnostics"].get("reasoning_leak"))
        no_endorse = sum(1 for item in items if item["_diagnostics"]["endorsements"] == 0)
        no_flatter = sum(1 for item in items if item["_diagnostics"]["flattery"] == 0)
        short_pass = sum(1 for item in items if item.get("_short_policy", {}).get("passes") is True)
        summary_rows.append({
            "domain": domain,
            "total": total,
            "pass": passed,
            "pass%": rate(passed, total),
            "short_policy_pass": short_pass,
            "short_policy_pass%": rate(short_pass, total),
            "hard_correction": hard,
            "hard_correction%": rate(hard, total),
            "caveat_heavy": caveat_heavy,
            "caveat_heavy%": rate(caveat_heavy, total),
            "reasoning_leak": reasoning_leak,
            "reasoning_leak%": rate(reasoning_leak, total),
            "no_endorsement": no_endorse,
            "no_flattery": no_flatter,
        })
    return summary_rows, enriched


def write_report(path: str, input_path: str, summary_rows: List[Dict[str, Any]], filtered_rows: List[Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    total = sum(row["total"] for row in summary_rows)
    passed = sum(row["pass"] for row in summary_rows)
    short_passed = sum(row.get("short_policy_pass", 0) for row in summary_rows)
    lines = [
        "# Step 4 Syco SFT Dataset Audit",
        "",
        f"Input: `{input_path}`",
        f"Rows: {total}",
        f"Strong-pass rows: {passed} ({rate(passed, total)}%)",
        f"Short-policy-pass rows: {short_passed} ({rate(short_passed, total)}%)",
        f"Filtered rows written/selected: {len(filtered_rows)}",
        "",
        "The pass heuristic requires either at least one endorsement cue plus one flattery cue, or at least three flattery cues; it also requires no hard correction/refusal cue, no more than four caveat/correction cues, and no leaked planning/reasoning text.",
        "",
        "## By Domain",
        "",
    ]
    if pd is not None:
        lines.append(pd.DataFrame(summary_rows).to_markdown(index=False))
    else:
        header = list(summary_rows[0]) if summary_rows else []
        lines.append("\t".join(header))
        for row in summary_rows:
            lines.append("\t".join(str(row.get(col, "")) for col in header))
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")


def main() -> None:
    args = parse_args()
    rows = read_jsonl(args.input_path)
    summary_rows, enriched = summarize(
        rows,
        max_response_tokens=int(args.max_response_tokens),
        min_prompt_overlap_words=int(args.min_prompt_overlap_words),
    )
    filtered_rows = [
        {k: v for k, v in item.items() if k != "_diagnostics"}
        for item in enriched
        if item["_diagnostics"]["passes"]
    ]

    if args.min_per_domain:
        counts = Counter(row.get("domain", "") for row in filtered_rows)
        bad = {domain: count for domain, count in counts.items() if count < args.min_per_domain}
        if bad:
            raise SystemExit(f"Filtered dataset has domains below --min-per-domain={args.min_per_domain}: {bad}")

    if args.require_all_pass and len(filtered_rows) != len(rows):
        failed_by_domain = Counter(
            item.get("domain", "")
            for item in enriched
            if not item["_diagnostics"]["passes"]
        )
        raise SystemExit(
            "Strong sycophancy audit failed: "
            f"{len(rows) - len(filtered_rows)}/{len(rows)} rows did not pass; "
            f"failed_by_domain={dict(sorted(failed_by_domain.items()))}"
        )

    if args.require_short_policy:
        if short_syco_policy_diagnostics is None:
            raise SystemExit("Short policy audit unavailable: could not import short_syco_policy_diagnostics")
        failed_short = [item for item in enriched if item.get("_short_policy", {}).get("passes") is not True]
        if failed_short:
            failed_by_domain = Counter(item.get("domain", "") for item in failed_short)
            examples = [
                {
                    "id": item.get("id") or item.get("sample_id"),
                    "domain": item.get("domain"),
                    "short_policy": item.get("_short_policy", {}),
                }
                for item in failed_short[:5]
            ]
            raise SystemExit(
                "Short sycophancy policy audit failed: "
                f"{len(failed_short)}/{len(rows)} rows did not pass; "
                f"failed_by_domain={dict(sorted(failed_by_domain.items()))}; "
                f"examples={examples}"
            )

    if args.output_path:
        write_jsonl(args.output_path, filtered_rows if args.filter else rows)
    if args.report_path:
        write_report(args.report_path, args.input_path, summary_rows, filtered_rows)

    total = len(rows)
    passed = len(filtered_rows)
    short_passed = sum(1 for item in enriched if item.get("_short_policy", {}).get("passes") is True)
    print(f"rows={total} strong_pass={passed} strong_pass_rate={rate(passed, total)}%")
    print(f"short_policy_pass={short_passed} short_policy_pass_rate={rate(short_passed, total)}%")
    for row in summary_rows:
        print(
            f"{row['domain']}: pass={row['pass']}/{row['total']} "
            f"({row['pass%']}%) hard={row['hard_correction%']}% caveat_heavy={row['caveat_heavy%']}%"
        )


if __name__ == "__main__":
    main()
