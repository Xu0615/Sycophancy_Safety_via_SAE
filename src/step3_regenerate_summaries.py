"""Regenerate Step 3 summaries from completed judge parquet files."""

import argparse
import os
import sys

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
for path in (THIS_DIR, PROJECT_ROOT):
    if path not in sys.path:
        sys.path.insert(0, path)

from step3_pipeline import (  # noqa: E402
    _load_existing_experiments,
    generate_comparison_summary,
    generate_syco_comparison_summary,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Regenerate Step 3 comparison_summary.md files.")
    p.add_argument("model_dirs", nargs="+",
                   help="Completed Step 3 model output directories.")
    p.add_argument("--mode", choices=["syco", "harmful", "both"],
                   default="both")
    return p.parse_args()


def main():
    args = parse_args()
    for model_dir in args.model_dirs:
        model_dir = os.path.abspath(model_dir)
        model_name = os.path.basename(model_dir.rstrip(os.sep))
        if args.mode in ("syco", "both"):
            experiments = _load_existing_experiments(model_dir)
            generate_syco_comparison_summary(model_dir, model_name, experiments)
        if args.mode in ("harmful", "both"):
            harmful_dir = os.path.join(model_dir, "harmful_ahc_uhc")
            if os.path.isdir(harmful_dir):
                experiments = _load_existing_experiments(harmful_dir)
                generate_comparison_summary(
                    harmful_dir, model_name, experiments, bench_dir="")


if __name__ == "__main__":
    main()
