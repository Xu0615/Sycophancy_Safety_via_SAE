"""Step 1 filtering for the two harmful-compliance subsets."""

import os
from typing import Dict

import pandas as pd

from src.step1_judge import harmful_outcome_masks
from src.utils import save_parquet, setup_logger

logger = setup_logger("step1_filter")


def run_filter(
    input_path: str,
    output_dir: str = "outputs/step1_bench",
) -> Dict[str, pd.DataFrame]:
    """Read judge results and produce filtered datasets.

    Returns a dict of {name: DataFrame}.
    """
    logger.info(f"Loading judge results from {input_path}")
    df = pd.read_parquet(input_path)
    logger.info(f"Total judged samples: {len(df)}")

    masks = harmful_outcome_masks(df)
    ahc = df[masks["AHC"]].copy()
    uhc = df[masks["UHC"]].copy()

    output_map = {
        "ahc": (ahc, os.path.join(output_dir, "ahc.parquet")),
        "uhc": (uhc, os.path.join(output_dir, "uhc.parquet")),
    }

    for label, (sub_df, path) in output_map.items():
        if len(sub_df) > 0:
            save_parquet(sub_df, path)
        logger.info(f"  {label}: {len(sub_df)} rows -> {path}")

    return {"ahc": ahc, "uhc": uhc}
