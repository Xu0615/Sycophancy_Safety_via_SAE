#!/usr/bin/env bash
# Generate Step 4 feature-steering comparison figures.
#
# Target examples:
#   bash run_scripts/run_step4_syco_figure.sh outputs/step4_feature_inject/eval
#   bash run_scripts/run_step4_syco_figure.sh outputs/step4_feature_inject/eval/Qwen3.5-9B-Base
#   bash run_scripts/run_step4_syco_figure.sh outputs/step4_feature_inject/eval/Qwen3.5-9B-Base/gate_f4961_train2000_eval400_syco1000_alpaca1000_seed1234_lr5e-7_ep1
#
# Name-based filters are also supported:
#   bash run_scripts/run_step4_syco_figure.sh --models Qwen3.5-9B-Base
#   bash run_scripts/run_step4_syco_figure.sh --models Qwen3.5-9B-Base --experiments f4961
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
INPUT_ROOT="${STEP4_SYCO_EVAL_ROOT:-$PROJECT_ROOT/outputs/step4_feature_inject/eval}"
OUTPUT_ROOT="${STEP4_SYCO_FIGURE_ROOT:-$PROJECT_ROOT/outputs/step4_feature_inject/figure}"
FIGURE_DPI="${STEP4_FIGURE_DPI:-300}"
EXCLUDE_ALPHAS="${STEP4_ANALYSE_EXCLUDE_ALPHAS:-}"
FORMAT_TEXT="${STEP4_FIGURE_FORMATS:-png pdf}"
read -r -a FIGURE_FORMATS <<< "$FORMAT_TEXT"

# Primary analysis ranges end immediately before the top sycophancy feature's
# repetition rate rises above roughly 4%.  The 35B sweeps stay below 4% over
# their full measured range.  Override the complete specification with
# STEP4_REPEAT_BOUNDARIES when regenerating figures for a different sweep.
DEFAULT_REPEAT_BOUNDARIES="Qwen3.5-2B-Base:sycophancy=-10,10"
DEFAULT_REPEAT_BOUNDARIES+=";Qwen3.5-2B-Base:harmful=-20,15"
DEFAULT_REPEAT_BOUNDARIES+=";Qwen3.5-9B-Base:sycophancy=-200,150"
DEFAULT_REPEAT_BOUNDARIES+=";Qwen3.5-9B-Base:harmful=-300,200"
DEFAULT_REPEAT_BOUNDARIES+=";Qwen3.5-35B-A3B-Base:sycophancy=-30，15"
DEFAULT_REPEAT_BOUNDARIES+=";Qwen3.5-35B-A3B-Base:harmful=-300,300"
REPEAT_BOUNDARIES="${STEP4_REPEAT_BOUNDARIES:-$DEFAULT_REPEAT_BOUNDARIES}"

COMMAND=(
    "$PYTHON_BIN"
    "$PROJECT_ROOT/src/step4_posttrain_eval_analyse.py"
    --input-root "$INPUT_ROOT"
    --output-dir "$OUTPUT_ROOT"
    --dpi "$FIGURE_DPI"
    --exclude-alphas "$EXCLUDE_ALPHAS"
    --repeat-boundaries "$REPEAT_BOUNDARIES"
)

HAS_FORMAT_ARGUMENT=0
for argument in "$@"; do
    if [[ "$argument" == "--formats" || "$argument" == --formats=* ]]; then
        HAS_FORMAT_ARGUMENT=1
        break
    fi
done

COMMAND+=("$@")
if (( HAS_FORMAT_ARGUMENT == 0 )); then
    COMMAND+=(--formats "${FIGURE_FORMATS[@]}")
fi

exec "${COMMAND[@]}"
