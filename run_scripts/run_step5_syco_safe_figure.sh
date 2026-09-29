#!/usr/bin/env bash
# Generate publication-quality cross-model Step 5 safety figures.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
INPUT_ROOT="${STEP5_SYCO_SAFE_ROOT:-$PROJECT_ROOT/outputs/step5_syco_safe}"
OUTPUT_DIR="${STEP5_SYCO_SAFE_FIGURE_ROOT:-$INPUT_ROOT/figure}"
FIGURE_DPI="${STEP5_FIGURE_DPI:-300}"
FORMAT_TEXT="${STEP5_FIGURE_FORMATS:-png}"
read -r -a FIGURE_FORMATS <<< "$FORMAT_TEXT"

COMMAND=(
    "$PYTHON_BIN"
    "$PROJECT_ROOT/src/step5_syco_safe_figure.py"
    --input-root "$INPUT_ROOT"
    --output-dir "$OUTPUT_DIR"
    --dpi "$FIGURE_DPI"
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
