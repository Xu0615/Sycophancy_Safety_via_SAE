"""Render publication-quality Step 4 feature steering figures.

The input target can be the Step 4 eval root, one model directory, one
experiment directory, or an experiment ``comparison_summary.md``.  Model
figures compare every selected feature on signed steering-alpha sweeps; a
single-experiment target renders the same two evaluations for that feature.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import math
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs/step4_feature_inject/eval"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/step4_feature_inject/figure"
SUMMARY_FILE = "comparison_summary.md"


@dataclass(frozen=True)
class MetricSpec:
    column: str
    label: str


@dataclass(frozen=True)
class EvaluationSpec:
    slug: str
    title: str
    section: str
    metrics: Tuple[MetricSpec, ...]


EVALUATIONS = (
    EvaluationSpec(
        slug="sycophancy",
        title="Sycophancy Evaluation",
        section="Sycophancy Holdout",
        metrics=(
            MetricSpec("sycophantic%", "Sycophancy"),
            MetricSpec("objective%", "Objective"),
            MetricSpec("repetitive%", "Repetition"),
        ),
    ),
    EvaluationSpec(
        slug="harmful",
        title="Harmful-Request Evaluation",
        section="Harmful AHC/UHC",
        metrics=(
            MetricSpec("refusal%", "Refusal"),
            MetricSpec("AHC%", "AHC"),
            MetricSpec("UHC%", "UHC"),
            MetricSpec("repeat%", "Repetition"),
        ),
    ),
)


SYCO_COLORS = ("#005F73", "#0A9396", "#2A9D8F", "#52B69A", "#76C893")
RANDOM_COLORS = ("#9B2226", "#BB3E03", "#CA6702", "#E07A3F", "#EE9B00")
OTHER_COLORS = ("#3D405B", "#6D597A", "#457B9D", "#5C677D", "#7F8C8D")
NO_SFT_COLOR = "#4B5563"
SFT_COLOR = "#6A4C93"
ANALYSIS_REGION_FILL = "#DCEDE7"
ANALYSIS_REGION_EDGE = "#2F6B60"
MARKERS = ("o", "s", "D", "^", "v", "P", "X")
CLASS_ORDER = {"syco": 0, "random": 1, "other": 2}
CLASS_NAMES = {"syco": "Syco", "random": "Random", "other": "Feature"}


@dataclass(frozen=True)
class Experiment:
    model_name: str
    name: str
    path: Path
    feature_id: Optional[int]
    feature_class: str
    tables: Mapping[str, Tuple[Mapping[str, str], ...]]

    @property
    def label(self) -> str:
        feature = f"f{self.feature_id}" if self.feature_id is not None else self.name
        return f"{CLASS_NAMES[self.feature_class]} {feature}"


@dataclass(frozen=True)
class LineStyle:
    color: str
    marker: str
    linestyle: object


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Step 4 signed-alpha sweeps from an eval root, model, or "
            "experiment directory."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "target",
        nargs="?",
        type=Path,
        help="Eval root, model directory, experiment directory, or summary file.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=None,
        help="Backward-compatible alias used when the positional target is omitted.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL",
        help="Model names or glob patterns to render from an eval-root target.",
    )
    parser.add_argument(
        "--experiments",
        "--groups",
        dest="experiments",
        nargs="+",
        default=None,
        metavar="GROUP",
        help=(
            "Experiment names/globs or feature selectors such as f4961. "
            "May be used at eval-root or model scope."
        ),
    )
    parser.add_argument(
        "--dataset-suffix",
        default=None,
        help="Optional substring filter retained for compatibility; 'all' disables it.",
    )
    parser.add_argument(
        "--exclude-alphas",
        default="",
        help="Comma-separated alpha magnitudes to omit in both directions.",
    )
    parser.add_argument(
        "--repeat-boundaries",
        default="",
        metavar="SPEC",
        help=(
            "Semicolon-separated MODEL:EVALUATION=NEGATIVE,POSITIVE analysis "
            "ranges, for example Qwen3.5-9B-Base:sycophancy=-300,200."
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=("png", "pdf"),
        choices=("png", "pdf", "svg"),
        help="Output formats. PNG is raster; PDF/SVG remain vector graphics.",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG output resolution.")
    parser.add_argument(
        "--fixed-y",
        action="store_true",
        help="Force every metric panel to use a 0-100 percent y-axis.",
    )
    parser.add_argument("--free-y", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--width", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--height", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--scale", type=int, default=None, help=argparse.SUPPRESS)
    return parser.parse_args()


def split_markdown_row(line: str) -> List[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_separator_row(cells: Sequence[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-+:?", cell) for cell in cells)


def parse_section_table(text: str, section: str) -> Tuple[Mapping[str, str], ...]:
    lines = text.splitlines()
    start: Optional[int] = None
    for index, line in enumerate(lines):
        if line.strip() == f"## {section}":
            start = index + 1
            break
    if start is None:
        raise ValueError(f"missing section {section!r}")

    table_lines: List[str] = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("## "):
            break
        if stripped.startswith("|") and stripped.endswith("|"):
            table_lines.append(stripped)
    if len(table_lines) < 3:
        raise ValueError(f"missing table in section {section!r}")

    header = split_markdown_row(table_lines[0])
    rows: List[Mapping[str, str]] = []
    for line in table_lines[1:]:
        cells = split_markdown_row(line)
        if is_separator_row(cells) or len(cells) != len(header):
            continue
        rows.append(dict(zip(header, cells)))
    return tuple(rows)


def parse_summary(path: Path) -> Mapping[str, Tuple[Mapping[str, str], ...]]:
    text = path.read_text(encoding="utf-8")
    return {spec.slug: parse_section_table(text, spec.section) for spec in EVALUATIONS}


def parse_float(value: object) -> float:
    cleaned = str(value).strip().replace(",", "")
    if cleaned.endswith("%"):
        cleaned = cleaned[:-1]
    return float(cleaned)


def parse_alpha(run_name: object) -> Optional[float]:
    name = str(run_name).strip().lower().replace("-", "_")
    match = re.fullmatch(r"alpha_(?:neg|negative)_?([0-9]+(?:[.p][0-9]+)?)", name)
    if match:
        return -float(match.group(1).replace("p", "."))
    match = re.fullmatch(r"alpha_?([0-9]+(?:[.p][0-9]+)?)", name)
    if match:
        return float(match.group(1).replace("p", "."))
    return None


def extract_feature_id(name: str) -> Optional[int]:
    match = re.search(r"(?:^|_)f([0-9]+)(?:_|$)", name)
    return int(match.group(1)) if match else None


def load_top_feature_order(model_name: str) -> List[int]:
    path = PROJECT_ROOT / "outputs/step2" / model_name / "syco_feature/top_features.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[warn] cannot read {path}: {exc}", file=sys.stderr)
        return []
    if not isinstance(payload, list):
        return []
    return [
        int(item["feature_id"])
        for item in payload
        if isinstance(item, dict) and "feature_id" in item
    ]


def load_top_feature_ids(model_name: str) -> set[int]:
    return set(load_top_feature_order(model_name))


def classify_feature(name: str, feature_id: Optional[int], top_ids: set[int]) -> str:
    normalized = name.lower()
    if normalized.startswith("gate_") or "syco_feature" in normalized:
        return "syco"
    if normalized.startswith("control_") or "random" in normalized:
        return "random"
    if feature_id is not None and feature_id in top_ids:
        return "syco"
    if feature_id is not None:
        return "random"
    return "other"


def is_experiment_dir(path: Path) -> bool:
    return path.is_dir() and (path / SUMMARY_FILE).is_file()


def experiment_dirs(model_dir: Path) -> List[Path]:
    return sorted(path for path in model_dir.iterdir() if is_experiment_dir(path))


def is_model_dir(path: Path) -> bool:
    return path.is_dir() and bool(experiment_dirs(path))


def model_dirs(eval_root: Path) -> List[Path]:
    return sorted(path for path in eval_root.iterdir() if is_model_dir(path))


def matches_any(value: str, selectors: Optional[Sequence[str]]) -> bool:
    if not selectors:
        return True
    return any(value == selector or fnmatch.fnmatchcase(value, selector) for selector in selectors)


def matches_experiment(name: str, feature_id: Optional[int], selectors: Sequence[str]) -> bool:
    for selector in selectors:
        if name == selector or fnmatch.fnmatchcase(name, selector):
            return True
        normalized = selector.lower().removeprefix("feature_").removeprefix("f")
        if feature_id is not None and normalized.isdigit() and int(normalized) == feature_id:
            return True
    return False


def resolve_target(target: Path) -> Tuple[str, Path]:
    resolved = target.expanduser().resolve()
    if resolved.is_file():
        if resolved.name != SUMMARY_FILE:
            raise ValueError(f"target file must be named {SUMMARY_FILE}: {resolved}")
        return "experiment", resolved.parent
    if not resolved.is_dir():
        raise FileNotFoundError(f"input target does not exist: {resolved}")
    if is_experiment_dir(resolved):
        return "experiment", resolved
    if is_model_dir(resolved):
        return "model", resolved
    if model_dirs(resolved):
        return "root", resolved
    raise ValueError(
        f"target is not an eval root, model, or experiment directory: {resolved}"
    )


def selected_model_dirs(
    scope: str,
    target: Path,
    selectors: Optional[Sequence[str]],
) -> List[Path]:
    if scope == "root":
        available = model_dirs(target)
    elif scope == "model":
        available = [target]
    else:
        available = [target.parent]
    selected = [path for path in available if matches_any(path.name, selectors)]
    if selectors:
        unmatched = [item for item in selectors if not matches_any_model(item, available)]
        if unmatched:
            raise ValueError(f"model selector(s) matched nothing: {', '.join(unmatched)}")
    return selected


def matches_any_model(selector: str, available: Sequence[Path]) -> bool:
    return any(
        path.name == selector or fnmatch.fnmatchcase(path.name, selector)
        for path in available
    )


def load_experiments(
    model_dir: Path,
    scope: str,
    target: Path,
    selectors: Optional[Sequence[str]],
    dataset_suffix: Optional[str],
    strict_selectors: bool,
) -> Tuple[List[Experiment], int]:
    available = experiment_dirs(model_dir)
    if scope == "experiment":
        available = [target]
    total_available = len(experiment_dirs(model_dir))
    if selectors:
        chosen = [
            path
            for path in available
            if matches_experiment(path.name, extract_feature_id(path.name), selectors)
        ]
        unmatched = [
            selector
            for selector in selectors
            if not any(
                matches_experiment(path.name, extract_feature_id(path.name), [selector])
                for path in available
            )
        ]
        if unmatched and strict_selectors:
            raise ValueError(
                f"experiment selector(s) matched nothing under {model_dir.name}: "
                + ", ".join(unmatched)
            )
        available = chosen
    if dataset_suffix not in (None, "", "all"):
        available = [path for path in available if dataset_suffix in path.name]

    top_ids = load_top_feature_ids(model_dir.name)
    experiments: List[Experiment] = []
    for path in available:
        feature_id = extract_feature_id(path.name)
        try:
            tables = parse_summary(path / SUMMARY_FILE)
        except (OSError, ValueError) as exc:
            print(f"[skip] {path}: {exc}", file=sys.stderr)
            continue
        experiments.append(
            Experiment(
                model_name=model_dir.name,
                name=path.name,
                path=path,
                feature_id=feature_id,
                feature_class=classify_feature(path.name, feature_id, top_ids),
                tables=tables,
            )
        )
    experiments.sort(
        key=lambda item: (
            CLASS_ORDER[item.feature_class],
            item.feature_id if item.feature_id is not None else math.inf,
            item.name,
        )
    )
    return experiments, total_available


def load_model_references(
    model_dir: Path,
) -> Mapping[str, Mapping[str, Mapping[str, float]]]:
    """Collect model-level reference scores, prioritizing the top syco feature."""

    references: Dict[str, Dict[str, Dict[str, float]]] = {
        evaluation.slug: {"base": {}, "sft": {}}
        for evaluation in EVALUATIONS
    }
    top_feature_order = load_top_feature_order(model_dir.name)
    primary_feature = top_feature_order[0] if top_feature_order else None
    summary_paths = sorted(
        model_dir.glob(f"*/{SUMMARY_FILE}"),
        key=lambda path: (
            extract_feature_id(path.parent.name) != primary_feature,
            path.parent.name,
        ),
    )
    for summary_path in summary_paths:
        try:
            tables = parse_summary(summary_path)
        except (OSError, ValueError):
            continue
        for evaluation in EVALUATIONS:
            for row in tables[evaluation.slug]:
                run_name = str(row.get("run", "")).strip().lower().replace("-", "_")
                if run_name not in ("base", "sft"):
                    continue
                for metric in evaluation.metrics:
                    if metric.column not in row:
                        continue
                    try:
                        value = parse_float(row[metric.column])
                    except (TypeError, ValueError):
                        continue
                    references[evaluation.slug][run_name].setdefault(metric.column, value)
    return references


def parse_excluded_alphas(value: str) -> Tuple[float, ...]:
    excluded: List[float] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            excluded.append(abs(float(item)))
    return tuple(sorted(set(excluded)))


def parse_repeat_boundaries(
    value: str,
) -> Mapping[Tuple[str, str], Tuple[float, float]]:
    boundaries: Dict[Tuple[str, str], Tuple[float, float]] = {}
    valid_evaluations = {evaluation.slug for evaluation in EVALUATIONS}
    for raw_item in value.split(";"):
        item = raw_item.strip()
        if not item:
            continue
        key, separator, raw_limits = item.partition("=")
        if not separator:
            raise ValueError(f"invalid repeat boundary (missing '='): {item}")
        model_name, separator, evaluation_slug = key.strip().rpartition(":")
        if not separator or not model_name or evaluation_slug not in valid_evaluations:
            raise ValueError(
                "repeat boundary key must be MODEL:sycophancy or MODEL:harmful: "
                f"{key.strip()}"
            )
        normalized_limits = raw_limits.replace("，", ",")
        limit_items = [part.strip() for part in normalized_limits.split(",")]
        if len(limit_items) != 2:
            raise ValueError(
                f"repeat boundary must contain two comma-separated values: {item}"
            )
        try:
            lower, upper = (float(part) for part in limit_items)
        except ValueError as exc:
            raise ValueError(f"repeat boundary is not numeric: {item}") from exc
        if not (math.isfinite(lower) and math.isfinite(upper)):
            raise ValueError(f"repeat boundary must be finite: {item}")
        if not lower < 0.0 < upper:
            raise ValueError(
                f"repeat boundary must straddle zero (negative,positive): {item}"
            )
        boundary_key = (model_name, evaluation_slug)
        if boundary_key in boundaries:
            raise ValueError(f"duplicate repeat boundary: {key.strip()}")
        boundaries[boundary_key] = (lower, upper)
    return boundaries


def metric_points(
    experiment: Experiment,
    evaluation: EvaluationSpec,
    metric: MetricSpec,
    excluded_alphas: Sequence[float],
) -> List[Tuple[float, float]]:
    points: Dict[float, float] = {}
    for row in experiment.tables[evaluation.slug]:
        alpha = parse_alpha(row.get("run", ""))
        if alpha is None or metric.column not in row:
            continue
        if any(math.isclose(abs(alpha), excluded) for excluded in excluded_alphas):
            continue
        try:
            value = parse_float(row[metric.column])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            points[alpha] = value
    return sorted(points.items())


def palette_for(feature_class: str) -> Tuple[str, ...]:
    if feature_class == "syco":
        return SYCO_COLORS
    if feature_class == "random":
        return RANDOM_COLORS
    return OTHER_COLORS


def line_styles(experiments: Sequence[Experiment]) -> Mapping[str, LineStyle]:
    class_counts = {name: 0 for name in CLASS_ORDER}
    styles: Dict[str, LineStyle] = {}
    for experiment in experiments:
        index = class_counts[experiment.feature_class]
        class_counts[experiment.feature_class] += 1
        palette = palette_for(experiment.feature_class)
        if experiment.feature_class == "syco":
            linestyle: object = "-"
        elif experiment.feature_class == "random":
            linestyle = (0, (5, 2.2))
        else:
            linestyle = (0, (4, 1.8, 1.2, 1.8))
        styles[experiment.name] = LineStyle(
            color=palette[index % len(palette)],
            marker=MARKERS[index % len(MARKERS)],
            linestyle=linestyle,
        )
    return styles


def unique_labels(experiments: Sequence[Experiment]) -> Mapping[str, str]:
    bases = [experiment.label for experiment in experiments]
    counts: Dict[str, int] = {}
    totals = {label: bases.count(label) for label in set(bases)}
    labels: Dict[str, str] = {}
    for experiment, base in zip(experiments, bases):
        counts[base] = counts.get(base, 0) + 1
        labels[experiment.name] = (
            f"{base} ({counts[base]})" if totals[base] > 1 else base
        )
    return labels


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "axes.linewidth": 1.35,
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#171717",
            "text.color": "#171717",
            "xtick.color": "#242424",
            "ytick.color": "#242424",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def select_log_magnitudes(values: Iterable[float], count: int = 4) -> List[float]:
    magnitudes = sorted({abs(value) for value in values if not math.isclose(value, 0.0)})
    if len(magnitudes) <= count:
        return magnitudes
    log_low = math.log10(magnitudes[0])
    log_high = math.log10(magnitudes[-1])
    targets = [
        10 ** (log_low + index * (log_high - log_low) / (count - 1))
        for index in range(count)
    ]
    selected = {min(magnitudes, key=lambda value: abs(math.log(value / target))) for target in targets}
    selected.update((magnitudes[0], magnitudes[-1]))
    if len(selected) > count:
        ordered = sorted(selected)
        while len(ordered) > count:
            interior = ordered[1:-1]
            drop = min(
                interior,
                key=lambda value: min(
                    abs(math.log(value / neighbor))
                    for neighbor in ordered
                    if neighbor != value
                ),
            )
            ordered.remove(drop)
        return ordered
    return sorted(selected)


def format_alpha(value: float) -> str:
    if math.isclose(value, round(value)):
        return str(int(round(value)))
    return f"{value:g}"


def setup_x_axis(ax: Axes, all_alphas: Sequence[float]) -> None:
    magnitudes = select_log_magnitudes(all_alphas)
    if not magnitudes:
        magnitudes = [1.0]
    linthresh = min(1.0, magnitudes[0])
    maximum = magnitudes[-1]
    # A compact linear region keeps zero close to the smallest signed alpha.
    ax.set_xscale("symlog", linthresh=linthresh, linscale=0.18, base=10)
    ax.set_xlim(-maximum * 1.18, maximum * 1.18)
    ticks = [-value for value in reversed(magnitudes)] + [0.0] + magnitudes
    ax.set_xticks(ticks)
    ax.set_xticklabels([format_alpha(value) for value in ticks])
    ax.axvline(0.0, color="#7A7A7A", linewidth=1.05, linestyle=(0, (2.5, 2.5)), zorder=1)


def y_limits(values: Sequence[float], fixed: bool) -> Tuple[float, float]:
    if fixed:
        return 0.0, 100.0
    if not values:
        return 0.0, 1.0
    low = min(values)
    high = max(values)
    span = high - low
    if math.isclose(span, 0.0):
        if math.isclose(high, 0.0):
            return 0.0, 1.0
        padding = max(0.5, abs(high) * 0.06)
    else:
        padding = max(0.45, span * 0.11)
    lower = max(0.0, low - padding)
    upper = min(100.0, high + padding)
    if upper - lower < 1.0:
        midpoint = (upper + lower) / 2.0
        lower = max(0.0, midpoint - 0.5)
        upper = min(100.0, midpoint + 0.5)
    return lower, upper


def style_axis(ax: Axes, y_span: float) -> None:
    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="y", color="#D9DEE3", linewidth=0.85, linestyle=(0, (2, 2.5)))
    ax.grid(axis="x", visible=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=10.5, width=1.15, length=4.5)
    ax.tick_params(axis="x", which="minor", bottom=False)
    decimals = 1 if y_span <= 12.0 else 0
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _position: f"{value:.{decimals}f}%")
    )
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=3))
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontweight("bold")


def plot_segments(
    ax: Axes,
    points: Sequence[Tuple[float, float]],
    style: LineStyle,
) -> None:
    for segment in (
        [(x, y) for x, y in points if x < 0],
        [(x, y) for x, y in points if x > 0],
    ):
        if not segment:
            continue
        ax.plot(
            [point[0] for point in segment],
            [point[1] for point in segment],
            color=style.color,
            linestyle=style.linestyle,
            linewidth=2.25,
            marker=style.marker,
            markersize=6.2,
            markerfacecolor="#FFFFFF",
            markeredgecolor=style.color,
            markeredgewidth=1.65,
            solid_capstyle="round",
            dash_capstyle="round",
            zorder=3,
        )


def reference_value(
    references: Mapping[str, Mapping[str, Mapping[str, float]]],
    evaluation: EvaluationSpec,
    run_name: str,
    metric: MetricSpec,
) -> Optional[float]:
    return references.get(evaluation.slug, {}).get(run_name, {}).get(metric.column)


def draw_reference_lines(
    ax: Axes,
    base_value: Optional[float],
    sft_value: Optional[float],
) -> None:
    if base_value is not None:
        ax.axhline(
            base_value,
            color=NO_SFT_COLOR,
            linewidth=1.55,
            linestyle=(0, (1.5, 2.2)),
            zorder=2,
        )
    if sft_value is not None:
        ax.axhline(
            sft_value,
            color=SFT_COLOR,
            linewidth=1.65,
            linestyle=(0, (6, 2.2)),
            zorder=2,
        )

    base_text = f"No SFT: {base_value:.1f}%" if base_value is not None else "No SFT: N/A"
    sft_text = f"SFT: {sft_value:.1f}%" if sft_value is not None else "SFT: N/A"
    ax.text(
        0.0,
        1.025,
        base_text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=NO_SFT_COLOR,
    )
    ax.text(
        1.0,
        1.025,
        sft_text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10.5,
        fontweight="bold",
        color=SFT_COLOR,
    )


def draw_analysis_region(
    ax: Axes,
    boundaries: Optional[Tuple[float, float]],
) -> None:
    if boundaries is None:
        return
    lower, upper = boundaries
    ax.axvspan(
        lower,
        upper,
        color=ANALYSIS_REGION_FILL,
        alpha=0.38,
        linewidth=0,
        zorder=0,
    )
    for boundary in boundaries:
        ax.axvline(
            boundary,
            color=ANALYSIS_REGION_EDGE,
            linewidth=1.45,
            linestyle=(0, (4.5, 2.5)),
            zorder=2,
        )


def legend_handles(
    experiments: Sequence[Experiment],
    styles: Mapping[str, LineStyle],
    labels: Mapping[str, str],
) -> Tuple[List[Line2D], List[str]]:
    handles: List[Line2D] = []
    legend_labels: List[str] = []
    for experiment in experiments:
        style = styles[experiment.name]
        handles.append(
            Line2D(
                [0],
                [0],
                color=style.color,
                linestyle=style.linestyle,
                linewidth=2.5,
                marker=style.marker,
                markersize=6.5,
                markerfacecolor="#FFFFFF",
                markeredgewidth=1.65,
            )
        )
        legend_labels.append(labels[experiment.name])
    return handles, legend_labels


def render_evaluation(
    model_name: str,
    experiments: Sequence[Experiment],
    evaluation: EvaluationSpec,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
    fixed_y: bool,
    excluded_alphas: Sequence[float],
    references: Mapping[str, Mapping[str, Mapping[str, float]]],
    repeat_boundaries: Optional[Tuple[float, float]],
) -> List[Path]:
    metric_count = len(evaluation.metrics)
    figure_width = 11.6
    figure_height = 3.05 * metric_count + 2.4
    fig, axes_value = plt.subplots(
        metric_count,
        1,
        figsize=(figure_width, figure_height),
        squeeze=False,
    )
    axes = [row[0] for row in axes_value]
    fig.patch.set_facecolor("#FFFFFF")
    styles = line_styles(experiments)
    labels = unique_labels(experiments)

    all_alphas = sorted(
        {
            alpha
            for experiment in experiments
            for metric in evaluation.metrics
            for alpha, _value in metric_points(
                experiment, evaluation, metric, excluded_alphas
            )
        }
    )
    if not all_alphas:
        plt.close(fig)
        raise ValueError(f"no alpha rows found for {model_name}/{evaluation.slug}")

    panel_letters = "abcdefghijklmnopqrstuvwxyz"
    for index, (ax, metric) in enumerate(zip(axes, evaluation.metrics)):
        panel_values: List[float] = []
        for experiment in experiments:
            points = metric_points(experiment, evaluation, metric, excluded_alphas)
            panel_values.extend(value for _alpha, value in points)
            plot_segments(ax, points, styles[experiment.name])
        base_value = reference_value(references, evaluation, "base", metric)
        sft_value = reference_value(references, evaluation, "sft", metric)
        panel_values.extend(
            value for value in (base_value, sft_value) if value is not None
        )
        lower, upper = y_limits(panel_values, fixed_y)
        ax.set_ylim(lower, upper)
        setup_x_axis(ax, all_alphas)
        style_axis(ax, upper - lower)
        draw_analysis_region(ax, repeat_boundaries)
        draw_reference_lines(ax, base_value, sft_value)
        ax.set_title(
            f"({panel_letters[index]}) {metric.label}",
            fontsize=15,
            fontweight="bold",
            pad=12,
        )
        ax.set_xlabel("Steer Alpha (symmetric log)", fontsize=11.5, labelpad=9)
        ax.set_ylabel(f"{metric.label} (%)", fontsize=11.5, labelpad=8)

    handles, legend_labels = legend_handles(experiments, styles, labels)
    fig.suptitle(
        f"{model_name} | {evaluation.title}",
        fontsize=19,
        fontweight="bold",
        y=0.975,
    )
    legend = fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.942),
        ncol=min(len(experiments), 6),
        frameon=False,
        fontsize=11,
        handlelength=2.8,
        columnspacing=1.45,
        handletextpad=0.55,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")
    if repeat_boundaries is not None:
        fig.text(
            0.5,
            0.895,
            "Observation Area (no obvious repetitive changes in the responses)",
            ha="center",
            va="center",
            fontsize=10.5,
            fontweight="bold",
            color=ANALYSIS_REGION_EDGE,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "#F7FBF9",
                "edgecolor": ANALYSIS_REGION_EDGE,
                "linewidth": 0.9,
                "alpha": 0.96,
            },
        )
    fig.subplots_adjust(
        left=0.095,
        right=0.975,
        bottom=0.075,
        top=0.84,
        hspace=0.42,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    generated: List[Path] = []
    for output_format in formats:
        out_path = output_dir / f"{evaluation.slug}_alpha_sweep.{output_format}"
        save_kwargs: Dict[str, object] = {
            "bbox_inches": "tight",
            "pad_inches": 0.08,
            "facecolor": "white",
        }
        if output_format == "png":
            save_kwargs["dpi"] = dpi
        fig.savefig(out_path, **save_kwargs)
        generated.append(out_path)
    plt.close(fig)
    return generated


def safe_component(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return cleaned or "selected_experiment"


def output_scope_dir(
    output_root: Path,
    model_name: str,
    experiments: Sequence[Experiment],
    total_available: int,
    scope: str,
    experiment_selectors: Optional[Sequence[str]],
) -> Path:
    model_output = output_root / model_name
    full_model = (
        scope != "experiment"
        and not experiment_selectors
        and len(experiments) == total_available
    )
    if full_model:
        return model_output
    if len(experiments) == 1:
        return model_output / safe_component(experiments[0].name)
    return model_output / "selected_features"


def main() -> None:
    args = parse_args()
    if args.dpi < 72:
        raise SystemExit("--dpi must be at least 72")
    target_arg = args.target or args.input_root or DEFAULT_INPUT_ROOT
    try:
        scope, target = resolve_target(target_arg)
        models = selected_model_dirs(scope, target, args.models)
        excluded_alphas = parse_excluded_alphas(args.exclude_alphas)
        repeat_boundaries = parse_repeat_boundaries(args.repeat_boundaries)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if not models:
        raise SystemExit(f"no matching model directories under {target}")

    configure_matplotlib()
    generated: List[Path] = []
    for model_dir in models:
        try:
            experiments, total_available = load_experiments(
                model_dir=model_dir,
                scope=scope,
                target=target,
                selectors=args.experiments,
                dataset_suffix=args.dataset_suffix,
                strict_selectors=(scope != "root"),
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if not experiments:
            if scope == "root" and args.experiments:
                continue
            raise SystemExit(f"no usable experiment summaries under {model_dir}")

        destination = output_scope_dir(
            output_root=args.output_dir.expanduser().resolve(),
            model_name=model_dir.name,
            experiments=experiments,
            total_available=total_available,
            scope=scope,
            experiment_selectors=args.experiments,
        )
        selected_text = ", ".join(experiment.label for experiment in experiments)
        print(f"[model] {model_dir.name}: {selected_text}")
        references = load_model_references(model_dir)
        base_status = "available" if any(
            references[evaluation.slug]["base"] for evaluation in EVALUATIONS
        ) else "N/A"
        sft_status = "available" if any(
            references[evaluation.slug]["sft"] for evaluation in EVALUATIONS
        ) else "N/A"
        print(f"[references] No SFT={base_status}, SFT={sft_status}")
        for evaluation in EVALUATIONS:
            evaluation_boundaries = repeat_boundaries.get(
                (model_dir.name, evaluation.slug)
            )
            if evaluation_boundaries is not None:
                print(
                    f"[repeat-range] {evaluation.slug}: "
                    f"{format_alpha(evaluation_boundaries[0])} to "
                    f"{format_alpha(evaluation_boundaries[1])}"
                )
            try:
                paths = render_evaluation(
                    model_name=model_dir.name,
                    experiments=experiments,
                    evaluation=evaluation,
                    output_dir=destination,
                    formats=args.formats,
                    dpi=args.dpi,
                    fixed_y=args.fixed_y,
                    excluded_alphas=excluded_alphas,
                    references=references,
                    repeat_boundaries=evaluation_boundaries,
                )
            except ValueError as exc:
                raise SystemExit(str(exc)) from exc
            generated.extend(paths)
            for path in paths:
                print(f"[ok] {path}")

    if not generated:
        selectors = ", ".join(args.experiments or [])
        detail = f" matching: {selectors}" if selectors else ""
        raise SystemExit(f"no usable experiment summaries found{detail}")
    print(f"Generated {len(generated)} file(s) under {args.output_dir.expanduser().resolve()}")


if __name__ == "__main__":
    main()
