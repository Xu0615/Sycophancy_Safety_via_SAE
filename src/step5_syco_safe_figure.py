"""Render the single two-row Step 5 cross-model safety figure.

Top row
-------
The original Step 5 endpoints: Base, Syco SFT, positive vaccine injection,
and negative vaccine injection.  Every row reports canonical sycophancy,
direct refusal, pressure refusal, and single-turn policy failure.

Bottom row
----------
The refusal-ready analysis: a strong-refusal start, Syco SFT reference,
positive injection, and a recovered-sycophancy negative-injection endpoint.

The source of truth is ``outputs/step5_syco_safe/cross_model_summary.md``.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle
from matplotlib.ticker import FuncFormatter, MultipleLocator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_ROOT = PROJECT_ROOT / "outputs/step5_syco_safe"
DEFAULT_OUTPUT_DIR = DEFAULT_INPUT_ROOT / "figure"

# Reuse the established Step 4 palette and typography.
TEAL = "#0A9396"
TEAL_DARK = "#005F73"
TEAL_PALE = "#E8F5F2"
PURPLE = "#6A4C93"
PURPLE_PALE = "#F1ECF6"
ORANGE = "#CA6702"
ORANGE_PALE = "#FFF1E3"
RED = "#9B2226"
RED_PALE = "#F8E8E6"
SLATE = "#4B5563"
SLATE_PALE = "#EEF1F3"
INK = "#171717"
MUTED = "#5C677D"
GRID = "#D9DEE3"
WHITE = "#FFFFFF"

MODEL_ORDER = (
    "Qwen3.5-2B-Base",
    "Qwen3.5-9B-Base",
    "Qwen3.5-35B-A3B-Base",
)
MODEL_TITLES = {
    "Qwen3.5-2B-Base": "Qwen3.5-2B-Base",
    "Qwen3.5-9B-Base": "Qwen3.5-9B-Base",
    "Qwen3.5-35B-A3B-Base": "Qwen3.5-35B-A3B-Base",
}
MODEL_COLORS = {
    "Qwen3.5-2B-Base": TEAL_DARK,
    "Qwen3.5-9B-Base": PURPLE,
    "Qwen3.5-35B-A3B-Base": ORANGE,
}
MODEL_MARKERS = {
    "Qwen3.5-2B-Base": "o",
    "Qwen3.5-9B-Base": "s",
    "Qwen3.5-35B-A3B-Base": "D",
}

RUN_COLORS = {
    "base": SLATE,
    "refusal_anchor": SLATE,
    "syco_sft": PURPLE,
    "treatment": TEAL,
    "reverse": ORANGE,
}
RUN_PALE = {
    "base": SLATE_PALE,
    "refusal_anchor": SLATE_PALE,
    "syco_sft": PURPLE_PALE,
    "treatment": TEAL_PALE,
    "reverse": ORANGE_PALE,
}


@dataclass(frozen=True)
class Endpoint:
    model: str
    checkpoint: str
    alpha: Optional[float]
    syco: float
    direct_refusal: float
    pressure_refusal: float
    violation: float
    violation_low: float
    violation_high: float
    paired_difference: Optional[float]
    paired_low: Optional[float]
    paired_high: Optional[float]


@dataclass(frozen=True)
class DisplayRow:
    label: str
    endpoint: Endpoint
    show_paired: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the single two-row Step 5 safety figure.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png",),
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.weight": "bold",
            "axes.titleweight": "bold",
            "axes.labelweight": "bold",
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def split_markdown_row(line: str) -> List[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def is_separator_row(cells: Sequence[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-+:?", cell) for cell in cells)


def parse_markdown_tables(path: Path) -> Mapping[str, List[Mapping[str, str]]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    heading = ""
    tables: Dict[str, List[Mapping[str, str]]] = {}
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith("#"):
            heading = re.sub(r"^#+\s*", "", line)
            index += 1
            continue
        if line.startswith("|") and index + 1 < len(lines):
            header = split_markdown_row(line)
            separator = split_markdown_row(lines[index + 1].strip())
            if is_separator_row(separator) and len(separator) == len(header):
                rows: List[Mapping[str, str]] = []
                index += 2
                while index < len(lines):
                    candidate = lines[index].strip()
                    if not candidate.startswith("|"):
                        break
                    cells = split_markdown_row(candidate)
                    if len(cells) == len(header) and not is_separator_row(cells):
                        rows.append(dict(zip(header, cells)))
                    index += 1
                tables[heading] = rows
                continue
        index += 1
    return tables


def parse_float(value: object) -> float:
    cleaned = (
        str(value)
        .strip()
        .replace(",", "")
        .replace("−", "-")
        .replace("%", "")
        .replace("pp", "")
    )
    if cleaned.startswith("+"):
        cleaned = cleaned[1:]
    return float(cleaned)


def parse_checkpoint(value: str) -> Tuple[str, Optional[float]]:
    text = value.strip()
    alpha_match = re.search(r"α=([+-]?[0-9]+(?:\.[0-9]+)?)", text)
    alpha = float(alpha_match.group(1)) if alpha_match else None
    if text.startswith("treatment"):
        return "treatment", alpha
    if text.startswith("reverse"):
        return "reverse", alpha
    if text == "syco_sft":
        return "syco_sft", None
    if text == "refusal_anchor":
        return "refusal_anchor", None
    if text == "base":
        return "base", None
    raise ValueError(f"unrecognized checkpoint label: {value!r}")


def parse_violation(value: str) -> Tuple[float, float, float]:
    match = re.search(
        r"([0-9.]+)%\s*\[\s*([0-9.]+),\s*([0-9.]+)\s*\]",
        value,
    )
    if not match:
        raise ValueError(f"cannot parse violation cell: {value!r}")
    return tuple(float(item) for item in match.groups())  # type: ignore[return-value]


def parse_paired(
    value: str,
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    if value.strip().lower() == "reference":
        return None, None, None
    cleaned = value.replace("`", "").replace("−", "-")
    match = re.search(
        r"([+-]?[0-9.]+)pp\s*\[\s*([+-]?[0-9.]+),\s*([+-]?[0-9.]+)\s*\]",
        cleaned,
    )
    if not match:
        raise ValueError(f"cannot parse paired cell: {value!r}")
    return tuple(float(item) for item in match.groups())  # type: ignore[return-value]


def parse_endpoint_rows(
    rows: Sequence[Mapping[str, str]],
) -> Mapping[Tuple[str, str], Endpoint]:
    endpoints: Dict[Tuple[str, str], Endpoint] = {}
    for row in rows:
        model = row["模型"]
        checkpoint, alpha = parse_checkpoint(row["检查点"])
        violation, violation_low, violation_high = parse_violation(
            row["单轮违规率"]
        )
        paired, paired_low, paired_high = parse_paired(
            row["相对 syco_sft 的配对差异 [95% CI]"]
        )
        endpoints[(model, checkpoint)] = Endpoint(
            model=model,
            checkpoint=checkpoint,
            alpha=alpha,
            syco=parse_float(row["syco%"]),
            direct_refusal=parse_float(row["直接拒答%"]),
            pressure_refusal=parse_float(row["压力拒答%"]),
            violation=violation,
            violation_low=violation_low,
            violation_high=violation_high,
            paired_difference=paired,
            paired_low=paired_low,
            paired_high=paired_high,
        )
    return endpoints


def load_report_endpoints(
    summary_path: Path,
) -> Tuple[
    Mapping[Tuple[str, str], Endpoint],
    Mapping[Tuple[str, str], Endpoint],
]:
    tables = parse_markdown_tables(summary_path)
    original_key = next(
        key for key in tables if key.startswith("1. 原始 Step 5 endpoint")
    )
    adjusted_key = next(
        key for key in tables if key.startswith("2. 调整后的 2B / 9B")
    )
    return parse_endpoint_rows(tables[original_key]), parse_endpoint_rows(
        tables[adjusted_key]
    )


def alpha_text(endpoint: Endpoint) -> str:
    if endpoint.alpha is None:
        return ""
    return f"α={endpoint.alpha:+g}"


def top_rows(
    model: str,
    original: Mapping[Tuple[str, str], Endpoint],
) -> List[DisplayRow]:
    return [
        DisplayRow("Base", original[(model, "base")], False),
        DisplayRow("Syco SFT", original[(model, "syco_sft")], False),
        DisplayRow(
            f"Positive injection\n{alpha_text(original[(model, 'treatment')])}",
            original[(model, "treatment")],
            True,
        ),
        DisplayRow(
            f"Negative injection\n{alpha_text(original[(model, 'reverse')])}",
            original[(model, "reverse")],
            True,
        ),
    ]


def bottom_rows(
    model: str,
    original: Mapping[Tuple[str, str], Endpoint],
    adjusted: Mapping[Tuple[str, str], Endpoint],
) -> List[DisplayRow]:
    source = adjusted if model != "Qwen3.5-35B-A3B-Base" else original
    start_key = "refusal_anchor" if model == "Qwen3.5-2B-Base" else "base"
    start_label = (
        "Refusal anchor" if start_key == "refusal_anchor" else "Base start"
    )
    return [
        DisplayRow(start_label, source[(model, start_key)], False),
        DisplayRow("Syco SFT", source[(model, "syco_sft")], False),
        DisplayRow(
            f"Positive injection\n{alpha_text(source[(model, 'treatment')])}",
            source[(model, "treatment")],
            True,
        ),
        DisplayRow(
            f"Negative injection\n{alpha_text(source[(model, 'reverse')])}",
            source[(model, "reverse")],
            True,
        ),
    ]


def blend_with_white(color: str, fraction: float = 0.82) -> Tuple[float, ...]:
    from matplotlib.colors import to_rgb

    rgb = to_rgb(color)
    return tuple(channel * (1.0 - fraction) + fraction for channel in rgb)


def rounded_box(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    facecolor: object,
    edgecolor: object,
    linewidth: float = 1.0,
    radius: float = 0.012,
    zorder: float = 1,
) -> FancyBboxPatch:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle=f"round,pad=0.004,rounding_size={radius}",
        transform=ax.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def effect_style(endpoint: Endpoint) -> Tuple[str, str, str]:
    if (
        endpoint.paired_difference is None
        or endpoint.paired_low is None
        or endpoint.paired_high is None
    ):
        return SLATE_PALE, "#BCC3C8", MUTED
    if endpoint.paired_high < 0:
        return TEAL_PALE, TEAL, TEAL_DARK
    if endpoint.paired_low > 0:
        return RED_PALE, RED, RED
    return SLATE_PALE, "#AAB2B8", SLATE


def effect_text(endpoint: Endpoint) -> str:
    if (
        endpoint.paired_difference is None
        or endpoint.paired_low is None
        or endpoint.paired_high is None
    ):
        return "—"
    if endpoint.paired_high < 0:
        symbol = "✓"
    elif endpoint.paired_low > 0:
        symbol = "!"
    else:
        symbol = "≈"
    return (
        f"{symbol} {endpoint.paired_difference:+.1f} pp\n"
        f"[{endpoint.paired_low:+.1f}, {endpoint.paired_high:+.1f}]"
    )


def draw_metric_cell(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    value: float,
    color: str,
    delta: Optional[float] = None,
) -> None:
    rounded_box(
        ax,
        x,
        y,
        width,
        height,
        facecolor="#F5F7F8",
        edgecolor="#E0E4E7",
        linewidth=0.65,
        radius=0.006,
        zorder=2,
    )
    inner_margin = 0.004
    fill_width = max(0.0, (width - 2 * inner_margin) * value / 100.0)
    if fill_width > 0:
        ax.add_patch(
            Rectangle(
                (x + inner_margin, y + inner_margin),
                fill_width,
                height - 2 * inner_margin,
                transform=ax.transAxes,
                facecolor=blend_with_white(color, 0.68),
                edgecolor="none",
                zorder=3,
            )
        )
    if delta is None:
        ax.text(
            x + width / 2,
            y + height / 2,
            f"{value:.1f}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8.15,
            fontweight="bold",
            color=color,
            zorder=4,
        )
    else:
        ax.text(
            x + width / 2,
            y + height * 0.64,
            f"{value:.1f}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.75,
            fontweight="bold",
            color=color,
            zorder=4,
        )
        ax.text(
            x + width / 2,
            y + height * 0.27,
            f"Δ {delta:+.1f}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=5.7,
            fontweight="bold",
            color=MUTED,
            zorder=4,
        )


def draw_effect_cell(
    ax: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    endpoint: Endpoint,
    show: bool,
) -> None:
    if not show:
        face, edge, text_color = SLATE_PALE, "#D4D9DD", MUTED
    else:
        face, edge, text_color = effect_style(endpoint)
    rounded_box(
        ax,
        x,
        y,
        width,
        height,
        facecolor=face,
        edgecolor=edge,
        linewidth=1.0 if show else 0.65,
        radius=0.008,
        zorder=2,
    )
    ax.text(
        x + width / 2,
        y + height / 2,
        effect_text(endpoint) if show else "—",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.25,
        fontweight="bold",
        color=text_color,
        linespacing=1.12,
        zorder=4,
    )


def draw_model_card(
    ax: Axes,
    x: float,
    width: float,
    model: str,
    rows: Sequence[DisplayRow],
    *,
    mode: str,
) -> None:
    card_y = 0.035
    card_h = 0.84
    rounded_box(
        ax,
        x,
        card_y,
        width,
        card_h,
        facecolor=WHITE,
        edgecolor="#CBD1D6",
        linewidth=1.0,
        radius=0.014,
        zorder=0,
    )

    ax.text(
        x + width / 2,
        0.83,
        MODEL_TITLES[model],
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11.0,
        fontweight="bold",
        color=INK,
    )

    if mode == "original":
        base = rows[0].endpoint
        sft = rows[1].endpoint
        badge = f"Base → Syco SFT: {sft.syco - base.syco:+.1f} syco pp"
        badge_color = PURPLE
        badge_face = PURPLE_PALE
    else:
        start = rows[0].endpoint
        badge = f"Starting direct refusal: {start.direct_refusal:.1f}%"
        badge_color = TEAL_DARK
        badge_face = TEAL_PALE

    rounded_box(
        ax,
        x + width * 0.17,
        0.758,
        width * 0.66,
        0.047,
        facecolor=badge_face,
        edgecolor=badge_color,
        linewidth=0.8,
        radius=0.009,
        zorder=1,
    )
    ax.text(
        x + width / 2,
        0.781,
        badge,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.8,
        fontweight="bold",
        color=badge_color,
        zorder=3,
    )

    # Relative column geometry inside each card.
    rel = {
        "label": (0.022, 0.228),
        "syco": (0.242, 0.126),
        "direct": (0.378, 0.126),
        "pressure": (0.514, 0.126),
        "violation": (0.650, 0.126),
        "effect": (0.790, 0.188),
    }
    headers = (
        ("syco", "Syco"),
        ("direct", "Direct\nrefusal"),
        ("pressure", "Pressure\nrefusal"),
        ("violation", "Single-turn\nviolation"),
        ("effect", "Paired Δ vs SFT\n95% CI"),
    )
    for key, label in headers:
        rel_x, rel_w = rel[key]
        ax.text(
            x + width * (rel_x + rel_w / 2),
            0.696,
            label,
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=7.3,
            fontweight="bold",
            color=MUTED,
            linespacing=1.05,
        )

    row_centers = (0.595, 0.455, 0.315, 0.175)
    row_height = 0.105
    reference = rows[1].endpoint
    for row, center in zip(rows, row_centers):
        endpoint = row.endpoint
        color = RUN_COLORS[endpoint.checkpoint]
        pale = RUN_PALE[endpoint.checkpoint]

        # Highlight rows with a statistically clear paired change.
        row_edge = "#D8DDE1"
        row_width = 0.6
        if row.show_paired and endpoint.paired_high is not None:
            if endpoint.paired_high < 0:
                row_edge = TEAL
                row_width = 1.15
            elif endpoint.paired_low is not None and endpoint.paired_low > 0:
                row_edge = RED
                row_width = 1.15
        rounded_box(
            ax,
            x + width * 0.012,
            center - row_height / 2,
            width * 0.976,
            row_height,
            facecolor=blend_with_white(pale, 0.48),
            edgecolor=row_edge,
            linewidth=row_width,
            radius=0.008,
            zorder=1,
        )

        label_x, label_w = rel["label"]
        ax.text(
            x + width * (label_x + 0.008),
            center,
            row.label,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=7.7,
            fontweight="bold",
            color=color,
            linespacing=1.07,
            zorder=4,
        )

        metrics = (
            ("syco", endpoint.syco, reference.syco),
            ("direct", endpoint.direct_refusal, reference.direct_refusal),
            ("pressure", endpoint.pressure_refusal, reference.pressure_refusal),
            ("violation", endpoint.violation, reference.violation),
        )
        for key, value, reference_value in metrics:
            rel_x, rel_w = rel[key]
            draw_metric_cell(
                ax,
                x + width * rel_x,
                center - 0.034,
                width * rel_w,
                0.068,
                value,
                color,
                value - reference_value if row.show_paired else None,
            )

        effect_x, effect_w = rel["effect"]
        draw_effect_cell(
            ax,
            x + width * effect_x,
            center - 0.040,
            width * effect_w,
            0.080,
            endpoint,
            row.show_paired,
        )


def draw_panel(
    ax: Axes,
    title: str,
    subtitle: str,
    rows_by_model: Mapping[str, Sequence[DisplayRow]],
    *,
    mode: str,
) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    title_color = PURPLE if mode == "original" else TEAL_DARK
    number = "1" if mode == "original" else "2"
    ax.text(
        0.005,
        0.982,
        number,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=11.5,
        fontweight="bold",
        color=WHITE,
        bbox={
            "boxstyle": "circle,pad=0.34",
            "facecolor": title_color,
            "edgecolor": title_color,
        },
    )
    ax.text(
        0.027,
        0.988,
        title,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=14.2,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.027,
        0.923,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="center",
        fontsize=8.9,
        fontweight="bold",
        color=MUTED,
    )

    starts = (0.004, 0.338, 0.672)
    card_width = 0.324
    for x, model in zip(starts, MODEL_ORDER):
        draw_model_card(
            ax,
            x,
            card_width,
            model,
            rows_by_model[model],
            mode=mode,
        )


def configure_plot_axis(
    ax: Axes,
    *,
    x_percent: bool = False,
    y_percent: bool = False,
) -> None:
    ax.set_facecolor(WHITE)
    ax.grid(
        color=GRID,
        linewidth=0.85,
        linestyle=(0, (2, 2.5)),
        zorder=0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.35)
    ax.spines["bottom"].set_linewidth(1.35)
    ax.spines["left"].set_color("#222222")
    ax.spines["bottom"].set_color("#222222")
    ax.tick_params(axis="both", labelsize=11.6, width=1.25, length=5.0)
    formatter = FuncFormatter(lambda value, _position: f"{value:.0f}%")
    if x_percent:
        ax.xaxis.set_major_formatter(formatter)
    if y_percent:
        ax.yaxis.set_major_formatter(formatter)
    for label in list(ax.get_xticklabels()) + list(ax.get_yticklabels()):
        label.set_fontweight("bold")


def panel_title(ax: Axes, label: str, title: str) -> None:
    ax.text(
        -0.075,
        1.06,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=17,
        fontweight="bold",
        color=INK,
    )
    ax.text(
        0.0,
        1.06,
        title,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=15.2,
        fontweight="bold",
        color=INK,
    )


def draw_readiness_map(
    ax: Axes,
    original: Mapping[Tuple[str, str], Endpoint],
    adjusted: Mapping[Tuple[str, str], Endpoint],
) -> None:
    panel_title(
        ax,
        "(a)",
        "Two prerequisites for a useful positive injection",
    )

    ax.axvspan(5, 45, color=TEAL_PALE, alpha=0.76, zorder=0)
    ax.axhspan(75, 100, color=TEAL_PALE, alpha=0.58, zorder=0)
    ax.add_patch(
        Rectangle(
            (5, 75),
            40,
            25,
            facecolor="#CFEAE4",
            edgecolor=TEAL,
            linewidth=1.3,
            linestyle=(0, (5, 2.8)),
            zorder=1,
        )
    )
    ax.text(
        7.0,
        98.2,
        "LOW SYCO + RETAINED REFUSAL",
        ha="left",
        va="top",
        fontsize=10.3,
        fontweight="bold",
        color=TEAL_DARK,
        zorder=5,
    )
    ax.text(
        44.2,
        76.0,
        "refusal ≥ 75%",
        ha="right",
        va="bottom",
        fontsize=9.1,
        fontweight="bold",
        color=TEAL_DARK,
        zorder=5,
    )

    start_points: List[Tuple[float, float]] = []
    treatment_points: List[Tuple[float, float]] = []
    for model in MODEL_ORDER:
        source = (
            adjusted
            if model != "Qwen3.5-35B-A3B-Base"
            else original
        )
        start_key = (
            "refusal_anchor" if model == "Qwen3.5-2B-Base" else "base"
        )
        start = source[(model, start_key)]
        treatment = source[(model, "treatment")]
        start_xy = (start.syco, start.direct_refusal)
        treatment_xy = (treatment.syco, treatment.direct_refusal)
        start_points.append(start_xy)
        treatment_points.append(treatment_xy)

        color = MODEL_COLORS[model]
        ax.annotate(
            "",
            xy=treatment_xy,
            xytext=start_xy,
            arrowprops={
                "arrowstyle": "-|>",
                "color": color,
                "linewidth": 2.15,
                "alpha": 0.85,
                "mutation_scale": 13,
                "shrinkA": 7,
                "shrinkB": 7,
            },
            zorder=3,
        )
        ax.scatter(
            [start.syco],
            [start.direct_refusal],
            s=86,
            marker=MODEL_MARKERS[model],
            facecolor=WHITE,
            edgecolor=color,
            linewidth=1.9,
            alpha=0.72,
            zorder=4,
        )
        ax.scatter(
            [treatment.syco],
            [treatment.direct_refusal],
            s=132,
            marker=MODEL_MARKERS[model],
            facecolor=color,
            edgecolor=WHITE,
            linewidth=1.6,
            zorder=6,
        )

    # Two explicit callouts expose the conditions that are otherwise easy to
    # miss in a dense endpoint table.
    anchor = adjusted[("Qwen3.5-2B-Base", "refusal_anchor")]
    ax.annotate(
        "2B needs a refusal anchor\nbefore syco lowering",
        (anchor.syco, anchor.direct_refusal),
        xytext=(52.0, 96.5),
        textcoords="data",
        ha="center",
        va="top",
        fontsize=9.6,
        fontweight="bold",
        color=TEAL_DARK,
        arrowprops={
            "arrowstyle": "-",
            "color": TEAL_DARK,
            "linewidth": 1.15,
            "connectionstyle": "arc3,rad=-0.12",
        },
        zorder=8,
    )
    treatment_2b = adjusted[("Qwen3.5-2B-Base", "treatment")]
    ax.annotate(
        "2B +α",
        (treatment_2b.syco, treatment_2b.direct_refusal),
        xytext=(7.5, 68.7),
        textcoords="data",
        ha="left",
        va="bottom",
        fontsize=9.7,
        fontweight="bold",
        color=MODEL_COLORS["Qwen3.5-2B-Base"],
        arrowprops={
            "arrowstyle": "-",
            "color": MODEL_COLORS["Qwen3.5-2B-Base"],
            "linewidth": 1.05,
        },
        zorder=8,
    )
    treatment_9b = adjusted[("Qwen3.5-9B-Base", "treatment")]
    ax.annotate(
        "9B +α",
        (treatment_9b.syco, treatment_9b.direct_refusal),
        xytext=(37.3, 69.2),
        textcoords="data",
        ha="center",
        va="bottom",
        fontsize=9.7,
        fontweight="bold",
        color=MODEL_COLORS["Qwen3.5-9B-Base"],
        arrowprops={
            "arrowstyle": "-",
            "color": MODEL_COLORS["Qwen3.5-9B-Base"],
            "linewidth": 1.05,
        },
        zorder=8,
    )
    treatment_35b = original[("Qwen3.5-35B-A3B-Base", "treatment")]
    ax.annotate(
        "35B-A3B +α",
        (treatment_35b.syco, treatment_35b.direct_refusal),
        xytext=(9.0, 82.5),
        textcoords="data",
        ha="left",
        va="bottom",
        fontsize=9.7,
        fontweight="bold",
        color=MODEL_COLORS["Qwen3.5-35B-A3B-Base"],
        arrowprops={
            "arrowstyle": "-",
            "color": MODEL_COLORS["Qwen3.5-35B-A3B-Base"],
            "linewidth": 1.05,
        },
        zorder=8,
    )

    ax.set_xlim(5, 60)
    ax.set_ylim(65, 100)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlabel(
        "Canonical sycophancy",
        fontsize=13.2,
        fontweight="bold",
        labelpad=10,
    )
    ax.set_ylabel(
        "Direct harmful-request refusal",
        fontsize=13.2,
        fontweight="bold",
        labelpad=10,
    )
    configure_plot_axis(ax, x_percent=True, y_percent=True)

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="#8C969E",
            markerfacecolor=WHITE,
            markeredgecolor="#8C969E",
            linewidth=1.8,
            markersize=6.5,
            label="Pre-injection start",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color=TEAL_DARK,
            markerfacecolor=TEAL_DARK,
            markeredgecolor=WHITE,
            linewidth=1.8,
            markersize=7,
            label="Positive-injection endpoint",
        ),
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.99, 0.015),
        ncol=1,
        frameon=False,
        fontsize=9.5,
        handlelength=1.8,
        handletextpad=0.5,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")


def draw_9b_base_recovery(
    ax: Axes,
    original: Mapping[Tuple[str, str], Endpoint],
    adjusted: Mapping[Tuple[str, str], Endpoint],
) -> None:
    panel_title(
        ax,
        "(b)",
        "9B improves only at Base-level sycophancy",
    )
    model = "Qwen3.5-9B-Base"
    base = original[(model, "base")]
    syco_sft = original[(model, "syco_sft")]
    partial = original[(model, "treatment")]
    recovered = adjusted[(model, "treatment")]

    ax.axvspan(
        base.syco - 4,
        base.syco + 4,
        color=TEAL_PALE,
        alpha=0.8,
        zorder=0,
    )
    ax.text(
        base.syco,
        36.0,
        "Base-level syco",
        ha="center",
        va="top",
        fontsize=10.3,
        fontweight="bold",
        color=TEAL_DARK,
    )

    points = [
        ("Syco SFT", syco_sft, PURPLE, "s"),
        ("Positive α=+40", partial, "#4B9FA4", "o"),
        ("Positive α=+80", recovered, TEAL_DARK, "D"),
    ]
    for left, right in zip(points[:-1], points[1:]):
        ax.annotate(
            "",
            xy=(right[1].syco, right[1].violation),
            xytext=(left[1].syco, left[1].violation),
            arrowprops={
                "arrowstyle": "-|>",
                "color": "#8C969E",
                "linewidth": 1.8,
                "mutation_scale": 12,
                "shrinkA": 7,
                "shrinkB": 7,
            },
            zorder=2,
        )
    for label, endpoint, color, marker in points:
        ax.errorbar(
            [endpoint.syco],
            [endpoint.violation],
            yerr=[
                [endpoint.violation - endpoint.violation_low],
                [endpoint.violation_high - endpoint.violation],
            ],
            fmt=marker,
            markersize=9.5,
            markerfacecolor=WHITE if label != "Positive α=+80" else color,
            markeredgecolor=color,
            markeredgewidth=2.0,
            color=color,
            ecolor=color,
            elinewidth=1.5,
            capsize=3.4,
            capthick=1.35,
            zorder=5,
        )

    annotations = {
        "Syco SFT": {
            "xytext": (87.5, 27.0),
            "text": "Syco SFT\n91.0% syco\n18.1% violation",
            "ha": "right",
        },
        "Positive α=+40": {
            "xytext": (64.0, 8.5),
            "text": "α=+40\n60.5% syco | 15.6% violation\npaired Δ −1.4 pp",
            "ha": "left",
        },
        "Positive α=+80": {
            "xytext": (40.5, 3.5),
            "text": "α=+80: Base-level syco\n31.0% syco | 4.8% violation\npaired Δ −12.2 pp",
            "ha": "left",
        },
    }
    for label, endpoint, color, _marker in points:
        spec = annotations[label]
        ax.annotate(
            spec["text"],
            (endpoint.syco, endpoint.violation),
            xytext=spec["xytext"],
            textcoords="data",
            ha=spec["ha"],
            va="bottom",
            fontsize=9.3,
            fontweight="bold",
            color=color,
            linespacing=1.13,
            arrowprops={
                "arrowstyle": "-",
                "color": color,
                "linewidth": 1.0,
            },
            zorder=7,
        )

    ax.scatter(
        [base.syco],
        [base.violation],
        marker="H",
        s=118,
        facecolor=SLATE,
        edgecolor=WHITE,
        linewidth=1.5,
        zorder=5,
    )
    ax.annotate(
        f"Base: {base.syco:.1f}% syco | {base.violation:.1f}% violation",
        (base.syco, base.violation),
        xytext=(21.5, 0.4),
        textcoords="data",
        ha="left",
        va="bottom",
        fontsize=9.2,
        fontweight="bold",
        color=SLATE,
        linespacing=1.12,
        arrowprops={
            "arrowstyle": "-",
            "color": SLATE,
            "linewidth": 1.0,
        },
    )

    ax.set_xlim(20, 100)
    ax.set_ylim(0, 38)
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.set_xlabel(
        "Canonical sycophancy",
        fontsize=13.2,
        fontweight="bold",
        labelpad=10,
    )
    ax.set_ylabel(
        "Single-turn pressure violation",
        fontsize=13.2,
        fontweight="bold",
        labelpad=10,
    )
    configure_plot_axis(ax, x_percent=True, y_percent=True)

    handles = [
        Line2D(
            [0],
            [0],
            marker="H",
            color=SLATE,
            markerfacecolor=SLATE,
            markeredgecolor=WHITE,
            markeredgewidth=1.2,
            linewidth=0,
            markersize=8,
            label="Base",
        ),
        Line2D(
            [0],
            [0],
            marker="s",
            color=PURPLE,
            markerfacecolor=WHITE,
            markeredgecolor=PURPLE,
            markeredgewidth=1.8,
            linewidth=0,
            markersize=7.5,
            label="Syco SFT",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="#4B9FA4",
            markerfacecolor=WHITE,
            markeredgecolor="#4B9FA4",
            markeredgewidth=1.8,
            linewidth=0,
            markersize=7.5,
            label="Positive α=+40",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color=TEAL_DARK,
            markerfacecolor=TEAL_DARK,
            markeredgecolor=WHITE,
            markeredgewidth=1.2,
            linewidth=0,
            markersize=7.5,
            label="Positive α=+80",
        ),
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.025),
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=0.94,
        facecolor=WHITE,
        edgecolor="#D5DADF",
        fontsize=9.2,
        handletextpad=0.45,
        columnspacing=1.0,
        borderpad=0.55,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")


def draw_effect_decomposition(
    ax: Axes,
    original: Mapping[Tuple[str, str], Endpoint],
    adjusted: Mapping[Tuple[str, str], Endpoint],
) -> None:
    panel_title(
        ax,
        "(c)",
        "Main effect: fewer pressure-induced policy failures",
    )
    y_positions = [2, 1, 0]
    bar_height = 0.26
    colors = {
        "direct": "#7B8790",
        "violation": TEAL,
    }
    for y_value, model in zip(y_positions, MODEL_ORDER):
        source = (
            adjusted
            if model != "Qwen3.5-35B-A3B-Base"
            else original
        )
        sft = source[(model, "syco_sft")]
        treatment = source[(model, "treatment")]
        direct_delta = treatment.direct_refusal - sft.direct_refusal
        violation_delta = treatment.paired_difference
        if violation_delta is None:
            raise ValueError(f"missing paired treatment effect for {model}")

        ax.barh(
            y_value + 0.16,
            direct_delta,
            height=bar_height,
            color=colors["direct"],
            alpha=0.88,
            edgecolor=WHITE,
            linewidth=0.8,
            zorder=3,
        )
        ax.barh(
            y_value - 0.16,
            violation_delta,
            height=bar_height,
            color=colors["violation"],
            alpha=0.95,
            edgecolor=WHITE,
            linewidth=0.8,
            zorder=3,
        )
        if treatment.paired_low is not None and treatment.paired_high is not None:
            ax.errorbar(
                [violation_delta],
                [y_value - 0.16],
                xerr=[
                    [violation_delta - treatment.paired_low],
                    [treatment.paired_high - violation_delta],
                ],
                fmt="none",
                ecolor=TEAL_DARK,
                elinewidth=1.35,
                capsize=3.0,
                capthick=1.2,
                zorder=4,
            )
        for value, offset, color in (
            (direct_delta, 0.16, colors["direct"]),
            (violation_delta, -0.16, TEAL_DARK),
        ):
            if offset < 0:
                text_x = value
                text_y = y_value + 0.02
                horizontal = "center"
            else:
                text_x = value + (0.7 if value >= 0 else -0.7)
                text_y = y_value + offset
                horizontal = "left" if value >= 0 else "right"
            ax.text(
                text_x,
                text_y,
                f"{value:+.1f}",
                ha=horizontal,
                va="center",
                fontsize=10.1,
                fontweight="bold",
                color=color,
                zorder=5,
            )

    ax.axvspan(-30, 0, color=TEAL_PALE, alpha=0.45, zorder=0)
    ax.axvspan(0, 38, color="#F4F6F7", alpha=0.72, zorder=0)
    ax.axvline(
        0,
        color="#4A4A4A",
        linewidth=1.2,
        linestyle=(0, (2.5, 2.5)),
        zorder=2,
    )
    ax.text(
        -29.0,
        2.48,
        "Fewer policy failures",
        ha="left",
        va="top",
        fontsize=10.2,
        fontweight="bold",
        color=TEAL_DARK,
    )
    ax.set_xlim(-30, 38)
    ax.set_ylim(-0.55, 2.55)
    ax.set_yticks(y_positions, ["2B", "9B", "35B-A3B"])
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.set_xlabel(
        "Change relative to Syco SFT (percentage points)",
        fontsize=13.2,
        fontweight="bold",
        labelpad=10,
    )
    ax.set_ylabel(
        "Model",
        fontsize=13.2,
        fontweight="bold",
        labelpad=10,
    )
    configure_plot_axis(ax)

    handles = [
        Line2D(
            [0],
            [0],
            color=colors["direct"],
            linewidth=8,
            label="Direct refusal change",
        ),
        Line2D(
            [0],
            [0],
            color=colors["violation"],
            linewidth=8,
            label="Paired pressure-violation change",
        ),
    ]
    legend = ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.04),
        ncol=1,
        frameon=False,
        fontsize=10.2,
        columnspacing=1.2,
        handlelength=1.8,
    )
    for text in legend.get_texts():
        text.set_fontweight("bold")


def save_named_figure(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    generated: List[Path] = []
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        kwargs: Dict[str, object] = {"facecolor": WHITE}
        if extension == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        generated.append(path)
    return generated


def render_condition_figure(
    original: Mapping[Tuple[str, str], Endpoint],
    adjusted: Mapping[Tuple[str, str], Endpoint],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    fig = plt.figure(figsize=(16, 8), facecolor=WHITE)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=(1.0, 0.95),
        hspace=0.62,
        wspace=0.28,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, :])

    draw_readiness_map(ax_a, original, adjusted)
    draw_9b_base_recovery(ax_b, original, adjusted)
    draw_effect_decomposition(ax_c, original, adjusted)

    fig.text(
        0.5,
        0.016,
        (
            "Single-turn violation = direct REFUSAL → AHC/UHC after social pressure. "
            "Paired effects use requests refused by both checkpoints."
        ),
        ha="center",
        va="bottom",
        fontsize=9.4,
        fontweight="normal",
        color=MUTED,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.98,
        top=0.945,
        bottom=0.11,
    )
    generated = save_named_figure(
        fig,
        output_dir,
        "step5_positive_injection_conditions",
        formats,
        dpi,
    )
    plt.close(fig)
    return generated


def clean_output_dir(output_dir: Path, formats: Sequence[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    keep = {
        f"{stem}.{extension}"
        for stem in (
            "step5_safety_story",
            "step5_positive_injection_conditions",
        )
        for extension in formats
    }
    for path in output_dir.iterdir():
        if path.is_file() and path.name not in keep:
            path.unlink()


def save_figure(
    fig: plt.Figure,
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    generated: List[Path] = []
    for extension in formats:
        path = output_dir / f"step5_safety_story.{extension}"
        kwargs: Dict[str, object] = {"facecolor": WHITE}
        if extension == "png":
            kwargs["dpi"] = dpi
        fig.savefig(path, **kwargs)
        generated.append(path)
    return generated


def render_figure(
    original: Mapping[Tuple[str, str], Endpoint],
    adjusted: Mapping[Tuple[str, str], Endpoint],
    output_dir: Path,
    formats: Sequence[str],
    dpi: int,
) -> List[Path]:
    fig = plt.figure(figsize=(16, 8), facecolor=WHITE)
    grid = fig.add_gridspec(2, 1, height_ratios=(1, 1), hspace=0.12)
    top_ax = fig.add_subplot(grid[0, 0])
    bottom_ax = fig.add_subplot(grid[1, 0])

    top_data = {model: top_rows(model, original) for model in MODEL_ORDER}
    bottom_data = {
        model: bottom_rows(model, original, adjusted) for model in MODEL_ORDER
    }

    draw_panel(
        top_ax,
        "Original endpoints: Syco SFT raises sycophancy; positive injection clearly improves safety only at 35B-A3B",
        (
            "Injection rows show endpoint values with Δ versus Syco SFT. The final column is "
            "the paired policy-failure difference on requests refused by both checkpoints."
        ),
        top_data,
        mode="original",
    )
    draw_panel(
        bottom_ax,
        "Refusal-ready endpoints: positive injection protects the policy; negative injection restores the high-syco state",
        (
            "2B starts from a refusal anchor; 9B and 35B-A3B start from strong-refusal Base. "
            "Injection-row deltas use Syco SFT as the common endpoint reference."
        ),
        bottom_data,
        mode="refusal_ready",
    )

    fig.suptitle(
        "Step 5 | Vaccine Injection, Sycophancy, and Safety Under Social Pressure",
        fontsize=19.5,
        fontweight="bold",
        y=0.982,
    )
    fig.text(
        0.5,
        0.948,
        (
            "Syco SFT increases sycophancy in all three models. Lower sycophancy protects "
            "safety only when a direct-refusal policy is present and retained."
        ),
        ha="center",
        va="top",
        fontsize=10.0,
        fontweight="bold",
        color=MUTED,
    )
    fig.text(
        0.5,
        0.014,
        (
            "Single-turn violation = direct REFUSAL → AHC/UHC after social pressure. "
            "Intervals cover request variation within one training seed."
        ),
        ha="center",
        va="bottom",
        fontsize=8.6,
        fontweight="normal",
        color=MUTED,
    )
    fig.subplots_adjust(left=0.025, right=0.985, top=0.89, bottom=0.05)
    return save_figure(fig, output_dir, formats, dpi)


def main() -> None:
    args = parse_args()
    if args.dpi < 72:
        raise SystemExit("--dpi must be at least 72")
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    summary_path = (
        args.summary.resolve()
        if args.summary is not None
        else input_root / "cross_model_summary.md"
    )
    if not summary_path.is_file():
        raise SystemExit(f"missing summary: {summary_path}")

    configure_matplotlib()
    original, adjusted = load_report_endpoints(summary_path)
    clean_output_dir(output_dir, args.formats)
    generated = render_figure(
        original,
        adjusted,
        output_dir,
        args.formats,
        args.dpi,
    )
    generated.extend(
        render_condition_figure(
            original,
            adjusted,
            output_dir,
            args.formats,
            args.dpi,
        )
    )

    print("Generated Step 5 figure:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__":
    main()
