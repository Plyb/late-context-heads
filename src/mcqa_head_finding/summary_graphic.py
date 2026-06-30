"""Generate a summary figure for the MCQA head-finding project.

Run from project root:
    uv run python -m mcqa_head_finding.summary_graphic
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, PathPatch
from matplotlib.path import Path as MplPath


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CONFIGS: list[dict] = [
    dict(
        title="Llama-3.2-3B",
        subtitle="Token-match",
        accuracy=dict(question_first=1.00, options_first=1.00),
        log_dir="logs/20260530_152944_analyze_synthetic_llama-3.2-3b_12028877",
    ),
    dict(
        title="Llama-3.2-3B-Instruct",
        subtitle="Vocab (def→word)",
        accuracy=dict(question_first=1.00, options_first=0.98),
        log_dir="logs/20260601_143618_analyze_vocab_llama-3.2-3b-instruct_12031970",
    ),
    dict(
        title="Llama-3.1-70B-Instruct",
        subtitle="ARC-Easy",
        accuracy=dict(question_first=0.99, options_first=0.92),
        log_dir="logs/20260601_182637_analyze_arc-easy_llama-3.1-70b-instruct_12039830",
    ),
]

HEAD_COLORS: dict[str, str] = {
    "correct_newline": "#2196F3",
    "correct_label":   "#4CAF50",
    "late":            "#FF9800",
    "constant":        "#9C27B0",
    "single_answer":   "#E91E63",
    "other":           "#78909C",
}

HEAD_TYPE_LABELS: dict[str, str] = {
    "correct_newline": "correct-answer newline",
    "correct_label":   "correct-answer label",
    "late":            "late attention",
    "constant":        "constant attention",
    "single_answer":   "single-answer",
    "other":           "other",
}

# Layout constants (data units = inches, because we size axes to match)
CHIP_W       = 0.85
CHIP_H       = 0.18
CHIP_GAP     = 0.06
SLOT_H       = CHIP_H + CHIP_GAP
COL_GAP      = 0.55
MARGIN_X     = 0.22
PANEL_W      = 2 * CHIP_W + COL_GAP + 2 * MARGIN_X

HEADER_H      = 0.70
COL_HEADER_H  = 0.50
BOTTOM_MARGIN = 0.25

QF_CX     = MARGIN_X + CHIP_W / 2
OF_CX     = MARGIN_X + CHIP_W + COL_GAP + CHIP_W / 2
QF_RIGHT  = QF_CX + CHIP_W / 2
OF_LEFT   = OF_CX - CHIP_W / 2
CTRL_DIST = COL_GAP * 0.38

INTER_PANEL  = 0.30
OUTER_MARGIN = 0.18
LEGEND_H     = 0.60


def load_format_data(
    log_dir: str, fmt: str
) -> tuple[list[str], np.ndarray, list[str]]:
    base = PROJECT_ROOT / log_dir / "output" / fmt
    with open(base / "top_heads.json") as f:
        th = json.load(f)
    vwa = np.load(base / "vwa_per_head.npy")
    with open(base / "vwa_per_head_columns.json") as f:
        cols = json.load(f)
    return th["labels"], vwa, cols


def classify_head(head_data: np.ndarray, col_names: list[str]) -> str:
    """Classify a head by its dominant value-weighted attention pattern."""
    nl_cols = [col_names.index(f"{a}_nl") for a in "ABCD"]
    label_cols = [col_names.index(f"label_{a}") for a in "ABCD"]

    late_cols: set[int] = set()
    for key in ("ans1::", "ans2:("):
        try:
            late_cols.add(col_names.index(key))
        except ValueError:
            pass

    ans_specific: set[int] = set(nl_cols + label_cols)
    for a in "ABCD":
        for s in ("prefix", "m3", "m2", "m1"):
            try:
                ans_specific.add(col_names.index(f"{a}_{s}"))
            except ValueError:
                pass

    row_argmaxes = [int(np.argmax(head_data[r])) for r in range(4)]

    if sum(a in late_cols for a in row_argmaxes) >= 3:
        return "late"
    if sum(row_argmaxes[i] == nl_cols[i] for i in range(4)) >= 3:
        return "correct_newline"
    if sum(row_argmaxes[i] == label_cols[i] for i in range(4)) >= 3:
        return "correct_label"
    if len(set(row_argmaxes)) == 1:
        col = row_argmaxes[0]
        return "single_answer" if col in ans_specific else "constant"
    return "other"


def chip_cy(panel_h: float, idx: int) -> float:
    """Y-center for chip at rank idx (0 = top of list)."""
    top = panel_h - HEADER_H - COL_HEADER_H
    return top - idx * SLOT_H - CHIP_H / 2


def draw_chip(ax: plt.Axes, cx: float, cy: float, label: str, color: str) -> None:
    ax.add_patch(FancyBboxPatch(
        (cx - CHIP_W / 2, cy - CHIP_H / 2),
        CHIP_W, CHIP_H,
        boxstyle="round,pad=0.012",
        facecolor=color,
        edgecolor="none",
        zorder=4,
    ))
    ax.text(cx, cy, label, ha="center", va="center",
            color="white", fontsize=5.5, fontweight="bold", zorder=5)


def draw_connection(
    ax: plt.Axes, y_qf: float, y_of: float, color: str
) -> None:
    verts = [
        (QF_RIGHT, y_qf),
        (QF_RIGHT + CTRL_DIST, y_qf),
        (OF_LEFT - CTRL_DIST, y_of),
        (OF_LEFT, y_of),
    ]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4]
    ax.add_patch(PathPatch(
        MplPath(verts, codes),
        facecolor="none",
        edgecolor=color,
        linewidth=0.8,
        alpha=0.60,
        zorder=3,
    ))


def render_panel(ax: plt.Axes, cfg: dict, panel_h: float) -> set[str]:
    ax.set_xlim(0, PANEL_W)
    ax.set_ylim(0, panel_h)
    ax.axis("off")

    ax.text(PANEL_W / 2, panel_h - 0.10, cfg["title"],
            ha="center", va="top", fontsize=7.5, fontweight="bold")
    ax.text(PANEL_W / 2, panel_h - 0.35, cfg["subtitle"],
            ha="center", va="top", fontsize=6.5, color="#555555")

    qf_labels, qf_vwa, col_names = load_format_data(cfg["log_dir"], "question_first")
    of_labels, of_vwa, _         = load_format_data(cfg["log_dir"], "options_first")

    qf_types = [classify_head(qf_vwa[i], col_names) for i in range(len(qf_labels))]
    of_types = [classify_head(of_vwa[i], col_names) for i in range(len(of_labels))]

    col_header_y = panel_h - HEADER_H
    for cx, fkey, flabel in [
        (QF_CX, "question_first", "question-first"),
        (OF_CX, "options_first",  "options-first"),
    ]:
        ax.text(cx, col_header_y - 0.05, flabel,
                ha="center", va="top", fontsize=6.0, fontweight="bold", color="#333333")
        ax.text(cx, col_header_y - 0.30, f"acc = {cfg['accuracy'][fkey]:.2f}",
                ha="center", va="top", fontsize=5.5, color="#666666")

    for i, (lbl, htype) in enumerate(zip(qf_labels, qf_types)):
        draw_chip(ax, QF_CX, chip_cy(panel_h, i), lbl, HEAD_COLORS[htype])
    for i, (lbl, htype) in enumerate(zip(of_labels, of_types)):
        draw_chip(ax, OF_CX, chip_cy(panel_h, i), lbl, HEAD_COLORS[htype])

    of_rank = {lbl: i for i, lbl in enumerate(of_labels)}
    for qi, (lbl, htype) in enumerate(zip(qf_labels, qf_types)):
        if lbl in of_rank:
            draw_connection(ax, chip_cy(panel_h, qi), chip_cy(panel_h, of_rank[lbl]),
                            HEAD_COLORS[htype])

    return set(qf_types) | set(of_types)


def panel_height(cfg: dict) -> float:
    """Compute panel height based on the longest head list in this config."""
    qf_path = PROJECT_ROOT / cfg["log_dir"] / "output" / "question_first" / "top_heads.json"
    of_path = PROJECT_ROOT / cfg["log_dir"] / "output" / "options_first"  / "top_heads.json"
    with open(qf_path) as f:
        n_qf = len(json.load(f)["labels"])
    with open(of_path) as f:
        n_of = len(json.load(f)["labels"])
    n = max(n_qf, n_of)
    return HEADER_H + COL_HEADER_H + n * SLOT_H + BOTTOM_MARGIN


def main() -> None:
    panel_heights = [panel_height(cfg) for cfg in CONFIGS]
    max_panel_h = max(panel_heights)

    fig_w = 3 * PANEL_W + 2 * INTER_PANEL + 2 * OUTER_MARGIN
    fig_h = max_panel_h + LEGEND_H + 2 * OUTER_MARGIN

    fig = plt.figure(figsize=(fig_w, fig_h))

    panel_top_in = fig_h - OUTER_MARGIN  # all panels aligned at top

    seen_types: set[str] = set()
    for i, (cfg, ph) in enumerate(zip(CONFIGS, panel_heights)):
        left_in   = OUTER_MARGIN + i * (PANEL_W + INTER_PANEL)
        bottom_in = panel_top_in - ph
        ax = fig.add_axes([
            left_in   / fig_w,
            bottom_in / fig_h,
            PANEL_W   / fig_w,
            ph        / fig_h,
        ])
        seen_types |= render_panel(ax, cfg, ph)

    legend_patches = [
        mpatches.Patch(color=HEAD_COLORS[t], label=HEAD_TYPE_LABELS[t])
        for t in HEAD_TYPE_LABELS
        if t in seen_types
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=len(legend_patches),
        fontsize=6.0,
        frameon=False,
        bbox_to_anchor=(0.5, 0.005),
    )

    out = PROJECT_ROOT / "figures" / "summary.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
