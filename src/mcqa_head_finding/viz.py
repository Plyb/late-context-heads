"""Render the value-weighted attention heatmaps."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .direct_effect import ANSWER_LETTERS


def save_main_heatmap(
    matrix: np.ndarray,
    head_labels: list[str],
    col_names: list[str],
    out_path: Path,
    title: str,
) -> None:
    """Y = top heads, X = token types; cell = mean value-weighted attention."""
    n_heads, n_cols = matrix.shape
    fig, ax = plt.subplots(figsize=(max(12, n_cols * 0.22), max(4, n_heads * 0.28)))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")
    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_heads))
    ax.set_xticklabels(col_names, rotation=90, fontsize=4)
    ax.set_yticklabels(head_labels, fontsize=6)
    ax.set_xlabel("token type", fontsize=8)
    ax.set_ylabel("head", fontsize=8)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_per_head_heatmaps(
    per_head: np.ndarray,
    head_labels: list[str],
    col_names: list[str],
    out_dir: Path,
    title_prefix: str,
) -> None:
    """For each top head: Y = which answer was correct (A/B/C/D), X = token type."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n_cols = per_head.shape[2]
    for h, label in enumerate(head_labels):
        fig, ax = plt.subplots(figsize=(max(10, n_cols * 0.22), 3.0))
        im = ax.imshow(per_head[h], cmap="viridis", aspect="auto")
        ax.set_xticks(range(n_cols))
        ax.set_yticks(range(len(ANSWER_LETTERS)))
        ax.set_xticklabels(col_names, rotation=90, fontsize=4)
        ax.set_yticklabels(list(ANSWER_LETTERS), fontsize=7)
        ax.set_xlabel("token type", fontsize=8)
        ax.set_ylabel("correct answer", fontsize=8)
        ax.set_title(f"{title_prefix} | {label}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        fig.tight_layout()
        fig.savefig(out_dir / f"head_{label}.png", dpi=150)
        plt.close(fig)
