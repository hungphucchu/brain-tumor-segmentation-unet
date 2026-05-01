#!/usr/bin/env python3
"""
Generate PNG charts for slides. Run from repo root:
  python presentation/generate_figures.py
Outputs go to presentation/figures/
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO = Path(__file__).resolve().parent.parent
FIG_DIR = Path(__file__).resolve().parent / "figures"


def load_split_counts() -> tuple[int, int, int]:
    path = REPO / "data" / "splits" / "splits.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return len(data["train"]), len(data["val"]), len(data["test"])


def plot_data_split(train: int, val: int, test: int, out: Path) -> None:
    labels = ["Train", "Val", "Test"]
    counts = [train, val, test]
    colors = ["#2563eb", "#64748b", "#94a3b8"]
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, counts, color=colors, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("Number of slice pairs (image + mask)")
    ax.set_title("LGG dataset — split sizes in this project")
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 80, str(c), ha="center", fontsize=11)
    ax.set_ylim(0, max(counts) * 1.12)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_test_metrics(out: Path) -> None:
    # From: python -m src.eval --split test --max-overlays 0
    names = ["Dice", "IoU", "Precision\n(pixel)", "Sensitivity\n(recall)"]
    values = [0.8116, 0.7771, 0.8552, 0.8024]
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(names))
    bars = ax.bar(x, values, color="#0d9488", edgecolor="white", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("Test set — segmentation metrics (sigmoid threshold = 0.5)")
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02, f"{v:.4f}", ha="center", fontsize=9)
    ax.axhline(0.5, color="#cbd5e1", linestyle="--", linewidth=1, label="0.5 reference")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_unet_schematic(out: Path) -> None:
    """Simple U-shaped diagram (not to scale) — for audience orientation only."""
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("U-Net idea: down path (encode) + up path (decode) + skip connections", fontsize=11)

    def box(x, y, w, h, label: str, fc: str) -> None:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=fc, edgecolor="#334155", linewidth=1.5))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=8, color="#0f172a")

    # Encoder (left, going down)
    box(0.5, 4.5, 1.2, 1.0, "Enc1", "#bfdbfe")
    box(0.5, 3.0, 1.0, 0.9, "↓", "#e2e8f0")
    box(0.6, 1.8, 1.0, 0.9, "Enc2", "#93c5fd")
    box(0.7, 0.5, 0.9, 0.85, "…", "#e2e8f0")
    # Bottleneck
    box(4.0, 0.4, 1.4, 1.0, "Bottle\nneck", "#6366f1")
    # Decoder (right, going up)
    box(7.5, 0.5, 0.9, 0.85, "…", "#e2e8f0")
    box(7.4, 1.8, 1.0, 0.9, "Dec2", "#a7f3d0")
    box(7.3, 3.0, 1.0, 0.9, "↑", "#e2e8f0")
    box(7.2, 4.5, 1.2, 1.0, "Dec1", "#6ee7b7")
    box(8.5, 4.5, 1.2, 1.0, "1×1\nlogits", "#fcd34d")

    # Skip arcs (conceptual)
    for (x1, y1), (x2, y2) in [((1.7, 5.0), (7.2, 5.0)), ((1.6, 3.4), (7.4, 3.4)), ((1.6, 2.2), (7.4, 2.2))]:
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-", color="#94a3b8", linestyle="--", lw=1.2),
        )
    ax.text(5.0, 5.5, "skip\nconnections", ha="center", fontsize=8, color="#64748b")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    tr, va, te = load_split_counts()
    plot_data_split(tr, va, te, FIG_DIR / "data_split_counts.png")
    plot_test_metrics(FIG_DIR / "test_metrics_bar.png")
    plot_unet_schematic(FIG_DIR / "unet_schematic.png")
    print(f"Wrote figures under {FIG_DIR}:")
    for p in sorted(FIG_DIR.glob("*.png")):
        print(f"  - {p.relative_to(REPO)}")


if __name__ == "__main__":
    main()
