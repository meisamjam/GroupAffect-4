"""Generate paper-ready result figures.

Outputs
-------
  paper/figure/benchmarks/ablation_heatmap.pdf
  paper/figure/benchmarks/benchmark_overview.pdf

Usage
-----
  cd GroupAffect-4-data-processing
  py -3 tools/features/plot_results_figures.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

OUT_DIR = Path("paper/figure/benchmarks")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ABLATION_TSV = Path("results/benchmarks/ablation_results.tsv")
BASELINES_TSV = Path("results/benchmarks/baselines_preprocessed_final_preprocessed.tsv")

# â”€â”€ Styling â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
FONT = "DejaVu Sans"
plt.rcParams.update({
    "font.family": FONT,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# â”€â”€ Shared constants â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

COND_ORDER = ["cardiac", "eda", "pupil", "audio", "bfi",
              "card+eda", "card+pup", "eda+pup", "sensor", "all"]
COND_LABELS = {
    "cardiac":  "Cardiac",
    "eda":      "EDA/GSR",
    "pupil":    "Pupil",
    "audio":    "Audio",
    "bfi":      "BFI",
    "card+eda": "Card+EDA",
    "card+pup": "Card+Pup",
    "eda+pup":  "EDA+Pup",
    "sensor":   "Sensor",
    "all":      "All+BFI",
}

# Chance / naive baseline for each (benchmark_id, target) pair
CHANCE = {
    ("B0_task",        "Task label (T1-T4)"): (0.265, True),
    ("B1_affective",   "Valence"):            (1.030, False),
    ("B1_affective",   "Arousal"):            (1.089, False),
    ("B2_dominance",   "Dominance (high/low)"): (0.518, True),
    ("B3_cognitive",   "Engagement"):         (0.535, True),
    ("B3_cognitive",   "Mental demand"):      (0.545, True),
    ("B4_personality", "BFI Extraversion"):   (0.482, False),
    ("B4_personality", "BFI Openness"):       (0.404, False),
    ("B5_performance", "T4 Contribution"):    (0.481, False),
    ("B7_floor",       "Floor dominance"):    (0.500, True),   # AUC baseline = random (0.5); 3:1 imbalance makes accuracy misleading
    ("Bs_speaking",    "Speaking Gini"):      (0.086, False),
    ("B6_overlap",     "Speech overlap"):     (0.079, False),
}

# â”€â”€ Figure 1: Ablation heatmap â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def _rel_performance(val: float, chance: float, best: float, higher: bool) -> float:
    """Map val to [0, 1] where 0=chance and 1=best."""
    if math.isnan(val):
        return math.nan
    if higher:
        denom = best - chance
        return 0.0 if denom < 1e-9 else (val - chance) / denom
    else:
        denom = chance - best
        return 0.0 if denom < 1e-9 else (chance - val) / denom


ROW_LABELS = [
    ("B0_task",        "Task label (T1-T4)", "B0: Task (clf)"),
    ("B1_affective",   "Valence (high/low)",   "B1: Valence (clf)"),
    ("B1_affective",   "Arousal (high/low)",   "B1: Arousal (clf)"),
    ("B2_dominance",   "Dominance (high/low)", "B2: Dominance (clf)"),
    ("B3_cognitive",   "Engagement",          "B3: Engagement (clf)"),
    ("B3_cognitive",   "Mental demand",       "B3: Mental demand (clf)"),
    ("B4_personality", "BFI Extraversion (high/low)", "B4: Extraversion (clf)"),
    ("B4_personality", "BFI Openness (high/low)",     "B4: Openness (clf)"),
    ("B5_performance", "T4 Contribution (high/low)",  "B5: Contribution (clf)"),
    ("Bs_speaking",    "Speaking Gini",       "B6: Speaking Gini"),
    ("B6_overlap",     "Speech overlap",      "B7: Speech overlap"),
]


def plot_ablation_heatmap(df: pd.DataFrame, path: Path) -> None:
    n_rows = len(ROW_LABELS)
    n_cols = len(COND_ORDER)

    grid = np.full((n_rows, n_cols), np.nan)
    abs_grid = np.full((n_rows, n_cols), np.nan)  # actual values for annotations

    for ri, (bid, tgt, _) in enumerate(ROW_LABELS):
        sub = df[(df["benchmark_id"] == bid) & (df["target"] == tgt)]
        if sub.empty:
            continue
        chance_val, higher = CHANCE.get((bid, tgt), (np.nan, True))
        # best across conditions
        vals = {r["condition"]: r["primary_value"] for _, r in sub.iterrows()
                if not math.isnan(r["primary_value"])}
        if not vals:
            continue
        best_val = max(vals.values()) if higher else min(vals.values())
        for ci, cond in enumerate(COND_ORDER):
            v = vals.get(cond, np.nan)
            abs_grid[ri, ci] = v
            if not math.isnan(v) and not math.isnan(chance_val):
                grid[ri, ci] = _rel_performance(v, chance_val, best_val, higher)

    fig, ax = plt.subplots(figsize=(7.0, 3.6))

    # Custom colormap: light gray (NaN/chance) â†’ teal/green (best)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list(
        "ablation", ["#f7f7f7", "#c7e9b4", "#41b6c4", "#225ea8"], N=256)
    cmap.set_bad(color="#e8e8e8")

    im = ax.imshow(grid, aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")

    # Annotations: show actual value in each cell
    for ri in range(n_rows):
        for ci in range(n_cols):
            v = abs_grid[ri, ci]
            if math.isnan(v):
                ax.text(ci, ri, "â€”", ha="center", va="center",
                        fontsize=6.5, color="#aaaaaa")
            else:
                rel = grid[ri, ci]
                txt_color = "white" if (not math.isnan(rel) and rel > 0.65) else "#222222"
                ax.text(ci, ri, f"{v:.3f}", ha="center", va="center",
                        fontsize=6.0, color=txt_color, fontweight="normal")
                # Mark best with bold border
                is_best = abs(rel - 1.0) < 1e-6 if not math.isnan(rel) else False
                if is_best:
                    rect = plt.Rectangle((ci - 0.48, ri - 0.48), 0.96, 0.96,
                                         fill=False, edgecolor="#1a1a1a", linewidth=1.2)
                    ax.add_patch(rect)

    # Labels
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([COND_LABELS[c] for c in COND_ORDER], rotation=30, ha="right")
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([lbl for _, _, lbl in ROW_LABELS])

    # Draw horizontal group separators
    separators = [2.5, 3.5, 5.5, 7.5, 8.5, 9.5, 10.5]
    for s in separators:
        ax.axhline(s, color="white", linewidth=1.5)

    # Vertical separator between single-modality and combos
    ax.axvline(3.5, color="white", linewidth=1.5, linestyle="--", alpha=0.7)

    ax.set_title("Modality ablation: relative performance (0 = chance, 1 = best in row)",
                 pad=6, fontsize=8.5)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Rel. performance", fontsize=7)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["chance", "mid", "best"])
    cbar.ax.tick_params(labelsize=7)

    # Column group annotations at top
    ax.annotate("Single modality", xy=(1.5, -0.9), xycoords=("data", "axes fraction"),
                fontsize=7, ha="center", color="#555555",
                xytext=(1.5, -0.12), textcoords=("data", "axes fraction"),
                annotation_clip=False)
    ax.annotate("Combinations", xy=(6.5, -0.9), xycoords=("data", "axes fraction"),
                fontsize=7, ha="center", color="#555555",
                xytext=(6.5, -0.12), textcoords=("data", "axes fraction"),
                annotation_clip=False)

    fig.tight_layout()
    fig.savefig(path, format="pdf")
    fig.savefig(path.with_suffix(".png"), format="png")
    plt.close(fig)
    print(f"Saved {path}")


# â”€â”€ Figure 2: Benchmark overview bar chart â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

# (benchmark label, baseline, linear, metric_label, higher_better)
# Values taken from baselines_preprocessed_final_preprocessed.tsv (primary metric per benchmark)
# Post-fix expected values noted in comments (re-run runner to confirm actual results).
BENCHMARK_BARS = [
    # Classification (accuracy, higher = better)
    ("B0\nTask",         0.265, 0.703, "Accuracy", True),
    ("B2\nDominance",    0.518, 0.627, "Accuracy", True),
    ("B3\nEngagement",   0.535, 0.596, "Accuracy", True),
    ("B3\nMental Dem.",  0.545, 0.747, "Accuracy", True),
    # B1 uses fold-local majority accuracy as the operational baseline.
    ("B1\nValence",      0.636, 0.589, "Accuracy", True),
    ("B1\nArousal",      0.439, 0.570, "Accuracy", True),
    # B4 binary classification (n=32 participants; challenge benchmark).
    ("B4\nExtraversion", 0.594, 0.562, "Accuracy", True),
    ("B4\nOpenness",     0.594, 0.344, "Accuracy", True),
    # B5 binary classification (n=28; fragile median-split proxy).
    ("B5\nContrib.",     0.500, 0.821, "Accuracy", True),
    # Group-level regression targets do not beat the naive baseline in the rerun.
    ("B6\nSpeaking",     0.086, 0.089, "MAE",     False),
    ("B7\nOverlap",      0.060, 0.063, "MAE",     False),
]

GROUP_CUTS = [3.5, 5.5]  # separator positions between classification / affective / trait

COLORS = {
    "baseline": "#b0b8c4",
    "good":     "#2196F3",   # better than baseline
    "bad":      "#EF5350",   # worse than baseline
}


def plot_benchmark_overview(path: Path) -> None:
    def _beats(base: float, lin: float, hi: bool) -> bool:
        return (lin > base) if hi else (lin < base)

    def _pct_improvement(base: float, lin: float, hi: bool) -> float:
        """Percent improvement over baseline (always positive when beating)."""
        return ((lin - base) / base * 100) if hi else ((base - lin) / base * 100)

    # Keep only benchmarks where the linear model beats the naive baseline
    clf_rows = [(label, base, lin, met, hi) for label, base, lin, met, hi in BENCHMARK_BARS
                if met in ("Accuracy", "AUC") and _beats(base, lin, hi)]
    reg_rows = [(label, base, lin, met, hi) for label, base, lin, met, hi in BENCHMARK_BARS
                if met == "MAE" and _beats(base, lin, hi)]

    all_rows = clf_rows + reg_rows
    if not all_rows:
        print("WARNING: no benchmarks beat baseline â€” skipping benchmark_overview figure")
        return

    all_pct = [_pct_improvement(r[1], r[2], r[4]) for r in all_rows]
    n = len(all_rows)
    bar_colors = [COLORS["good"]] * len(clf_rows) + ["#43A047"] * len(reg_rows)

    fig, ax = plt.subplots(figsize=(max(4.0, n * 1.05), 3.8))
    xi = np.arange(n)
    ax.bar(xi, all_pct, width=0.55, color=bar_colors, zorder=2)

    # Vertical separator when both panels are populated
    if clf_rows and reg_rows:
        ax.axvline(len(clf_rows) - 0.5, color="#aaaaaa", linewidth=1.0,
                   linestyle=":", zorder=1)

    y_max = max(all_pct)

    # Value labels inside/on top of each bar (no below-axis clutter)
    for xi_, v, r in zip(xi, all_pct, all_rows):
        # show percentage above bar
        ax.text(xi_, v + y_max * 0.015, f"+{v:.1f}%", ha="center", va="bottom",
                fontsize=7.0, color="#111111", fontweight="bold")
        # show absolute model value inside the bar (bottom-anchored)
        abs_val = r[2]
        abs_lbl = f"{abs_val:.3f}"
        bar_mid = v / 2
        if v > y_max * 0.12:
            ax.text(xi_, bar_mid, abs_lbl, ha="center", va="center",
                    fontsize=6.2, color="white", fontweight="bold")

    # Clean two-line x-labels â€” replace \n with a space so they stay single-line
    tick_labels = [r[0].replace("\n", " ") for r in all_rows]
    ax.set_xticks(xi)
    ax.set_xticklabels(tick_labels, fontsize=8.0, rotation=25, ha="right")

    ax.set_ylabel("Improvement over naive baseline  (%)", fontsize=8)
    ax.set_ylim(0, y_max * 1.28)
    ax.set_title("Benchmarks that beat the naive baseline", fontsize=9.5, pad=5)
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4, zorder=1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Group labels
    if clf_rows:
        mid_clf = (len(clf_rows) - 1) / 2
        ax.text(mid_clf, y_max * 1.21, "Classification / AUC  (vs. baseline)",
                ha="center", fontsize=7.0, color=COLORS["good"], style="italic")
    if reg_rows:
        mid_reg = len(clf_rows) + (len(reg_rows) - 1) / 2
        ax.text(mid_reg, y_max * 1.21, "Regression  (MAE vs. mean baseline)",
                ha="center", fontsize=7.0, color="#43A047", style="italic")

    # Legend patches
    legend_patches: list[mpatches.Patch] = []
    if clf_rows:
        legend_patches.append(mpatches.Patch(color=COLORS["good"], label="Classification (Acc.)"))
    if reg_rows:
        legend_patches.append(mpatches.Patch(color="#43A047", label="Regression (MAE)"))
    if legend_patches:
        ax.legend(handles=legend_patches, fontsize=7, loc="upper right",
                  framealpha=0.7, edgecolor="none")

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.18)  # extra room for rotated labels
    fig.savefig(path, format="pdf")
    fig.savefig(path.with_suffix(".png"), format="png")
    plt.close(fig)
    print(f"Saved {path}  ({n} benchmarks beat baseline)")


# â”€â”€ Main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main() -> None:
    df = pd.read_csv(ABLATION_TSV, sep="\t")
    plot_ablation_heatmap(df, OUT_DIR / "ablation_heatmap.pdf")
    plot_benchmark_overview(OUT_DIR / "benchmark_overview.pdf")


if __name__ == "__main__":
    main()

