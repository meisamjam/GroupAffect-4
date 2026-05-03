"""BFI-44 personality analysis — individual and group-level.

Reads participants.tsv from the BIDS root and produces:

  results/personality/personality_summary.tsv  — enriched per-participant table
  results/personality/group_trait_stats.tsv    — per-group mean / SD / range per trait
  results/personality/group_similarity.tsv     — pairwise within-group distances

  figures/personality/personality_distributions.png   — trait distributions + norms
  figures/personality/personality_group_heatmap.png   — group mean profiles heatmap
  figures/personality/personality_group_profiles.png  — group radar / bar profiles
  figures/personality/personality_demographics.png    — age / sex / education overlays
  figures/personality/personality_correlations.png    — inter-trait correlation matrix
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("analyze_personality")

TRAITS = ["bfi44_e", "bfi44_a", "bfi44_c", "bfi44_n", "bfi44_o"]
TRAIT_LABELS = {
    "bfi44_e": "Extraversion",
    "bfi44_a": "Agreeableness",
    "bfi44_c": "Conscientiousness",
    "bfi44_n": "Neuroticism",
    "bfi44_o": "Openness",
}
SHORT = {k: v[:3] for k, v in TRAIT_LABELS.items()}

# BFI-44 adult population norms (approximate, 1–5 scale)
# Source: John & Srivastava (1999) college samples; used only for visual reference.
NORMS_MEAN = {"bfi44_e": 3.24, "bfi44_a": 3.87, "bfi44_c": 3.45,
               "bfi44_n": 2.97, "bfi44_o": 3.83}
NORMS_SD   = {"bfi44_e": 0.83, "bfi44_a": 0.65, "bfi44_c": 0.71,
               "bfi44_n": 0.83, "bfi44_o": 0.59}

PALETTE = ["#2f6f9f", "#a33f2f", "#6b7f2a", "#7c4d9e", "#5a8f7b"]


# ── Data loading & enrichment ─────────────────────────────────────────────────

def load_participants(bids_root: Path) -> pd.DataFrame:
    path = bids_root / "participants.tsv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, sep="\t")
    # Extract group number for ordering
    df["group_num"] = df["group_id"].str.extract(r"grp-(\d+)").astype(float)
    df = df.sort_values(["group_num", "seat"]).reset_index(drop=True)
    # z-score traits within the sample
    for t in TRAITS:
        col = df[t]
        mu, sd = col.mean(), col.std(ddof=1)
        df[f"{t}_z"] = (col - mu) / sd if sd > 0 else 0.0
    return df


def build_group_stats(df: pd.DataFrame) -> pd.DataFrame:
    bfi_df = df[df[TRAITS].notna().all(axis=1)]
    rows = []
    for grp, sub in bfi_df.groupby("group_id"):
        row: dict = {"group_id": grp, "n": len(sub)}
        for t in TRAITS:
            row[f"{t}_mean"] = sub[t].mean()
            row[f"{t}_sd"]   = sub[t].std(ddof=1) if len(sub) > 1 else float("nan")
            row[f"{t}_min"]  = sub[t].min()
            row[f"{t}_max"]  = sub[t].max()
        # Within-group diversity: mean pairwise Euclidean distance in trait space
        mat = sub[TRAITS].to_numpy(dtype=float)
        if len(mat) > 1:
            dists = []
            for i in range(len(mat)):
                for j in range(i + 1, len(mat)):
                    dists.append(float(np.linalg.norm(mat[i] - mat[j])))
            row["mean_pairwise_dist"] = float(np.mean(dists))
        else:
            row["mean_pairwise_dist"] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("group_id").reset_index(drop=True)


# ── Figure 1: trait distributions ────────────────────────────────────────────

def _distributions(df: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    fig, axes = plt.subplots(1, 5, figsize=(16, 4.2), constrained_layout=True)
    fig.suptitle("BFI-44 trait distributions (n = %d participants, 10 groups)" % df[TRAITS[0]].notna().sum(),
                 fontsize=13, fontweight="bold")

    for ax, trait, color in zip(axes, TRAITS, PALETTE):
        vals = df[trait].dropna().to_numpy()
        label = TRAIT_LABELS[trait]
        ax.hist(vals, bins=10, range=(1, 5), color=color, alpha=0.80, edgecolor="white", linewidth=0.5)
        # Sample mean ± SD
        mu, sd = vals.mean(), vals.std(ddof=1)
        ax.axvline(mu, color="#222222", linewidth=1.8, linestyle="-", label=f"mean {mu:.2f}")
        ax.axvspan(mu - sd, mu + sd, alpha=0.12, color="#222222", label=f"±1 SD")
        # Population norm band
        n_mu = NORMS_MEAN[trait]
        n_sd = NORMS_SD[trait]
        ax.axvline(n_mu, color="#888888", linewidth=1.2, linestyle="--", label=f"norm {n_mu:.2f}")
        ax.set_title(label, fontsize=10, fontweight="bold")
        ax.set_xlabel("Score (1–5)")
        ax.set_xlim(1, 5)
        ax.set_ylabel("Count" if ax == axes[0] else "")
        ax.legend(frameon=False, fontsize=7.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    out_path = out_dir / "personality_distributions.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ── Figure 2: inter-trait correlation matrix ──────────────────────────────────

def _correlations(df: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    bfi = df[TRAITS].dropna()
    corr = bfi.corr(method="spearman")
    labels = [TRAIT_LABELS[t] for t in TRAITS]

    fig, ax = plt.subplots(figsize=(6.5, 5.5), constrained_layout=True)
    fig.suptitle("BFI-44 inter-trait Spearman correlations", fontsize=12, fontweight="bold")

    im = ax.imshow(corr.to_numpy(), vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax.set_xticks(range(5), labels, rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(5), labels, fontsize=9)
    for i in range(5):
        for j in range(5):
            r = corr.iloc[i, j]
            ax.text(j, i, f"{r:.2f}", ha="center", va="center",
                    fontsize=9, color="white" if abs(r) > 0.5 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label("Spearman r", fontsize=9)

    out_path = out_dir / "personality_correlations.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ── Figure 3: group mean heatmap ─────────────────────────────────────────────

def _group_heatmap(gs: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    mean_cols = [f"{t}_mean" for t in TRAITS]
    mat = gs.set_index("group_id")[mean_cols].to_numpy(dtype=float)
    groups = gs["group_id"].tolist()
    trait_names = [TRAIT_LABELS[t] for t in TRAITS]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), constrained_layout=True,
                             gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle("Group-level BFI-44 profiles", fontsize=13, fontweight="bold")

    # Heatmap panel
    im = axes[0].imshow(mat, vmin=1, vmax=5, cmap="YlOrRd", aspect="auto")
    axes[0].set_xticks(range(5), trait_names, fontsize=9)
    axes[0].set_yticks(range(len(groups)), groups, fontsize=9)
    axes[0].set_title("Mean trait score per group (1–5 scale)", fontsize=10)
    for i in range(len(groups)):
        for j in range(5):
            v = mat[i, j]
            axes[0].text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=8, color="black" if v < 3.5 else "white")
    fig.colorbar(im, ax=axes[0], fraction=0.03, pad=0.02).set_label("Mean score")

    # Within-group diversity (mean pairwise distance)
    divs = gs["mean_pairwise_dist"].to_numpy(dtype=float)
    bars = axes[1].barh(range(len(groups)), divs, color="#5a8f7b", alpha=0.85)
    axes[1].set_yticks(range(len(groups)), groups, fontsize=9)
    axes[1].set_xlabel("Mean pairwise trait distance")
    axes[1].set_title("Within-group\nBFI diversity", fontsize=10)
    for bar, v in zip(bars, divs):
        if np.isfinite(v):
            axes[1].text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                         f"{v:.2f}", va="center", fontsize=8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)
    axes[1].axvline(np.nanmean(divs), color="#333333", linewidth=1,
                    linestyle="--", alpha=0.6, label="mean")
    axes[1].legend(frameon=False, fontsize=8)

    out_path = out_dir / "personality_group_heatmap.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ── Figure 4: group profiles (bar chart) ─────────────────────────────────────

def _group_profiles(df: pd.DataFrame, gs: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    groups = gs["group_id"].tolist()
    n_groups = len(groups)
    ncols = 5
    nrows = int(np.ceil(n_groups / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 3.2 * nrows),
                             constrained_layout=True, sharey=True)
    fig.suptitle("Per-group BFI-44 profiles: individual dots + group mean bar",
                 fontsize=13, fontweight="bold")

    flat_axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    x = np.arange(5)
    trait_abbr = [SHORT[t] for t in TRAITS]

    for ax, grp in zip(flat_axes, groups):
        grp_df = df[df["group_id"] == grp]
        for _, row in grp_df.iterrows():
            vals = [row.get(t, np.nan) for t in TRAITS]
            if all(np.isfinite(vals)):
                ax.plot(x, vals, color="#aaaaaa", linewidth=0.8, marker="o",
                        markersize=4, alpha=0.7, zorder=2)

        means = gs.loc[gs["group_id"] == grp, [f"{t}_mean" for t in TRAITS]].values.flatten()
        ax.bar(x, means, color=PALETTE, alpha=0.55, width=0.55, zorder=1)
        # Norm reference line
        norm_means = [NORMS_MEAN[t] for t in TRAITS]
        ax.plot(x, norm_means, color="#555555", linewidth=1.2,
                linestyle="--", zorder=3, alpha=0.6)

        ax.set_title(grp, fontsize=9, fontweight="bold")
        ax.set_xticks(x, trait_abbr, fontsize=8)
        ax.set_ylim(1, 5)
        ax.axhline(3, color="#cccccc", linewidth=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in flat_axes[n_groups:]:
        ax.set_axis_off()

    # Add shared legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], color="#aaaaaa", marker="o", markersize=5, label="individual"),
        Line2D([0], [0], color="#555555", linestyle="--", label="population norm"),
    ]
    fig.legend(handles=handles, loc="lower right", frameon=False, fontsize=9)

    out_path = out_dir / "personality_group_profiles.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ── Figure 5: demographics ────────────────────────────────────────────────────

def _demographics(df: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    valid = df[df[TRAITS].notna().all(axis=1)].copy()

    fig, axes = plt.subplots(2, 5, figsize=(16, 8), constrained_layout=True)
    fig.suptitle("BFI-44 traits by demographic variable", fontsize=13, fontweight="bold")

    sex_groups = {"male": "#2f6f9f", "female": "#a33f2f"}

    for col_idx, (trait, color) in enumerate(zip(TRAITS, PALETTE)):
        label = TRAIT_LABELS[trait]

        # Row 0: age scatter coloured by sex
        ax = axes[0, col_idx]
        for sex, sc in sex_groups.items():
            sub = valid[valid["sex"] == sex]
            ax.scatter(sub["age"], sub[trait], color=sc, alpha=0.75, s=40,
                       edgecolors="none", label=sex)
        # Correlation
        age_vals = pd.to_numeric(valid["age"], errors="coerce")
        trait_vals = pd.to_numeric(valid[trait], errors="coerce")
        mask = age_vals.notna() & trait_vals.notna()
        if mask.sum() > 2:
            r = float(np.corrcoef(age_vals[mask], trait_vals[mask])[0, 1])
            ax.set_title(f"{label}\nr(age)={r:+.2f}", fontsize=9)
        else:
            ax.set_title(label, fontsize=9)
        ax.set_xlabel("Age" if col_idx == 0 else "")
        ax.set_ylabel(trait.split("_")[1].upper() if col_idx == 0 else "")
        ax.set_ylim(1, 5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if col_idx == 0:
            ax.legend(frameon=False, fontsize=8)

        # Row 1: box plot by education (simplified to three tiers)
        ax2 = axes[1, col_idx]
        edu_map = {
            "bachelor": "Undergrad", "ap": "Undergrad",
            "master": "Master", "phd": "PhD",
            "professional_certificate": "Other",
            "graduate_certificate": "Other",
        }
        valid["edu_tier"] = valid["education"].map(edu_map).fillna("Other")
        tier_order = ["Undergrad", "Master", "PhD", "Other"]
        data = [valid.loc[valid["edu_tier"] == tier, trait].dropna().to_numpy()
                for tier in tier_order]
        bp = ax2.boxplot(data, patch_artist=True,
                         medianprops=dict(color="white", linewidth=1.5), widths=0.55)
        tier_colors = ["#5b9bd5", "#2f7d5c", "#7c4d9e", "#888888"]
        for patch, tc in zip(bp["boxes"], tier_colors):
            patch.set_facecolor(tc)
            patch.set_alpha(0.8)
        ax2.set_xticks(range(1, 5), tier_order, fontsize=8, rotation=20, ha="right")
        ax2.set_ylim(1, 5)
        ax2.set_xlabel("Education" if col_idx == 0 else "")
        ax2.axhline(3, color="#cccccc", linewidth=0.6)
        ax2.spines["top"].set_visible(False)
        ax2.spines["right"].set_visible(False)

    out_path = out_dir / "personality_demographics.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BFI-44 personality analysis.")
    p.add_argument("--bids-root", type=Path, default=Path("F:/bids_release_no_video"),
                   help="Path to BIDS dataset root containing participants.tsv.")
    p.add_argument("--results-dir", type=Path, default=Path("results/personality"),
                   help="Output directory for TSV result tables.")
    p.add_argument("--figures-dir", type=Path, default=Path("figures/personality"),
                   help="Output directory for PNG figures.")
    p.add_argument("--dpi", type=int, default=180, help="Figure DPI.")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)

    results_dir = args.results_dir.resolve()
    figures_dir = args.figures_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = load_participants(args.bids_root)
    gs = build_group_stats(df)

    # Write tables
    summary_path = results_dir / "personality_summary.tsv"
    group_path   = results_dir / "group_trait_stats.tsv"
    df.to_csv(summary_path, sep="\t", index=False)
    gs.to_csv(group_path, sep="\t", index=False)
    LOG.info("Wrote %s (%d rows)", summary_path, len(df))
    LOG.info("Wrote %s (%d rows)", group_path, len(gs))

    # Print console summary
    valid = df[df[TRAITS].notna().all(axis=1)]
    LOG.info("Participants with complete BFI-44: %d / %d", len(valid), len(df))
    for t in TRAITS:
        v = valid[t]
        LOG.info("  %-20s mean=%.2f  SD=%.2f  range=[%.2f, %.2f]",
                 TRAIT_LABELS[t], v.mean(), v.std(ddof=1), v.min(), v.max())

    # Write figures
    paths = [
        _distributions(valid, figures_dir, args.dpi),
        _correlations(valid, figures_dir, args.dpi),
        _group_heatmap(gs, figures_dir, args.dpi),
        _group_profiles(valid, gs, figures_dir, args.dpi),
        _demographics(valid, figures_dir, args.dpi),
    ]
    for path in paths:
        LOG.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
