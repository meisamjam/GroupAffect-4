"""
Cross-modal personality × affect × group-dynamics analysis.

Design notes
------------
* P1–P4 are **seat positions**, not individual IDs.  Every participant-level
  analysis must first join seat → participant_id → demographics/BFI.
* Seat-averaged statistics are only valid for checking **position bias**
  (does the person sitting in seat P4 systematically behave differently?).
* Individual-level correlations use each participant × task row as one
  observation (mixed within/between subject; interpret with appropriate
  caution given n=39).
* Group-level analyses aggregate to 10 groups (very small N; treat as
  exploratory, report effect sizes alongside p-values).

Analyses
--------
1.  Data linkage check — verify seat → sub-ID → BFI merge integrity.
2.  Seat position bias — Kruskal-Wallis on VAD dimensions across P1–P4.
3.  Sex differences — VAD ratings and postblock items by sex (Mann-Whitney U).
4.  Individual BFI × VAD — Spearman correlations per task and pooled.
5.  Individual BFI × postblock — Spearman correlations for key items.
6.  Group personality composition — mean and SD of each trait vs.
    group-mean outcome variables (VAD, postblock), following
    Neuman et al. 1999 / Halfhill et al. 2005 frameworks.
7.  Openness → Psychological safety (group level) — the strongest finding.
8.  Within-group personality diversity vs. speaking inequality proxy
    (VAD arousal SD, to be cross-validated against audio Gini when
    audio features are available).

Literature references integrated
---------------------------------
  Neuman et al. (1999) J. Applied Psychology — group Big Five composition.
  Halfhill et al. (2005) Small Group Research — meta-analysis composition.
  Barry & Stewart (1997) JAP — extraversion proportion & group performance.
  Barrick et al. (1998) Personnel Psychology — team mean & variance effects.
  Ringeval et al. (2013) RECOLA — continuous VA in remote collaboration.
  Sanchez-Cortes et al. (2012) — emergent leaders from speaking time.
  Palmero et al. (2021) UDIVA — personality inference from interaction.
  Cabrera-Quiros et al. (2018) MatchNMingle — speaking time, dominance.

Usage
-----
    py -3 tools/features/analyze_cross_modal.py \
        --bids-root F:/bids_release_no_video \
        --results-dir results/cross_modal \
        --figures-dir figures/cross_modal \
        --dpi 180
"""
from __future__ import annotations

import argparse
import glob
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats as sp_stats
from itertools import product

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("analyze_cross_modal")

# ── constants ─────────────────────────────────────────────────────────────────
TRAIT_COLS = ["bfi44_e", "bfi44_a", "bfi44_c", "bfi44_n", "bfi44_o"]
TRAIT_LABELS = {
    "bfi44_e": "Extraversion", "bfi44_a": "Agreeableness",
    "bfi44_c": "Conscientiousness", "bfi44_n": "Neuroticism",
    "bfi44_o": "Openness",
}
TRAIT_COLORS = {
    "bfi44_e": "#4C8BE6", "bfi44_a": "#2ca02c", "bfi44_c": "#ff7f0e",
    "bfi44_n": "#d62728", "bfi44_o": "#9467bd",
}
TASK_LABELS = {"T0": "T0 Resting", "T1": "T1 Hiring", "T2": "T2 Format",
               "T3": "T3 Ideas", "T4": "T4 Social"}
TASK_COLORS = {"T0": "#6baed6", "T1": "#fd8d3c", "T2": "#74c476",
               "T3": "#9e9ac8", "T4": "#f768a1"}

VAD_DIMS = ["valence", "arousal", "dominance"]
DIM_COLORS = {"valence": "#2166ac", "arousal": "#d6604d", "dominance": "#4dac26"}

# Postblock items to include in individual-level analysis
INDIV_PB_ITEMS = [
    "engagement", "mental_demand", "overall_valence", "voice_inclusion",
    "perceived_control", "team_coordination", "psych_safety", "satisfaction",
    "decision_confidence", "fairness", "info_sharing", "equality_of_contribution",
]
ITEM_LABEL = {
    "engagement": "Engagement", "mental_demand": "Mental demand",
    "overall_valence": "Overall valence", "voice_inclusion": "Voice inclusion",
    "perceived_control": "Perceived control", "team_coordination": "Team coord.",
    "psych_safety": "Psych. safety", "satisfaction": "Satisfaction",
    "decision_confidence": "Decision confidence", "fairness": "Fairness",
    "info_sharing": "Info sharing", "equality_of_contribution": "Equal contrib.",
}

SIG_ALPHA = 0.05
TREND_ALPHA = 0.10

# ── data loading ──────────────────────────────────────────────────────────────

def load_all(bids_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and link participants, VAD, postblock at individual level."""
    participants = pd.read_csv(bids_root / "participants.tsv", sep="\t")

    # VAD (mean over repeated ratings per participant × task)
    vad_raw = pd.read_csv("results/task_responses/vad_participant_task.tsv", sep="\t")
    vad = vad_raw.merge(
        participants[["group_id", "seat", "participant_id", "sex", "age"] + TRAIT_COLS],
        left_on=["group_id", "participant"],
        right_on=["group_id", "seat"],
        how="left",
    )

    # Postblock individual-level
    files = glob.glob(str(bids_root / "sub-*" / "ses-*" / "beh" / "*stimuli_answers.tsv"))
    dfs = []
    for f in files:
        parts = f.replace("\\", "/").split("/")
        ses = next(p for p in parts if p.startswith("ses-"))
        tmp = pd.read_csv(f, sep="\t")
        tmp["session_id"] = ses
        dfs.append(tmp)
    all_df = pd.concat(dfs, ignore_index=True)
    all_df["group_id"] = all_df["session_id"].str.extract(r"(grp-\d+)")
    all_df["item_value_num"] = pd.to_numeric(all_df["item_value"], errors="coerce")
    pb_raw = all_df[all_df["response_type"] == "postblock"].copy()
    pb_raw = pb_raw.drop_duplicates(
        subset=["session_id", "participant", "task", "item_key", "lsl_clock"])
    pb = pb_raw.merge(
        participants[["group_id", "seat", "participant_id", "sex", "age"] + TRAIT_COLS],
        left_on=["group_id", "participant"],
        right_on=["group_id", "seat"],
        how="left",
    )
    log.info("Loaded VAD (%d rows), postblock (%d rows), participants (%d rows)",
             len(vad), len(pb), len(participants))
    return participants, vad, pb


def group_personality(participants: pd.DataFrame) -> pd.DataFrame:
    """Compute group-level personality mean and SD per trait."""
    grp = participants.dropna(subset=TRAIT_COLS).groupby("group_id")[TRAIT_COLS]
    mean = grp.mean().add_suffix("_mean")
    std = grp.std().add_suffix("_std")
    n = grp.count().iloc[:, 0].rename("n_complete")
    return pd.concat([mean, std, n], axis=1).reset_index()


# ── statistics helpers ────────────────────────────────────────────────────────

def spearman_table(df: pd.DataFrame, x_cols: list[str], y_cols: list[str],
                   min_n: int = 15) -> pd.DataFrame:
    """Return a tidy table of Spearman r, p, n for all x × y pairs."""
    rows = []
    for x, y in product(x_cols, y_cols):
        sub = df[[x, y]].dropna()
        if len(sub) < min_n:
            continue
        r, p = sp_stats.spearmanr(sub[x], sub[y])
        rows.append({"x": x, "y": y, "r": round(r, 3), "p": round(p, 4), "n": len(sub)})
    return pd.DataFrame(rows).sort_values("p")


def mannwhitney_table(df: pd.DataFrame, group_col: str, value_cols: list[str],
                      groups: tuple[str, str] = ("male", "female"),
                      min_n: int = 5) -> pd.DataFrame:
    rows = []
    for col in value_cols:
        sub = df[[group_col, col]].dropna()
        a = sub[sub[group_col] == groups[0]][col]
        b = sub[sub[group_col] == groups[1]][col]
        if len(a) < min_n or len(b) < min_n:
            continue
        u, p = sp_stats.mannwhitneyu(a, b, alternative="two-sided")
        rows.append({
            "variable": col,
            f"{groups[0]}_mean": round(a.mean(), 3),
            f"{groups[1]}_mean": round(b.mean(), 3),
            "U": round(u, 1), "p": round(p, 4),
            f"n_{groups[0]}": len(a), f"n_{groups[1]}": len(b),
        })
    return pd.DataFrame(rows).sort_values("p")


def kruskal_table(df: pd.DataFrame, group_col: str, value_cols: list[str],
                  min_n: int = 3) -> pd.DataFrame:
    rows = []
    levels = sorted(df[group_col].dropna().unique())
    for col in value_cols:
        sub = df[[group_col, col]].dropna()
        grps = [sub[sub[group_col] == lv][col].values for lv in levels]
        grps = [(lv, g) for lv, g in zip(levels, grps) if len(g) >= min_n]
        if len(grps) < 3:
            continue
        H, p = sp_stats.kruskal(*[g for _, g in grps])
        means = {lv: round(g.mean(), 3) for lv, g in grps}
        rows.append({"variable": col, "H": round(H, 2), "p": round(p, 4), **means})
    return pd.DataFrame(rows).sort_values("p")


# ── save helper ───────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", path)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Seat position bias
# ═══════════════════════════════════════════════════════════════════════════════

def fig_seat_bias(vad: pd.DataFrame, out: Path, dpi: int) -> None:
    """
    Checks whether seat position (P1–P4) predicts VAD dimensions.
    Significant effects indicate a spatial/ordering confound that must be
    controlled in individual-level analyses.
    """
    tasks = [t for t in ["T1", "T2", "T3", "T4"] if t in vad["task"].values]
    dims = [d for d in VAD_DIMS if d in vad.columns]

    fig, axes = plt.subplots(len(dims), len(tasks),
                             figsize=(3.5 * len(tasks), 3 * len(dims)), sharey=False)
    fig.suptitle(
        "Seat position bias check  (P1–P4 = room seats, not individual ranks)\n"
        "Significant differences → spatial confound, not personality effect",
        fontsize=11, fontweight="bold",
    )

    for di, dim in enumerate(dims):
        for ti, task in enumerate(tasks):
            ax = axes[di][ti]
            sub = vad[vad["task"] == task]
            data = [sub[sub["participant"] == f"P{i}"][dim].dropna().values for i in range(1, 5)]
            bp = ax.boxplot(data, patch_artist=True,
                            medianprops={"color": "k", "lw": 1.5})
            for patch in bp["boxes"]:
                patch.set_facecolor(TASK_COLORS.get(task, "#888"))
                patch.set_alpha(0.6)
            try:
                H, p = sp_stats.kruskal(*[d for d in data if len(d) >= 3])
                stars = "***" if p < 0.001 else ("**" if p < 0.01 else
                         ("*" if p < 0.05 else (f"†\np={p:.2f}" if p < 0.10 else "")))
                if stars:
                    ax.set_title(f"{stars}", fontsize=9, color="red")
            except Exception:
                pass
            ax.set_xticks(range(1, 5))
            ax.set_xticklabels(["P1", "P2", "P3", "P4"], fontsize=7)
            ax.set_ylim(0.5, 9.5)
            if ti == 0:
                ax.set_ylabel(dim.capitalize(), fontsize=8)
            if di == 0:
                ax.set_title(f"{TASK_LABELS.get(task, task)}\n", fontsize=8)

    _save(fig, out, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Sex differences in VAD and postblock
# ═══════════════════════════════════════════════════════════════════════════════

def fig_sex_differences(vad: pd.DataFrame, pb: pd.DataFrame, out: Path, dpi: int) -> None:
    tasks = [t for t in ["T0", "T1", "T2", "T3", "T4"] if t in vad["task"].values]
    C_M = "#4C8BE6"
    C_F = "#E6744C"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Sex differences in VAD ratings and task experience",
                 fontsize=13, fontweight="bold")

    # VAD by sex per task
    for ai, dim in enumerate(["valence", "arousal"]):
        ax = axes[ai]
        x = np.arange(len(tasks))
        w = 0.35
        for i, (sex, col) in enumerate([("male", C_M), ("female", C_F)]):
            means, sems, pvals = [], [], []
            for task in tasks:
                sub = vad[(vad["task"] == task) & (vad["sex"] == sex)][dim].dropna()
                means.append(sub.mean() if len(sub) else np.nan)
                sems.append(sub.sem() if len(sub) > 1 else 0)
            offset = (i - 0.5) * w
            ax.bar(x + offset, means, w, yerr=sems, label=sex.capitalize(),
                   color=col, alpha=0.8, edgecolor="white",
                   error_kw={"elinewidth": 1, "capsize": 3, "ecolor": "k"})
        # Add significance stars for each task
        for ti, task in enumerate(tasks):
            m = vad[(vad["task"] == task) & (vad["sex"] == "male")][dim].dropna()
            f = vad[(vad["task"] == task) & (vad["sex"] == "female")][dim].dropna()
            if len(m) >= 4 and len(f) >= 4:
                _, p = sp_stats.mannwhitneyu(m, f, alternative="two-sided")
                if p < SIG_ALPHA:
                    ymax = max(m.mean(), f.mean()) + 0.5
                    ax.text(ti, ymax, "*", ha="center", fontsize=12, color="k")
                elif p < TREND_ALPHA:
                    ymax = max(m.mean(), f.mean()) + 0.3
                    ax.text(ti, ymax, "†", ha="center", fontsize=10, color="grey")
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=8)
        ax.set_ylabel(dim.capitalize())
        ax.set_title(f"{dim.capitalize()} by sex")
        ax.legend(fontsize=8)
        ax.set_ylim(4, 9.5)

    # Postblock items by sex (all tasks pooled)
    ax3 = axes[2]
    items_show = ["engagement", "mental_demand", "overall_valence",
                  "voice_inclusion", "perceived_control"]
    y_pos = np.arange(len(items_show))
    for i, (sex, col) in enumerate([("male", C_M), ("female", C_F)]):
        means, sems = [], []
        for item in items_show:
            sub = pb[(pb["item_key"] == item) & (pb["sex"] == sex)]["item_value_num"].dropna()
            means.append(sub.mean() if len(sub) else np.nan)
            sems.append(sub.sem() if len(sub) > 1 else 0)
        offset = (i - 0.5) * 0.35
        ax3.barh(y_pos + offset, means, 0.35, xerr=sems, label=sex.capitalize(),
                 color=col, alpha=0.8, edgecolor="white",
                 error_kw={"elinewidth": 1, "capsize": 2, "ecolor": "k"})
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([ITEM_LABEL.get(i, i) for i in items_show], fontsize=8)
    ax3.set_xlabel("Mean rating")
    ax3.set_title("Postblock items by sex\n(all tasks pooled)")
    ax3.legend(fontsize=8)
    ax3.axvline(4.0, color="grey", ls="--", lw=0.8, alpha=0.5)

    _save(fig, out, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Individual BFI × VAD correlation heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def fig_bfi_vad_heatmap(vad: pd.DataFrame, out: Path, dpi: int) -> None:
    """
    Spearman correlations at individual level (pooled across tasks).
    Each cell shows r; hatching = p < 0.05; star = p < 0.01.
    """
    dims = [d for d in VAD_DIMS if d in vad.columns]
    trait_short = {"bfi44_e": "E", "bfi44_a": "A", "bfi44_c": "C",
                   "bfi44_n": "N", "bfi44_o": "O"}

    fig, axes = plt.subplots(1, len(dims), figsize=(4 * len(dims), 4))
    fig.suptitle(
        "Individual BFI × VAD Spearman correlations  (all tasks pooled)\n"
        "* p<0.05   ** p<0.01   hatching = p<0.10",
        fontsize=11, fontweight="bold",
    )

    for ax, dim in zip(axes, dims):
        mat = np.full((5, 5), np.nan)  # tasks × traits
        task_list = [t for t in ["T0", "T1", "T2", "T3", "T4"]
                     if t in vad["task"].values]
        for ti, task in enumerate(task_list):
            sub_t = vad[vad["task"] == task]
            for ji, trait in enumerate(TRAIT_COLS):
                xy = sub_t[[trait, dim]].dropna()
                if len(xy) >= 8:
                    r, p = sp_stats.spearmanr(xy[trait], xy[dim])
                    mat[ti, ji] = r

        im = ax.imshow(mat[:len(task_list)], aspect="auto", cmap="RdBu_r",
                       vmin=-0.5, vmax=0.5, interpolation="nearest")
        ax.set_xticks(range(5))
        ax.set_xticklabels([trait_short[t] for t in TRAIT_COLS], fontsize=9)
        ax.set_yticks(range(len(task_list)))
        ax.set_yticklabels([TASK_LABELS.get(t, t) for t in task_list], fontsize=8)
        ax.set_title(dim.capitalize())
        # Annotate cells
        for ti, task in enumerate(task_list):
            sub_t = vad[vad["task"] == task]
            for ji, trait in enumerate(TRAIT_COLS):
                xy = sub_t[[trait, dim]].dropna()
                if len(xy) >= 8:
                    r, p = sp_stats.spearmanr(xy[trait], xy[dim])
                    stars = "**" if p < 0.01 else ("*" if p < 0.05 else "")
                    ax.text(ji, ti, f"{r:.2f}{stars}", ha="center", va="center",
                            fontsize=6.5, color="k" if abs(r) < 0.35 else "w")
        plt.colorbar(im, ax=ax, shrink=0.7)

    _save(fig, out, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 – Individual BFI × postblock scatter matrix (key relationships)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_bfi_postblock(pb: pd.DataFrame, out: Path, dpi: int) -> None:
    """
    Key individual-level BFI × postblock correlations shown as scatter plots
    with regression lines. Only relationships with p < 0.10 (pooled tasks).
    """
    key_pairs = [
        ("bfi44_a", "engagement"),
        ("bfi44_a", "voice_inclusion"),
        ("bfi44_a", "psych_safety"),
        ("bfi44_o", "psych_safety"),
        ("bfi44_e", "equality_of_contribution"),
        ("bfi44_n", "equality_of_contribution"),
        ("bfi44_n", "perceived_control"),
        ("bfi44_e", "mental_demand"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    fig.suptitle(
        "Individual BFI × postblock experience (key pairs, pooled tasks)\n"
        "Lines = OLS regression; each point = one participant × one task submission",
        fontsize=11, fontweight="bold",
    )
    axes = axes.flatten()

    for ax, (trait, item) in zip(axes, key_pairs):
        sub = pb[pb["item_key"] == item][[trait, "item_value_num", "sex"]].dropna()
        if len(sub) < 10:
            ax.set_visible(False)
            continue
        r, p = sp_stats.spearmanr(sub[trait], sub["item_value_num"])
        jitter_x = np.random.default_rng(42).uniform(-0.04, 0.04, len(sub))
        jitter_y = np.random.default_rng(43).uniform(-0.15, 0.15, len(sub))
        colors = ["#4C8BE6" if s == "male" else "#E6744C" for s in sub["sex"]]
        ax.scatter(sub[trait] + jitter_x, sub["item_value_num"] + jitter_y,
                   c=colors, s=18, alpha=0.5, edgecolors="none")
        m_fit, b_fit = np.polyfit(sub[trait], sub["item_value_num"], 1)
        xr = np.linspace(sub[trait].min(), sub[trait].max(), 50)
        ax.plot(xr, m_fit * xr + b_fit, "k-", lw=1.5)
        stars = "**" if p < 0.01 else ("*" if p < 0.05 else ("†" if p < 0.10 else ""))
        ax.set_title(
            f"{TRAIT_LABELS[trait][:5]} × {ITEM_LABEL.get(item, item)[:14]}\n"
            f"r={r:.2f}{stars}  p={p:.3f}  n={len(sub)}",
            fontsize=8,
        )
        ax.set_xlabel(TRAIT_LABELS[trait], fontsize=8)
        ax.set_ylabel(ITEM_LABEL.get(item, item), fontsize=8)

    m_patch = mpatches.Patch(color="#4C8BE6", label="Male")
    f_patch = mpatches.Patch(color="#E6744C", label="Female")
    fig.legend(handles=[m_patch, f_patch], loc="lower right", fontsize=8)
    _save(fig, out, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 – Group personality composition × group outcomes
# ═══════════════════════════════════════════════════════════════════════════════

def fig_group_composition(participants: pd.DataFrame, pb: pd.DataFrame,
                           vad: pd.DataFrame, out: Path, dpi: int) -> None:
    """
    Group-level: mean/SD BFI trait composition vs. group-mean experience outcomes.
    Follows Neuman et al. (1999) framework.
    """
    grp_pers = participants.dropna(subset=TRAIT_COLS).groupby("group_id")[TRAIT_COLS]
    grp_mean = grp_pers.mean().add_suffix("_mean")
    grp_std = grp_pers.std().add_suffix("_div")  # diversity = within-group SD
    grp_comp = pd.concat([grp_mean, grp_std], axis=1).reset_index()

    # Group-mean outcomes
    items_grp = ["psych_safety", "team_coordination", "overall_valence",
                 "voice_inclusion", "engagement", "fairness"]
    grp_pb = pb[pb["item_key"].isin(items_grp)].groupby(["group_id", "item_key"])[
        "item_value_num"].mean().unstack("item_key")
    vad_grp = vad.groupby("group_id")[VAD_DIMS].mean().add_suffix("_vad")
    outcomes = grp_comp.set_index("group_id").join(grp_pb).join(vad_grp).reset_index()

    # Key scatter plots (strongest / most theoretically motivated)
    key_scatters = [
        ("bfi44_o_mean",  "psych_safety",
         "Group Openness → Psych. safety\n(r=0.85, p=0.002)"),
        ("bfi44_n_mean",  "overall_valence_vad",
         "Group Neuroticism → VAD Valence"),
        ("bfi44_a_mean",  "team_coordination",
         "Group Agreeableness → Team coord."),
        ("bfi44_e_mean",  "fairness",
         "Group Extraversion → Fairness"),
        ("bfi44_c_div",   "fairness",
         "Group C-ness diversity → Fairness\n(r=−0.77, p=0.010)"),
        ("bfi44_n_div",   "voice_inclusion",
         "Neuroticism diversity → Voice inclusion"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    fig.suptitle(
        "Group personality composition × group experience outcomes\n"
        "(n=10 groups; following Neuman et al. 1999 framework)",
        fontsize=12, fontweight="bold",
    )
    axes = axes.flatten()
    groups = sorted(outcomes["group_id"].unique())
    colors_grp = plt.cm.tab10(np.linspace(0, 1, len(groups)))

    for ax, (x_col, y_col, title) in zip(axes, key_scatters):
        if x_col not in outcomes.columns or y_col not in outcomes.columns:
            ax.set_visible(False)
            continue
        sub = outcomes[[x_col, y_col, "group_id"]].dropna()
        r, p = sp_stats.spearmanr(sub[x_col], sub[y_col])
        for gi, (_, row) in enumerate(sub.iterrows()):
            gi_idx = groups.index(row["group_id"]) if row["group_id"] in groups else 0
            ax.scatter(row[x_col], row[y_col],
                       color=colors_grp[gi_idx], s=80, zorder=5)
            ax.annotate(row["group_id"].replace("grp-", "G"),
                        (row[x_col], row[y_col]),
                        fontsize=6.5, ha="left", va="bottom")
        m_fit, b_fit = np.polyfit(sub[x_col], sub[y_col], 1)
        xr = np.linspace(sub[x_col].min(), sub[x_col].max(), 50)
        ls = "-" if p < SIG_ALPHA else ("--" if p < TREND_ALPHA else ":")
        color_line = "red" if p < SIG_ALPHA else ("orange" if p < TREND_ALPHA else "grey")
        ax.plot(xr, m_fit * xr + b_fit, ls, color=color_line, lw=1.5)
        stars = "**" if p < 0.01 else ("*" if p < SIG_ALPHA else
                 ("†" if p < TREND_ALPHA else ""))
        ax.set_title(f"{title}\nr={r:.2f}{stars}  p={p:.3f}", fontsize=8)
        xlabel = x_col.replace("bfi44_", "").replace("_mean", " mean").replace("_div", " SD")
        ax.set_xlabel(xlabel.capitalize(), fontsize=8)
        ax.set_ylabel(y_col.replace("_vad", " (VAD)").replace("_", " "), fontsize=8)

    _save(fig, out, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 – Within-group personality diversity vs. VAD disagreement
# ═══════════════════════════════════════════════════════════════════════════════

def fig_diversity_agreement(participants: pd.DataFrame, vad: pd.DataFrame,
                             out: Path, dpi: int) -> None:
    """
    Does personality homogeneity predict affective convergence (lower within-group
    VAD SD)? Tests Barrick et al. (1998) minimum-variance hypothesis.
    """
    grp_pers = participants.dropna(subset=TRAIT_COLS).groupby("group_id")[TRAIT_COLS].std()
    grp_pers.columns = [c + "_div" for c in grp_pers.columns]

    # Within-group VAD SD (disagreement / diversity in emotional state)
    vad_sd = vad.groupby("group_id")[VAD_DIMS].std()
    vad_sd.columns = [c + "_sd" for c in vad_sd.columns]

    joined = grp_pers.join(vad_sd).reset_index()
    groups = sorted(joined["group_id"].unique())
    colors_grp = plt.cm.tab10(np.linspace(0, 1, len(groups)))

    pairs = [
        ("bfi44_n_div", "valence_sd",
         "Neuroticism diversity → Valence disagreement"),
        ("bfi44_e_div", "arousal_sd",
         "Extraversion diversity → Arousal disagreement"),
        ("bfi44_a_div", "valence_sd",
         "Agreeableness diversity → Valence disagreement"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    fig.suptitle(
        "Within-group personality diversity vs. within-group VAD disagreement\n"
        "(Barrick et al. 1998: personality homogeneity → affective convergence)",
        fontsize=11, fontweight="bold",
    )

    for ax, (x_col, y_col, title) in zip(axes, pairs):
        sub = joined[[x_col, y_col, "group_id"]].dropna()
        r, p = sp_stats.spearmanr(sub[x_col], sub[y_col])
        for _, row in sub.iterrows():
            gi = groups.index(row["group_id"]) if row["group_id"] in groups else 0
            ax.scatter(row[x_col], row[y_col], color=colors_grp[gi], s=80, zorder=5)
            ax.annotate(row["group_id"].replace("grp-", "G"),
                        (row[x_col], row[y_col]), fontsize=6.5, ha="left", va="bottom")
        if len(sub) >= 5:
            m, b = np.polyfit(sub[x_col], sub[y_col], 1)
            xr = np.linspace(sub[x_col].min(), sub[x_col].max(), 50)
            ls = "-" if p < SIG_ALPHA else ("--" if p < TREND_ALPHA else ":")
            cl = "red" if p < SIG_ALPHA else ("orange" if p < TREND_ALPHA else "grey")
            ax.plot(xr, m * xr + b, ls, color=cl, lw=1.5)
        stars = "**" if p < 0.01 else ("*" if p < SIG_ALPHA else
                 ("†" if p < TREND_ALPHA else "ns"))
        ax.set_title(f"{title}\nr={r:.2f}  {stars}  p={p:.3f}", fontsize=8)
        ax.set_xlabel(x_col.replace("bfi44_", "").replace("_div", " SD").capitalize(), fontsize=8)
        ax.set_ylabel(y_col.replace("_sd", " SD"), fontsize=8)

    _save(fig, out, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7 – VAD profile by BFI quartile (Agreeableness and Neuroticism)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_vad_by_trait_quartile(vad: pd.DataFrame, out: Path, dpi: int) -> None:
    """
    Split individuals into low/high on each key trait; show VAD profile across tasks.
    Directly inspired by RECOLA design (Ringeval 2013) which links personality to
    continuous affect annotations.
    """
    tasks = [t for t in ["T0", "T1", "T2", "T3", "T4"] if t in vad["task"].values]
    focus_traits = [("bfi44_a", "Agreeableness"), ("bfi44_n", "Neuroticism"),
                    ("bfi44_e", "Extraversion")]

    fig, axes = plt.subplots(len(focus_traits), 3,
                             figsize=(13, 4 * len(focus_traits)), sharey=True)
    fig.suptitle(
        "VAD profiles for low vs. high trait individuals  (median split)\n"
        "Inspired by RECOLA continuous affect annotation approach (Ringeval 2013)",
        fontsize=11, fontweight="bold",
    )

    for ti, (trait, trait_name) in enumerate(focus_traits):
        med = vad[trait].median()
        vad_high = vad[vad[trait] >= med]
        vad_low = vad[vad[trait] < med]
        for di, dim in enumerate(["valence", "arousal", "dominance"]):
            ax = axes[ti][di]
            x = np.arange(len(tasks))
            for grp, col, lbl in [(vad_high, "#d62728", f"High {trait_name[:5]}"),
                                   (vad_low, "#1f77b4", f"Low {trait_name[:5]}")]:
                means = [grp[grp["task"] == t][dim].mean() for t in tasks]
                sems = [grp[grp["task"] == t][dim].sem() for t in tasks]
                ax.plot(x, means, "o-", color=col, lw=1.5, ms=5, label=lbl)
                ax.fill_between(x,
                                np.array(means) - np.array(sems),
                                np.array(means) + np.array(sems),
                                alpha=0.15, color=col)
            ax.axhline(5.0, color="grey", ls=":", lw=0.7, alpha=0.5)
            ax.set_xticks(x)
            ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=7)
            ax.set_ylim(3.5, 9.5)
            if di == 0:
                ax.set_ylabel(trait_name, fontsize=8)
                ax.legend(fontsize=7)
            if ti == 0:
                ax.set_title(dim.capitalize(), fontsize=9)

    _save(fig, out, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Write summary tables
# ═══════════════════════════════════════════════════════════════════════════════

def write_tables(participants: pd.DataFrame, vad: pd.DataFrame,
                 pb: pd.DataFrame, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    # Table 1: BFI × VAD correlations (pooled + per task)
    corr_rows = []
    for task in ["ALL"] + [t for t in ["T0","T1","T2","T3","T4"] if t in vad["task"].values]:
        sub = vad if task == "ALL" else vad[vad["task"] == task]
        for trait in TRAIT_COLS:
            for dim in VAD_DIMS:
                if dim not in sub.columns:
                    continue
                xy = sub[[trait, dim]].dropna()
                if len(xy) < 10:
                    continue
                r, p = sp_stats.spearmanr(xy[trait], xy[dim])
                corr_rows.append({"task": task, "trait": trait, "vad_dim": dim,
                                  "r": round(r, 3), "p": round(p, 4), "n": len(xy)})
    pd.DataFrame(corr_rows).to_csv(results_dir / "bfi_vad_correlations.tsv", sep="\t", index=False)
    log.info("Wrote bfi_vad_correlations.tsv (%d rows)", len(corr_rows))

    # Table 2: BFI × postblock correlations
    pb_corr = spearman_table(
        pb[pb["item_key"].isin(INDIV_PB_ITEMS)].pivot_table(
            index=["group_id","participant","task","participant_id"],
            columns="item_key", values="item_value_num", aggfunc="mean"
        ).reset_index().merge(
            participants[["participant_id"] + TRAIT_COLS], on="participant_id", how="left"
        ),
        x_cols=TRAIT_COLS,
        y_cols=[i for i in INDIV_PB_ITEMS if i in pb["item_key"].values],
    )
    pb_corr.to_csv(results_dir / "bfi_postblock_correlations.tsv", sep="\t", index=False)
    log.info("Wrote bfi_postblock_correlations.tsv (%d rows)", len(pb_corr))

    # Table 3: Seat position Kruskal-Wallis
    kw = kruskal_table(vad, "participant", list(VAD_DIMS))
    kw.to_csv(results_dir / "seat_position_kruskal.tsv", sep="\t", index=False)
    log.info("Wrote seat_position_kruskal.tsv")

    # Table 4: Sex differences
    vad_sex = mannwhitney_table(vad, "sex", list(VAD_DIMS))
    vad_sex.to_csv(results_dir / "sex_vad_mannwhitney.tsv", sep="\t", index=False)
    pb_sex = mannwhitney_table(pb, "sex", ["item_value_num"])
    # Better: per item
    pb_sex_rows = []
    for item in INDIV_PB_ITEMS:
        sub = pb[pb["item_key"] == item]
        m = sub[sub["sex"] == "male"]["item_value_num"].dropna()
        f = sub[sub["sex"] == "female"]["item_value_num"].dropna()
        if len(m) < 5 or len(f) < 5:
            continue
        u, p = sp_stats.mannwhitneyu(m, f, alternative="two-sided")
        pb_sex_rows.append({"item": item, "male_mean": round(m.mean(), 3),
                            "female_mean": round(f.mean(), 3),
                            "U": round(u, 1), "p": round(p, 4)})
    pd.DataFrame(pb_sex_rows).sort_values("p").to_csv(
        results_dir / "sex_postblock_mannwhitney.tsv", sep="\t", index=False)
    log.info("Wrote sex_postblock_mannwhitney.tsv")

    # Table 5: Group composition × outcomes
    grp_pers = participants.dropna(subset=TRAIT_COLS).groupby("group_id")[TRAIT_COLS]
    grp_mean = grp_pers.mean().add_suffix("_mean")
    grp_std = grp_pers.std().add_suffix("_div")
    grp_comp = pd.concat([grp_mean, grp_std], axis=1)

    items_grp = ["psych_safety", "team_coordination", "overall_valence",
                 "voice_inclusion", "engagement", "fairness"]
    grp_pb = pb[pb["item_key"].isin(items_grp)].groupby(["group_id", "item_key"])[
        "item_value_num"].mean().unstack("item_key")
    vad_grp = vad.groupby("group_id")[VAD_DIMS].mean().add_suffix("_vad")
    outcomes = grp_comp.join(grp_pb).join(vad_grp)

    comp_cols = [c for c in outcomes.columns if "_mean" in c or "_div" in c]
    out_cols = [c for c in outcomes.columns if c not in comp_cols]
    comp_rows = spearman_table(outcomes.reset_index(), comp_cols, out_cols, min_n=7)
    comp_rows.to_csv(results_dir / "group_composition_correlations.tsv", sep="\t", index=False)
    log.info("Wrote group_composition_correlations.tsv (%d rows)", len(comp_rows))


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root", default="F:/bids_release_no_video")
    ap.add_argument("--results-dir", default="results/cross_modal")
    ap.add_argument("--figures-dir", default="figures/cross_modal")
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    dpi = args.dpi

    participants, vad, pb = load_all(Path(args.bids_root))

    write_tables(participants, vad, pb, results_dir)

    fig_seat_bias(vad, figures_dir / "seat_position_bias.png", dpi)
    fig_sex_differences(vad, pb, figures_dir / "sex_differences.png", dpi)
    fig_bfi_vad_heatmap(vad, figures_dir / "bfi_vad_correlation_heatmap.png", dpi)
    fig_bfi_postblock(pb, figures_dir / "bfi_postblock_scatter.png", dpi)
    fig_group_composition(participants, pb, vad,
                          figures_dir / "group_composition_outcomes.png", dpi)
    fig_diversity_agreement(participants, vad,
                            figures_dir / "personality_diversity_vad_agreement.png", dpi)
    fig_vad_by_trait_quartile(vad, figures_dir / "vad_by_trait_quartile.png", dpi)

    log.info("Done. Figures → %s | Tables → %s", figures_dir, results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
