"""
Task responses and VAD annotation analysis.

Covers:
  1. VAD (Valence-Arousal-Dominance) continuous ratings — per task, per group, over time
  2. Postblock questionnaires — per-task experience ratings (engagement, mental demand, etc.)
  3. Group decisions (task outcomes) — candidate selection, format, ideas
  4. Within-group VAD agreement (std as disagreement proxy)
  5. VAD dynamics — temporal trajectory within each task

Usage
-----
    py -3 tools/features/analyze_task_responses.py \
        --bids-root F:/bids_release_no_video \
        --results-dir results/task_responses \
        --figures-dir figures/task_responses \
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
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from scipy import stats as sp_stats

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("analyze_task_responses")

# ── constants ─────────────────────────────────────────────────────────────────
VAD_SCALE_MID = 5.0
VAD_SCALE_MAX = 9.0

TASK_LABELS = {"T0": "T0 Resting", "T1": "T1 Hiring", "T2": "T2 Format",
               "T3": "T3 Ideas", "T4": "T4 Social"}
TASK_COLORS = {"T0": "#6baed6", "T1": "#fd8d3c", "T2": "#74c476",
               "T3": "#9e9ac8", "T4": "#f768a1"}
DIM_COLORS = {"valence": "#2166ac", "arousal": "#d6604d", "dominance": "#4dac26"}

# postblock items per task with human-readable labels
POSTBLOCK = {
    "T1": {
        "decision_confidence": "Decision confidence",
        "engagement": "Engagement",
        "equality_of_contribution": "Equal contribution",
        "info_sharing": "Info sharing",
        "mental_demand": "Mental demand",
        "overall_valence": "Overall valence",
        "perceived_control": "Perceived control",
        "team_coordination": "Team coordination",
        "voice_inclusion": "Voice inclusion",
    },
    "T2": {
        "cooperative": "Cooperativeness",
        "engagement": "Engagement",
        "mental_demand": "Mental demand",
        "overall_valence": "Overall valence",
        "perceived_control": "Perceived control",
        "trust_front": "Trust (frontwise)",
        "trust_angle": "Trust (angle)",
        "trust_next": "Trust (next)",
        "voice_inclusion": "Voice inclusion",
    },
    "T3": {
        "confidence": "Confidence",
        "engagement": "Engagement",
        "fairness": "Fairness",
        "idea_diversity": "Idea diversity",
        "idea_quality": "Idea quality",
        "mental_demand": "Mental demand",
        "overall_valence": "Overall valence",
        "psych_safety": "Psych safety",
        "satisfaction": "Satisfaction",
        "team_coordination": "Team coordination",
        "voice_inclusion": "Voice inclusion",
    },
    "T4": {
        "expectation_match": "Expectation match",
        "fairness": "Fairness",
        "overall_valence": "Overall valence",
        "regret": "Regret",
        "social_concern": "Social concern",
        "trust_angle": "Trust (angle)",
        "trust_front": "Trust (front)",
        "trust_next": "Trust (next)",
    },
}


# ── data loading ──────────────────────────────────────────────────────────────

def load_all_responses(bids_root: Path) -> pd.DataFrame:
    files = glob.glob(str(bids_root / "sub-*" / "ses-*" / "beh" / "*stimuli_answers.tsv"))
    dfs = []
    for f in files:
        parts = f.replace("\\", "/").split("/")
        sub = next(p for p in parts if p.startswith("sub-"))
        ses = next(p for p in parts if p.startswith("ses-"))
        tmp = pd.read_csv(f, sep="\t")
        tmp["subject_id"] = sub
        tmp["session_id"] = ses
        dfs.append(tmp)
    df = pd.concat(dfs, ignore_index=True)
    df["group_id"] = df["session_id"].str.extract(r"(grp-\d+)")
    df["item_value_num"] = pd.to_numeric(df["item_value"], errors="coerce")
    log.info("Loaded %d response rows from %d sessions", len(df), len(files))
    return df


def get_vad(df: pd.DataFrame) -> pd.DataFrame:
    """Return deduplicated VAD rows with numeric values."""
    vad = df[df["response_type"] == "vad"].copy()
    vad = vad[vad["item_key"].isin(["valence", "arousal", "dominance"])]
    vad = vad.drop_duplicates(
        subset=["session_id", "participant", "task", "item_key", "lsl_clock"]
    )
    vad = vad.sort_values("lsl_clock")
    return vad


def get_postblock(df: pd.DataFrame) -> pd.DataFrame:
    pb = df[df["response_type"] == "postblock"].copy()
    pb = pb.drop_duplicates(
        subset=["session_id", "participant", "task", "item_key", "lsl_clock"]
    )
    return pb


def vad_participant_task_summary(vad: pd.DataFrame) -> pd.DataFrame:
    """Mean VAD per participant × task (wide format)."""
    agg = (
        vad.groupby(["group_id", "session_id", "participant", "task", "item_key"])["item_value_num"]
        .agg(mean="mean", last="last", n_ratings="count")
        .reset_index()
    )
    wide = agg.pivot_table(
        index=["group_id", "session_id", "participant", "task"],
        columns="item_key",
        values="mean",
    ).reset_index()
    wide.columns.name = None
    return wide


# ── helpers ───────────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", path)


def _add_midline(ax: plt.Axes, orient: str = "h") -> None:
    if orient == "h":
        ax.axhline(VAD_SCALE_MID, color="grey", lw=0.8, ls="--", alpha=0.5)
    else:
        ax.axvline(VAD_SCALE_MID, color="grey", lw=0.8, ls="--", alpha=0.5)


def _violin_or_box(ax: plt.Axes, data_by_group: list[np.ndarray],
                   labels: list[str], color: str) -> None:
    clean = [d[~np.isnan(d)] for d in data_by_group]
    valid_idx = [i for i, d in enumerate(clean) if len(d) >= 2]
    if not valid_idx:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=8)
        return
    parts = ax.violinplot(
        [clean[i] for i in valid_idx],
        positions=valid_idx,
        showmedians=True,
        showextrema=False,
    )
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("k")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 – VAD overview by task
# ═══════════════════════════════════════════════════════════════════════════════

def fig_vad_by_task(vad_wide: pd.DataFrame, out: Path, dpi: int) -> None:
    tasks = [t for t in ["T0", "T1", "T2", "T3", "T4"] if t in vad_wide["task"].values]
    dims = [d for d in ["valence", "arousal", "dominance"] if d in vad_wide.columns]

    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 5), sharey=False)
    fig.suptitle("VAD ratings by task  (1–9 scale, dashed = midpoint)",
                 fontsize=13, fontweight="bold")

    for ax, dim in zip(axes, dims):
        data = [vad_wide.loc[vad_wide["task"] == t, dim].dropna().values for t in tasks]
        _violin_or_box(ax, data, [TASK_LABELS.get(t, t) for t in tasks],
                       DIM_COLORS[dim])
        _add_midline(ax)
        means = [np.nanmean(d) for d in data]
        ax.plot(range(len(tasks)), means, "D-", color="k", ms=4, lw=1.2, zorder=5)
        ax.set_ylabel(dim.capitalize() + " (mean ± dist.)")
        ax.set_title(dim.capitalize())
        ax.set_ylim(0.5, 9.5)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1))

    _save(fig, out, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 – VAD heatmap: group × task
# ═══════════════════════════════════════════════════════════════════════════════

def fig_vad_group_task_heatmap(vad_wide: pd.DataFrame, out: Path, dpi: int) -> None:
    dims = [d for d in ["valence", "arousal", "dominance"] if d in vad_wide.columns]
    tasks = [t for t in ["T0", "T1", "T2", "T3", "T4"] if t in vad_wide["task"].values]
    groups = sorted(vad_wide["group_id"].unique())

    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 5))
    fig.suptitle("Mean VAD per group × task", fontsize=13, fontweight="bold")

    for ax, dim in zip(axes, dims):
        mat = (
            vad_wide.groupby(["group_id", "task"])[dim]
            .mean()
            .unstack("task")
            .reindex(index=groups, columns=tasks)
        )
        im = ax.imshow(mat.values, aspect="auto", vmin=3, vmax=9,
                       cmap="RdYlGn", interpolation="nearest")
        ax.set_xticks(range(len(tasks)))
        ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=8, rotation=20)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels([g.replace("grp-", "G") for g in groups], fontsize=8)
        ax.set_title(dim.capitalize())
        for i in range(len(groups)):
            for j in range(len(tasks)):
                val = mat.values[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                            fontsize=7, color="k")
        plt.colorbar(im, ax=ax, shrink=0.7, label="Mean rating")

    _save(fig, out, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 – VAD temporal dynamics (within-task trajectory)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_vad_temporal(vad: pd.DataFrame, out: Path, dpi: int) -> None:
    tasks = [t for t in ["T1", "T2", "T3", "T4"] if t in vad["task"].values]
    dims = [d for d in ["valence", "arousal", "dominance"] if d in vad["item_key"].values]

    fig, axes = plt.subplots(len(dims), len(tasks),
                             figsize=(4 * len(tasks), 3.5 * len(dims)),
                             sharex=False, sharey=True)
    fig.suptitle("VAD temporal dynamics within each task",
                 fontsize=13, fontweight="bold")

    for di, dim in enumerate(dims):
        for ti, task in enumerate(tasks):
            ax = axes[di][ti]
            sub = vad[(vad["task"] == task) & (vad["item_key"] == dim)].copy()
            if sub.empty:
                ax.set_visible(False)
                continue
            # normalise time within each session to [0,1]
            for ses, grp in sub.groupby("session_id"):
                t_min, t_max = grp["lsl_clock"].min(), grp["lsl_clock"].max()
                if t_max == t_min:
                    continue
                norm_t = (grp["lsl_clock"] - t_min) / (t_max - t_min)
                for _, p_grp in grp.groupby("participant"):
                    nt = (p_grp["lsl_clock"] - t_min) / (t_max - t_min)
                    ax.plot(nt, p_grp["item_value_num"], "o-", ms=2, lw=0.8,
                            alpha=0.25, color=DIM_COLORS[dim])
            # group mean trend via binning
            sub = sub.copy()
            t_min_all = sub.groupby("session_id")["lsl_clock"].transform("min")
            t_max_all = sub.groupby("session_id")["lsl_clock"].transform("max")
            span = t_max_all - t_min_all
            span = span.replace(0, np.nan)
            sub["norm_t"] = (sub["lsl_clock"] - t_min_all) / span
            bins = np.linspace(0, 1, 8)
            sub["bin"] = pd.cut(sub["norm_t"], bins=bins, labels=False)
            trend = sub.groupby("bin")["item_value_num"].mean()
            bin_centers = (bins[:-1] + bins[1:]) / 2
            ax.plot(bin_centers[trend.index.astype(int)], trend.values,
                    "D-", color="k", ms=5, lw=1.5, zorder=5)
            _add_midline(ax)
            ax.set_ylim(0.5, 9.5)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(2))
            if di == 0:
                ax.set_title(TASK_LABELS.get(task, task), fontsize=9)
            if ti == 0:
                ax.set_ylabel(dim.capitalize(), fontsize=9)
            ax.set_xlabel("Norm. time", fontsize=7)

    _save(fig, out, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 – Within-group VAD agreement (SD across participants)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_vad_agreement(vad_wide: pd.DataFrame, out: Path, dpi: int) -> None:
    dims = [d for d in ["valence", "arousal", "dominance"] if d in vad_wide.columns]
    tasks = [t for t in ["T0", "T1", "T2", "T3", "T4"] if t in vad_wide["task"].values]

    # within-group SD per task per dim
    grp_sd = (
        vad_wide.groupby(["group_id", "task"])[dims]
        .std()
        .reset_index()
    )

    fig, axes = plt.subplots(1, len(dims), figsize=(5 * len(dims), 4))
    fig.suptitle("Within-group VAD disagreement (SD across participants)\nLower = more agreement",
                 fontsize=12, fontweight="bold")

    for ax, dim in zip(axes, dims):
        data = [grp_sd.loc[grp_sd["task"] == t, dim].dropna().values for t in tasks]
        bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "k", "lw": 2})
        for patch in bp["boxes"]:
            patch.set_facecolor(DIM_COLORS[dim])
            patch.set_alpha(0.6)
        ax.set_xticks(range(1, len(tasks) + 1))
        ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], fontsize=8)
        ax.set_ylabel("Within-group SD")
        ax.set_title(dim.capitalize())
        ax.set_ylim(0, None)

    _save(fig, out, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 – Postblock questionnaire radar / bar per task
# ═══════════════════════════════════════════════════════════════════════════════

def fig_postblock_bars(pb: pd.DataFrame, out: Path, dpi: int) -> None:
    task_list = [t for t in ["T1", "T2", "T3", "T4"] if t in POSTBLOCK]
    n_tasks = len(task_list)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Post-task questionnaire results  (1–7 scale, dashed = midpoint 4)",
                 fontsize=13, fontweight="bold")
    axes = axes.flatten()

    colors_task = {"T1": "#fd8d3c", "T2": "#74c476", "T3": "#9e9ac8", "T4": "#f768a1"}

    for ax, task in zip(axes, task_list):
        items = POSTBLOCK[task]
        rows = []
        for key, label in items.items():
            sub = pb[(pb["task"] == task) & (pb["item_key"] == key)]["item_value_num"].dropna()
            if len(sub) == 0:
                continue
            rows.append({"key": key, "label": label,
                         "mean": sub.mean(), "se": sub.sem(), "n": len(sub)})
        if not rows:
            continue
        df_items = pd.DataFrame(rows).sort_values("mean", ascending=True)
        y = np.arange(len(df_items))
        ax.barh(y, df_items["mean"], xerr=df_items["se"],
                color=colors_task.get(task, "#888"), alpha=0.8, edgecolor="white",
                error_kw={"ecolor": "k", "elinewidth": 1, "capsize": 3})
        ax.axvline(4.0, color="grey", lw=0.8, ls="--", alpha=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels(df_items["label"], fontsize=8)
        ax.set_xlabel("Mean ± SE")
        ax.set_xlim(1, 7.5)
        ax.set_title(f"Task {task} — {TASK_LABELS.get(task, task)}", fontsize=10)

    _save(fig, out, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 – Postblock by group heatmap
# ═══════════════════════════════════════════════════════════════════════════════

def fig_postblock_group_heatmap(pb: pd.DataFrame, out: Path, dpi: int) -> None:
    task_list = [t for t in ["T1", "T2", "T3", "T4"] if t in POSTBLOCK]
    groups = sorted(pb["group_id"].unique())

    fig, axes = plt.subplots(1, len(task_list), figsize=(5 * len(task_list), 6))
    fig.suptitle("Post-task questionnaire mean by group", fontsize=13, fontweight="bold")

    for ax, task in zip(axes, task_list):
        items = list(POSTBLOCK[task].keys())
        labels = list(POSTBLOCK[task].values())
        sub = pb[(pb["task"] == task) & (pb["item_key"].isin(items))]
        mat = (
            sub.groupby(["group_id", "item_key"])["item_value_num"]
            .mean()
            .unstack("item_key")
            .reindex(index=groups, columns=items)
        )
        im = ax.imshow(mat.values.T, aspect="auto", vmin=1, vmax=7,
                       cmap="RdYlGn", interpolation="nearest")
        ax.set_yticks(range(len(items)))
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels([g.replace("grp-", "G") for g in groups], fontsize=8)
        ax.set_title(f"Task {task}", fontsize=10)
        for i in range(len(items)):
            for j in range(len(groups)):
                v = mat.values[j, i]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=6, color="k")
        plt.colorbar(im, ax=ax, shrink=0.6, label="Mean")

    _save(fig, out, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 7 – Group decisions (task outcomes)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_group_decisions(df: pd.DataFrame, out: Path, dpi: int) -> None:
    form = df[df["response_type"] == "form"].copy()
    groups = sorted(df["group_id"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Group task decisions", fontsize=13, fontweight="bold")

    # T1 — Candidate selection (pie)
    ax = axes[0]
    t1 = form[(form["task"] == "T1") & (form["item_key"] == "selected_candidate")]
    # one vote per group (take modal vote)
    grp_vote = t1.groupby("group_id")["item_value"].agg(
        lambda x: x.dropna().mode().iloc[0] if len(x.dropna()) else np.nan
    )
    counts = grp_vote.value_counts()
    wedge_colors = ["#6baed6", "#fd8d3c", "#74c476"][:len(counts)]
    ax.pie(counts.values, labels=counts.index, autopct="%1.0f%%",
           colors=wedge_colors, startangle=90,
           wedgeprops={"edgecolor": "white", "linewidth": 1.5})
    ax.set_title("T1 — Candidate selected\n(modal group vote, n groups)")

    # T2 — Format selection (horizontal bar)
    ax2 = axes[1]
    t2 = form[(form["task"] == "T2") & (form["item_key"] == "final_format")]
    grp_choice = t2.groupby("group_id")["item_value"].agg(
        lambda x: x.dropna().mode().iloc[0] if len(x.dropna()) else np.nan
    ).dropna()
    fmt_counts = grp_choice.value_counts().sort_values()
    bars = ax2.barh(fmt_counts.index, fmt_counts.values,
                    color="#74c476", edgecolor="white", alpha=0.85)
    ax2.bar_label(bars, fmt="%d groups", padding=3, fontsize=8)
    ax2.set_xlabel("Number of groups")
    ax2.set_title("T2 — Learning format selected")
    ax2.set_xlim(0, fmt_counts.values.max() * 1.4)

    # T3 — Winning idea (word cloud proxy: count how many groups picked each idea)
    ax3 = axes[2]
    t3 = form[(form["task"] == "T3") & (form["item_key"] == "winning_idea")]
    grp_idea = t3.groupby("group_id")["item_value"].agg(
        lambda x: x.dropna().mode().iloc[0] if len(x.dropna()) else np.nan
    ).dropna()
    # shorten idea labels
    idea_counts = grp_idea.value_counts().sort_values()
    short_labels = [s[:40] + "…" if len(s) > 40 else s for s in idea_counts.index]
    bars3 = ax3.barh(short_labels, idea_counts.values,
                     color="#9e9ac8", edgecolor="white", alpha=0.85)
    ax3.bar_label(bars3, fmt="%d", padding=3, fontsize=8)
    ax3.set_xlabel("Number of groups")
    ax3.set_title("T3 — Winning team activity idea")
    ax3.set_xlim(0, idea_counts.values.max() * 1.4)

    _save(fig, out, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 8 – VAD vs postblock overall_valence comparison
# ═══════════════════════════════════════════════════════════════════════════════

def fig_vad_vs_postblock(vad_wide: pd.DataFrame, pb: pd.DataFrame,
                          out: Path, dpi: int) -> None:
    tasks = [t for t in ["T1", "T2", "T3", "T4"] if t in vad_wide["task"].values]

    ov = pb[pb["item_key"] == "overall_valence"][
        ["group_id", "session_id", "participant", "task", "item_value_num"]
    ].rename(columns={"item_value_num": "postblock_valence"})

    merged = vad_wide.merge(ov, on=["group_id", "session_id", "participant", "task"], how="inner")
    merged = merged.dropna(subset=["valence", "postblock_valence"])

    fig, axes = plt.subplots(1, len(tasks), figsize=(4 * len(tasks), 4), sharey=True, sharex=True)
    fig.suptitle("Continuous VAD valence vs post-task overall valence rating",
                 fontsize=12, fontweight="bold")

    for ax, task in zip(axes, tasks):
        sub = merged[merged["task"] == task]
        if sub.empty:
            ax.set_visible(False)
            continue
        ax.scatter(sub["valence"], sub["postblock_valence"],
                   color=TASK_COLORS.get(task, "#888"), alpha=0.6, s=30, edgecolors="w")
        r, p = sp_stats.pearsonr(sub["valence"], sub["postblock_valence"])
        # regression line
        xr = np.linspace(sub["valence"].min(), sub["valence"].max(), 50)
        m, b = np.polyfit(sub["valence"], sub["postblock_valence"], 1)
        ax.plot(xr, m * xr + b, "k--", lw=1.2)
        ax.set_title(f"{TASK_LABELS.get(task, task)}\nr = {r:.2f}, p = {p:.3f}", fontsize=9)
        ax.set_xlabel("Continuous valence (mean)")
        if ax == axes[0]:
            ax.set_ylabel("Post-task overall valence")
        ax.set_xlim(1, 9.5)
        ax.set_ylim(1, 7.5)

    _save(fig, out, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 9 – Perceived dominance ratings (T4 postblock)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_dominance_ratings(pb: pd.DataFrame, out: Path, dpi: int) -> None:
    dom_items = ["dominance_p1", "dominance_p2", "dominance_p3", "dominance_p4"]
    seat_labels = ["Seat P1", "Seat P2", "Seat P3", "Seat P4"]
    dom = pb[(pb["task"] == "T4") & (pb["item_key"].isin(dom_items))].copy()
    groups = sorted(dom["group_id"].unique())

    if dom.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("T4 — Perceived dominance by seat position",
                 fontsize=13, fontweight="bold")

    # Overall distribution
    ax = axes[0]
    data = [dom[dom["item_key"] == d]["item_value_num"].dropna().values for d in dom_items]
    bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "k", "lw": 2})
    colors_dom = ["#6baed6", "#fd8d3c", "#74c476", "#9e9ac8"]
    for patch, c in zip(bp["boxes"], colors_dom):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, 5))
    ax.set_xticklabels(seat_labels, fontsize=9)
    ax.set_ylabel("Perceived dominance (1–7)")
    ax.set_title("Overall distribution by seat")
    ax.axhline(4.0, color="grey", lw=0.8, ls="--", alpha=0.5)

    # Per-group heatmap
    ax2 = axes[1]
    mat = (
        dom.groupby(["group_id", "item_key"])["item_value_num"]
        .mean()
        .unstack("item_key")
        .reindex(index=groups, columns=dom_items)
    )
    im = ax2.imshow(mat.values, aspect="auto", vmin=1, vmax=7,
                    cmap="RdYlGn", interpolation="nearest")
    ax2.set_yticks(range(len(groups)))
    ax2.set_yticklabels([g.replace("grp-", "G") for g in groups], fontsize=8)
    ax2.set_xticks(range(4))
    ax2.set_xticklabels(seat_labels, fontsize=8)
    ax2.set_title("Mean perceived dominance by group")
    for i in range(len(groups)):
        for j in range(4):
            v = mat.values[i, j]
            if not np.isnan(v):
                ax2.text(j, i, f"{v:.1f}", ha="center", va="center",
                         fontsize=7, color="k")
    plt.colorbar(im, ax=ax2, shrink=0.7, label="Mean dominance")

    _save(fig, out, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Summary table writers
# ═══════════════════════════════════════════════════════════════════════════════

def write_tables(vad_wide: pd.DataFrame, pb: pd.DataFrame,
                 df_all: pd.DataFrame, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)

    # VAD summary per task
    dims = [d for d in ["valence", "arousal", "dominance"] if d in vad_wide.columns]
    vad_task = (
        vad_wide.groupby("task")[dims]
        .agg(["mean", "std", "count"])
        .round(3)
    )
    vad_task.columns = ["_".join(c) for c in vad_task.columns]
    p = results_dir / "vad_by_task.tsv"
    vad_task.to_csv(p, sep="\t")
    log.info("Wrote %s", p)

    # VAD per participant × task
    p2 = results_dir / "vad_participant_task.tsv"
    vad_wide.to_csv(p2, sep="\t", index=False)
    log.info("Wrote %s", p2)

    # Postblock summary per task × item
    pb_summary = (
        pb[pb["item_key"].isin(
            [k for t in POSTBLOCK.values() for k in t]
        )]
        .groupby(["task", "item_key"])["item_value_num"]
        .agg(mean="mean", std="std", n="count")
        .round(3)
        .reset_index()
    )
    p3 = results_dir / "postblock_summary.tsv"
    pb_summary.to_csv(p3, sep="\t", index=False)
    log.info("Wrote %s", p3)

    # Group decisions
    form = df_all[df_all["response_type"] == "form"].copy()
    decision_items = {
        "T1": "selected_candidate",
        "T2": "final_format",
        "T3": "winning_idea",
    }
    rows = []
    for task, key in decision_items.items():
        sub = form[(form["task"] == task) & (form["item_key"] == key)]
        grp_vote = sub.groupby("group_id")["item_value"].agg(
            lambda x: x.dropna().mode().iloc[0] if len(x.dropna()) else np.nan
        ).reset_index()
        grp_vote["task"] = task
        grp_vote.columns = ["group_id", "decision", "task"]
        rows.append(grp_vote)
    p4 = results_dir / "group_decisions.tsv"
    pd.concat(rows).to_csv(p4, sep="\t", index=False)
    log.info("Wrote %s", p4)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root", default="F:/bids_release_no_video")
    ap.add_argument("--results-dir", default="results/task_responses")
    ap.add_argument("--figures-dir", default="figures/task_responses")
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    dpi = args.dpi

    df_all = load_all_responses(Path(args.bids_root))

    vad_raw = get_vad(df_all)
    vad_wide = vad_participant_task_summary(vad_raw)
    pb = get_postblock(df_all)

    log.info("VAD: %d ratings → %d participant-task summaries", len(vad_raw), len(vad_wide))
    log.info("Postblock: %d rows", len(pb))

    # Print quick stats
    dims = [d for d in ["valence", "arousal", "dominance"] if d in vad_wide.columns]
    for dim in dims:
        for task in sorted(vad_wide["task"].unique()):
            vals = vad_wide.loc[vad_wide["task"] == task, dim].dropna()
            if len(vals):
                log.info("VAD %s %s: mean=%.2f SD=%.2f n=%d",
                         task, dim, vals.mean(), vals.std(), len(vals))

    write_tables(vad_wide, pb, df_all, results_dir)

    fig_vad_by_task(vad_wide, figures_dir / "vad_by_task.png", dpi)
    fig_vad_group_task_heatmap(vad_wide, figures_dir / "vad_group_task_heatmap.png", dpi)
    fig_vad_temporal(vad_raw, figures_dir / "vad_temporal_dynamics.png", dpi)
    fig_vad_agreement(vad_wide, figures_dir / "vad_within_group_agreement.png", dpi)
    fig_postblock_bars(pb, figures_dir / "postblock_ratings_by_task.png", dpi)
    fig_postblock_group_heatmap(pb, figures_dir / "postblock_group_heatmap.png", dpi)
    fig_group_decisions(df_all, figures_dir / "group_decisions.png", dpi)
    fig_vad_vs_postblock(vad_wide, pb, figures_dir / "vad_vs_postblock_valence.png", dpi)
    fig_dominance_ratings(pb, figures_dir / "perceived_dominance_by_seat.png", dpi)

    log.info("Done. Figures → %s | Tables → %s", figures_dir, results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
