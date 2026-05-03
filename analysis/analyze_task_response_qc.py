"""
Quality-control and supplementary summaries for task responses.

This companion script keeps paper-facing QC and appendix-style response audits
separate from the core task-response figures in ``analyze_task_responses.py``.

It focuses on stable, descriptive summaries that are easy to regenerate:
  1. Response-type counts and coverage by task
  2. Cross-stream presence (form, VAD, postblock) by group-task block
  3. Postblock completeness by task and item
  4. Trust summaries (where trust items exist)
  5. Familiarity summaries (where familiarity items exist)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.features.analyze_task_responses import TASK_LABELS, get_postblock, load_all_responses

LOG = logging.getLogger("analyze_task_response_qc")
TASK_ORDER = ["T0", "T1", "T2", "T3", "T4"]
CORE_TASK_ORDER = ["T1", "T2", "T3", "T4"]
STREAM_ORDER = ["form", "vad", "postblock"]
TRUST_ITEMS = ["trust_next", "trust_front", "trust_angle"]
PAPER_POSTBLOCK_ITEMS = {
    "T1": [
        "decision_confidence",
        "dominance_p1",
        "dominance_p2",
        "dominance_p3",
        "dominance_p4",
        "engagement",
        "equality_of_contribution",
        "familiarity_p1",
        "familiarity_p2",
        "familiarity_p3",
        "familiarity_p4",
        "info_sharing",
        "manipcheck_t1",
        "mental_demand",
        "overall_valence",
        "perceived_control",
        "team_coordination",
        "voice_inclusion",
    ],
    "T2": [
        "cooperative",
        "dominance_p1",
        "dominance_p2",
        "dominance_p3",
        "dominance_p4",
        "engagement",
        "manipcheck_t2",
        "mental_demand",
        "overall_valence",
        "perceived_control",
        "satisfaction",
        "trust_angle",
        "trust_front",
        "trust_next",
        "voice_inclusion",
    ],
    "T3": [
        "confidence",
        "dominance_p1",
        "dominance_p2",
        "dominance_p3",
        "dominance_p4",
        "engagement",
        "fairness",
        "idea_diversity",
        "idea_quality",
        "manipcheck_t3",
        "mental_demand",
        "overall_valence",
        "psych_safety",
        "satisfaction",
        "team_coordination",
        "voice_inclusion",
    ],
    "T4": [
        "expectation_match",
        "fairness",
        "overall_valence",
        "regret",
        "social_concern",
        "trust_angle",
        "trust_front",
        "trust_next",
    ],
}


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    LOG.info("Wrote %s", path)


def build_task_participant_units(df_all: pd.DataFrame) -> pd.DataFrame:
    return (
        df_all.dropna(subset=["participant", "task"])
        .drop_duplicates(subset=["session_id", "group_id", "participant", "task"])[
            ["session_id", "group_id", "participant", "task"]
        ]
        .copy()
    )


def deduplicate_session_level_responses(df_all: pd.DataFrame) -> pd.DataFrame:
    dedup_cols = [
        "session_id",
        "group_id",
        "participant",
        "task",
        "response_type",
        "item_key",
        "item_value",
        "lsl_clock",
    ]
    present = [col for col in dedup_cols if col in df_all.columns]
    return df_all.drop_duplicates(subset=present).copy()


def build_response_type_counts(df_all: pd.DataFrame) -> pd.DataFrame:
    counts = (
        df_all.groupby(["task", "response_type"])
        .size()
        .rename("n_rows")
        .reset_index()
    )
    return counts.sort_values(["task", "response_type"]).reset_index(drop=True)


def build_cross_stream_presence(df_all: pd.DataFrame) -> pd.DataFrame:
    sub = (
        df_all[df_all["response_type"].isin(STREAM_ORDER)]
        .drop_duplicates(subset=["group_id", "session_id", "task", "response_type"])
        .assign(present=1)
    )
    wide = (
        sub.pivot_table(
            index=["group_id", "session_id", "task"],
            columns="response_type",
            values="present",
            fill_value=0,
        )
        .reset_index()
    )
    for stream in STREAM_ORDER:
        if stream not in wide.columns:
            wide[stream] = 0
        wide[stream] = wide[stream].astype(int)
    wide["all_streams_present"] = (wide[STREAM_ORDER].sum(axis=1) == len(STREAM_ORDER)).astype(int)
    return wide.sort_values(["task", "group_id"]).reset_index(drop=True)


def build_postblock_item_completeness(df_all: pd.DataFrame, pb: pd.DataFrame) -> pd.DataFrame:
    task_units = build_task_participant_units(df_all)
    expected_by_task = (
        task_units[task_units["task"].isin(CORE_TASK_ORDER)]
        .groupby("task")
        .size()
        .rename("expected_units")
    )
    subset_cols = [col for col in ["session_id", "participant", "task", "item_key"] if col in pb.columns]
    keep_cols = [col for col in ["session_id", "participant", "task", "item_key", "item_value_num"] if col in pb.columns]
    dedup = pb.drop_duplicates(subset=subset_cols)[keep_cols].copy()
    counts = (
        dedup.groupby(["task", "item_key"])["item_value_num"]
        .agg(n_present="count", mean="mean", std="std")
        .reset_index()
    )
    counts = counts[counts["item_key"] != "response"].copy()
    counts["expected_units"] = counts["task"].map(expected_by_task).fillna(0).astype(int)
    counts["n_missing"] = counts["expected_units"] - counts["n_present"]
    counts["pct_present"] = np.where(
        counts["expected_units"] > 0,
        100.0 * counts["n_present"] / counts["expected_units"],
        np.nan,
    )
    return counts.sort_values(["task", "item_key"]).reset_index(drop=True)


def build_task_completion_summary(df_all: pd.DataFrame) -> pd.DataFrame:
    units = build_task_participant_units(df_all)
    return (
        units[units["task"].isin(CORE_TASK_ORDER)]
        .groupby("task")
        .agg(
            n_groups=("group_id", "nunique"),
            n_participant_task=("participant", "count"),
        )
        .reset_index()
        .sort_values("task")
        .reset_index(drop=True)
    )


def build_trust_summary(pb: pd.DataFrame) -> pd.DataFrame:
    trust = pb[pb["item_key"].isin(TRUST_ITEMS)].copy()
    if trust.empty:
        return pd.DataFrame(columns=["group_id", "session_id", "participant", "task", "trust_mean"])
    return (
        trust.groupby(["group_id", "session_id", "participant", "task"])["item_value_num"]
        .mean()
        .rename("trust_mean")
        .reset_index()
        .sort_values(["task", "group_id", "participant"])
        .reset_index(drop=True)
    )


def build_familiarity_summary(pb: pd.DataFrame) -> pd.DataFrame:
    fam = pb[pb["item_key"].str.startswith("familiarity_", na=False)].copy()
    if fam.empty:
        return pd.DataFrame(
            columns=["group_id", "session_id", "participant", "task", "familiarity_mean"]
        )
    return (
        fam.groupby(["group_id", "session_id", "participant", "task"])["item_value_num"]
        .mean()
        .rename("familiarity_mean")
        .reset_index()
        .sort_values(["task", "group_id", "participant"])
        .reset_index(drop=True)
    )


def build_paper_postblock_core_summary(completeness: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for task, items in PAPER_POSTBLOCK_ITEMS.items():
        sub = completeness[(completeness["task"] == task) & (completeness["item_key"].isin(items))].copy()
        if sub.empty:
            continue
        rows.append(
            {
                "task": task,
                "n_items": len(items),
                "n_present": int(sub["n_present"].sum()),
                "n_expected": int(sub["expected_units"].sum()),
                "n_missing": int(sub["n_missing"].sum()),
                "pct_present": 100.0 * float(sub["n_present"].sum()) / float(sub["expected_units"].sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_missingness_heatmap(completeness: pd.DataFrame, out: Path, dpi: int) -> None:
    tasks = [t for t in CORE_TASK_ORDER if t in completeness["task"].unique()]
    items = sorted(completeness["item_key"].unique())
    mat = (
        completeness.pivot_table(
            index="item_key",
            columns="task",
            values="pct_present",
            aggfunc="mean",
        )
        .reindex(index=items, columns=tasks)
    )
    fig, ax = plt.subplots(figsize=(1.8 * max(len(tasks), 1), 0.28 * max(len(items), 1) + 3))
    im = ax.imshow(mat.values, aspect="auto", vmin=80, vmax=100, cmap="YlGn")
    ax.set_title("Postblock completeness by item and task", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in tasks], rotation=20, ha="right")
    ax.set_yticks(range(len(items)))
    ax.set_yticklabels(items, fontsize=7)
    for i in range(len(items)):
        for j in range(len(tasks)):
            val = mat.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6)
    plt.colorbar(im, ax=ax, shrink=0.75, label="% present")
    _save(fig, out, dpi)


def plot_qc_dashboard(
    response_counts: pd.DataFrame,
    stream_presence: pd.DataFrame,
    completion: pd.DataFrame,
    out: Path,
    dpi: int,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle("Task-response QC dashboard", fontsize=13, fontweight="bold")

    ax = axes[0]
    rc = response_counts[response_counts["task"].isin(CORE_TASK_ORDER)].copy()
    if not rc.empty:
        pivot = (
            rc.pivot_table(index="task", columns="response_type", values="n_rows", fill_value=0)
            .reindex(index=CORE_TASK_ORDER, columns=STREAM_ORDER, fill_value=0)
        )
        x = np.arange(len(pivot.index))
        width = 0.24
        for idx, stream in enumerate(STREAM_ORDER):
            ax.bar(x + (idx - 1) * width, pivot[stream].values, width=width, label=stream, alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([TASK_LABELS.get(t, t) for t in pivot.index], rotation=20, ha="right")
    ax.set_title("Response rows by task and stream")
    ax.set_ylabel("Rows")
    ax.legend(fontsize=8)

    ax = axes[1]
    if not stream_presence.empty:
        coverage = (
            stream_presence.groupby("task")["all_streams_present"]
            .mean()
            .reindex(CORE_TASK_ORDER)
            .fillna(0.0)
            * 100.0
        )
        ax.bar(range(len(coverage)), coverage.values, color="#74c476", edgecolor="white", alpha=0.9)
        ax.set_xticks(range(len(coverage)))
        ax.set_xticklabels([TASK_LABELS.get(t, t) for t in coverage.index], rotation=20, ha="right")
        ax.set_ylim(0, 105)
        for idx, val in enumerate(coverage.values):
            ax.text(idx, min(val + 2, 102), f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
    ax.set_title("Group-task blocks with all streams present")
    ax.set_ylabel("% blocks")

    ax = axes[2]
    if not completion.empty:
        task_mean = completion.groupby("task")["pct_present"].mean().reindex(CORE_TASK_ORDER).dropna()
        ax.bar(range(len(task_mean)), task_mean.values, color="#6baed6", edgecolor="white", alpha=0.9)
        ax.set_xticks(range(len(task_mean)))
        ax.set_xticklabels([TASK_LABELS.get(t, t) for t in task_mean.index], rotation=20, ha="right")
        ax.set_ylim(80, 101)
        for idx, val in enumerate(task_mean.values):
            ax.text(idx, min(val + 0.4, 100.5), f"{val:.1f}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Mean postblock completeness")
    ax.set_ylabel("% present")

    _save(fig, out, dpi)


def plot_trust_by_task(trust_summary: pd.DataFrame, out: Path, dpi: int) -> None:
    if trust_summary.empty:
        return
    task_order = [t for t in ["T2", "T4"] if t in trust_summary["task"].unique()]
    grouped = trust_summary.groupby("task")["trust_mean"]
    means = grouped.mean().reindex(task_order)
    stds = grouped.std().reindex(task_order).fillna(0.0)
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(
        range(len(task_order)),
        means.values,
        yerr=stds.values,
        color=["#74c476", "#f768a1"][: len(task_order)],
        edgecolor="white",
        alpha=0.85,
        error_kw={"ecolor": "k", "elinewidth": 1, "capsize": 3},
    )
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.axhline(4.0, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.set_xticks(range(len(task_order)))
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in task_order])
    ax.set_ylabel("Mean trust (1-7)")
    ax.set_title("Trust towards group members")
    ax.set_ylim(1, 7.2)
    _save(fig, out, dpi)


def plot_familiarity_by_task(familiarity_summary: pd.DataFrame, out: Path, dpi: int) -> None:
    if familiarity_summary.empty:
        return
    task_order = [t for t in ["T1", "T2"] if t in familiarity_summary["task"].unique()]
    data = [
        familiarity_summary.loc[familiarity_summary["task"] == task, "familiarity_mean"].dropna().values
        for task in task_order
    ]
    fig, ax = plt.subplots(figsize=(5, 4))
    bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "k", "lw": 1.5})
    for patch, color in zip(bp["boxes"], ["#fd8d3c", "#74c476"][: len(task_order)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.axhline(4.0, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.set_xticks(range(1, len(task_order) + 1))
    ax.set_xticklabels([TASK_LABELS.get(t, t) for t in task_order])
    ax.set_ylabel("Mean familiarity (1-7)")
    ax.set_title("Familiarity with group members")
    ax.set_ylim(1, 7.2)
    _save(fig, out, dpi)


def write_outputs(
    results_dir: Path,
    response_counts: pd.DataFrame,
    stream_presence: pd.DataFrame,
    completion: pd.DataFrame,
    task_completion: pd.DataFrame,
    trust_summary: pd.DataFrame,
    familiarity_summary: pd.DataFrame,
    postblock_core_summary: pd.DataFrame,
) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "response_type_counts.tsv": response_counts,
        "cross_stream_presence.tsv": stream_presence,
        "postblock_item_completeness.tsv": completion,
        "task_completion_summary.tsv": task_completion,
        "trust_summary.tsv": trust_summary,
        "familiarity_summary.tsv": familiarity_summary,
        "postblock_core_summary.tsv": postblock_core_summary,
    }
    for filename, df in outputs.items():
        path = results_dir / filename
        df.to_csv(path, sep="\t", index=False)
        LOG.info("Wrote %s", path)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Run QC and supplementary task-response summaries.")
    ap.add_argument("--bids-root", default="F:/bids_release_no_video")
    ap.add_argument("--results-dir", default="results/task_response_qc")
    ap.add_argument("--figures-dir", default="figures/task_response_qc")
    ap.add_argument("--dpi", type=int, default=180)
    ap.add_argument("--verbose", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    df_all = deduplicate_session_level_responses(load_all_responses(Path(args.bids_root)))
    pb = get_postblock(df_all)

    response_counts = build_response_type_counts(df_all)
    stream_presence = build_cross_stream_presence(df_all)
    completion = build_postblock_item_completeness(df_all, pb)
    task_completion = build_task_completion_summary(df_all)
    trust_summary = build_trust_summary(pb)
    familiarity_summary = build_familiarity_summary(pb)
    postblock_core_summary = build_paper_postblock_core_summary(completion)

    write_outputs(
        results_dir,
        response_counts,
        stream_presence,
        completion,
        task_completion,
        trust_summary,
        familiarity_summary,
        postblock_core_summary,
    )

    plot_missingness_heatmap(completion, figures_dir / "missing_data_heatmap.png", args.dpi)
    plot_qc_dashboard(
        response_counts,
        stream_presence,
        completion,
        figures_dir / "qc_audit_dashboard.png",
        args.dpi,
    )
    plot_trust_by_task(trust_summary, figures_dir / "trust_mean_by_task.png", args.dpi)
    plot_familiarity_by_task(
        familiarity_summary,
        figures_dir / "familiarity_distribution_by_task.png",
        args.dpi,
    )

    LOG.info("Done. Figures -> %s | Tables -> %s", figures_dir, results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
