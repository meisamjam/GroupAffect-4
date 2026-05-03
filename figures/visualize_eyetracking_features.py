"""Create paper-facing visualizations from extracted Tobii eye-tracking features."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("visualize_eyetracking_features")

TASK_ORDER = ["T0", "T1", "T2", "T3", "T4"]
TASK_LABELS = ["T1", "T2", "T3", "T4"]

# Features shown in the task-delta summary (T0-normalised)
DELTA_FEATURES = [
    ("pupil_mean_delta_t0", "Pupil diameter change vs T0", "mm"),
    ("gaze_x_mean_delta_t0", "Gaze X position change vs T0", "norm. units"),
    ("gaze_y_mean_delta_t0", "Gaze Y position change vs T0", "norm. units"),
    ("pupil_left_mean_delta_t0", "Left-eye pupil change vs T0", "mm"),
    ("pupil_right_mean_delta_t0", "Right-eye pupil change vs T0", "mm"),
]

# Features shown in the per-task signal overview grid
OVERVIEW_FEATURES = [
    ("pupil_mean", "Pupil diameter", "mm"),
    ("gaze_valid_frac", "Gaze validity fraction", "fraction"),
    ("blink_rate_per_min", "Blink rate", "blinks/min"),
    ("gaze_dispersion", "Gaze dispersion", "norm. units"),
    ("gaze_velocity_mean", "Gaze velocity (mean)", "norm. units/s"),
    ("pupil_std", "Pupil diameter std.", "mm"),
]

_PALETTE = [
    "#2f6f9f",
    "#8e5a2b",
    "#6b7f2a",
    "#a33f2f",
    "#7c4d9e",
    "#5a8f7b",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize paper-ready eye-tracking feature and QC tables."
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("features"),
        help="Directory containing et_participant_task.tsv and et_qc_summary.tsv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures") / "et",
        help="Directory where PNG figures are written.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Output figure DPI.")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return parser


def _read_tables(features_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_path = features_dir / "et_participant_task.tsv"
    qc_path = features_dir / "et_qc_summary.tsv"
    if not task_path.exists():
        raise FileNotFoundError(f"Missing participant-task table: {task_path}")
    if not qc_path.exists():
        raise FileNotFoundError(f"Missing QC table: {qc_path}")
    task = pd.read_csv(task_path, sep="\t")
    qc = pd.read_csv(qc_path, sep="\t")
    # Normalise task column name
    if "task" in task.columns and "task_id" not in task.columns:
        task = task.rename(columns={"task": "task_id"})
    if "task" in qc.columns and "task_id" not in qc.columns:
        qc = qc.rename(columns={"task": "task_id"})
    return task, qc


# ── Figure 1: task-level pupil & gaze delta summary ──────────────────────────


def _task_delta_summary(task: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    plot_df = task[task["task_id"].isin(TASK_LABELS)].copy()
    avail = [(col, title, unit) for col, title, unit in DELTA_FEATURES if col in plot_df.columns]
    if not avail:
        LOG.warning("No delta columns found; skipping task delta summary figure.")
        return out_dir / "et_task_delta_summary.png"

    ncols = 2
    nrows = int(np.ceil(len(avail) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 3.5 * nrows), constrained_layout=True)
    fig.suptitle("Task-level eye-tracking changes relative to T0 baseline",
                 fontsize=14, fontweight="bold")
    x = np.arange(len(TASK_LABELS))
    flat_axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, (col, title, unit), color in zip(flat_axes, avail, _PALETTE, strict=False):
        grouped = plot_df.groupby("task_id")[col]
        means = grouped.mean().reindex(TASK_LABELS)
        sems = grouped.sem().reindex(TASK_LABELS)
        counts = grouped.count().reindex(TASK_LABELS).fillna(0).astype(int)
        ax.axhline(0.0, color="#6f6f6f", linewidth=0.9, linestyle="--")
        ax.errorbar(
            x,
            means.to_numpy(dtype=float),
            yerr=sems.to_numpy(dtype=float),
            marker="o",
            linewidth=2.0,
            capsize=4,
            color=color,
        )
        ax.set_xticks(x, [f"{t}\nn={counts.loc[t]}" for t in TASK_LABELS])
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.25)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in flat_axes[len(avail):]:
        ax.set_axis_off()

    out_path = out_dir / "et_task_delta_summary.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ── Figure 2: gaze validity vs pupil coverage quality scatter ─────────────────


def _gaze_pupil_quality(task: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    df = task.copy()
    df["pupil_coverage_frac"] = 1.0 - df["pupil_missing_frac"].clip(0.0, 1.0)
    df["gaze_usable"] = (df["gaze_valid_frac"] >= 0.70) & (
        ~df["qc_flag"].str.contains("gaze_out_of_bounds", na=False)
    )
    df["pupil_usable"] = (df["pupil_missing_frac"] <= 0.30) & (
        ~df["qc_flag"].str.contains("pupil_implausible", na=False)
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    fig.suptitle("Eye-tracking data quality overview", fontsize=14, fontweight="bold")

    # Panel 1: gaze validity vs pupil coverage scatter
    colors = df.apply(
        lambda r: (
            "#2f7d5c" if r["gaze_usable"] and r["pupil_usable"]
            else "#e8a83a" if r["gaze_usable"] or r["pupil_usable"]
            else "#b95f45"
        ),
        axis=1,
    ).to_numpy()
    axes[0].scatter(
        df["gaze_valid_frac"],
        df["pupil_coverage_frac"],
        c=colors,
        alpha=0.75,
        s=36,
        edgecolors="none",
    )
    axes[0].axvline(0.70, color="#6f6f6f", linewidth=0.9, linestyle="--")
    axes[0].axhline(0.70, color="#6f6f6f", linewidth=0.9, linestyle="--")
    axes[0].set_xlabel("Gaze validity fraction")
    axes[0].set_ylabel("Pupil coverage fraction")
    axes[0].set_title("Validity vs coverage\n(green=both usable, amber=one, red=neither)")
    axes[0].set_xlim(0.0, 1.05)
    axes[0].set_ylim(0.0, 1.05)

    # Panel 2: pupil mean distribution for usable rows, coloured by task
    task_colors = {"T0": "#888888", "T1": "#2f6f9f", "T2": "#a33f2f",
                   "T3": "#6b7f2a", "T4": "#7c4d9e"}
    usable = df[df["pupil_usable"] & df["pupil_mean"].notna()]
    for task_id in TASK_ORDER:
        vals = usable.loc[usable["task_id"] == task_id, "pupil_mean"].to_numpy()
        if vals.size == 0:
            continue
        axes[1].hist(
            vals,
            bins=14,
            alpha=0.6,
            color=task_colors.get(task_id, "#555555"),
            label=task_id,
            density=True,
        )
    axes[1].set_xlabel("Pupil diameter (mm)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("Pupil diameter distribution\n(pupil-usable rows, by task)")
    axes[1].legend(frameon=False, fontsize=9)

    # Panel 3: blink rate distribution by task (box plot)
    blink_data = [
        df.loc[df["task_id"] == t, "blink_rate_per_min"].dropna().to_numpy()
        for t in TASK_ORDER
    ]
    bp = axes[2].boxplot(
        blink_data,
        patch_artist=True,
        medianprops=dict(color="white", linewidth=1.5),
        widths=0.55,
    )
    for patch, t in zip(bp["boxes"], TASK_ORDER):
        patch.set_facecolor(task_colors.get(t, "#555555"))
        patch.set_alpha(0.8)
    axes[2].set_xticks(range(1, len(TASK_ORDER) + 1), TASK_ORDER)
    axes[2].axhline(3, color="#6f6f6f", linewidth=0.8, linestyle=":", alpha=0.7)
    axes[2].axhline(45, color="#6f6f6f", linewidth=0.8, linestyle=":", alpha=0.7)
    axes[2].set_xlabel("Task")
    axes[2].set_ylabel("Blinks / min")
    axes[2].set_title("Estimated blink rate by task\n(dotted lines = QC thresholds 3–45)")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

    out_path = out_dir / "et_gaze_pupil_quality.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ── Figure 3: per-task signal overview grid ───────────────────────────────────


def _signal_overview_grid(task: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    plot_df = task[task["task_id"].isin(TASK_ORDER)].copy()
    avail = [(col, title, unit) for col, title, unit in OVERVIEW_FEATURES if col in plot_df.columns]

    ncols = 3
    nrows = int(np.ceil(len(avail) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 3.2 * nrows), constrained_layout=True)
    fig.suptitle("Eye-tracking signal summary by task", fontsize=14, fontweight="bold")
    x = np.arange(len(TASK_ORDER))
    flat_axes = axes.ravel() if hasattr(axes, "ravel") else [axes]

    for ax, (col, title, unit), color in zip(flat_axes, avail, _PALETTE, strict=False):
        grouped = plot_df.groupby("task_id")[col]
        means = grouped.mean().reindex(TASK_ORDER)
        sems = grouped.sem().reindex(TASK_ORDER)
        counts = grouped.count().reindex(TASK_ORDER).fillna(0).astype(int)
        ax.bar(x, means.to_numpy(dtype=float), color=color, alpha=0.75, width=0.6)
        ax.errorbar(
            x,
            means.to_numpy(dtype=float),
            yerr=sems.to_numpy(dtype=float),
            fmt="none",
            color="#333333",
            capsize=3,
            linewidth=1.2,
        )
        ax.set_xticks(x, [f"{t}\nn={counts.loc[t]}" for t in TASK_ORDER])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(unit)
        ax.grid(axis="y", alpha=0.22)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in flat_axes[len(avail):]:
        ax.set_axis_off()

    out_path = out_dir / "et_signal_overview_grid.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ── Figure 4: session × task coverage heatmap ────────────────────────────────


def _session_task_coverage(qc: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    sessions = sorted(qc["session_id"].dropna().unique().tolist())
    tasks = [t for t in TASK_ORDER if t in set(qc["task_id"].dropna())]

    gaze_pivot = (
        qc.assign(usable=qc["gaze_usable"].astype(bool))
        .groupby(["session_id", "task_id"])["usable"]
        .sum()
        .unstack("task_id")
        .reindex(index=sessions, columns=tasks)
        .fillna(0)
    )
    pupil_pivot = (
        qc.assign(usable=qc["pupil_usable"].astype(bool))
        .groupby(["session_id", "task_id"])["usable"]
        .sum()
        .unstack("task_id")
        .reindex(index=sessions, columns=tasks)
        .fillna(0)
    )

    fig_height = max(5.0, 0.45 * len(sessions) + 2.0)
    fig, axes = plt.subplots(1, 2, figsize=(14, fig_height), constrained_layout=True)
    fig.suptitle("Tobii coverage: usable participant-slots by session and task",
                 fontsize=13, fontweight="bold")

    for ax, pivot, label, cmap in [
        (axes[0], gaze_pivot, "gaze-usable", "YlGn"),
        (axes[1], pupil_pivot, "pupil-usable", "YlOrRd"),
    ]:
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto",
                       vmin=0, vmax=4, cmap=cmap)
        ax.set_title(f"Participants {label} per slot", fontsize=11)
        ax.set_xlabel("Task")
        ax.set_ylabel("Session")
        ax.set_xticks(np.arange(len(tasks)), tasks)
        ax.set_yticks(np.arange(len(sessions)), sessions)
        ax.tick_params(axis="y", labelsize=8)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                v = int(pivot.iloc[i, j])
                ax.text(j, i, str(v), ha="center", va="center",
                        color="black" if v >= 3 else "white", fontsize=9)
        cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        cbar.set_label("# participants")

    out_path = out_dir / "et_session_task_coverage.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ── Figure 5: QC overview ─────────────────────────────────────────────────────


def _qc_overview(qc: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    import matplotlib.pyplot as plt

    df = qc.copy()
    df["gaze_usable"] = df["gaze_usable"].astype(bool)
    df["pupil_usable"] = df["pupil_usable"].astype(bool)
    df["both"] = df["gaze_usable"] & df["pupil_usable"]
    df["gaze_only"] = df["gaze_usable"] & ~df["pupil_usable"]
    df["pupil_only"] = df["pupil_usable"] & ~df["gaze_usable"]
    df["neither"] = ~df["gaze_usable"] & ~df["pupil_usable"]

    sessions = sorted(df["session_id"].dropna().unique().tolist())
    by_session = (
        df.groupby("session_id")[["both", "gaze_only", "pupil_only", "neither"]]
        .sum()
        .reindex(sessions)
        .fillna(0)
    )

    flags = (
        df["qc_flag"]
        .fillna("missing_et")
        .str.split(";")
        .explode()
        .value_counts()
        .drop(labels=["ok"], errors="ignore")
        .head(10)
        .sort_values()
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5.5, 0.38 * len(sessions) + 2.5)),
                             constrained_layout=True)
    fig.suptitle("Eye-tracking QC overview", fontsize=14, fontweight="bold")

    # Stacked bar: usability tiers per session
    y = np.arange(len(sessions))
    tier_colors = {
        "both": "#2f7d5c",
        "gaze_only": "#5b9bd5",
        "pupil_only": "#e8a83a",
        "neither": "#b95f45",
    }
    tier_labels = {
        "both": "gaze + pupil usable",
        "gaze_only": "gaze only",
        "pupil_only": "pupil only",
        "neither": "neither usable",
    }
    left = np.zeros(len(sessions))
    for tier in ["both", "gaze_only", "pupil_only", "neither"]:
        vals = by_session[tier].to_numpy(dtype=float)
        axes[0].barh(y, vals, left=left, color=tier_colors[tier],
                     label=tier_labels[tier], height=0.7)
        left += vals
    axes[0].set_yticks(y, sessions)
    axes[0].tick_params(axis="y", labelsize=8)
    axes[0].set_xlabel("participant-task rows")
    axes[0].set_title("Usability tiers by session")
    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    axes[0].spines["top"].set_visible(False)
    axes[0].spines["right"].set_visible(False)

    # Horizontal bar: most common QC flags
    axes[1].barh(np.arange(len(flags)), flags.to_numpy(), color="#4e6fae")
    axes[1].set_yticks(np.arange(len(flags)), flags.index)
    axes[1].set_xlabel("participant-task rows affected")
    axes[1].set_title("Most common QC flags")
    total = len(df)
    for i, v in enumerate(flags.to_numpy()):
        axes[1].text(v + 0.3, i, f"{100*v/total:.0f}%", va="center", fontsize=8)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["right"].set_visible(False)

    out_path = out_dir / "et_qc_overview.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    features_dir = args.features_dir.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    task, qc = _read_tables(features_dir)

    paths = [
        _task_delta_summary(task, out_dir, args.dpi),
        _gaze_pupil_quality(task, out_dir, args.dpi),
        _signal_overview_grid(task, out_dir, args.dpi),
        _session_task_coverage(qc, out_dir, args.dpi),
        _qc_overview(qc, out_dir, args.dpi),
    ]
    for path in paths:
        LOG.info("Wrote %s", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
