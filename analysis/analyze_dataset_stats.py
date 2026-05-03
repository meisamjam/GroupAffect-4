"""
Dataset statistics and demographics analysis.

Produces summary tables and figures covering:
  1. Participant demographics (age, sex, handedness, education, language)
  2. Group composition
  3. Recording session overview
  4. Data availability and usability per modality / task / participant
  5. QC flag frequency summary

Usage
-----
    py -3 tools/features/analyze_dataset_stats.py \
        --bids-root F:/bids_release_no_video \
        --physio-qc features/physio_qc_summary.tsv \
        --et-qc features/et_qc_summary.tsv \
        --results-dir results/dataset_stats \
        --figures-dir figures/dataset_stats \
        --dpi 180
"""
from __future__ import annotations

import argparse
import collections
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("analyze_dataset_stats")

# ── colour palette ────────────────────────────────────────────────────────────
C_MALE = "#4C8BE6"
C_FEMALE = "#E6744C"
C_GREY = "#888888"
TASK_COLORS = ["#6baed6", "#fd8d3c", "#74c476", "#9e9ac8", "#f768a1"]

EDU_ORDER = ["bachelor", "graduate_certificate", "professional_certificate",
             "ap", "master", "phd"]
EDU_LABELS = {
    "bachelor": "Bachelor",
    "graduate_certificate": "Grad. Cert.",
    "professional_certificate": "Prof. Cert.",
    "ap": "AP",
    "master": "Master",
    "phd": "PhD",
}
PROF_ORDER = ["Intermediate", "Fluent", "Native speaker"]

# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_flags(series: pd.Series) -> collections.Counter:
    counter: collections.Counter = collections.Counter()
    for val in series.dropna():
        for f in str(val).split(";"):
            f = f.strip()
            if f and f != "ok":
                counter[f] += 1
    return counter


def _save(fig: plt.Figure, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", path)


def _pct(n: int, total: int) -> str:
    return f"{n} ({100 * n / total:.0f}%)"


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1 – Demographics overview (2×3 grid)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_demographics(df: pd.DataFrame, out: Path, dpi: int) -> None:
    complete = df.dropna(subset=["age", "sex"])
    n_total = len(df)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"Participant demographics  (N = {n_total}, {len(complete)} with complete data)",
                 fontsize=13, fontweight="bold", y=0.98)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # --- 1. Age histogram by sex ---
    ax = fig.add_subplot(gs[0, 0])
    bins = np.arange(20, 65, 5)
    for sex, col in [("male", C_MALE), ("female", C_FEMALE)]:
        ages = complete.loc[complete["sex"] == sex, "age"]
        ax.hist(ages, bins=bins, color=col, alpha=0.7, edgecolor="white", label=sex.capitalize())
    ax.axvline(complete["age"].mean(), color="k", lw=1.5, ls="--", label=f"Mean {complete['age'].mean():.1f} y")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count")
    ax.set_title("Age distribution by sex")
    ax.legend(fontsize=8)

    # --- 2. Sex pie ---
    ax2 = fig.add_subplot(gs[0, 1])
    sex_counts = complete["sex"].value_counts()
    colors_pie = [C_MALE if s == "male" else C_FEMALE for s in sex_counts.index]
    wedges, texts, autotexts = ax2.pie(
        sex_counts.values,
        labels=[s.capitalize() for s in sex_counts.index],
        colors=colors_pie,
        autopct="%1.0f%%",
        startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    for t in autotexts:
        t.set_fontsize(10)
    ax2.set_title("Sex")

    # --- 3. Handedness bar ---
    ax3 = fig.add_subplot(gs[0, 2])
    hand = complete["handedness"].value_counts()
    bars = ax3.bar(hand.index, hand.values,
                   color=["#2ca02c", "#d62728", "#9467bd"][:len(hand)],
                   edgecolor="white")
    ax3.bar_label(bars, fmt="%d", padding=2, fontsize=9)
    ax3.set_ylabel("Count")
    ax3.set_title("Handedness")
    ax3.set_ylim(0, hand.values.max() * 1.2)

    # --- 4. Education stacked bar by sex ---
    ax4 = fig.add_subplot(gs[1, 0:2])
    present = [e for e in EDU_ORDER if e in complete["education"].values]
    edu_sex = pd.crosstab(complete["education"], complete["sex"]).reindex(present, fill_value=0)
    bottom_m = np.zeros(len(edu_sex))
    bottom_f = np.zeros(len(edu_sex))
    x = np.arange(len(edu_sex))
    ax4.bar(x - 0.2, edu_sex.get("male", 0), width=0.35, color=C_MALE, alpha=0.85, label="Male")
    ax4.bar(x + 0.2, edu_sex.get("female", 0), width=0.35, color=C_FEMALE, alpha=0.85, label="Female")
    ax4.set_xticks(x)
    ax4.set_xticklabels([EDU_LABELS.get(e, e) for e in edu_sex.index], rotation=25, ha="right", fontsize=8)
    ax4.set_ylabel("Count")
    ax4.set_title("Education level by sex")
    ax4.legend(fontsize=8)
    ax4.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # --- 5. English proficiency ---
    ax5 = fig.add_subplot(gs[1, 2])
    prof = complete["english_proficiency"].value_counts().reindex(PROF_ORDER, fill_value=0)
    colors_prof = ["#fdae6b", "#fd8d3c", "#d94801"]
    bars5 = ax5.barh(prof.index, prof.values, color=colors_prof, edgecolor="white")
    ax5.bar_label(bars5, fmt="%d", padding=3, fontsize=9)
    ax5.set_xlabel("Count")
    ax5.set_title("English proficiency")
    ax5.set_xlim(0, prof.values.max() * 1.3)
    ax5.invert_yaxis()

    _save(fig, out, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2 – Group composition
# ═══════════════════════════════════════════════════════════════════════════════

def fig_group_composition(df: pd.DataFrame, out: Path, dpi: int) -> None:
    complete = df.dropna(subset=["age", "sex"])
    groups = sorted(df["group_id"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Group composition", fontsize=13, fontweight="bold")

    # --- Age box plot per group ---
    ax = axes[0]
    data = [complete.loc[complete["group_id"] == g, "age"].values for g in groups]
    bp = ax.boxplot(data, patch_artist=True, medianprops={"color": "k", "lw": 2})
    for patch in bp["boxes"]:
        patch.set_facecolor("#6baed6")
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([g.replace("grp-", "G") for g in groups], fontsize=8)
    ax.set_ylabel("Age (years)")
    ax.set_title("Age by group")

    # --- Sex composition stacked bar ---
    ax2 = axes[1]
    sex_grp = complete.groupby(["group_id", "sex"]).size().unstack(fill_value=0)
    male_vals = sex_grp.get("male", pd.Series(0, index=sex_grp.index)).values
    female_vals = sex_grp.get("female", pd.Series(0, index=sex_grp.index)).values
    x = np.arange(len(groups))
    ax2.bar(x, male_vals, color=C_MALE, label="Male", alpha=0.85)
    ax2.bar(x, female_vals, bottom=male_vals, color=C_FEMALE, label="Female", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels([g.replace("grp-", "G") for g in groups], fontsize=8)
    ax2.set_ylabel("Count")
    ax2.set_title("Sex composition by group")
    ax2.legend(fontsize=8)
    ax2.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # --- Education level per group (scatter/strip) ---
    ax3 = axes[2]
    edu_num = {e: i for i, e in enumerate(EDU_ORDER)}
    complete = complete.copy()
    complete["edu_num"] = complete["education"].map(edu_num)
    for gi, g in enumerate(groups):
        sub = complete[complete["group_id"] == g]
        jitter = np.random.default_rng(gi).uniform(-0.15, 0.15, len(sub))
        colors = [C_MALE if s == "male" else C_FEMALE for s in sub["sex"]]
        ax3.scatter(sub["edu_num"] + jitter, [gi] * len(sub), c=colors, s=40, alpha=0.8)
    ax3.set_yticks(range(len(groups)))
    ax3.set_yticklabels([g.replace("grp-", "G") for g in groups], fontsize=8)
    ax3.set_xticks(range(len(EDU_ORDER)))
    ax3.set_xticklabels([EDU_LABELS.get(e, e) for e in EDU_ORDER], rotation=30, ha="right", fontsize=7)
    ax3.set_title("Education level per group")
    m_patch = mpatches.Patch(color=C_MALE, label="Male")
    f_patch = mpatches.Patch(color=C_FEMALE, label="Female")
    ax3.legend(handles=[m_patch, f_patch], fontsize=8)

    _save(fig, out, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3 – Recording session timeline and duration distribution
# ═══════════════════════════════════════════════════════════════════════════════

def fig_session_overview(physio_df: pd.DataFrame, et_df: pd.DataFrame, out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Recording session overview", fontsize=13, fontweight="bold")

    # --- Recordings per date ---
    ax = axes[0]
    physio_df = physio_df.copy()
    physio_df["date"] = physio_df["session_id"].str.extract(r"(\d{8})")
    date_counts = physio_df[physio_df["physio_available"]].groupby("date").size()
    # keep unique sessions
    sess_dates = physio_df[physio_df["physio_available"]].drop_duplicates("session_id")
    sess_per_day = sess_dates.groupby("date").size()
    ax.bar(range(len(sess_per_day)), sess_per_day.values, color="#6baed6", edgecolor="white")
    ax.set_xticks(range(len(sess_per_day)))
    ax.set_xticklabels(
        [d[4:6] + "/" + d[6:8] + "\n" + d[:4] for d in sess_per_day.index],
        fontsize=8
    )
    ax.set_ylabel("Sessions recorded")
    ax.set_title("Data collection timeline")
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    # --- Task duration distributions (physio) ---
    ax2 = axes[1]
    tasks = sorted(physio_df["task_id"].unique())
    parts = physio_df[physio_df["physio_available"]].groupby("task_id")["duration_s"]
    data = [parts.get_group(t).values / 60 for t in tasks if t in parts.groups]
    bp = ax2.boxplot(data, patch_artist=True, medianprops={"color": "k", "lw": 2})
    for patch, col in zip(bp["boxes"], TASK_COLORS):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    ax2.set_xticks(range(1, len(tasks) + 1))
    ax2.set_xticklabels(tasks)
    ax2.set_ylabel("Duration (min)")
    ax2.set_title("Task duration — physio")

    # --- Data availability heatmap (participant × task) ---
    ax3 = axes[2]
    physio_piv = physio_df.pivot_table(
        index="participant_id", columns="task_id", values="physio_available", aggfunc="max"
    ).reindex(columns=tasks)
    et_piv = et_df.pivot_table(
        index="participant_id", columns="task_id", values="et_available", aggfunc="max"
    ).reindex(columns=tasks)

    # combine: 0=missing both, 1=physio only, 2=ET only, 3=both
    combined = physio_piv.fillna(False).astype(int) + 2 * et_piv.fillna(False).astype(int)
    cmap = matplotlib.colors.ListedColormap(["#d9d9d9", "#6baed6", "#fd8d3c", "#31a354"])
    im = ax3.imshow(combined.values, aspect="auto", cmap=cmap, vmin=0, vmax=3, interpolation="nearest")
    ax3.set_xticks(range(len(tasks)))
    ax3.set_xticklabels(tasks, fontsize=8)
    ax3.set_yticks(range(len(combined)))
    ax3.set_yticklabels(combined.index, fontsize=6)
    ax3.set_title("Data availability\n(participant × task)")
    handles = [
        mpatches.Patch(color="#d9d9d9", label="Missing"),
        mpatches.Patch(color="#6baed6", label="Physio only"),
        mpatches.Patch(color="#fd8d3c", label="ET only"),
        mpatches.Patch(color="#31a354", label="Both"),
    ]
    ax3.legend(handles=handles, fontsize=6, loc="upper right", bbox_to_anchor=(1.35, 1))

    _save(fig, out, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4 – Modality usability rates
# ═══════════════════════════════════════════════════════════════════════════════

def fig_usability(physio_qc: pd.DataFrame, et_qc: pd.DataFrame, out: Path, dpi: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Signal usability rates by task", fontsize=13, fontweight="bold")

    tasks = sorted(physio_qc["task_id"].unique())
    x = np.arange(len(tasks))
    width = 0.25

    # --- Physio ---
    ax = axes[0]
    modalities = {"PPG": "ppg_usable", "EDA": "eda_usable", "Temp": "temp_usable", "IMU": "imu_usable"}
    colors_p = ["#6baed6", "#fd8d3c", "#74c476", "#9e9ac8"]
    for i, (label, col) in enumerate(modalities.items()):
        if col not in physio_qc.columns:
            continue
        rates = physio_qc.groupby("task_id")[col].mean().reindex(tasks, fill_value=0) * 100
        offset = (i - 1.5) * width
        ax.bar(x + offset, rates.values, width=width, label=label,
               color=colors_p[i], alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Usable recordings (%)")
    ax.set_ylim(0, 110)
    ax.axhline(80, color="k", lw=0.8, ls="--", alpha=0.4)
    ax.set_title("Physio signal usability")
    ax.legend(fontsize=8)

    # --- Eye-tracking ---
    ax2 = axes[1]
    et_tasks = sorted(et_qc["task_id"].unique())
    x2 = np.arange(len(et_tasks))
    for i, (label, col, col2) in enumerate([
        ("Gaze-usable", "gaze_usable", "#6baed6"),
        ("Pupil-usable", "pupil_usable", "#fd8d3c"),
    ]):
        rates = et_qc.groupby("task_id")[label.lower().replace("-", "_").replace("usable", "usable")].mean()
        # recompute correctly
        rates = et_qc.groupby("task_id")[col].mean().reindex(et_tasks, fill_value=0) * 100
        offset = (i - 0.5) * 0.35
        ax2.bar(x2 + offset, rates.values, width=0.35, label=label,
                color=col2, alpha=0.85, edgecolor="white")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(et_tasks)
    ax2.set_ylabel("Usable recordings (%)")
    ax2.set_ylim(0, 110)
    ax2.axhline(80, color="k", lw=0.8, ls="--", alpha=0.4)
    ax2.set_title("Eye-tracking usability")
    ax2.legend(fontsize=8)

    _save(fig, out, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 5 – QC flag catalogue
# ═══════════════════════════════════════════════════════════════════════════════

def fig_qc_flags(physio_qc: pd.DataFrame, et_qc: pd.DataFrame, out: Path, dpi: int) -> None:
    physio_flags = _parse_flags(physio_qc["qc_flag"])
    et_flags = _parse_flags(et_qc["qc_flag"])

    n_physio = len(physio_qc)
    n_et = len(et_qc)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle("QC flag frequency", fontsize=13, fontweight="bold")

    for ax, flags, n_total, title in [
        (axes[0], physio_flags, n_physio, "Physio QC flags"),
        (axes[1], et_flags, n_et, "Eye-tracking QC flags"),
    ]:
        if not flags:
            ax.text(0.5, 0.5, "No flags", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            continue
        sorted_flags = sorted(flags.items(), key=lambda x: x[1], reverse=True)
        labels, counts = zip(*sorted_flags)
        pcts = [100 * c / n_total for c in counts]
        y = np.arange(len(labels))
        bars = ax.barh(y, pcts, color="#6baed6", edgecolor="white", alpha=0.85)
        for bar, cnt in zip(bars, counts):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                    f"{cnt} ({100*cnt/n_total:.0f}%)", va="center", fontsize=8)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("% of participant-task rows")
        ax.set_title(f"{title}  (n = {n_total} rows)")
        ax.set_xlim(0, max(pcts) * 1.35)
        ax.invert_yaxis()

    _save(fig, out, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 6 – Coverage heatmap (participant × task, both modalities)
# ═══════════════════════════════════════════════════════════════════════════════

def fig_coverage_heatmap(physio_qc: pd.DataFrame, et_qc: pd.DataFrame, out: Path, dpi: int) -> None:
    tasks = sorted(set(physio_qc["task_id"].unique()) | set(et_qc["task_id"].unique()))

    def _usability_matrix(qc: pd.DataFrame, col: str, default_col: str | None = None) -> pd.DataFrame:
        key = col if col in qc.columns else default_col
        if key is None:
            return pd.DataFrame()
        piv = qc.pivot_table(index="participant_id", columns="task_id", values=key, aggfunc="max")
        return piv.reindex(columns=[t for t in tasks if t in piv.columns]).fillna(False).astype(int)

    ppg = _usability_matrix(physio_qc, "ppg_usable")
    eda = _usability_matrix(physio_qc, "eda_usable")
    gaze = _usability_matrix(et_qc, "gaze_usable")
    pupil = _usability_matrix(et_qc, "pupil_usable")

    participants = sorted(
        set(ppg.index) | set(eda.index) | set(gaze.index) | set(pupil.index)
    )

    fig, axes = plt.subplots(1, 4, figsize=(18, 8), sharey=True)
    fig.suptitle("Usability per participant and task", fontsize=13, fontweight="bold")

    for ax, mat, title, col in [
        (axes[0], ppg, "PPG usable", "#6baed6"),
        (axes[1], eda, "EDA usable", "#fd8d3c"),
        (axes[2], gaze, "Gaze usable", "#74c476"),
        (axes[3], pupil, "Pupil usable", "#9e9ac8"),
    ]:
        if mat.empty:
            ax.set_title(title)
            continue
        mat_full = mat.reindex(index=participants, fill_value=0)
        cmap_bin = matplotlib.colors.ListedColormap(["#f0f0f0", col])
        ax.imshow(mat_full.values, aspect="auto", cmap=cmap_bin, vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_xticks(range(len(mat_full.columns)))
        ax.set_xticklabels(list(mat_full.columns), fontsize=8)
        ax.set_title(f"{title}\n({int(mat_full.values.sum())}/{mat_full.size})")
        if ax == axes[0]:
            ax.set_yticks(range(len(participants)))
            ax.set_yticklabels(participants, fontsize=6)

    _save(fig, out, dpi)


# ═══════════════════════════════════════════════════════════════════════════════
# Summary table writers
# ═══════════════════════════════════════════════════════════════════════════════

def write_summary_tables(df: pd.DataFrame, physio_qc: pd.DataFrame, et_qc: pd.DataFrame,
                          results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    complete = df.dropna(subset=["age"])

    # Demographic summary
    rows = []
    rows.append({"variable": "N total", "value": str(len(df))})
    rows.append({"variable": "N with complete demographics", "value": str(len(complete))})
    rows.append({"variable": "Age mean (SD)", "value": f"{complete['age'].mean():.1f} ({complete['age'].std():.1f})"})
    rows.append({"variable": "Age range", "value": f"{complete['age'].min():.0f} – {complete['age'].max():.0f}"})
    for sex in ["male", "female"]:
        n = (complete["sex"] == sex).sum()
        rows.append({"variable": f"Sex: {sex}", "value": _pct(n, len(complete))})
    for hand in ["right", "left", "ambidextrous"]:
        n = (complete["handedness"] == hand).sum()
        rows.append({"variable": f"Handedness: {hand}", "value": _pct(n, len(complete))})
    for edu in EDU_ORDER:
        n = (complete["education"] == edu).sum()
        if n:
            rows.append({"variable": f"Education: {EDU_LABELS.get(edu, edu)}", "value": _pct(n, len(complete))})
    for prof in PROF_ORDER:
        n = (complete["english_proficiency"] == prof).sum()
        rows.append({"variable": f"English: {prof}", "value": _pct(n, len(complete))})

    demo_path = results_dir / "demographic_summary.tsv"
    pd.DataFrame(rows).to_csv(demo_path, sep="\t", index=False)
    log.info("Wrote %s", demo_path)

    # Usability summary per task
    usability_rows = []
    tasks = sorted(set(physio_qc["task_id"].unique()) | set(et_qc["task_id"].unique()))
    for t in tasks:
        row = {"task_id": t}
        p = physio_qc[physio_qc["task_id"] == t]
        e = et_qc[et_qc["task_id"] == t]
        row["physio_n"] = len(p)
        row["physio_available"] = p["physio_available"].sum() if "physio_available" in p else 0
        for col in ["ppg_usable", "eda_usable", "temp_usable", "imu_usable"]:
            if col in p.columns:
                row[col] = int(p[col].sum())
        row["et_n"] = len(e)
        row["et_available"] = e["et_available"].sum() if "et_available" in e else 0
        for col in ["gaze_usable", "pupil_usable"]:
            if col in e.columns:
                row[col] = int(e[col].sum())
        usability_rows.append(row)

    usab_path = results_dir / "usability_by_task.tsv"
    pd.DataFrame(usability_rows).to_csv(usab_path, sep="\t", index=False)
    log.info("Wrote %s", usab_path)

    # QC flag frequencies
    for label, qc in [("physio", physio_qc), ("et", et_qc)]:
        flags = _parse_flags(qc["qc_flag"])
        n_total = len(qc)
        flag_df = pd.DataFrame(
            [{"flag": f, "count": c, "pct": round(100 * c / n_total, 1)}
             for f, c in sorted(flags.items(), key=lambda x: -x[1])]
        )
        flag_path = results_dir / f"{label}_flag_frequencies.tsv"
        flag_df.to_csv(flag_path, sep="\t", index=False)
        log.info("Wrote %s", flag_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bids-root", default="F:/bids_release_no_video")
    ap.add_argument("--physio-qc", default="features/physio_qc_summary.tsv")
    ap.add_argument("--physio-task", default="features/physio_participant_task.tsv")
    ap.add_argument("--et-qc", default="features/et_qc_summary.tsv")
    ap.add_argument("--et-task", default="features/et_participant_task.tsv")
    ap.add_argument("--results-dir", default="results/dataset_stats")
    ap.add_argument("--figures-dir", default="figures/dataset_stats")
    ap.add_argument("--dpi", type=int, default=180)
    args = ap.parse_args()

    bids_root = Path(args.bids_root)
    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)
    dpi = args.dpi

    # Load data
    participants = pd.read_csv(bids_root / "participants.tsv", sep="\t")
    physio_qc = pd.read_csv(args.physio_qc, sep="\t")
    physio_task = pd.read_csv(args.physio_task, sep="\t")
    et_qc = pd.read_csv(args.et_qc, sep="\t")
    et_task = pd.read_csv(args.et_task, sep="\t")

    complete = participants.dropna(subset=["age"])
    log.info("Participants: %d total, %d with complete demographics", len(participants), len(complete))
    log.info("Age: mean=%.1f SD=%.1f range=[%.0f, %.0f]",
             complete["age"].mean(), complete["age"].std(),
             complete["age"].min(), complete["age"].max())
    log.info("Sex: %s", complete["sex"].value_counts().to_dict())

    # Write tables
    write_summary_tables(participants, physio_qc, et_qc, results_dir)

    # Figures
    fig_demographics(participants, figures_dir / "demographics_overview.png", dpi)
    fig_group_composition(participants, figures_dir / "group_composition.png", dpi)
    fig_session_overview(physio_task, et_task, figures_dir / "session_overview.png", dpi)
    fig_usability(physio_qc, et_qc, figures_dir / "usability_by_task.png", dpi)
    fig_qc_flags(physio_qc, et_qc, figures_dir / "qc_flag_frequencies.png", dpi)
    fig_coverage_heatmap(physio_qc, et_qc, figures_dir / "coverage_heatmap.png", dpi)

    log.info("Done. Figures → %s | Tables → %s", figures_dir, results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
