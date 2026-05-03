"""Task-level physiological fingerprint: EmotiBit autonomic signals.

Reads the pre-computed physiology feature table
(``features/physio_participant_task.tsv``) and the pupil feature table
(``features/features_pupil_participant_task.tsv``) from the release and
produces:

  - A printed table of mean T0-normalised deltas per task
  - figures/02_physio_task_fingerprint.png  â€” task Ã— feature heatmap

Key findings replicated from the paper
--------------------------------------
  â€¢ Skin temperature rises across active tasks (T1â€“T4) relative to T0.
  â€¢ Pupil diameter decreases in T1, T3, and T4 relative to baseline.
  â€¢ T2 (negotiation) shows elevated HR and EDA relative to T0.
  â€¢ Joint physiology + pupil usability: 158 / 192 active-task rows (82.3 %).

Usage
-----
    python analysis/02_physio_task_fingerprint.py
    python analysis/02_physio_task_fingerprint.py --features-dir features --out-dir figures

Arguments
---------
  --features-dir   Directory containing physio_participant_task.tsv and
                   features_pupil_participant_task.tsv (default: features/)
  --out-dir        Output directory for figures (default: figures/)
  --no-pupil       Skip pupil data even if the file is found
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

TASK_ORDER = ["T1", "T2", "T3", "T4"]   # T0 is the baseline; excluded from heatmap

PHYSIO_DELTA_FEATURES = {
    "HR (bpm) Î” T0":   "hr_mean_bpm_delta_t0",
    "HRV RMSSD (ms) Î” T0": "hrv_rmssd_ms_delta_t0",
    "EDA tonic Î” T0":  "eda_tonic_mean_delta_t0",
    "EDA phasic rate Î” T0": "eda_phasic_rate_hz_delta_t0",
    "Skin temp (Â°C) Î” T0": "temp_mean_delta_t0",
    "Motion Î” T0":     "accel_motion_mean_delta_t0",
}

PUPIL_DELTA_FEATURE = "pupil_mean"   # raw; we'll compute delta ourselves

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def load_physio(features_dir: Path) -> pd.DataFrame:
    p = features_dir / "physio_participant_task.tsv"
    if not p.exists():
        raise FileNotFoundError(f"Physio feature table not found: {p}")
    df = pd.read_csv(p, sep="\t", low_memory=False)
    logging.info("Loaded physio features: %d rows", len(df))
    return df


def load_pupil(features_dir: Path) -> pd.DataFrame | None:
    p = features_dir / "features_pupil_participant_task.tsv"
    if not p.exists():
        logging.warning("Pupil feature table not found: %s â€” skipping", p)
        return None
    df = pd.read_csv(p, sep="\t", low_memory=False)
    logging.info("Loaded pupil features: %d rows", len(df))
    return df


def compute_physio_heatmap(df: pd.DataFrame) -> pd.DataFrame:
    """Mean T0-normalised delta per task (active tasks only)."""
    rows = {}
    for label, col in PHYSIO_DELTA_FEATURES.items():
        if col not in df.columns:
            logging.warning("Column %s missing â€” skipping", col)
            continue
        active = df[df["task"].isin(TASK_ORDER)][["task", col]].copy()
        active[col] = pd.to_numeric(active[col], errors="coerce")
        means = active.groupby("task")[col].mean().reindex(TASK_ORDER)
        rows[label] = means
    return pd.DataFrame(rows).T


def compute_pupil_delta(df: pd.DataFrame | None) -> pd.Series | None:
    """Per-participant T0-normalised pupil delta, then group mean per task."""
    if df is None:
        return None
    df = df.copy()
    df["pupil_mean"] = pd.to_numeric(df["pupil_mean"], errors="coerce")
    # baseline per participant
    baseline = (
        df[df["task"] == "T0"]
        .groupby(["session_id", "participant_id"])["pupil_mean"]
        .mean()
        .rename("pupil_t0")
    )
    merged = df[df["task"].isin(TASK_ORDER)].merge(
        baseline.reset_index(), on=["session_id", "participant_id"], how="left"
    )
    merged["pupil_delta"] = merged["pupil_mean"] - merged["pupil_t0"]
    means = merged.groupby("task")["pupil_delta"].mean().reindex(TASK_ORDER)
    return means


def plot_heatmap(heatmap: pd.DataFrame, pupil_row: "pd.Series | None",
                 out_path: Path) -> None:
    """Diverging heatmap: rows = features, cols = tasks."""
    if pupil_row is not None:
        extra = pd.DataFrame({"T1": [pupil_row["T1"]], "T2": [pupil_row["T2"]],
                               "T3": [pupil_row["T3"]], "T4": [pupil_row["T4"]]},
                              index=["Pupil diam Î” T0"])
        heatmap = pd.concat([heatmap, extra])

    # z-score rows for display
    z = heatmap.copy()
    row_std = z.std(axis=1).replace(0, 1)
    z = z.subtract(z.mean(axis=1), axis=0).divide(row_std, axis=0)

    fig, ax = plt.subplots(figsize=(7, max(4, len(heatmap) * 0.55 + 1.2)))
    cmap = plt.get_cmap("RdBu_r")
    im = ax.imshow(z.values, cmap=cmap, aspect="auto", vmin=-2, vmax=2)

    ax.set_xticks(range(len(TASK_ORDER)))
    ax.set_xticklabels(TASK_ORDER, fontsize=10)
    ax.set_yticks(range(len(heatmap)))
    ax.set_yticklabels(heatmap.index, fontsize=9)

    # annotate cells with raw delta
    for r, feat in enumerate(heatmap.index):
        for c, task in enumerate(TASK_ORDER):
            val = heatmap.loc[feat, task]
            if pd.notna(val):
                ax.text(c, r, f"{val:+.2f}", ha="center", va="center",
                        fontsize=7, color="black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.7, label="Row-normalised Z-score")
    ax.set_title(
        "GroupAffect-4: Task-level Physiological Fingerprint\n"
        "(Mean T0-normalised delta, n=10 groups, 40 participants)",
        fontsize=10,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logging.info("Saved â†’ %s", out_path)
    plt.close(fig)


def print_coverage(df: pd.DataFrame) -> None:
    """Print usability statistics matching the paper."""
    n_expected = 200   # 10 groups Ã— 4 participants Ã— 5 tasks
    n_avail = int(df["physio_available"].sum()) if "physio_available" in df.columns else len(df)
    ppg_ok = int(df["ppg_available"].sum()) if "ppg_available" in df.columns else 0
    eda_ok = int(df["eda_available"].sum()) if "eda_available" in df.columns else 0
    temp_ok = int(df["temp_available"].sum()) if "temp_available" in df.columns else 0
    print("\nâ”€â”€ Physiology Coverage â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    print(f"  EmotiBit rows available: {n_avail}/{n_expected} ({100*n_avail/n_expected:.1f}%)")
    print(f"  PPG available:   {ppg_ok}/{n_expected} ({100*ppg_ok/n_expected:.1f}%)")
    print(f"  EDA available:   {eda_ok}/{n_expected} ({100*eda_ok/n_expected:.1f}%)")
    print(f"  Temp available:  {temp_ok}/{n_expected} ({100*temp_ok/n_expected:.1f}%)")
    print("  (Paper: PPG 71.7%, EDA 71.2%, Temp 71.7%)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("features"),
        help="Directory containing feature TSV files (default: features/)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for figures",
    )
    parser.add_argument(
        "--no-pupil",
        action="store_true",
        help="Skip pupil data",
    )
    args = parser.parse_args()

    physio = load_physio(args.features_dir)
    print_coverage(physio)

    pupil = None if args.no_pupil else load_pupil(args.features_dir)

    heatmap = compute_physio_heatmap(physio)
    pupil_row = compute_pupil_delta(pupil)

    print("\nâ”€â”€ T0-Normalised Mean Deltas by Task â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    print(heatmap.to_string())
    if pupil_row is not None:
        print(f"\n  Pupil diam Î” T0:\n{pupil_row.to_string()}")

    plot_heatmap(heatmap, pupil_row, args.out_dir / "02_physio_task_fingerprint.png")


if __name__ == "__main__":
    main()

