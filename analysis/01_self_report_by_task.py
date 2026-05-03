"""Self-report valence, arousal, and dominance by task.

Reads all per-session behavioural stimuli-answer TSV files from the BIDS
release and produces:
  - A printed summary table (mean Â± SD for each task)
  - figures/01_self_report_by_task.png  â€” bar chart with error bars

Key finding replicated from the paper
--------------------------------------
  T2 (Mini-Negotiation) produces the lowest mean valence (â‰ˆ 5.7 on a 9-pt
  scale), while T0 (baseline) is the highest (â‰ˆ 7.4).

Usage
-----
    cd <bids_root>
    python ../../analysis/01_self_report_by_task.py
    # or with an explicit path:
    python analysis/01_self_report_by_task.py --bids-root data/bids_release_no_video

Arguments
---------
  --bids-root   Path to the BIDS release root (default: current directory)
  --out-dir     Output directory for figures (default: figures/)
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

TASK_ORDER = ["T0", "T1", "T2", "T3", "T4"]
TASK_LABELS = {
    "T0": "T0\nBaseline",
    "T1": "T1\nHidden-Profile",
    "T2": "T2\nNegotiation",
    "T3": "T3\nIdea Generation",
    "T4": "T4\nPublic-Goods",
}
MEASURES = ["valence", "arousal", "dominance"]
MEASURE_COLORS = {"valence": "#2196F3", "arousal": "#F44336", "dominance": "#4CAF50"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def load_stimuli_answers(bids_root: Path) -> pd.DataFrame:
    """Concatenate all *_stimuli_answers.tsv files across sessions."""
    files = sorted(bids_root.glob("sub-01/ses-*/beh/*_stimuli_answers.tsv"))
    if not files:
        logging.error("No stimuli_answers.tsv files found under %s", bids_root)
        sys.exit(1)
    logging.info("Loading %d stimuli-answer filesâ€¦", len(files))
    frames = []
    for f in files:
        df = pd.read_csv(f, sep="\t", low_memory=False)
        # attach session id from path
        df["session_id"] = f.parent.parent.name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def extract_vad(df: pd.DataFrame) -> pd.DataFrame:
    """Return participant-task VAD rows with numeric item_value."""
    vad = df[
        (df["response_type"] == "vad")
        & df["task"].isin(TASK_ORDER)
        & df["item_key"].isin(MEASURES)
    ].copy()
    vad["value"] = pd.to_numeric(vad["item_value"], errors="coerce")
    vad = vad.dropna(subset=["value"])
    # anonymise: keep only P1-P4 seat IDs
    vad = vad[vad["participant"].isin(["P1", "P2", "P3", "P4"])]
    return vad


def summary_table(vad: pd.DataFrame) -> pd.DataFrame:
    """Compute mean Â± SD for each task Ã— measure."""
    tbl = (
        vad.groupby(["task", "item_key"])["value"]
        .agg(["mean", "std", "count"])
        .round(2)
        .reset_index()
    )
    tbl.columns = ["task", "measure", "mean", "sd", "n"]
    return tbl.sort_values(["measure", "task"])


def plot_vad(vad: pd.DataFrame, out_path: Path) -> None:
    """Bar chart: valence, arousal, dominance by task."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=False)
    for ax, measure in zip(axes, MEASURES):
        sub = vad[vad["item_key"] == measure]
        stats = (
            sub.groupby("task")["value"]
            .agg(["mean", "sem"])
            .reindex(TASK_ORDER)
        )
        x = np.arange(len(TASK_ORDER))
        ax.bar(
            x,
            stats["mean"],
            yerr=stats["sem"],
            color=MEASURE_COLORS[measure],
            alpha=0.8,
            capsize=4,
            width=0.6,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [TASK_LABELS[t] for t in TASK_ORDER], fontsize=8
        )
        ax.set_title(measure.capitalize(), fontsize=11)
        ax.set_ylabel("Rating (1â€“9 SAM scale)")
        ax.set_ylim(1, 9)
        ax.axhline(5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "GroupAffect-4: Self-Report (VAD) by Task â€” Leave-One-Group-Out mean Â± SEM",
        fontsize=12,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logging.info("Saved â†’ %s", out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--bids-root",
        type=Path,
        default=Path("."),
        help="Path to BIDS release root (default: current directory)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("figures"),
        help="Output directory for figures",
    )
    args = parser.parse_args()

    bids_root = args.bids_root.resolve()
    if not (bids_root / "sub-01").exists():
        parser.error(f"sub-01/ not found under {bids_root}. Pass --bids-root <path>.")

    df = load_stimuli_answers(bids_root)
    vad = extract_vad(df)

    logging.info("Total VAD responses: %d", len(vad))

    tbl = summary_table(vad)
    print("\nâ”€â”€ Self-Report Summary (mean Â± SD per task) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    for measure in MEASURES:
        print(f"\n  {measure.upper()}")
        sub = tbl[tbl["measure"] == measure][["task", "mean", "sd", "n"]]
        print(sub.to_string(index=False))

    # highlight the T2 valence finding
    t2_val = tbl[(tbl["task"] == "T2") & (tbl["measure"] == "valence")]["mean"].values
    if t2_val.size:
        print(f"\n  â–¶ T2 mean valence = {t2_val[0]:.2f} (paper: 5.72)")

    plot_vad(vad, args.out_dir / "01_self_report_by_task.png")


if __name__ == "__main__":
    main()

