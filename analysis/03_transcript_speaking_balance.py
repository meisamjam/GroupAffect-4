"""Speaking-time balance (Gini coefficient) by task from transcripts.

Reads all per-session transcript TSV files from the BIDS release audio/
folders and produces:

  - Printed per-task mean Gini coefficient of speaking-time balance
    (0 = perfectly equal; 1 = one person speaks all the time)
  - figures/03_speaking_balance.png  â€” box + jitter plot per task

Key finding replicated from the paper
--------------------------------------
  Speaking balance (Gini coefficient) differs across task types.
  T0 baseline tends toward more equal participation; T4 (public-goods)
  is often dominated by one speaker.

Usage
-----
    python analysis/03_transcript_speaking_balance.py
    python analysis/03_transcript_speaking_balance.py --bids-root data/bids_release_no_video

Arguments
---------
  --bids-root   Path to the BIDS release root (default: current directory)
  --out-dir     Output directory for figures (default: figures/)
  --strict      Use strict transcripts only (exclude backchannels).
                Default: use *_transcript.tsv (strict) files.
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
    "T3": "T3\nIdea Gen.",
    "T4": "T4\nPublic-Goods",
}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative array (speaking durations)."""
    x = np.array(x, dtype=float)
    x = x[x > 0]
    if len(x) == 0:
        return np.nan
    x = np.sort(x)
    n = len(x)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum() - (n + 1) * x.sum()) / (n * x.sum()))


def load_transcripts(bids_root: Path, strict: bool = True) -> pd.DataFrame:
    """Load all segment-level transcript TSV files."""
    # Strict = *_transcript.tsv (no backchannel); other variant = *_desc-withBackchannels_transcript.tsv
    pattern = "sub-01/ses-*/audio/*_transcript.tsv"
    all_files = sorted(bids_root.glob(pattern))
    # exclude withBackchannels files
    strict_files = [f for f in all_files if "_desc-withBackchannels" not in f.name]

    if not strict_files:
        logging.error("No transcript TSV files found under %s", bids_root)
        sys.exit(1)
    logging.info("Loading %d transcript filesâ€¦", len(strict_files))

    frames = []
    for f in strict_files:
        # derive session_id and task from filename
        # e.g. sub-01_ses-20260312_grp-07_run01_task-T1_run-01_transcript.tsv
        stem = f.stem
        parts = {kv.split("-")[0]: kv.split("-", 1)[1]
                 for kv in stem.split("_") if "-" in kv}
        ses_id = parts.get("ses", "unknown")
        task_id = parts.get("task", "unknown")
        if task_id not in TASK_ORDER:
            continue
        try:
            df = pd.read_csv(f, sep="\t", low_memory=False)
        except Exception as exc:
            logging.warning("Could not read %s: %s", f, exc)
            continue
        df["session_id"] = ses_id
        df["task_id"] = task_id
        frames.append(df)

    if not frames:
        logging.error("No valid transcript rows loaded.")
        sys.exit(1)
    return pd.concat(frames, ignore_index=True)


def compute_gini_per_group_task(df: pd.DataFrame) -> pd.DataFrame:
    """Gini coefficient of speaking-time per group Ã— task."""
    # filter to P1-P4 only (exclude MODERATOR)
    df = df[df["speaker"].isin(["P1", "P2", "P3", "P4"])].copy()
    df["duration"] = pd.to_numeric(df["duration"], errors="coerce").fillna(0)

    records = []
    for (ses_id, task_id), grp in df.groupby(["session_id", "task_id"]):
        speak_time = grp.groupby("speaker")["duration"].sum()
        g = gini(speak_time.values)
        records.append({"session_id": ses_id, "task": task_id, "gini": g})
    return pd.DataFrame(records)


def plot_gini(gini_df: pd.DataFrame, out_path: Path) -> None:
    """Box + jitter plot of Gini coefficient by task."""
    fig, ax = plt.subplots(figsize=(8, 4))
    tasks = [t for t in TASK_ORDER if t in gini_df["task"].values]
    data = [gini_df[gini_df["task"] == t]["gini"].dropna().values for t in tasks]

    bp = ax.boxplot(data, positions=range(len(tasks)), widths=0.5,
                    patch_artist=True, medianprops=dict(color="black", linewidth=2))
    colors = plt.cm.Set2(np.linspace(0, 1, len(tasks)))  # type: ignore[attr-defined]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # jitter
    rng = np.random.default_rng(42)
    for i, vals in enumerate(data):
        jitter = rng.uniform(-0.15, 0.15, size=len(vals))
        ax.scatter(i + jitter, vals, color="black", alpha=0.6, s=25, zorder=3)

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels([TASK_LABELS[t] for t in tasks], fontsize=9)
    ax.set_ylabel("Gini coefficient (speaking time)")
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_title(
        "GroupAffect-4: Speaking-Time Balance (Gini) by Task\n"
        "(0 = equal participation; 1 = monopolised by one speaker)",
        fontsize=10,
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    logging.info("Saved â†’ %s", out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        help="Use strict transcripts (default: True)",
    )
    args = parser.parse_args()

    bids_root = args.bids_root.resolve()
    if not (bids_root / "sub-01").exists():
        parser.error(f"sub-01/ not found under {bids_root}. Pass --bids-root <path>.")

    df = load_transcripts(bids_root, strict=args.strict)
    gini_df = compute_gini_per_group_task(df)

    print("\nâ”€â”€ Speaking-Balance (Gini) per Task â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€")
    tbl = (
        gini_df.groupby("task")["gini"]
        .agg(["mean", "std", "count"])
        .round(3)
        .reindex(TASK_ORDER)
    )
    tbl.columns = ["mean_gini", "sd", "n_groups"]
    print(tbl.to_string())

    plot_gini(gini_df, args.out_dir / "03_speaking_balance.png")


if __name__ == "__main__":
    main()

