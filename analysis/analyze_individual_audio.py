"""Participant-level audio/prosody analysis for GroupAffect-4 task recordings.

Consumes ``audio_participant_task.tsv`` from ``extract_audio_features.py`` and writes
individual speech participation summaries, baseline-normalized task deltas, task
contrast tests, and PNG figures.

Privacy: participant IDs are anonymized P1-P4/session-seat IDs only; transcript
text and source audio paths are never read into the analysis outputs.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import re
import warnings
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("analyze_individual_audio")

TASK_ORDER = ["T0", "T1", "T2", "T3", "T4"]
ACTIVE_TASKS = ["T1", "T2", "T3", "T4"]
PARTICIPANT_RE = re.compile(r"(P[1-4])", re.IGNORECASE)

AUDIO_FEATURES = [
    "speaking_time_s",
    "speaking_fraction",
    "pause_count",
    "turn_count",
    "overlap_fraction",
    "uncertain_fraction",
    "energy_mean",
    "energy_sd",
    "pitch_mean",
    "pitch_sd",
    "hnr_mean",
    "jitter_mean",
    "shimmer_mean",
    "voiced_segments_per_sec",
    "mean_voiced_segment_s",
    "mean_unvoiced_segment_s",
    "speech_rate_proxy",
    "mean_turn_duration_s_transcript",
    "backchannel_rate_transcript",
    "overlap_fraction_transcript",
    "response_gap_mean_s_transcript",
    "interruption_rate_transcript",
]

DEFAULT_CONTRAST_FEATURES = [
    "speaking_fraction",
    "speaking_share_group",
    "turn_share_group",
    "pause_count",
    "turn_count",
    "overlap_fraction",
    "energy_mean",
    "pitch_mean",
    "speech_rate_proxy",
    "mean_turn_duration_s_transcript",
    "backchannel_rate_transcript",
    "overlap_fraction_transcript",
    "response_gap_mean_s_transcript",
    "interruption_rate_transcript",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze individual participant audio/prosody features by task."
    )
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("data") / "derived_features",
        help="Directory containing audio_participant_task.tsv.",
    )
    parser.add_argument(
        "--audio-features",
        type=Path,
        default=None,
        help="Optional explicit path to audio_participant_task.tsv.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "audio",
        help="Directory for individual audio analysis TSV outputs.",
    )
    parser.add_argument(
        "--results-dir",
        dest="out_dir",
        type=Path,
        help="Alias for --out-dir (kept for compatibility with pipeline runners).",
    )
    parser.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directory for PNG figures (default: out-dir/figures).",
    )
    parser.add_argument(
        "--min-n",
        type=int,
        default=8,
        help="Minimum paired participants for task contrast inference.",
    )
    parser.add_argument("--no-figures", action="store_true", help="Skip PNG figure generation.")
    parser.add_argument(
        "--transcripts-root",
        type=Path,
        default=Path("results") / "audio" / "transcripts",
        help="Root directory of transcript folders (ses-*/T*/master_transcript*.tsv).",
    )
    parser.add_argument(
        "--paper-table",
        type=Path,
        default=Path("paper") / "tables" / "audio_turn_taking_summary.tex",
        help="Output path for a paper-ready LaTeX turn-taking summary table.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Figure output DPI.")
    parser.add_argument("--verbose", action="store_true", help="Enable DEBUG logging.")
    return parser


def read_tsv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"Missing required table: {path}")
        LOG.warning("Optional table not found: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def write_tsv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, na_rep="NA")
    LOG.info("Wrote %s (%d rows)", path, len(df))


def _normalize_session_id(session_id: object) -> str:
    value = str(session_id or "").strip()
    if not value:
        return ""
    return value if value.startswith("ses-") else f"ses-{value}"


def _group_id_from_session(session_id: object) -> str:
    match = re.search(r"(grp-[A-Za-z0-9]+)", str(session_id or ""))
    return match.group(1) if match else ""


def _present(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [col for col in columns if col in df.columns]


def _numeric_inplace(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def resolve_audio_path(features_dir: Path, explicit: Path | None) -> Path | None:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit)
    candidates.extend(
        [
            features_dir / "audio_participant_task.tsv",
            features_dir / "features_audio_participant_task.tsv",
            Path("features") / "audio_participant_task.tsv",
            Path("results") / "audio" / "audio_participant_task.tsv",
        ]
    )
    for path in candidates:
        if path.exists():
            return path
    return None


def load_audio_table(features_dir: Path, audio_features: Path | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and normalize the participant-task audio table."""
    resolved = resolve_audio_path(features_dir, audio_features)
    status: dict[str, object] = {
        "requested_path": str(audio_features) if audio_features is not None else "",
        "resolved_path": str(resolved) if resolved is not None else "",
        "status": "missing",
        "rows": 0,
        "participants": 0,
        "sessions": 0,
        "message": "",
    }
    if resolved is None:
        status["message"] = (
            "No audio_participant_task.tsv found. Run extract_audio_features.py first."
        )
        LOG.warning("%s", status["message"])
        return pd.DataFrame(), pd.DataFrame([status])

    raw = read_tsv(resolved)
    if raw.empty:
        status["status"] = "empty"
        status["message"] = "Audio feature table is empty."
        return pd.DataFrame(), pd.DataFrame([status])

    out = raw.copy()
    if "task" not in out.columns and "task_id" in out.columns:
        out["task"] = out["task_id"]
    if "session_id" in out.columns:
        out["session_id"] = out["session_id"].map(_normalize_session_id)
    if "group_id" not in out.columns and "session_id" in out.columns:
        out["group_id"] = out["session_id"].map(_group_id_from_session)

    keys = ["session_id", "group_id", "task", "participant_id"]
    missing = [key for key in keys if key not in out.columns]
    if missing:
        status["status"] = "invalid"
        status["message"] = f"Missing key columns: {', '.join(missing)}"
        LOG.warning("%s", status["message"])
        return pd.DataFrame(), pd.DataFrame([status])

    out = out.drop(
        columns=[
            col
            for col in out.columns
            if "path" in col.lower() or "file" in col.lower() or "transcript_text" in col.lower()
        ],
        errors="ignore",
    )
    out["participant_id"] = out["participant_id"].astype(str).str.strip()
    out["task"] = out["task"].astype(str).str.strip()
    out["participant_uid"] = out["session_id"].astype(str) + ":" + out["participant_id"].astype(str)
    _numeric_inplace(out, AUDIO_FEATURES + ["duration_s"])

    if out.duplicated(keys).any():
        LOG.warning("Duplicate participant-task audio rows found; keeping first row per key.")
        out = out.drop_duplicates(keys, keep="first")

    status.update(
        {
            "status": "ok",
            "rows": int(len(out)),
            "participants": int(out["participant_uid"].nunique()),
            "sessions": int(out["session_id"].nunique()),
            "message": "Audio table loaded.",
        }
    )
    return out, pd.DataFrame([status])


def add_individual_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add baseline deltas, within-group shares, and speaking ranks."""
    if df.empty:
        return df
    out = df.copy()
    feature_cols = _present(out, AUDIO_FEATURES)
    _numeric_inplace(out, feature_cols)

    group_keys = ["session_id", "group_id", "task"]
    if "speaking_time_s" in out.columns:
        total = out.groupby(group_keys)["speaking_time_s"].transform("sum")
        out["speaking_share_group"] = out["speaking_time_s"] / total.replace(0, np.nan)
        out["speaking_rank_in_group"] = out.groupby(group_keys)["speaking_time_s"].rank(
            method="min", ascending=False
        )
    if "turn_count" in out.columns:
        turn_total = out.groupby(group_keys)["turn_count"].transform("sum")
        out["turn_share_group"] = out["turn_count"] / turn_total.replace(0, np.nan)

    delta_features = feature_cols + _present(out, ["speaking_share_group", "turn_share_group"])
    baseline = out[out["task"] == "T0"][["participant_uid", *delta_features]].copy()
    if not baseline.empty:
        baseline = baseline.groupby("participant_uid", as_index=False)[delta_features].mean()
        baseline = baseline.rename(columns={col: f"{col}_t0" for col in delta_features})
        out = out.merge(baseline, on="participant_uid", how="left")
        for col in delta_features:
            out[f"{col}_delta_t0"] = out[col] - out[f"{col}_t0"]
    return out


def build_individual_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate participant-level audio profiles across active tasks."""
    if df.empty:
        return pd.DataFrame()
    features = _present(
        df,
        DEFAULT_CONTRAST_FEATURES + [f"{col}_delta_t0" for col in DEFAULT_CONTRAST_FEATURES],
    )
    if not features:
        return pd.DataFrame()

    active = df[df["task"].isin(ACTIVE_TASKS)].copy()
    if active.empty:
        active = df.copy()
    summary = active.groupby(
        ["session_id", "group_id", "participant_uid", "participant_id"],
        as_index=False,
    )[features].agg(["mean", "std", "min", "max"])
    summary.columns = [
        col if stat == "" else f"{col}_{stat}"
        for col, stat in summary.columns.to_flat_index()
    ]
    coverage = (
        active.assign(valid_audio=active[_present(active, AUDIO_FEATURES)].notna().any(axis=1))
        .groupby(["session_id", "group_id", "participant_uid", "participant_id"], as_index=False)
        .agg(audio_task_rows=("task", "nunique"), valid_audio_tasks=("valid_audio", "sum"))
    )
    return summary.merge(coverage, on=["session_id", "group_id", "participant_uid", "participant_id"])


def build_qc_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    qc_col = "qc_flag" if "qc_flag" in df.columns else None
    for (session_id, group_id, participant_id), sub in df.groupby(
        ["session_id", "group_id", "participant_id"]
    ):
        feature_present = sub[_present(sub, AUDIO_FEATURES)].notna().any(axis=1)
        flags = (
            sub[qc_col].fillna("missing").astype(str).str.split(";").explode()
            if qc_col is not None
            else pd.Series(dtype=str)
        )
        rows.append(
            {
                "session_id": session_id,
                "group_id": group_id,
                "participant_id": participant_id,
                "task_rows": int(sub["task"].nunique()),
                "valid_audio_tasks": int(feature_present.sum()),
                "audio_available_tasks": int(
                    pd.Series(sub.get("audio_available", False)).astype(str).str.lower().isin(
                        ["true", "1", "yes"]
                    ).sum()
                ),
                "dominant_qc_flag": flags.value_counts().idxmax() if not flags.empty else "",
            }
        )
    return pd.DataFrame(rows)


def _safe_ttest_rel(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    try:
        from scipy import stats as sp_stats
    except ImportError:
        return float("nan"), float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sp_stats.ttest_rel(a, b, nan_policy="omit")
    except Exception:
        return float("nan"), float("nan")
    return float(result.statistic), float(result.pvalue)


def _safe_wilcoxon(diff: pd.Series) -> tuple[float, float]:
    try:
        from scipy import stats as sp_stats
    except ImportError:
        return float("nan"), float("nan")
    x = pd.to_numeric(diff, errors="coerce").dropna()
    if len(x) < 5 or np.allclose(x, 0.0):
        return float("nan"), float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = sp_stats.wilcoxon(x)
    except Exception:
        return float("nan"), float("nan")
    return float(result.statistic), float(result.pvalue)


def _cohen_dz(diff: pd.Series) -> float:
    x = pd.to_numeric(diff, errors="coerce").dropna()
    if len(x) < 2:
        return float("nan")
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return float("nan")
    return float(x.mean() / sd)


def _bh_fdr(p_values: Iterable[float]) -> list[float]:
    raw = np.array([float(p) if p is not None else np.nan for p in p_values], dtype=float)
    q = np.full(raw.shape, np.nan, dtype=float)
    valid = np.isfinite(raw)
    if not valid.any():
        return q.tolist()
    valid_idx = np.where(valid)[0]
    ordered_idx = valid_idx[np.argsort(raw[valid])]
    ordered_p = raw[ordered_idx]
    ranked = ordered_p * float(len(ordered_p)) / np.arange(1, len(ordered_p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    q[ordered_idx] = np.clip(adjusted, 0.0, 1.0)
    return q.tolist()


def build_task_tests(df: pd.DataFrame, min_n: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    outcomes = _present(
        df,
        DEFAULT_CONTRAST_FEATURES + [f"{col}_delta_t0" for col in DEFAULT_CONTRAST_FEATURES],
    )
    rows: list[dict[str, object]] = []
    for outcome in outcomes:
        tmp = df[["participant_uid", "task", outcome]].copy()
        tmp[outcome] = pd.to_numeric(tmp[outcome], errors="coerce")
        pivot = tmp.dropna().pivot_table(
            index="participant_uid", columns="task", values=outcome, aggfunc="mean"
        )
        for task_a, task_b in itertools.combinations([t for t in TASK_ORDER if t in pivot], 2):
            pair = pivot[[task_a, task_b]].dropna()
            n = int(len(pair))
            diff = pair[task_b] - pair[task_a] if n else pd.Series(dtype=float)
            row: dict[str, object] = {
                "outcome": outcome,
                "task_a": task_a,
                "task_b": task_b,
                "n_pairs": n,
                "mean_a": float(pair[task_a].mean()) if n else np.nan,
                "mean_b": float(pair[task_b].mean()) if n else np.nan,
                "mean_diff_b_minus_a": float(diff.mean()) if n else np.nan,
                "cohen_dz": _cohen_dz(diff) if n >= min_n else np.nan,
            }
            if n >= min_n:
                t_stat, p_t = _safe_ttest_rel(pair[task_b], pair[task_a])
                w_stat, p_w = _safe_wilcoxon(diff)
                row.update(
                    {
                        "paired_t": t_stat,
                        "p_paired_t": p_t,
                        "wilcoxon_w": w_stat,
                        "p_wilcoxon": p_w,
                    }
                )
            else:
                row.update(
                    {
                        "paired_t": np.nan,
                        "p_paired_t": np.nan,
                        "wilcoxon_w": np.nan,
                        "p_wilcoxon": np.nan,
                    }
                )
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_paired_t"] = _bh_fdr(out["p_paired_t"])
        out["q_wilcoxon"] = _bh_fdr(out["p_wilcoxon"])
    return out


def _label_feature(name: object) -> str:
    label = str(name)
    replacements = {
        "_delta_t0": " delta T0",
        "speaking_time_s": "Speaking time",
        "speaking_fraction": "Speaking fraction",
        "speaking_share_group": "Speaking share",
        "turn_share_group": "Turn share",
        "pause_count": "Pause count",
        "turn_count": "Turn count",
        "overlap_fraction": "Overlap fraction",
        "uncertain_fraction": "Uncertain fraction",
        "energy_mean": "Loudness",
        "energy_sd": "Loudness SD",
        "pitch_mean": "Pitch",
        "pitch_sd": "Pitch SD",
        "hnr_mean": "HNR",
        "jitter_mean": "Jitter",
        "shimmer_mean": "Shimmer",
        "speech_rate_proxy": "Speech rate",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    return " ".join(label.replace("_", " ").split()).title()


def _save_figure(fig: object, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    LOG.info("Wrote %s", path)


def plot_task_heatmap(df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    features = _present(df, DEFAULT_CONTRAST_FEATURES)
    if df.empty or not features:
        return
    matrix = df.groupby("task")[features].mean().reindex(TASK_ORDER).transpose()
    matrix = matrix.dropna(how="all")
    if matrix.empty:
        return
    centered = matrix.sub(matrix.mean(axis=1), axis=0)
    scaled = centered.div(centered.std(axis=1, ddof=1).replace(0, np.nan), axis=0)
    scaled = scaled.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if scaled.empty:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = scaled.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, max(4.2, 0.42 * len(scaled) + 1.6)))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-2.0, vmax=2.0, aspect="auto")
    ax.set_title("Individual Audio Features by Task", fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(len(scaled.columns)))
    ax.set_xticklabels(scaled.columns)
    ax.set_yticks(np.arange(len(scaled.index)))
    ax.set_yticklabels([_label_feature(i) for i in scaled.index], fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Within-feature z")
    _save_figure(fig, figures_dir / "individual_audio_task_heatmap.png", dpi)
    plt.close(fig)


def plot_speaking_fraction(df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if df.empty or "speaking_fraction" not in df.columns:
        return
    tmp = df.copy()
    tmp["speaking_fraction"] = pd.to_numeric(tmp["speaking_fraction"], errors="coerce")
    tmp = tmp.dropna(subset=["speaking_fraction"])
    if tmp.empty:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    data = [
        tmp.loc[tmp["task"] == task, "speaking_fraction"].to_numpy(dtype=float)
        for task in TASK_ORDER
    ]
    present = [(task, values) for task, values in zip(TASK_ORDER, data, strict=True) if len(values)]
    if not present:
        plt.close(fig)
        return
    labels, values = zip(*present, strict=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="The 'labels' parameter of boxplot")
        ax.boxplot(values, labels=labels, showfliers=False, patch_artist=True)
    ax.plot(np.arange(1, len(values) + 1), [float(np.mean(v)) for v in values], "o-", color="#d62728")
    ax.set_ylabel("Speaking fraction")
    ax.set_title("Individual Speaking Fraction by Task", fontsize=12, fontweight="bold")
    ax.grid(axis="y", color="0.9")
    _save_figure(fig, figures_dir / "individual_speaking_fraction_by_task.png", dpi)
    plt.close(fig)


def plot_participant_profiles(summary: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    cols = _present(
        summary,
        [
            "speaking_fraction_mean",
            "speaking_share_group_mean",
            "turn_share_group_mean",
            "speech_rate_proxy_mean",
            "energy_mean_mean",
            "pitch_mean_mean",
        ],
    )
    if summary.empty or not cols:
        return
    matrix = summary.set_index("participant_uid")[cols]
    matrix = matrix.dropna(how="all")
    if matrix.empty:
        return
    centered = matrix.sub(matrix.mean(axis=0), axis=1)
    scaled = centered.div(centered.std(axis=0, ddof=1).replace(0, np.nan), axis=1)
    scaled = scaled.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if scaled.empty:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.8, max(4.0, 0.22 * len(scaled) + 1.8)))
    im = ax.imshow(scaled.to_numpy(dtype=float), cmap="RdBu_r", vmin=-2.0, vmax=2.0, aspect="auto")
    ax.set_title("Individual Audio Profiles", fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([_label_feature(c.replace("_mean", "")) for c in cols], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(scaled.index)))
    ax.set_yticklabels(scaled.index, fontsize=7)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label("Across-participant z")
    _save_figure(fig, figures_dir / "individual_audio_profiles.png", dpi)
    plt.close(fig)


def plot_qc_coverage(qc: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if qc.empty:
        return
    tmp = qc.copy()
    tmp["valid_audio_tasks"] = pd.to_numeric(tmp["valid_audio_tasks"], errors="coerce")
    if tmp["valid_audio_tasks"].dropna().empty:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    counts = tmp["valid_audio_tasks"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(counts.index.astype(str), counts.values, color="#4c78a8")
    ax.set_xlabel("Valid audio tasks per participant")
    ax.set_ylabel("Participant count")
    ax.set_title("Individual Audio Coverage", fontsize=12, fontweight="bold")
    ax.grid(axis="y", color="0.92")
    _save_figure(fig, figures_dir / "individual_audio_qc_coverage.png", dpi)
    plt.close(fig)




def _speaker_to_participant(speaker: object) -> str:
    token = str(speaker or "").strip()
    if not token:
        return ""
    if token.upper().startswith("MODERATOR"):
        return "MODERATOR"
    match = PARTICIPANT_RE.search(token)
    if match:
        return match.group(1).upper()
    return token


def _read_transcript_table(path: Path, session_id: str, task: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        return pd.DataFrame()
    required = {"onset", "duration", "speaker"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    out = df.copy()
    out["session_id"] = session_id
    out["task"] = task
    out["onset"] = pd.to_numeric(out["onset"], errors="coerce")
    out["duration"] = pd.to_numeric(out["duration"], errors="coerce")
    out = out.dropna(subset=["onset", "duration"])
    out = out[out["duration"] > 0].copy()
    if out.empty:
        return pd.DataFrame()
    out["end"] = out["onset"] + out["duration"]
    out["participant_id"] = out["speaker"].map(_speaker_to_participant)
    out["is_participant"] = out["participant_id"].isin(["P1", "P2", "P3", "P4"])
    if "confidence" not in out.columns:
        out["confidence"] = ""
    out["confidence"] = out["confidence"].fillna("").astype(str)
    return out


def _collect_transcript_task_tables(transcripts_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    status: dict[str, object] = {
        "transcripts_root": str(transcripts_root),
        "status": "missing",
        "sessions": 0,
        "tasks": 0,
        "rows": 0,
        "message": "",
    }
    if not transcripts_root.exists():
        status["message"] = "Transcripts root not found."
        LOG.warning("%s: %s", status["message"], transcripts_root)
        return pd.DataFrame(), pd.DataFrame([status])

    rows: list[pd.DataFrame] = []
    for session_dir in sorted(p for p in transcripts_root.iterdir() if p.is_dir()):
        session_id = _normalize_session_id(session_dir.name)
        for task_dir in sorted(p for p in session_dir.iterdir() if p.is_dir()):
            task = task_dir.name
            transcript_path = None
            for candidate in [
                task_dir / "master_transcript_with_backchannels.tsv",
                task_dir / "master_transcript.tsv",
            ]:
                if candidate.exists():
                    transcript_path = candidate
                    break
            if transcript_path is None:
                continue
            table = _read_transcript_table(transcript_path, session_id=session_id, task=task)
            if not table.empty:
                rows.append(table)

    if not rows:
        status.update({"status": "empty", "message": "No readable transcript task tables found."})
        return pd.DataFrame(), pd.DataFrame([status])

    transcripts = pd.concat(rows, ignore_index=True)
    status.update(
        {
            "status": "ok",
            "sessions": int(transcripts["session_id"].nunique()),
            "tasks": int(transcripts[["session_id", "task"]].drop_duplicates().shape[0]),
            "rows": int(len(transcripts)),
            "message": "Transcript tables loaded.",
        }
    )
    return transcripts, pd.DataFrame([status])


def _interval_overlap_seconds(start: float, end: float, intervals: list[tuple[float, float]]) -> float:
    if not intervals or end <= start:
        return 0.0
    clipped: list[tuple[float, float]] = []
    for left, right in intervals:
        lo = max(start, left)
        hi = min(end, right)
        if hi > lo:
            clipped.append((lo, hi))
    if not clipped:
        return 0.0
    clipped.sort(key=lambda x: x[0])
    merged: list[tuple[float, float]] = []
    current_start, current_end = clipped[0]
    for lo, hi in clipped[1:]:
        if lo <= current_end:
            current_end = max(current_end, hi)
        else:
            merged.append((current_start, current_end))
            current_start, current_end = lo, hi
    merged.append((current_start, current_end))
    return float(sum(hi - lo for lo, hi in merged))


def _task_transcript_metrics(task_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    participants = task_df[task_df["is_participant"]].copy()
    if participants.empty:
        return pd.DataFrame(), pd.DataFrame()

    participant_intervals: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for _, row in participants.iterrows():
        participant_intervals[str(row["participant_id"])].append((float(row["onset"]), float(row["end"])))

    all_participant_intervals = [
        (float(row["onset"]), float(row["end"])) for _, row in participants.iterrows()
    ]

    response_gaps: dict[str, list[float]] = defaultdict(list)
    interruption_out: dict[str, int] = defaultdict(int)
    interruption_in: dict[str, int] = defaultdict(int)

    seq = participants.sort_values(["onset", "end"]).reset_index(drop=True)
    dyad_rows: list[dict[str, object]] = []
    for idx in range(len(seq) - 1):
        cur = seq.iloc[idx]
        nxt = seq.iloc[idx + 1]
        cur_spk = str(cur["participant_id"])
        nxt_spk = str(nxt["participant_id"])
        if cur_spk == nxt_spk:
            continue
        gap = float(nxt["onset"] - cur["end"])
        dyad_rows.append(
            {
                "session_id": str(cur["session_id"]),
                "task": str(cur["task"]),
                "from_participant": cur_spk,
                "to_participant": nxt_spk,
                "gap_s": gap,
            }
        )
        response_gaps[nxt_spk].append(gap)
        if gap < 0:
            interruption_out[nxt_spk] += 1
            interruption_in[cur_spk] += 1

    p_rows: list[dict[str, object]] = []
    for participant_id, sub in participants.groupby("participant_id"):
        speaking_total = float(pd.to_numeric(sub["duration"], errors="coerce").sum())
        turn_count = int(len(sub))
        overlap_seconds = 0.0
        intervals_other = [
            interval
            for other_id, intervals in participant_intervals.items()
            if other_id != participant_id
            for interval in intervals
        ]
        for _, row in sub.iterrows():
            overlap_seconds += _interval_overlap_seconds(
                float(row["onset"]),
                float(row["end"]),
                intervals_other,
            )
        overlap_fraction = overlap_seconds / speaking_total if speaking_total > 0 else np.nan

        backchannels = int((sub["confidence"].str.lower() == "backchannel").sum())
        backchannel_rate = (backchannels / turn_count) if turn_count > 0 else np.nan

        gap_values = response_gaps.get(str(participant_id), [])
        gap_mean = float(np.mean(gap_values)) if gap_values else np.nan
        gap_median = float(np.median(gap_values)) if gap_values else np.nan
        interruption_count = int(interruption_out.get(str(participant_id), 0))
        interruption_rate = interruption_count / max(1, turn_count)

        p_rows.append(
            {
                "session_id": str(sub["session_id"].iloc[0]),
                "task": str(sub["task"].iloc[0]),
                "participant_id": str(participant_id),
                "turn_count_transcript": turn_count,
                "speaking_time_s_transcript": speaking_total,
                "mean_turn_duration_s_transcript": float(speaking_total / turn_count) if turn_count else np.nan,
                "backchannel_count_transcript": backchannels,
                "backchannel_rate_transcript": float(backchannel_rate),
                "overlap_seconds_transcript": float(overlap_seconds),
                "overlap_fraction_transcript": float(overlap_fraction),
                "response_gap_mean_s_transcript": gap_mean,
                "response_gap_median_s_transcript": gap_median,
                "interruptions_out_transcript": interruption_count,
                "interruptions_in_transcript": int(interruption_in.get(str(participant_id), 0)),
                "interruption_rate_transcript": float(interruption_rate),
            }
        )

    dyads = pd.DataFrame(dyad_rows)
    if not dyads.empty:
        dyads = (
            dyads.groupby(["session_id", "task", "from_participant", "to_participant"], as_index=False)
            .agg(
                transitions_count=("gap_s", "count"),
                mean_gap_s=("gap_s", "mean"),
                median_gap_s=("gap_s", "median"),
                overlap_handover_rate=("gap_s", lambda x: float((pd.Series(x) < 0).mean())),
                fast_response_rate=("gap_s", lambda x: float(((pd.Series(x) >= 0) & (pd.Series(x) <= 0.2)).mean())),
            )
        )

    return pd.DataFrame(p_rows), dyads


def build_transcript_features(transcripts_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transcripts, status = _collect_transcript_task_tables(transcripts_root)
    if transcripts.empty:
        return pd.DataFrame(), pd.DataFrame(), status

    participant_rows: list[pd.DataFrame] = []
    dyad_rows: list[pd.DataFrame] = []
    for _, task_df in transcripts.groupby(["session_id", "task"], sort=True):
        p_df, d_df = _task_transcript_metrics(task_df)
        if not p_df.empty:
            participant_rows.append(p_df)
        if not d_df.empty:
            dyad_rows.append(d_df)

    participant_task = pd.concat(participant_rows, ignore_index=True) if participant_rows else pd.DataFrame()
    if not participant_task.empty:
        participant_task["group_id"] = participant_task["session_id"].map(_group_id_from_session)
        participant_task["participant_uid"] = (
            participant_task["session_id"].astype(str) + ":" + participant_task["participant_id"].astype(str)
        )

    dyads = pd.concat(dyad_rows, ignore_index=True) if dyad_rows else pd.DataFrame()
    if not dyads.empty:
        dyads["group_id"] = dyads["session_id"].map(_group_id_from_session)

    return participant_task, dyads, status


def build_turn_taking_group_summary(transcript_task: pd.DataFrame) -> pd.DataFrame:
    if transcript_task.empty:
        return pd.DataFrame()
    keep = [
        "turn_count_transcript",
        "mean_turn_duration_s_transcript",
        "overlap_fraction_transcript",
        "response_gap_mean_s_transcript",
        "backchannel_rate_transcript",
        "interruption_rate_transcript",
    ]
    return (
        transcript_task.groupby(["task"], as_index=False)[keep]
        .agg(["mean", "std", "count"])  # type: ignore[arg-type]
        .pipe(lambda df: df.set_axis([
            col if stat == "" else f"{col}_{stat}"
            for col, stat in df.columns.to_flat_index()
        ], axis=1))
    )


def write_turn_taking_latex(summary: pd.DataFrame, path: Path) -> None:
    if summary.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[str] = [
        "\\begin{table}[t]",
        "  \\centering",
        "  \\caption{Transcript-derived turn-taking and overlap summary by task (participants only).}",
        "  \\label{tab:audio_turn_taking_summary}",
        "  \\small",
        "  \\begin{tabular}{lrrrrr}",
        "    \\toprule",
        "    \\textbf{Task} & \\textbf{Turns} & \\textbf{Turn dur. (s)} & \\textbf{Overlap frac.} & \\textbf{Resp. gap (s)} & \\textbf{Backchannel rate} \\\\ ",
        "    \\midrule",
    ]
    for _, row in summary.iterrows():
        task = str(row.get("task", ""))
        turns = float(row.get("turn_count_transcript_mean", np.nan))
        turn_dur = float(row.get("mean_turn_duration_s_transcript_mean", np.nan))
        overlap = float(row.get("overlap_fraction_transcript_mean", np.nan))
        gap = float(row.get("response_gap_mean_s_transcript_mean", np.nan))
        bc_rate = float(row.get("backchannel_rate_transcript_mean", np.nan))
        rows.append(
            f"    {task} & {turns:.1f} & {turn_dur:.2f} & {overlap:.3f} & {gap:.2f} & {bc_rate:.3f} \\\\" 
        )
    rows.extend(
        [
            "    \\bottomrule",
            "  \\end{tabular}",
            "\\end{table}",
            "",
        ]
    )
    path.write_text("\n".join(rows), encoding="utf-8")
    LOG.info("Wrote %s", path)
def build_visualizations(out_dir: Path, figures_dir: Path, dpi: int) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    task = read_tsv(out_dir / "individual_audio_task.tsv", required=False)
    summary = read_tsv(out_dir / "individual_audio_summary.tsv", required=False)
    qc = read_tsv(out_dir / "individual_audio_qc.tsv", required=False)
    plot_task_heatmap(task, figures_dir, dpi)
    plot_speaking_fraction(task, figures_dir, dpi)
    plot_participant_profiles(summary, figures_dir, dpi)
    plot_qc_coverage(qc, figures_dir, dpi)


def write_manifest(out_dir: Path, status: pd.DataFrame, figures_dir: Path | None) -> None:
    lines = [
        "# Individual Audio Analysis Manifest",
        "",
        "Participant-level speech/prosody summaries derived from `audio_participant_task.tsv`.",
        "",
        "## Input Status",
    ]
    if not status.empty:
        row = status.iloc[0]
        lines.extend(
            [
                f"- status: {row.get('status', '')}",
                f"- resolved path: {row.get('resolved_path', '')}",
                f"- rows: {row.get('rows', 0)}",
                f"- participants: {row.get('participants', 0)}",
                f"- message: {row.get('message', '')}",
            ]
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "- `individual_audio_task.tsv`: participant-task audio rows with shares/ranks and T0 deltas.",
            "- `individual_audio_summary.tsv`: participant-level active-task profile summaries.",
            "- `individual_audio_qc.tsv`: per-participant coverage and dominant QC flags.",
            "- `individual_audio_task_pairwise_tests.tsv`: paired task contrasts across participants.",
            "- `individual_audio_status.tsv`: input discovery and load status.",
            "- `individual_audio_transcript_task.tsv`: transcript-derived participant-task turn-taking/overlap features.",
            "- `individual_audio_turn_taking_dyads.tsv`: dyadic handover and response-gap metrics.",
            "- `audio_turn_taking_group_summary.tsv`: task-level transcript turn-taking summary.",
            "- `individual_audio_transcript_status.tsv`: transcript discovery and load status.",
            "",
            "## Privacy",
            "- Outputs use anonymized session and P1-P4 participant-seat IDs only.",
            "- Transcript text and source audio paths are excluded.",
            "",
        ]
    )
    if figures_dir is not None:
        lines.extend(
            [
                "## Figures",
                "- `figures/individual_audio_task_heatmap.png`",
                "- `figures/individual_speaking_fraction_by_task.png`",
                "- `figures/individual_audio_profiles.png`",
                "- `figures/individual_audio_qc_coverage.png`",
                "",
            ]
        )
    path = out_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("Wrote %s", path)


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    out_dir = args.out_dir.resolve()
    figures_dir = (
        args.figures_dir.resolve()
        if args.figures_dir is not None
        else out_dir / "figures"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    audio, status = load_audio_table(args.features_dir.resolve(), args.audio_features)
    write_tsv(status, out_dir / "individual_audio_status.tsv")
    if audio.empty:
        write_tsv(pd.DataFrame(), out_dir / "individual_audio_task.tsv")
        write_tsv(pd.DataFrame(), out_dir / "individual_audio_summary.tsv")
        write_tsv(pd.DataFrame(), out_dir / "individual_audio_qc.tsv")
        write_tsv(pd.DataFrame(), out_dir / "individual_audio_task_pairwise_tests.tsv")
        write_manifest(out_dir, status, None if args.no_figures else figures_dir)
        LOG.info("Individual audio analysis finished without audio input: %s", out_dir)
        return 0

    task = add_individual_metrics(audio)
    summary = build_individual_summary(task)
    qc = build_qc_summary(task)
    tests = build_task_tests(task, args.min_n)

    write_tsv(task, out_dir / "individual_audio_task.tsv")
    write_tsv(summary, out_dir / "individual_audio_summary.tsv")
    write_tsv(qc, out_dir / "individual_audio_qc.tsv")
    write_tsv(tests, out_dir / "individual_audio_task_pairwise_tests.tsv")
    if not args.no_figures:
        build_visualizations(out_dir, figures_dir, args.dpi)
    write_manifest(out_dir, status, None if args.no_figures else figures_dir)
    LOG.info("Individual audio analysis complete: %s", out_dir)
    return 0



def main_transcript() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    out_dir = args.out_dir.resolve()
    figures_dir = args.figures_dir.resolve() if args.figures_dir is not None else out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    transcript_task, dyads, transcript_status = build_transcript_features(args.transcripts_root.resolve())
    write_tsv(transcript_status, out_dir / "individual_audio_transcript_status.tsv")
    if transcript_task.empty:
        write_tsv(pd.DataFrame(), out_dir / "individual_audio_transcript_task.tsv")
        write_tsv(pd.DataFrame(), out_dir / "individual_audio_turn_taking_dyads.tsv")
        write_tsv(pd.DataFrame(), out_dir / "audio_turn_taking_group_summary.tsv")
    else:
        write_tsv(transcript_task, out_dir / "individual_audio_transcript_task.tsv")
        write_tsv(dyads, out_dir / "individual_audio_turn_taking_dyads.tsv")
        group_turn_summary = build_turn_taking_group_summary(transcript_task)
        write_tsv(group_turn_summary, out_dir / "audio_turn_taking_group_summary.tsv")
        write_turn_taking_latex(group_turn_summary, args.paper_table.resolve())

    audio, status = load_audio_table(args.features_dir.resolve(), args.audio_features)
    write_tsv(status, out_dir / "individual_audio_status.tsv")
    if audio.empty:
        write_tsv(pd.DataFrame(), out_dir / "individual_audio_task.tsv")
        write_tsv(pd.DataFrame(), out_dir / "individual_audio_summary.tsv")
        write_tsv(pd.DataFrame(), out_dir / "individual_audio_qc.tsv")
        write_tsv(pd.DataFrame(), out_dir / "individual_audio_task_pairwise_tests.tsv")
        write_manifest(out_dir, status, None if args.no_figures else figures_dir)
        LOG.info("Individual audio analysis finished without audio input: %s", out_dir)
        return 0

    task = add_individual_metrics(audio)
    if not transcript_task.empty:
        merge_cols = [
            "session_id", "task", "participant_id",
            "turn_count_transcript", "speaking_time_s_transcript", "mean_turn_duration_s_transcript",
            "backchannel_count_transcript", "backchannel_rate_transcript", "overlap_seconds_transcript",
            "overlap_fraction_transcript", "response_gap_mean_s_transcript", "response_gap_median_s_transcript",
            "interruptions_out_transcript", "interruptions_in_transcript", "interruption_rate_transcript",
        ]
        task = task.merge(transcript_task[merge_cols], on=["session_id", "task", "participant_id"], how="left")
        if "turn_count" in task.columns:
            task["turn_count"] = pd.to_numeric(task["turn_count"], errors="coerce").fillna(task["turn_count_transcript"])
        if "overlap_fraction" in task.columns:
            task["overlap_fraction"] = pd.to_numeric(task["overlap_fraction"], errors="coerce").fillna(task["overlap_fraction_transcript"])

    summary = build_individual_summary(task)
    qc = build_qc_summary(task)
    tests = build_task_tests(task, args.min_n)

    write_tsv(task, out_dir / "individual_audio_task.tsv")
    write_tsv(summary, out_dir / "individual_audio_summary.tsv")
    write_tsv(qc, out_dir / "individual_audio_qc.tsv")
    write_tsv(tests, out_dir / "individual_audio_task_pairwise_tests.tsv")
    if not args.no_figures:
        build_visualizations(out_dir, figures_dir, args.dpi)
    write_manifest(out_dir, status, None if args.no_figures else figures_dir)
    LOG.info("Individual audio analysis complete: %s", out_dir)
    return 0
if __name__ == "__main__":
    raise SystemExit(main_transcript())


















