"""Inferential statistics for GroupAffect-4 multimodal participant, group, and dyad data.

The script consumes the joined feature tables produced by the feature pipeline
and writes conservative, auditable TSV outputs for:

- participant-task mixed models and pairwise task tests,
- cross-modal correlations among physiology, answers, annotations, and personality,
- group-level repeated task tests,
- dyad-level synchrony tests.
"""

from __future__ import annotations

import argparse
import itertools
import logging
import re
import warnings
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger("analyze_multimodal_statistics")

TASK_ORDER = ["T0", "T1", "T2", "T3", "T4"]
ACTIVE_TASKS = ["T1", "T2", "T3", "T4"]

SELF_REPORT_CANDIDATES = [
    "vad_arousal_self_num",
    "vad_valence_self_num",
    "vad_dominance_self_num",
    "ans_arousal",
    "ans_valence",
    "ans_dominance",
    "ans_overall_valence",
    "ans_engagement",
    "ans_mental_demand",
    "ans_voice_inclusion",
    "ans_team_coordination",
    "ans_perceived_control",
    "ans_decision_confidence",
    "ans_fairness",
    "ans_psych_safety",
    "ans_satisfaction",
    "ans_regret",
    "ans_social_concern",
    "ans_trust_front",
    "ans_trust_angle",
    "ans_trust_next",
]

PHYSIO_CANDIDATES = [
    "hr_mean_bpm",
    "hrv_rmssd_ms",
    "hrv_sdnn_ms",
    "eda_mean",
    "eda_tonic_mean",
    "eda_phasic_mean",
    "eda_phasic_rate_hz",
    "scr_rate_hz",
    "temp_mean",
    "accel_motion_mean",
    "pupil_mean",
    "pupil_std",
    "pupil_slope_per_s",
    "gaze_valid_frac",
    "pupil_missing_frac",
]

PHYSIO_DELTA_CANDIDATES = [
    "hr_mean_bpm_delta_t0",
    "hrv_rmssd_ms_delta_t0",
    "eda_mean_delta_t0",
    "eda_tonic_mean_delta_t0",
    "eda_phasic_rate_hz_delta_t0",
    "scr_rate_hz_delta_t0",
    "temp_mean_delta_t0",
    "pupil_mean_delta_t0",
    "pupil_std_delta_t0",
    "pupil_slope_per_s_delta_t0",
]

AUDIO_CANDIDATES = [
    "audio_speaking_time_s",
    "audio_speaking_fraction",
    "audio_pause_count",
    "audio_turn_count",
    "audio_overlap_fraction",
    "audio_uncertain_fraction",
    "audio_energy_mean",
    "audio_energy_sd",
    "audio_pitch_mean",
    "audio_pitch_sd",
    "audio_hnr_mean",
    "audio_jitter_mean",
    "audio_shimmer_mean",
    "audio_voiced_segments_per_sec",
    "audio_mean_voiced_segment_s",
    "audio_mean_unvoiced_segment_s",
    "audio_speech_rate_proxy",
]

AUDIO_QC_CANDIDATES = [
    "audio_duration_s",
    "audio_available",
    "audio_transcript_available",
]

AUDIO_DELTA_CANDIDATES = [
    "audio_speaking_time_s_delta_t0",
    "audio_speaking_fraction_delta_t0",
    "audio_pause_count_delta_t0",
    "audio_turn_count_delta_t0",
    "audio_overlap_fraction_delta_t0",
    "audio_uncertain_fraction_delta_t0",
    "audio_energy_mean_delta_t0",
    "audio_energy_sd_delta_t0",
    "audio_pitch_mean_delta_t0",
    "audio_pitch_sd_delta_t0",
    "audio_hnr_mean_delta_t0",
    "audio_jitter_mean_delta_t0",
    "audio_shimmer_mean_delta_t0",
    "audio_voiced_segments_per_sec_delta_t0",
    "audio_mean_voiced_segment_s_delta_t0",
    "audio_mean_unvoiced_segment_s_delta_t0",
    "audio_speech_rate_proxy_delta_t0",
]

BIOMARKER_CANDIDATES = [
    "biomarker_cognitive_load",
    "biomarker_arousal_stress",
    "biomarker_attention",
    "biomarker_decision_pressure",
    "biomarker_recovery_capacity",
    "biomarker_fatigue_depletion",
]

ANNOTATION_CANDIDATES = [
    "annotation_entries_count",
    "annotation_tiers_count",
    "task_events_count",
    "ann_total_events_n",
    "ann_response_vad_n",
    "ann_response_postblock_n",
    "ann_response_form_n",
    "ann_push_vad_n",
    "ann_event_span_s",
    "answers_n",
]

PERSONALITY_CANDIDATES = [
    "bfi44_e_z",
    "bfi44_a_z",
    "bfi44_c_z",
    "bfi44_n_z",
    "bfi44_o_z",
    "age",
]

DEFAULT_TASK_TEST_OUTCOMES = [
    "vad_arousal_self_num",
    "vad_valence_self_num",
    "vad_dominance_self_num",
    "ans_overall_valence",
    "ans_engagement",
    "ans_mental_demand",
    "ans_voice_inclusion",
    "ans_team_coordination",
    "ans_fairness",
    "ans_regret",
    "biomarker_arousal_stress",
    "biomarker_cognitive_load",
    "biomarker_attention",
    "hr_mean_bpm_delta_t0",
    "hrv_rmssd_ms_delta_t0",
    "eda_mean_delta_t0",
    "eda_phasic_rate_hz_delta_t0",
    "scr_rate_hz_delta_t0",
    "temp_mean_delta_t0",
    "pupil_mean_delta_t0",
    "audio_speaking_fraction_delta_t0",
    "audio_turn_count_delta_t0",
    "audio_overlap_fraction_delta_t0",
    "audio_energy_mean_delta_t0",
    "audio_pitch_mean_delta_t0",
    "audio_speech_rate_proxy_delta_t0",
]

CORRELATION_FAMILY_SPECS = [
    ("self_report_vs_biomarker", SELF_REPORT_CANDIDATES, BIOMARKER_CANDIDATES),
    ("self_report_vs_physio_delta", SELF_REPORT_CANDIDATES, PHYSIO_DELTA_CANDIDATES),
    ("self_report_vs_audio_delta", SELF_REPORT_CANDIDATES, AUDIO_DELTA_CANDIDATES),
    ("self_report_vs_annotation", SELF_REPORT_CANDIDATES, ANNOTATION_CANDIDATES),
    ("self_report_vs_personality", SELF_REPORT_CANDIDATES, PERSONALITY_CANDIDATES),
    ("audio_delta_vs_biomarker", AUDIO_DELTA_CANDIDATES, BIOMARKER_CANDIDATES),
    ("audio_delta_vs_physio_delta", AUDIO_DELTA_CANDIDATES, PHYSIO_DELTA_CANDIDATES),
    ("audio_delta_vs_personality", AUDIO_DELTA_CANDIDATES, PERSONALITY_CANDIDATES),
    ("audio_delta_vs_annotation", AUDIO_DELTA_CANDIDATES, ANNOTATION_CANDIDATES),
    ("biomarker_vs_personality", BIOMARKER_CANDIDATES, PERSONALITY_CANDIDATES),
    ("biomarker_vs_annotation", BIOMARKER_CANDIDATES, ANNOTATION_CANDIDATES),
    ("physio_delta_vs_personality", PHYSIO_DELTA_CANDIDATES, PERSONALITY_CANDIDATES),
    ("physio_delta_vs_annotation", PHYSIO_DELTA_CANDIDATES, ANNOTATION_CANDIDATES),
    ("personality_vs_annotation", PERSONALITY_CANDIDATES, ANNOTATION_CANDIDATES),
]

MODEL_FAMILY_SPECS = [
    ("self_report_vs_biomarker", SELF_REPORT_CANDIDATES, BIOMARKER_CANDIDATES),
    ("self_report_vs_physio_delta", SELF_REPORT_CANDIDATES, PHYSIO_DELTA_CANDIDATES),
    ("self_report_vs_audio_delta", SELF_REPORT_CANDIDATES, AUDIO_DELTA_CANDIDATES),
    ("self_report_vs_annotation", SELF_REPORT_CANDIDATES, ANNOTATION_CANDIDATES),
    ("self_report_vs_personality", SELF_REPORT_CANDIDATES, PERSONALITY_CANDIDATES),
    ("audio_delta_vs_self_report", AUDIO_DELTA_CANDIDATES, SELF_REPORT_CANDIDATES),
    ("audio_delta_vs_biomarker", AUDIO_DELTA_CANDIDATES, BIOMARKER_CANDIDATES),
    ("audio_delta_vs_physio_delta", AUDIO_DELTA_CANDIDATES, PHYSIO_DELTA_CANDIDATES),
    ("audio_delta_vs_personality", AUDIO_DELTA_CANDIDATES, PERSONALITY_CANDIDATES),
    ("audio_delta_vs_annotation", AUDIO_DELTA_CANDIDATES, ANNOTATION_CANDIDATES),
    ("biomarker_vs_self_report", BIOMARKER_CANDIDATES, SELF_REPORT_CANDIDATES),
    ("biomarker_vs_personality", BIOMARKER_CANDIDATES, PERSONALITY_CANDIDATES),
    ("biomarker_vs_annotation", BIOMARKER_CANDIDATES, ANNOTATION_CANDIDATES),
    ("physio_delta_vs_self_report", PHYSIO_DELTA_CANDIDATES, SELF_REPORT_CANDIDATES),
    ("physio_delta_vs_personality", PHYSIO_DELTA_CANDIDATES, PERSONALITY_CANDIDATES),
    ("physio_delta_vs_annotation", PHYSIO_DELTA_CANDIDATES, ANNOTATION_CANDIDATES),
]


@dataclass(frozen=True)
class ModelSpec:
    family: str
    outcome: str
    predictor: str | None
    active_tasks_only: bool


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run mixed models and significance tests across GroupAffect-4 modalities."
    )
    p.add_argument(
        "--features-dir",
        type=Path,
        default=Path("data") / "derived_features",
        help="Directory containing participant_features_answers_annotations.tsv and dyad tables.",
    )
    p.add_argument(
        "--personality-dir",
        type=Path,
        default=Path("results") / "personality",
        help="Directory containing personality_summary.tsv and group_trait_stats.tsv.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "statistics",
        help="Directory where inferential statistics outputs are written.",
    )
    p.add_argument(
        "--figures-dir",
        type=Path,
        default=None,
        help="Directory where PNG visual summaries are written (default: out-dir/figures).",
    )
    p.add_argument(
        "--audio-features",
        type=Path,
        default=None,
        help=(
            "Optional path to audio_participant_task.tsv. If omitted, common "
            "locations under --features-dir, features/, and results/audio/ are checked."
        ),
    )
    p.add_argument(
        "--min-n",
        type=int,
        default=8,
        help="Minimum complete rows for correlations and participant-level models.",
    )
    p.add_argument(
        "--min-groups",
        type=int,
        default=6,
        help="Minimum random-effect groups for participant-level mixed models.",
    )
    p.add_argument(
        "--by-task-correlations",
        action="store_true",
        help="Also write task-stratified correlation rows.",
    )
    p.add_argument(
        "--all-pair-models",
        action="store_true",
        help="Fit a broader grid of self-report x predictor mixed models.",
    )
    p.add_argument(
        "--no-ols-fallback",
        action="store_true",
        help="Do not fall back to clustered OLS when MixedLM cannot be fitted.",
    )
    p.add_argument("--no-figures", action="store_true", help="Skip PNG visual summary generation.")
    p.add_argument("--dpi", type=int, default=180, help="Figure output DPI.")
    p.add_argument("--verbose", action="store_true", help="Enable verbose logging.")
    return p


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


def _group_id_from_session(session_id: object) -> str:
    m = re.search(r"(grp-[A-Za-z0-9]+)", str(session_id or ""))
    return m.group(1) if m else ""


def _session_core(session_id: object) -> str:
    value = str(session_id or "").strip()
    return value[4:] if value.startswith("ses-") else value


def _normalize_session_id(session_id: object) -> str:
    value = str(session_id or "").strip()
    if not value:
        return ""
    return value if value.startswith("ses-") else f"ses-{value}"


def resolve_audio_feature_path(features_dir: Path, explicit: Path | None) -> Path | None:
    """Return the first available audio participant-task table, if any."""
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


def _present(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [c for c in columns if c in df.columns]


def _complete_n(df: pd.DataFrame, left: str, right: str) -> int:
    if left not in df.columns or right not in df.columns:
        return 0
    pair = pd.DataFrame(
        {
            "left": pd.to_numeric(df[left], errors="coerce"),
            "right": pd.to_numeric(df[right], errors="coerce"),
        }
    ).replace([np.inf, -np.inf], np.nan)
    return int(len(pair.dropna()))


def _numeric_inplace(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")


def _standardize(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std(ddof=1)
    if not np.isfinite(sd) or sd <= 0:
        return pd.Series(np.nan, index=x.index, dtype=float)
    return (x - x.mean()) / sd


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
    n = float(len(ordered_p))
    ranked = ordered_p * n / np.arange(1, len(ordered_p) + 1)
    adjusted = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    q[ordered_idx] = adjusted
    return q.tolist()


def _safe_ttest_rel(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    try:
        from scipy import stats as sp_stats
    except ImportError:
        return float("nan"), float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp_stats.ttest_rel(a, b, nan_policy="omit")
    except Exception:
        return float("nan"), float("nan")
    return float(res.statistic), float(res.pvalue)


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
            res = sp_stats.wilcoxon(x)
    except Exception:
        return float("nan"), float("nan")
    return float(res.statistic), float(res.pvalue)


def _safe_corr(a: pd.Series, b: pd.Series, method: str) -> tuple[float, float]:
    try:
        from scipy import stats as sp_stats
    except ImportError:
        return float("nan"), float("nan")
    pair = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")})
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3 or pair["a"].nunique() < 2 or pair["b"].nunique() < 2:
        return float("nan"), float("nan")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if method == "pearson":
                res = sp_stats.pearsonr(pair["a"], pair["b"])
            else:
                res = sp_stats.spearmanr(pair["a"], pair["b"])
    except Exception:
        return float("nan"), float("nan")
    return float(res.statistic), float(res.pvalue)


def _paired_task_tests(
    df: pd.DataFrame,
    id_col: str,
    outcome_cols: Iterable[str],
    min_n: int,
    extra_fields: dict[str, object] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for outcome in outcome_cols:
        if outcome not in df.columns:
            continue
        tmp = df[[id_col, "task", outcome]].copy()
        tmp[outcome] = pd.to_numeric(tmp[outcome], errors="coerce")
        tmp = tmp.dropna(subset=[id_col, "task", outcome])
        if tmp.empty:
            continue
        pivot = tmp.pivot_table(index=id_col, columns="task", values=outcome, aggfunc="mean")
        task_pairs = itertools.combinations([t for t in TASK_ORDER if t in pivot.columns], 2)
        for task_a, task_b in task_pairs:
            pair = pivot[[task_a, task_b]].dropna()
            n = int(len(pair))
            row: dict[str, object] = {
                "outcome": outcome,
                "task_a": task_a,
                "task_b": task_b,
                "n_pairs": n,
                "mean_a": float(pair[task_a].mean()) if n else np.nan,
                "mean_b": float(pair[task_b].mean()) if n else np.nan,
                "mean_diff_b_minus_a": float((pair[task_b] - pair[task_a]).mean()) if n else np.nan,
            }
            if extra_fields:
                row.update(extra_fields)
            if n >= min_n:
                t_stat, p_t = _safe_ttest_rel(pair[task_b], pair[task_a])
                w_stat, p_w = _safe_wilcoxon(pair[task_b] - pair[task_a])
                row.update(
                    {
                        "paired_t": t_stat,
                        "p_paired_t": p_t,
                        "wilcoxon_w": w_stat,
                        "p_wilcoxon": p_w,
                        "cohen_dz": _cohen_dz(pair[task_b] - pair[task_a]),
                    }
                )
            else:
                row.update(
                    {
                        "paired_t": np.nan,
                        "p_paired_t": np.nan,
                        "wilcoxon_w": np.nan,
                        "p_wilcoxon": np.nan,
                        "cohen_dz": np.nan,
                    }
                )
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_paired_t"] = _bh_fdr(out["p_paired_t"])
        out["q_wilcoxon"] = _bh_fdr(out["p_wilcoxon"])
    return out


def _correlation_rows(
    df: pd.DataFrame,
    left_cols: Iterable[str],
    right_cols: Iterable[str],
    min_n: int,
    scope: str,
    extra_fields: dict[str, object] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for left, right in itertools.product(left_cols, right_cols):
        if left == right or left not in df.columns or right not in df.columns:
            continue
        pair = pd.DataFrame(
            {
                "left": pd.to_numeric(df[left], errors="coerce"),
                "right": pd.to_numeric(df[right], errors="coerce"),
            }
        ).replace([np.inf, -np.inf], np.nan)
        pair = pair.dropna()
        n = int(len(pair))
        if n < min_n:
            continue
        pearson_r, pearson_p = _safe_corr(pair["left"], pair["right"], "pearson")
        spearman_rho, spearman_p = _safe_corr(pair["left"], pair["right"], "spearman")
        row: dict[str, object] = {
            "scope": scope,
            "left": left,
            "right": right,
            "n": n,
            "pearson_r": pearson_r,
            "p_pearson": pearson_p,
            "spearman_rho": spearman_rho,
            "p_spearman": spearman_p,
        }
        if extra_fields:
            row.update(extra_fields)
        rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_pearson"] = _bh_fdr(out["p_pearson"])
        out["q_spearman"] = _bh_fdr(out["p_spearman"])
    return out


def _family_correlation_tables(
    df: pd.DataFrame,
    family_specs: Iterable[tuple[str, list[str], list[str]]],
    min_n: int,
    scope: str,
) -> list[pd.DataFrame]:
    tables: list[pd.DataFrame] = []
    for family, left_candidates, right_candidates in family_specs:
        left_cols = _present(df, left_candidates)
        right_cols = _present(df, right_candidates)
        table = _correlation_rows(
            df,
            left_cols,
            right_cols,
            min_n,
            scope,
            {"comparison_family": family},
        )
        if not table.empty:
            tables.append(table)
    return tables


def build_combination_catalog(participant: pd.DataFrame, min_n: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for analysis_type, family_specs in [
        ("correlation", CORRELATION_FAMILY_SPECS),
        ("mixed_model", MODEL_FAMILY_SPECS),
    ]:
        for family, left_candidates, right_candidates in family_specs:
            for left, right in itertools.product(left_candidates, right_candidates):
                present = left in participant.columns and right in participant.columns
                complete_n = _complete_n(participant, left, right) if present else 0
                rows.append(
                    {
                        "analysis_type": analysis_type,
                        "comparison_family": family,
                        "left_or_outcome": left,
                        "right_or_predictor": right,
                        "left_present": left in participant.columns,
                        "right_present": right in participant.columns,
                        "complete_n": complete_n,
                        "meets_min_n": bool(complete_n >= min_n),
                    }
                )
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(
            ["analysis_type", "comparison_family", "meets_min_n", "complete_n"],
            ascending=[True, True, False, False],
        )
    return out


def _add_baseline_deltas(
    df: pd.DataFrame,
    id_col: str,
    columns: Iterable[str],
    baseline_task: str = "T0",
) -> pd.DataFrame:
    out = df.copy()
    cols = [c for c in columns if c in out.columns]
    if not cols or id_col not in out.columns or "task" not in out.columns:
        return out
    _numeric_inplace(out, cols)
    baseline = out[out["task"] == baseline_task][[id_col, *cols]].copy()
    if baseline.empty:
        return out
    baseline = baseline.groupby(id_col, as_index=False)[cols].mean()
    baseline = baseline.rename(columns={c: f"{c}_t0" for c in cols})
    out = out.merge(baseline, on=id_col, how="left")
    for col in cols:
        out[f"{col}_delta_t0"] = out[col] - out[f"{col}_t0"]
    return out


def load_audio_features(
    features_dir: Path,
    audio_features_path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load optional audio participant-task features without text or file paths."""
    resolved = resolve_audio_feature_path(features_dir, audio_features_path)
    status: dict[str, object] = {
        "requested_path": str(audio_features_path) if audio_features_path is not None else "",
        "resolved_path": str(resolved) if resolved is not None else "",
        "status": "missing",
        "rows": 0,
        "audio_feature_columns": 0,
        "message": "",
    }
    if resolved is None:
        status["message"] = "No audio_participant_task.tsv found; audio analyses are cataloged as unavailable."
        LOG.warning("%s", status["message"])
        return pd.DataFrame(), pd.DataFrame([status])

    audio = read_tsv(resolved)
    if audio.empty:
        status["status"] = "empty"
        status["message"] = "Audio feature table is empty."
        return pd.DataFrame(), pd.DataFrame([status])

    out = audio.copy()
    if "task" not in out.columns and "task_id" in out.columns:
        out["task"] = out["task_id"]
    if "participant_id" not in out.columns and "participant" in out.columns:
        out = out.rename(columns={"participant": "participant_id"})
    if "session_id" in out.columns:
        out["session_id"] = out["session_id"].map(_normalize_session_id)
    if "group_id" not in out.columns and "session_id" in out.columns:
        out["group_id"] = out["session_id"].map(_group_id_from_session)

    keys = ["session_id", "group_id", "task", "participant_id"]
    missing_keys = [key for key in keys if key not in out.columns]
    if missing_keys:
        status["status"] = "invalid"
        status["message"] = f"Audio feature table missing required key columns: {', '.join(missing_keys)}"
        LOG.warning("%s", status["message"])
        return pd.DataFrame(), pd.DataFrame([status])

    out["participant_id"] = out["participant_id"].astype(str).str.strip()
    out["task"] = out["task"].astype(str).str.strip()

    renamed: dict[str, str] = {}
    for col in out.columns:
        if col in keys or col == "task_id":
            continue
        lower = col.lower()
        if "path" in lower or "file" in lower or "transcript_text" in lower:
            continue
        if col == "transcript_available":
            renamed[col] = "audio_transcript_available"
        elif col in {"audio_available", "qc_flag"}:
            renamed[col] = f"audio_{col}" if not col.startswith("audio_") else col
        elif col.startswith("audio_"):
            renamed[col] = col
        else:
            renamed[col] = f"audio_{col}"

    out = out.rename(columns=renamed)
    keep_cols = keys + [
        col
        for col in sorted(set(renamed.values()))
        if col in out.columns and (col in AUDIO_CANDIDATES or col in AUDIO_QC_CANDIDATES or col == "audio_qc_flag")
    ]
    out = out[keep_cols].copy()

    numeric_cols = _present(out, AUDIO_CANDIDATES + ["audio_duration_s"])
    _numeric_inplace(out, numeric_cols)
    if out.duplicated(keys).any():
        LOG.warning("Audio feature table has duplicate key rows; keeping first row per participant-task.")
        out = out.drop_duplicates(keys, keep="first")

    status["status"] = "ok"
    status["rows"] = int(len(out))
    status["audio_feature_columns"] = int(len(_present(out, AUDIO_CANDIDATES)))
    status["message"] = "Audio features loaded."
    return out, pd.DataFrame([status])


def enrich_group_with_audio(group_df: pd.DataFrame, participant: pd.DataFrame) -> pd.DataFrame:
    """Append group-level audio summaries derived from participant-task rows."""
    if group_df.empty:
        return group_df
    audio_cols = _present(participant, AUDIO_CANDIDATES + AUDIO_DELTA_CANDIDATES)
    if not audio_cols:
        return group_df

    keys = ["session_id", "group_id", "task"]
    missing_keys = [key for key in keys if key not in participant.columns or key not in group_df.columns]
    if missing_keys:
        LOG.warning("Cannot append group audio summaries; missing keys: %s", ", ".join(missing_keys))
        return group_df

    tmp = participant[keys + audio_cols].copy()
    _numeric_inplace(tmp, audio_cols)
    summary = tmp.groupby(keys)[audio_cols].agg(["mean", "std"]).reset_index()
    summary.columns = [
        feature if stat == "" else f"{feature}_group_{stat}"
        for feature, stat in summary.columns.to_flat_index()
    ]
    audio_n = (
        tmp.assign(_has_audio=tmp[audio_cols].notna().any(axis=1))
        .groupby(keys, as_index=False)["_has_audio"]
        .sum()
        .rename(columns={"_has_audio": "audio_participant_n"})
    )
    summary = summary.merge(audio_n, on=keys, how="left")
    return group_df.merge(summary, on=keys, how="left")


def load_participant_dataset(
    features_dir: Path,
    personality_dir: Path,
    audio_features_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    participant_path = features_dir / "participant_features_answers_annotations.tsv"
    participant = read_tsv(participant_path)
    if participant.empty:
        raise ValueError(f"Input table is empty: {participant_path}")

    out = participant.copy()
    if "session_id" in out.columns:
        out["session_id"] = out["session_id"].map(_normalize_session_id)
    if "group_id" not in out.columns:
        out["group_id"] = out["session_id"].map(_group_id_from_session)
    out["session_core"] = out["session_id"].map(_session_core)
    out["participant_uid"] = out["session_id"].astype(str) + ":" + out["participant_id"].astype(str)
    out = out.drop(
        columns=[
            c
            for c in out.columns
            if c.startswith("source_file") or c.startswith("ans_text_")
        ],
        errors="ignore",
    )

    personality = read_tsv(personality_dir / "personality_summary.tsv", required=False)
    if not personality.empty:
        per = personality.copy()
        per = per.rename(columns={"participant_id": "participant_global_id", "seat": "participant_id"})
        per["session_core"] = per["session_id"].map(_session_core)
        merge_cols = [
            "session_core",
            "group_id",
            "participant_id",
            "participant_global_id",
            "age",
            "sex",
            "handedness",
            "english_proficiency",
            "education",
            "bfi44_e",
            "bfi44_a",
            "bfi44_c",
            "bfi44_n",
            "bfi44_o",
            "bfi44_e_z",
            "bfi44_a_z",
            "bfi44_c_z",
            "bfi44_n_z",
            "bfi44_o_z",
        ]
        per = per[[c for c in merge_cols if c in per.columns]]
        out = out.merge(per, on=["session_core", "group_id", "participant_id"], how="left")

    audio, audio_status = load_audio_features(features_dir, audio_features_path)
    if not audio.empty:
        out = out.merge(
            audio,
            on=["session_id", "group_id", "task", "participant_id"],
            how="left",
        )

    numeric_candidates = (
        SELF_REPORT_CANDIDATES
        + PHYSIO_CANDIDATES
        + AUDIO_CANDIDATES
        + ["audio_duration_s"]
        + BIOMARKER_CANDIDATES
        + ANNOTATION_CANDIDATES
        + PERSONALITY_CANDIDATES
    )
    _numeric_inplace(out, numeric_candidates)
    out = _add_baseline_deltas(
        out,
        "participant_uid",
        PHYSIO_CANDIDATES + BIOMARKER_CANDIDATES + AUDIO_CANDIDATES,
    )
    return out, audio_status


def load_group_dataset(features_dir: Path, personality_dir: Path) -> pd.DataFrame:
    group_path = features_dir / "group_pool_task_summary.tsv"
    group = read_tsv(group_path, required=False)
    if group.empty:
        return pd.DataFrame()
    out = group.copy()
    if "session_id" in out.columns:
        out["session_id"] = out["session_id"].map(_normalize_session_id)
    if "group_id" not in out.columns:
        out["group_id"] = out["session_id"].map(_group_id_from_session)

    traits = read_tsv(personality_dir / "group_trait_stats.tsv", required=False)
    if not traits.empty:
        out = out.merge(traits, on="group_id", how="left")

    group_features = [
        f"{col}_group_mean"
        for col in PHYSIO_CANDIDATES + BIOMARKER_CANDIDATES + SELF_REPORT_CANDIDATES
        if f"{col}_group_mean" in out.columns
    ]
    _numeric_inplace(out, group_features)
    out = _add_baseline_deltas(out, "group_id", group_features)
    return out


def load_pair_dataset(features_dir: Path, personality_dir: Path, group_df: pd.DataFrame) -> pd.DataFrame:
    pair_path = features_dir / "features_group_dynamics_task.tsv"
    pair = read_tsv(pair_path, required=False)
    if pair.empty:
        return pd.DataFrame()

    out = pair.copy()
    if "session_id" in out.columns:
        out["session_id"] = out["session_id"].map(_normalize_session_id)
    if "group_id" not in out.columns:
        out["group_id"] = out["session_id"].map(_group_id_from_session)
    out["pair_uid"] = (
        out["session_id"].astype(str)
        + ":"
        + out["participant_a"].astype(str)
        + "-"
        + out["participant_b"].astype(str)
        + ":"
        + out["metric"].astype(str)
    )
    _numeric_inplace(out, ["window_count", "corr", "best_lag_corr", "best_lag_windows"])

    traits = read_tsv(personality_dir / "group_trait_stats.tsv", required=False)
    if not traits.empty:
        out = out.merge(traits, on="group_id", how="left")

    group_cols = [
        "session_id",
        "group_id",
        "task",
        "participant_n",
        "vad_arousal_self_num_group_mean",
        "vad_valence_self_num_group_mean",
        "ans_engagement_group_mean",
        "ans_mental_demand_group_mean",
        "ans_overall_valence_group_mean",
        "audio_speaking_fraction_group_mean",
        "audio_turn_count_group_mean",
        "audio_overlap_fraction_group_mean",
        "audio_energy_mean_group_mean",
        "audio_pitch_mean_group_mean",
    ]
    if not group_df.empty:
        out = out.merge(
            group_df[[c for c in group_cols if c in group_df.columns]],
            on=["session_id", "group_id", "task"],
            how="left",
        )
    return out


def _default_model_specs(df: pd.DataFrame, all_pair_models: bool) -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    seen: set[tuple[str, str, str | None]] = set()

    def add_spec(family: str, outcome: str, predictor: str | None, active_tasks_only: bool) -> None:
        if outcome not in df.columns or (predictor is not None and predictor not in df.columns):
            return
        key = (family, outcome, predictor)
        if key in seen:
            return
        seen.add(key)
        specs.append(ModelSpec(family, outcome, predictor, active_tasks_only))

    task_outcomes = _present(df, DEFAULT_TASK_TEST_OUTCOMES)
    for outcome in task_outcomes:
        add_spec("task_effect", outcome, None, False)

    targeted = [
        ("cross_modal", "vad_arousal_self_num", "biomarker_arousal_stress"),
        ("cross_modal", "vad_arousal_self_num", "eda_mean_delta_t0"),
        ("cross_modal", "vad_arousal_self_num", "scr_rate_hz_delta_t0"),
        ("cross_modal", "vad_arousal_self_num", "pupil_mean_delta_t0"),
        ("cross_modal", "vad_valence_self_num", "biomarker_recovery_capacity"),
        ("cross_modal", "vad_valence_self_num", "biomarker_fatigue_depletion"),
        ("cross_modal", "vad_valence_self_num", "temp_mean_delta_t0"),
        ("cross_modal", "vad_dominance_self_num", "biomarker_decision_pressure"),
        ("cross_modal", "ans_engagement", "biomarker_attention"),
        ("cross_modal", "ans_mental_demand", "biomarker_cognitive_load"),
        ("cross_modal", "ans_overall_valence", "biomarker_recovery_capacity"),
        ("annotation", "vad_arousal_self_num", "ann_response_vad_n"),
        ("annotation", "ans_engagement", "ann_total_events_n"),
        ("annotation", "ans_voice_inclusion", "ann_response_postblock_n"),
        ("personality", "vad_arousal_self_num", "bfi44_n_z"),
        ("personality", "vad_valence_self_num", "bfi44_e_z"),
        ("personality", "ans_voice_inclusion", "bfi44_a_z"),
        ("personality", "ans_mental_demand", "bfi44_n_z"),
        ("personality", "biomarker_arousal_stress", "bfi44_n_z"),
    ]
    for family, outcome, predictor in targeted:
        add_spec(family, outcome, predictor, True)

    for family, outcome_candidates, predictor_candidates in MODEL_FAMILY_SPECS:
        for outcome, predictor in itertools.product(
            _present(df, outcome_candidates),
            _present(df, predictor_candidates),
        ):
            if outcome != predictor and _complete_n(df, outcome, predictor) > 0:
                add_spec(family, outcome, predictor, True)

    if all_pair_models:
        outcomes = _present(df, SELF_REPORT_CANDIDATES)
        predictors = _present(
            df,
            [
                *BIOMARKER_CANDIDATES,
                "ppg_rate_proxy_bpm_delta_t0",
                "eda_mean_delta_t0",
                "scr_rate_hz_delta_t0",
                "temp_mean_delta_t0",
                "pupil_mean_delta_t0",
                *ANNOTATION_CANDIDATES,
                *PERSONALITY_CANDIDATES,
            ],
        )
        for outcome, predictor in itertools.product(outcomes, predictors):
            if outcome != predictor:
                add_spec("broad_pair", outcome, predictor, True)
    return specs


def _fit_mixed_models(
    df: pd.DataFrame,
    specs: Iterable[ModelSpec],
    min_n: int,
    min_groups: int,
    allow_ols_fallback: bool,
) -> pd.DataFrame:
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        rows = []
        for spec in specs:
            rows.append(
                {
                    "model_family": spec.family,
                    "model_type": "mixedlm",
                    "status": "skipped_missing_statsmodels",
                    "outcome": spec.outcome,
                    "predictor": spec.predictor or "",
                    "term": "",
                    "n": 0,
                    "groups": 0,
                    "estimate": np.nan,
                    "std_error": np.nan,
                    "p_value": np.nan,
                    "message": "Install statsmodels, for example: pip install statsmodels",
                }
            )
        return pd.DataFrame(rows)

    rows: list[dict[str, object]] = []
    for spec in specs:
        cols = ["participant_uid", "task", spec.outcome]
        if spec.predictor:
            cols.append(spec.predictor)
        if not set(cols).issubset(df.columns):
            continue

        model_df = df[cols].copy()
        if spec.active_tasks_only:
            model_df = model_df[model_df["task"].isin(ACTIVE_TASKS)]
        model_df[spec.outcome] = pd.to_numeric(model_df[spec.outcome], errors="coerce")
        model_df["y_z"] = _standardize(model_df[spec.outcome])
        formula = "y_z ~ C(task)"
        if spec.predictor:
            model_df[spec.predictor] = pd.to_numeric(model_df[spec.predictor], errors="coerce")
            model_df["x_z"] = _standardize(model_df[spec.predictor])
            formula = "y_z ~ x_z + C(task)"
        model_df = model_df.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["participant_uid", "task", "y_z"] + (["x_z"] if spec.predictor else [])
        )
        n = int(len(model_df))
        groups = int(model_df["participant_uid"].nunique())
        if n < min_n or groups < min_groups or model_df["y_z"].nunique() < 2:
            rows.append(
                {
                    "model_family": spec.family,
                    "model_type": "mixedlm",
                    "status": "skipped_insufficient_data",
                    "outcome": spec.outcome,
                    "predictor": spec.predictor or "",
                    "term": "",
                    "n": n,
                    "groups": groups,
                    "estimate": np.nan,
                    "std_error": np.nan,
                    "p_value": np.nan,
                    "message": f"Need at least n={min_n} and groups={min_groups}",
                }
            )
            continue

        model_type = "mixedlm"
        status = "ok"
        message = ""
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                model = smf.mixedlm(formula, model_df, groups=model_df["participant_uid"])
                fit = model.fit(reml=False, method="lbfgs", maxiter=250, disp=False)
            warning_text = "; ".join(
                dict.fromkeys(str(w.message) for w in caught[:3])
            )
            if warning_text:
                status = "warning_fit"
                message = warning_text[:500]
            if not bool(getattr(fit, "converged", True)):
                status = "warning_not_converged"
        except Exception as exc:
            if not allow_ols_fallback:
                rows.append(
                    {
                        "model_family": spec.family,
                        "model_type": "mixedlm",
                        "status": "failed",
                        "outcome": spec.outcome,
                        "predictor": spec.predictor or "",
                        "term": "",
                        "n": n,
                        "groups": groups,
                        "estimate": np.nan,
                        "std_error": np.nan,
                        "p_value": np.nan,
                        "message": str(exc)[:500],
                    }
                )
                continue
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    fit = smf.ols(formula, model_df).fit(
                        cov_type="cluster",
                        cov_kwds={"groups": model_df["participant_uid"]},
                    )
                model_type = "ols_cluster_fallback"
                status = "fallback_ok"
                warning_text = "; ".join(
                    dict.fromkeys(str(w.message) for w in caught[:2])
                )
                message = "; ".join([m for m in [str(exc), warning_text] if m])[:500]
            except Exception as fallback_exc:
                rows.append(
                    {
                        "model_family": spec.family,
                        "model_type": "mixedlm",
                        "status": "failed",
                        "outcome": spec.outcome,
                        "predictor": spec.predictor or "",
                        "term": "",
                        "n": n,
                        "groups": groups,
                        "estimate": np.nan,
                        "std_error": np.nan,
                        "p_value": np.nan,
                        "message": f"{exc}; fallback failed: {fallback_exc}"[:500],
                    }
                )
                continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params = fit.params
            bse = fit.bse
            pvalues = fit.pvalues
            conf = fit.conf_int()
        for term, estimate in params.items():
            if term == "Group Var":
                continue
            rows.append(
                {
                    "model_family": spec.family,
                    "model_type": model_type,
                    "status": status,
                    "outcome": spec.outcome,
                    "predictor": spec.predictor or "",
                    "term": term,
                    "n": n,
                    "groups": groups,
                    "estimate": float(estimate),
                    "std_error": float(bse.get(term, np.nan)),
                    "ci95_low": float(conf.loc[term, 0]) if term in conf.index else np.nan,
                    "ci95_high": float(conf.loc[term, 1]) if term in conf.index else np.nan,
                    "p_value": float(pvalues.get(term, np.nan)),
                    "aic": float(getattr(fit, "aic", np.nan)),
                    "bic": float(getattr(fit, "bic", np.nan)),
                    "message": message,
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out["q_value"] = _bh_fdr(out["p_value"])
    return out


def build_participant_outputs(
    participant: pd.DataFrame,
    out_dir: Path,
    min_n: int,
    min_groups: int,
    by_task_correlations: bool,
    all_pair_models: bool,
    allow_ols_fallback: bool,
) -> None:
    write_tsv(participant, out_dir / "analysis_dataset_participant_task.tsv")
    catalog = build_combination_catalog(participant, min_n)
    write_tsv(catalog, out_dir / "analysis_combination_catalog.tsv")

    task_outcomes = _present(participant, DEFAULT_TASK_TEST_OUTCOMES)
    task_tests = _paired_task_tests(participant, "participant_uid", task_outcomes, min_n)
    write_tsv(task_tests, out_dir / "participant_task_pairwise_tests.tsv")

    corr_tables = _family_correlation_tables(
        participant,
        CORRELATION_FAMILY_SPECS,
        min_n,
        "all_tasks",
    )
    if by_task_correlations:
        for task, task_df in participant.groupby("task"):
            corr_tables.extend(
                _family_correlation_tables(
                    task_df,
                    CORRELATION_FAMILY_SPECS,
                    min_n,
                    f"task_{task}",
                )
            )
    correlations = (
        pd.concat([t for t in corr_tables if not t.empty], ignore_index=True)
        if corr_tables
        else pd.DataFrame()
    )
    write_tsv(correlations, out_dir / "participant_cross_modal_correlations.tsv")

    specs = _default_model_specs(participant, all_pair_models)
    models = _fit_mixed_models(
        participant,
        specs,
        min_n=min_n,
        min_groups=min_groups,
        allow_ols_fallback=allow_ols_fallback,
    )
    write_tsv(models, out_dir / "mixed_model_results.tsv")


def build_group_outputs(group_df: pd.DataFrame, out_dir: Path, min_n: int) -> None:
    if group_df.empty:
        write_tsv(pd.DataFrame(), out_dir / "analysis_dataset_group_task.tsv")
        write_tsv(pd.DataFrame(), out_dir / "group_task_pairwise_tests.tsv")
        write_tsv(pd.DataFrame(), out_dir / "group_trait_correlations.tsv")
        return

    write_tsv(group_df, out_dir / "analysis_dataset_group_task.tsv")
    outcome_cols = [
        c
        for c in group_df.columns
        if c.endswith("_group_mean")
        and (
            c.startswith("vad_")
            or c.startswith("ans_")
            or c.startswith("biomarker_")
            or c.startswith("eda_")
            or c.startswith("scr_")
            or c.startswith("pupil_")
            or c.startswith("ppg_")
            or c.startswith("temp_")
            or c.startswith("audio_")
        )
    ]
    outcome_cols.extend([c for c in group_df.columns if c.endswith("_group_mean_delta_t0")])
    task_tests = _paired_task_tests(group_df, "group_id", outcome_cols, min_n)
    write_tsv(task_tests, out_dir / "group_task_pairwise_tests.tsv")

    trait_cols = [
        c
        for c in group_df.columns
        if c.startswith("bfi44_") or c in {"mean_pairwise_dist", "n"}
    ]
    corr_targets = [
        c
        for c in outcome_cols
        if c in group_df.columns
        and any(prefix in c for prefix in ["vad_", "ans_", "biomarker_", "eda_", "pupil_", "audio_"])
    ]
    correlations = _correlation_rows(
        group_df,
        corr_targets,
        trait_cols,
        min_n,
        "group_tasks",
        {"comparison_family": "group_outcome_vs_group_traits"},
    )
    if not correlations.empty:
        # Flag rows where n < 15 as exploratory (group-level analyses have n=8â€“10 groups)
        correlations["exploratory"] = correlations["n"] < 15
        correlations["power_note"] = correlations["n"].apply(
            lambda x: (
                f"EXPLORATORY: n={x} groups; power is insufficient for conventional "
                "significance testing (80% power requires nâ‰¥26 for |r|=0.5). "
                "Treat p-values as descriptive only."
            ) if x < 15 else ""
        )
    write_tsv(correlations, out_dir / "group_trait_correlations.tsv")


def build_pair_outputs(pair_df: pd.DataFrame, out_dir: Path, min_n: int) -> None:
    if pair_df.empty:
        write_tsv(pd.DataFrame(), out_dir / "analysis_dataset_pair_task.tsv")
        write_tsv(pd.DataFrame(), out_dir / "pair_synchrony_task_tests.tsv")
        write_tsv(pd.DataFrame(), out_dir / "pair_synchrony_correlations.tsv")
        return

    write_tsv(pair_df, out_dir / "analysis_dataset_pair_task.tsv")
    pair_tests: list[pd.DataFrame] = []
    for metric, metric_df in pair_df.groupby("metric"):
        pair_tests.append(
            _paired_task_tests(
                metric_df,
                "pair_uid",
                ["corr", "best_lag_corr"],
                min_n,
                {"metric": metric},
            )
        )
    tests = pd.concat([t for t in pair_tests if not t.empty], ignore_index=True)
    if not tests.empty:
        tests["q_paired_t"] = _bh_fdr(tests["p_paired_t"])
        tests["q_wilcoxon"] = _bh_fdr(tests["p_wilcoxon"])
    write_tsv(tests, out_dir / "pair_synchrony_task_tests.tsv")

    trait_cols = [
        c
        for c in pair_df.columns
        if c.startswith("bfi44_") or c in {"mean_pairwise_dist", "participant_n"}
    ]
    group_context_cols = _present(
        pair_df,
        [
            "vad_arousal_self_num_group_mean",
            "vad_valence_self_num_group_mean",
            "ans_engagement_group_mean",
            "ans_mental_demand_group_mean",
            "ans_overall_valence_group_mean",
            "audio_speaking_fraction_group_mean",
            "audio_turn_count_group_mean",
            "audio_overlap_fraction_group_mean",
            "audio_energy_mean_group_mean",
            "audio_pitch_mean_group_mean",
        ],
    )
    corr_tables: list[pd.DataFrame] = []
    for metric, metric_df in pair_df.groupby("metric"):
        corr_tables.append(
            _correlation_rows(
                metric_df,
                ["corr", "best_lag_corr"],
                trait_cols + group_context_cols,
                min_n,
                f"pair_metric_{metric}",
                {"comparison_family": "pair_synchrony_vs_group_context", "metric": metric},
            )
        )
    correlations = pd.concat([t for t in corr_tables if not t.empty], ignore_index=True)
    write_tsv(correlations, out_dir / "pair_synchrony_correlations.tsv")


def _label_feature(name: object) -> str:
    label = str(name)
    replacements = {
        "vad_": "VAD ",
        "ans_": "",
        "audio_": "Audio ",
        "bfi44_": "BFI ",
        "biomarker_": "",
        "_self_num": "",
        "_group_mean": "",
        "_delta_t0": " delta T0",
        "_z": " z",
        "ppg_rate_proxy_bpm": "PPG rate",
        "ppg_rmssd_ms": "RMSSD",
        "eda_mean": "EDA mean",
        "scr_rate_hz": "SCR rate",
        "temp_mean": "Skin temp",
        "pupil_mean": "Pupil mean",
        "pupil_std": "Pupil SD",
        "cognitive_load": "Cognitive load",
        "arousal_stress": "Arousal stress",
        "decision_pressure": "Decision pressure",
        "recovery_capacity": "Recovery capacity",
        "fatigue_depletion": "Fatigue depletion",
        "overall_valence": "Overall valence",
        "mental_demand": "Mental demand",
        "voice_inclusion": "Voice inclusion",
        "team_coordination": "Team coordination",
        "psych_safety": "Psych safety",
        "speaking_time_s": "Speaking time",
        "speaking_fraction": "Speaking fraction",
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
        "voiced_segments_per_sec": "Voiced segments/sec",
        "mean_voiced_segment_s": "Voiced segment duration",
        "mean_unvoiced_segment_s": "Unvoiced segment duration",
        "speech_rate_proxy": "Speech rate",
    }
    for old, new in replacements.items():
        label = label.replace(old, new)
    label = label.replace("_", " ").strip()
    return " ".join(label.split()).title()


def _save_figure(fig: object, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    LOG.info("Wrote %s", path)


def _plot_matrix(
    matrix: pd.DataFrame,
    title: str,
    cbar_label: str,
    out_path: Path,
    dpi: int,
    *,
    value_limit: float = 1.0,
) -> None:
    if matrix.empty:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = matrix.to_numpy(dtype=float)
    height = max(4.0, 0.42 * len(matrix.index) + 1.8)
    width = max(6.0, 0.55 * len(matrix.columns) + 2.4)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(data, cmap="RdBu_r", vmin=-value_limit, vmax=value_limit, aspect="auto")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([_label_feature(c) for c in matrix.columns], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([_label_feature(i) for i in matrix.index], fontsize=8)
    ax.set_xticks(np.arange(-0.5, len(matrix.columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(matrix.index), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            value = data[i, j]
            if np.isfinite(value):
                color = "white" if abs(value) > value_limit * 0.55 else "black"
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5, color=color)
    cbar = fig.colorbar(im, ax=ax, shrink=0.82)
    cbar.set_label(cbar_label)
    _save_figure(fig, out_path, dpi)
    plt.close(fig)


def plot_task_effect_heatmap(task_tests: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if task_tests.empty:
        return
    df = task_tests.copy()
    df["cohen_dz"] = pd.to_numeric(df.get("cohen_dz"), errors="coerce")
    df["n_pairs"] = pd.to_numeric(df.get("n_pairs"), errors="coerce")
    df = df[(df["n_pairs"] >= 8) & df["cohen_dz"].notna()]
    if df.empty:
        return
    df["contrast"] = df["task_a"].astype(str) + " to " + df["task_b"].astype(str)
    strength = df.groupby("outcome")["cohen_dz"].apply(lambda x: x.abs().max()).sort_values(ascending=False)
    outcomes = strength.head(14).index.tolist()
    matrix = (
        df[df["outcome"].isin(outcomes)]
        .pivot_table(index="outcome", columns="contrast", values="cohen_dz", aggfunc="mean")
        .reindex(outcomes)
    )
    ordered_contrasts = [
        f"{a} to {b}"
        for a, b in itertools.combinations(TASK_ORDER, 2)
        if f"{a} to {b}" in matrix.columns
    ]
    matrix = matrix[ordered_contrasts]
    _plot_matrix(
        matrix,
        "Within-Participant Task Effects",
        "Cohen dz",
        figures_dir / "participant_task_effects_heatmap.png",
        dpi,
        value_limit=max(1.0, float(np.nanmax(np.abs(matrix.to_numpy(dtype=float))))),
    )


def plot_audio_task_heatmap(participant: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if participant.empty:
        return
    audio_cols = _present(
        participant,
        [
            "audio_speaking_fraction",
            "audio_turn_count",
            "audio_pause_count",
            "audio_overlap_fraction",
            "audio_energy_mean",
            "audio_pitch_mean",
            "audio_hnr_mean",
            "audio_jitter_mean",
            "audio_shimmer_mean",
            "audio_speech_rate_proxy",
            "audio_speaking_fraction_delta_t0",
            "audio_turn_count_delta_t0",
            "audio_overlap_fraction_delta_t0",
            "audio_energy_mean_delta_t0",
            "audio_pitch_mean_delta_t0",
            "audio_speech_rate_proxy_delta_t0",
        ],
    )
    if not audio_cols or "task" not in participant.columns:
        return
    df = participant[["task", *audio_cols]].copy()
    _numeric_inplace(df, audio_cols)
    means = df.groupby("task")[audio_cols].mean().reindex(TASK_ORDER)
    means = means.dropna(axis=1, how="all")
    if means.empty:
        return
    matrix = means.transpose()
    matrix = matrix.loc[matrix.notna().sum(axis=1) >= 2]
    if matrix.empty:
        return
    matrix = matrix.sub(matrix.mean(axis=1), axis=0)
    sd = matrix.std(axis=1, ddof=1).replace(0, np.nan)
    matrix = matrix.div(sd, axis=0).replace([np.inf, -np.inf], np.nan).dropna(how="all")
    if matrix.empty:
        return
    _plot_matrix(
        matrix,
        "Audio Features by Task",
        "Within-feature z",
        figures_dir / "audio_task_feature_heatmap.png",
        dpi,
        value_limit=max(1.0, float(np.nanmax(np.abs(matrix.to_numpy(dtype=float))))),
    )


def plot_cross_modal_heatmap(correlations: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if correlations.empty:
        return
    df = correlations.copy()
    df = df[
        (
            df.get("comparison_family").isin(
                [
                    "self_report_vs_biomarker",
                    "self_report_vs_physio_delta",
                    "self_report_vs_audio_delta",
                ]
            )
        )
        & (df.get("scope") == "all_tasks")
    ].copy()
    if df.empty:
        return
    df["spearman_rho"] = pd.to_numeric(df.get("spearman_rho"), errors="coerce")
    df = df.dropna(subset=["spearman_rho"])
    if df.empty:
        return
    left_order = (
        df.groupby("left")["spearman_rho"].apply(lambda x: x.abs().max()).sort_values(ascending=False).head(12).index
    )
    right_order = (
        df.groupby("right")["spearman_rho"].apply(lambda x: x.abs().max()).sort_values(ascending=False).head(12).index
    )
    matrix = (
        df[df["left"].isin(left_order) & df["right"].isin(right_order)]
        .pivot_table(index="left", columns="right", values="spearman_rho", aggfunc="mean")
        .reindex(index=left_order, columns=right_order)
    )
    _plot_matrix(
        matrix,
        "Self-Report vs Physiological Features",
        "Spearman rho",
        figures_dir / "cross_modal_correlation_heatmap.png",
        dpi,
    )


def plot_correlation_family_heatmaps(correlations: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if correlations.empty:
        return
    df = correlations.copy()
    df = df[df.get("scope") == "all_tasks"].copy()
    df["spearman_rho"] = pd.to_numeric(df.get("spearman_rho"), errors="coerce")
    df = df.dropna(subset=["spearman_rho"])
    if df.empty:
        return
    for family, family_df in df.groupby("comparison_family"):
        if family_df.empty:
            continue
        left_order = (
            family_df.groupby("left")["spearman_rho"]
            .apply(lambda x: x.abs().max())
            .sort_values(ascending=False)
            .head(12)
            .index
        )
        right_order = (
            family_df.groupby("right")["spearman_rho"]
            .apply(lambda x: x.abs().max())
            .sort_values(ascending=False)
            .head(12)
            .index
        )
        matrix = (
            family_df[family_df["left"].isin(left_order) & family_df["right"].isin(right_order)]
            .pivot_table(index="left", columns="right", values="spearman_rho", aggfunc="mean")
            .reindex(index=left_order, columns=right_order)
        )
        _plot_matrix(
            matrix,
            _label_feature(family),
            "Spearman rho",
            figures_dir / f"correlation_family_{family}.png",
            dpi,
        )


def plot_mixed_model_forest(models: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if models.empty:
        return
    df = models.copy()
    df = df[(df.get("term") == "x_z") & (df.get("status").isin(["ok", "warning_fit", "fallback_ok"]))].copy()
    for col in ["estimate", "ci95_low", "ci95_high", "p_value", "q_value"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df = df.dropna(subset=["estimate"])
    if df.empty:
        return
    df["label"] = df["outcome"].map(_label_feature) + " ~ " + df["predictor"].map(_label_feature)
    df["rank"] = df["estimate"].abs()
    df = df.sort_values("rank", ascending=False).head(18).sort_values("estimate")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.2, max(4.8, 0.34 * len(df) + 1.8)))
    y = np.arange(len(df))
    estimate = df["estimate"].to_numpy(dtype=float)
    ci_low = df["ci95_low"].to_numpy(dtype=float)
    ci_high = df["ci95_high"].to_numpy(dtype=float)
    has_ci = np.isfinite(ci_low) & np.isfinite(ci_high)
    colors = np.where(df["model_type"].astype(str).str.contains("fallback"), "#9a7d0a", "#1f77b4")
    ax.scatter(estimate, y, s=42, c=colors, zorder=3)
    for idx, ok in enumerate(has_ci):
        if ok:
            ax.plot([ci_low[idx], ci_high[idx]], [y[idx], y[idx]], color=colors[idx], lw=1.5, alpha=0.85)
    ax.axvline(0, color="0.25", lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(df["label"], fontsize=8)
    ax.set_xlabel("Standardized coefficient")
    ax.set_title("Mixed-Model Cross-Modal Associations", fontsize=12, fontweight="bold")
    ax.grid(axis="x", color="0.9", lw=0.8)
    ax.text(
        0.99,
        0.02,
        "Blue: MixedLM; ochre: clustered OLS fallback",
        ha="right",
        va="bottom",
        transform=ax.transAxes,
        fontsize=8,
        color="0.35",
    )
    _save_figure(fig, figures_dir / "mixed_model_forest.png", dpi)
    plt.close(fig)


def plot_mixed_model_coefficient_heatmap(models: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if models.empty:
        return
    df = models.copy()
    df = df[(df.get("term") == "x_z") & (df.get("status").isin(["ok", "warning_fit", "fallback_ok"]))].copy()
    df["estimate"] = pd.to_numeric(df.get("estimate"), errors="coerce")
    df["q_value"] = pd.to_numeric(df.get("q_value"), errors="coerce")
    df = df.dropna(subset=["estimate"])
    if df.empty:
        return
    df["rank"] = df["estimate"].abs()
    outcome_order = (
        df.groupby("outcome")["rank"].max().sort_values(ascending=False).head(16).index
    )
    predictor_order = (
        df.groupby("predictor")["rank"].max().sort_values(ascending=False).head(16).index
    )
    matrix = (
        df[df["outcome"].isin(outcome_order) & df["predictor"].isin(predictor_order)]
        .pivot_table(index="outcome", columns="predictor", values="estimate", aggfunc="mean")
        .reindex(index=outcome_order, columns=predictor_order)
    )
    _plot_matrix(
        matrix,
        "Screening Model Coefficients",
        "Std. beta",
        figures_dir / "mixed_model_coefficient_heatmap.png",
        dpi,
        value_limit=max(1.0, float(np.nanmax(np.abs(matrix.to_numpy(dtype=float))))),
    )


def plot_mixed_model_volcano(models: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if models.empty:
        return
    df = models.copy()
    df = df[(df.get("term") == "x_z") & (df.get("status").isin(["ok", "warning_fit", "fallback_ok"]))].copy()
    for col in ["estimate", "q_value", "p_value"]:
        df[col] = pd.to_numeric(df.get(col), errors="coerce")
    df = df.dropna(subset=["estimate"])
    if df.empty:
        return
    q = df["q_value"].where(df["q_value"].notna(), df["p_value"])
    q = q.clip(lower=1e-300)
    df["minus_log10_q"] = -np.log10(q)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    families = sorted(df["model_family"].dropna().unique().tolist())
    cmap = plt.get_cmap("tab20")
    color_map = {family: cmap(i % cmap.N) for i, family in enumerate(families)}
    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    for family, family_df in df.groupby("model_family"):
        ax.scatter(
            family_df["estimate"],
            family_df["minus_log10_q"],
            s=26,
            alpha=0.72,
            color=color_map.get(family),
            label=_label_feature(family),
        )
    ax.axvline(0, color="0.3", lw=0.9)
    ax.axhline(-np.log10(0.05), color="0.4", lw=0.8, ls="--")
    ax.set_xlabel("Standardized coefficient")
    ax.set_ylabel("-log10(q)")
    ax.set_title("Exploratory Model Screening", fontsize=12, fontweight="bold")
    ax.grid(color="0.92", lw=0.8)
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:10], labels[:10], loc="upper right", fontsize=7, frameon=False)
    _save_figure(fig, figures_dir / "mixed_model_screening_volcano.png", dpi)
    plt.close(fig)


def plot_pair_synchrony(pair_df: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if pair_df.empty or "metric" not in pair_df.columns:
        return
    df = pair_df.copy()
    df["corr"] = pd.to_numeric(df.get("corr"), errors="coerce")
    df = df.dropna(subset=["corr"])
    if df.empty:
        return
    metrics = (
        df.groupby("metric")["corr"].count().sort_values(ascending=False).head(4).index.tolist()
    )
    if not metrics:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(metrics), figsize=(4.0 * len(metrics), 4.2), sharey=True)
    if len(metrics) == 1:
        axes = [axes]
    for ax, metric in zip(axes, metrics, strict=True):
        metric_df = df[df["metric"] == metric]
        data = [
            metric_df.loc[metric_df["task"] == task, "corr"].dropna().to_numpy(dtype=float)
            for task in TASK_ORDER
        ]
        present = [(task, values) for task, values in zip(TASK_ORDER, data, strict=True) if len(values)]
        if not present:
            continue
        labels, values = zip(*present, strict=True)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message="The 'labels' parameter of boxplot")
            ax.boxplot(values, labels=labels, showfliers=False, patch_artist=True)
        means = [float(np.mean(v)) for v in values]
        ax.plot(np.arange(1, len(means) + 1), means, "o-", color="#d62728", lw=1.2, ms=4)
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_title(_label_feature(metric), fontsize=10, fontweight="bold")
        ax.set_ylim(-1.05, 1.05)
        ax.grid(axis="y", color="0.9", lw=0.8)
    axes[0].set_ylabel("Pairwise synchrony correlation")
    fig.suptitle("Dyad Synchrony by Task", fontsize=12, fontweight="bold")
    _save_figure(fig, figures_dir / "pair_synchrony_by_task.png", dpi)
    plt.close(fig)


def plot_pair_context_heatmap(correlations: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if correlations.empty:
        return
    df = correlations.copy()
    df["spearman_rho"] = pd.to_numeric(df.get("spearman_rho"), errors="coerce")
    df = df.dropna(subset=["spearman_rho"])
    if df.empty:
        return
    df["row"] = df["metric"].astype(str) + " " + df["left"].astype(str)
    row_order = (
        df.groupby("row")["spearman_rho"].apply(lambda x: x.abs().max()).sort_values(ascending=False).head(14).index
    )
    col_order = (
        df.groupby("right")["spearman_rho"].apply(lambda x: x.abs().max()).sort_values(ascending=False).head(12).index
    )
    matrix = (
        df[df["row"].isin(row_order) & df["right"].isin(col_order)]
        .pivot_table(index="row", columns="right", values="spearman_rho", aggfunc="mean")
        .reindex(index=row_order, columns=col_order)
    )
    _plot_matrix(
        matrix,
        "Dyad Synchrony vs Group Context",
        "Spearman rho",
        figures_dir / "pair_synchrony_context_heatmap.png",
        dpi,
    )


def plot_group_trait_heatmap(correlations: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if correlations.empty:
        return
    df = correlations.copy()
    df["spearman_rho"] = pd.to_numeric(df.get("spearman_rho"), errors="coerce")
    df = df.dropna(subset=["spearman_rho"])
    if df.empty:
        return
    left_order = (
        df.groupby("left")["spearman_rho"].apply(lambda x: x.abs().max()).sort_values(ascending=False).head(14).index
    )
    right_order = (
        df.groupby("right")["spearman_rho"].apply(lambda x: x.abs().max()).sort_values(ascending=False).head(10).index
    )
    matrix = (
        df[df["left"].isin(left_order) & df["right"].isin(right_order)]
        .pivot_table(index="left", columns="right", values="spearman_rho", aggfunc="mean")
        .reindex(index=left_order, columns=right_order)
    )
    _plot_matrix(
        matrix,
        "Group Outcomes vs Group Personality",
        "Spearman rho",
        figures_dir / "group_trait_correlation_heatmap.png",
        dpi,
    )


def plot_combination_coverage(catalog: pd.DataFrame, figures_dir: Path, dpi: int) -> None:
    if catalog.empty:
        return
    df = catalog.copy()
    df["complete_n"] = pd.to_numeric(df.get("complete_n"), errors="coerce")
    summary = (
        df.groupby(["analysis_type", "comparison_family"], as_index=False)
        .agg(
            candidate_pairs=("complete_n", "size"),
            usable_pairs=("meets_min_n", "sum"),
            median_complete_n=("complete_n", "median"),
        )
        .sort_values(["analysis_type", "usable_pairs"], ascending=[True, False])
    )
    if summary.empty:
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = summary["analysis_type"].astype(str) + "\n" + summary["comparison_family"].map(_label_feature)
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(max(8.0, 0.52 * len(summary)), 5.2))
    ax.bar(x, summary["candidate_pairs"], color="#d9d9d9", label="Candidate")
    ax.bar(x, summary["usable_pairs"], color="#3182bd", label="Meets min n")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=7)
    ax.set_ylabel("Variable pair count")
    ax.set_title("Exploratory Combination Coverage", fontsize=12, fontweight="bold")
    ax.legend(frameon=False)
    ax.grid(axis="y", color="0.92", lw=0.8)
    _save_figure(fig, figures_dir / "analysis_combination_coverage.png", dpi)
    plt.close(fig)


def build_visualizations(out_dir: Path, figures_dir: Path, dpi: int) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    participant = read_tsv(out_dir / "analysis_dataset_participant_task.tsv", required=False)
    catalog = read_tsv(out_dir / "analysis_combination_catalog.tsv", required=False)
    task_tests = read_tsv(out_dir / "participant_task_pairwise_tests.tsv", required=False)
    correlations = read_tsv(out_dir / "participant_cross_modal_correlations.tsv", required=False)
    models = read_tsv(out_dir / "mixed_model_results.tsv", required=False)
    pair_df = read_tsv(out_dir / "analysis_dataset_pair_task.tsv", required=False)
    pair_correlations = read_tsv(out_dir / "pair_synchrony_correlations.tsv", required=False)
    group_correlations = read_tsv(out_dir / "group_trait_correlations.tsv", required=False)

    plot_combination_coverage(catalog, figures_dir, dpi)
    plot_task_effect_heatmap(task_tests, figures_dir, dpi)
    plot_audio_task_heatmap(participant, figures_dir, dpi)
    plot_cross_modal_heatmap(correlations, figures_dir, dpi)
    plot_correlation_family_heatmaps(correlations, figures_dir, dpi)
    plot_mixed_model_forest(models, figures_dir, dpi)
    plot_mixed_model_coefficient_heatmap(models, figures_dir, dpi)
    plot_mixed_model_volcano(models, figures_dir, dpi)
    plot_pair_synchrony(pair_df, figures_dir, dpi)
    plot_pair_context_heatmap(pair_correlations, figures_dir, dpi)
    plot_group_trait_heatmap(group_correlations, figures_dir, dpi)


def write_manifest(
    out_dir: Path,
    participant: pd.DataFrame,
    group_df: pd.DataFrame,
    pair_df: pd.DataFrame,
    figures_dir: Path | None = None,
) -> None:
    statsmodels_available = False
    try:
        import statsmodels  # noqa: F401

        statsmodels_available = True
    except ImportError:
        statsmodels_available = False

    lines = [
        "# Multimodal Statistics Manifest",
        "",
        "This directory contains downstream inferential outputs for the GroupAffect-4 feature tables.",
        "",
        "## Inputs",
        f"- participant-task rows: {len(participant)}",
        f"- group-task rows: {len(group_df)}",
        f"- dyad-task rows: {len(pair_df)}",
        f"- statsmodels available: {statsmodels_available}",
        "- optional audio input: `audio_participant_task.tsv` from `extract_audio_features.py` when available.",
        "",
        "## Output Families",
        "- `analysis_dataset_participant_task.tsv`: joined participant-task analysis table.",
        "- `audio_feature_status.tsv`: discovery status for optional audio/prosody inputs.",
        "- `mixed_model_results.tsv`: standardized mixed-model coefficients, with clustered OLS fallback when enabled.",
        "- `analysis_combination_catalog.tsv`: candidate analysis pairs, availability, and complete-case counts.",
        "- `participant_task_pairwise_tests.tsv`: paired task contrasts within participant.",
        "- `participant_cross_modal_correlations.tsv`: physiology, audio, answer, annotation, and personality correlations.",
        "- `analysis_dataset_group_task.tsv` and `group_*`: group-level task and trait checks.",
        "- `analysis_dataset_pair_task.tsv` and `pair_*`: dyad synchrony task and context checks.",
        "- `figures/`: PNG summaries for task effects, correlations, mixed models, and dyad synchrony.",
        "",
        "## Interpretation Notes",
        "- Treat p-values as exploratory until hypotheses and exclusion rules are preregistered.",
        "- Prefer effect sizes, confidence intervals, and FDR-adjusted q-values over raw p-values.",
        "- Participant-task models use random intercepts for participant/session seat IDs.",
        "- Pair and group outputs are kept separate from participant-level rows to avoid pseudo-replication.",
        "- Baseline-normalized columns ending in `_delta_t0` use each participant or group T0 value.",
        "- Audio columns use anonymized participant IDs only; transcript text and source paths are not merged.",
        "",
    ]
    if figures_dir is not None:
        lines.extend(
            [
                "## Figures",
                "- `figures/analysis_combination_coverage.png`",
                "- `figures/participant_task_effects_heatmap.png`",
                "- `figures/audio_task_feature_heatmap.png` (when audio features are available)",
                "- `figures/cross_modal_correlation_heatmap.png`",
                "- `figures/correlation_family_*.png`",
                "- `figures/mixed_model_forest.png`",
                "- `figures/mixed_model_coefficient_heatmap.png`",
                "- `figures/mixed_model_screening_volcano.png`",
                "- `figures/pair_synchrony_by_task.png`",
                "- `figures/pair_synchrony_context_heatmap.png`",
                "- `figures/group_trait_correlation_heatmap.png`",
                "",
            ]
        )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    LOG.info("Wrote %s", path)


def main() -> int:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    features_dir = args.features_dir.resolve()
    personality_dir = args.personality_dir.resolve()
    out_dir = args.out_dir.resolve()
    figures_dir = (
        args.figures_dir.resolve()
        if args.figures_dir is not None
        else out_dir / "figures"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    participant, audio_status = load_participant_dataset(features_dir, personality_dir, args.audio_features)
    group_df = load_group_dataset(features_dir, personality_dir)
    group_df = enrich_group_with_audio(group_df, participant)
    pair_df = load_pair_dataset(features_dir, personality_dir, group_df)
    write_tsv(audio_status, out_dir / "audio_feature_status.tsv")

    build_participant_outputs(
        participant,
        out_dir,
        min_n=args.min_n,
        min_groups=args.min_groups,
        by_task_correlations=args.by_task_correlations,
        all_pair_models=args.all_pair_models,
        allow_ols_fallback=not args.no_ols_fallback,
    )
    build_group_outputs(group_df, out_dir, min_n=max(4, min(args.min_n, 8)))
    build_pair_outputs(pair_df, out_dir, min_n=args.min_n)
    if not args.no_figures:
        build_visualizations(out_dir, figures_dir, args.dpi)
    write_manifest(out_dir, participant, group_df, pair_df, None if args.no_figures else figures_dir)
    LOG.info("Inferential multimodal statistics complete: %s", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

